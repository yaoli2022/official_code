#!/usr/bin/env python3
import argparse, csv, importlib.util, os, sys, time
import numpy as np

BASE = os.environ.get('GA_BASE', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EN_ALPHAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
EN_L1S = [0.1, 0.5, 0.9, 1.0]
MLP_GRID = [(d, w, dr, lr) for d in (2, 3, 4) for w in (256, 512)
            for dr in (0.1, 0.3) for lr in (1e-3, 3e-4)]
MLP_EPOCHS = 200

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['en', 'mlp'], required=True)
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--max-cpgs', type=int, default=None, help='smoke test only')
    ap.add_argument('--quick', action='store_true', help='smoke test only: tiny grid')
    a = ap.parse_args()

    import pandas as pd
    t = load('ga_train', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.py'))
    wb = load('ga_wave0', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wave0_baselines.py'))
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import mean_absolute_error

    alphas, l1s = (EN_ALPHAS, EN_L1S) if not a.quick else ([0.01, 0.1], [0.5])
    grid, epochs = (MLP_GRID, MLP_EPOCHS) if not a.quick else ([(2, 256, 0.1, 1e-3)], 20)

    out = os.path.join(BASE, 'baselines_%s.csv' % a.model)
    gout = os.path.join(BASE, 'baselines_%s_grid.csv' % a.model)
    pdir = os.path.join(BASE, 'baselines_preds'); os.makedirs(pdir, exist_ok=True)
    print('GA_BASE=%s  model=%s  device=%s  quick=%s' % (BASE, a.model, wb.device, a.quick), flush=True)

    train_combined, test_combined = t.load_split()
    meta = list(t.META_COLS)
    cpg_cols = [c for c in train_combined.columns if c not in meta]
    if a.max_cpgs:
        cpg_cols = cpg_cols[:a.max_cpgs]
    keep = cpg_cols + meta
    train_combined, test_combined = train_combined[keep], test_combined[keep]
    Xte, yte = wb.xy(test_combined)
    print('train %d  test %d  CpGs %d' % (len(train_combined), len(test_combined), len(cpg_cols)), flush=True)

    done = set()
    if os.path.exists(out):
        for r in csv.DictReader(open(out)):
            done.add((int(r['fold']), int(r['seed'])))
    fh = open(out, 'a', newline=''); w = csv.writer(fh)
    if os.path.getsize(out) == 0:
        w.writerow(['model', 'fold', 'seed', 'config', 'val_mae', 'test_mae', 'secs'])
    gh = open(gout, 'a', newline=''); gw = csv.writer(gh)
    if os.path.getsize(gout) == 0:
        gw.writerow(['fold', 'seed', 'config', 'val_mae', 'secs'])

    def save(f, s, pred):
        pd.DataFrame({'true_age': yte, 'predicted_age': pred}).to_csv(
            os.path.join(pdir, '%s_fold%d_seed%d.csv' % (a.model, f, s)), index=False)

    for fold in a.folds:
        todo = [s for s in a.seeds if (fold, s) not in done]
        if not todo:
            print('fold %d: all seeds done, skip' % fold, flush=True); continue
        ftr, fva = t.fold_frames(train_combined, fold)
        Xtr, ytr = wb.xy(ftr); Xva, yva = wb.xy(fva)
        print('\n=== fold %d  train%s  val%s  test%s ===' % (fold, Xtr.shape, Xva.shape, Xte.shape), flush=True)

        if a.model == 'en':
            t0 = time.time(); best = None
            for l1 in l1s:
                for al in alphas:
                    t1 = time.time()
                    m = ElasticNet(alpha=al, l1_ratio=l1, max_iter=3000, tol=1e-3,
                                   selection='random', random_state=wb.SEED)
                    m.fit(Xtr, ytr)
                    v = mean_absolute_error(yva, m.predict(Xva))
                    nz = int((m.coef_ != 0).sum())
                    cfg = 'alpha=%g,l1=%g' % (al, l1)
                    gw.writerow([fold, 'all', cfg + ',nonzero=%d' % nz, '%.4f' % v, '%.0f' % (time.time() - t1)]); gh.flush()
                    print('  alpha=%-6g l1=%-4g val MAE=%6.3f  nonzero=%-6d (%.0fs)' % (al, l1, v, nz, time.time() - t1), flush=True)
                    if best is None or v < best[0]:
                        best = (v, m, cfg, nz)
            v, m, cfg, nz = best
            pred = m.predict(Xte); s_mae = mean_absolute_error(yte, pred)
            print('  best %s nonzero=%d -> val %.4f  test %.4f' % (cfg, nz, v, s_mae), flush=True)
            for s in todo:
                w.writerow(['ElasticNet', fold, s, cfg + ',nonzero=%d' % nz, '%.4f' % v, '%.4f' % s_mae, '%.0f' % (time.time() - t0)])
                save(fold, s, pred)
            fh.flush()
            continue

        orig_set_seed = wb.set_seed
        for s in todo:
            t0 = time.time(); best = None
            wb.set_seed = (lambda _s=s: (lambda *args, **kw: orig_set_seed(_s)))()
            try:
                for depth, width, dr, lr in grid:
                    t1 = time.time()
                    v, pred = wb.fit_mlp(Xtr, ytr, Xva, yva, Xte, depth, width, dr, lr, epochs)
                    cfg = 'depth=%d,width=%d,dropout=%g,lr=%g' % (depth, width, dr, lr)
                    gw.writerow([fold, s, cfg, '%.4f' % v, '%.0f' % (time.time() - t1)]); gh.flush()
                    print('  f%d s%d %s  val MAE=%6.3f  (%.0fs)' % (fold, s, cfg, v, time.time() - t1), flush=True)
                    if best is None or v < best[0]:
                        best = (v, pred, cfg)
            finally:
                wb.set_seed = orig_set_seed
            v, pred, cfg = best
            s_mae = mean_absolute_error(yte, pred)
            w.writerow(['MLP', fold, s, cfg, '%.4f' % v, '%.4f' % s_mae, '%.0f' % (time.time() - t0)]); fh.flush()
            save(fold, s, pred)
            print('  -> MLP fold %d seed %d  best %s  val %.4f  test %.4f  (%.0fs)' % (fold, s, cfg, v, s_mae, time.time() - t0), flush=True)
    fh.close(); gh.close()

    rows = list(csv.DictReader(open(out)))
    v = [float(r['test_mae']) for r in rows]
    if v:
        m = sum(v) / len(v); sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** .5 if len(v) > 1 else 0.0
        print('\n%s: test MAE %.4f +/- %.4f  (n=%d)  -> %s' % (a.model, m, sd, len(v), out))

if __name__ == '__main__':
    main()
