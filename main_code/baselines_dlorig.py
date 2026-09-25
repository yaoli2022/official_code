#!/usr/bin/env python3
import argparse, copy, csv, fcntl, importlib.util, os, random, sys, time
import numpy as np

BASE = os.environ.get('GA_BASE', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CFG = {
    'deepmage':  dict(lr=1e-4, wd=1e-5, batch=64, epochs=500, patience=10, factor=0.5, top_k=1000, dropout=0.3),
    'resnetage': dict(lr=5e-4, wd=1e-4, batch=64, epochs=500, patience=15, factor=0.5, dropout=0.3),
}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['deepmage', 'resnetage'], required=True)
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--max-cpgs', type=int, default=None, help='smoke test only')
    ap.add_argument('--quick', action='store_true', help='smoke test only: 2 epochs, writes *_smoke files')
    a = ap.parse_args()

    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    t = load('ga_train', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.py'))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    c = dict(CFG[a.model])
    if a.quick:
        c['epochs'] = 2
    tag = a.model + ('_smoke' if a.quick else '')

    class DeepMAge(nn.Module):
        def __init__(self, input_dim, hidden_size=512, dropout=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_size), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_size, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    class ResBlock(nn.Module):
        def __init__(self, channels, kernel_size=3):
            super().__init__()
            pad = kernel_size // 2
            self.block = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, padding=pad), nn.ELU(), nn.BatchNorm1d(channels),
                nn.Conv1d(channels, channels, kernel_size, padding=pad), nn.ELU(), nn.BatchNorm1d(channels))

        def forward(self, x):
            return x + self.block(x)

    class ResnetAge(nn.Module):
        def __init__(self, input_dim, dropout=0.3):
            super().__init__()
            self.initial = nn.Sequential(nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.ELU(), nn.BatchNorm1d(32))
            self.res_blocks = nn.Sequential(
                ResBlock(32), nn.Conv1d(32, 64, kernel_size=1),
                ResBlock(64), nn.Conv1d(64, 128, kernel_size=1),
                ResBlock(128), nn.Conv1d(128, 128, kernel_size=1),
                ResBlock(128), nn.Conv1d(128, 64, kernel_size=1),
                ResBlock(64))
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ELU(), nn.Dropout(dropout), nn.Linear(32, 1))

        def forward(self, x):
            x = self.initial(x.unsqueeze(1))
            x = self.res_blocks(x)
            return self.head(self.pool(x)).squeeze(-1)

    def set_seed(s):
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    def predict(model, loader):
        model.eval(); p, y = [], []
        with torch.no_grad():
            for xb, yb in loader:
                p.extend(model(xb.to(device)).cpu().numpy()); y.extend(yb.numpy())
        return np.array(y), np.array(p)

    out = os.path.join(BASE, 'baselines_%s.csv' % tag)
    pdir = os.path.join(BASE, 'baselines_preds'); os.makedirs(pdir, exist_ok=True)
    cdir = os.path.join(BASE, 'baselines_curves'); os.makedirs(cdir, exist_ok=True)
    print('GA_BASE=%s  model=%s  device=%s  cfg=%s' % (BASE, a.model, device, c), flush=True)

    train_combined, test_combined = t.load_split()
    meta = list(t.META_COLS)
    cpg_cols = [col for col in train_combined.columns if col not in meta]
    if a.max_cpgs:
        cpg_cols = cpg_cols[:a.max_cpgs]
    print('train %d  test %d  CpGs %d' % (len(train_combined), len(test_combined), len(cpg_cols)), flush=True)

    HEADER = ['model', 'fold', 'seed', 'best_epoch', 'val_mae', 'test_mae', 'test_mse', 'test_r2', 'n_params', 'secs']

    def read_rows():
        if not os.path.exists(out):
            return []
        return [r for r in csv.DictReader(open(out)) if str(r.get('fold', '')).isdigit()]

    def append_row(row):
        with open(out, 'a', newline='') as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            w = csv.writer(fh)
            if os.fstat(fh.fileno()).st_size == 0:
                w.writerow(HEADER)
            w.writerow(row)
            fh.flush()
            fcntl.flock(fh, fcntl.LOCK_UN)

    done = set((int(r['fold']), int(r['seed'])) for r in read_rows())

    for fold in a.folds:
        todo = [s for s in a.seeds if (fold, s) not in done]
        if not todo:
            print('fold %d: all seeds done, skip' % fold, flush=True); continue
        fold_train, fold_val = t.fold_frames(train_combined, fold)

        if a.model == 'deepmage':
            X_train = fold_train[cpg_cols].astype('float')
            y_train = fold_train['age'].values.astype('float32')
            X_val = fold_val[cpg_cols].astype('float')
            X_test = test_combined[cpg_cols].astype('float')
            corr = X_train.corrwith(pd.Series(y_train, index=X_train.index)).abs()
            top = corr.nlargest(min(c['top_k'], len(cpg_cols))).index.tolist()
            Xtr, Xva, Xte = (X_train[top].values.astype('float32'), X_val[top].values.astype('float32'),
                             X_test[top].values.astype('float32'))
            print('\n=== fold %d  top-%d CpGs by |r| on fold training set ===' % (fold, len(top)), flush=True)
        else:
            Xtr = fold_train[cpg_cols].astype('float32').values
            Xva = fold_val[cpg_cols].astype('float32').values
            Xte = test_combined[cpg_cols].astype('float32').values
            print('\n=== fold %d  all %d CpGs ===' % (fold, len(cpg_cols)), flush=True)
        ytr = fold_train['age'].values.astype('float32')
        yva = fold_val['age'].values.astype('float32')
        yte = test_combined['age'].values.astype('float32')
        print('  train %s  val %s  test %s' % (Xtr.shape, Xva.shape, Xte.shape), flush=True)

        for s in todo:
            t0 = time.time()
            set_seed(s)
            train_loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=c['batch'], shuffle=True)
            val_loader = DataLoader(TensorDataset(torch.tensor(Xva), torch.tensor(yva)), batch_size=c['batch'], shuffle=False)
            test_loader = DataLoader(TensorDataset(torch.tensor(Xte), torch.tensor(yte)), batch_size=c['batch'], shuffle=False)
            if a.model == 'deepmage':
                model = DeepMAge(Xtr.shape[1], 512, c['dropout']).to(device)
            else:
                model = ResnetAge(Xtr.shape[1], c['dropout']).to(device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            optimizer = optim.Adam(model.parameters(), lr=c['lr'], weight_decay=c['wd'])
            criterion = nn.MSELoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=c['patience'], factor=c['factor'])

            best_val, best_epoch, best_state, curve = float('inf'), 0, None, []
            for epoch in range(c['epochs']):
                model.train(); losses = []
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                vt, vp = predict(model, val_loader)
                val_mae = mean_absolute_error(vt, vp)
                lr_now = optimizer.param_groups[0]['lr']
                scheduler.step(val_mae)
                curve.append((epoch + 1, lr_now, float(np.mean(losses)), val_mae))
                if val_mae < best_val:
                    best_val, best_epoch = val_mae, epoch + 1
                    best_state = copy.deepcopy(model.state_dict())
                if (epoch + 1) % 50 == 0 or epoch == 0:
                    print('  f%d s%d epoch %3d/%d  train_loss=%.3f  val_mae=%.3f  best=%.3f@%d  lr=%.1e  (%.0fs)' % (
                        fold, s, epoch + 1, c['epochs'], np.mean(losses), val_mae, best_val, best_epoch, lr_now,
                        time.time() - t0), flush=True)

            model.load_state_dict(best_state)
            tt, tp = predict(model, test_loader)
            mae, mse, r2 = mean_absolute_error(tt, tp), mean_squared_error(tt, tp), r2_score(tt, tp)
            append_row([a.model, fold, s, best_epoch, '%.4f' % best_val, '%.4f' % mae, '%.4f' % mse, '%.4f' % r2,
                        n_params, '%.0f' % (time.time() - t0)])
            pd.DataFrame({'true_age': tt, 'predicted_age': tp}).to_csv(
                os.path.join(pdir, '%s_fold%d_seed%d.csv' % (tag, fold, s)), index=False)
            with open(os.path.join(cdir, '%s_fold%d_seed%d.csv' % (tag, fold, s)), 'w', newline='') as ch:
                cw = csv.writer(ch); cw.writerow(['epoch', 'lr', 'train_loss', 'val_mae'])
                cw.writerows([(e, '%.6e' % l, '%.6f' % tl, '%.6f' % v) for e, l, tl, v in curve])
            print('  -> %s fold %d seed %d  best epoch %d  val %.4f  test MAE %.4f  (%d params, %.0fs)' % (
                a.model, fold, s, best_epoch, best_val, mae, n_params, time.time() - t0), flush=True)
            del model, optimizer, best_state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    rows = read_rows()
    v = [float(r['test_mae']) for r in rows]
    if v:
        m = sum(v) / len(v); sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** .5 if len(v) > 1 else 0.0
        print('\n%s: test MAE %.4f +/- %.4f  (n=%d)  -> %s' % (a.model, m, sd, len(v), out), flush=True)


if __name__ == '__main__':
    main()
