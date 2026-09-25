import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

BASE = os.environ.get('GA_BASE', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(BASE, 'cache')
DATA_DIR = os.path.join(BASE, 'data/all-organs4/all_organs')
CPG_LIST = os.path.join(BASE, 'data/multi_platform_cpgs.pkl')
CPG_INFO = os.path.join(BASE, 'data/cpgsite-info/GPL8490_HumanMethylation27_270596_v.1.2.csv')

META_COLS = ['age', 'gender', 'dataset', 'tissue_type']
K_FOLDS = 5
EDGE_DIM = 3


def log(m=''):
    print(m, flush=True)


def section(t):
    log('\n' + '=' * 74); log(t); log('=' * 74)


def atomic_save(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_information():
    info = pd.read_csv(CPG_INFO, skiprows=7, low_memory=False)
    info.dropna(subset=['Chr'], inplace=True)
    info[['start', 'end']] = (info.CPG_ISLAND_LOCATIONS.fillna('0:0-0')
                              .str.split(':').str[1].str.split('-', expand=True).astype(int))
    info['CPG_ISLAND'] = info['CPG_ISLAND'].astype(int)
    info['CPG_ISLAND_LEN'] = info.end - info.start
    info.MapInfo = info.MapInfo.astype(int)

    sc = MinMaxScaler()
    for col in ('MapInfo', 'TSS_Coordinate'):
        info['Normalized_' + col] = info.groupby('Chr')[col].transform(
            lambda x: sc.fit_transform(x.values.reshape(-1, 1)).flatten())
    info = pd.get_dummies(info, columns=['Gene_Strand'])
    for col in ('start', 'end', 'CPG_ISLAND_LEN'):
        info[col] = info.groupby('Chr')[col].transform(
            lambda x: sc.fit_transform(x.values.reshape(-1, 1)).flatten())
    info = pd.get_dummies(info, columns=['Next_Base'])
    info['Distance_to_TSS'] = info.groupby('Chr')['Distance_to_TSS'].transform(
        lambda x: sc.fit_transform(x.values.reshape(-1, 1)).flatten())
    info['Distance_to_TSS'] = info['Distance_to_TSS'].fillna(1)
    info = pd.get_dummies(info, columns=['SourceStrand'])
    chrom = info.Chr.tolist()
    info = pd.get_dummies(info, columns=['Chr'])
    info['Chr'] = chrom
    info.index = info.IlmnID
    return info


def frozen_order():
    p = os.path.join(BASE, 'cache', 'file_order.json')
    if os.path.exists(p):
        import json
        return json.load(open(p))['order']
    return None


def load_split(max_cpgs=None):
    cpgs = np.array(pd.read_pickle(CPG_LIST)).tolist()
    cpg_set = set(cpgs)
    frames_tr, frames_te = [], []
    order = frozen_order()
    for fn in (order if order is not None else os.listdir(DATA_DIR)):
        if not fn.endswith('.pkl'):
            continue
        fp = os.path.join(DATA_DIR, fn)
        if order is not None and not os.path.exists(fp):
            raise SystemExit(f'FATAL: pinned order lists {fn}, which is missing '
                             f'from {DATA_DIR}')
        d = pd.read_pickle(fp)
        a = d[[c for c in d.columns if c in cpg_set] + META_COLS]
        a = a[a['tissue_type'].str.lower().str.contains('blood')].dropna()
        if len(a) == 0:
            continue
        tr, te = train_test_split(a, test_size=0.2, random_state=42)
        frames_tr.append(tr); frames_te.append(te)

    train_combined = pd.concat(frames_tr)
    test_combined = pd.concat(frames_te)

    ref = os.path.join(BASE, 'data/reference_test_predictions.csv')
    if os.path.exists(ref):
        want = pd.read_csv(ref).true_age.values.astype(float)
        got = test_combined.age.values.astype(float)
        if len(got) == len(want) and np.allclose(got, want):
            log('  [ok]   split order matches the reference test predictions; '
                'fold indices are consistent with the reference run')
        else:
            raise SystemExit(
                '\n  FATAL: the reconstructed split does not match '
                'data/reference_test_predictions.csv.\n'
                '  os.listdir returned a different order than the reference run used, so\n'
                '  fold k here would not be fold k there.  Do not build caches until\n'
                '  this is resolved -- every downstream comparison depends on it.')
    else:
        log('  [WARN] reference predictions not found; fold indices unverified')

    selected = [c for c in train_combined.columns if c not in META_COLS]
    if max_cpgs:
        selected = selected[:max_cpgs]
        log(f'  [smoke] restricted to the first {max_cpgs} CpGs')
    return train_combined, test_combined, selected


def fold_train_frame(train_combined, fold):
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    tr_idx, va_idx = list(kf.split(train_combined))[fold]
    return (train_combined.iloc[tr_idx, :].sample(frac=1, random_state=42),
            train_combined.iloc[va_idx, :])


def make_graph_original(info, meth, thr_corr, thr_dist):
    chrom = np.array(info.Chr.values)
    genes = np.array(info.Symbol.values)
    chrom_adj = (chrom[:, None] == chrom).astype(np.float32)
    genes_adj = (genes[:, None] == genes).astype(np.float32)
    dist = squareform(pdist(info.MapInfo.values.reshape(-1, 1)))
    adj = np.corrcoef(meth.to_numpy(), rowvar=False)

    sec, ter = thr_corr - 0.02, thr_corr - 0.04
    src, dst = np.where(
        (((chrom_adj == 1) & ((np.abs(adj) > sec) |
                              ((abs(dist) < thr_dist) & (np.abs(adj) > ter))))
         | (np.abs(adj) > thr_corr))
        & (np.arange(adj.shape[0])[:, None] != np.arange(adj.shape[1])))
    w = np.column_stack([adj[src, dst], chrom_adj[src, dst], genes_adj[src, dst]])
    return (torch.stack([torch.tensor(src, dtype=torch.int64),
                         torch.tensor(dst, dtype=torch.int64)]),
            torch.tensor(w, dtype=torch.float32).reshape(-1, EDGE_DIM))


def make_graph_chunked(info, meth, thr_corr, thr_dist, chunk=1024):
    chrom = np.asarray(info.Chr.values)
    genes = np.asarray(info.Symbol.values)
    mapinfo = info.MapInfo.values.astype(np.float64)
    n = len(chrom)

    X = meth.to_numpy().astype(np.float64)
    Z = X - X.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0, ddof=1, keepdims=True)
    sd[sd == 0] = np.inf
    Z = Z / sd
    denom = X.shape[0] - 1

    sec, ter = thr_corr - 0.02, thr_corr - 0.04
    src_l, dst_l, corr_l, chr_l, gene_l = [], [], [], [], []

    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        corr = (Z[:, a:b].T @ Z) / denom
        np.clip(corr, -1.0, 1.0, out=corr)
        acorr = np.abs(corr)

        same_chr = chrom[a:b, None] == chrom[None, :]
        near = np.abs(mapinfo[a:b, None] - mapinfo[None, :]) < thr_dist
        keep = ((same_chr & ((acorr > sec) | (near & (acorr > ter))))
                | (acorr > thr_corr))
        keep[np.arange(b - a), np.arange(a, b)] = False

        i, j = np.where(keep)
        if len(i) == 0:
            continue
        src_l.append(i + a)
        dst_l.append(j)
        corr_l.append(corr[i, j])
        chr_l.append(same_chr[i, j].astype(np.float32))
        gene_l.append((genes[i + a] == genes[j]).astype(np.float32))

    src = np.concatenate(src_l); dst = np.concatenate(dst_l)
    w = np.column_stack([np.concatenate(corr_l),
                         np.concatenate(chr_l),
                         np.concatenate(gene_l)])
    return (torch.stack([torch.tensor(src, dtype=torch.int64),
                         torch.tensor(dst, dtype=torch.int64)]),
            torch.tensor(w, dtype=torch.float32).reshape(-1, EDGE_DIM))


def compute_deg(edge_index, n_train, mode='legacy'):
    dst = edge_index[1]
    if mode == 'legacy':
        max_degree = int(dst.max())
        return torch.bincount(dst, minlength=max_degree + 1) * n_train
    indeg = torch.bincount(dst, minlength=int(dst.max()) + 1)
    return torch.bincount(indeg, minlength=int(indeg.max()) + 1)


def cache_path(fold, thr_corr, thr_dist, tag):
    return os.path.join(CACHE, f'graph_fold{fold}_corr{thr_corr}_dist{thr_dist:g}{tag}.pt')


def build_one(fold, thr_corr, thr_dist, info_f, train_combined, selected,
              args, tag):
    path = cache_path(fold, thr_corr, thr_dist, tag)
    if os.path.exists(path) and not args.force:
        try:
            g = torch.load(path, map_location='cpu', weights_only=False)
            log(f'  [cached] fold {fold} corr={thr_corr}: '
                f'{g["edge_index"].shape[1]:,} edges  ->  {path}')
            return g
        except Exception as e:
            log(f'  [stale]  {path} unreadable ({e}); rebuilding')

    ftrain, fval = fold_train_frame(train_combined, fold)
    X = ftrain[selected].astype('float')
    log(f'  fold {fold}: train={len(ftrain)} val={len(fval)} cpgs={len(selected)}')

    t0 = time.time()
    edge_index, edge_attr = make_graph_chunked(
        info_f, X, thr_corr, thr_dist, chunk=args.chunk)
    secs = time.time() - t0

    deg = compute_deg(edge_index, len(ftrain), mode=args.deg_mode)
    n = len(selected)
    g = dict(edge_index=edge_index, edge_attr=edge_attr, deg=deg,
             fold=fold, thr_corr=thr_corr, thr_dist=thr_dist,
             n_cpgs=n, n_train=len(ftrain), n_val=len(fval),
             deg_mode=args.deg_mode, cpgs=selected, build_secs=round(secs, 1))
    atomic_save(g, path)

    e = edge_index.shape[1]
    log(f'  [built]  fold {fold} corr={thr_corr}: {e:,} edges '
        f'(density {e / (n * (n - 1)) * 100:.3f}%, mean degree {e / n:.1f}) '
        f'in {secs:.0f}s  ->  {path}')
    return g


def verify(info_f, train_combined, selected, args):
    section('verify: chunked implementation vs. the original full-matrix one')
    ftrain, _ = fold_train_frame(train_combined, args.folds[0])
    X = ftrain[selected].astype('float')
    thr = args.thr_corr[0]

    t0 = time.time()
    ei_a, ea_a = make_graph_original(info_f, X, thr, args.thr_dist)
    t_orig = time.time() - t0
    t0 = time.time()
    ei_b, ea_b = make_graph_chunked(info_f, X, thr, args.thr_dist, chunk=args.chunk)
    t_chunk = time.time() - t0

    log(f'  original: {ei_a.shape[1]:,} edges in {t_orig:.1f}s')
    log(f'  chunked : {ei_b.shape[1]:,} edges in {t_chunk:.1f}s')

    same_shape = ei_a.shape == ei_b.shape
    same_edges = same_shape and torch.equal(ei_a, ei_b)
    max_dw = float((ea_a - ea_b).abs().max()) if same_shape else float('nan')
    log(f'  identical edge list : {same_edges}')
    log(f'  max |edge_attr diff|: {max_dw:.3e}')
    ok = same_edges and max_dw < 1e-5
    log(f'\n  {"PASS -- chunked build is equivalent" if ok else "FAIL -- do not use the cache"}')
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--folds', type=int, nargs='+', default=[2])
    ap.add_argument('--thr-corr', type=float, nargs='+', default=[0.70])
    ap.add_argument('--thr-dist', type=float, default=1e5)
    ap.add_argument('--chunk', type=int, default=1024)
    ap.add_argument('--deg-mode', choices=['legacy', 'histogram'], default='legacy')
    ap.add_argument('--max-cpgs', type=int, default=None,
                    help='smoke-test with a CpG subset')
    ap.add_argument('--verify', action='store_true',
                    help='check the chunked build against the original implementation')
    ap.add_argument('--force', action='store_true', help='rebuild even if cached')
    args = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(os.path.join(BASE, 'results'), exist_ok=True)
    log(f'BASE  = {BASE}')
    log(f'CACHE = {CACHE}')

    section('loading annotation and split')
    info = load_information()
    train_combined, test_combined, selected = load_split(args.max_cpgs)
    info_f = info[info.IlmnID.isin(selected)]
    selected = [c for c in selected if c in set(info_f.IlmnID)]
    info_f = info_f.loc[selected]
    log(f'  train_combined={len(train_combined)}  test={len(test_combined)}  '
        f'cpgs={len(selected)}')

    tag = f'_n{len(selected)}' if args.max_cpgs else ''

    if args.verify:
        ok = verify(info_f, train_combined, selected, args)
        if not ok:
            raise SystemExit(1)

    section('building / loading caches')
    index = []
    for fold in args.folds:
        for thr in args.thr_corr:
            g = build_one(fold, thr, args.thr_dist, info_f, train_combined,
                          selected, args, tag)
            index.append(dict(fold=fold, thr_corr=thr, thr_dist=args.thr_dist,
                              n_edges=int(g['edge_index'].shape[1]),
                              n_cpgs=int(g['n_cpgs']), n_train=int(g['n_train']),
                              deg_mode=g['deg_mode'],
                              path=cache_path(fold, thr, args.thr_dist, tag)))

    idx_path = os.path.join(CACHE, 'graph_index.csv')
    df = pd.DataFrame(index)
    if os.path.exists(idx_path):
        old = pd.read_csv(idx_path)
        df = (pd.concat([old, df])
              .drop_duplicates(subset=['fold', 'thr_corr', 'thr_dist', 'n_cpgs', 'deg_mode'],
                               keep='last'))
    df.to_csv(idx_path, index=False)

    section('summary')
    log(df.to_string(index=False))
    log(f'\n-> {idx_path}')

    if len(args.folds) > 1:
        section('cross-fold graph stability')
        keys = [(f, args.thr_corr[0]) for f in args.folds]
        sets = {}
        for f, thr in keys:
            g = torch.load(cache_path(f, thr, args.thr_dist, tag),
                           map_location='cpu', weights_only=False)
            ei = g['edge_index'].numpy()
            sets[f] = set(map(tuple, ei.T.tolist()))
        rows = []
        fs = sorted(sets)
        for i, a in enumerate(fs):
            for b in fs[i + 1:]:
                inter = len(sets[a] & sets[b]); union = len(sets[a] | sets[b])
                rows.append(dict(fold_a=a, fold_b=b, edges_a=len(sets[a]),
                                 edges_b=len(sets[b]), shared=inter,
                                 jaccard=inter / union))
        sdf = pd.DataFrame(rows)
        sdf.to_csv(os.path.join(BASE, 'results', 'c5_graph_stability.csv'),
                   index=False)
        log(sdf.round(4).to_string(index=False))
        log(f'\n  mean pairwise Jaccard = {sdf.jaccard.mean():.4f}')
        log('  -> results/c5_graph_stability.csv')


if __name__ == '__main__':
    main()
