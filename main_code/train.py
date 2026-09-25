import argparse
import json
import os
import random
import signal
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GATConv, GCNConv, GINConv, PNAConv, SAGEConv

warnings.filterwarnings('ignore')

BASE = os.environ.get('GA_BASE', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(BASE, 'cache')
RUNS = os.path.join(BASE, 'runs')
DATA_DIR = os.path.join(BASE, 'data/all-organs4/all_organs')
CPG_LIST = os.path.join(BASE, 'data/multi_platform_cpgs.pkl')
CPG_INFO = os.path.join(BASE, 'data/cpgsite-info/GPL8490_HumanMethylation27_270596_v.1.2.csv')

META_COLS = ['age', 'gender', 'dataset', 'tissue_type']
POS_FEATURES = ['CPG_ISLAND', 'CPG_ISLAND_LEN', 'Distance_to_TSS',
                'Next_Base_A', 'Next_Base_C', 'Next_Base_T',
                'start', 'end', 'Normalized_MapInfo']
K_FOLDS = 5
SEQ_LEN = 122
AGE_NAMES = ['0', '0-20', '20-45', '45-55', '55-65', '65-75', '75-80', '80+']

STOP = {'flag': False, 'sig': None}
LOGF = {'fh': None}


def log(m=''):
    print(m, flush=True)
    if LOGF['fh']:
        LOGF['fh'].write(str(m) + '\n')
        LOGF['fh'].flush()


def section(t):
    log('\n' + '=' * 74); log(t); log('=' * 74)


def _on_signal(sig, frame):
    STOP['flag'] = True
    STOP['sig'] = sig
    log(f'\n  [signal {sig}] will checkpoint and exit after this epoch')


def atomic_torch_save(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def sweep_tmp(run_dir):
    for f in os.listdir(run_dir):
        if f.endswith('.tmp'):
            try:
                os.remove(os.path.join(run_dir, f))
                log(f'  [cleanup] removed stale {f}')
            except OSError:
                pass


def atomic_write_text(text, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def rng_state():
    return dict(python=random.getstate(),
                numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(st):
    random.setstate(st['python'])
    np.random.set_state(st['numpy'])
    torch.set_rng_state(st['torch'].cpu() if hasattr(st['torch'], 'cpu') else st['torch'])
    if st.get('cuda') is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu() for s in st['cuda']])
        except Exception as e:
            log(f'  [warn] could not restore CUDA RNG ({e})')


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


def load_split():
    cpg_set = set(np.array(pd.read_pickle(CPG_LIST)).tolist())
    tr_f, te_f = [], []
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
        t1, t2 = train_test_split(a, test_size=0.2, random_state=42)
        tr_f.append(t1); te_f.append(t2)
    train_combined = pd.concat(tr_f)
    test_combined = pd.concat(te_f)

    ref = os.path.join(BASE, 'data/reference_test_predictions.csv')
    if os.path.exists(ref):
        want = pd.read_csv(ref).true_age.values.astype(float)
        got = test_combined.age.values.astype(float)
        if not (len(got) == len(want) and np.allclose(got, want)):
            raise SystemExit('FATAL: split order differs from data/reference_test_predictions.csv; '
                             'fold indices would not match. Aborting.')
    return train_combined, test_combined


def fold_frames(train_combined, fold):
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    tr_idx, va_idx = list(kf.split(train_combined))[fold]
    return (train_combined.iloc[tr_idx, :].sample(frac=1, random_state=42),
            train_combined.iloc[va_idx, :])


def stat_features(seqs):
    out = np.zeros((len(seqs), 8), dtype=np.float32)
    for i, seq in enumerate(seqs):
        if not isinstance(seq, str) or len(seq) == 0:
            continue
        s = seq.replace('[CG]', 'CG').upper()
        n = len(s)
        up, down = s[:60], s[62:]
        ctx = s[55:65] if n >= 65 else s
        out[i] = [(s.count('G') + s.count('C')) / max(n, 1),
                  s.count('CG') / max(n - 1, 1),
                  (up.count('G') + up.count('C')) / max(len(up), 1),
                  (down.count('G') + down.count('C')) / max(len(down), 1),
                  ctx.count('A') / max(len(ctx), 1),
                  ctx.count('T') / max(len(ctx), 1),
                  ctx.count('C') / max(len(ctx), 1),
                  ctx.count('G') / max(len(ctx), 1)]
    return out


def onehot_features(seqs):
    idx = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    out = np.zeros((len(seqs), SEQ_LEN, 4), dtype=np.float32)
    for i, seq in enumerate(seqs):
        if not isinstance(seq, str) or len(seq) == 0:
            continue
        s = seq.replace('[CG]', 'CG').upper()[:SEQ_LEN]
        for j, nt in enumerate(s):
            if nt in idx:
                out[i, j, idx[nt]] = 1.0
            else:
                out[i, j, :] = 0.25
    return out


def build_seq(variant, info_f, seed):
    if variant == 'none':
        return None, 'no sequence input'
    seqs = info_f['TopGenomicSeq'].values
    if variant == 'cnn':
        return torch.tensor(onehot_features(seqs)), f'one-hot [{len(seqs)}, {SEQ_LEN}, 4]'
    if variant == 'cnn_perm':
        oh = onehot_features(seqs)
        rng = np.random.default_rng(10_000 + seed)
        return (torch.tensor(oh[rng.permutation(len(oh))]),
                f'one-hot, row-permuted across CpGs [{len(seqs)}, {SEQ_LEN}, 4]')
    if variant == 'cnn_rand':
        rng = np.random.default_rng(30_000 + seed)
        bases = rng.integers(0, 4, size=(len(seqs), SEQ_LEN))
        return (torch.tensor(np.eye(4, dtype=np.float32)[bases]),
                f'one-hot, i.i.d. random bases per CpG [{len(seqs)}, {SEQ_LEN}, 4]')

    feats = stat_features(seqs)
    if variant == 'perm':
        rng = np.random.default_rng(10_000 + seed)
        feats = feats[rng.permutation(len(feats))]
        desc = 'statistics, row-permuted across CpGs'
    elif variant == 'rand':
        rng = np.random.default_rng(20_000 + seed)
        feats = rng.normal(feats.mean(0), feats.std(0) + 1e-8,
                           size=feats.shape).astype(np.float32)
        desc = 'gaussian noise matched to feature moments'
    else:
        desc = '8 sequence statistics'
    return torch.tensor(feats), f'{desc} [{feats.shape[0]}, 8]'


USES_GATE = {'stat', 'cnn', 'gate_only', 'perm', 'rand', 'noposition', 'cnn_perm', 'cnn_rand'}
USES_PROJ = {'stat', 'cnn', 'proj_only', 'perm', 'rand', 'noposition', 'cnn_perm', 'cnn_rand'}


def pna_in_dim(variant, original_dim, seq_stat_dim=8):
    if variant == 'none':
        return original_dim
    if variant == 'gate_only':
        return original_dim
    if variant == 'concat':
        return original_dim + seq_stat_dim
    if variant == 'noposition':
        return 1 + 2
    return original_dim + 2


class Net(nn.Module):
    def __init__(self, deg, variant, original_dim, num_cpgs,
                 gnn='pna', edge_dim=3, mlp_first=1024, seq_embed_dim=32,
                 leaky=0.0):
        super().__init__()
        self.variant = variant
        self.gnn = gnn
        self.leaky = leaky
        self.in_dim = pna_in_dim(variant, original_dim)

        if variant in ('cnn', 'cnn_perm', 'cnn_rand'):
            self.seq_cnn = nn.Sequential(
                nn.Conv1d(4, 16, 3, padding=1), nn.BatchNorm1d(16), nn.ReLU(),
                nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, seq_embed_dim, 7, padding=3),
                nn.BatchNorm1d(seq_embed_dim), nn.ReLU())
            self.seq_pool = nn.AdaptiveMaxPool1d(1)
            gate_in, proj_hidden = seq_embed_dim, 16
        else:
            gate_in, proj_hidden = 8, 8

        if variant in USES_GATE:
            self.importance_gate = nn.Sequential(
                nn.Linear(gate_in, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
        if variant in USES_PROJ:
            self.seq_proj = nn.Sequential(
                nn.Linear(gate_in, proj_hidden), nn.ReLU(), nn.Linear(proj_hidden, 2))

        aggr = ['mean', 'max', 'std', 'min']
        scal = ['identity', 'amplification', 'attenuation']
        if gnn == 'pna':
            self.conv = PNAConv(self.in_dim, 1, aggregators=aggr, scalers=scal,
                                deg=deg, edge_dim=edge_dim, towers=1,
                                pre_layers=1, post_layers=1, divide_input=False)
        elif gnn == 'gcn':
            self.conv = GCNConv(self.in_dim, 1)
        elif gnn == 'gat':
            self.conv = GATConv(self.in_dim, 1, heads=1, edge_dim=edge_dim)
        elif gnn == 'sage':
            self.conv = SAGEConv(self.in_dim, 1)
        elif gnn == 'gin':
            self.conv = GINConv(nn.Sequential(nn.Linear(self.in_dim, 16),
                                              nn.ReLU(), nn.Linear(16, 1)))
        elif gnn == 'nograph':
            self.conv = nn.Linear(self.in_dim, 1)
        elif gnn == 'mlp':
            self.conv = None
        else:
            raise ValueError(gnn)

        self.mlp = nn.Sequential(
            nn.Linear(num_cpgs, mlp_first), nn.ReLU(),
            nn.Linear(mlp_first, 656), nn.SELU(),
            nn.Linear(656, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 124), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(124, 64), nn.SELU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 8), nn.ReLU(),
            nn.Linear(8, 1))

    def encode_seq(self, x_seq):
        if self.variant not in ('cnn', 'cnn_perm', 'cnn_rand'):
            return x_seq
        h = self.seq_cnn(x_seq.permute(0, 2, 1))
        return self.seq_pool(h).squeeze(-1)

    def build_nodes(self, x_orig, x_seq):
        if self.variant == 'none':
            return x_orig
        h = self.encode_seq(x_seq)
        meth = x_orig[:, 0:1]
        if self.variant == 'concat':
            return torch.cat([x_orig, h], dim=-1)
        if self.variant in USES_GATE:
            meth = meth * self.importance_gate(h)
        parts = [meth]
        if self.variant != 'noposition':
            parts.append(x_orig[:, 1:])
        if self.variant in USES_PROJ:
            parts.append(self.seq_proj(h))
        return torch.cat(parts, dim=-1)

    def forward(self, x_orig, x_seq, edge_index, edge_attr, n_graphs, n_nodes):
        x = self.build_nodes(x_orig, x_seq)
        if self.gnn == 'pna' or self.gnn == 'gat':
            x = self.conv(x, edge_index, edge_attr)
        elif self.gnn == 'gcn':
            x = self.conv(x, edge_index, edge_attr[:, 0].abs())
        elif self.gnn == 'nograph':
            x = self.conv(x)
        elif self.gnn == 'mlp':
            x = x[:, 0:1]
        else:
            x = self.conv(x, edge_index)
        x = F.leaky_relu(x, self.leaky) if self.leaky > 0 else F.relu(x)
        return self.mlp(x.view(n_graphs, n_nodes)).flatten()


def replicate_graph(edge_index, edge_attr, n_nodes, b):
    if b == 1:
        return edge_index, edge_attr
    offs = (torch.arange(b, device=edge_index.device) * n_nodes).repeat_interleave(
        edge_index.shape[1])
    ei = edge_index.repeat(1, b) + offs
    return ei, edge_attr.repeat(b, 1)


def age_group(a):
    if a <= 0: return 0
    if a <= 20: return 1
    if a <= 45: return 2
    if a <= 55: return 3
    if a <= 65: return 4
    if a <= 75: return 5
    if a <= 80: return 6
    return 7


def group_metrics(truth, pred):
    rows = []
    for g in range(8):
        m = np.array([age_group(t) == g for t in truth])
        if m.sum() == 0:
            rows.append(dict(age_group=AGE_NAMES[g], n=0, mae=None, mse=None, r2=None))
            continue
        t, p = truth[m], pred[m]
        rows.append(dict(age_group=AGE_NAMES[g], n=int(m.sum()),
                         mae=mean_absolute_error(t, p), mse=mean_squared_error(t, p),
                         r2=r2_score(t, p) if m.sum() > 1 else None))
    return pd.DataFrame(rows)


def run_epoch(model, X, y, seq, edge_index, edge_attr, n_nodes, pos,
              optimizer=None, batch_size=1, device='cpu', amp=False,
              grad_clip=0.0):
    train = optimizer is not None
    model.train() if train else model.eval()
    crit = nn.MSELoss()
    n = X.shape[0]
    total_loss, preds = 0.0, []
    ei_cache = {}

    for i in range(0, n, batch_size):
        b = min(batch_size, n - i)
        meth = X[i:i + b].to(device, non_blocking=True)
        x_orig = torch.cat([meth.reshape(-1, 1),
                            pos.repeat(b, 1)], dim=1)
        x_seq = None if seq is None else (seq if b == 1 else seq.repeat(b, *([1] * (seq.dim() - 1))))
        if b not in ei_cache:
            ei_cache[b] = replicate_graph(edge_index, edge_attr, n_nodes, b)
        ei, ea = ei_cache[b]
        yb = y[i:i + b].to(device, non_blocking=True)

        ctx = torch.autocast('cuda', dtype=torch.bfloat16) if (amp and device == 'cuda') \
            else torch.autocast('cpu', enabled=False)
        if train:
            optimizer.zero_grad(set_to_none=True)
            with ctx:
                out = model(x_orig, x_seq, ei, ea, b, n_nodes)
                loss = crit(out.float(), yb)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += float(loss) * b
        else:
            with torch.no_grad(), ctx:
                out = model(x_orig, x_seq, ei, ea, b, n_nodes)
        preds.append(out.detach().float().cpu().numpy())

    pred = np.concatenate(preds)
    truth = y.cpu().numpy()
    return (float(total_loss / n) if train else None,
            float(mean_absolute_error(truth, pred)),
            float(mean_squared_error(truth, pred)),
            float(r2_score(truth, pred)), truth, pred)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant', required=True,
                    choices=['none', 'stat', 'cnn', 'gate_only', 'proj_only',
                             'concat', 'perm', 'rand', 'noposition', 'cnn_perm', 'cnn_rand'])
    ap.add_argument('--fold', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--thr-corr', type=float, default=0.70)
    ap.add_argument('--thr-dist', type=float, default=1e5)
    ap.add_argument('--gnn', default='pna',
                    choices=['pna', 'gcn', 'gat', 'sage', 'gin', 'nograph', 'mlp'])
    ap.add_argument('--mlp-first', type=int, default=1024)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--max-epochs', type=int, default=None,
                    help='stop early (for timing); the run stays resumable')
    ap.add_argument('--lr', type=float, default=0.5e-3 * 0.4 ** 2)
    ap.add_argument('--weight-decay', type=float, default=6e-4)
    ap.add_argument('--factor', type=float, default=0.4)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--min-lr', type=float, default=1e-11)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--amp', action='store_true', help='bfloat16 autocast on CUDA')
    ap.add_argument('--shuffle', action='store_true',
                    help='reshuffle training order each epoch (default: fixed order)')
    ap.add_argument('--max-cpgs', type=int, default=None, help='smoke test')
    ap.add_argument('--tag', default='', help='suffix for the run directory')
    ap.add_argument('--force-restart', action='store_true')
    ap.add_argument('--grad-clip', type=float, default=0.0,
                    help='clip gradient norm (0 = off); guards against the '
                         'first steps killing every ReLU')
    ap.add_argument('--leaky', type=float, default=0.0,
                    help='negative slope for the post-convolution activation '
                         '(0 = plain ReLU)')
    ap.add_argument('--keep-ckpt', action='store_true',
                    help='keep ckpt.pt after the run finishes (default: delete it; '
                         'it is resume scratch state, not a result)')
    ap.add_argument('--min-free-gb', type=float, default=3.0,
                    help='refuse to start if the filesystem has less free space')
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    tag = f'_n{args.max_cpgs}' if args.max_cpgs else ''
    run_id = (f'{args.variant}_fold{args.fold}_seed{args.seed}'
              f'_corr{args.thr_corr}_{args.gnn}{tag}{args.tag}')
    run_dir = os.path.join(RUNS, run_id)
    os.makedirs(run_dir, exist_ok=True)
    LOGF['fh'] = open(os.path.join(run_dir, 'log.txt'), 'a')

    CKPT = os.path.join(run_dir, 'ckpt.pt')
    BEST = os.path.join(run_dir, 'best_val.pt')
    METRICS = os.path.join(run_dir, 'metrics.csv')
    STATUS = os.path.join(run_dir, 'status.json')

    section(f'run {run_id}')
    log(f'  started {time.strftime("%Y-%m-%d %H:%M:%S")}')
    atomic_write_text(json.dumps(vars(args), indent=2, default=str),
                      os.path.join(run_dir, 'config.json'))

    if os.path.exists(STATUS) and not args.force_restart:
        try:
            st = json.load(open(STATUS))
            if st.get('state') == 'done':
                log(f'  already finished at epoch {st.get("epoch")} '
                    f'(best val MAE {st.get("best_val_mae"):.4f}); nothing to do')
                return
        except Exception:
            pass

    sweep_tmp(run_dir)
    avail = free_gb(BASE)
    log(f'  disk free: {avail:.1f} GB')
    if avail < args.min_free_gb:
        raise SystemExit(
            f'FATAL: only {avail:.1f} GB free, need at least {args.min_free_gb} GB.\n'
            f'  A run needs ~0.35 GB while training and ~0.09 GB once finished.\n'
            f'  Free space, or lower the bar with --min-free-gb.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu = torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'
    log(f'  device={device} ({gpu})  torch={torch.__version__}')
    set_seed(args.seed)

    gp = os.path.join(CACHE, f'graph_fold{args.fold}_corr{args.thr_corr}'
                             f'_dist{args.thr_dist:g}{tag}.pt')
    if not os.path.exists(gp):
        raise SystemExit(f'FATAL: graph cache missing: {gp}\n'
                         f'  run:  python build_graph.py --folds {args.fold} '
                         f'--thr-corr {args.thr_corr}')
    g = torch.load(gp, map_location='cpu', weights_only=False)
    selected = g['cpgs']
    n_nodes = len(selected)
    edge_index = g['edge_index'].to(device)
    edge_attr = g['edge_attr'].to(device)
    deg = g['deg']
    log(f'  graph: {n_nodes} nodes, {edge_index.shape[1]:,} edges, deg_mode={g["deg_mode"]}')

    info = load_information()
    train_combined, test_combined = load_split()
    ftrain, fval = fold_frames(train_combined, args.fold)
    info_f = info.loc[selected]

    pos = torch.tensor(info_f[POS_FEATURES].to_numpy().astype(np.float32)).to(device)
    original_dim = 1 + len(POS_FEATURES)

    def mat(df):
        return (torch.tensor(df[selected].to_numpy().astype(np.float32)),
                torch.tensor(df.age.to_numpy().astype(np.float32)))

    Xtr, ytr = mat(ftrain); Xva, yva = mat(fval); Xte, yte = mat(test_combined)
    log(f'  samples: train={len(ytr)} val={len(yva)} test={len(yte)}')

    seq, seq_desc = build_seq(args.variant, info_f, args.seed)
    if seq is not None:
        seq = seq.to(device)
    log(f'  sequence input: {seq_desc}')

    model = Net(deg, args.variant, original_dim, n_nodes, gnn=args.gnn,
                edge_dim=edge_attr.shape[1], mlp_first=args.mlp_first,
                leaky=args.leaky).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    n_seq_par = sum(p.numel() for n, p in model.named_parameters()
                    if n.startswith(('seq_', 'importance_gate')))
    log(f'  PNA input dim={model.in_dim}  params={n_par:,}  '
        f'(sequence pathway {n_seq_par:,})')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.factor,
                                  patience=args.patience, min_lr=args.min_lr)

    start_epoch, best_val, best_epoch, best_test = 1, float('inf'), 0, None
    if os.path.exists(CKPT) and not args.force_restart:
        try:
            ck = torch.load(CKPT, map_location=device, weights_only=False)
            model.load_state_dict(ck['model'])
            optimizer.load_state_dict(ck['optimizer'])
            scheduler.load_state_dict(ck['scheduler'])
            restore_rng(ck['rng'])
            start_epoch = ck['epoch'] + 1
            best_val = ck['best_val']; best_epoch = ck['best_epoch']
            best_test = ck.get('best_test')
            log(f'  [resume] from epoch {start_epoch} '
                f'(best val {best_val:.4f} @ epoch {best_epoch})')
        except Exception as e:
            log(f'  [warn] checkpoint unreadable ({e}); starting fresh')
    else:
        log('  [fresh] no checkpoint found')

    if not os.path.exists(METRICS):
        atomic_write_text('epoch,lr,secs,train_loss,train_mae,val_mae,val_mse,val_r2,'
                          'test_mae,test_mse,test_r2\n', METRICS)

    last_epoch = args.epochs if args.max_epochs is None else min(
        args.epochs, start_epoch + args.max_epochs - 1)
    if start_epoch > args.epochs:
        log('  nothing left to do')
    section(f'training epochs {start_epoch} .. {last_epoch}  (target {args.epochs})')

    order = torch.arange(len(ytr))
    for epoch in range(start_epoch, last_epoch + 1):
        t0 = time.time()
        if args.shuffle:
            order = torch.randperm(len(ytr))
        tl, tr_mae, _, _, _, _ = run_epoch(
            model, Xtr[order], ytr[order], seq, edge_index, edge_attr, n_nodes, pos,
            optimizer=optimizer, batch_size=args.batch_size, device=device,
            amp=args.amp, grad_clip=args.grad_clip)
        _, v_mae, v_mse, v_r2, v_t, v_p = run_epoch(
            model, Xva, yva, seq, edge_index, edge_attr, n_nodes, pos,
            batch_size=args.batch_size, device=device, amp=args.amp)
        _, s_mae, s_mse, s_r2, s_t, s_p = run_epoch(
            model, Xte, yte, seq, edge_index, edge_attr, n_nodes, pos,
            batch_size=args.batch_size, device=device, amp=args.amp)

        scheduler.step(v_mae)
        lr = optimizer.param_groups[0]['lr']
        secs = time.time() - t0

        with open(METRICS, 'a') as f:
            f.write(f'{epoch},{lr:.6e},{secs:.1f},{tl:.6f},{tr_mae:.6f},'
                    f'{v_mae:.6f},{v_mse:.6f},{v_r2:.6f},'
                    f'{s_mae:.6f},{s_mse:.6f},{s_r2:.6f}\n')

        improved = v_mae < best_val
        if improved:
            best_val, best_epoch, best_test = v_mae, epoch, s_mae
            atomic_torch_save(dict(model=model.state_dict(), epoch=epoch,
                                   val_mae=v_mae, test_mae=s_mae, config=vars(args)), BEST)
            pd.DataFrame({'true_age': s_t, 'predicted_age': s_p}).to_csv(
                os.path.join(run_dir, 'test_predictions.csv'), index=False)
            pd.DataFrame({'true_age': v_t, 'predicted_age': v_p}).to_csv(
                os.path.join(run_dir, 'val_predictions.csv'), index=False)
            group_metrics(s_t, s_p).to_csv(
                os.path.join(run_dir, 'test_age_groups.csv'), index=False)

        atomic_torch_save(dict(epoch=epoch, model=model.state_dict(),
                               optimizer=optimizer.state_dict(),
                               scheduler=scheduler.state_dict(), rng=rng_state(),
                               best_val=best_val, best_epoch=best_epoch,
                               best_test=best_test, config=vars(args)), CKPT)

        done = epoch >= args.epochs
        atomic_write_text(json.dumps(dict(
            run_id=run_id, state='done' if done else 'running', epoch=int(epoch),
            target_epochs=int(args.epochs), best_val_mae=float(best_val),
            best_epoch=int(best_epoch),
            test_mae_at_best_val=None if best_test is None else float(best_test),
            last_val_mae=float(v_mae), last_test_mae=float(s_mae),
            train_loss=float(tl), lr=float(lr), secs_per_epoch=round(float(secs), 1),
            gpu=gpu, torch_version=torch.__version__,
            eta_hours=round(float(secs) * (args.epochs - epoch) / 3600, 2),
            updated=time.strftime('%Y-%m-%d %H:%M:%S')),
            indent=2, default=float), STATUS)

        log(f'  ep {epoch:03d}/{args.epochs} | {secs:6.1f}s | lr {lr:.2e} | '
            f'train {tl:8.3f} | val {v_mae:6.3f}{" *" if improved else "  "} | '
            f'test {s_mae:6.3f} | best val {best_val:.3f}@{best_epoch}')

        if STOP['flag']:
            log(f'\n  interrupted after epoch {epoch}; checkpoint saved.')
            log(f'  resume with the identical command.')
            LOGF['fh'].close()
            sys.exit(130)

    section('summary')
    log(f'  best val MAE      {best_val:.4f}  (epoch {best_epoch})')
    log(f'  test MAE at that  {best_test:.4f}')

    if (not args.keep_ckpt and os.path.exists(CKPT)
            and epoch >= args.epochs):
        size = os.path.getsize(CKPT) / 1e6
        os.remove(CKPT)
        log(f'  removed ckpt.pt ({size:.0f} MB of resume state; '
            f'pass --keep-ckpt to retain it)')

    log(f'  -> {run_dir}')
    LOGF['fh'].close()


if __name__ == '__main__':
    main()
