import itertools
import os

BASE = os.environ.get('GA_BASE', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, 'configs')
os.makedirs(OUT, exist_ok=True)

FOLDS = [2, 0, 1, 3, 4]
SEEDS = [0, 1, 2]
VARIANTS = ['none', 'cnn', 'stat']
EXTRA = os.environ.get('GA_TRAIN_EXTRA', '')


def write(name, rows):
    path = os.path.join(OUT, name)
    with open(path, 'w') as f:
        for r in rows:
            f.write(' '.join(r) + (f' {EXTRA}' if EXTRA else '') + '\n')
    print(f'  {name:<16} {len(rows):>4} runs   -> {path}')
    return len(rows)


def a2():
    return [(f'--variant {v}', f'--fold {f}', f'--seed {s}')
            for f, s, v in itertools.product(FOLDS, SEEDS, VARIANTS)]


def b_ablation():
    variants = ['gate_only', 'proj_only', 'concat', 'perm', 'rand', 'noposition']
    return [(f'--variant {v}', '--fold 2', f'--seed {s}')
            for s, v in itertools.product(SEEDS, variants)]


def c1_threshold():
    return [('--variant stat', '--fold 2', '--seed 0', f'--thr-corr {t}')
            for t in [0.60, 0.65, 0.75, 0.80]]


def c4_operator():
    return [('--variant stat', '--fold 2', f'--seed {s}', f'--gnn {g}')
            for s, g in itertools.product(SEEDS, ['nograph', 'gcn', 'sage', 'gat', 'gin'])]


def i2_capacity():
    return [('--variant stat', '--fold 2', f'--seed {s}', f'--mlp-first {w}')
            for w, s in itertools.product([256, 512], SEEDS)]


if __name__ == '__main__':
    print(f'writing to {OUT}' + (f'   (extra args: {EXTRA})' if EXTRA else ''))
    total = 0
    total += write('a2.tsv', a2())
    total += write('b_ablation.tsv', b_ablation())
    total += write('c1_threshold.tsv', c1_threshold())
    total += write('c4_operator.tsv', c4_operator())
    total += write('i2_capacity.tsv', i2_capacity())
    print(f'\n  {total} runs total')
    print('\n  each line is the argument list for one `python main_code/train.py` call;')
    print('  run the lines in order so any stopping point leaves complete comparisons.')
    print('\n  a2.tsv order: fold 2 first, three variants per (fold, seed) block,')
    print('  so every 3 completed runs is one full controlled comparison.')
