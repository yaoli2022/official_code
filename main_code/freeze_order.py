import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

BASE = os.environ.get('GA_BASE', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE, 'data/all-organs4/all_organs')
CPG_LIST = os.path.join(BASE, 'data/multi_platform_cpgs.pkl')
REF_PRED = os.path.join(BASE, 'data/reference_test_predictions.csv')
OUT = os.path.join(BASE, 'cache', 'file_order.json')
META = ['age', 'gender', 'dataset', 'tissue_type']
MAX_SOLUTIONS = 10


def load_test_ages():
    cpg = set(np.array(pd.read_pickle(CPG_LIST)).tolist())
    ages = {}
    for fn in sorted(os.listdir(DATA_DIR)):
        if not fn.endswith('.pkl'):
            continue
        d = pd.read_pickle(os.path.join(DATA_DIR, fn))
        a = d[[c for c in d.columns if c in cpg] + META]
        a = a[a['tissue_type'].str.lower().str.contains('blood')].dropna()
        if len(a) == 0:
            continue
        _, te = train_test_split(a[META], test_size=0.2, random_state=42)
        ages[fn] = te.age.values.astype(float)
    return ages


def solve(ages, ref):
    names = sorted(ages)
    n = len(ref)
    out = []

    def walk(pos, used, acc):
        if len(out) >= MAX_SOLUTIONS:
            return
        if pos == n:
            out.append(list(acc))
            return
        for fn in names:
            if fn in used:
                continue
            a = ages[fn]
            if pos + len(a) <= n and np.allclose(a, ref[pos:pos + len(a)]):
                acc.append(fn)
                walk(pos + len(a), used | {fn}, acc)
                acc.pop()

    walk(0, frozenset(), [])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--show', action='store_true', help='print without writing')
    args = ap.parse_args()

    print(f'BASE = {BASE}')
    ref = pd.read_csv(REF_PRED).true_age.values.astype(float)
    ages = load_test_ages()
    print(f'  {len(ages)} blood datasets, {sum(len(v) for v in ages.values())} test samples')

    sols = solve(ages, ref)
    if not sols:
        raise SystemExit('FATAL: no dataset order reproduces the reference test predictions.\n'
                         '  The data under data/all-organs4/all_organs does not match the\n'
                         '  files the reference predictions were produced from.')
    print(f'  {len(sols)} order(s) consistent with {os.path.basename(REF_PRED)}')

    order = sols[0]
    if len(sols) > 1:
        diff = [i for i, (a, b) in enumerate(zip(sols[0], sols[1])) if a != b]
        print(f'  ambiguous at position(s) {diff}: '
              f'{[sols[0][i] for i in diff]} vs {[sols[1][i] for i in diff]}')
        print('  (adjacent all-neonate blocks with identical ages; the test set is')
        print('   unaffected, only fold composition. Pinning the first order.)')

    rec = dict(order=order, n_datasets=len(order), n_test=len(ref),
               n_solutions=len(sols), source=os.path.basename(REF_PRED))
    if args.show:
        print('\n' + json.dumps(rec, indent=1))
        return

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, OUT)
    print(f'\n  -> {OUT}')
    print(f'  first 3: {order[:3]}')


if __name__ == '__main__':
    main()
