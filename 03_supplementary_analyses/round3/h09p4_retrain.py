# -*- coding: utf-8 -*-
"""Retrain M1'/M2' (P arm, FS-C, LOSO) per week for FULL_RERUN (21 features) and NO4T (20 features, no is_4tech).
Usage:  python h09p4_retrain.py --W 8 [--W 10 ...] --out results      (one week per call keeps each call < 300 s)
"""
import argparse
import os
import platform
import sys
import tempfile
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import h09p4_lib as H  # noqa: E402

C = H.C
L = H.L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--W', type=int, action='append', required=True)
    ap.add_argument('--out', default='results')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import ews_experiment as E
    import scipy
    import sklearn
    import xgboost
    logp = os.path.join(a.out, 'run_retrain.log')
    with open(logp, 'a', encoding='utf-8') as lg:
        def log(*x):
            msg = ' '.join(str(v) for v in x)
            print(msg, flush=True)
            lg.write(msg + '\n')
            lg.flush()

        log('# H-09p4 retrain  clock=%s' % time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        log('python', platform.python_version(), 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__,
            'numpy', np.__version__, 'pandas', pd.__version__, 'scipy', scipy.__version__, 'seeds', L.SEEDS)
        for f in ['oof_scores_no_ids.csv', 'class_membership5.csv', 'd7_semester_scores.csv']:
            log('sha256', f, C.sha256(os.path.join(C.IN, f)))
        for f in ['H09p4_plan.md', 'h09p4_lib.py', 'h09p4_retrain.py']:
            log('sha256', f, C.sha256(os.path.join(HERE, f)))
        tmp = tempfile.mkdtemp()
        for W in a.W:
            t0 = time.time()
            etl, data, n = H.build_data(W, tmp)
            assert n == 2652
            fs_full = list(etl.FS_C)
            fs_no = [f for f in fs_full if f != 'is_4tech']
            log(f'W{W}: n={n} features full={len(fs_full)} no4t={len(fs_no)} (is_4tech removed: {set(fs_full) - set(fs_no)})')
            for cfg, feats in [('full', fs_full), ('no4t', fs_no)]:
                df = E.run_folds(etl, data[W], E.fold_defs('P'), feats, False, {'m1': 1, 'm2': 1})
                df = H.canonical(df)
                df.insert(0, 'W', f'W{W}')
                keep = df[['W', 'semester', 'final', 'y', 'm1', 'm2']]
                keep.to_csv(os.path.join(a.out, f'H09p4_oof_{cfg}_W{W}.csv'), index=False, encoding='utf-8-sig')
                log(f'W{W} {cfg}: rows={len(df)} fails={int(df.y.sum())} auc m1={L.auc_fail(df.y.to_numpy(int), df.m1.to_numpy(float)):.6f} '
                    f'm2={L.auc_fail(df.y.to_numpy(int), df.m2.to_numpy(float)):.6f}  ({time.time() - t0:.0f}s)')
    print('done')


if __name__ == '__main__':
    main()
