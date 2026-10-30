# -*- coding: utf-8 -*-
"""Run one H-07 arm (rebuild -> inject -> A1b-flow pipeline [-> ablation]). Spec: H07_addendum3_arms.md.
Usage: python h07_arms_run.py --arm ORIG_W8 --out results"""
import argparse
import json
import os
import pickle
import sys
import tempfile
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import h07_arms_lib as A  # noqa: E402

R = A.R
C = A.C
L = A.L
GROUPS = {
    'reading': ['active_weeks', 'consistency_score', 'weekly_minutes_std', 'cram_pattern_score', 'late_night_ratio',
                'early_start_ratio', 'total_learning_minutes', 'reading_inflation_ratio', 'policy_gaming_flag'],
    'quiz': ['quz_first_attempt_accuracy', 'quz_final_accuracy', 'quz_score_delta', 'quz_total_attempts',
             'quz_pass_rate', 'quz_cramming_ratio'],
    'sequence': ['s_cluster_encoded', 'MM_ratio', 'MQ_ratio', 'QM_ratio', 'QQ_ratio'],
    'class_proxy': ['is_4tech'],
}
ABLATE_ARMS = ['ORIG_W8', 'STRICT', 'EXCL_W8']
ORIG_OF = {'EXCL_W6': 'ORIG_W6', 'EXCL_W7': 'ORIG_W7', 'EXCL_W8': 'ORIG_W8', 'STRICT': 'ORIG_W8'}


def prep(m):
    import ews_experiment as E
    tmp = tempfile.mkdtemp()
    d = os.path.join(tmp, 'out_W8')
    os.makedirs(d)
    json.dump(m, open(os.path.join(d, 'step9_inputs.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    etl = L.Etl(os.path.join(C.PACK, '02_pipeline', 'etl_v7_15_34'))
    L.init_bas(etl)
    data = E.load_all(etl, tmp, [8])
    fr = C.frozen_frame(8)
    fin = {(r.semester, r.anon_id): float(r.final) for r in fr.itertuples()}
    for s in C.D7:
        for r in data[8][s]:
            r['final'] = fin[(s, r['anon_id'])]
            r['is_fail'] = int(r['final'] < 60)
    return E, etl, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--out', default='results')
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'arms'), exist_ok=True)
    t0 = time.time()
    name = a.arm
    rebuilt = A.rebuild_arm(name)
    info = rebuilt[4]
    if name.startswith('ORIG'):
        pickle.dump(rebuilt, open(os.path.join(a.out, 'arms', f'_{name}.pkl'), 'wb'))
    elif not name.endswith('_R'):
        orig = pickle.load(open(os.path.join(a.out, 'arms', f'_{ORIG_OF[name]}.pkl'), 'rb'))
        rebuilt = A.fix_retake(rebuilt, orig)
    m = R.merged_frozen_step9(8)
    m2 = R.inject(m, *rebuilt[:4])
    E, etl, data = prep(m2)
    df = E.run_folds(etl, data[8], E.fold_defs('P'), etl.FS_C, False, {'m1': 1, 'm2': 1})
    df = df.sort_values(['semester', 'anon_id']).reset_index(drop=True)
    fr = C.frozen_frame(8)
    assert (df.semester.to_numpy() == fr.semester.to_numpy()).all() and (df.final.to_numpy() == fr.final.to_numpy()).all()
    df[['semester', 'final', 'y', 'm1', 'm2']].to_csv(os.path.join(a.out, 'arms', f'{name}_oof.csv'), index=False, encoding='utf-8-sig')
    y = df.y.to_numpy(int)
    msg = f'{name}: events kept quiz={info["n_quiz_events"]} reading={info["n_reading_events"]} auc M1={L.auc_fail(y, df.m1.to_numpy(float)):.6f} M2={L.auc_fail(y, df.m2.to_numpy(float)):.6f}'
    if name in ABLATE_ARMS:
        fs = list(etl.FS_C)
        assert sorted(sum(GROUPS.values(), [])) == sorted(fs), 'groups must partition FS-C'
        for g, feats in GROUPS.items():
            d2 = E.run_folds(etl, data[8], E.fold_defs('P'), feats, False, {'m2': 1})
            d2 = d2.sort_values(['semester', 'anon_id']).reset_index(drop=True)
            d2[['semester', 'final', 'y', 'm2']].to_csv(os.path.join(a.out, 'arms', f'{name}_abl_{g}.csv'), index=False, encoding='utf-8-sig')
            msg += f' | {g}={L.auc_fail(y, d2.m2.to_numpy(float)):.4f}'
    msg += f'  ({time.time() - t0:.0f}s)'
    print(msg, flush=True)
    with open(os.path.join(a.out, 'run_arms.log'), 'a', encoding='utf-8') as lg:
        lg.write(msg + '\n')


if __name__ == '__main__':
    main()
