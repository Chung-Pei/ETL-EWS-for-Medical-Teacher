# -*- coding: utf-8 -*-
"""H-07 gate A1: reproduce the frozen Week-8 anchors with the shipped step9 inputs and ETL v7.15.34.
Plan: plan/H07_plan.md. Run from the return root: python -I H-07/h07_gate_a1.py"""
import hashlib
import json
import os
import platform
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
P2 = os.path.join(ROOT, 'materials', 'pack2')
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
sys.path.insert(0, os.path.join(P2, '04_pipeline_code'))
import ews_experiment as EX  # noqa: E402
import ews_lib as L  # noqa: E402
import h_lib as H  # noqa: E402
import linkage as K  # noqa: E402

OUT = os.path.join(ROOT, 'H-07')
CACHE = os.path.join(ROOT, 'work_cache', 'step9_merged')
OOF = os.path.join(ROOT, 'materials', 'pack1', '04_data', 'oof_scores_no_ids.csv')
ANCHORS = {"M2'_auc": 0.7092, "M3'_auc": 0.727}
TOL = 0.003


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main():
    log = open(os.path.join(OUT, 'run.log'), 'w', encoding='utf-8')
    import sklearn
    import xgboost
    print(f'python {platform.python_version()} numpy {np.__version__} pandas {pd.__version__} '
          f'xgboost {xgboost.__version__} sklearn {sklearn.__version__}', file=log)
    etl_dir = os.path.join(P2, '07_etl_v7_15_34')
    for p in (os.path.join(etl_dir, 'lms_etl', 'modules', '11_cross_analysis.py'), os.path.join(P2, '04_pipeline_code', 'ews_lib.py'),
              os.path.join(P2, '04_pipeline_code', 'ews_experiment.py'), OOF):
        print(f'sha256 {os.path.relpath(p, ROOT)} {sha(p)}', file=log)

    os.makedirs(os.path.join(CACHE, 'out_W8'), exist_ok=True)
    merged = K.load_step9(P2, 8)
    json.dump(merged, open(os.path.join(CACHE, 'out_W8', 'step9_inputs.json'), 'w', encoding='utf-8'), ensure_ascii=False)

    etl = L.Etl(etl_dir)
    L.init_bas(etl)
    data = EX.load_all(etl, CACHE, [8])
    counts = {s: len(v) for s, v in data[8].items()}
    n = sum(counts.values())
    oof = H.load_oof(OOF)['W8']
    oofid = K.load_oof_with_ids(OOF, P2)['W8']
    sem_final = {(str(sm), aid): float(f) for sm, aid, f in zip(oofid['semester'], oofid['anon_id'], oofid['final'])}

    def run_gate(dat):
        fr = EX.run_folds(etl, dat, EX.fold_defs('P'), etl.FS_C, False, {'m1': 1, 'm2': 1})
        df = EX.canon(EX.add_m3(fr))
        y = df['y'].to_numpy(int)
        auc = {"M1'": H.auc_fail(y, df['m1'].to_numpy(float)), "M2'": H.auc_fail(y, df['m2'].to_numpy(float)),
               "M3'": H.auc_fail(y, df['m3'].to_numpy(float))}
        cmp = {}
        for c in ('m1', 'm2', 'm3'):
            x, z = df[c].to_numpy(float), oof[c].to_numpy(float)
            cmp[c] = {'max_abs_diff': float(np.abs(x - z).max()), 'mean_abs_diff': float(np.abs(x - z).mean()),
                      'pearson': float(np.corrcoef(x, z)[0, 1]),
                      'spearman': float(pd.Series(x).corr(pd.Series(z), method='spearman'))}
        anchors = {}
        for k, v in (("M2'_auc", auc["M2'"]), ("M3'_auc", auc["M3'"])):
            anchors[k] = {'value': v, 'anchor': ANCHORS[k], 'tolerance': TOL, 'diff': v - ANCHORS[k],
                          'pass': bool(abs(v - ANCHORS[k]) <= TOL + 1e-12)}
        return df, auc, cmp, anchors

    df1, auc1, cmp1, anc1 = run_gate(data[8])
    assert (oof['semester'].to_numpy() == df1['semester'].to_numpy(int)).all()
    data_b = {W: {sm: [dict(r) for r in recs] for sm, recs in by.items()} for W, by in data.items()}
    for sm, recs in data_b[8].items():
        for r in recs:
            r['final'] = sem_final[(sm, r['anon_id'])]
            r['is_fail'] = int(r['final'] < 60)
    df2, auc2, cmp2, anc2 = run_gate(data_b[8])
    res = {'gate': 'A1 and A1b', 'n_records': n, 'semester_counts': {str(k): v for k, v in counts.items()},
           'A1_outcome_fail_count': int(df1['y'].sum()), 'A1b_outcome_fail_count': int(df2['y'].sum()),
           'anchors': anc1, 'gate_A1_pass': bool(all(a['pass'] for a in anc1.values())),
           'rerun_auc': auc1, 'score_comparison_with_frozen_oof': cmp1,
           'anchors_A1b': anc2, 'gate_A1b_pass': bool(all(a['pass'] for a in anc2.values())),
           'rerun_auc_A1b': auc2, 'score_comparison_with_frozen_oof_A1b': cmp2,
           'frozen_oof_auc': {"M1'": H.auc_fail(oof['y'].to_numpy(int), oof['m1'].to_numpy(float)),
                              "M2'": H.auc_fail(oof['y'].to_numpy(int), oof['m2'].to_numpy(float)),
                              "M3'": H.auc_fail(oof['y'].to_numpy(int), oof['m3'].to_numpy(float))},
           'versions': {'xgboost': xgboost.__version__, 'sklearn': sklearn.__version__, 'etl': 'v7.15.34 (version warning, README_pack2 section 4)'}}
    json.dump(res, open(os.path.join(OUT, 'H07_gate_A1.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(json.dumps({k: res[k] for k in ('n_records', 'anchors', 'gate_A1_pass', 'rerun_auc', 'anchors_A1b', 'gate_A1b_pass', 'rerun_auc_A1b', 'frozen_oof_auc')}, indent=1), file=log)
    log.close()
    print(json.dumps({'A1_pass': res['gate_A1_pass'], 'A1b_pass': res['gate_A1b_pass'], 'A1b_anchors': anc2, 'A1b_auc': auc2,
                      'frozen_auc': res['frozen_oof_auc'], 'A1b_cmp': cmp2}, indent=1))


if __name__ == '__main__':
    main()
