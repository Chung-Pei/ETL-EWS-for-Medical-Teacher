# -*- coding: utf-8 -*-
"""Reproduction gate for R5-5 / R5-9: rebuild the frozen LOSO scores (arm P, feature set C) from the packaged step9 inputs with
the pack's own pipeline code, XGBoost 3.4.1 (CPU), five seeds, and compare with the keyed frozen scores."""
import sys
import time

import numpy as np
import pandas as pd

import r5_lib as L
import r5_pipeline as P

weeks = [int(w) for w in sys.argv[1:]] or [8]
EL, EE = P.import_pipeline()
import xgboost
print('xgboost', xgboost.__version__)
root = P.merge_step9()
etl = EL.Etl(str(P.ETL_DIR))
EL.init_bas(etl)
base, scores = L.load_base()
y = base.y.to_numpy()
rows = []
t0 = time.time()
for W in weeks:
    data = EE.load_all(etl, str(root), [W])
    P.apply_semester_outcome(data, base)
    df = EE.canon(EE.add_m3(EE.run_folds(etl, data[W], EE.fold_defs('P'), etl.FS_C, False, {'m1': 1, 'm2': 1})))
    assert list(zip(df.semester, df.anon_id)) == list(zip(base.semester, base.anon_id))
    fro = scores[f'W{W}']
    for col, nm in (('m1', "M1' BAS"), ('m2', "M2' XGBoost"), ('m3', "M3' blend")):
        a_new, a_fro = L.auc_fail(y, df[col]), L.auc_fail(y, fro[col])
        r_new, r_fro = L.pearson(df[col], base.final), L.pearson(fro[col], base.final)
        rows.append({'week': f'W{W}', 'score': nm, 'AUC_rebuilt': a_new, 'AUC_frozen': a_fro, 'AUC_diff': a_new - a_fro,
                     'r_rebuilt': r_new, 'r_frozen': r_fro, 'r_diff': r_new - r_fro,
                     'corr_score_rebuilt_vs_frozen': float(np.corrcoef(df[col], fro[col])[0, 1]),
                     'max_abs_diff_score': float(np.abs(df[col].to_numpy() - fro[col].to_numpy()).max())})
    print(f'W{W} done {time.time() - t0:.0f}s', flush=True)
T = pd.DataFrame(rows)
T['within_0.003_AUC'] = T.AUC_diff.abs() <= 0.003
pd.set_option('display.width', 250)
print(T.to_string())
T.to_csv(L.RES / 'R5-5_9_reproduction_gate.csv', index=False, encoding='utf-8-sig')
