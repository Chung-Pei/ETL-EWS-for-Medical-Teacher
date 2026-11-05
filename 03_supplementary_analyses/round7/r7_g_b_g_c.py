# -*- coding: utf-8 -*-
"""Gate R7-G items G-b (Tier B E2 two folds through the R7-1 retraining pipeline) and G-c (BAS / blend rebuild anchors)."""
import r7_retrain as R
import r7_common as C
import json
import time
import numpy as np
import pandas as pd

L = C.L
t0 = time.time()
EL, EE, etl, base, scores, data = R.setup()
print(f'setup done {time.time() - t0:.0f}s')
y_all = base.y.to_numpy()

# ---------------------------------------------------------------- G-c: pooled LOSO rebuild at W8 ------------------------------------
df = EE.canon(EE.add_m3(EE.run_folds(etl, data[8], EE.fold_defs('P'), etl.FS_C, False, {'m1': 1, 'm2': 1})))
assert list(zip(df.semester, df.anon_id)) == list(zip(base.semester, base.anon_id))
fro = scores['W8']
rows = []
for col, nm in (('m1', 'BAS'), ('m2', 'XGBoost'), ('m3', 'blend')):
    a_new, a_fro = L.auc_fail(y_all, df[col]), L.auc_fail(y_all, fro[col])
    rows.append({'score': nm, 'AUC_rebuilt': a_new, 'AUC_frozen': a_fro, 'AUC_diff': a_new - a_fro,
                 'max_abs_diff_score': float(np.abs(df[col].to_numpy() - fro[col].to_numpy()).max())})
G_c = pd.DataFrame(rows)
print('\nG-c  (pooled LOSO, W8, rebuilt vs frozen)')
print(G_c.round(5).to_string(index=False))
G_c.to_csv(C.RES / 'gate_G_c.csv', index=False, encoding='utf-8-sig')
print(f'G-c done {time.time() - t0:.0f}s')

# ---------------------------------------------------------------- G-b: Tier B E2 two folds ----------------------------------------------
frames = {}
removed = {}
for W in C.W_ALL:
    parts = []
    for tr, te in (('1141', '1132'), ('1132', '1141')):
        d, rm = R.fit_pair(EE, etl, data[W], [tr], te)
        parts.append(d)
        removed[f'W{W}|train {tr}->test {te}'] = rm
    e2 = EE.canon(EE.add_m3(pd.concat(parts, ignore_index=True)))     # era-level blend: pooled percentile over both folds
    frames[W] = e2
    print(f'  E2 folds W{W} done {time.time() - t0:.0f}s', flush=True)
ref = frames[12]
assert all((frames[W].anon_id.to_numpy() == ref.anon_id.to_numpy()).all() for W in C.W_ALL)
rows = []
for col, nm in (('m1', "M1' BAS"), ('m2', "M2' XGBoost"), ('m3', "M3' blend")):
    for W in C.W_ALL:
        pm = EL.point_metrics(frames[W], col)
        rows.append({'model': nm, 'W': f'W{W}', 'n': pm['n'], 'n_fail': pm['n_fail'], 'auc': pm['auc'], 'r': pm['r']})
T = pd.DataFrame(rows)
piv = T.pivot(index='model', columns='W', values=['auc', 'r'])
print('\nG-b  Tier B E2 two folds (train 1141->test 1132; train 1132->test 1141), era-merged')
print(T.round(4).to_string(index=False))
for nm in ("M1' BAS", "M2' XGBoost", "M3' blend"):
    g = T[T.model == nm].set_index('W')
    print(f"  {nm}: W8 point loss AUC {g.loc['W12','auc'] - g.loc['W8','auc']:+.4f}   r {g.loc['W12','r'] - g.loc['W8','r']:+.4f}")
T.to_csv(C.RES / 'gate_G_b_E2_points.csv', index=False, encoding='utf-8-sig')
json.dump({'removed_training_records': removed}, open(C.RES / 'gate_G_b_removed.json', 'w'), indent=1)
print('removed training records (retaker rule):', removed)
print(f'total {time.time() - t0:.0f}s')
