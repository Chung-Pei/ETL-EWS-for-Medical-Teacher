# -*- coding: utf-8 -*-
"""Self-checks (evidence, not impression): vectorized weighted AUC / r versus explicit resampling for selected draws;
uniform-scale corners equal the identity; Tier A machine versus Supplementary Table 1 re-selected weeks."""
import r7_common as C
import r7_tiera as TA
import json
import numpy as np
import pandas as pd

L = C.L
base, scores = C.load_base()
y = base.y.to_numpy()
f = base.final.to_numpy(float)
N = len(base)
out = []


def rec(name, ok, detail):
    out.append({'test': name, 'result': 'PASS' if ok else 'FAIL', 'detail': detail})
    print(('PASS ' if ok else 'FAIL ') + name + ': ' + detail)


# (1) weighted AUC / r vs explicit resampling, standing stream (seed 20261031) and Tier A stream (seed 20261006)
for label, Wm in (('standing stream 20261031', L.make_weights(base.stu.to_numpy())), ('Tier A stream 20261006', C.counts_sorted_students(base.anon_id))):
    S = np.column_stack([scores['W8'].m3.to_numpy(float), scores['W12'].m3.to_numpy(float)])
    a_d, _ = L.auc_weighted(Wm, S, y, [np.arange(N)])
    r_d, _ = L.corr_weighted(Wm, S, f, [np.arange(N)])
    worst_a = worst_r = 0.0
    for b in (0, 777, 1999):
        idx = np.repeat(np.arange(N), Wm[b].astype(int))
        for c in (0, 1):
            worst_a = max(worst_a, abs(L.auc_fail(y[idx], S[idx, c]) - a_d[b, 0, c]))
            worst_r = max(worst_r, abs(L.pearson(S[idx, c], f[idx]) - r_d[b, 0, c]))
    rec(f'weighted AUC/r equal explicit resampling ({label}; draws 0, 777, 1999; W8, W12 blend)', worst_a < 1e-9 and worst_r < 1e-9, f'max |dAUC| {worst_a:.2e}, max |dr| {worst_r:.2e}')

# (2) R7-5 subsample machinery against explicit resampling (subsample j = 0 of the primary run)
rng = np.random.default_rng(C.SEED_TIERA + 1_000_000)
E1 = np.flatnonzero(base.semester.isin(['1111', '1112', '1121', '1122', '1131']).to_numpy())
sub = np.sort(rng.choice(E1, size=745, replace=False))
Wb = C.counts_sorted_students(base.anon_id.to_numpy()[sub], 1000, C.SEED_TIERA)
S = np.column_stack([scores['W8'].m3.to_numpy(float)[sub], scores['W12'].m3.to_numpy(float)[sub]])
a_d, _ = L.auc_weighted(Wb, S, y[sub], [np.arange(745)])
a_d = a_d[:, 0, :]
loss_vec = a_d[:, 1] - a_d[:, 0]
loss_exp = np.empty(1000)
for b in range(1000):
    idx = np.repeat(np.arange(745), Wb[b].astype(int))
    loss_exp[b] = L.auc_fail(y[sub][idx], S[idx, 1]) - L.auc_fail(y[sub][idx], S[idx, 0])
rec('R7-5 subsample j=0: vectorized loss draws equal explicit resampling (1000 draws)', float(np.abs(loss_vec - loss_exp).max()) < 1e-9,
    f'max abs diff {np.abs(loss_vec - loss_exp).max():.2e}; width {np.percentile(loss_vec, 97.5) - np.percentile(loss_vec, 2.5):.5f}')

# (3) uniform-scale corners = identity
D = pd.read_csv(C.RES / 'R7-2_variants.csv', encoding='utf-8-sig')
num = [c for c in D.columns if c.startswith(('pooled_', 'deploy_')) and D[c].dtype.kind == 'f']
i0 = D.iloc[0][num].astype(float).to_numpy()
for v in ('V13 corner (0.5,0.5,0.5)', 'V20 corner (1.5,1.5,1.5)'):
    d = float(np.max(np.abs(D[D.variant == v].iloc[0][num].astype(float).to_numpy() - i0)))
    rec(f'R7-2: {v} equals identity after normalization', d < 1e-12, f'max abs diff over {len(num)} numeric columns {d:.2e}')

# (4) Tier A machine vs Supplementary Table 1 (re-selected weeks)
pub = {'pooled': ('W10', 'W8', 'W8'), 'E1': ('W10', 'W10', 'W10'), 'E2': ('W10', 'W12', 'W10')}      # BAS, XGBoost, blend
Wm = TA.full_weights(base)
masks = {'pooled': np.ones(N, bool), 'E1': base.semester.isin(['1111', '1112', '1121', '1122', '1131']).to_numpy(), 'E2': base.semester.isin(['1132', '1141']).to_numpy()}
for k, m in masks.items():
    ws = TA.select_on_subset(base, scores, m, Wm)['sel']['wstar']
    got = (ws['M1'], ws['M2'], ws['M3'])
    rec(f'Tier A machine re-selected weeks equal Supplementary Table 1 ({k})', got == pub[k], f'BAS/XGBoost/blend {got} vs published {pub[k]}')
pd.DataFrame(out).to_csv(C.RES / 'R7_selfcheck.csv', index=False, encoding='utf-8-sig')
print('ALL PASS' if all(o['result'] == 'PASS' for o in out) else 'FAILURES PRESENT')
