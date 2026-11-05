# -*- coding: utf-8 -*-
"""R7-4: distribution of the pooled-definition W8 AUC-loss upper limit in the R5-9 refitting resamples (pure arithmetic)."""
import r7_common as C
import r7_rows as RW
import numpy as np
import pandas as pd

d = C.STAGING / '06_r5_9_per_resample'
F = {nm: pd.read_csv(d / f'R5-9_per_resample_{nm}.csv', encoding='utf-8-sig') for nm in ('refit_pooled', 'frozen_pooled', 'refit_deploy', 'frozen_deploy')}
assert all(len(v) == 200 for v in F.values()), {k: len(v) for k, v in F.items()}
assert (F['refit_pooled'].b.to_numpy() == F['frozen_pooled'].b.to_numpy()).all(), 'resample index differs between refit and frozen'
rows = []


def dist(v):
    v = np.asarray(v, float)
    return dict(med=float(np.median(v)), q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)),
                p90=float(np.percentile(v, 90)))


# gate G-d record
fr = F['refit_pooled'].groupby(['sel_week', 'sel_pred']).size() / 200 * 100
RW.add(rows, 'Gate G-d', 'refit_pooled', 'rows', n=200, estimate=200, note='expected 200')
RW.add(rows, 'Gate G-d', 'refit_pooled', 'selected policy Week 10 + blend (%)', 'W10', 200, float(fr[('W10', 'M3')]), note='published 46.0')
RW.add(rows, 'Gate G-d', 'refit_pooled', 'selected policy Week 8 + blend (%)', 'W8', 200, float(fr[('W8', 'M3')]), note='published 36.5')

NOTE = {'ucl95_auc_W8_M3': 'column named in the spec (95th percentile of the inner loss draws)',
        'ucl97_auc_W8_M3': '97.5th percentile of the inner loss draws = convention of the selection rule, Table 15 Panel B (median 0.029; 78.5% > 0.020 reproduced exactly by this column in the deployment files) and the published 0.014'}
for col in ('ucl95_auc_W8_M3', 'ucl97_auc_W8_M3'):
    for nm in ('refit_pooled', 'frozen_pooled', 'refit_deploy', 'frozen_deploy'):
        v = F[nm][col].to_numpy()
        s = dist(v)
        pr = 'spec' if col.startswith('ucl95') else 'like-for-like with published'
        RW.add(rows, 'R7-4', nm, f'{col}: median [IQR]', 'W8', 200, s['med'], s['q1'], s['q3'], f'{pr}; {NOTE[col]}')
        RW.add(rows, 'R7-4', nm, f'{col}: 90th percentile', 'W8', 200, s['p90'], note=pr)
        RW.add(rows, 'R7-4', nm, f'{col}: share > 0.020 (%)', 'W8', 200, float(np.mean(v > 0.020) * 100), note=pr)
for nm in ('refit_pooled', 'frozen_pooled', 'refit_deploy', 'frozen_deploy'):
    x = F[nm]
    lit = (x.auc_W8_M3 - x.auc_W12_M3).to_numpy()          # literal spec sign: W8 - W12 (negative = W8 lower)
    s = dist(lit)
    RW.add(rows, 'R7-4', nm, 'auc_W8_M3 - auc_W12_M3 (spec sign; negative = Week 8 lower): median [IQR]', 'W8', 200, s['med'], s['q1'], s['q3'],
           'per-resample point difference')
    loss = -lit                                              # published sign: W12 - W8 (positive = loss)
    s = dist(loss)
    RW.add(rows, 'R7-4', nm, 'AUC loss = auc_W12_M3 - auc_W8_M3 (published sign; positive = loss): median [IQR]', 'W8', 200, s['med'], s['q1'], s['q3'],
           'same numbers, sign reversed')
    RW.add(rows, 'R7-4', nm, 'point loss > 0.020 (share, %)', 'W8', 200, float(np.mean(loss > 0.020) * 100), note='point loss itself beyond the AUC bound')
RW.write('R7-4', rows)
for r in rows:
    print(f"{r['item']:9s} {r['basis']:14s} {r['comparison'][:92]:92s} {r['estimate']:9.4f} {r['ci_lo']:9.4f} {r['ci_hi']:9.4f}")
