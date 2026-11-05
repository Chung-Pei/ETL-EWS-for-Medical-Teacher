# -*- coding: utf-8 -*-
"""R5-9 add-on: paired comparison of the selected policy per outer resample, refit scores vs frozen scores (same 200 draws),
and Wilson 95% intervals for the headline frequencies (binomial sampling error of B = 200)."""
import numpy as np
import pandas as pd

import r5_lib as L


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


rows = []
for defn in ('deploy', 'pooled'):
    r = pd.read_csv(L.RES / f'R5-9_per_resample_refit_{defn}.csv')
    f = pd.read_csv(L.RES / f'R5-9_per_resample_frozen_{defn}.csv')
    assert (r.b.values == f.b.values).all()
    pr = r.sel_week + ' + ' + r.sel_pred
    pf = f.sel_week + ' + ' + f.sel_pred
    ct = pd.crosstab(pf, pr)
    for a in ct.index:
        for b in ct.columns:
            rows.append({'definition': defn, 'table': 'paired_selected_policy (rows: frozen, cols: refit)', 'frozen_policy': a,
                         'refit_policy': b, 'count': int(ct.loc[a, b]), 'n': len(r)})
    n = len(r)
    for lab, mask_r, mask_f in (('W10 + M3', pr == 'W10 + M3', pf == 'W10 + M3'), ('W8 + M3', pr == 'W8 + M3', pf == 'W8 + M3')):
        for src, m in (('refit', mask_r), ('frozen', mask_f)):
            k = int(m.sum())
            lo, hi = wilson(k, n)
            rows.append({'definition': defn, 'table': 'headline_frequency_with_Wilson95', 'frozen_policy': src, 'refit_policy': lab,
                         'count': k, 'n': n, 'frequency': k / n, 'ci_lo': lo, 'ci_hi': hi})
    # resamples where the selected policy differs
    rows.append({'definition': defn, 'table': 'policy_differs_refit_vs_frozen', 'count': int((pr != pf).sum()), 'n': n,
                 'frequency': float((pr != pf).mean())})
    # blend W* week per resample, paired shift
    for p in L.PRED:
        a, b = f[f'wstar_{p}'], r[f'wstar_{p}']
        W = {'W6': 6, 'W8': 8, 'W10': 10, 'W12': 12, 'none': 99}
        later = int((b.map(W) > a.map(W)).sum())
        earlier = int((b.map(W) < a.map(W)).sum())
        rows.append({'definition': defn, 'table': 'W*_shift_refit_vs_frozen', 'frozen_policy': p, 'refit_policy': 'later|earlier',
                     'count': later, 'n': n, 'frequency': earlier})
T = pd.DataFrame(rows)
T.to_csv(L.RES / 'R5-9_paired_policy.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 200)
print(T.to_string())
