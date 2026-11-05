# -*- coding: utf-8 -*-
"""R5-3 -- bootstrap selection frequencies under the DEPLOYMENT definition (frozen out-of-fold scores).

Outer: 2,000 student-cluster resamples (standing draws, seed 20261031).  Inner: 1,000 draws from each outer resample
re-estimate the loss intervals; the Appendix B rule (within-semester metrics) is re-applied in every outer resample.
Candidate weeks W6, W8, W10, W12 (W12 = reference; it always qualifies with loss 0, so W14/W16 can never be selected)."""
import sys
import time

import numpy as np
import pandas as pd

import r5_lib as L

B_OUT = int(sys.argv[1]) if len(sys.argv) > 1 else L.BOOT_B
D_IN = 1000
base, scores = L.load_base()
N = len(base)
y = base.y.to_numpy().astype(int)
f = base.final.to_numpy(float)
stu = base.stu.to_numpy()
n_stu = int(stu.max()) + 1
counts = L.make_counts(n_stu, B_OUT, L.SEED)
names, S = L.build_candidates(base, scores, 'deploy')
groups = L.sem_groups_of(base.sem_i)
col = {n: i for i, n in enumerate(names)}

wstar_log, pol_log, equiv_flag = [], [], 0
t0 = time.time()
for b in range(B_OUT):
    Wb = counts[b][stu][None, :]                                  # outer resample as record weights (1 x N)
    a_pt, _ = L.metrics_draws(Wb, S, y, f, groups, 'deploy')
    pt = {n: float(a_pt[0, i]) for i, n in enumerate(names)}
    rng = np.random.default_rng(np.random.SeedSequence([L.SEED, b]))
    Wi = L.inner_weights(counts[b], stu, D_IN, rng)               # 1,000 inner draws from the outer resample
    a_d, r_d = L.metrics_draws(Wi, S, y, f, groups, 'deploy')
    sel = L.select_rule(a_d, r_d, names, pt)
    wstar_log.append({p: sel['wstar'][p] or 'none' for p in L.PRED})
    pol_log.append((sel['sel_week'] or 'none', sel['sel_pred'] or 'none'))
    equiv_flag += int(any(c['entirely_within_band'] for c in sel['compare']))
    if (b + 1) % 200 == 0:
        el = time.time() - t0
        print(f'outer {b + 1}/{B_OUT}  {el:.0f}s elapsed  ({el / (b + 1):.3f}s per outer)', flush=True)

rows = []
W_ORDER = ['W6', 'W8', 'W10', 'W12', 'none']
for p in L.PRED:
    vc = pd.Series([w[p] for w in wstar_log]).value_counts()
    for W in W_ORDER:
        rows.append({'table': 'selected_week_per_predictor', 'predictor': {'M1': "M1' (BAS)", 'M2': "M2' (XGBoost)",
                     'M3': "M3' (blend)"}[p], 'week': W, 'policy': '', 'count': int(vc.get(W, 0)),
                     'frequency': float(vc.get(W, 0)) / B_OUT})
pol = pd.Series([f'{w}|{p}' for w, p in pol_log]).value_counts()
for k, v in pol.items():
    w, p = k.split('|')
    rows.append({'table': 'selected_policy (week x predictor)', 'predictor': p, 'week': w, 'policy': f'{w} + {p}',
                 'count': int(v), 'frequency': v / B_OUT})
n_w10m3 = int(sum(1 for w, p in pol_log if w == 'W10' and p == 'M3'))
rows.append({'table': 'headline', 'predictor': 'M3', 'week': 'W10', 'policy': 'Week 10 + blend selected', 'count': n_w10m3,
             'frequency': n_w10m3 / B_OUT})
rows.append({'table': 'headline', 'predictor': '', 'week': '', 'policy': 'resamples in which a third-rule comparison fell '
             'entirely inside the +-0.02 band (equivalence fallback used)', 'count': equiv_flag, 'frequency': equiv_flag / B_OUT})
T = pd.DataFrame(rows)
T.to_csv(L.RES / 'R5-3.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 200)
print(T.to_string())
print(f'total {time.time() - t0:.0f}s; B_outer={B_OUT}, D_inner={D_IN}, seed={L.SEED}')
