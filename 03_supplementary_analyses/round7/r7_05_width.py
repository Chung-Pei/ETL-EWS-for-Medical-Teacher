# -*- coding: utf-8 -*-
"""R7-5: sampling distribution of the Tier A AUC-loss interval width at n = 745 (Era E1 record subsamples)."""
import r7_common as C
import r7_tiera as TA
import r7_rows as RW
import json
import time
import numpy as np
import pandas as pd

L = C.L
t0 = time.time()
base, scores = C.load_base()
y = base.y.to_numpy()
f = base.final.to_numpy(float)
N = len(base)
E1 = base.semester.isin(['1111', '1112', '1121', '1122', '1131']).to_numpy()
E2 = base.semester.isin(['1132', '1141']).to_numpy()
rows = []

# ---------------- step 0: full intervals (Tier A machine) ----------------------------------------------------------------
Wm = TA.full_weights(base)
S0 = {}
for nm, m in (('pooled', np.ones(N, bool)), ('E1', E1), ('E2', E2)):
    res = TA.select_on_subset(base, scores, m, Wm)
    S0[nm] = TA.blend_w8_loss(res)
    S0[nm]['n'] = int(m.sum())
    S0[nm]['n_fail'] = int(y[m].sum())
    S0[nm]['width'] = S0[nm]['auc_loss_hi'] - S0[nm]['auc_loss_lo']
W_E2, W_P = S0['E2']['width'], S0['pooled']['width']
print('Step 0 (blend W8 AUC loss = W12 - W8; percentile interval, B=2000, seed 20261006)')
for nm in ('pooled', 'E1', 'E2'):
    s = S0[nm]
    print(f"  {nm:6s} n={s['n']} fail={s['n_fail']}  point {s['auc_loss_pt']:+.4f}  [{s['auc_loss_lo']:+.4f}, {s['auc_loss_hi']:+.4f}]  width {s['width']:.4f}")
print(f'  ratio W_E2 / W_pooled = {W_E2 / W_P:.3f}  (published 1.87);  sqrt(2652/745) = {np.sqrt(2652 / 745):.3f} (published 1.89)')
for nm in ('pooled', 'E1', 'E2'):
    s = S0[nm]
    RW.add(rows, 'R7-5 step0', f'Tier A machine, {nm}', 'blend W8 AUC loss (W12 - W8), full interval', 'W8', s['n'], s['auc_loss_pt'], s['auc_loss_lo'], s['auc_loss_hi'],
           f"n_fail {s['n_fail']}; width {s['width']:.5f}; B=2000 seed 20261006")
RW.add(rows, 'R7-5 step0', 'Tier A machine', 'width W_E2 (numerator of 1.87)', 'W8', 745, W_E2, note='upper limit minus lower limit, Era E2')
RW.add(rows, 'R7-5 step0', 'Tier A machine', 'width pooled (denominator of 1.87)', 'W8', 2652, W_P, note='upper limit minus lower limit, pooled')
RW.add(rows, 'R7-5 step0', 'Tier A machine', 'ratio W_E2 / W_pooled', 'W8', estimate=W_E2 / W_P, note='published 1.87')
RW.add(rows, 'R7-5 step0', 'Tier A machine', 'sqrt(N_pooled / N_E2) = sqrt(2652/745)', 'W8', estimate=float(np.sqrt(2652 / 745)), note='published 1.89')

# ---------------- step 1: 200 subsamples of n = 745 from E1 ------------------------------------------------------------------
ix_e1 = np.flatnonzero(E1)
S = np.column_stack([scores['W8'].m3.to_numpy(float), scores['W12'].m3.to_numpy(float)])   # frozen pooled-percentile blend


def one_run(j, draw_seed_offset):
    rng = np.random.default_rng(C.SEED_TIERA + draw_seed_offset + j)
    sub = np.sort(rng.choice(ix_e1, size=745, replace=False))
    anon = base.anon_id.to_numpy()[sub]
    Wb = C.counts_sorted_students(anon, 1000, C.SEED_TIERA + j)               # (1000 x 745)
    a_d, _ = L.auc_weighted(Wb, S[sub], y[sub], [np.arange(745)])
    a_d = a_d[:, 0, :]                                  # (B x 2): columns [W8, W12]
    la = a_d[:, 1] - a_d[:, 0]
    lo, hi = float(np.nanpercentile(la, 2.5)), float(np.nanpercentile(la, 97.5))
    a_p, _ = L.auc_weighted(np.ones((1, 745), np.float32), S[sub], y[sub], [np.arange(745)])
    return {'j': j, 'n_fail': int(y[sub].sum()), 'n_students': int(len(np.unique(anon))), 'pt_loss': float(a_p[0, 0, 1] - a_p[0, 0, 0]),
            'lo': lo, 'hi': hi, 'width': hi - lo}


out = {}
for tag, off in (('primary (subsample seed = 20261006 + 1,000,000 + j)', 1_000_000), ('check (subsample seed = 20261006 + 2,000,000 + j)', 2_000_000)):
    D = pd.DataFrame([one_run(j, off) for j in range(200)])
    out[tag] = D
    print(f'\n{tag}  [{time.time() - t0:.0f}s]')
    w = D.width.to_numpy()
    print(f"  width: median {np.median(w):.4f}  IQR [{np.percentile(w, 25):.4f}, {np.percentile(w, 75):.4f}]  min {w.min():.4f} max {w.max():.4f}")
    print(f"  W_E2 {W_E2:.4f} percentile in the distribution: {np.mean(w < W_E2) * 100:.1f}% below (mid-rank {(np.mean(w < W_E2) + 0.5 * np.mean(w == W_E2)) * 100:.1f}%)")
    print(f"  upper limit > 0.020: {np.mean(D.hi > 0.020) * 100:.1f}%   upper-limit median {D.hi.median():.4f} IQR [{D.hi.quantile(.25):.4f}, {D.hi.quantile(.75):.4f}]")
    print(f"  point loss median {D.pt_loss.median():+.4f}; subsample failing count median {D.n_fail.median():.0f} [{D.n_fail.min()}, {D.n_fail.max()}]")

P = out['primary (subsample seed = 20261006 + 1,000,000 + j)']
Cc = out['check (subsample seed = 20261006 + 2,000,000 + j)']
for tag, D, role in (('primary', P, 'primary'), ('check', Cc, 'check draw seeds')):
    w = D.width.to_numpy()
    pct = float(np.mean(w < W_E2) * 100)
    RW.add(rows, 'R7-5', f'E1 subsamples n=745 ({role})', 'interval width: median [IQR]', 'W8', 745, float(np.median(w)), float(np.percentile(w, 25)), float(np.percentile(w, 75)),
           'R=200; within-subsample student-cluster bootstrap B=1000, seed 20261006+j')
    RW.add(rows, 'R7-5', f'E1 subsamples n=745 ({role})', 'interval width: min / max', 'W8', 745, float(w.min()), float(w.min()), float(w.max()), 'estimate = min; ci_lo = min; ci_hi = max')
    RW.add(rows, 'R7-5', f'E1 subsamples n=745 ({role})', 'percentile of W_E2 in width distribution (% of subsamples narrower)', 'W8', 745, pct, note=f'W_E2 = {W_E2:.5f}')
    RW.add(rows, 'R7-5', f'E1 subsamples n=745 ({role})', 'upper limit: median [IQR]', 'W8', 745, float(D.hi.median()), float(D.hi.quantile(.25)), float(D.hi.quantile(.75)), '')
    RW.add(rows, 'R7-5', f'E1 subsamples n=745 ({role})', 'share of subsamples with upper limit > 0.020 (%)', 'W8', 745, float(np.mean(D.hi > 0.020) * 100), note=f'E2 observed upper limit {S0["E2"]["auc_loss_hi"]:.4f}')
    RW.add(rows, 'R7-5', f'E1 subsamples n=745 ({role})', 'ratio median width / W_pooled', 'W8', 745, float(np.median(w) / W_P), note='for comparison with 1.87 and 1.89')
RW.write('R7-5', rows)
P.assign(role='primary').to_csv(C.RES / 'R7-5_subsamples_primary.csv', index=False, encoding='utf-8-sig')
Cc.assign(role='check').to_csv(C.RES / 'R7-5_subsamples_check.csv', index=False, encoding='utf-8-sig')
json.dump(S0, open(C.RES / 'R7-5_step0.json', 'w'), indent=1)
print(f'\ntotal {time.time() - t0:.0f}s')
