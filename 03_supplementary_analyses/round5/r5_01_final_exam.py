# -*- coding: utf-8 -*-
"""R5-1 -- final-examination outcome check (frozen scores; no retraining)."""
import numpy as np
import pandas as pd

import r5_lib as L

base, scores = L.load_base()
N = len(base)
y = (base.final_score < 60).astype(int).to_numpy()          # outcome: final examination < 60 (d7)
sem = base.semester.to_numpy()
groups = L.sem_groups_of(base.sem_i)
Wm = L.make_weights(base.stu.to_numpy())
ones = np.ones((1, N), np.float32)

rows = []


def add(pred, week, metric, est, lo=np.nan, hi=np.nan, note=''):
    rows.append({'predictor': pred, 'week': week, 'metric': metric, 'estimate': est, 'ci_lo': lo, 'ci_hi': hi,
                 'note': note})


add('outcome', 'all', 'n_records', N)
add('outcome', 'all', 'n_fail_final_exam_lt60', int(y.sum()))
add('outcome', 'all', 'fail_rate_final_exam_lt60', float(y.mean()))

# --- predictor columns -------------------------------------------------------------------------------
cols, names = [], []
mid = base.midterm_score.to_numpy(float)
mid_pct_w = np.empty(N)
for s in np.unique(sem):
    ix = np.flatnonzero(sem == s)
    mid_pct_w[ix] = L.pct_rank(mid[ix])
for W in ('W8', 'W10'):
    m3 = scores[W].m3.to_numpy(float)
    m3_pct_w = np.empty(N)
    for s in np.unique(sem):
        ix = np.flatnonzero(sem == s)
        m3_pct_w[ix] = L.pct_rank(m3[ix])
    cols += [m3, (m3_pct_w + mid_pct_w) / 2]
    names += [('M3p', W), ('Combo', W)]
cols.append(mid)
names.append(('Midterm', 'both'))
S = np.column_stack(cols)
C = S.shape[1]

# --- point estimates ------------------------------------------------------------------------------------
frame = pd.DataFrame({'semester': sem, 'y': y})
listed = np.zeros((N, C), bool)
rec_pt = np.zeros(C)
for c in range(C):
    frame['s'] = S[:, c]
    rec_pt[c], _, listed[:, c] = L.recall_at(frame, 's')
auc_pt = np.array([L.auc_fail(y, S[:, c]) for c in range(C)])

# --- bootstrap draws ----------------------------------------------------------------------------------------
a_pool, _ = L.auc_weighted(Wm, S, y, [np.arange(N)])
a_pool = a_pool[:, 0, :]
a_sem, n_sem = L.auc_weighted(Wm, S, y, groups)
ones_pt, n_pt = L.auc_weighted(ones, S, y, groups)
agg_dr, _ = L.aggregate_deploy(a_sem, np.zeros_like(a_sem) + 0.1, n_sem)       # r placeholder unused
agg_pt, _ = L.aggregate_deploy(ones_pt, np.zeros_like(ones_pt) + 0.1, n_pt)
fail_w = Wm @ y.astype(np.float32)
rec_dr = (Wm @ (listed & (y[:, None] == 1)).astype(np.float32)) / fail_w[:, None]

for c, (p, W) in enumerate(names):
    lo, hi = L.ci(a_pool[:, c])
    add(p, W, 'AUC_pooled', auc_pt[c], lo, hi, 'pooled construction; failure = final exam < 60')
    lo, hi = L.ci(rec_dr[:, c])
    add(p, W, 'Recall@20%', rec_pt[c], lo, hi, 'lists within semester (ceil(0.2n), tie-break seed 7); lists fixed, '
        'students resampled')
    lo, hi = L.ci(agg_dr[:, c])
    add(p, W, 'AUC_within_semester_nweighted', float(agg_pt[0, c]), lo, hi, 'n-weighted mean of within-semester AUCs')
    for gi, s in enumerate(L.D7):
        lo, hi = L.ci(a_sem[:, gi, c])
        add(p, W, f'AUC_within_semester_{s}', float(ones_pt[0, gi, c]), lo, hi,
            f'n={int(n_pt[0, gi])}, fails={int(y[groups[gi]].sum())}')

# --- paired differences (same resamples) ---------------------------------------------------------------------
idx = {n: i for i, n in enumerate(names)}
mid_i = idx[('Midterm', 'both')]
for W in ('W8', 'W10'):
    for a, b in [('M3p', 'Midterm'), ('Combo', 'Midterm'), ('Combo', 'M3p')]:
        ia = idx[(a, W)]
        ib = mid_i if b == 'Midterm' else idx[(b, W)]
        d = a_pool[:, ia] - a_pool[:, ib]
        lo, hi = L.ci(d)
        add(f'{a} minus {b}', W, 'AUC_pooled_difference', auc_pt[ia] - auc_pt[ib], lo, hi, 'paired bootstrap')
        d = rec_dr[:, ia] - rec_dr[:, ib]
        lo, hi = L.ci(d)
        add(f'{a} minus {b}', W, 'Recall@20%_difference', rec_pt[ia] - rec_pt[ib], lo, hi, 'paired bootstrap')

out = pd.DataFrame(rows)
out.to_csv(L.RES / 'R5-1.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 200, 'display.max_colwidth', 70)
print(out[~out.metric.str.startswith('AUC_within_semester_1')].to_string())
