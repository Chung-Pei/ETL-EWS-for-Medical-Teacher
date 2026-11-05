# -*- coding: utf-8 -*-
"""R5-6 (optional per-semester AUC table; included, costless) and R5-7 (retake-cohort exclusion sensitivity).
Frozen scores; training folds unchanged -- evaluation-subset sensitivity, not retraining."""
import math

import numpy as np
import pandas as pd

import r5_lib as L

base, scores = L.load_base()
N = len(base)
y = base.y.to_numpy().astype(int)
f = base.final.to_numpy(float)
sem = base.semester.to_numpy()
m3 = scores['W8'].m3.to_numpy(float)
Wm = L.make_weights(base.stu.to_numpy())

# ---------------- R5-6 ----------------------------------------------------------------------------------
groups = L.sem_groups_of(base.sem_i)
a_d, n_d = L.auc_weighted(Wm, m3.reshape(-1, 1), y, groups)
a_p, n_p = L.auc_weighted(np.ones((1, N), np.float32), m3.reshape(-1, 1), y, groups)
rows = []
for gi, s in enumerate(L.D7):
    lo, hi = L.ci(a_d[:, gi, 0])
    rows.append({'semester': s, 'n': int(n_p[0, gi]), 'n_fail': int(y[groups[gi]].sum()),
                 'auc_m3_W8_pooled_construction': float(a_p[0, gi, 0]), 'ci_lo': lo, 'ci_hi': hi,
                 'note': 'frozen pooled-percentile blend m3 (arm P, feature set C) evaluated within the semester; '
                         'student-cluster bootstrap B=2000 seed 20261031'})
R6 = pd.DataFrame(rows)
R6.to_csv(L.RES / 'R5-6.csv', index=False, encoding='utf-8-sig')
print(R6.drop(columns='note').to_string())

# ---------------- R5-7 ----------------------------------------------------------------------------------
b1 = L.b1_scores(base)                       # failure-oriented, frozen training folds
b1_pass = -b1


def headline(b, scr_m3, b1_pass_s, y_s, f_s, Wm_s, tag):
    S = np.column_stack([scr_m3, b1_pass_s])
    allix = [np.arange(len(y_s))]
    a, _ = L.auc_weighted(Wm_s, S, y_s, allix)
    r, _ = L.corr_weighted(Wm_s, S[:, :1], f_s, allix)
    a_pt = [L.auc_fail(y_s, S[:, 0]), L.auc_fail(y_s, S[:, 1])]
    r_pt = L.pearson(S[:, 0], f_s)
    fr = pd.DataFrame({'semester': b.semester.to_numpy(), 'y': y_s, 's': S[:, 0]})
    rec_pt, _, lst = L.recall_at(fr, 's')
    fail_w = Wm_s @ y_s.astype(np.float32)
    rec_d = (Wm_s @ (lst & (y_s == 1)).astype(np.float32)) / fail_w
    out = {
        'AUC blend W8 (pooled)': (a_pt[0], a[:, 0, 0]),
        'r(blend, semester score) W8': (r_pt, r[:, 0, 0]),
        'Recall@20% W8 (list within semester)': (rec_pt, rec_d),
        'B1 AUC (rebuilt, Appendix A)': (a_pt[1], a[:, 0, 1]),
        'Increment over B1 (blend AUC - B1 AUC)': (a_pt[0] - a_pt[1], a[:, 0, 0] - a[:, 0, 1]),
    }
    n_stu = len(np.unique(b.stu))
    return out, n_stu


full, nstu_full = headline(base, m3, b1_pass, y, f, Wm, 'full')
keep = (base.retake == 0).to_numpy()
assert int((~keep).sum()) == 52 and set(base.semester[~keep]) == {'1141'}      # validation: 52, all 1141
bsub, ssub = L.subset_view(base, scores, keep)
ysub, fsub = y[keep], f[keep]
Wsub = L.make_weights(bsub.stu.to_numpy())
sub, nstu_sub = headline(bsub, m3[keep], b1_pass[keep], ysub, fsub, Wsub, 'sub')
rows = []
for q in full:
    rows.append({'quantity': q, 'full_sample_value': full[q][0], 'full_ci_lo': L.ci(full[q][1])[0],
                 'full_ci_hi': L.ci(full[q][1])[1], 'subset_value_excl_retake': sub[q][0],
                 'subset_ci_lo': L.ci(sub[q][1])[0], 'subset_ci_hi': L.ci(sub[q][1])[1],
                 'note': 'scores and training folds frozen (evaluation-subset sensitivity, not retraining)'})
rows.append({'quantity': 'n records / failing / students', 'full_sample_value': f'{N} / {int(y.sum())} / {nstu_full}',
             'subset_value_excl_retake': f'{keep.sum()} / {int(ysub.sum())} / {nstu_sub}',
             'note': 'excluded: 52 retake records, all Semester 1141 (class_membership.csv class_track=retake)'})
R7 = pd.DataFrame(rows)
R7.to_csv(L.RES / 'R5-7.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 220, 'display.max_colwidth', 60)
print(R7.drop(columns='note').to_string())
