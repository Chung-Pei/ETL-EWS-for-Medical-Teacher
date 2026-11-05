# -*- coding: utf-8 -*-
"""Independent numeric check of the probe results: a second implementation (scikit-learn weighted AUC, numpy correlation,
explicit loops over draws) recomputes the point estimates, the per-semester/pooled values and the first 300 bootstrap draws
of the pooled paired differences from the OOF file, and compares them with PW_probe_results.csv / PW_boot_draws.npz."""
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('PW_ROOT', HERE.parent))
os.environ.setdefault('R5_RES', str(ROOT / 'results' / '_r5lib_scratch'))
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import r5_lib as L          # used ONLY for the standing draws (make_counts) and the base table, never for a metric

RES = ROOT / 'results'
EVAL = ['1121', '1122', '1131']
o = pd.read_csv(RES / 'PW_W9_oof_scores.csv', dtype={'anon_id': str, 'semester': str}, float_precision='round_trip')
R = pd.read_csv(RES / 'PW_probe_results.csv')
base, _ = L.load_base()
assert list(zip(o.semester, o.anon_id)) == list(zip(base.semester, base.anon_id))

# independent deployment blend: within-semester average rank / n of m1 and m2, averaged
def blend(m1, m2):
    return (rankdata(m1) / len(m1) + rankdata(m2) / len(m2)) / 2
S = {}
for W, c1, c2 in ((9, 'm1_BAS_W9', 'm2_XGBoost_W9'), (10, 'm1_BAS_W10_samebuild', 'm2_XGBoost_W10_samebuild')):
    s = np.empty(len(o))
    for sm in o.semester.unique():
        ix = np.flatnonzero(o.semester.to_numpy() == sm)
        s[ix] = blend(o[c1].to_numpy()[ix], o[c2].to_numpy()[ix])
    S[W] = s
assert np.allclose(S[9], o.score_W9_deployment_blend_pass_oriented) and np.allclose(S[10], o.score_W10_deployment_blend_pass_oriented_samebuild)

y = o.outcome_fail_semester_score_lt60.to_numpy()
f = o.semester_score.to_numpy(float)
sem = o.semester.to_numpy()
stu = base.stu.to_numpy()
idx = {s: np.flatnonzero(sem == s) for s in EVAL}


def wauc(s, yy, w):
    return roc_auc_score(yy, -s, sample_weight=w)          # failure is the positive class; lower pass-score = higher risk


def wcorr(s, ff, w):
    m1, m2 = np.average(s, weights=w), np.average(ff, weights=w)
    c = np.average((s - m1) * (ff - m2), weights=w)
    return c / math.sqrt(np.average((s - m1) ** 2, weights=w) * np.average((ff - m2) ** 2, weights=w))


def pooled(W, counts_row):
    w_all = counts_row[stu]
    A, Rr, N = [], [], []
    for sm in EVAL:
        ix = idx[sm]
        A.append(wauc(S[W][ix], y[ix], w_all[ix]))
        Rr.append(wcorr(S[W][ix], f[ix], w_all[ix]))
        N.append(w_all[ix].sum())
    N = np.array(N)
    auc = float(np.sum(np.array(A) * N) / N.sum())
    z = np.arctanh(np.array(Rr))
    r = float(np.tanh(np.sum(z * (N - 3)) / np.sum(N - 3)))
    return auc, r, A, Rr


ones = np.ones(stu.max() + 1)
chk = []
pt = {W: pooled(W, ones) for W in (9, 10)}
pool = R[R.row.str.startswith('pooled')].iloc[0]
chk.append(('pooled AUC W9 point', pt[9][0], pool.AUC_W9))
chk.append(('pooled AUC W10 point', pt[10][0], pool.AUC_W10))
chk.append(('pooled r W9 point', pt[9][1], pool.r_W9))
chk.append(('pooled r W10 point', pt[10][1], pool.r_W10))
chk.append(('pooled dAUC point', pt[10][0] - pt[9][0], pool.dAUC_W10_minus_W9))
chk.append(('pooled dr point', pt[10][1] - pt[9][1], pool.dr_W10_minus_W9))
for k, sm in enumerate(EVAL):
    row = R[R.row == sm].iloc[0]
    chk.append((f'{sm} AUC W9', pt[9][2][k], row.AUC_W9))
    chk.append((f'{sm} AUC W10', pt[10][2][k], row.AUC_W10))
    chk.append((f'{sm} r W9', pt[9][3][k], row.r_W9))
    chk.append((f'{sm} r W10', pt[10][3][k], row.r_W10))
    chk.append((f'{sm} n', len(idx[sm]), row.n))
    chk.append((f'{sm} failing', int(y[idx[sm]].sum()), row.n_failing))

# Recall@20%: explicit list construction (lowest ceil(0.2 n) pass-scores, tie-break rng seed 7, one stream over sorted semesters)
for W in (9, 10):
    rng = np.random.default_rng(7)
    tp_tot = fl_tot = 0
    for k, sm in enumerate(EVAL):
        ix = idx[sm]
        order = np.lexsort((rng.random(len(ix)), S[W][ix]))
        listed = ix[order[:math.ceil(0.2 * len(ix))]]
        tp, fl = int(y[listed].sum()), int(y[ix].sum())
        chk.append((f'{sm} Recall@20% W{W}', tp / fl, R[R.row == sm].iloc[0][f'Recall20_W{W}']))
        tp_tot += tp
        fl_tot += fl
    chk.append((f'pooled Recall@20% W{W}', tp_tot / fl_tot, pool[f'Recall20_W{W}']))

# first 300 standing bootstrap draws, recomputed draw by draw
cnt = L.make_counts(int(stu.max()) + 1)
dr = np.load(RES / 'PW_boot_draws.npz')
dA, dR = [], []
for b in range(300):
    a9, r9, _, _ = pooled(9, cnt[b].astype(float))
    a10, r10, _, _ = pooled(10, cnt[b].astype(float))
    dA.append(a10 - a9)
    dR.append(r10 - r9)
chk.append(('boot draws 0-299 max |dAUC diff|', float(np.max(np.abs(np.array(dA) - dr['dAUC'][:300]))), 0.0))
chk.append(('boot draws 0-299 max |dr diff|', float(np.max(np.abs(np.array(dR) - dr['dr'][:300]))), 0.0))
# percentile limits from the stored 2000 draws
chk.append(('pooled dAUC upper 95% CL', float(np.percentile(dr['dAUC'], 97.5)), pool.dAUC_W10_minus_W9_hi))
chk.append(('pooled dr upper 95% CL', float(np.percentile(dr['dr'], 97.5)), pool.dr_W10_minus_W9_hi))

C = pd.DataFrame(chk, columns=['quantity', 'independent', 'reported'])
C['abs_diff'] = (C.independent - C.reported).abs()
C['ok'] = C.abs_diff <= np.where(C.quantity.str.startswith('boot'), 1e-5, 1e-9)
C.to_csv(RES / 'PW_selfcheck.csv', index=False)
pd.set_option('display.width', 200)
print(C.to_string(index=False))
print('ALL OK' if C.ok.all() else 'MISMATCH')
sys.exit(0 if C.ok.all() else 1)
