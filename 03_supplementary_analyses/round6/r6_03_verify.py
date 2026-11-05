# -*- coding: utf-8 -*-
"""R6 independent verification.  Second implementation of every row of R6_results.csv:
explicit resampling of record indices (no weights), AUC by r5_lib.auc_fail (rankdata), within-semester mean by
np.average -- no code shared with r6_lib.auc_groups / eval_cols.  Compares estimate, ci_lo, ci_hi row by row."""
import re
import sys

import numpy as np
import pandas as pd

import r5_lib as L
import r6_lib as R

base, scores = L.load_base()
N = len(base)
sem_i = base.sem_i.to_numpy()
S = R.build_scores(base, scores)
SUB = R.subsets(base)
V = R.variant1_labels(base)
Y = {'original': V['y0'], 'variant1': V['y1']}
res = pd.read_csv(L.RES / 'R6_results.csv', encoding='utf-8-sig')
LOGF = open(L.RES / 'r6_verify_log.txt', 'w', encoding='utf-8')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOGF.write(s + '\n')
    LOGF.flush()


def explicit_idx(protocol):
    if protocol == 'H02':
        inv = np.unique(base.anon_id.to_numpy(), return_inverse=True)[1].reshape(-1)
        seed = R.SEED_H02
    else:
        inv = base.stu.to_numpy()
        seed = R.SEED_R5
    n = int(inv.max()) + 1
    rows = [np.flatnonzero(inv == i) for i in range(n)]
    draws = np.random.default_rng(seed).integers(0, n, size=(R.BOOT_B, n))
    return [np.concatenate([rows[c] for c in d]) for d in draws]


def metric(ix, y, s, mask, basis):
    ix = ix[mask[ix]]
    if basis == 'pooled':
        return L.auc_fail(y[ix], s[ix])
    a, w = [], []
    for k in range(len(R.D7)):
        m = sem_i[ix] == k
        if not m.any():
            continue
        yk = y[ix][m]
        if yk.sum() in (0, m.sum()):
            continue
        a.append(L.auc_fail(yk, s[ix][m]))
        w.append(int(m.sum()))
    return float(np.average(a, weights=w))


MASK = {k: np.isin(np.arange(N), v) for k, v in SUB.items()}
IDX = {p: explicit_idx(p) for p in ('H02', 'R5')}
PT = np.arange(N)

# row definitions (must mirror what r6_02_variant1.py wrote; matched by label text, not by shared code)
EST = {'E1a': ('pooled', 'all', 'blend_pool_W8', 'B1', 'H02'), 'E1b': ('within', 'all', 'blend_pool_W8', 'B1', 'H02'),
       'E2a': ('pooled', 'main', 'blend_pool_W8', 'B2', 'H02'), 'E2b': ('pooled', 'clean', 'blend_pool_W8', 'B2', 'H02'),
       'E2c': ('within', 'clean', 'blend_pool_W8', 'B2', 'H02'), 'E3a': ('pooled', 'all', 'blend_pool_W8', 'B3_W8', 'H02'),
       'E3b': ('within', 'all', 'blend_pool_W8', 'B3_W8', 'H02'), 'E4': ('within', 'all', 'B3_W10', 'blend_dep_W10', 'R5')}
SCOL = {('blend', 'W8'): 'blend_pool_W8', ('blend', 'W10'): 'blend_dep_W10', ('B1', 'W8'): 'B1', ('B2', 'W8'): 'B2',
        ('B3', 'W8'): 'B3_W8', ('B3', 'W10'): 'B3_W10'}
SUBOF = {2652: 'all', 2180: 'main', 1907: 'clean'}

worst = 0.0
nrow = 0
for _, r in res.iterrows():
    y = Y[r.variant]
    mask = MASK[SUBOF[int(r.n)]]
    m = re.match(r'^(E\d[abc]?) ', r.comparison)
    if m:
        basis, sub, a, b, proto = EST[m.group(1)]
        assert sub == SUBOF[int(r.n)] and basis == r.basis
        pt = metric(PT, y, S[a], mask, basis) - metric(PT, y, S[b], mask, basis)
        d = np.array([metric(ix, y, S[a], mask, basis) - metric(ix, y, S[b], mask, basis) for ix in IDX[proto]])
    else:
        sc = re.match(r'^AUC (\w+)$', r.comparison).group(1)
        col = SCOL[(sc, r.week)]
        pt = metric(PT, y, S[col], mask, r.basis)
        d = np.array([metric(ix, y, S[col], mask, r.basis) for ix in IDX['R5']])
    lo, hi = L.ci(d)
    diff = max(abs(pt - r.estimate), abs(lo - r.ci_lo), abs(hi - r.ci_hi))
    worst = max(worst, diff)
    nrow += 1
    if diff > 2e-9:
        log('MISMATCH', r.variant, r.comparison, r.week, r.basis, 'diff %.2e' % diff)
log('rows compared: %d; max abs difference in estimate / ci_lo / ci_hi (CSV stores 10 decimals): %.2e' % (nrow, worst))
ok = worst <= 1e-8
log('INDEPENDENT VERIFICATION:', 'PASS' if ok else 'FAIL')
import json
json.dump({'rows': nrow, 'max_abs_diff': worst, 'pass': bool(ok)}, open(L.RES / 'r6_verify.json', 'w'))
LOGF.close()
sys.exit(0 if ok else 1)
