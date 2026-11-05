# -*- coding: utf-8 -*-
"""R6 Gate 1 -- reproduction with the ORIGINAL outcome.  Frozen scores; no refitting; semester 1142 not used.

Checks, in order:
  V1  weight matrix of the H02 protocol == ews_lib.Boot draw multiplicities (source of Boot loaded from ews_lib.py by ast)
  V2  unit-weight AUC == r5_lib.auc_fail on explicit data;  weighted draw AUC == AUC of explicitly resampled rows
  G1  published increments (main-text Table 2 + Supplementary Table 17 E4): point within +-0.003
Writes results/gate1_matrix.csv, results/gate1_decision.json, results/gate1_log.txt.  Exit code 1 if Gate 1 fails."""
import ast
import hashlib
import json
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import r5_lib as L
import r6_lib as R

T0 = time.time()
LOG = open(L.RES / 'gate1_log.txt', 'w', encoding='utf-8')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + '\n')
    LOG.flush()


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


log('# R6 Gate 1  utc=%s' % time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
log('python', platform.python_version(), 'numpy', np.__version__, 'pandas', pd.__version__,
    'scipy', __import__('scipy').__version__)
log('NOTE: published environment was numpy 2.4.4 / pandas 3.0.2 / scipy 1.17.1; V1 below checks the RNG stream directly')
for f in ['oof_scores_keyed.csv', 'd7_semester_scores.csv',
          'datapack_complete/08_class_membership_and_1142_outcomes/class_membership5.csv',
          'datapack_complete/03_class_membership/class_membership.csv']:
    log('sha256', f, sha(L.PACK / f))
for f in ['r5_lib.py', 'r6_lib.py', 'r6_01_gate1.py']:
    log('sha256', f, sha(L.HERE / f))
log('bootstrap: B=%d; H02 seed %d (ews_lib.Boot order); R5 seed %d (r5_lib order)' % (R.BOOT_B, R.SEED_H02, R.SEED_R5))

base, scores = L.load_base()
assert len(base) == 2652 and int(base.y.sum()) == 560 and base.stu.nunique() == 2473
assert sorted(base.semester.unique()) == L.D7                      # development semesters only
sem_i = base.sem_i.to_numpy()
y0 = base.y.to_numpy().astype(int)
S = R.build_scores(base, scores)
SUB = R.subsets(base)
log('base n=%d failing=%d students=%d; subsets %s' % (len(base), y0.sum(), base.stu.nunique(),
                                                      {k: len(v) for k, v in SUB.items()}))
log('B1 (r5_lib.b1_scores) == B1 (h02 b1_risk): asserted exactly equal in build_scores')

# ----------------------------------------------------------------------------------------------------------------
# V1: weight matrix equals ews_lib.Boot
# ----------------------------------------------------------------------------------------------------------------
EWS = L.PACK / 'datapack_complete/04_pipeline_code/ews_lib.py'
src = open(EWS, encoding='utf-8').read()
tree = ast.parse(src)
ns = {'np': np, 'rankdata': rankdata}
for node in tree.body:
    if isinstance(node, ast.Assign) and any(getattr(t, 'id', '') in ('BOOT_B', 'BOOT_SEED') for t in node.targets):
        exec(ast.get_source_segment(src, node), ns)
    if isinstance(node, ast.FunctionDef) and node.name == 'auc_fail':
        exec(ast.get_source_segment(src, node), ns)
    if isinstance(node, ast.ClassDef) and node.name == 'Boot':
        exec(ast.get_source_segment(src, node), ns)
assert ns['BOOT_SEED'] == R.SEED_H02 and ns['BOOT_B'] == R.BOOT_B
boot = ns['Boot'](base.anon_id.to_numpy())
W_H02 = R.boot_weights(base, 'H02')
N = len(base)
bad = 0
for b in range(R.BOOT_B):
    if not np.array_equal(np.bincount(boot.idx[b], minlength=N).astype(float), W_H02[b]):
        bad += 1
log('V1 H02 weight matrix vs ews_lib.Boot (source from ews_lib.py): draws differing = %d of %d -> %s'
    % (bad, R.BOOT_B, 'PASS' if bad == 0 else 'FAIL'))
assert bad == 0
W_R5 = R.boot_weights(base, 'R5')
assert np.array_equal(W_R5, L.make_weights(base.stu.to_numpy()).astype(np.float64))
log('V1b R5 weight matrix == r5_lib.make_weights: PASS')

# ----------------------------------------------------------------------------------------------------------------
# V2: AUC machinery
# ----------------------------------------------------------------------------------------------------------------
cols = ['blend_pool_W8', 'blend_dep_W8', 'B1', 'B3_W8']
Sm = np.column_stack([S[c] for c in cols])
worst = 0.0
for basis in ('pooled', 'within'):
    pt, _ = R.eval_cols(np.ones((1, N)), Sm, y0, SUB['all'], sem_i, basis)
    if basis == 'pooled':
        ref = [L.auc_fail(y0, Sm[:, j]) for j in range(len(cols))]
    else:
        ref = []
        for j in range(len(cols)):
            a, n = [], []
            for s in range(7):
                m = sem_i == s
                a.append(L.auc_fail(y0[m], Sm[m, j]))
                n.append(m.sum())
            ref.append(float(np.average(a, weights=n)))
    worst = max(worst, float(np.max(np.abs(pt - np.array(ref)))))
log('V2a unit-weight AUC vs r5_lib.auc_fail (pooled and record-weighted within-semester): max abs diff = %.2e' % worst)
assert worst < 1e-12
w2 = 0.0
for b in (0, 1, 2, 999, 1999):
    ix = boot.idx[b]
    ref = np.array([ns['auc_fail'](y0[ix], Sm[ix, j]) for j in range(len(cols))])
    g, _ = R.auc_groups(W_H02[b:b + 1], Sm, y0, [np.arange(N)])
    w2 = max(w2, float(np.max(np.abs(g[0, 0, :] - ref))))
log('V2b weighted draw AUC vs AUC of explicitly resampled rows (ews_lib.auc_fail), draws 0,1,2,999,1999: max abs diff = %.2e' % w2)
assert w2 < 1e-12

# ----------------------------------------------------------------------------------------------------------------
# G1: grid of candidate computations under both protocols
# ----------------------------------------------------------------------------------------------------------------
PUB = {'E1a': (0.081, 0.054, 0.108), 'E1b': (0.056, 0.029, 0.083), 'E2a': (0.033, 0.002, 0.065),
       'E2b': (0.027, -0.005, 0.060), 'E2c': (0.047, 0.016, 0.078), 'E3a': (0.012, -0.006, 0.031),
       'E3b': (-0.009, None, None), 'E4': (0.036, 0.020, 0.051)}
# (id, candidate label, basis, subset, minuend, subtrahend)
CAND = [('E1a', 'blend_pool', 'pooled', 'all', 'blend_pool_W8', 'B1'),
        ('E1b', 'blend_pool', 'within', 'all', 'blend_pool_W8', 'B1'),
        ('E1b', 'blend_dep', 'within', 'all', 'blend_dep_W8', 'B1'),
        ('E2a', 'blend_pool', 'pooled', 'main', 'blend_pool_W8', 'B2'),
        ('E2b', 'blend_pool', 'pooled', 'clean', 'blend_pool_W8', 'B2'),
        ('E2c', 'main|blend_pool', 'within', 'main', 'blend_pool_W8', 'B2'),
        ('E2c', 'main|blend_dep', 'within', 'main', 'blend_dep_W8', 'B2'),
        ('E2c', 'clean|blend_pool', 'within', 'clean', 'blend_pool_W8', 'B2'),
        ('E2c', 'clean|blend_dep', 'within', 'clean', 'blend_dep_W8', 'B2'),
        ('E3a', 'blend_pool', 'pooled', 'all', 'blend_pool_W8', 'B3_W8'),
        ('E3b', 'blend_pool', 'within', 'all', 'blend_pool_W8', 'B3_W8'),
        ('E3b', 'blend_dep', 'within', 'all', 'blend_dep_W8', 'B3_W8'),
        ('E4', 'blend_dep', 'within', 'all', 'B3_W10', 'blend_dep_W10'),
        ('E4', 'blend_pool', 'within', 'all', 'B3_W10', 'blend_pool_W10')]
NEED = {}
for _id, _lab, basis, sub, a, b in CAND:
    NEED.setdefault((basis, sub), set()).update([a, b])

rows = []
cache = {}
for proto, Wm in (('H02', W_H02), ('R5', W_R5)):
    for (basis, sub), names in NEED.items():
        names = sorted(names)
        Smat = np.column_stack([S[n] for n in names])
        pt, dr = R.eval_cols(Wm, Smat, y0, SUB[sub], sem_i, basis)
        for j, n in enumerate(names):
            cache[(proto, basis, sub, n)] = (float(pt[j]), dr[:, j])
    for _id, lab, basis, sub, a, b in CAND:
        pa, da = cache[(proto, basis, sub, a)]
        pb, db = cache[(proto, basis, sub, b)]
        est = pa - pb
        lo, hi = L.ci(da - db)
        pe, pl, ph = PUB[_id]
        rows.append({'protocol': proto, 'estimand': _id, 'candidate': lab, 'basis': basis, 'subset': sub,
                     'est': est, 'lo': lo, 'hi': hi, 'pub_est': pe, 'pub_lo': pl, 'pub_hi': ph,
                     'd_est': est - pe,
                     'd_lo': (lo - pl) if pl is not None else np.nan, 'd_hi': (hi - ph) if ph is not None else np.nan,
                     'point_ok': abs(round(est, 3) - pe) <= 0.003 + 1e-9,
                     'ci_ok': (pl is not None) and abs(round(lo, 3) - pl) < 1e-9 and abs(round(hi, 3) - ph) < 1e-9})
M = pd.DataFrame(rows)
M.to_csv(L.RES / 'gate1_matrix.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250, 'display.max_columns', 30)
log('\nG1 matrix (estimate / 95% percentile CI vs published; point_ok = |round(est,3)-published| <= 0.003; '
    'ci_ok = both ends equal published at 3 decimals)')
log(M.round(4).to_string(index=False))

# ---- robustness: stored m3 column instead of the recomputed pooled rank blend --------------------------------
dm = float(np.abs(S['blend_m3_W8'] - S['blend_pool_W8']).max())
log('\nstored m3 vs recomputed pooled rank blend (W8): max abs value difference = %.3e' % dm)
mx = 0.0
for _id, lab, basis, sub, a, b in CAND:
    if 'blend_pool_W8' not in (a, b):
        continue
    a2, b2 = a.replace('blend_pool', 'blend_m3'), b.replace('blend_pool', 'blend_m3')
    Smat = np.column_stack([S[a2], S[b2]])
    pt, dr = R.eval_cols(W_H02, Smat, y0, SUB[sub], sem_i, basis)
    est = pt[0] - pt[1]
    lo, hi = L.ci(dr[:, 0] - dr[:, 1])
    ref = M[(M.protocol == 'H02') & (M.estimand == _id) & (M.candidate == lab)].iloc[0]
    d = max(abs(est - ref.est), abs(lo - ref.lo), abs(hi - ref.hi))
    mx = max(mx, d)
log('stored-m3 column: max abs difference in estimate / CI end over the blend_pool candidates (H02 protocol) = %.2e' % mx)

# ---- optional: compare with the prior H-02 result files at full precision --------------------------------------
import os
from pathlib import Path
PRIOR = Path(os.environ.get('R6_PRIOR', L.HERE.parent.parent / 'r6s' / '03_prior_h02'))
fB = PRIOR / 'rerun3_A_H02_B1B2/results/H02B_delta_auc.csv'
f3 = PRIOR / 'phase2_H02/H02_delta_auc.csv'
if fB.exists() and f3.exists():
    h = pd.read_csv(fB)
    h3 = pd.read_csv(f3)
    pr = [('E1a', 'blend_pool', "M3' - B1", 'E_all', 'pooled'), ('E1b', 'blend_pool', "M3' - B1", 'E_all', 'wmean'),
          ('E2a', 'blend_pool', "M3' - B2", 'E_b2', 'pooled'), ('E2b', 'blend_pool', "M3' - B2", 'E_b2_classprior', 'pooled'),
          ('E2c', 'clean|blend_pool', "M3' - B2", 'E_b2_classprior', 'wmean'),
          ('E2c', 'main|blend_pool', "M3' - B2", 'E_b2', 'wmean')]
    mxp = 0.0
    for e, c, cmp_, es, sc in pr:
        a = M[(M.protocol == 'H02') & (M.estimand == e) & (M.candidate == c)].iloc[0]
        b = h[(h.comparison == cmp_) & (h.eval_set == es) & (h.scope == sc)].iloc[0]
        mxp = max(mxp, abs(a.est - b.delta_auc), abs(a.lo - b.ci_lo), abs(a.hi - b.ci_hi))
    for e, sc in (('E3a', 'pooled'), ('E3b', 'weighted_within_semester')):
        a = M[(M.protocol == 'H02') & (M.estimand == e) & (M.candidate == 'blend_pool')].iloc[0]
        b = h3[(h3.baseline == 'B3 quiz pass rate') & (h3['subset'] == 'all') & (h3.metric == sc)].iloc[0]
        mxp = max(mxp, abs(a.est - b.delta_auc), abs(a.lo - b.lo), abs(a.hi - b.hi))
    log('vs prior H-02 result files (H02B_delta_auc.csv, H02_delta_auc.csv), 8 comparisons incl. E2c main and clean: '
        'max abs difference in estimate / CI end = %.2e' % mxp)
else:
    log('prior H-02 result files not found at %s; comparison skipped' % PRIOR)

# ---- per-score AUC anchors (informational) -------------------------------------------------------------------
AUCP = [('blend W8 pooled, all', 'pooled', 'all', 'blend_pool_W8', 0.727, None, None),
        ('blend W8 pooled, main subset', 'pooled', 'main', 'blend_pool_W8', 0.719, None, None),
        ('blend W8 within (pooled rank blend)', 'within', 'all', 'blend_pool_W8', 0.730, None, None),
        ('blend W8 within (deploy blend)', 'within', 'all', 'blend_dep_W8', 0.730, None, None),
        ('B1 W8 pooled, all', 'pooled', 'all', 'B1', 0.647, None, None),
        ('B2 pooled, main subset', 'pooled', 'main', 'B2', 0.686, None, None),
        ('B3 W8 pooled', 'pooled', 'all', 'B3_W8', 0.715, 0.691, 0.741),
        ('B3 W10 pooled', 'pooled', 'all', 'B3_W10', 0.747, 0.724, 0.769),
        ('B3 W8 within', 'within', 'all', 'B3_W8', 0.739, None, None),
        ('B3 W10 within (deployment)', 'within', 'all', 'B3_W10', 0.771, 0.748, 0.794)]
arows = []
for lab, basis, sub, col, pe, pl, ph in AUCP:
    key = [k for k in cache if k[1] == basis and k[2] == sub and k[3] == col]
    if not key:
        pt, dr = R.eval_cols(W_R5, S[col][:, None], y0, SUB[sub], sem_i, basis)
        p, d = float(pt[0]), dr[:, 0]
        p_h, d_h = p, R.eval_cols(W_H02, S[col][:, None], y0, SUB[sub], sem_i, basis)[1][:, 0]
    else:
        p, d = cache[('R5',) + key[0][1:]][0], cache[('R5',) + key[0][1:]][1]
        p_h, d_h = cache[('H02',) + key[0][1:]][0], cache[('H02',) + key[0][1:]][1]
    arows.append({'score': lab, 'point': p, 'H02_lo': L.ci(d_h)[0], 'H02_hi': L.ci(d_h)[1],
                  'R5_lo': L.ci(d)[0], 'R5_hi': L.ci(d)[1], 'pub': pe, 'pub_lo': pl, 'pub_hi': ph,
                  'point_ok': abs(round(p, 3) - pe) <= 0.003 + 1e-9})
A = pd.DataFrame(arows)
A.to_csv(L.RES / 'gate1_auc_anchors.csv', index=False, encoding='utf-8-sig')
log('\nG1 AUC anchors (informational)')
log(A.round(4).to_string(index=False))

# ---- Gate 1 verdict (point estimates only, per handoff section 3) -------------------------------------------
ids = list(PUB)
ok_by_id = {i: bool(M[(M.estimand == i) & M.point_ok].shape[0] > 0) for i in ids}
log('\nGate 1 verdict: an estimand passes if at least one defined candidate reproduces its point estimate within +-0.003')
log(ok_by_id)
passed = all(ok_by_id.values())
log('GATE 1:', 'PASS' if passed else 'FAIL')
json.dump({'gate1_pass': bool(passed), 'by_estimand': ok_by_id,
           'elapsed_s': round(time.time() - T0, 1)}, open(L.RES / 'gate1_decision.json', 'w'), indent=1)
LOG.close()
sys.exit(0 if passed else 1)
