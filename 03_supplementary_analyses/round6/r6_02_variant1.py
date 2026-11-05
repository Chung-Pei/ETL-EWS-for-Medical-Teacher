# -*- coding: utf-8 -*-
"""R6 -- increments under the bonus-free outcome (Variant 1).  Post hoc.  Frozen scores; semester 1142 not used.

Order: (0) Gate 1 must have passed; (1) Gate 2 construction asserts; (2) estimands under the ORIGINAL and the
VARIANT 1 outcome, same scores, same weights; (3) write results/R6_results.csv.
Configuration frozen at Gate 1 (see results/gate1_log.txt):
  E1a-E3b : H02 bootstrap (seed 20261006, ews_lib.Boot student order), pooled rank blend of frozen m1/m2
  E2c     : clean subset S3 (n = 1,907)
  E4      : R5 bootstrap (seed 20261031), deployment blend (within-semester ranks), Week 10
  per-score AUC rows: R5 bootstrap (reproduces the published B3 intervals and the R5-8 blend rows)
B1 and B2 are built once from the ORIGINAL y and held fixed (handoff section 2)."""
import hashlib
import json
import platform
import sys
import time

import numpy as np
import pandas as pd

import r5_lib as L
import r6_lib as R

T0 = time.time()
LOG = open(L.RES / 'r6_variant1_log.txt', 'w', encoding='utf-8')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + '\n')
    LOG.flush()


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


# ---- (0) Gate 1 must have passed ---------------------------------------------------------------------------------
g1 = json.load(open(L.RES / 'gate1_decision.json'))
if not g1['gate1_pass']:
    sys.exit('Gate 1 did not pass: stop (do not proceed to Variant 1)')
log('# R6 Variant 1  utc=%s' % time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
log('python', platform.python_version(), 'numpy', np.__version__, 'pandas', pd.__version__,
    'scipy', __import__('scipy').__version__)
for f in ['r5_lib.py', 'r6_lib.py', 'r6_01_gate1.py', 'r6_02_variant1.py']:
    log('sha256', f, sha(L.HERE / f))
log('Gate 1 decision file: PASS')

base, scores = L.load_base()
N = len(base)
sem_i = base.sem_i.to_numpy()
S = R.build_scores(base, scores)
SUB = R.subsets(base)
V = R.variant1_labels(base)
y0, y1 = V['y0'], V['y1']
assert sorted(base.semester.unique()) == L.D7, 'only development semesters may be present'

# ---- (1) Gate 2 --------------------------------------------------------------------------------------------------
n_change = V['to_pass'] + V['to_fail']
g2 = {'bonus_counts': V['cnt'], 'bonus_total': int(V['add'].sum()), 'rec>=60_comp<60': V['n56'],
      'label_changes': n_change, 'fail_to_pass': V['to_pass'], 'pass_to_fail': V['to_fail'],
      'failing_y0': int(y0.sum()), 'failing_y1': int(y1.sum())}
rec = ((base.anon_id == 'WITHHELD') & (base.semester == '1141')).to_numpy()
assert rec.sum() == 1
g2['WITHHELD'] = {'y0': int(y0[rec][0]), 'y1': int(y1[rec][0]), 'd7_semester_score': float(base.semester_score[rec].iloc[0]),
                  'oof_final': float(base.final[rec].iloc[0]), 'composite': float(V['comp'][rec][0])}
log('Gate 2 construction:', g2)
ok2 = (V['cnt'] == {'1121': 5, '1131': 58, '1141': 204} and g2['bonus_total'] == 267 and V['n56'] == 56
       and n_change == 57 and V['to_pass'] == 1 and V['to_fail'] == 56 and int(y1.sum()) == 615
       and g2['WITHHELD']['y0'] == 1 and g2['WITHHELD']['y1'] == 0 and abs(g2['WITHHELD']['composite'] - 64.0) < 1e-9)
log('GATE 2 (57 label changes; failing 560 -> 615; bonus counts 5/58/204; WITHHELD as specified):', 'PASS' if ok2 else 'FAIL')
if not ok2:
    sys.exit('Gate 2 failed: stop')
assert (y1[~V['add']] == y0[~V['add']]).all(), 'labels outside the 267 bonus records changed'
log('labels differ from the original only on bonus-carrying records: confirmed; y0 sha %s y1 sha %s'
    % (hashlib.sha256(y0.tobytes()).hexdigest()[:16], hashlib.sha256(y1.tobytes()).hexdigest()[:16]))
log('score arrays (frozen, identical for both outcomes) sha: ' +
    ', '.join('%s %s' % (k, hashlib.sha256(np.nan_to_num(v).tobytes()).hexdigest()[:10]) for k, v in sorted(S.items())))

W = {'H02': R.boot_weights(base, 'H02'), 'R5': R.boot_weights(base, 'R5')}

# ---- (2) estimand definitions -------------------------------------------------------------------------------------
# id, label, basis, subset, minuend, subtrahend, protocol, week, definition note
EST = [
    ('E1a', 'blend - B1', 'pooled', 'all', 'blend_pool_W8', 'B1', 'H02', 'W8', 'pooled; all records'),
    ('E1b', 'blend - B1', 'within', 'all', 'blend_pool_W8', 'B1', 'H02', 'W8', 'within-semester; all records'),
    ('E2a', 'blend - B2', 'pooled', 'main', 'blend_pool_W8', 'B2', 'H02', 'W8', 'pooled; main subset (semester != 1111)'),
    ('E2b', 'blend - B2', 'pooled', 'clean', 'blend_pool_W8', 'B2', 'H02', 'W8', 'pooled; clean subset S3'),
    ('E2c', 'blend - B2', 'within', 'clean', 'blend_pool_W8', 'B2', 'H02', 'W8',
     'within-semester; clean subset S3 (the subset that reproduces the published value at Gate 1)'),
    ('E3a', 'blend - B3', 'pooled', 'all', 'blend_pool_W8', 'B3_W8', 'H02', 'W8', 'pooled; all records'),
    ('E3b', 'blend - B3', 'within', 'all', 'blend_pool_W8', 'B3_W8', 'H02', 'W8',
     'within-semester; all records; no interval was published, interval computed here'),
    ('E4', 'B3 - blend', 'within', 'all', 'B3_W10', 'blend_dep_W10', 'R5', 'W10',
     'deployment (within-semester ranks for the blend); paired'),
]
# per-score AUC rows: score label, column, basis, subset, week
AUC = [
    ('blend', 'blend_pool_W8', 'pooled', 'all', 'W8'), ('blend', 'blend_pool_W8', 'pooled', 'main', 'W8'),
    ('blend', 'blend_pool_W8', 'pooled', 'clean', 'W8'), ('blend', 'blend_pool_W8', 'within', 'all', 'W8'),
    ('blend', 'blend_pool_W8', 'within', 'clean', 'W8'),
    ('B1', 'B1', 'pooled', 'all', 'W8'), ('B1', 'B1', 'within', 'all', 'W8'),
    ('B2', 'B2', 'pooled', 'main', 'W8'), ('B2', 'B2', 'pooled', 'clean', 'W8'), ('B2', 'B2', 'within', 'clean', 'W8'),
    ('B3', 'B3_W8', 'pooled', 'all', 'W8'), ('B3', 'B3_W8', 'within', 'all', 'W8'),
    ('blend', 'blend_dep_W10', 'within', 'all', 'W10'), ('B3', 'B3_W10', 'within', 'all', 'W10'),
]
SUBN = {'all': 'all records', 'main': 'main subset (semester != 1111)', 'clean': 'clean subset S3'}

need = {}
for _id, lab, basis, sub, a, b, proto, wk, nt in EST:
    need.setdefault((proto, basis, sub), set()).update([a, b])
for lab, col, basis, sub, wk in AUC:
    need.setdefault(('R5', basis, sub), set()).add(col)

out_rows = []
res = {}
for vname, y in (('original', y0), ('variant1', y1)):
    for (proto, basis, sub), names in sorted(need.items()):
        names = sorted(names)
        pt, dr = R.eval_cols(W[proto], np.column_stack([S[n] for n in names]), y, SUB[sub], sem_i, basis)
        for j, n in enumerate(names):
            res[(vname, proto, basis, sub, n)] = (float(pt[j]), dr[:, j])

    for _id, lab, basis, sub, a, b, proto, wk, nt in EST:
        pa, da = res[(vname, proto, basis, sub, a)]
        pb, db = res[(vname, proto, basis, sub, b)]
        lo, hi = L.ci(da - db)
        nf = int(y[SUB[sub]].sum())
        out_rows.append({'variant': vname, 'basis': basis, 'comparison': '%s %s' % (_id, lab), 'week': wk,
                         'n': len(SUB[sub]), 'estimate': pa - pb, 'ci_lo': lo, 'ci_hi': hi,
                         'note': 'paired delta AUC; %s; failing=%d; bootstrap %s seed %d B=%d'
                                 % (nt, nf, proto, R.SEED_H02 if proto == 'H02' else R.SEED_R5, R.BOOT_B)})
    for lab, col, basis, sub, wk in AUC:
        p, d = res[(vname, 'R5', basis, sub, col)]
        lo, hi = L.ci(d)
        nf = int(y[SUB[sub]].sum())
        defn = {'blend_pool_W8': 'pooled rank blend of frozen m1,m2 (within-semester AUC of that score on the within basis)',
                'blend_dep_W10': 'deployment blend (within-semester ranks)'}.get(col, 'score as in main-text Table 2 / Table 17')
        out_rows.append({'variant': vname, 'basis': basis, 'comparison': 'AUC %s' % lab, 'week': wk,
                         'n': len(SUB[sub]), 'estimate': p, 'ci_lo': lo, 'ci_hi': hi,
                         'note': 'own AUC; %s; %s; failing=%d; bootstrap R5 seed %d B=%d'
                                 % (SUBN[sub], defn, nf, R.SEED_R5, R.BOOT_B)})

# ---- R5-8 cross-check of the Variant 1 labels (informational Gate 2 supplement) --------------------------------------
Wm5 = W['R5']
pt8, dr8 = R.eval_cols(Wm5, S['blend_m3_W8'][:, None], y1, SUB['all'], sem_i, 'pooled')
pt10, dr10 = R.eval_cols(Wm5, S['blend_dep_W10'][:, None], y1, SUB['all'], sem_i, 'within')
log('R5-8 anchor check (Variant 1 labels, R5 protocol, stored m3 as R5-8): blend W8 pooled AUC %.4f %s (anchor 0.7196 [0.6976, 0.7417]);'
    ' blend W10 deployment AUC %.4f %s (anchor 0.7364 [0.7132, 0.7584])'
    % (pt8[0], tuple(round(x, 4) for x in L.ci(dr8[:, 0])), pt10[0], tuple(round(x, 4) for x in L.ci(dr10[:, 0]))))
assert abs(pt8[0] - 0.7196) < 5e-4 and abs(pt10[0] - 0.7364) < 5e-4

# ---- robustness: stored m3 instead of the recomputed pooled blend; alternative bootstrap seed ----------------------
mxm3 = 0.0
for _id, lab, basis, sub, a, b, proto, wk, nt in EST:
    if 'blend_pool_W8' not in (a, b):
        continue
    a2 = a.replace('blend_pool', 'blend_m3')
    pt, dr = R.eval_cols(W[proto], np.column_stack([S[a2], S[b]]), y1, SUB[sub], sem_i, basis)
    est = pt[0] - pt[1]
    lo, hi = L.ci(dr[:, 0] - dr[:, 1])
    ref = [r for r in out_rows if r['variant'] == 'variant1' and r['comparison'].startswith(_id + ' ')][0]
    mxm3 = max(mxm3, abs(est - ref['estimate']), abs(lo - ref['ci_lo']), abs(hi - ref['ci_hi']))
log('robustness (Variant 1): stored m3 column instead of recomputed pooled blend, max abs change in estimate / CI end = %.2e' % mxm3)

alt = {}
for _id, lab, basis, sub, a, b, proto, wk, nt in EST:
    p2 = 'R5' if proto == 'H02' else 'H02'
    pt, dr = R.eval_cols(W[p2], np.column_stack([S[a], S[b]]), y1, SUB[sub], sem_i, basis)
    alt[_id] = (p2, float(pt[0] - pt[1]), *L.ci(dr[:, 0] - dr[:, 1]))
log('robustness (Variant 1): interval under the OTHER seed (estimand: protocol, est, lo, hi):')
for k, v in alt.items():
    log('  %s %s %+.4f [%+.4f, %+.4f]' % (k, v[0], v[1], v[2], v[3]))

# ---- (3) write -------------------------------------------------------------------------------------------------------
OUT = pd.DataFrame(out_rows)
OUT.to_csv(L.RES / 'R6_results.csv', index=False, encoding='utf-8-sig', float_format='%.10f')
pd.set_option('display.width', 250, 'display.max_columns', 30, 'display.max_colwidth', 60)
log('\nR6_results.csv (%d rows)' % len(OUT))
log(OUT.drop(columns='note').round(4).to_string(index=False))
json.dump({'gate2': g2, 'stored_m3_max_change': mxm3,
           'alt_seed': {k: list(v) for k, v in alt.items()}, 'elapsed_s': round(time.time() - T0, 1)},
          open(L.RES / 'r6_variant1_extra.json', 'w'), indent=1, default=str)
LOG.close()
