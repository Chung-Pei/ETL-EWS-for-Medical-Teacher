# -*- coding: utf-8 -*-
"""VALIDATION GATE (spec: run first, report before item results)."""
import json
import time

import numpy as np
import pandas as pd

import r5_lib as L

t0 = time.time()
base, scores = L.load_base()
N = len(base)
rows = []


def rec(level, anchor, expected, observed, tol, passed, note=''):
    rows.append({'level': level, 'anchor': anchor, 'expected': expected, 'observed': observed,
                 'tolerance': tol, 'pass': bool(passed), 'note': note})


# ---- file-level anchors (arm P, feature_set C, Week 8) ---------------------------------------------
w8 = scores['W8']
y = base.y.to_numpy()
rec('file', 'records in analysis set', 2652, N, 'exact', N == 2652)
rec('file', 'failing records (semester outcome)', 560, int(y.sum()), 'exact', int(y.sum()) == 560)
a3 = L.auc_fail(y, w8.m3)
r3 = L.pearson(w8.m3, base.final)
rec('file', 'W8 blend m3 pooled AUC', 0.7274, round(a3, 4), '4 dp', round(a3, 4) == 0.7274)
rec('file', 'W8 r(m3, final)', 0.4484, round(r3, 4), '4 dp', round(r3, 4) == 0.4484)
ap = L.auc_fail(y, 1 - w8.pfail)
rec('file', 'W8 pfail pooled AUC (pass-oriented 1-pfail)', 0.7092, round(ap, 4), '4 dp', round(ap, 4) == 0.7092)
dm = float(np.abs(w8.m2 - (1 - w8.pfail)).max())
rec('file', 'm2 == 1 - pfail', 'exact', f'max abs diff {dm:.3g}', '<= 1e-15 (float64 rounding)', dm <= 1e-15,
    'identical up to float64 representation')
am1 = L.auc_fail(y, w8.m1)
rec('file', 'm1 (BAS by elimination) W8 AUC, informational', 'n/a', round(am1, 4), 'info', True)
# percentile function: frozen blend reproduced exactly for every week
worst = 0.0
for W in ['W6', 'W8', 'W10', 'W12', 'W14', 'W16']:
    d = float(np.abs(L.pooled_blend(scores[W].m1, scores[W].m2) - scores[W].m3).max())
    worst = max(worst, d)
rec('file', 'percentile function pct_rank = rankdata/n reproduces frozen m3 (all 6 weeks)', 'exact',
    f'max abs diff {worst:.3g}', '<= 1e-15', worst <= 1e-15, 'function of ews_lib.pct_rank')

# ---- package-level anchors ------------------------------------------------------------------------
# B3 raw feature
rows_b3 = []
for s in L.D7:
    d = json.load(open(L.PACK / f'datapack_complete/01_step9_inputs/inputs_D7/out_W8/{s}_step9_inputs.json',
                       encoding='utf-8'))
    for r in d['scatter_data']:
        rows_b3.append((s, r['anon_id'], r['features'].get('quz_pass_rate')))
b3 = pd.DataFrame(rows_b3, columns=['semester', 'anon_id', 'b3'])
mm = base[['semester', 'anon_id']].merge(b3, on=['semester', 'anon_id'], how='left', validate='one_to_one')
assert mm.b3.notna().all()
ab3 = L.auc_fail(y, mm.b3.to_numpy())
rec('package', 'B3 pooled Week-8 AUC (step9 quz_pass_rate, no missing in analysis set)', 0.7154, round(ab3, 4),
    '4 dp', round(ab3, 4) == 0.7154)
# retake count
n_rt = int(base.retake.sum())
rt_sem = base.loc[base.retake == 1, 'semester'].unique().tolist()
rec('package', 'retake count (class_membership.csv class_track=retake, join on anon_id+semester)', 52, n_rt,
    'exact', n_rt == 52 and rt_sem == ['1141'],
    'all in 1141; a join on anon_id alone would return %d rows because retake students also appear in other '
    'semesters' % int(base.anon_id.isin(set(base.anon_id[base.retake == 1])).sum()))
# B1 rebuild
b1 = L.b1_scores(base)
ab1 = L.auc_fail(y, -b1)
inc = a3 - ab1
rec('package', 'B1 increment (W8 blend AUC - B1 AUC), Appendix A rebuild', 0.081, round(inc, 3), '3 dp',
    round(inc, 3) == 0.081, f'B1 AUC {ab1:.4f}; five classes (program5_key) + repeat-student exclusion in training')
alt = {}
for col in ['class_track', 'program5_key']:
    for ex in (False, True):
        alt[f'{col}|excl={ex}'] = round(a3 - L.auc_fail(y, -L.b1_scores(base, col, ex)), 4)
rec('package', 'B1 variants tried (informational)', 'n/a', json.dumps(alt), 'info', True,
    'only five classes + exclusion of test-semester students from training gives +0.081')
# five-class counts
five = base.program5_key.value_counts().to_dict()
exp5 = {'2yr_gen': 1008, '2yr_working': 732, 'post': 169, '4yr': 691, 'retake': 52}
rec('package', 'five class counts (R5-2 Addendum 2)', json.dumps(exp5), json.dumps(five), 'exact', five == exp5)
rt5 = set(base.loc[base.program5_key == 'retake', 'anon_id'])
rt3 = set(base.loc[base.retake == 1, 'anon_id'])
rec('package', 'class_membership5 retake set == class_membership retake set', 'identical',
    'identical' if rt5 == rt3 else f'differ: {sorted(rt5 ^ rt3)}', 'exact', rt5 == rt3)
# d7 / discrepancy
rec('package', 'd7 rows / empty-score rows', '2673 / 11', f'{len(pd.read_csv(L.PACK / "d7_semester_scores.csv"))} / '
    f'{int(pd.read_csv(L.PACK / "d7_semester_scores.csv").final_score.isna().sum())}', 'info', True,
    'analysis set = 2,652 keyed records')
disc = base[(base.anon_id == 'WITHHELD') & (base.semester == '1141')]
rec('package', 'known discrepancy WITHHELD (1141)', 'd7 mid 54, final exam 74, sem 66.8; OOF final 53.2, y=1',
    disc[['semester', 'midterm_score', 'final_score', 'semester_score', 'final', 'y']].to_dict('records').__str__(),
    'info', True)
ex_final = int((base.final_score < 60).sum())
rec('package', 'final-examination failing count (d7 final_score < 60)', 492, ex_final, 'exact', ex_final == 492)

# ---- selection-rule reproduction on frozen Week-8 scores ------------------------------------------
Wm = L.make_weights(base.stu.to_numpy())
f = base.final.to_numpy(float)
res_sel = {}
for defn in ('pooled', 'deploy'):
    names, S = L.build_candidates(base, scores, defn)
    groups = L.sem_groups_of(base.sem_i)
    a_d, r_d = L.metrics_draws(Wm, S, y, f, groups, defn)
    ones = np.ones((1, N), np.float32)
    a_pt, r_pt = L.metrics_draws(ones, S, y, f, groups, defn)
    pt = {n: float(a_pt[0, i]) for i, n in enumerate(names)}
    rpt = {n: float(r_pt[0, i]) for i, n in enumerate(names)}
    sel = L.select_rule(a_d, r_d, names, pt)
    res_sel[defn] = (sel, names, a_d, r_d, pt, rpt)
    col = {n: i for i, n in enumerate(names)}
    d = a_d[:, col[('W8', 'M3')]] - a_d[:, col[('W8', 'M2')]]
    lo, hi = L.ci(d)
    print(f'[{defn}] W*: {sel["wstar"]}  selected: {sel["sel_week"]}+{sel["sel_pred"]}')
    for cr in sel['compare']:
        print('    third-rule comparison', cr)
    print(f'[{defn}] W8 blend - XGBoost: {pt[("W8", "M3")] - pt[("W8", "M2")]:+.4f} [{lo:.4f}, {hi:.4f}]')
    for dd in sel['detail']:
        print('   ', dd)
    # alternative upper-limit convention (one-sided 95th percentile), informational
    loss = a_d[:, col[('W12', 'M3')]] - a_d[:, col[('W8', 'M3')]]
    print(f'[{defn}] W8 blend AUC-loss upper limit: 97.5th pct = {np.percentile(loss, 97.5):.4f}; '
          f'95th pct = {np.percentile(loss, 95):.4f}')
sel_p = res_sel['pooled'][0]
ok_choice = (sel_p['sel_week'] == 'W8' and sel_p['sel_pred'] == 'M3')
pt = res_sel['pooled'][4]
names = res_sel['pooled'][1]
col = {n: i for i, n in enumerate(names)}
d = res_sel['pooled'][2][:, col[('W8', 'M3')]] - res_sel['pooled'][2][:, col[('W8', 'M2')]]
lo, hi = L.ci(d)
rec('package', 'Appendix B rule on frozen scores (pooled) reproduces Week-8 blend choice', 'W8 + blend',
    f'{sel_p["sel_week"]} + {sel_p["sel_pred"]}', 'exact', ok_choice,
    f'W*: {sel_p["wstar"]}')
rec('package', 'blend - XGBoost at W8 (pooled AUC): published +0.018 [0.006, 0.032]', '+0.018 [0.006, 0.032]',
    f'{pt[("W8", "M3")] - pt[("W8", "M2")]:+.3f} [{lo:.3f}, {hi:.3f}]', '3 dp (seed differs from the published run)',
    abs((pt[("W8", "M3")] - pt[("W8", "M2")]) - 0.018) < 0.0006)

G = pd.DataFrame(rows)
G.to_csv(L.RES / 'R5-0_validation_gate.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250, 'display.max_colwidth', 110)
print(G[['level', 'anchor', 'expected', 'observed', 'pass']].to_string())
print('FILE-LEVEL ALL PASS:', bool(G[G.level == 'file']['pass'].all()))
print('PACKAGE-LEVEL FAILS:', G[(G.level == 'package') & (~G['pass'])]['anchor'].tolist())
print(f'elapsed {time.time() - t0:.1f}s')
