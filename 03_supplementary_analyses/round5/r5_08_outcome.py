# -*- coding: utf-8 -*-
"""R5-8 -- outcome-construction sensitivity (reading-time bonus).  Frozen scores; both selection definitions."""
import numpy as np
import pandas as pd

import r5_lib as L

base, scores = L.load_base()
N = len(base)
comp = ((base.midterm_score + base.final_score) / 2).to_numpy(float)
rec_score = base.semester_score.to_numpy(float)
add = (rec_score - comp) > 0.05
cnt = base[add].semester.value_counts().sort_index().to_dict()
n56 = int((add & (rec_score >= 60) & (comp < 60)).sum())
gate_ok = (cnt == {'1121': 5, '1131': 58, '1141': 204}) and add.sum() == 267 and n56 == 56
print('gate: counts', cnt, 'total', int(add.sum()), 'recorded>=60 & composite<60:', n56, 'PASS' if gate_ok else 'FAIL')
assert gate_ok, 'R5-8 validation gate failed: stop this item and say so in the memo'
rows = [{'variant': 'gate', 'definition': '', 'row_type': 'counts', 'predictor': '', 'week': '', 'estimate': int(add.sum()),
         'ci_lo': np.nan, 'ci_hi': np.nan,
         'note': f'addition-carrying records by semester {cnt}; recorded>=60 with pure composite<60: {n56} (gate PASS)'}]

# ---- primary: relabel the 267 records from the pure composite; frozen scores/ranks unchanged ----------------------------
y0 = base.y.to_numpy().astype(int)
f0 = base.final.to_numpy(float)
y1 = y0.copy()
f1 = f0.copy()
y1[add] = (comp[add] < 60).astype(int)
f1[add] = comp[add]
flips_to_pass = int(((y0 == 1) & (y1 == 0)).sum())
flips_to_fail = int(((y0 == 0) & (y1 == 1)).sum())
rows.append({'variant': 'primary_relabel', 'definition': '', 'row_type': 'label_changes', 'predictor': '', 'week': '',
             'estimate': flips_to_pass + flips_to_fail, 'ci_lo': np.nan, 'ci_hi': np.nan,
             'note': f'fail->pass {flips_to_pass}; pass->fail {flips_to_fail}; failing {int(y0.sum())} -> {int(y1.sum())}; '
                     f'record WITHHELD (1141): d7 semester score 66.8 vs OOF final 53.2 -> composite {comp[(base.anon_id == "WITHHELD").to_numpy() & (base.semester == "1141").to_numpy()][0]:.1f}'})
Wm = L.make_weights(base.stu.to_numpy())
for defn, bw in (('pooled', 'W8'), ('deploy', 'W10')):
    res = L.run_rule(base, scores, y1, f1, defn, Wm, recompute_pooled_m3=False)
    rows += L.rule_rows('primary_relabel', defn, res, bw, 'frozen scores and frozen ranks; continuous outcome for r = '
                        'pure composite on the 267 records')
    s = res['sel']
    print('PRIMARY', defn, s['wstar'], s['sel_week'], s['sel_pred'])
    col = {n: i for i, n in enumerate(res['names'])}
    k = (bw, 'M3')
    print('   blend', bw, round(res['auc_pt'][k], 4), L.ci(res['auc_d'][:, col[k]]), round(res['r_pt'][k], 4),
          L.ci(res['r_d'][:, col[k]]))
    for c in s['compare']:
        print('   third rule', c)
    for d in s['detail']:
        if d['week'] in ('W8',):
            print('   ', d)

# ---- secondary: drop the 267; recompute percentile ranks within the retained set ----------------------------------------------
keep = ~add
b, sc = L.subset_view(base, scores, keep)
y2, f2 = y0[keep], f0[keep]
rows.append({'variant': 'secondary_exclusion', 'definition': '', 'row_type': 'counts', 'predictor': '', 'week': '',
             'estimate': int(keep.sum()), 'ci_lo': np.nan, 'ci_hi': np.nan,
             'note': f'retained n={int(keep.sum())}, failing={int(y2.sum())}, students={b.stu.nunique()}; percentile ranks '
                     'recomputed over the retained set'})
W2 = L.make_weights(b.stu.to_numpy())
for defn, bw in (('pooled', 'W8'), ('deploy', 'W10')):
    res = L.run_rule(b, sc, y2, f2, defn, W2, recompute_pooled_m3=True)
    rows += L.rule_rows('secondary_exclusion', defn, res, bw, 'ranks recomputed over the retained set')
    s = res['sel']
    print('SECONDARY', defn, s['wstar'], s['sel_week'], s['sel_pred'])
    col = {n: i for i, n in enumerate(res['names'])}
    k = (bw, 'M3')
    print('   blend', bw, round(res['auc_pt'][k], 4), L.ci(res['auc_d'][:, col[k]]), round(res['r_pt'][k], 4),
          L.ci(res['r_d'][:, col[k]]))
    for c in s['compare']:
        print('   third rule', c)
pd.DataFrame(rows).to_csv(L.RES / 'R5-8.csv', index=False, encoding='utf-8-sig')
