# -*- coding: utf-8 -*-
"""R5-12 -- leave-one-semester-out influence: exclusion of Semester 1132 (frozen scores; ranks recomputed within the
retained set; training folds unchanged)."""
import numpy as np
import pandas as pd

import r5_lib as L

base, scores = L.load_base()
y_all = base.y.to_numpy().astype(int)
drop = (base.semester == '1132').to_numpy()
assert drop.sum() == 236 and y_all[drop].sum() == 121                      # validation anchors
assert (~drop).sum() == 2416 and y_all[~drop].sum() == 439
b, sc = L.subset_view(base, scores, ~drop)
y = b.y.to_numpy().astype(int)
f = b.final.to_numpy(float)
Wm = L.make_weights(b.stu.to_numpy())
rows = [{'variant': 'anchors', 'definition': '', 'row_type': 'counts', 'predictor': '', 'week': '',
         'estimate': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan,
         'note': f'dropped 1132: n=236, failing=121 (51.3%); retained n={len(b)}, failing={int(y.sum())}; '
                 f'students in retained set={b.stu.nunique()}'}]
for defn, bw in (('pooled', 'W8'), ('deploy', 'W10')):
    res = L.run_rule(b, sc, y, f, defn, Wm, recompute_pooled_m3=True)
    rows += L.rule_rows('exclude_1132', defn, res, bw, 'pooled percentile ranks recomputed over the retained set'
                        if defn == 'pooled' else 'within-semester ranks unchanged; aggregate over 6 retained semesters')
    s = res['sel']
    print(defn, 'W*', s['wstar'], 'selected', s['sel_week'], s['sel_pred'])
    col = {n: i for i, n in enumerate(res['names'])}
    k = (bw, 'M3')
    print('   blend', bw, 'AUC', round(res['auc_pt'][k], 4), L.ci(res['auc_d'][:, col[k]]), 'r', round(res['r_pt'][k], 4),
          L.ci(res['r_d'][:, col[k]]))
    for c in s['compare']:
        print('   third rule', c)
R = pd.DataFrame(rows)
R.to_csv(L.RES / 'R5-12.csv', index=False, encoding='utf-8-sig')
