# -*- coding: utf-8 -*-
"""R5-4 -- admit B3 (step9 quz_pass_rate) as a fourth selection candidate; Appendix B rule under both definitions."""
import numpy as np
import pandas as pd

import r5_lib as L

base, scores = L.load_base()
N = len(base)
y = base.y.to_numpy().astype(int)
f = base.final.to_numpy(float)
Wm = L.make_weights(base.stu.to_numpy())
b3 = {}
for W in WEEKS if (WEEKS := L.WEEKS) else []:
    feat = L.load_step9_features(base, L.WNUM[W])
    assert feat.quz_pass_rate.notna().all()
    b3[W] = feat.quz_pass_rate.to_numpy(float)          # pass-oriented (higher = safer); raw feature, not a fitted score
extra = {(W, 'B3'): b3[W] for W in L.WEEKS}
rows, detail_rows = [], []
for defn in ('pooled', 'deploy'):
    for cand_set, sel_preds, ex in (('M1,M2,M3 (published set)', L.PRED, None),
                                    ('M1,M2,M3,B3 (B3 admitted)', L.PRED + ['B3'], extra)):
        res = L.run_rule(base, scores, y, f, defn, Wm, extra=ex, sel_preds=sel_preds)
        s = res['sel']
        for p in sel_preds:
            rows.append({'candidate_set': cand_set, 'definition': defn, 'candidate': p if p != 'M1' else "M1' (BAS)",
                         'W_star': s['wstar'][p] or 'none'})
        rows.append({'candidate_set': cand_set, 'definition': defn, 'candidate': 'SELECTED POLICY',
                     'W_star': f"{s['sel_week']} + {s['sel_pred']}"})
        col = {n: i for i, n in enumerate(res['names'])}
        for d in s['detail']:
            detail_rows.append({'candidate_set': cand_set, 'definition': defn, 'candidate': d['pred'], 'week': d['week'],
                                'auc_loss_ucl_975': d['auc_loss_ucl'], 'r_loss_ucl_975': d['r_loss_ucl'],
                                'qualifies': d['qualifies']})
        for c in s['compare']:
            rows.append({'candidate_set': cand_set, 'definition': defn, 'candidate': f"third-rule {c['a']} minus {c['b']} at "
                         f"{c['week']}", 'W_star': f"{c['delta_point']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]; entirely within "
                         f"+-0.02 band: {c['entirely_within_band']}"})
        if ex:
            for W in L.WEEKS:
                k = (W, 'B3')
                lo, hi = L.ci(res['auc_d'][:, col[k]])
                rows.append({'candidate_set': cand_set, 'definition': defn, 'candidate': f'B3 AUC at {W}',
                             'W_star': f"{res['auc_pt'][k]:.4f} [{lo:.4f}, {hi:.4f}]"})
                lo, hi = L.ci(res['r_d'][:, col[k]])
                rows.append({'candidate_set': cand_set, 'definition': defn, 'candidate': f'B3 r at {W}',
                             'W_star': f"{res['r_pt'][k]:.4f} [{lo:.4f}, {hi:.4f}]"})
        print(defn, cand_set, s['wstar'], s['sel_week'], s['sel_pred'])
        for c in s['compare']:
            print('   ', c)
T = pd.DataFrame(rows)
T.to_csv(L.RES / 'R5-4.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(detail_rows).to_csv(L.RES / 'R5-4_loss_limits.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 200, 'display.max_colwidth', 100)
print(T.to_string())
