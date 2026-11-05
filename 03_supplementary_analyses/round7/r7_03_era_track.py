# -*- coding: utf-8 -*-
"""R7-3: Tier A re-selection inside the track-pure cells (frozen scores, no retraining).
Tracks from class_membership.csv class_track (see ledger Ruling): F = '4yr'; S = '2yr' + 'postbac' (retake excluded)."""
import r7_common as C
import r7_tiera as TA
import r7_rows as RW
import json
import numpy as np
import pandas as pd

L = C.L
base, scores = C.load_base()
Wm = TA.full_weights(base)                      # Tier A: same resamples in every subset
sem = base.semester
ct = base.class_track
E1m = sem.isin(['1111', '1112', '1121', '1122', '1131']).to_numpy()
E2m = sem.isin(['1132', '1141']).to_numpy()
cells = {
    'standard|pooled': np.ones(len(base), bool),
    'standard|E1': E1m, 'standard|E2': E2m,
    'F (4yr)|E1': (E1m & (ct == '4yr')).to_numpy(),
    'F (4yr)|E2': (E2m & (ct == '4yr')).to_numpy(),
    'S (2yr+postbac)|E1': (E1m & ct.isin(['2yr', 'postbac'])).to_numpy(),
    'S (2yr+postbac)|E2': (E2m & ct.isin(['2yr', 'postbac'])).to_numpy(),
}
# composition check against the semester list given in the spec
comp = {k: sorted(sem[m].unique().tolist()) for k, m in cells.items()}
print('composition (semesters):', json.dumps(comp))
assert comp['F (4yr)|E1'] == ['1112', '1122'] and comp['F (4yr)|E2'] == ['1132']
assert comp['S (2yr+postbac)|E1'] == ['1111', '1121', '1131'] and comp['S (2yr+postbac)|E2'] == ['1141']
assert int((E2m & ct.isin(['2yr', 'postbac'])).sum()) == 509 - 52

NAME = {'M1': 'BAS', 'M2': 'XGBoost', 'M3': 'blend'}
rows, raw = [], []
for cell, m in cells.items():
    res = TA.select_on_subset(base, scores, m, Wm)
    sel = res['sel']
    ls = TA.blend_w8_loss(res)
    n, nf = int(m.sum()), int(base.y[m].sum())
    in_a = ls['auc_loss_pt'] <= 0.02
    in_r = ls['r_loss_pt'] <= 0.03
    ucl_a = ls['auc_loss_hi'] > 0.02
    ucl_r = ls['r_loss_hi'] > 0.03
    ws = {NAME[p]: sel['wstar'][p] for p in ('M1', 'M2', 'M3')}
    # W10 feasibility of the blend (upper limits against the bounds)
    d10 = [d for d in sel['detail'] if d['pred'] == 'M3' and d['week'] == 'W10'][0]
    d8 = [d for d in sel['detail'] if d['pred'] == 'M3' and d['week'] == 'W8'][0]
    raw.append({'cell': cell, 'semesters': '+'.join(sorted(sem[m].unique())), 'n': n, 'n_fail': nf, 'fail_rate_pct': 100 * nf / n,
                'blend_W8_AUC': ls['auc_w8'], 'blend_W12_AUC': ls['auc_w12'], 'blend_W8_r': ls['r_w8'], 'blend_W12_r': ls['r_w12'],
                'auc_loss_pt': ls['auc_loss_pt'], 'auc_loss_lo': ls['auc_loss_lo'], 'auc_loss_ucl': ls['auc_loss_hi'],
                'r_loss_pt': ls['r_loss_pt'], 'r_loss_lo': ls['r_loss_lo'], 'r_loss_ucl': ls['r_loss_hi'],
                'pt_loss_within_bounds_auc': bool(in_a), 'pt_loss_within_bounds_r': bool(in_r),
                'ucl_beyond_bound_auc': bool(ucl_a), 'ucl_beyond_bound_r': bool(ucl_r),
                'W_star_BAS': ws['BAS'], 'W_star_XGBoost': ws['XGBoost'], 'W_star_blend': ws['blend'],
                'selected_policy': f"{sel['sel_week']}+{NAME.get(sel['sel_pred'], sel['sel_pred'])}",
                'blend_W10_auc_loss_ucl': d10['auc_loss_ucl'], 'blend_W10_r_loss_ucl': d10['r_loss_ucl'], 'blend_W10_qualifies': d10['qualifies'],
                'blend_W8_r_loss_ucl_rule': d8['r_loss_ucl'], 'n_nan_draws': ls['n_nan_draws']})
    note = (f"semesters {'+'.join(sorted(sem[m].unique()))}; n_fail {nf}; W* BAS {ws['BAS']}, XGBoost {ws['XGBoost']}, blend {ws['blend']}; "
            f"selected {sel['sel_week']}+{NAME.get(sel['sel_pred'], sel['sel_pred'])}")
    RW.add(rows, 'R7-3', cell, 'blend W8 AUC loss (W12 - W8): point [95% CI]', 'W8', n, ls['auc_loss_pt'], ls['auc_loss_lo'], ls['auc_loss_hi'], note)
    RW.add(rows, 'R7-3', cell, 'blend W8 r loss (W12 - W8): point [95% CI]', 'W8', n, ls['r_loss_pt'], ls['r_loss_lo'], ls['r_loss_hi'], note)
    for p in ('M1', 'M2', 'M3'):
        RW.add(rows, 'R7-3', cell, f're-selected week, {NAME[p]}', sel['wstar'][p] or 'none', n, note=f'n_fail {nf}')
    RW.add(rows, 'R7-3', cell, 'blend W10 AUC-loss upper limit (rule: <= 0.020)', 'W10', n, d10['auc_loss_ucl'], note=f"r-loss upper {d10['r_loss_ucl']:.4f} (<= 0.030); qualifies {d10['qualifies']}")
R = pd.DataFrame(raw)
pd.set_option('display.width', 250, 'display.max_columns', 40)
print(R[['cell', 'semesters', 'n', 'n_fail', 'fail_rate_pct', 'auc_loss_pt', 'auc_loss_ucl', 'r_loss_pt', 'r_loss_ucl', 'W_star_BAS', 'W_star_XGBoost', 'W_star_blend', 'selected_policy']].round(4).to_string(index=False))
print()
print(R[['cell', 'pt_loss_within_bounds_auc', 'pt_loss_within_bounds_r', 'ucl_beyond_bound_auc', 'ucl_beyond_bound_r', 'blend_W10_auc_loss_ucl', 'blend_W10_r_loss_ucl', 'blend_W10_qualifies', 'n_nan_draws']].round(4).to_string(index=False))
R.to_csv(C.RES / 'R7-3_cells.csv', index=False, encoding='utf-8-sig')
RW.write('R7-3', rows)
