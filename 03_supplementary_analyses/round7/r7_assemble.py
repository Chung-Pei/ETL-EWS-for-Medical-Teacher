# -*- coding: utf-8 -*-
"""Assemble R7_results.csv (long format) from the per-item parts, add gate rows and R7-2 summary rows."""
import r7_retrain as R
import r7_common as C
import r7_rows as RW
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

L = C.L
res = C.RES
rows = []

# ---------------- gates ---------------------------------------------------------------------------------------------
ga = json.load(open(res / 'gate_G_a_variants.json'))['same_resamples|blend_recomputed_in_subset=False']
for sub, anchor in (('pooled', 0.014), ('E2', 0.032), ('E1', 0.021)):
    v = ga[sub]['M3']['auc_loss_ucl']
    ok = abs(v - anchor) <= 0.003
    gate = 'Gate G-a' if sub != 'E1' else 'Gate G-a (informational)'
    RW.add(rows, gate, f'Tier A machine, {sub}', 'blend W8 AUC-loss upper limit (97.5th pct of W12 - W8)', 'W8', ga[sub]['M3']['n'], v,
           note=f'published {anchor} (Supplementary Table 1), tolerance +-0.003: {"PASS" if ok else "FAIL"}; protocol = documented Tier A (frozen scores, same resamples in every era, seed 20261006)')
gb = pd.read_csv(res / 'gate_G_b_E2_points.csv', encoding='utf-8-sig')
g = gb[gb.model == "M3' blend"].set_index('W')
for nm, key, anchor in (('AUC', 'auc', 0.051), ('r', 'r', 0.095)):
    v = float(g.loc['W12', key] - g.loc['W8', key])
    RW.add(rows, 'Gate G-b', 'Tier B Era E2 two folds (train 1141->test 1132; train 1132->test 1141), R7 pipeline', f'blend W8 point loss, {nm} (W12 - W8)', 'W8', 745, v,
           note=f'published +{anchor}, tolerance +-0.003: {"PASS" if abs(v - anchor) <= 0.003 else "FAIL"}')
gc = pd.read_csv(res / 'gate_G_c.csv', encoding='utf-8-sig').set_index('score')
for nm, anchor, gate_tol in (('BAS', 0.0008, True), ('blend', -0.0020, True), ('XGBoost', -0.0043, False)):
    v = float(gc.loc[nm, 'AUC_diff'])
    RW.add(rows, 'Gate G-c', 'pooled LOSO rebuild (XGBoost 3.4.1), W8', f'{nm} AUC rebuilt - frozen', 'W8', 2652, v,
           note=f'documented {anchor:+.4f}' + (f', tolerance +-0.003: {"PASS" if abs(v - anchor) <= 0.003 else "FAIL"}' if gate_tol else '; informational (documented build offset)'))
RW.write('gates_GaGbGc', rows)
rows = []

# ---------------- R7-2 ---------------------------------------------------------------------------------------------
D = pd.read_csv(res / 'R7-2_variants.csv', encoding='utf-8-sig')
D['distinct'] = ~D.variant.isin(['V13 corner (0.5,0.5,0.5)', 'V20 corner (1.5,1.5,1.5)'])
base, scores = C.load_base()
y, f = base.y.to_numpy(), base.final.to_numpy(float)
groups = L.sem_groups_of(base.sem_i)
one = np.ones((1, len(y)), np.float32)


def frozen_aucs():
    out = {}
    for definition in ('pooled', 'deploy'):
        names, S = L.build_candidates(base, scores, definition, weeks=['W8', 'W10', 'W12'])
        a, _ = L.metrics_draws(one, S, y, f, groups, definition)
        for i, (W, p) in enumerate(names):
            out[(definition, p, W)] = float(a[0, i])
    return out


FR = frozen_aucs()
ident = D.iloc[0]
Q = [('pooled', 'BAS'), ('pooled', 'blend'), ('deploy', 'BAS'), ('deploy', 'blend')]
for _, r in D.iterrows():
    lab = r.variant
    note = f"weights (quiz-mastery, passive-reversed, log-attempts) = ({r.w_qmi:.4f}, {r.w_passive:.4f}, {r.w_log_att:.4f}); multipliers {r.mult}" + ('' if r.distinct else '; numerically identical to V00 identity')
    for W in ('W8', 'W10', 'W12'):
        RW.add(rows, 'R7-2 variant', lab, 'BAS pooled AUC', W, 2652, r[f'pooled_BAS_auc_{W}'], note=note)
    for W in ('W8', 'W10', 'W12'):
        RW.add(rows, 'R7-2 variant', lab, 'blend pooled AUC', W, 2652, r[f'pooled_blend_auc_{W}'], note=note)
    for W in ('W8', 'W10', 'W12'):
        RW.add(rows, 'R7-2 variant', lab, 'blend deployment AUC', W, 2652, r[f'deploy_blend_auc_{W}'], note=note)
    for d_, p_ in Q:
        RW.add(rows, 'R7-2 variant', lab, f'{p_} selected week ({d_} definition)', r[f'{d_}_{p_}_Wstar'], 2652, note=note)
        RW.add(rows, 'R7-2 variant', lab, f'{p_} W8 AUC-loss upper limit ({d_} definition)', 'W8', 2652, r[f'{d_}_{p_}_W8_auc_loss_ucl'], note='97.5th percentile of W12 - W8, seed 20261031, B=2000')

# summary counts
for d_, p_ in Q:
    col = f'{d_}_{p_}_Wstar'
    for scope, sub in (('21 variants as specified', D), ('19 distinct weight vectors', D[D.distinct])):
        vc = sub[col].value_counts()
        for W in ('W6', 'W8', 'W10', 'W12'):
            RW.add(rows, 'R7-2 summary', scope, f'{p_} selected week = {W} ({d_} definition): number of variants', W, len(sub), int(vc.get(W, 0)),
                   note=f'identity selects {ident[col]}')
    chg = D[(D[col] != ident[col]) & D.distinct].variant.tolist()
    RW.add(rows, 'R7-2 summary', 'vs identity', f'{p_} ({d_}): variants whose selected week differs from identity', ident[col], 19, len(chg), note='; '.join(chg) if chg else 'none')
# blend W8 AUC band
for d_ in ('pooled', 'deploy'):
    for p_ in ('blend', 'BAS'):
        c = f'{d_}_{p_}_auc_W8'
        sub = D[D.distinct]
        imin, imax = sub[c].idxmin(), sub[c].idxmax()
        RW.add(rows, 'R7-2 summary', '19 distinct weight vectors', f'{p_} W8 AUC ({d_}): identity [min, max] across variants', 'W8', 19, float(ident[c]), float(sub[c].min()), float(sub[c].max()),
               f"min = {D.variant[imin]}; max = {D.variant[imax]}")
        RW.add(rows, 'R7-2 summary', '19 distinct weight vectors', f'{p_} W8 AUC ({d_}): difference from identity [min, max]', 'W8', 19, 0.0, float(sub[c].min() - ident[c]), float(sub[c].max() - ident[c]), 'paired to the rebuilt identity variant')
        rk = int((sub[c] > ident[c] + 1e-12).sum())
        RW.add(rows, 'R7-2 summary', '19 distinct weight vectors', f'{p_} W8 AUC ({d_}): number of the other 18 distinct variants with higher AUC than identity', 'W8', 18, rk)
# identity vs frozen
for W in ('W8', 'W10', 'W12'):
    RW.add(rows, 'R7-2 identity vs frozen', 'V00 identity (rebuilt BAS + frozen XGBoost) minus frozen', 'BAS pooled AUC', W, 2652, float(ident[f'pooled_BAS_auc_{W}'] - FR[('pooled', 'M1', W)]),
           note='rebuilt BAS differs from the frozen BAS (build constant); read variants against V00, not against frozen')
    RW.add(rows, 'R7-2 identity vs frozen', 'V00 identity minus frozen', 'blend pooled AUC', W, 2652, float(ident[f'pooled_blend_auc_{W}'] - FR[('pooled', 'M3', W)]))
    RW.add(rows, 'R7-2 identity vs frozen', 'V00 identity minus frozen', 'blend deployment AUC', W, 2652, float(ident[f'deploy_blend_auc_{W}'] - FR[('deploy', 'M3', W)]))
EL, EE, etl, base2, scores2, data = R.setup([8, 10, 12])
for W in (8, 10, 12):
    df = EE.canon(EE.run_folds(etl, data[W], EE.fold_defs('P'), etl.FS_C, False, {'m1': 1}))
    rho = float(spearmanr(df.m1.to_numpy(), scores[f'W{W}'].m1.to_numpy())[0])
    RW.add(rows, 'R7-2 identity vs frozen', 'V00 identity BAS vs frozen m1', 'Spearman correlation of the two score vectors', f'W{W}', 2652, rho, note='rank agreement of the rebuilt and the frozen BAS')
RW.write('R7-2', rows)

# ---------------- concatenate ----------------------------------------------------------------------------------------
order = ['gates_GaGbGc', 'R7-4', 'R7-1', 'R7-2', 'R7-3', 'R7-5']
parts = [pd.read_csv(res / 'parts' / f'{n}.csv', encoding='utf-8-sig') for n in order]
# R7-4 part already holds the G-d rows
ALL = pd.concat(parts, ignore_index=True)
ALL.to_csv(res / 'R7_results.csv', index=False, encoding='utf-8-sig')
print(ALL.groupby('item').size().to_string())
print('rows total', len(ALL))
