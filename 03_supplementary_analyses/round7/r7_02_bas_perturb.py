# -*- coding: utf-8 -*-
"""R7-2: BAS weight perturbation.  BAS rebuilt through the pack's own pipeline (ews_lib.bas_fold via run_folds) with perturbed weights;
XGBoost = frozen m2 (never retrained); blend recomposed on both percentile bases; Appendix B rule with the standing bootstrap
(student cluster, B = 2000, seed 20261031, paired)."""
import r7_retrain as R
import r7_common as C
import r7_rows as RW
import itertools
import json
import sys
import time
import numpy as np
import pandas as pd

L = C.L
t0 = time.time()
EL, EE, etl, base, scores, data = R.setup()
N = len(base)
y = base.y.to_numpy()
f = base.final.to_numpy(float)
Wm = L.make_weights(base.stu.to_numpy())            # standing stream: seed 20261031, B = 2000, factorize student order
assert L.SEED == 20261031 and L.BOOT_B == 2000
ORIG_W0 = dict(etl.BAS_W)                            # {'midterm': .35, 'qmi': .30, 'neg_passive': .20, 'log_attempts': .15}
KEYS = ['qmi', 'neg_passive', 'log_attempts']
W0 = np.array([ORIG_W0[k] for k in KEYS]) / 0.65     # exact pipeline weights (0.4615, 0.3077, 0.2308; rounded 0.462, 0.308, 0.231)

# ---- variants ------------------------------------------------------------------------------------------
variants = [('V00 identity', (1.0, 1.0, 1.0), 'identity')]
names = ['quiz-mastery', 'passive-reversed', 'log-attempts']
k = 1
for i in range(3):
    for fac in (0.5, 0.75, 1.25, 1.5):
        m = [1.0, 1.0, 1.0]
        m[i] = fac
        variants.append((f'V{k:02d} single {names[i]} x{fac}', tuple(m), 'single'))
        k += 1
for combo in itertools.product((0.5, 1.5), repeat=3):
    variants.append((f'V{k:02d} corner ({combo[0]},{combo[1]},{combo[2]})', combo, 'corner'))
    k += 1
assert len(variants) == 21


def weights_of(mult):
    w = W0 * np.array(mult)
    return w / w.sum()


def rebuild_bas(wvec, identity):
    """Rebuild the frozen-protocol BAS (LOSO, retaker rule, training-fold standardization) for W6/8/10/12 with weight vector wvec."""
    out = {}
    if identity:
        EL.BAS_W0 = dict(ORIG_W0)
    else:
        EL.BAS_W0 = {'midterm': ORIG_W0['midterm'], 'qmi': wvec[0] * 0.65, 'neg_passive': wvec[1] * 0.65, 'log_attempts': wvec[2] * 0.65}
    for W in C.W_ALL:
        df = EE.canon(EE.run_folds(etl, data[W], EE.fold_defs('P'), etl.FS_C, False, {'m1': 1}))
        assert list(zip(df.semester, df.anon_id)) == list(zip(base.semester, base.anon_id))
        out[f'W{W}'] = df.m1.to_numpy(float)
    EL.BAS_W0 = dict(ORIG_W0)
    return out


def effective_w(wvec):
    return {k: float(v) for k, v in zip(KEYS, wvec)}


NAME = {'M1': 'BAS', 'M3': 'blend'}
rows, raw = [], []
sc_frozen_m2 = {W: scores[W].m2.to_numpy(float) for W in ['W6', 'W8', 'W10', 'W12']}
for vi, (label, mult, kind) in enumerate(variants):
    wvec = weights_of(mult)
    m1 = rebuild_bas(wvec, identity=(vi == 0))
    rec = {'variant': label, 'kind': kind, 'mult': str(mult), 'w_qmi': wvec[0], 'w_passive': wvec[1], 'w_log_att': wvec[2]}
    for definition in ('pooled', 'deploy'):
        sc = {}
        for W in ['W6', 'W8', 'W10', 'W12']:
            sc[W] = pd.DataFrame({'m1': m1[W], 'm2': sc_frozen_m2[W], 'm3': L.pooled_blend(m1[W], sc_frozen_m2[W]),
                                  'pfail': 1 - sc_frozen_m2[W], 'qmi_none': 0.0})
        res = L.run_rule(base, sc, y, f, definition, Wm, preds=['M1', 'M3'])
        sel = res['sel']
        colx = {n: i for i, n in enumerate(res['names'])}
        for p in ('M1', 'M3'):
            tag = f"{definition}_{NAME[p]}"
            rec[f'{tag}_Wstar'] = sel['wstar'][p]
            d8 = [d for d in sel['detail'] if d['pred'] == p and d['week'] == 'W8'][0]
            rec[f'{tag}_W8_auc_loss_ucl'] = d8['auc_loss_ucl']
            rec[f'{tag}_W8_r_loss_ucl'] = d8['r_loss_ucl']
            for W in ('W8', 'W10', 'W12'):
                rec[f'{tag}_auc_{W}'] = res['auc_pt'][(W, p)]
            rec[f'{tag}_W8_auc_loss_pt'] = res['auc_pt'][('W12', p)] - res['auc_pt'][('W8', p)]
        rec[f'{definition}_selected_policy'] = f"{sel['sel_week']}+{NAME.get(sel['sel_pred'], sel['sel_pred'])}"
    # rebuild vs frozen BAS (informational, per week)
    for W in ('W8', 'W10', 'W12'):
        fro = scores[W].m1.to_numpy(float)
        rec[f'bas_rebuilt_minus_frozen_auc_{W}'] = L.auc_fail(y, m1[W]) - L.auc_fail(y, fro)
        rec[f'bas_max_abs_diff_vs_frozen_{W}'] = float(np.abs(m1[W] - fro).max())
        rec[f'bas_n_diff_gt_1e-9_{W}'] = int((np.abs(m1[W] - fro) > 1e-9).sum())
    raw.append(rec)
    print(f"[{time.time() - t0:5.0f}s] {label:42s} pooled: BAS {rec['pooled_BAS_Wstar']} blend {rec['pooled_blend_Wstar']} | deploy: BAS {rec['deploy_BAS_Wstar']} blend {rec['deploy_blend_Wstar']} | "
          f"blend W8 AUC pooled {rec['pooled_blend_auc_W8']:.4f} deploy {rec['deploy_blend_auc_W8']:.4f}", flush=True)
    pd.DataFrame(raw).to_csv(C.RES / 'R7-2_variants.csv', index=False, encoding='utf-8-sig')

D = pd.DataFrame(raw)
D.to_csv(C.RES / 'R7-2_variants.csv', index=False, encoding='utf-8-sig')
print(f'done {time.time() - t0:.0f}s')
