# -*- coding: utf-8 -*-
"""Gate R7-G, items G-a (Tier A machine reproduction) and G-d (R7-4 sanity of the refit_pooled file).
Tier A machine (documented): frozen out-of-fold scores (arm P, feature set C, pooled blend m3), student-cluster bootstrap,
B = 2000, seed 20261006, the same resamples in every era, upper limit = 97.5th percentile of the loss draws (W12 - W8)."""
import r7_common as C
import json
import numpy as np
import pandas as pd

L = C.L
base, scores = C.load_base()
y = base.y.to_numpy()
f = base.final.to_numpy(float)
N = len(base)
E1 = base.semester.isin(['1111', '1112', '1121', '1122', '1131']).to_numpy()
E2 = base.semester.isin(['1132', '1141']).to_numpy()
rows = []


def tierA(mask, Wm_all, recompute_blend=False, label=''):
    """Return dict of W8/W12 point + loss + upper limits for blend/BAS/XGB on the record subset `mask`."""
    ix = np.flatnonzero(mask)
    out = {}
    for pname, col in (('M3', 'm3'), ('M1', 'm1'), ('M2', 'm2')):
        cols = []
        for W in ('W8', 'W12'):
            sc = scores[W]
            v = sc[col].to_numpy(float)[ix]
            if pname == 'M3' and recompute_blend:
                v = L.pooled_blend(sc.m1.to_numpy()[ix], sc.m2.to_numpy()[ix])
            cols.append(v)
        S = np.column_stack(cols)                       # [W8, W12]
        Wm = Wm_all[:, ix]
        a_d, _ = L.auc_weighted(Wm, S, y[ix], [np.arange(len(ix))])
        r_d, _ = L.corr_weighted(Wm, S, f[ix], [np.arange(len(ix))])
        a_d, r_d = a_d[:, 0, :], r_d[:, 0, :]
        one = np.ones((1, len(ix)), np.float32)
        a_p, _ = L.auc_weighted(one, S, y[ix], [np.arange(len(ix))])
        r_p, _ = L.corr_weighted(one, S, f[ix], [np.arange(len(ix))])
        a_p, r_p = a_p[0, 0, :], r_p[0, 0, :]
        la = a_d[:, 1] - a_d[:, 0]
        lr = r_d[:, 1] - r_d[:, 0]
        out[pname] = {'n': len(ix), 'n_fail': int(y[ix].sum()), 'auc_w8': a_p[0], 'r_w8': r_p[0],
                      'auc_loss_pt': a_p[1] - a_p[0], 'r_loss_pt': r_p[1] - r_p[0],
                      'auc_loss_lo': float(np.percentile(la, 2.5)), 'auc_loss_ucl': float(np.percentile(la, 97.5)),
                      'r_loss_lo': float(np.percentile(lr, 2.5)), 'r_loss_ucl': float(np.percentile(lr, 97.5)),
                      'auc_loss_width': float(np.percentile(la, 97.5) - np.percentile(la, 2.5))}
    return out


# ---- protocol variants -------------------------------------------------------------------------------
Wm_same = C.counts_sorted_students(base.anon_id, 2000, C.SEED_TIERA)        # same resamples in every era (documented)
variants = {}
for rb in (False, True):
    variants[f'same_resamples|blend_recomputed_in_subset={rb}'] = {
        'pooled': tierA(np.ones(N, bool), Wm_same, rb), 'E1': tierA(E1, Wm_same, rb), 'E2': tierA(E2, Wm_same, rb)}
# alternative: bootstrap drawn separately inside each subset with the same seed (ews_lib.Boot on subset ids)
alt = {}
for nm, m in (('pooled', np.ones(N, bool)), ('E1', E1), ('E2', E2)):
    Wsub = C.counts_sorted_students(base.anon_id[m], 2000, C.SEED_TIERA)
    full = np.zeros((2000, N), np.float32)
    full[:, np.flatnonzero(m)] = Wsub
    alt[nm] = tierA(m, full, False)
variants['separate_draws_per_subset|blend_frozen'] = alt

pub = {'pooled': dict(auc_loss_pt=-0.001, auc_ucl=0.014, r_loss_pt=0.002, r_ucl=0.020),
       'E1': dict(auc_loss_pt=0.006, auc_ucl=0.021, r_loss_pt=0.014, r_ucl=0.032),
       'E2': dict(auc_loss_pt=0.006, auc_ucl=0.032, r_loss_pt=0.016, r_ucl=0.054)}
pubpt = {'E1': (0.748, 0.475), 'E2': (0.714, 0.435)}
print('Published (Supp. Table 1/2): blend W8 AUC-loss [pt, upper]; r-loss [pt, upper]; W8 AUC, r')
for vn, v in variants.items():
    print('\n==', vn)
    for sub in ('pooled', 'E1', 'E2'):
        o = v[sub]['M3']
        p = pub[sub]
        extra = f"  W8 AUC {o['auc_w8']:.3f} r {o['r_w8']:.3f}"
        print(f"  {sub:6s} n={o['n']} fail={o['n_fail']}  AUC-loss pt {o['auc_loss_pt']:+.4f} ucl {o['auc_loss_ucl']:.4f}"
              f" (pub {p['auc_loss_pt']:+.3f}/{p['auc_ucl']:.3f})  r-loss pt {o['r_loss_pt']:+.4f} ucl {o['r_loss_ucl']:.4f}"
              f" (pub {p['r_loss_pt']:+.3f}/{p['r_ucl']:.3f}){extra}")
json.dump({k: v for k, v in variants.items()}, open(C.RES / 'gate_G_a_variants.json', 'w'), indent=1, default=float)

# ---- G-d -------------------------------------------------------------------------------------------------
rp = pd.read_csv(C.STAGING / '06_r5_9_per_resample/R5-9_per_resample_refit_pooled.csv', encoding='utf-8-sig')
n = len(rp)
vc = rp.groupby(['sel_week', 'sel_pred']).size().sort_values(ascending=False)
print('\nG-d: rows =', n)
print((vc / n * 100).round(1).to_string())
