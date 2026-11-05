# -*- coding: utf-8 -*-
"""Tier A machine (documented): frozen out-of-fold scores, pooled-percentile blend m3, student-cluster bootstrap
(B = 2000, seed 20261006, np.unique-sorted student order, the same resamples in every subset), rule = Appendix B."""
import r7_common as C
import numpy as np

L = C.L


def full_weights(base, B=2000, seed=C.SEED_TIERA):
    return C.counts_sorted_students(base.anon_id, B, seed)


def select_on_subset(base, scores, mask, Wm_full):
    """Appendix B on the records in `mask`; returns r5_lib.run_rule() result (selection, point metrics, draws)."""
    ix = np.flatnonzero(mask)
    b = base.iloc[ix].reset_index(drop=True)
    sc = {W: scores[W].iloc[ix].reset_index(drop=True) for W in scores}
    return L.run_rule(b, sc, b.y.to_numpy(), b.final.to_numpy(float), 'pooled', Wm_full[:, ix])


def blend_w8_loss(res, pred='M3'):
    """W8 loss (W12 - W8; positive = loss) for `pred`: point, 2.5th and 97.5th percentiles, for AUC and r."""
    col = {n: i for i, n in enumerate(res['names'])}
    i8, i12 = col[('W8', pred)], col[('W12', pred)]
    la = res['auc_d'][:, i12] - res['auc_d'][:, i8]
    lr = res['r_d'][:, i12] - res['r_d'][:, i8]
    return {'auc_w8': res['auc_pt'][('W8', pred)], 'auc_w12': res['auc_pt'][('W12', pred)],
            'r_w8': res['r_pt'][('W8', pred)], 'r_w12': res['r_pt'][('W12', pred)],
            'auc_loss_pt': res['auc_pt'][('W12', pred)] - res['auc_pt'][('W8', pred)],
            'r_loss_pt': res['r_pt'][('W12', pred)] - res['r_pt'][('W8', pred)],
            'auc_loss_lo': float(np.nanpercentile(la, 2.5)), 'auc_loss_hi': float(np.nanpercentile(la, 97.5)),
            'r_loss_lo': float(np.nanpercentile(lr, 2.5)), 'r_loss_hi': float(np.nanpercentile(lr, 97.5)),
            'n_nan_draws': int(np.isnan(la).sum())}
