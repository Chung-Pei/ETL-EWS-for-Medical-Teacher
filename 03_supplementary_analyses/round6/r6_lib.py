# -*- coding: utf-8 -*-
"""Paper A Round 6 -- shared library (identifiers ASCII-only).  Post hoc.  No refitting: frozen scores only.

Builds on r5_lib.py (unaltered Round 5 copy).  Contents:
  - b1_risk, b2_risk : copied VERBATIM (ast.get_source_segment) from h02_b1b2.py, sha256 b065b7f8cf8435bc7187d3a98baf85cab0e21fa46edeec640cae7249f6fb7739
  - weighted float64 AUC, student-cluster bootstrap weights for both published protocols
  - Variant 1 relabelling exactly as r5_08_outcome.py ran it
"""
import numpy as np
import pandas as pd

import r5_lib as L

D7 = L.D7
CLS = ['2yr_gen', '2yr_working', 'post', '4yr', 'retake']      # h02_baselines.py CLASSES (= pa_common.CLASS5)
BOOT_B = 2000
SEED_H02 = 20261006     # ews_lib.BOOT_SEED: protocol of the H-02 runs that produced main-text Table 2
SEED_R5 = L.SEED        # 20261031: Round 5 standing protocol (r5_lib.make_weights, factorize order)


# ---- verbatim from h02_b1b2.py (uses CLS and D7 of this module) --------------------------------------------
def b1_risk(df, y_col='y', exclude_same_student=True):
    """B1: per LOSO fold, class failure rate from the training fold (saturated lookup); overall rate if class absent.
    Returns (risk array aligned with df, table of fold-class rates)."""
    risk = np.full(len(df), np.nan)
    sem = df['semester'].to_numpy()
    cls = df['program5_key'].to_numpy()
    y = df[y_col].to_numpy(float)
    ids = df['anon_id'].to_numpy()
    tab = []
    for s in D7:
        te = sem == s
        tr = sem != s
        if exclude_same_student:
            tr = tr & ~np.isin(ids, ids[te])
        overall = y[tr].mean()
        for c in CLS:
            mc = tr & (cls == c)
            n = int(mc.sum())
            rate = float(y[mc].mean()) if n > 0 else overall
            tab.append({'test_semester': s, 'class': c, 'n_train': n, 'fail_train': int(y[mc].sum()) if n else 0,
                        'rate_used': rate, 'fallback_to_fold_overall': int(n == 0), 'fold_overall_rate': overall,
                        'n_test_in_class': int((te & (cls == c)).sum())})
        for c in CLS:
            m = te & (cls == c)
            if m.any():
                risk[m] = [t for t in tab if t['test_semester'] == s and t['class'] == c][0]['rate_used']
    return risk, pd.DataFrame(tab)


def b2_risk(df, y_col='y'):
    """B2: class failure rate over all D7 semesters strictly before the record's semester (expanding window).
    Class absent in the prior window -> prior-window overall rate (G1). 1111 -> NaN (undefined)."""
    risk = np.full(len(df), np.nan)
    sem = df['semester'].to_numpy()
    cls = df['program5_key'].to_numpy()
    y = df[y_col].to_numpy(float)
    tab = []
    for i, s in enumerate(D7):
        if i == 0:
            continue
        prior = np.isin(sem, D7[:i])
        overall = y[prior].mean()
        for c in CLS:
            mc = prior & (cls == c)
            n = int(mc.sum())
            rate = float(y[mc].mean()) if n > 0 else overall
            nt = int(((sem == s) & (cls == c)).sum())
            tab.append({'semester': s, 'class': c, 'n_prior': n, 'fail_prior': int(y[mc].sum()) if n else 0,
                        'rate_used': rate, 'fallback_to_prior_overall': int(n == 0), 'prior_overall_rate': overall,
                        'n_test_in_class': nt})
            if nt:
                risk[(sem == s) & (cls == c)] = rate
    return risk, pd.DataFrame(tab)


# ---------------------------------------------------------------------------------------------------------------
# bootstrap weights
# ---------------------------------------------------------------------------------------------------------------
def boot_weights(base, protocol):
    """(B x N) float64 record weights.
    'H02': ews_lib.Boot semantics: students = np.unique(anon_id) (sorted order), seed 20261006,
           draws = default_rng(seed).integers(0, n_stu, size=(B, n_stu)); record weight = student draw count.
    'R5' : r5_lib.make_weights semantics: students in first-appearance order (base.stu), seed 20261031."""
    if protocol == 'H02':
        stu = np.unique(base.anon_id.to_numpy(), return_inverse=True)[1].reshape(-1)
        seed = SEED_H02
    elif protocol == 'R5':
        stu = base.stu.to_numpy()
        seed = SEED_R5
    else:
        raise ValueError(protocol)
    cnt = L.make_counts(int(stu.max()) + 1, BOOT_B, seed)
    return cnt[:, stu].astype(np.float64)


# ---------------------------------------------------------------------------------------------------------------
# weighted AUC (float64) -- same estimator as r5_lib.auc_weighted; fail = positive class, pass-oriented scores
# ---------------------------------------------------------------------------------------------------------------
def auc_groups(Wm, S, y, groups, chunk=200):
    """Returns (B x G x C) AUC (NaN where a group lacks fail or pass weight) and (B x G) group weight sums."""
    Bn = Wm.shape[0]
    C = S.shape[1]
    out = np.full((Bn, len(groups), C), np.nan)
    nn = np.zeros((Bn, len(groups)))
    for gi, ix in enumerate(groups):
        ix = np.asarray(ix)
        nn[:, gi] = Wm[:, ix].sum(1)
        yy = y[ix]
        F = ix[yy == 1]
        P = ix[yy == 0]
        if len(F) == 0 or len(P) == 0:
            continue
        Mcat = np.concatenate([((S[P, c][None, :] > S[F, c][:, None]) + 0.5 * (S[P, c][None, :] == S[F, c][:, None]))
                               .astype(np.float64) for c in range(C)], axis=1)
        for a in range(0, Bn, chunk):
            b = min(Bn, a + chunk)
            Wf = Wm[a:b][:, F]
            Wp = Wm[a:b][:, P]
            T = (Wf @ Mcat).reshape(b - a, C, len(P))
            num = (T * Wp[:, None, :]).sum(axis=2)
            den = Wf.sum(1) * Wp.sum(1)
            with np.errstate(invalid='ignore', divide='ignore'):
                out[a:b, gi, :] = np.where(den[:, None] > 0, num / den[:, None], np.nan)
    return out, nn


def wmean_auc(a_g, n_g):
    """Within-semester aggregation (h02 _wmean / r5 aggregate_deploy AUC part): group-weight-weighted mean of
    per-semester AUC over semesters with both outcome classes.  a_g (B x G x C), n_g (B x G)."""
    ok = ~np.isnan(a_g)
    w = np.where(ok, n_g[:, :, None], 0.0)
    return (np.where(ok, a_g, 0.0) * w).sum(1) / w.sum(1)


def make_groups(rows, sem_i, basis):
    rows = np.asarray(rows)
    if basis == 'pooled':
        return [rows]
    return [rows[sem_i[rows] == s] for s in range(len(D7)) if (sem_i[rows] == s).any()]


def eval_cols(Wm, S, y, rows, sem_i, basis):
    """Point (C,) and draws (B x C) of AUC for the columns of S on the record subset `rows`.
    basis 'pooled': one AUC over the subset; 'within': record-count-weighted mean of per-semester AUCs."""
    assert not np.isnan(S[rows]).any(), 'NaN score inside evaluation subset'
    groups = make_groups(rows, sem_i, basis)
    ones = np.ones((1, len(y)))
    a1, n1 = auc_groups(ones, S, y, groups)
    ad, nd = auc_groups(Wm, S, y, groups)
    if basis == 'pooled':
        return a1[0, 0, :], ad[:, 0, :]
    return wmean_auc(a1, n1)[0], wmean_auc(ad, nd)


# ---------------------------------------------------------------------------------------------------------------
# scores (frozen; B1/B2 from the ORIGINAL outcome only)
# ---------------------------------------------------------------------------------------------------------------
def build_scores(base, scores):
    """All pass-oriented (higher = safer).  B1 and B2 are computed ONCE from the original y in base['y']."""
    y_before = base['y'].to_numpy().copy()
    S = {}
    for W in ('W8', 'W10'):
        S['blend_m3_' + W] = scores[W].m3.to_numpy(float)                                   # stored column (float noise 1e-16)
        S['blend_pool_' + W] = L.pooled_blend(scores[W].m1.to_numpy(), scores[W].m2.to_numpy())   # pooled rank blend of frozen m1, m2
        S['blend_dep_' + W] = L.deploy_blend(scores[W].m1.to_numpy(), scores[W].m2.to_numpy(),
                                             base.semester.to_numpy())                       # within-semester ranks
    S['B1'] = 1.0 - L.b1_scores(base, class_col='program5_key')
    r1_h02, _ = b1_risk(base)
    assert np.allclose(1.0 - r1_h02, S['B1'], atol=0, rtol=0), 'B1: r5_lib.b1_scores differs from h02 b1_risk'
    r2, _ = b2_risk(base)
    S['B2'] = 1.0 - r2                                                                      # NaN for semester 1111
    for W in (8, 10):
        feat = L.load_step9_features(base, W)
        assert feat.quz_pass_rate.notna().all()
        S['B3_W%d' % W] = feat.quz_pass_rate.to_numpy(float)
    assert (base['y'].to_numpy() == y_before).all(), 'original y was modified'
    return S


def subsets(base):
    sem = base.semester.to_numpy()
    cls = base.program5_key.to_numpy()
    allr = np.arange(len(base))
    in_b2 = sem != '1111'
    clean = in_b2 & ~((sem == '1112') & (cls == '4yr')) & ~((sem == '1141') & (cls == 'retake'))
    assert int(in_b2.sum()) == 2180 and int(clean.sum()) == 1907
    return {'all': allr, 'main': np.flatnonzero(in_b2), 'clean': np.flatnonzero(clean)}


# ---------------------------------------------------------------------------------------------------------------
# Variant 1 outcome: exactly r5_08_outcome.py (primary relabel)
# ---------------------------------------------------------------------------------------------------------------
def variant1_labels(base):
    """Return dict with y0, y1, add mask, composite and the R5-8 gate counts.  Gate 2 asserts live in the caller."""
    comp = ((base.midterm_score + base.final_score) / 2).to_numpy(float)
    rec_score = base.semester_score.to_numpy(float)
    add = (rec_score - comp) > 0.05
    cnt = base[add].semester.value_counts().sort_index().to_dict()
    n56 = int((add & (rec_score >= 60) & (comp < 60)).sum())
    y0 = base.y.to_numpy().astype(int)
    y1 = y0.copy()
    y1[add] = (comp[add] < 60).astype(int)
    return {'y0': y0, 'y1': y1, 'add': add, 'comp': comp, 'cnt': cnt, 'n56': n56,
            'to_pass': int(((y0 == 1) & (y1 == 0)).sum()), 'to_fail': int(((y0 == 0) & (y1 == 1)).sum())}
