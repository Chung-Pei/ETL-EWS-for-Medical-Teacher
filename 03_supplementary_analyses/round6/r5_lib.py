# -*- coding: utf-8 -*-
"""Paper A Round 5 -- shared library (identifiers ASCII-only).

Frozen-score analyses only: no model is retrained here.
Definitions follow 00_README_Round5_spec.md (Appendix B selection rule; standing bootstrap).
"""
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
PACK = Path(os.environ.get('R5_PACK', HERE.parent / 'materials' / 'handoff_pack'))
RES = Path(os.environ.get('R5_RES', HERE.parent / 'results'))
RES.mkdir(parents=True, exist_ok=True)

SEED = 20261031          # standing bootstrap seed of this round (spec: report it)
BOOT_B = 2000
D7 = ['1111', '1112', '1121', '1122', '1131', '1132', '1141']
S16P2 = ['1121', '1122', '1131']          # 16+2 semesters (midterm in Week 8)
WEEKS = ['W6', 'W8', 'W10', 'W12']        # documented candidate set; W12 = reference
WNUM = {'W6': 6, 'W8': 8, 'W10': 10, 'W12': 12, 'W14': 14, 'W16': 16}
AUC_BOUND = 0.02
R_BOUND = 0.03
EQ_BAND = 0.02
PRED = ['M1', 'M2', 'M3']                 # M1' (BAS), M2' (XGBoost), M3' (blend)
PRED_COL = {'M1': 'm1', 'M2': 'm2', 'M3': 'm3'}


# ---------------------------------------------------------------------------
# percentile function of the original construction: ews_lib.pct_rank
# ---------------------------------------------------------------------------
def pct_rank(x):
    x = np.asarray(x, float)
    return rankdata(x) / len(x)


def midrank_pct(x):
    """(rank - 0.5) / n with midranks for ties (spec: midterm percentile rank in R5-11)."""
    x = np.asarray(x, float)
    return (rankdata(x) - 0.5) / len(x)


def auc_fail(y, pass_score):
    """AUC for failure; the pass-oriented score is reversed (identical to ews_lib.auc_fail)."""
    y = np.asarray(y, int)
    r = rankdata(-np.asarray(pass_score, float))
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def recall_at(frame, score_col, frac=0.2, seed=7, y_col='y', sem_col='semester'):
    """Within-semester list of ceil(frac*n) lowest pass-scores (ews_lib.recall_precision_at logic,
    random tie-break with seed 7 as in the frozen pipeline).  Returns (recall, precision, listed_mask)."""
    rng = np.random.default_rng(seed)
    listed = np.zeros(len(frame), bool)
    for _sem, g in frame.groupby(sem_col, sort=True):
        s = g[score_col].to_numpy(float)
        k = int(math.ceil(frac * len(g)))
        order = np.lexsort((rng.random(len(g)), s))
        pos = np.flatnonzero(frame.index.isin(g.index))
        listed[pos[order[:k]]] = True
    y = frame[y_col].to_numpy(int)
    fails = int(y.sum())
    tp = int(y[listed].sum())
    return (tp / fails if fails else np.nan), (tp / listed.sum() if listed.sum() else np.nan), listed


def list_by_k(score, sem, k_by_sem, seed=7):
    """Within each semester list the k lowest pass-scores; random tie-break with the same rng stream as
    recall_at (seed 7, one draw of len(group) per semester in sorted order), so lists of different size
    are nested prefixes of one order.  k_by_sem: dict semester -> k or callable n -> k."""
    score = np.asarray(score, float)
    sem = np.asarray(sem)
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(score), bool)
    for s in sorted(np.unique(sem)):
        ix = np.flatnonzero(sem == s)
        order = np.lexsort((rng.random(len(ix)), score[ix]))
        k = k_by_sem(len(ix)) if callable(k_by_sem) else int(k_by_sem[s])
        mask[ix[order[:k]]] = True
    return mask


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_base():
    """Return (base, scores).  base: 2,652 analysis records (keyed file P/C, W8 order = canonical order).
    scores[W]: DataFrame aligned to base rows with m1, m2, m3, pfail, qmi_none."""
    k = pd.read_csv(PACK / 'oof_scores_keyed.csv', dtype={'anon_id': str, 'semester': str})
    pc = k[(k.arm == 'P') & (k.feature_set == 'C')]
    scores = {}
    ref_keys = None
    for W in ['W6', 'W8', 'W10', 'W12', 'W14', 'W16']:
        g = pc[pc.W == W].reset_index(drop=True)
        keys = list(zip(g.semester, g.anon_id))
        if ref_keys is None:
            ref_keys = keys
            base = g[['semester', 'anon_id', 'stratum', 'final', 'y']].copy()
        assert keys == ref_keys, f'record order differs at {W}'
        scores[W] = g[['m1', 'm2', 'm3', 'pfail', 'qmi_none']].copy()
    d7 = pd.read_csv(PACK / 'd7_semester_scores.csv', dtype={'anon_id': str, 'semester': str})
    base = base.merge(d7, on=['anon_id', 'semester'], how='left', validate='one_to_one')
    base['final'] = base['final'].astype(float)
    cm = pd.read_csv(PACK / 'datapack_complete/03_class_membership/class_membership.csv', dtype=str)
    c5 = pd.read_csv(PACK / 'datapack_complete/08_class_membership_and_1142_outcomes/class_membership5.csv',
                     dtype=str)
    base = base.merge(cm[['semester', 'anon_id', 'class_track']], on=['semester', 'anon_id'], how='left',
                      validate='one_to_one')
    base = base.merge(c5[['semester', 'anon_id', 'program5_key']], on=['semester', 'anon_id'], how='left',
                      validate='one_to_one')
    base['retake'] = (base['class_track'] == 'retake').astype(int)
    base['stu'] = pd.factorize(base['anon_id'])[0]
    base['sem_i'] = base['semester'].map({s: i for i, s in enumerate(D7)})
    return base, scores


def deploy_blend(m1, m2, sem):
    """Mean of the within-semester percentile ranks of m1 and m2 (original pct_rank function)."""
    m1 = np.asarray(m1, float)
    m2 = np.asarray(m2, float)
    sem = np.asarray(sem)
    out = np.empty(len(m1))
    for s in np.unique(sem):
        ix = np.flatnonzero(sem == s)
        out[ix] = (pct_rank(m1[ix]) + pct_rank(m2[ix])) / 2
    return out


def pooled_blend(m1, m2):
    return (pct_rank(m1) + pct_rank(m2)) / 2


def b1_scores(base, class_col='program5_key', exclude_test_students=True):
    """B1 rebuilt per Appendix A: per LOSO fold a saturated lookup of the class type's failure rate in
    the training folds (overall training rate where the class is absent).  Training records of students
    who also appear in the test semester are dropped, as in the frozen pipeline (run_folds).
    Returns the failure-oriented score (higher = more likely to fail)."""
    out = np.zeros(len(base))
    for s in D7:
        te = (base.semester == s).to_numpy()
        tr = ~te
        if exclude_test_students:
            tr &= ~base.anon_id.isin(set(base.anon_id[te])).to_numpy()
        ov = base.y[tr].mean()
        rate = base[tr].groupby(class_col).y.mean()
        out[te] = base.loc[te, class_col].map(rate).fillna(ov).to_numpy()
    return out


# ---------------------------------------------------------------------------
# student-cluster bootstrap by weights
# ---------------------------------------------------------------------------
def make_counts(n_stu, B=BOOT_B, seed=SEED):
    """(B x n_stu) per-student draw multiplicities of the standing student-cluster bootstrap."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_stu, size=(B, n_stu))
    cnt = np.zeros((B, n_stu), np.float32)
    for b in range(B):
        cnt[b] = np.bincount(draws[b], minlength=n_stu)
    return cnt


def make_weights(stu_idx, B=BOOT_B, seed=SEED, n_stu=None):
    """(B x N) record weights: draw n_stu students with replacement; all of a drawn student's records
    carry the draw multiplicity.  float32 (counts are small integers, exactly representable)."""
    stu_idx = np.asarray(stu_idx)
    n_stu = int(stu_idx.max()) + 1 if n_stu is None else n_stu
    return make_counts(n_stu, B, seed)[:, stu_idx]


def inner_weights(outer_counts_stu, stu_idx, D, rng):
    """Inner draws for the nested procedure: resample the outer resample's students (with their outer
    multiplicity) D times.  outer_counts_stu: per-student count vector of the outer resample."""
    c = np.asarray(outer_counts_stu, float)
    n = int(c.sum())
    nz = np.flatnonzero(c > 0)
    cnt = np.zeros((D, len(c)), np.float32)
    cnt[:, nz] = rng.multinomial(n, c[nz] / c[nz].sum(), size=D)
    return cnt[:, stu_idx]


def _chunks(B, size):
    for a in range(0, B, size):
        yield a, min(B, a + size)


def auc_weighted(Wm, S, y, groups, chunk=250):
    """Weighted AUC for failure.  S: (N x C) pass-oriented scores; y: (N,) 0/1; groups: list of index
    arrays.  Returns (B x G x C) AUC (NaN where a group has no fail or no pass weight) and (B x G) n."""
    Bn = Wm.shape[0]
    C = S.shape[1]
    out = np.full((Bn, len(groups), C), np.nan)
    nn = np.zeros((Bn, len(groups)))
    for gi, ix in enumerate(groups):
        ix = np.asarray(ix)
        yy = y[ix]
        F = ix[yy == 1]
        Pp = ix[yy == 0]
        if len(F) == 0 or len(Pp) == 0:
            continue
        # fail should have the LOWER pass-score:  M[i,j] = 1(s_j > s_i) + 0.5 1(s_j == s_i), i in F, j in P
        Ms = []
        for c in range(C):
            sf = S[F, c][:, None]
            sp = S[Pp, c][None, :]
            Ms.append(((sp > sf) + 0.5 * (sp == sf)).astype(np.float32))
        Mcat = np.concatenate(Ms, axis=1)
        for a, b in _chunks(Bn, chunk):
            Wf = Wm[a:b][:, F]
            Wp = Wm[a:b][:, Pp]
            T = (Wf @ Mcat).reshape(b - a, C, len(Pp))
            num = (T * Wp[:, None, :]).sum(axis=2)
            den = Wf.sum(1, dtype=np.float64) * Wp.sum(1, dtype=np.float64)
            with np.errstate(invalid='ignore', divide='ignore'):
                out[a:b, gi, :] = np.where(den[:, None] > 0, num / den[:, None], np.nan)
            nn[a:b, gi] = Wm[a:b][:, ix].sum(1, dtype=np.float64)
    return out, nn


def corr_weighted(Wm, X, f, groups, chunk=500):
    """Weighted Pearson r between each column of X (N x C) and f (N,) within each group.
    Returns (B x G x C) r and (B x G) n."""
    Bn = Wm.shape[0]
    C = X.shape[1]
    out = np.full((Bn, len(groups), C), np.nan)
    nn = np.zeros((Bn, len(groups)))
    for gi, ix in enumerate(groups):
        ix = np.asarray(ix)
        Xg = X[ix] - X[ix].mean(0)
        fg = f[ix] - f[ix].mean()
        feats = np.concatenate([np.ones((len(ix), 1)), Xg, fg[:, None], Xg ** 2, (fg ** 2)[:, None],
                                Xg * fg[:, None]], axis=1)
        for a, b in _chunks(Bn, chunk):
            Wg = Wm[a:b][:, ix].astype(np.float64)
            Mo = Wg @ feats
            sw = Mo[:, 0]
            sx = Mo[:, 1:1 + C]
            sf = Mo[:, 1 + C]
            sxx = Mo[:, 2 + C:2 + 2 * C]
            sff = Mo[:, 2 + 2 * C]
            sxf = Mo[:, 3 + 2 * C:3 + 3 * C]
            with np.errstate(invalid='ignore', divide='ignore'):
                cov = sxf - sx * sf[:, None] / sw[:, None]
                vx = sxx - sx ** 2 / sw[:, None]
                vf = sff - sf ** 2 / sw
                out[a:b, gi, :] = cov / np.sqrt(vx * vf[:, None])
            nn[a:b, gi] = sw
    return out, nn


def aggregate_deploy(auc_g, r_g, n_g):
    """Deployment definition: n-weighted mean of within-semester AUCs; Fisher-z (n-3)-weighted mean of
    within-semester r (ews_lib.macro_metrics).  Inputs (B x G x C), (B x G); NaN groups are skipped."""
    n = n_g[:, :, None]
    ok_a = ~np.isnan(auc_g)
    wa = np.where(ok_a, n, 0.0)
    auc = (np.where(ok_a, auc_g, 0.0) * wa).sum(1) / wa.sum(1)
    ok_r = ~np.isnan(r_g)
    wr = np.where(ok_r, n - 3, 0.0)
    z = np.arctanh(np.clip(np.where(ok_r, r_g, 0.0), -0.999, 0.999))
    r = np.tanh((z * wr).sum(1) / wr.sum(1))
    return auc, r


def metrics_draws(Wm, S, y, f, sem_groups, definition):
    """AUC and r draws (B x C) for columns of S under the pooled or the deployment definition."""
    if definition == 'pooled':
        allix = [np.arange(len(y))]
        a, _ = auc_weighted(Wm, S, y, allix)
        r, _ = corr_weighted(Wm, S, f, allix)
        return a[:, 0, :], r[:, 0, :]
    a, n = auc_weighted(Wm, S, y, sem_groups)
    r, n2 = corr_weighted(Wm, S, f, sem_groups)
    return aggregate_deploy(a, r, n)


def sem_groups_of(sem_i):
    sem_i = np.asarray(sem_i)
    return [np.flatnonzero(sem_i == s) for s in np.unique(sem_i)]


def ci(arr, level=0.95):
    a = (1 - level) / 2
    return float(np.nanpercentile(arr, 100 * a)), float(np.nanpercentile(arr, 100 * (1 - a)))


# ---------------------------------------------------------------------------
# Appendix B selection rule
# ---------------------------------------------------------------------------
def build_candidates(base, scores, definition, weeks=WEEKS, preds=PRED, extra=None, recompute_pooled_m3=False):
    """Return (names, S) where S is (N x C): one pass-oriented column per (week, predictor).
    pooled: frozen pooled columns m1, m2, m3 (m3 = frozen pooled-percentile blend).
    deploy: m1, m2 raw frozen; m3 replaced by the within-semester blend of m1 and m2."""
    names, cols = [], []
    for W in weeks:
        sc = scores[W]
        for p in preds:
            if p == 'M3' and definition == 'deploy':
                v = deploy_blend(sc.m1.to_numpy(), sc.m2.to_numpy(), base.semester.to_numpy())
            elif p == 'M3' and recompute_pooled_m3:
                v = pooled_blend(sc.m1.to_numpy(), sc.m2.to_numpy())   # ranks over the rows supplied (retained set)
            else:
                v = sc[PRED_COL[p]].to_numpy(float)
            names.append((W, p))
            cols.append(v)
    if extra:
        for nm, v in extra.items():
            names.append(nm)
            cols.append(np.asarray(v, float))
    return names, np.column_stack(cols)


def select_rule(auc, rr, names, point_auc, weeks=WEEKS, preds=PRED, ref='W12', ucl_pct=97.5):
    """Apply Appendix B to draws auc, rr (B x C) with column names (week, pred).
    Returns dict: wstar[pred], detail rows, selected week, selected policy, comparison info."""
    col = {n: i for i, n in enumerate(names)}
    detail = []
    wstar = {}
    for p in preds:
        wstar[p] = None
        for W in weeks:
            if (W, p) not in col:
                continue
            if W == ref:
                la = np.zeros(auc.shape[0])
                lr = np.zeros(auc.shape[0])
            else:
                la = auc[:, col[(ref, p)]] - auc[:, col[(W, p)]]
                lr = rr[:, col[(ref, p)]] - rr[:, col[(W, p)]]
            ua = float(np.nanpercentile(la, ucl_pct))
            ur = float(np.nanpercentile(lr, ucl_pct))
            ok = (ua <= AUC_BOUND) and (ur <= R_BOUND)
            detail.append({'pred': p, 'week': W, 'auc_loss_ucl': ua, 'r_loss_ucl': ur, 'qualifies': bool(ok)})
            if ok and wstar[p] is None:
                wstar[p] = W
    qual = {p: w for p, w in wstar.items() if w is not None}
    if not qual:
        return {'wstar': wstar, 'detail': detail, 'sel_week': None, 'sel_pred': None, 'compare': []}
    sel_week = min(qual.values(), key=lambda w: WNUM[w])
    cands = [p for p, w in qual.items() if w == sel_week]
    cmp_rows = []
    if len(cands) == 1:
        sel_pred = cands[0]
    else:
        order = sorted(cands, key=lambda p: -point_auc[(sel_week, p)])
        sel_pred = order[0]
        for other in order[1:]:
            d = auc[:, col[(sel_week, sel_pred)]] - auc[:, col[(sel_week, other)]]
            lo, hi = ci(d)
            within = (lo >= -EQ_BAND) and (hi <= EQ_BAND)
            cmp_rows.append({'week': sel_week, 'a': sel_pred, 'b': other,
                             'delta_point': point_auc[(sel_week, sel_pred)] - point_auc[(sel_week, other)],
                             'lo': lo, 'hi': hi, 'entirely_within_band': bool(within)})
    return {'wstar': wstar, 'detail': detail, 'sel_week': sel_week, 'sel_pred': sel_pred, 'compare': cmp_rows}


def subset_view(base, scores, keep):
    """Rows kept (boolean mask); students re-factorized within the subset (clusters = anon_id in the subset)."""
    keep = np.asarray(keep, bool)
    b = base[keep].reset_index(drop=True).copy()
    b['stu'] = pd.factorize(b['anon_id'])[0]
    sc = {W: scores[W][keep].reset_index(drop=True) for W in scores}
    return b, sc


def run_rule(b, sc, y, f, definition, Wm, recompute_pooled_m3=False, extra=None, weeks=WEEKS, preds=PRED,
             sel_preds=None):
    """Appendix B on the rows of b.  Returns dict with selection, point and draw metrics."""
    names, S = build_candidates(b, sc, definition, weeks=weeks, preds=preds, extra=extra,
                                recompute_pooled_m3=recompute_pooled_m3)
    groups = sem_groups_of(b.sem_i)
    a_d, r_d = metrics_draws(Wm, S, y, f, groups, definition)
    a_p, r_p = metrics_draws(np.ones((1, len(y)), np.float32), S, y, f, groups, definition)
    pt = {n: float(a_p[0, i]) for i, n in enumerate(names)}
    rpt = {n: float(r_p[0, i]) for i, n in enumerate(names)}
    sel = select_rule(a_d, r_d, names, pt, weeks=weeks, preds=sel_preds or preds)
    return {'sel': sel, 'names': names, 'S': S, 'auc_d': a_d, 'r_d': r_d, 'auc_pt': pt, 'r_pt': rpt}


def rule_rows(variant, definition, res, blend_week, extra_note=''):
    """Flatten a run_rule() result into long-format rows (W*, policy, loss limits, third-rule comparisons,
    blend metrics at blend_week with the standing bootstrap interval)."""
    sel, names = res['sel'], res['names']
    col = {n: i for i, n in enumerate(names)}
    rows = []

    def row(rt, pred='', week='', est=np.nan, lo=np.nan, hi=np.nan, note=''):
        rows.append({'variant': variant, 'definition': definition, 'row_type': rt, 'predictor': pred, 'week': week,
                     'estimate': est, 'ci_lo': lo, 'ci_hi': hi, 'note': (note + ' ' + extra_note).strip()})

    for p in PRED:
        row('W_star', p, sel['wstar'][p] or 'none')
    row('selected_policy', sel['sel_pred'] or 'none', sel['sel_week'] or 'none')
    for d in sel['detail']:
        row('loss_upper_limit', d['pred'], d['week'], d['auc_loss_ucl'], note=f"AUC-loss UCL (97.5th pct); r-loss UCL "
            f"{d['r_loss_ucl']:.4f}; qualifies={d['qualifies']} (bounds 0.020 / 0.030)")
    for c in sel['compare']:
        row('third_rule_comparison', f"{c['a']} minus {c['b']}", c['week'], c['delta_point'], c['lo'], c['hi'],
            f"entirely within +-0.02 band: {c['entirely_within_band']}")
    for p in PRED:
        key = (blend_week, p)
        if key in col:
            lo, hi = ci(res['auc_d'][:, col[key]])
            row('metric_AUC', p, blend_week, res['auc_pt'][key], lo, hi)
            lo, hi = ci(res['r_d'][:, col[key]])
            row('metric_r', p, blend_week, res['r_pt'][key], lo, hi)
    return rows


def load_step9_features(base, W):
    """Frozen step9 feature values (inputs_D7) aligned to the base rows by (semester, anon_id).  W in 6, 8, 10, 12."""
    import json
    rows = []
    for sm in D7:
        d = json.load(open(PACK / f'datapack_complete/01_step9_inputs/inputs_D7/out_W{W}/{sm}_step9_inputs.json',
                           encoding='utf-8'))
        for r in d['scatter_data']:
            rr = dict(r['features'])
            rr.update({'semester': sm, 'anon_id': r['anon_id']})
            rows.append(rr)
    df = pd.DataFrame(rows)
    out = base[['semester', 'anon_id']].merge(df, on=['semester', 'anon_id'], how='left', validate='one_to_one')
    assert len(out) == len(base)
    return out
