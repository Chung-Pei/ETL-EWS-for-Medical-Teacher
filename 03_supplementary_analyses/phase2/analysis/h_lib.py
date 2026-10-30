# -*- coding: utf-8 -*-
"""Shared helpers for the Paper A supplementary analyses (Phase 1 subset).

Definitions follow materials/.../05_code/ews_lib.py (auc_fail, pearson, pct_rank, recall_precision_at,
Boot draw mechanism). ews_lib itself is not imported because it needs xgboost and the ETL tree.
"""
import math
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.stats import t as student_t

W_ALL = ['W6', 'W8', 'W10', 'W12', 'W14', 'W16', 'Full']
MODEL_COL = {"M1'": 'm1', "M2'": 'm2', "M3'": 'm3'}
SIMPLICITY = ["M1'", "M2'", "M3'"]          # rule-based < XGBoost < blend
BOOT_B = 2000
BOOT_SEED = 20261006
D_AUC = 0.02
D_R = 0.03
EQUIV_MARGIN = 0.02


class AlignmentError(Exception):
    pass


# ----------------------------------------------------------------------------- data
def assert_aligned(frames):
    """Every week must hold the same records in the same row order (semester, final, y identical)."""
    ref_key = 'W6' if 'W6' in frames else sorted(frames)[0]
    ref = frames[ref_key]
    for W, df in frames.items():
        if len(df) != len(ref):
            raise AlignmentError(f'{W}: length {len(df)} != {len(ref)}')
        for col in ('semester', 'final', 'y'):
            if not np.array_equal(df[col].to_numpy(), ref[col].to_numpy()):
                raise AlignmentError(f'{W}: column {col} differs from {ref_key}')


def load_oof(path):
    d = pd.read_csv(path, encoding='utf-8-sig')
    d = d[(d['arm'] == 'P') & (d['feature_set'] == 'C')].copy()
    # Ruling (ledger): the shipped m3 equals the POOLED percentile blend (rank over all 2,652 records of a week)
    # to 1e-16, but its float noise breaks mathematical ties differently from the frozen pipeline, which moves
    # M3' AUC by up to 6e-6. Recomputing m3 from m1 and m2 with the same formula reproduces the frozen AUCs to
    # 1e-9; the shipped column is kept as m3_shipped for audit.
    out = {}
    for W, g in d.groupby('W'):
        g = g.reset_index(drop=True)
        g['m3_shipped'] = g['m3']
        g['m3'] = pooled_pct_blend(g['m1'].to_numpy(), g['m2'].to_numpy())
        out[W] = g[['semester', 'final', 'y', 'm1', 'm2', 'm3', 'm3_shipped']]
    assert_aligned(out)
    return out


# ----------------------------------------------------------------------------- metrics
def auc_fail(y, pass_score):
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


def auc_matrix(Y, S):
    """Row-wise AUC (failing = positive, risk = -score) for resampled matrices Y, S of shape (B, n)."""
    R = rankdata(-S, axis=1)
    n1 = Y.sum(axis=1).astype(float)
    n0 = Y.shape[1] - n1
    with np.errstate(divide='ignore', invalid='ignore'):
        auc = ((R * Y).sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    auc[(n1 == 0) | (n0 == 0)] = np.nan
    return auc


def corr_matrix(A, F):
    Ac = A - A.mean(axis=1, keepdims=True)
    Fc = F - F.mean(axis=1, keepdims=True)
    den = np.sqrt((Ac ** 2).sum(axis=1) * (Fc ** 2).sum(axis=1))
    with np.errstate(divide='ignore', invalid='ignore'):
        return (Ac * Fc).sum(axis=1) / den


def recall_precision_at(frame, score_col, frac=0.2, seed=7):
    """Within each semester flag the lowest ceil(frac*n) pass-scores; pool numerators and denominators."""
    rng = np.random.default_rng(seed)
    tp = flagged = fails = 0
    for _sem, g in frame.groupby('semester'):
        s = g[score_col].to_numpy(float)
        y = g['y'].to_numpy(int)
        k = int(math.ceil(frac * len(g)))
        order = np.lexsort((rng.random(len(g)), s))
        tp += int(y[order[:k]].sum())
        flagged += k
        fails += int(y.sum())
    return (tp / fails if fails else np.nan), (tp / flagged if flagged else np.nan)


def weighted_mean(vals, weights):
    vals = np.asarray(vals, float)
    weights = np.asarray(weights, float)
    return float((vals * weights).sum() / weights.sum())


def fisher_weighted_r(rs, ns):
    """Weighted Fisher-z mean of correlations along the last axis (weights n-3), back-transformed."""
    rs = np.asarray(rs, float)
    w = np.asarray(ns, float) - 3.0
    z = np.arctanh(np.clip(rs, -0.999, 0.999))
    return np.tanh((z * w).sum(axis=-1) / w.sum())


def pct_rank(x):
    x = np.asarray(x, float)
    return rankdata(x) / len(x)


def pct_ref(x, ref):
    """Mid-rank empirical percentile of x within a fixed reference sample."""
    ref = np.sort(np.asarray(ref, float))
    x = np.asarray(x, float)
    lo = np.searchsorted(ref, x, side='left')
    hi = np.searchsorted(ref, x, side='right')
    return (lo + 0.5 * (hi - lo)) / len(ref)


def pooled_pct_blend(m1, m2):
    """M3' as in the frozen outputs: mean of pooled percentile ranks (rankdata / N) of the two scores."""
    n = len(m1)
    return (rankdata(m1) / n + rankdata(m2) / n) / 2


def within_semester_blend(frame):
    """M3' variant: percentile ranks taken inside each semester (the holdout/deployment convention)."""
    out = np.empty(len(frame))
    sem = frame['semester'].to_numpy()
    for s_ in np.unique(sem):
        ix = sem == s_
        out[ix] = (pct_rank(frame['m1'].to_numpy()[ix]) + pct_rank(frame['m2'].to_numpy()[ix])) / 2
    return out


# ----------------------------------------------------------------------------- bootstrap
def record_draws(n, B=BOOT_B, seed=BOOT_SEED):
    """Same draw mechanism as ews_lib.Boot with one cluster per record."""
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def strat_draws(sizes, B=BOOT_B, seed=BOOT_SEED):
    """Independent within-semester draws; sizes maps semester -> n. Stream per semester = [seed, rank of semester]."""
    out = {}
    for i, sem in enumerate(sorted(sizes)):
        out[sem] = np.random.default_rng([seed, i]).integers(0, sizes[sem], size=(B, sizes[sem]))
    return out


def boot_record_series(frames, keys, draws):
    """keys = [(W, col)]; returns {(W, col): (auc_b, r_b)} for pooled records, one shared draw matrix."""
    out = {}
    for W, col in keys:
        df = frames[W]
        y = df['y'].to_numpy(int)[draws]
        s = df[col].to_numpy(float)[draws]
        f = df['final'].to_numpy(float)[draws]
        out[(W, col)] = (auc_matrix(y, s), corr_matrix(s, f))
    return out


def semester_rows(frame):
    sems = sorted(frame['semester'].unique())
    return sems, {s: np.where(frame['semester'].to_numpy() == s)[0] for s in sems}


def boot_strat_series(frames, keys, sdraws, score_override=None):
    """Semester-stratified bootstrap. Returns {(W, col): dict(sem_auc (B,k), sem_r (B,k), pooled_auc (B,), pooled_r (B,))}.
    score_override maps (W, col) -> array aligned with frames[W] rows (used for variant scores)."""
    out = {}
    for W, col in keys:
        df = frames[W]
        sems, rows = semester_rows(df)
        sc = df[col].to_numpy(float) if score_override is None or (W, col) not in score_override \
            else np.asarray(score_override[(W, col)], float)
        y_all = df['y'].to_numpy(int)
        f_all = df['final'].to_numpy(float)
        sa, sr, Ys, Ss, Fs = [], [], [], [], []
        for s in sems:
            ix = rows[s]
            d = sdraws[s]
            y, f, x = y_all[ix][d], f_all[ix][d], sc[ix][d]
            sa.append(auc_matrix(y, x))
            sr.append(corr_matrix(x, f))
            Ys.append(y)
            Ss.append(x)
            Fs.append(f)
        Y, S, F = np.hstack(Ys), np.hstack(Ss), np.hstack(Fs)
        out[(W, col)] = {'sem_auc': np.column_stack(sa), 'sem_r': np.column_stack(sr),
                         'pooled_auc': auc_matrix(Y, S), 'pooled_r': corr_matrix(S, F)}
    return out


def percentile_hi(arr, level):
    """Upper limit of a two-sided CI. level 0.975 -> 97.5th percentile; 'bonf' -> alpha .05/7 two-sided (99.3% CI)."""
    p = 100 * (1 - 0.05 / 14) if level == 'bonf' else 100 * float(level)
    return float(np.nanpercentile(arr, p))


def percentile_ci(arr, level=0.95):
    a = (1 - level) / 2
    return float(np.nanpercentile(arr, 100 * a)), float(np.nanpercentile(arr, 100 * (1 - a)))


# ----------------------------------------------------------------------------- rule and pooling
def qualifies(auc_hi, r_hi, d_auc=D_AUC, d_r=D_R):
    return bool(auc_hi <= d_auc + 1e-12 and r_hi <= d_r + 1e-12)


def earliest_week(ok_by_week):
    for w in sorted(ok_by_week):
        if ok_by_week[w]:
            return w
    return None


def choose_model(auc_by_model, diff_ci, margin=EQUIV_MARGIN):
    """Third decision rule: highest AUC wins unless its advantage over the runner-up has a CI wholly within
    +-margin, in which case the simpler model (order M1' < M2' < M3') is preferred."""
    ranked = sorted(auc_by_model, key=lambda m: auc_by_model[m], reverse=True)
    if len(ranked) == 1:
        return ranked[0]
    best, second = ranked[0], ranked[1]
    if diff_ci[0] >= -margin and diff_ci[1] <= margin:
        return sorted([best, second], key=SIMPLICITY.index)[0]
    return best


def dl_random_effects(est, se):
    """DerSimonian-Laird random-effects pooling with normal CI and t-based prediction interval (df = k-2)."""
    est = np.asarray(est, float)
    se = np.asarray(se, float)
    k = len(est)
    w = 1.0 / se ** 2
    fixed = (w * est).sum() / w.sum()
    q = float((w * (est - fixed) ** 2).sum())
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (q - (k - 1)) / c)
    ws = 1.0 / (se ** 2 + tau2)
    pooled = float((ws * est).sum() / ws.sum())
    se_p = math.sqrt(1.0 / ws.sum())
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    tcrit = student_t.ppf(0.975, k - 2) if k > 2 else np.nan
    half = tcrit * math.sqrt(tau2 + se_p ** 2) if k > 2 else np.nan
    return {'pooled': pooled, 'se': se_p, 'ci_lo': pooled - 1.959964 * se_p, 'ci_hi': pooled + 1.959964 * se_p,
            'tau2': float(tau2), 'i2': float(i2), 'q': q, 'k': k, 'pi_lo': pooled - half, 'pi_hi': pooled + half}


def log_versions(stream=None):
    import platform
    import scipy
    stream = stream or sys.stdout
    print(f'python {platform.python_version()} numpy {np.__version__} scipy {scipy.__version__} '
          f'pandas {pd.__version__}', file=stream)
