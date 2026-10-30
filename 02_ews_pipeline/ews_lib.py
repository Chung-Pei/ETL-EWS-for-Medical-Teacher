# -*- coding: utf-8 -*-
"""EWS timing experiment library (Spec v1.2 + amendment v1.3). Identifiers are ASCII-only.

Data flow:  out_W{W}/step9_inputs.json  ->  ETL's own build_records_from_memory / add_derived_features
            -> per-fold models (BAS via ETL functions; XGBoost with ETL hyper-parameters)
"""
import importlib.util
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats as sps
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

D7 = ['1111', '1112', '1121', '1122', '1131', '1132', '1141']
UPPER = ['1111', '1121', '1131', '1141']
LOWER = ['1112', '1122', '1132']
STRATUM = {s: ('upper' if s[-1] == '1' else 'lower') for s in D7 + ['1142']}
SEEDS = [42, 43, 44, 45, 46]
TIME5 = ['active_weeks', 'weekly_minutes_std', 'cram_pattern_score', 'late_night_ratio', 'early_start_ratio']
DROP5 = ['vid_completion_rate', 'aud_completion_rate', 'txt_completion_rate', 'sup_completion_rate',
         'tut_total_minutes']
BOOT_B = 2000
BOOT_SEED = 20261006


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Etl:
    """Wrapper that exposes the ETL module-11 objects the experiment must reuse unchanged."""

    def __init__(self, etl_dir):
        self.m11 = load_mod(os.path.join(etl_dir, 'lms_etl', 'modules', '11_cross_analysis.py'), 'm11_exp')
        m = self.m11
        self.FS_A = list(m.XGB_FEATURES)
        self.FS_B = [f for f in self.FS_A if f not in DROP5 + ['student_type_fail_rate_hist']]
        self.FS_C = [f for f in self.FS_B if f != 'midterm']
        self.FS_Bm = [f for f in self.FS_B if f not in TIME5]          # S12
        self.FS_Cm = [f for f in self.FS_C if f not in TIME5]
        self.QMI_W, self.PAS_W, self.BAS_W = m.QMI_WEIGHTS, m.PASSIVE_WEIGHTS, m.BAS_WEIGHTS
        assert len(self.FS_B) == 22 and len(self.FS_C) == 21, (len(self.FS_B), len(self.FS_C))


# ----------------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------------
def load_week(etl, root, W, tag=None):
    """Load one ETL run, rebuild production-identical records and attach the 5 time features."""
    path = os.path.join(root, tag or f'out_W{W}', 'step9_inputs.json')
    d = json.load(open(path, encoding='utf-8'))
    m = etl.m11
    recs, _meta = m.build_records_from_memory(d['scatter_data'], d['lsa_type_map'],
                                              d['incomplete_semesters'], d['transition_ratios'])
    recs = m.add_derived_features(recs)
    raw = {(s['anon_id'], s['semester']): s['features'] for s in d['scatter_data']}
    for r in recs:
        tf = d['time_features'].get(f"{r['anon_id']}||{r['semester']}", {})
        for k in TIME5:
            v = tf.get(k)
            r[k] = np.nan if v is None else float(v)
        r['_raw'] = raw.get((r['anon_id'], r['semester']), {})
    return recs


def num(v):
    if v is None:
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def build_matrix(recs, feats, nan_variant=False):
    rows = []
    for r in recs:
        row = []
        for f in feats:
            v = r.get(f)
            if nan_variant and f in r.get('_raw', {}) and r['_raw'][f] is None:
                v = None
            row.append(num(v))
        if nan_variant:
            row.append(1.0 if (r.get('quz_total_attempts') or 0) > 0 else 0.0)
        rows.append(row)
    return np.array(rows, dtype=float)


# ----------------------------------------------------------------------------
# BAS (M1) -- always through the ETL functions
# ----------------------------------------------------------------------------
PLACEHOLDER = dict(qmi_mean=0.0, qmi_std=1.0, pas_mean=0.0, pas_std=1.0, mid_mean=0.0, mid_std=1.0,
                   att_mean=0.0, att_std=1.0)


def bas_weights(include_midterm):
    w = dict(BAS_W0)
    if include_midterm:
        return w
    return {'midterm': 0.0, **{k: v / 0.65 for k, v in w.items() if k != 'midterm'}}


BAS_W0 = None  # set by init_bas()


def init_bas(etl):
    global BAS_W0
    BAS_W0 = dict(etl.BAS_W)


def _copies(recs):
    return [dict((k, v) for k, v in r.items() if k != '_raw') for r in recs]


def bas_fold(etl, train, test, include_midterm=True):
    """Frozen BAS (spec 5.3): 4 terms, z-statistics from the training fold only. Returns (bas_train, bas_test, stats)."""
    m = etl.m11
    w = bas_weights(include_midterm)
    tr, te = _copies(train), _copies(test)
    for r in tr + te:
        m.compute_qmi_passive_bas(r, PLACEHOLDER, etl.QMI_W, etl.PAS_W, w, True)
    stats_tr = m._safe_qmi_passive_stats([r for r in tr if r.get('final') is not None], include_log_attempts=True)
    for r in tr + te:
        m.compute_qmi_passive_bas(r, stats_tr, etl.QMI_W, etl.PAS_W, w, True)
    qmi_none_te = float(np.mean([r['qmi'] is None for r in te])) if te else 0.0
    return (np.array([r['bas_score'] for r in tr], float), np.array([r['bas_score'] for r in te], float),
            stats_tr, qmi_none_te)


def bas_deployed(etl, train, test):
    """S10: deployed variant (12_early_warning.add_qmi_bas): 3 terms, semester-local midterm statistics."""
    m = etl.m11
    tr, te = _copies(train), _copies(test)
    w = dict(etl.BAS_W)
    for r in tr:
        m.compute_qmi_passive_bas(r, PLACEHOLDER, etl.QMI_W, etl.PAS_W, w, True)
    st = m._safe_qmi_passive_stats([r for r in tr if r.get('final') is not None], include_log_attempts=False)
    mids = np.array([r['midterm'] for r in te if r.get('midterm') is not None], float)
    mm, ms = (float(mids.mean()), float(mids.std())) if len(mids) >= 2 else (60.0, 15.0)
    stats = {'qmi_mean': st['qmi_mean'], 'qmi_std': st['qmi_std'] if st['qmi_std'] > 0 else 1.0,
             'pas_mean': st['pas_mean'], 'pas_std': st['pas_std'] if st['pas_std'] > 0 else 1.0,
             'mid_mean': mm, 'mid_std': ms if ms > 0 else 1.0}
    for r in te:
        m.compute_qmi_passive_bas(r, stats, etl.QMI_W, etl.PAS_W, w, False)
    return np.array([r['bas_score'] for r in te], float)


def bas_refit(etl, train, test, stats_tr, include_midterm=True):
    """S7: re-fit non-negative weights on the training fold (least squares on final score), apply to test fold."""
    from scipy.optimize import nnls
    m = etl.m11
    tr, te = _copies(train), _copies(test)
    w = bas_weights(include_midterm)
    for r in tr + te:
        m.compute_qmi_passive_bas(r, stats_tr, etl.QMI_W, etl.PAS_W, w, True)

    def comp(r):
        z_mid = m._zscore(r['midterm'] if r.get('midterm') is not None else 0.0, stats_tr['mid_mean'], stats_tr['mid_std']) if include_midterm else 0.0
        z_q = m._zscore(r['qmi'], stats_tr['qmi_mean'], stats_tr['qmi_std']) if r['qmi'] is not None else np.nan
        z_p = m._zscore(-r['passive_index'], -stats_tr['pas_mean'], stats_tr['pas_std'])
        z_a = m._zscore(np.log1p(r.get('quz_total_attempts', 0.0) or 0.0), stats_tr['att_mean'], stats_tr['att_std'])
        return [z_mid, z_q, z_p, z_a]

    A = np.array([comp(r) for r in tr], float)
    y = np.array([r['final'] for r in tr], float)
    ok = ~np.isnan(A).any(axis=1)
    cols = [0, 1, 2, 3] if include_midterm else [1, 2, 3]
    coef, _ = nnls(A[ok][:, cols], y[ok] - y[ok].mean())
    wts = np.zeros(4)
    wts[cols] = coef
    out = []
    for r in te:
        c = np.array(comp(r), float)
        if np.isnan(c[1]):
            ww = wts.copy()
            ww[1] = 0.0
            out.append(float((ww * np.nan_to_num(c)).sum() / ww.sum()) if ww.sum() > 0 else 0.0)
        else:
            out.append(float((wts * c).sum() / wts.sum()) if wts.sum() > 0 else 0.0)
    return np.array(out, float), wts


# ----------------------------------------------------------------------------
# XGBoost (M2)
# ----------------------------------------------------------------------------
def xgb_fail_prob(Xtr, ytr, Xte, seeds=SEEDS):
    imp = SimpleImputer(strategy='median')
    a = imp.fit_transform(Xtr)
    b = imp.transform(Xte)
    p = np.zeros(len(Xte))
    for s in seeds:
        model = xgb.XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.1, subsample=0.9,
                                  colsample_bytree=0.8, scale_pos_weight=1.0, objective='binary:logistic',
                                  eval_metric='auc', random_state=s, verbosity=0, n_jobs=1)
        model.fit(a, ytr)
        p += model.predict_proba(b)[:, 1]
    return p / len(seeds)


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def auc_fail(y, pass_score):
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


def pct_rank(x):
    x = np.asarray(x, float)
    return rankdata(x) / len(x)


def recall_precision_at(frame, score_col, frac=0.2, seed=7):
    """Within each semester flag the lowest ceil(frac*n) pass-scores; pool numerators/denominators."""
    rng = np.random.default_rng(seed)
    tp = 0
    flagged = 0
    fails = 0
    for _sem, g in frame.groupby('semester'):
        s = g[score_col].to_numpy(float)
        y = g['y'].to_numpy(int)
        k = int(math.ceil(frac * len(g)))
        order = np.lexsort((rng.random(len(g)), s))
        tp += int(y[order[:k]].sum())
        flagged += k
        fails += int(y.sum())
    return (tp / fails if fails else np.nan), (tp / flagged if flagged else np.nan)


def point_metrics(frame, score_col):
    y = frame['y'].to_numpy(int)
    s = frame[score_col].to_numpy(float)
    f = frame['final'].to_numpy(float)
    rho = float(sps.spearmanr(s, f)[0])
    rec, prec = recall_precision_at(frame, score_col)
    return {'n': len(frame), 'n_fail': int(y.sum()), 'r': pearson(s, f), 'rho': rho, 'auc': auc_fail(y, s),
            'recall20': rec, 'precision20': prec}


def macro_metrics(frame, score_col):
    rs, aucs, ns = [], [], []
    for _sem, g in frame.groupby('semester'):
        y = g['y'].to_numpy(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        rs.append(pearson(g[score_col], g['final']))
        aucs.append(auc_fail(y, g[score_col].to_numpy(float)))
        ns.append(len(g))
    ns = np.array(ns, float)
    z = np.arctanh(np.clip(rs, -0.999, 0.999))
    wz = ((ns - 3) * z).sum() / (ns - 3).sum()
    return {'macro_r': float(np.tanh(wz)), 'macro_auc': float((np.array(aucs) * ns).sum() / ns.sum())}


def jackknife_se(frame, score_col):
    sems = sorted(frame['semester'].unique())
    vals_r, vals_a = [], []
    for s in sems:
        g = frame[frame['semester'] != s]
        vals_r.append(pearson(g[score_col], g['final']))
        vals_a.append(auc_fail(g['y'].to_numpy(int), g[score_col].to_numpy(float)))
    k = len(sems)

    def se(v):
        v = np.array(v, float)
        return float(math.sqrt((k - 1) / k * ((v - v.mean()) ** 2).sum()))
    return se(vals_r), se(vals_a)


def calibration(frame):
    p = np.clip(frame['pfail'].to_numpy(float), 1e-6, 1 - 1e-6)
    y = frame['y'].to_numpy(int)
    brier = float(np.mean((p - y) ** 2))
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(np.log(p / (1 - p)).reshape(-1, 1), y)
    return brier, float(lr.coef_[0][0])


# ----------------------------------------------------------------------------
# cluster bootstrap (by anon_id), shared draws for paired comparisons
# ----------------------------------------------------------------------------
class Boot:
    def __init__(self, anon_ids, B=BOOT_B, seed=BOOT_SEED):
        uniq, inv = np.unique(np.asarray(anon_ids), return_inverse=True)
        self.rows = [np.where(inv == i)[0] for i in range(len(uniq))]
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(uniq), size=(B, len(uniq)))
        self.idx = [np.concatenate([self.rows[c] for c in d]) for d in draws]
        self.B = B

    def metric_dist(self, y, s, f):
        """Return arrays (B,) of r and AUC for score s."""
        y = np.asarray(y, int)
        s = np.asarray(s, float)
        f = np.asarray(f, float)
        rr = np.empty(self.B)
        aa = np.empty(self.B)
        for b, ix in enumerate(self.idx):
            sb = s[ix]
            fb = f[ix]
            yb = y[ix]
            rr[b] = np.corrcoef(sb, fb)[0, 1]
            aa[b] = auc_fail(yb, sb)
        return rr, aa


def ci(arr, level=0.95):
    a = (1 - level) / 2
    return float(np.nanpercentile(arr, 100 * a)), float(np.nanpercentile(arr, 100 * (1 - a)))
