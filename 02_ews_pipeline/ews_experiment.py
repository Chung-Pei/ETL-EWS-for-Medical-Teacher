# -*- coding: utf-8 -*-
"""EWS timing experiment -- main analysis (Spec v1.2 + amendment v1.3).

Usage (PowerShell):
  python ews_experiment.py --etl-dir etl_orig --root . --out results --stage main
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

import ews_lib as L

FSB_WEEKS = [10, 12, 14, 16, 0]
FSC_WEEKS = [6, 8, 10, 12, 14, 16, 0]
WLABEL = {0: 'Full'}


def wl(w):
    return WLABEL.get(w, f'W{w}')


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_all(etl, root, weeks, tag_fmt=None):
    """data[W][sem] -> records (with final), restricted to D7 + 1142; sorted by anon_id for canonical alignment."""
    data = {}
    for W in weeks:
        recs = L.load_week(etl, root, W, None if tag_fmt is None else tag_fmt.format(W=W))
        by = {}
        for r in recs:
            if r.get('final') is None or r['semester'] not in L.D7 + ['1142']:
                continue
            by.setdefault(r['semester'], []).append(r)
        for s in by:
            by[s].sort(key=lambda r: r['anon_id'])
        data[W] = by
    return data


def check_alignment(data):
    ref = None
    for W, by in data.items():
        keys = [(s, r['anon_id']) for s in L.D7 for r in by.get(s, [])]
        if ref is None:
            ref = keys
        assert keys == ref, f'record alignment differs at W={W}'
    return len(ref)


# ----------------------------------------------------------------------------
# cross validation
# ----------------------------------------------------------------------------
def fold_defs(arm):
    if arm == 'P':
        return [(s, [t for t in L.D7 if t != s]) for s in L.D7]
    if arm == 'S':
        return [(s, [t for t in (L.UPPER if s in L.UPPER else L.LOWER) if t != s]) for s in L.D7]
    if arm == 'FWD':  # S1: forward chaining, at least 2 earlier semesters
        return [(L.D7[i], L.D7[:i]) for i in range(2, len(L.D7))]
    raise ValueError(arm)


def run_folds(etl, data, folds, feats, with_midterm, models, data_train=None, exclude=frozenset(),
              label=None, nan_variant=False, permute_seed=None, extra_test=None):
    """Return a DataFrame of out-of-fold scores (pass-propensity: higher = better)."""
    data_train = data if data_train is None else data_train
    out = []
    removed = {}
    for test_sem, train_sems in folds:
        test = [r for r in (extra_test or data)[test_sem] if (r['anon_id'], r['semester']) not in exclude]
        test_ids = {r['anon_id'] for r in test}
        train = [r for t in train_sems for r in data_train[t]
                 if r['anon_id'] not in test_ids and (r['anon_id'], r['semester']) not in exclude]
        removed[test_sem] = sum(len(data_train[t]) for t in train_sems) - len(train)
        if not train or not test:
            continue
        ytr = np.array([r['is_fail'] if label is None else label(r) for r in train], int)
        if permute_seed is not None:
            ytr = np.random.default_rng(permute_seed).permutation(ytr)
        yte = np.array([int(r['final'] < 60) if label is None else label(r) for r in test], int)
        fr = pd.DataFrame({'anon_id': [r['anon_id'] for r in test], 'semester': test_sem,
                           'stratum': L.STRATUM[test_sem], 'final': [r['final'] for r in test], 'y': yte})
        if 'm0' in models:
            fr['m0'] = [float(r['midterm']) if r.get('midterm') is not None else np.nan for r in test]
        need_bas = any(k in models for k in ('m1', 'm3b'))
        if need_bas:
            btr, bte, stats_tr, qn = L.bas_fold(etl, train, test, include_midterm=with_midterm)
            fr['m1'] = bte
            fr['qmi_none'] = qn
        if 'm2' in models or 'm3b' in models:
            ff = [f for f in feats if not (f == 'is_4tech' and models.get('const_is4tech'))]
            Xtr = L.build_matrix(train, ff, nan_variant)
            Xte = L.build_matrix(test, ff, nan_variant)
            if 'm2' in models:
                pf = L.xgb_fail_prob(Xtr, ytr, Xte)
                fr['pfail'] = pf
                fr['m2'] = 1.0 - pf
            if 'm3b' in models:
                pf2 = L.xgb_fail_prob(np.hstack([Xtr, btr.reshape(-1, 1)]), ytr, np.hstack([Xte, bte.reshape(-1, 1)]))
                fr['m3b'] = 1.0 - pf2
        if 'm1dep' in models:
            fr['m1dep'] = L.bas_deployed(etl, train, test)
        if 'm1refit' in models:
            fr['m1refit'], _ = L.bas_refit(etl, train, test, stats_tr, with_midterm)
        out.append(fr)
    df = pd.concat(out, ignore_index=True)
    df.attrs['removed'] = removed
    return df


def add_m3(df, by_stratum=False):
    """M3 = mean percentile of M1 and M2 (spec 5.3). Percentiles are label-free."""
    df = df.copy()
    if 'm1' not in df or 'm2' not in df:
        return df
    if by_stratum:
        df['m3'] = np.nan
        for _st, ix in df.groupby('stratum').groups.items():
            g = df.loc[ix]
            df.loc[ix, 'm3'] = (L.pct_rank(g['m1']) + L.pct_rank(g['m2'])) / 2
    else:
        df['m3'] = (L.pct_rank(df['m1']) + L.pct_rank(df['m2'])) / 2
    return df


def canon(df):
    return df.sort_values(['semester', 'anon_id']).reset_index(drop=True)


# ----------------------------------------------------------------------------
# main analysis
# ----------------------------------------------------------------------------
def build_main_frames(etl, data, cache_path=None, log=print):
    if cache_path and os.path.exists(cache_path):
        return pickle.load(open(cache_path, 'rb'))
    F = {}
    t0 = time.time()
    # --- P arm, FS-B -----------------------------------------------------------
    for W in FSB_WEEKS:
        df = run_folds(etl, data[W], fold_defs('P'), etl.FS_B, True, {'m0': 1, 'm1': 1, 'm2': 1, 'm3b': 1})
        F[('P', 'B', W)] = canon(add_m3(df))
        log(f'  P/FS-B/{wl(W)} done ({time.time() - t0:.0f}s)')
    # --- P arm, FS-C -----------------------------------------------------------
    for W in FSC_WEEKS:
        df = run_folds(etl, data[W], fold_defs('P'), etl.FS_C, False, {'m1': 1, 'm2': 1})
        F[('P', 'C', W)] = canon(add_m3(df))
        log(f'  P/FS-C/{wl(W)} done ({time.time() - t0:.0f}s)')
    # --- S arm (is_4tech is constant within a stratum -> dropped) ---------------
    for W in FSB_WEEKS:
        df = run_folds(etl, data[W], fold_defs('S'), etl.FS_B, True, {'m0': 1, 'm1': 1, 'm2': 1, 'const_is4tech': 1})
        F[('S', 'B', W)] = canon(add_m3(df, by_stratum=True))
        log(f'  S/FS-B/{wl(W)} done ({time.time() - t0:.0f}s)')
    for W in FSC_WEEKS:
        df = run_folds(etl, data[W], fold_defs('S'), etl.FS_C, False, {'m1': 1, 'm2': 1, 'const_is4tech': 1})
        F[('S', 'C', W)] = canon(add_m3(df, by_stratum=True))
        log(f'  S/FS-C/{wl(W)} done ({time.time() - t0:.0f}s)')
    if cache_path:
        pickle.dump(F, open(cache_path, 'wb'))
    return F


def model_cols(fs, arm):
    if fs == 'B':
        cols = {'M0': 'm0', 'M1': 'm1', 'M2': 'm2', 'M3': 'm3'}
        if arm == 'P':
            cols['M3b'] = 'm3b'
        return cols
    return {"M1'": 'm1', "M2'": 'm2', "M3'": 'm3'}


def summarize(F, log=print):
    """Point metrics, bootstrap CIs, jackknife SE, macro metrics for every (arm, fs, W, model)."""
    ref = F.get(('P', 'B', 12))
    if ref is None:
        ref = next(iter(F.values()))
    boot = L.Boot(ref['anon_id'])
    rows, bdist = [], {}
    for (arm, fs, W), df in sorted(F.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] if kv[0][2] else 99)):
        assert (df['anon_id'].to_numpy() == ref['anon_id'].to_numpy()).all()
        qn = float(df['qmi_none'].mean()) if 'qmi_none' in df else np.nan
        for mname, col in model_cols(fs, arm).items():
            pm = L.point_metrics(df, col)
            mm = L.macro_metrics(df, col)
            jr, ja = L.jackknife_se(df, col)
            rr, aa = boot.metric_dist(df['y'], df[col], df['final'])
            bdist[(arm, fs, W, mname)] = (rr, aa)
            rlo, rhi = L.ci(rr)
            alo, ahi = L.ci(aa)
            row = {'arm': arm, 'feature_set': fs, 'W': wl(W), 'W_num': W if W else 18, 'model': mname,
                   **pm, 'r_lo': rlo, 'r_hi': rhi, 'auc_lo': alo, 'auc_hi': ahi, **mm,
                   'jack_se_r': jr, 'jack_se_auc': ja, 'qmi_none_share': qn if col in ('m1', 'm3') else np.nan}
            if col == 'm2':
                row['brier'], row['calib_slope'] = L.calibration(df)
            rows.append(row)
        log(f'  summarized {arm}/FS-{fs}/{wl(W)}')
    T = pd.DataFrame(rows)
    # relative predictive power vs W16
    T['rpp_r'] = np.nan
    T['rpp_auc'] = np.nan
    for (arm, fs, model), g in T.groupby(['arm', 'feature_set', 'model']):
        base = g[g['W'] == 'W16']
        if base.empty:
            continue
        b_r, b_a = float(base['r'].iloc[0]), float(base['auc'].iloc[0])
        T.loc[g.index, 'rpp_r'] = 100 * g['r'] / b_r
        T.loc[g.index, 'rpp_auc'] = 100 * (g['auc'] - 0.5) / (b_a - 0.5)
    return T, bdist, boot


def contrasts(bdist, T):
    """Confirmatory C1-C7 (99.29% CI) and exploratory paired differences (95% CI)."""
    def d(key_a, key_b, which):
        ra, aa = bdist[key_a]
        rb, ab = bdist[key_b]
        return (ra - rb) if which == 'r' else (aa - ab)

    def pt(key, which):
        arm, fs, W, m = key
        w = wl(W)
        row = T[(T.arm == arm) & (T.feature_set == fs) & (T.W == w) & (T.model == m)].iloc[0]
        return float(row['r' if which == 'r' else 'auc'])

    out = []

    def add(name, ka, kb, which, conf, level):
        dist = d(ka, kb, which)
        lo, hi = L.ci(dist, level)
        out.append({'id': name, 'confirmatory': conf, 'metric': which, 'a': str(ka), 'b': str(kb),
                    'delta': pt(ka, which) - pt(kb, which), 'ci_lo': lo, 'ci_hi': hi, 'ci_level': level})

    P = 'P'
    conf = [
        ('C1', (P, 'B', 12, 'M2'), (P, 'B', 12, 'M1')),
        ('C2', (P, 'B', 12, 'M3'), (P, 'B', 12, 'M2')),
        ('C3', (P, 'B', 10, 'M2'), (P, 'B', 10, 'M0')),
        ('C4', (P, 'B', 12, 'M3'), (P, 'B', 10, 'M3')),
        ('C5', (P, 'B', 14, 'M3'), (P, 'B', 12, 'M3')),
        ('C6', (P, 'B', 16, 'M3'), (P, 'B', 14, 'M3')),
    ]
    for name, ka, kb in conf:
        add(name, ka, kb, 'auc', True, 1 - 0.05 / 7)
    # C7: separate-arm M2 (concatenated) minus pooled-arm M2, W12
    add('C7', ('S', 'B', 12, 'M2'), ('P', 'B', 12, 'M2'), 'auc', True, 1 - 0.05 / 7)
    # exploratory: model contrasts within each (arm, fs, W); adjacent-W contrasts per model
    for (arm, fs, W, m) in list(bdist):
        pass
    keys = list(bdist)
    for arm in ('P', 'S'):
        for fs in ('B', 'C'):
            ms = ['M0', 'M1', 'M2', 'M3', 'M3b'] if fs == 'B' else ["M1'", "M2'", "M3'"]
            ws = FSB_WEEKS if fs == 'B' else FSC_WEEKS
            for W in ws:
                for a, b in [(ms[i], ms[j]) for i in range(len(ms)) for j in range(i)]:
                    ka, kb = (arm, fs, W, a), (arm, fs, W, b)
                    if ka in bdist and kb in bdist:
                        for which in ('r', 'auc'):
                            add(f'X_model_{arm}_{fs}_{wl(W)}_{a}-{b}', ka, kb, which, False, 0.95)
            for m in ms:
                wl_sorted = [w for w in ws if w]
                pairs = list(zip(wl_sorted[1:], wl_sorted[:-1])) + [(0, 16)]
                for wa, wb in pairs:
                    ka, kb = (arm, fs, wa, m), (arm, fs, wb, m)
                    if ka in bdist and kb in bdist:
                        for which in ('r', 'auc'):
                            add(f'X_week_{arm}_{fs}_{m}_{wl(wa)}-{wl(wb)}', ka, kb, which, False, 0.95)
    # separate vs pooled by model (W grid), AUC and r
    for fs in ('B', 'C'):
        ms = ['M1', 'M2', 'M3'] if fs == 'B' else ["M1'", "M2'", "M3'"]
        for W in (FSB_WEEKS if fs == 'B' else FSC_WEEKS):
            for m in ms:
                ka, kb = ('S', fs, W, m), ('P', fs, W, m)
                if ka in bdist and kb in bdist:
                    for which in ('r', 'auc'):
                        add(f'X_sep_vs_pooled_{fs}_{wl(W)}_{m}', ka, kb, which, False, 0.95)
    return pd.DataFrame(out)


def wstar(T, bdist):
    """Spec 9.5: smallest W with r(W) >= 0.95*r(W16) and upper 95% CI of AUC(W16)-AUC(W) <= 0.02."""
    out = []
    for arm in ('P', 'S'):
        for fs, ms, ws in (('B', ['M1', 'M2', 'M3'], [10, 12, 14, 16]),
                           ('C', ["M1'", "M2'", "M3'"], [6, 8, 10, 12, 14, 16])):
            for m in ms:
                chosen = None
                detail = []
                for W in ws:
                    row = T[(T.arm == arm) & (T.feature_set == fs) & (T.W == wl(W)) & (T.model == m)]
                    base = T[(T.arm == arm) & (T.feature_set == fs) & (T.W == 'W16') & (T.model == m)]
                    if row.empty or base.empty:
                        continue
                    ra = float(row['r'].iloc[0]) >= 0.95 * float(base['r'].iloc[0])
                    dd = bdist[(arm, fs, 16, m)][1] - bdist[(arm, fs, W, m)][1]
                    hi = L.ci(dd)[1]
                    rb = hi <= 0.02
                    detail.append((W, bool(ra), round(hi, 4), bool(rb)))
                    if chosen is None and ra and rb:
                        chosen = W
                out.append({'arm': arm, 'feature_set': fs, 'model': m, 'W_star': chosen, 'detail(W,a,ci_hi,b)': str(detail)})
    return pd.DataFrame(out)


def per_semester(F):
    rows = []
    for (arm, fs, W), df in F.items():
        for mname, col in model_cols(fs, arm).items():
            for sem, g in df.groupby('semester'):
                y = g['y'].to_numpy(int)
                rows.append({'arm': arm, 'feature_set': fs, 'W': wl(W), 'model': mname, 'semester': sem,
                             'n': len(g), 'n_fail': int(y.sum()),
                             'r': L.pearson(g[col], g['final']), 'auc': L.auc_fail(y, g[col].to_numpy(float))})
    return pd.DataFrame(rows)


def strata_table(F, boot_cache={}):
    """Pooled vs separate arms within each stratum (M1, M2, M3)."""
    rows = []
    for (arm, fs, W), df in F.items():
        for st in ('upper', 'lower'):
            g = df[df.stratum == st].reset_index(drop=True)
            for mname, col in model_cols(fs, arm).items():
                if mname in ('M0', 'M3b'):
                    continue
                y = g['y'].to_numpy(int)
                rows.append({'arm': arm, 'feature_set': fs, 'W': wl(W), 'stratum': st, 'model': mname, 'n': len(g),
                             'n_fail': int(y.sum()), 'r': L.pearson(g[col], g['final']),
                             'auc': L.auc_fail(y, g[col].to_numpy(float))})
    T = pd.DataFrame(rows)
    # paired S - P with bootstrap CI
    out = []
    for (fs, W, st, m), g in T.groupby(['feature_set', 'W', 'stratum', 'model']):
        s = g[g.arm == 'S']
        p = g[g.arm == 'P']
        if s.empty or p.empty:
            continue
        Wn = 0 if W == 'Full' else int(W[1:])
        dfs = F[('S', fs, Wn)]
        dfp = F[('P', fs, Wn)]
        col = model_cols(fs, 'S')[m]
        gs = dfs[dfs.stratum == st].reset_index(drop=True)
        gp = dfp[dfp.stratum == st].reset_index(drop=True)
        b = L.Boot(gs['anon_id'])
        rs, as_ = b.metric_dist(gs['y'], gs[col], gs['final'])
        rp, ap = b.metric_dist(gp['y'], gp[col], gp['final'])
        out.append({'feature_set': fs, 'W': W, 'stratum': st, 'model': m, 'n': int(s['n'].iloc[0]),
                    'auc_pooled_arm': float(p['auc'].iloc[0]), 'auc_separate_arm': float(s['auc'].iloc[0]),
                    'auc_diff_S_minus_P': float(s['auc'].iloc[0] - p['auc'].iloc[0]),
                    'auc_diff_lo': L.ci(as_ - ap)[0], 'auc_diff_hi': L.ci(as_ - ap)[1],
                    'r_pooled_arm': float(p['r'].iloc[0]), 'r_separate_arm': float(s['r'].iloc[0]),
                    'r_diff_S_minus_P': float(s['r'].iloc[0] - p['r'].iloc[0]),
                    'r_diff_lo': L.ci(rs - rp)[0], 'r_diff_hi': L.ci(rs - rp)[1]})
    return T, pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--etl-dir', default='etl_orig')
    ap.add_argument('--root', default='.')
    ap.add_argument('--out', default='results')
    ap.add_argument('--stage', default='main', choices=['main'])
    ap.add_argument('--cache', default='')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    etl = L.Etl(a.etl_dir)
    L.init_bas(etl)
    weeks = sorted(set(FSB_WEEKS + FSC_WEEKS))
    data = load_all(etl, a.root, weeks)
    n = check_alignment(data)
    print(f'records aligned across W: n={n}')
    F = build_main_frames(etl, data, a.cache or None)
    T, bdist, boot = summarize(F)
    C = contrasts(bdist, T)
    WS = wstar(T, bdist)
    PS = per_semester(F)
    T.to_csv(f'{a.out}/table_main.csv', index=False, encoding='utf-8-sig')
    C.to_csv(f'{a.out}/table_contrasts.csv', index=False, encoding='utf-8-sig')
    WS.to_csv(f'{a.out}/table_wstar.csv', index=False, encoding='utf-8-sig')
    PS.to_csv(f'{a.out}/table_per_semester.csv', index=False, encoding='utf-8-sig')
    pickle.dump(F, open(f'{a.out}/frames.pkl', 'wb'))
    print('core tables written')
    ST, SC = strata_table(F)
    ST.to_csv(f'{a.out}/table_strata_point.csv', index=False, encoding='utf-8-sig')
    SC.to_csv(f'{a.out}/table_strata_sep_vs_pooled.csv', index=False, encoding='utf-8-sig')
    oof = pd.concat([df.assign(arm=k[0], feature_set=k[1], W=wl(k[2])) for k, df in F.items()], ignore_index=True)
    oof.drop(columns=['anon_id']).to_csv(f'{a.out}/oof_scores_no_ids.csv', index=False, encoding='utf-8-sig')
    pickle.dump(F, open(f'{a.out}/frames.pkl', 'wb'))
    print('main stage written to', a.out)


if __name__ == '__main__':
    main()
