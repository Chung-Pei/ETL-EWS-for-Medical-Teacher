# -*- coding: utf-8 -*-
"""H-02 supplementary: B1 (class only) and B2 (class + temporal prior failure rate). post hoc. Plan: H02B_plan.md.

Usage:  python h02_b1b2.py --out results
All predictors are PASS-scores (higher = safer); a risk r becomes score 1 - r.
"""
import argparse
import csv
import math
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import pa_common as C  # noqa: E402

L = C.L
D7 = C.D7
CLS = C.CLASS5
SEM_CODE = {s: i for i, s in enumerate(D7)}


# ----------------------------------------------------------------------------------------------------------------
# predictors
# ----------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------------------------
def _auc(y, s):
    return L.auc_fail(y, s)


def _wmean(y, s, semc):
    num = den = 0.0
    for k in range(len(D7)):
        m = semc == k
        yk = y[m]
        if yk.size == 0 or yk.sum() == 0 or yk.sum() == yk.size:
            continue
        num += _auc(yk, s[m]) * m.sum()
        den += m.sum()
    return num / den if den else np.nan


def point_auc(df, mask, score, with_class=True):
    y = df['y'].to_numpy(int)[mask]
    s = np.asarray(score, float)[mask]
    semc = np.array([SEM_CODE[v] for v in df['semester'].to_numpy()[mask]])
    out = {'pooled': _auc(y, s), 'wmean': _wmean(y, s, semc)}
    if with_class:
        cl = df['program5_key'].to_numpy()[mask]
        for c in CLS:
            m = cl == c
            out['class:' + c] = _auc(y[m], s[m]) if m.sum() else np.nan
    return out


def boot_scores(df, boot, mask, scores, with_class=True):
    """Bootstrap distributions (shared cluster draws; the Boot is built on all 2,652 rows, subsets restrict rows).
    scores: name -> full-length array. Returns {(name, scope): array(B)}."""
    y_all = df['y'].to_numpy(int)
    semc_all = np.array([SEM_CODE[v] for v in df['semester'].to_numpy()])
    cls_all = df['program5_key'].to_numpy()
    B = boot.B
    res = {(n, sc): np.full(B, np.nan) for n in scores for sc in ['pooled', 'wmean'] + (['class:' + c for c in CLS] if with_class else [])}
    for b, ix in enumerate(boot.idx):
        ix = ix[mask[ix]]
        yb = y_all[ix]
        sb_ = semc_all[ix]
        cb = cls_all[ix]
        for name, sc in scores.items():
            s = sc[ix]
            res[(name, 'pooled')][b] = _auc(yb, s)
            res[(name, 'wmean')][b] = _wmean(yb, s, sb_)
            if with_class:
                for c in CLS:
                    m = cb == c
                    if m.any():
                        res[(name, 'class:' + c)][b] = _auc(yb[m], s[m])
    return res


def ci_(a):
    return L.ci(np.asarray(a, float))


def top20_flags(frame, score_col, frac=0.2, seed=7):
    """Index labels of the top-20% (lowest pass-score) list; replicates ews_lib.recall_precision_at (same RNG flow)."""
    rng = np.random.default_rng(seed)
    out = []
    for _sem, g in frame.groupby('semester'):
        s = g[score_col].to_numpy(float)
        k = int(math.ceil(frac * len(g)))
        order = np.lexsort((rng.random(len(g)), s))
        out.extend(g.index[order[:k]].tolist())
    return out


def recall_prec(frame, score_col):
    rec, prec = L.recall_precision_at(frame, score_col)
    return rec, prec


def verdict(delta, lo, hi):
    if delta is None or np.isnan(delta) or np.isnan(lo):
        return 'n/a'
    if hi < 0:
        return 'M3 worse than baseline (CI < 0)'
    if lo <= 0 <= hi or delta < 0.02:
        return 'no substantive increment (CI includes 0 or delta < 0.02)'
    if delta >= 0.03 and lo > 0:
        return 'increment holds (delta >= 0.03, CI excludes 0)'
    return 'gray zone (0.02 <= delta < 0.03, CI excludes 0): criteria silent'


# ----------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, 'run.log'), 'w', encoding='utf-8')

    def log(*x):
        msg = ' '.join(str(v) for v in x)
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    import xgboost
    import scipy
    import sklearn
    log('# H-02 B1/B2 run (post hoc)  utc=%s' % time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    log('python', platform.python_version(), 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__,
        'numpy', np.__version__, 'pandas', pd.__version__, 'scipy', scipy.__version__)
    log('boot B=%d seed=%d (one Boot on all 2,652 rows; subsets restrict rows)' % (L.BOOT_B, L.BOOT_SEED))
    for f in ['oof_scores_no_ids.csv', 'class_membership5.csv', 'd7_semester_scores.csv']:
        log('sha256', f, C.sha256(os.path.join(C.IN, f)))
    for f in ['H02B_plan.md', 'h02_b1b2.py']:
        log('sha256', f, C.sha256(os.path.join(HERE, f)))

    df = C.frozen_frame(8)
    n = len(df)
    ar = np.ones(n, bool)
    r1, tab1 = b1_risk(df)
    r1s1, tab1s1 = b1_risk(df, exclude_same_student=False)
    r2, tab2 = b2_risk(df)
    df['b1'] = 1.0 - r1
    df['b1_s1'] = 1.0 - r1s1
    df['b2'] = 1.0 - r2
    tab1.to_csv(os.path.join(a.out, 'H02B_class_rates_B1_by_fold.csv'), index=False, encoding='utf-8-sig')
    tab2.to_csv(os.path.join(a.out, 'H02B_class_rates_B2_by_semester.csv'), index=False, encoding='utf-8-sig')
    log('B1 fallback cells:', tab1[tab1.fallback_to_fold_overall == 1][['test_semester', 'class', 'n_test_in_class']].query('n_test_in_class>0').to_dict('records'))
    log('B2 fallback cells:', tab2[(tab2.fallback_to_prior_overall == 1) & (tab2.n_test_in_class > 0)][['semester', 'class', 'n_test_in_class']].to_dict('records'))

    # ---- evaluation sets (plan section 4) ---------------------------------------------------------------------
    sem = df['semester'].to_numpy()
    cls = df['program5_key'].to_numpy()
    in_b2 = sem != '1111'
    nr = cls != 'retake'
    classprior = in_b2 & ~((sem == '1112') & (cls == '4yr')) & ~((sem == '1141') & (cls == 'retake'))
    sets = {
        'E_all': (ar, ["M3'", 'B1', 'B1_S1']),
        'E_noretake': (nr, ["M3'", 'B1']),
        'E_b2': (in_b2, ["M3'", 'B1', 'B2']),
        'E_b2_noretake': (in_b2 & nr, ["M3'", 'B1', 'B2']),
        'E_b2_classprior': (classprior, ["M3'", 'B1', 'B2']),
    }
    log('eval sizes:', {k: int(v[0].sum()) for k, v in sets.items()})
    assert int(in_b2.sum()) == 2180 and int(classprior.sum()) == 1907
    SC = {"M3'": df['m3'].to_numpy(float), 'B1': df['b1'].to_numpy(float), 'B1_S1': df['b1_s1'].to_numpy(float),
          'B2': df['b2'].to_numpy(float)}
    boot = L.Boot(df['anon_id'].to_numpy())

    metrics, delta_rows, class_rows = [], [], []
    for sname, (mask, preds) in sets.items():
        t0 = time.time()
        sub = df[mask]
        nf = int(sub['y'].sum())
        base = nf / len(sub)
        scores = {p: SC[p] for p in preds}
        bd = boot_scores(df, boot, mask, scores)
        for p in preds:
            pt = point_auc(df, mask, SC[p])
            fr = df.assign(_s=SC[p])[mask]
            rec, prec = recall_prec(fr, '_s')
            row = {'predictor': p, 'eval_set': sname, 'n': len(sub), 'n_fail': nf, 'base_rate': base,
                   'auc_pooled': pt['pooled'], 'auc_pooled_lo': ci_(bd[(p, 'pooled')])[0], 'auc_pooled_hi': ci_(bd[(p, 'pooled')])[1],
                   'auc_wmean': pt['wmean'], 'auc_wmean_lo': ci_(bd[(p, 'wmean')])[0], 'auc_wmean_hi': ci_(bd[(p, 'wmean')])[1],
                   'recall20': rec, 'precision20': prec, 'lift20': prec / base}
            metrics.append(row)
            for c in CLS:
                m = (cls == c) & mask
                class_rows.append({'predictor': p, 'eval_set': sname, 'class': c, 'n': int(m.sum()),
                                   'n_fail': int(df.loc[m, 'y'].sum()), 'auc': pt['class:' + c],
                                   'auc_lo': ci_(bd[(p, 'class:' + c)])[0], 'auc_hi': ci_(bd[(p, 'class:' + c)])[1]})
        for p in preds:
            if p == "M3'":
                continue
            pt_m = point_auc(df, mask, SC["M3'"])
            pt_p = point_auc(df, mask, SC[p])
            for scope in ['pooled', 'wmean'] + ['class:' + c for c in CLS]:
                d = pt_m[scope] - pt_p[scope]
                dist = bd[("M3'", scope)] - bd[(p, scope)]
                lo, hi = ci_(dist)
                delta_rows.append({'comparison': f"M3' - {p}", 'eval_set': sname, 'scope': scope, 'n': len(sub),
                                   'n_fail': nf, 'auc_m3': pt_m[scope], 'auc_baseline': pt_p[scope], 'delta_auc': d,
                                   'ci_lo': lo, 'ci_hi': hi, 'verdict': verdict(d, lo, hi)})
        log(f'{sname}: n={len(sub)} done in {time.time() - t0:.0f}s')

    M = pd.DataFrame(metrics)
    D = pd.DataFrame(delta_rows)
    K = pd.DataFrame(class_rows)
    M.to_csv(os.path.join(a.out, 'H02B_metrics.csv'), index=False, encoding='utf-8-sig')
    D.to_csv(os.path.join(a.out, 'H02B_delta_auc.csv'), index=False, encoding='utf-8-sig')
    K.to_csv(os.path.join(a.out, 'H02B_auc_by_class.csv'), index=False, encoding='utf-8-sig')

    # ---- per-semester AUC (point) -------------------------------------------------------------------------------
    rows = []
    for p in ["M3'", 'B1', 'B2']:
        for s in D7:
            m = (sem == s) & (in_b2 if p == 'B2' else ar)
            if not m.any():
                continue
            y = df.loc[m, 'y'].to_numpy(int)
            rows.append({'predictor': p, 'semester': s, 'n': int(m.sum()), 'n_fail': int(y.sum()),
                         'auc': _auc(y, SC[p][m]) if 0 < y.sum() < len(y) else np.nan,
                         'n_distinct_scores': int(len(np.unique(SC[p][m])))})
    pd.DataFrame(rows).to_csv(os.path.join(a.out, 'H02B_auc_by_semester.csv'), index=False, encoding='utf-8-sig')

    # ---- list composition ----------------------------------------------------------------------------------------
    comp = []
    for sname in ['E_all', 'E_b2']:
        mask = sets[sname][0]
        sub = df[mask].copy()
        for p in sets[sname][1]:
            if p == 'B1_S1':
                continue
            sub['_s'] = SC[p][mask]
            fl = top20_flags(sub, '_s')
            lst = sub.loc[fl]
            for c in CLS:
                nl = int((lst['program5_key'] == c).sum())
                npop = int((sub['program5_key'] == c).sum())
                comp.append({'eval_set': sname, 'predictor': p, 'class': c, 'list_n': nl, 'list_total': len(lst),
                             'share_of_list': nl / len(lst), 'pop_n': npop, 'share_of_pop': npop / len(sub),
                             'fail_rate_in_list_class': float(lst.loc[lst['program5_key'] == c, 'y'].mean()) if nl else np.nan,
                             'list_recall': float(lst['y'].sum() / sub['y'].sum()), 'list_precision': float(lst['y'].mean())})
    pd.DataFrame(comp).to_csv(os.path.join(a.out, 'H02B_list_composition.csv'), index=False, encoding='utf-8-sig')

    # ---- S4: outcome from d7_semester_scores.csv (point estimates only) ---------------------------------------
    ss = pd.read_csv(os.path.join(C.IN, 'd7_semester_scores.csv'), dtype={'anon_id': str, 'semester': str})
    d4 = df.merge(ss[['anon_id', 'semester', 'semester_score']], on=['anon_id', 'semester'], how='left')
    assert not d4['semester_score'].isna().any()
    d4['y'] = (d4['semester_score'] < 60).astype(int)
    r1_4, _ = b1_risk(d4)
    r2_4, _ = b2_risk(d4)
    s4 = []
    for name, sc, mask in [("M3'", d4['m3'].to_numpy(float), ar), ('B1', 1 - r1_4, ar), ("M3' (E_b2)", d4['m3'].to_numpy(float), in_b2), ('B2', 1 - r2_4, in_b2)]:
        pt = point_auc(d4, mask, sc, with_class=False)
        s4.append({'sensitivity': 'S4_outcome_from_d7_file', 'predictor': name, 'n_fail': int(d4.loc[mask, 'y'].sum()),
                   'auc_pooled': pt['pooled'], 'auc_wmean': pt['wmean']})
    pd.DataFrame(s4).to_csv(os.path.join(a.out, 'H02B_sensitivity_S4.csv'), index=False, encoding='utf-8-sig')
    log('S4:', s4)

    # ---- baseline-metrics file in the existing H02 format ---------------------------------------------------------
    existing = os.path.join(C.PACK, '05_plans_and_prior', 'H02_baseline_metrics.csv')
    with open(existing, encoding='utf-8-sig', newline='') as f:
        rd = list(csv.reader(f))
    header, body = rd[0], rd[1:]
    names = {'B1': 'B1 class only', 'B2': 'B2 class + prior failure rate', "M3'": "M3'"}

    def fmt(v):
        return '' if v is None or (isinstance(v, float) and np.isnan(v)) else repr(float(v)) if isinstance(v, (float, np.floating)) else str(v)

    def mk(p, sname, subset_label):
        r = M[(M.predictor == p) & (M.eval_set == sname)].iloc[0]
        d = {'predictor': names[p], 'subset': subset_label, 'n': float(r.n), 'n_fail': float(r.n_fail),
             'auc_pooled': r.auc_pooled, 'auc_pooled_lo': r.auc_pooled_lo, 'auc_pooled_hi': r.auc_pooled_hi,
             'auc_wmean': r.auc_wmean, 'auc_wmean_lo': r.auc_wmean_lo, 'auc_wmean_hi': r.auc_wmean_hi,
             'recall20': r.recall20, 'precision20': r.precision20, 'lift20': r.lift20, 'status': 'run'}
        return [fmt(d.get(h, '')) for h in header]

    out = []
    for r in body:
        if r[0].startswith('B1 class only'):
            out.append(mk('B1', 'E_all', 'all'))
        elif r[0].startswith('B2 class + prior'):
            out.append(mk('B2', 'E_b2', 'eval_1112_1141'))
        else:
            out.append(r)
    extra = [mk('B1', 'E_noretake', 'excl_retake'), mk('B2', 'E_b2_noretake', 'eval_1112_1141_excl_retake'),
             mk("M3'", 'E_b2', 'eval_1112_1141 (same set as B2)'), mk('B1', 'E_b2', 'eval_1112_1141 (same set as B2)'),
             mk('B2', 'E_b2_classprior', 'S3_classprior_only'), mk("M3'", 'E_b2_classprior', 'S3_classprior_only')]
    with open(os.path.join(a.out, 'H02_baseline_metrics_filled.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(out + extra)
    log('done')
    logf.close()


if __name__ == '__main__':
    main()
