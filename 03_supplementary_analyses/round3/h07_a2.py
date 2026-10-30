# -*- coding: utf-8 -*-
"""H-07 gates A2a (feature drift table) and A2 (event-rebuild AUC anchor). post hoc. Plan: H07_addendum2_plan.md.
Usage: python h07_a2.py --out results"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import h07_rebuild as R  # noqa: E402

C = R.C
L = R.L
ANCHORS = {"M2'": 0.7092, "M3'": 0.727}
TOL = 0.005
A1B = {"M1'": 0.693305278612401, "M2'": 0.704679817672767, "M3'": 0.7252727226167713}
NUM = ['quz_first_attempt_accuracy', 'quz_final_accuracy', 'quz_score_delta', 'quz_total_attempts', 'quz_pass_rate',
       'quz_cramming_ratio', 'consistency_score', 'pre_exam_intensity', 'reading_inflation_ratio', 'total_learning_minutes',
       'active_weeks', 'weekly_minutes_std', 'cram_pattern_score', 'late_night_ratio', 'early_start_ratio',
       'MM_ratio', 'MQ_ratio', 'QM_ratio', 'QQ_ratio']
CAT = ['policy_gaming_flag', 's_cluster']
FSC_OR_BAS = {  # where each compared item is used downstream
    'quz_first_attempt_accuracy': 'FS-C, BAS', 'quz_final_accuracy': 'FS-C, BAS', 'quz_score_delta': 'FS-C, BAS',
    'quz_total_attempts': 'FS-C, BAS', 'quz_pass_rate': 'FS-C', 'quz_cramming_ratio': 'FS-C, BAS',
    'consistency_score': 'FS-C', 'pre_exam_intensity': 'BAS', 'reading_inflation_ratio': 'FS-C',
    'total_learning_minutes': 'FS-C', 'policy_gaming_flag': 'FS-C', 'active_weeks': 'FS-C',
    'weekly_minutes_std': 'FS-C', 'cram_pattern_score': 'FS-C', 'late_night_ratio': 'FS-C',
    'early_start_ratio': 'FS-C', 'MM_ratio': 'FS-C', 'MQ_ratio': 'FS-C', 'QM_ratio': 'FS-C', 'QQ_ratio': 'FS-C',
    's_cluster': 'FS-C (s_cluster_encoded)'}


def assemble(feat, time_, lsa, ratios, keys):
    """One row per analysis record with every compared item."""
    f = feat.set_index(['anon_id', 'semester'])
    t = time_.set_index(['anon_id', 'semester'])
    rows = []
    for a, s in keys:
        row = {'anon_id': a, 'semester': s}
        for k in R.FEATS:
            row[k] = f.loc[(a, s), k]
        for k in R.TIME5:
            row[k] = t.loc[(a, s), k]
        ck = f'{a}||{s}'
        rr = ratios.get(ck, {})
        for k in ['MM_ratio', 'MQ_ratio', 'QM_ratio', 'QQ_ratio']:
            row[k] = rr.get(k)
        row['s_cluster'] = lsa.get(ck)
        rows.append(row)
    return pd.DataFrame(rows)


def drift_table(fz, rb, by=None):
    out = []
    groups = [('all', fz.index)] if by is None else [(g, ix) for g, ix in fz.groupby(by).groups.items()]
    for gname, ix in groups:
        for k in NUM + CAT:
            a, b = fz.loc[ix, k], rb.loc[ix, k]
            row = {'group': gname, 'feature': k, 'used_in': FSC_OR_BAS[k], 'n': len(ix)}
            if k in CAT:
                both_na = a.isna() & b.isna()
                eq = (a.astype(object) == b.astype(object)) | both_na
                row.update({'n_comparable': int((~(a.isna() | b.isna())).sum()), 'exact_share': float(eq.mean()),
                            'mean_abs_diff': np.nan, 'max_abs_diff': np.nan, 'pearson_r': np.nan,
                            'mean_frozen': np.nan, 'mean_rebuilt': np.nan,
                            'n_missing_mismatch': int((a.isna() ^ b.isna()).sum())})
            else:
                a = pd.to_numeric(a, errors='coerce')
                b = pd.to_numeric(b, errors='coerce')
                ok = a.notna() & b.notna()
                d = (a - b).abs()[ok]
                row.update({'n_comparable': int(ok.sum()), 'exact_share': float((d < 1e-9).mean()) if ok.any() else np.nan,
                            'mean_abs_diff': float(d.mean()) if ok.any() else np.nan,
                            'max_abs_diff': float(d.max()) if ok.any() else np.nan,
                            'pearson_r': float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 and a[ok].std() > 0 and b[ok].std() > 0 else np.nan,
                            'mean_frozen': float(a[ok].mean()), 'mean_rebuilt': float(b[ok].mean()),
                            'n_missing_mismatch': int((a.isna() ^ b.isna()).sum())})
            out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)
    lg = open(os.path.join(out, 'run.log'), 'w', encoding='utf-8')

    def log(*x):
        msg = ' '.join(str(v) for v in x)
        print(msg, flush=True)
        lg.write(msg + '\n')
        lg.flush()

    import scipy
    import sklearn
    import xgboost
    from scipy.stats import spearmanr
    log('# H-07 gates A2a / A2 (post hoc)  clock=%s' % time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    log('python', platform.python_version(), 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__, 'numpy',
        np.__version__, 'pandas', pd.__version__, 'scipy', scipy.__version__)
    for f in ['H07_addendum2_plan.md', 'h07_rebuild.py', 'h07_a2.py']:
        log('sha256', f, C.sha256(os.path.join(HERE, f)))
    for f in ['events_quiz.csv', 'events_reading.csv', 'class_membership5.csv']:
        log('sha256', f, C.sha256(os.path.join(C.IN, f)))
    log('sha256 calendar', C.sha256(R.CAL))

    t0 = time.time()
    feat, time_, lsa, ratios, info = R.rebuild()
    log('rebuild done', info, f'{time.time() - t0:.0f}s')
    m = R.merged_frozen_step9(8)
    ana = C.frozen_frame(8)
    keys = list(zip(ana.anon_id, ana.semester))
    fd, td, lt, rt = R.frozen_as_rebuilt(m)
    fz = assemble(fd, td, lt, rt, keys)
    rb = assemble(feat, time_, lsa, ratios, keys)
    fz['track'] = ana.program5_key.to_numpy()
    fz = fz.reset_index(drop=True)
    rb = rb.reset_index(drop=True)
    rb['track'] = fz['track']
    rebuilt_out = rb.drop(columns=['anon_id'])
    rebuilt_out.to_csv(os.path.join(out, 'rebuilt_features_W8.csv'), index=False, encoding='utf-8-sig')

    # ---- A2a ---------------------------------------------------------------------------------------------------------
    T = drift_table(fz, rb)
    T.to_csv(os.path.join(out, 'H07_A2a_feature_drift.csv'), index=False, encoding='utf-8-sig')
    TS = drift_table(fz, rb, by=fz['semester'])
    TS.rename(columns={'group': 'semester'}).to_csv(os.path.join(out, 'H07_A2a_drift_by_semester.csv'), index=False, encoding='utf-8-sig')
    TT = drift_table(fz, rb, by=fz['track'])
    TT.rename(columns={'group': 'class_track5'}).to_csv(os.path.join(out, 'H07_A2a_drift_by_track.csv'), index=False, encoding='utf-8-sig')
    pd.crosstab(fz['s_cluster'].fillna('NA'), rb['s_cluster'].fillna('NA'), rownames=['frozen'], colnames=['rebuilt']).to_csv(
        os.path.join(out, 'H07_A2a_s_cluster_confusion.csv'), encoding='utf-8-sig')
    log(T.drop(columns=['used_in']).round(4).to_string())

    # ---- A2 -------------------------------------------------------------------------------------------------------------
    m2 = R.inject(m, feat, time_, lsa, ratios)
    df = R.run_pipeline_w8(m2)
    y = df.y.to_numpy(int)
    auc = {"M1'": L.auc_fail(y, df.m1.to_numpy(float)), "M2'": L.auc_fail(y, df.m2.to_numpy(float)),
           "M3'": L.auc_fail(y, df.m3.to_numpy(float))}
    base = R.run_pipeline_w8(m)  # frozen features, A1b flow (reference for score comparison)
    comp = {}
    for k, col in [("M1'", 'm1'), ("M2'", 'm2'), ("M3'", 'm3')]:
        a, b = df[col].to_numpy(float), base[col].to_numpy(float)
        comp[k] = {'pearson': float(np.corrcoef(a, b)[0, 1]), 'spearman': float(spearmanr(a, b).correlation),
                   'mean_abs_diff': float(np.abs(a - b).mean()), 'max_abs_diff': float(np.abs(a - b).max())}
    anchors = {}
    for k in ["M2'", "M3'"]:
        anchors[k] = {'value': auc[k], 'anchor': ANCHORS[k], 'tolerance': TOL, 'diff': auc[k] - ANCHORS[k],
                      'pass': bool(abs(auc[k] - ANCHORS[k]) <= TOL)}
    gate = {'gate': 'A2 (event-rebuild anchor)', 'n_records': int(len(df)), 'outcome_fail_count': int(y.sum()),
            'rebuilt_auc': auc, 'A1b_reference_auc (frozen features, same pipeline)': A1B,
            'frozen_features_rerun_auc_check': {k: float(L.auc_fail(y, base[c].to_numpy(float))) for k, c in
                                                [("M1'", 'm1'), ("M2'", 'm2'), ("M3'", 'm3')]},
            'anchors': anchors, 'gate_A2_pass': bool(all(v['pass'] for v in anchors.values())),
            'score_comparison_rebuilt_vs_frozen_features': comp,
            'tolerance_note': 'Tolerance relaxed to +-0.005 by author ruling (rulings section B); A1b anchor gap -0.0045 disclosed.',
            'versions': {'python': platform.python_version(), 'xgboost': xgboost.__version__, 'sklearn': sklearn.__version__}}
    json.dump(gate, open(os.path.join(out, 'H07_gate_A2.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    df[['semester', 'final', 'y', 'm1', 'm2', 'm3']].to_csv(os.path.join(out, 'H07_A2_oof_rebuilt.csv'), index=False, encoding='utf-8-sig')
    log(json.dumps(gate, ensure_ascii=False, indent=1))
    log('done')
    lg.close()


if __name__ == '__main__':
    main()
