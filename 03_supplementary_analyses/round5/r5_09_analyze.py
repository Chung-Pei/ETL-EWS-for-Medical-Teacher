# -*- coding: utf-8 -*-
"""R5-9 (analysis stage) -- apply Appendix B to each refit resample.  Level: student-cluster bootstrap, refitting-inclusive,
frozen-feature path.  Same nesting convention as R5-3: the outer resample supplies the refit scores and the point metrics; the loss
intervals come from D=1,000 inner draws (same seeds as R5-3) over the refit scores.  The same rule is also applied to the frozen
scores on the SAME outer resamples for a paired comparison.  Usage: python r5_09_analyze.py FILE [FILE ...]"""
import re
import sys
import time

import numpy as np
import pandas as pd

import r5_lib as L

files = sys.argv[1:]
base, frozen = L.load_base()
N = len(base)
y = base.y.to_numpy().astype(int)
f = base.final.to_numpy(float)
stu = base.stu.to_numpy()
n_stu = int(stu.max()) + 1
groups = L.sem_groups_of(base.sem_i)
counts_all = L.make_counts(n_stu, 2000, L.SEED)
D_IN = 1000
WK = [6, 8, 10, 12]

# ---- collect refit scores -------------------------------------------------------------------------------------------------------
idx, secs, M = [], [], {}
for fn in files:
    z = np.load(fn)
    idx += z['idx'].tolist()
    secs += z['secs'].tolist()
    for k in z.files:
        if k.startswith('m'):
            M.setdefault(k, []).append(z[k])
M = {k: np.concatenate(v) for k, v in M.items()}
order = np.argsort(idx)
idx = np.array(idx)[order]
M = {k: v[order] for k, v in M.items()}
secs = np.array(secs)[order]
B = len(idx)
print(f'B = {B} refit resamples; mean {secs.mean():.1f}s per resample (sum {secs.sum():.0f}s over {len(files)} worker files)')
import os
assert B >= int(os.environ.get("R5_MIN_B", 100))


def refit_scores(j):
    sc = {}
    for W in WK:
        m1, m2 = M[f'm1_W{W}'][j], M[f'm2_W{W}'][j]
        sc[f'W{W}'] = pd.DataFrame({'m1': m1, 'm2': m2, 'm3': L.pooled_blend(m1, m2)})
    return sc


def one(sc, defn, b, rng_seed_b):
    names, S = L.build_candidates(base, sc, defn)
    col = {n: i for i, n in enumerate(names)}
    Wb = counts_all[b][stu][None, :]
    a_pt, _ = L.metrics_draws(Wb, S, y, f, groups, defn)
    pt = {n: float(a_pt[0, i]) for i, n in enumerate(names)}
    rng = np.random.default_rng(np.random.SeedSequence([L.SEED, b]))
    Wi = L.inner_weights(counts_all[b], stu, D_IN, rng)
    a_d, r_d = L.metrics_draws(Wi, S, y, f, groups, defn)
    sel = L.select_rule(a_d, r_d, names, pt)
    la = a_d[:, col[('W12', 'M3')]] - a_d[:, col[('W8', 'M3')]]
    lr = r_d[:, col[('W12', 'M3')]] - r_d[:, col[('W8', 'M3')]]
    out = {'b': b, 'sel_week': sel['sel_week'] or 'none', 'sel_pred': sel['sel_pred'] or 'none',
           'ucl97_auc_W8_M3': float(np.percentile(la, 97.5)), 'ucl95_auc_W8_M3': float(np.percentile(la, 95)),
           'ucl97_r_W8_M3': float(np.percentile(lr, 97.5)), 'auc_W8_M3': pt[('W8', 'M3')], 'auc_W10_M3': pt[('W10', 'M3')],
           'auc_W12_M3': pt[('W12', 'M3')], 'equiv': int(any(c['entirely_within_band'] for c in sel['compare']))}
    for p in L.PRED:
        out[f'wstar_{p}'] = sel['wstar'][p] or 'none'
    return out


t0 = time.time()
recs = {(src, d): [] for src in ('refit', 'frozen') for d in ('deploy', 'pooled')}
for j, b in enumerate(idx):
    sr = refit_scores(j)
    for d in ('deploy', 'pooled'):
        recs[('refit', d)].append(one(sr, d, int(b), None))
        recs[('frozen', d)].append(one(frozen, d, int(b), None))
    if (j + 1) % 50 == 0:
        print(f'analysed {j + 1}/{B}  {time.time() - t0:.0f}s', flush=True)
R = {k: pd.DataFrame(v) for k, v in recs.items()}
for (src, d), df in R.items():
    df.to_csv(L.RES / f'R5-9_per_resample_{src}_{d}.csv', index=False, encoding='utf-8-sig')

# ---- R5-3 reference frequencies (2,000 outer, frozen scores) -----------------------------------------------------------------------
r53 = pd.read_csv(L.RES / 'R5-3.csv')
rows = []


def add(**kw):
    base_row = {'table': '', 'definition': '', 'source': '', 'predictor': '', 'week': '', 'policy': '', 'statistic': '',
                'count': np.nan, 'n': np.nan, 'value': np.nan, 'note': ''}
    base_row.update(kw)
    rows.append(base_row)


tot = float(sum(secs))
add(table='run_info', statistic='level', note='student-cluster bootstrap (cluster = anon_id), REFITTING-INCLUSIVE, frozen-feature path '
    '(step9 features W6/W8/W10/W12; pipeline code ews_lib/ews_experiment + ETL v7.15.34 unmodified; XGBoost 3.4.1 CPU; arm P, '
    'feature set C, leave-one-semester-out). Events-level path NOT run.')
add(table='run_info', statistic='B_resamples', value=B, note='first B rows of the standing 2,000-draw stream (seed 20261031)')
add(table='run_info', statistic='per_resample_runtime_seconds_mean', value=float(secs.mean()),
    note='four weeks x seven folds x five XGBoost seeds, one CPU core per worker')
add(table='run_info', statistic='total_fit_cpu_seconds', value=tot, note=f'{len(files)} parallel workers; wall time ~ half')
add(table='run_info', statistic='inner_draws_per_resample', value=D_IN, note='loss intervals: inner student-cluster draws from the outer resample, same seeds as R5-3')

SRC = {'refit': 'refit (R5-9, refitting-inclusive)', 'frozen': f'frozen scores, same {B} outer resamples (paired)'}
for d in ('deploy', 'pooled'):
    dn = 'deployment (primary)' if d == 'deploy' else 'pooled (secondary)'
    for src in ('refit', 'frozen'):
        df = R[(src, d)]
        for p in L.PRED:
            vc = df[f'wstar_{p}'].value_counts()
            for W in ['W6', 'W8', 'W10', 'W12', 'none']:
                add(table='selected_week_per_predictor', definition=dn, source=SRC[src], predictor={'M1': "M1' (BAS)",
                    'M2': "M2' (XGBoost)", 'M3': "M3' (blend)"}[p], week=W, count=int(vc.get(W, 0)), n=B,
                    value=float(vc.get(W, 0)) / B)
        pol = (df.sel_week + ' + ' + df.sel_pred).value_counts()
        for k, v in pol.items():
            add(table='selected_policy', definition=dn, source=SRC[src], policy=k, count=int(v), n=B, value=v / B)
        n10 = int(((df.sel_week == 'W10') & (df.sel_pred == 'M3')).sum())
        add(table='headline', definition=dn, source=SRC[src], policy='Week 10 + blend selected', count=n10, n=B, value=n10 / B)
        add(table='headline', definition=dn, source=SRC[src], policy='Week 8 + blend selected', count=int(((df.sel_week == 'W8') &
            (df.sel_pred == 'M3')).sum()), n=B, value=float(((df.sel_week == 'W8') & (df.sel_pred == 'M3')).mean()))
        for col, lab in (('ucl97_auc_W8_M3', 'blend W8 AUC-loss upper limit (97.5th pct of inner draws = upper limit of 95% interval)'),
                         ('ucl95_auc_W8_M3', 'blend W8 AUC-loss upper limit (95th pct, one-sided; informational)'),
                         ('ucl97_r_W8_M3', 'blend W8 r-loss upper limit (97.5th pct)')):
            v = df[col].to_numpy()
            add(table='W8_blend_loss_upper_limit', definition=dn, source=SRC[src], statistic=lab + ' | median', value=float(np.median(v)), n=B)
            add(table='W8_blend_loss_upper_limit', definition=dn, source=SRC[src], statistic=lab + ' | 2.5th pct', value=float(np.percentile(v, 2.5)), n=B)
            add(table='W8_blend_loss_upper_limit', definition=dn, source=SRC[src], statistic=lab + ' | 97.5th pct', value=float(np.percentile(v, 97.5)), n=B)
        thr = AUC_T = 0.020
        for col, lab in (('ucl97_auc_W8_M3', '97.5th-pct limit'), ('ucl95_auc_W8_M3', '95th-pct limit')):
            k = int((df[col] > thr).sum())
            add(table='W8_blend_loss_upper_limit', definition=dn, source=SRC[src], statistic=f'share of resamples with {lab} > 0.020',
                count=k, n=B, value=k / B)
        for col, lab in (('auc_W8_M3', 'W8'), ('auc_W10_M3', 'W10'), ('auc_W12_M3', 'W12')):
            v = df[col].to_numpy()
            add(table='blend_point_AUC_across_resamples', definition=dn, source=SRC[src], week=lab, statistic='median [2.5th, 97.5th]',
                value=float(np.median(v)), note=f'[{np.percentile(v, 2.5):.4f}, {np.percentile(v, 97.5):.4f}]', n=B)
        add(table='headline', definition=dn, source=SRC[src], policy='resamples with a third-rule comparison entirely inside +-0.02',
            count=int(df.equiv.sum()), n=B, value=float(df.equiv.mean()))
# R5-3 frozen reference at B=2000
for _, r in r53.iterrows():
    if r['table'] in ('selected_week_per_predictor', 'headline') and str(r.get('policy', '')) in ('', 'Week 10 + blend selected', 'nan'):
        add(table='R5-3_reference_frozen_scores_2000_outer', definition='deployment (primary)', source='R5-3 frozen scores, B=2000',
            predictor=r['predictor'], week=r['week'], policy='' if pd.isna(r['policy']) else r['policy'], count=int(r['count']),
            n=2000, value=float(r['frequency']), note=r['table'])
T = pd.DataFrame(rows)
T.to_csv(L.RES / 'R5-9.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250, 'display.max_colwidth', 120, 'display.max_rows', 500)
print(T[T.table.isin(['run_info', 'headline', 'W8_blend_loss_upper_limit', 'blend_point_AUC_across_resamples'])].drop(columns=['table']).to_string())
print(T[T.table.isin(['selected_week_per_predictor', 'selected_policy']) & T.definition.str.startswith('deploy')].drop(columns=['table']).to_string())
print(f'analysis {time.time() - t0:.0f}s')
