# -*- coding: utf-8 -*-
"""H-09 (items 1-3): within-semester AUC, selection rule re-run on semester-stratified metrics, percentile reference.

Run from the return root:  python -I H-09/h09_within_semester.py
Plan: plan/supp_analysis_plan.md section 4. CIs: semester-stratified record-level bootstrap (plan R3), B = 2000,
seed 20261006 (stream [seed, semester rank]); approximate because the OOF file has no student IDs.
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
import h_lib as H  # noqa: E402

PACK = os.path.join(ROOT, 'materials', 'PaperA_supp_analyses_newchat_pack_20261026')
OOF = os.path.join(PACK, '04_data', 'oof_scores_no_ids.csv')
XLSX = os.path.join(PACK, '04_data', 'Table_A_data_era.xlsx')
OUT = os.path.join(ROOT, 'H-09')
CANDS = ['W6', 'W8', 'W10', 'W12']
REF = 'W12'


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def past_reference_blend(df, sems):
    """Variant b: percentile of m1 and m2 against all records of chronologically earlier semesters (same week)."""
    out = np.full(len(df), np.nan)
    sem = df['semester'].to_numpy()
    for s in sems[1:]:
        ix = sem == s
        ref = sem < s
        p1 = H.pct_ref(df['m1'].to_numpy()[ix], df['m1'].to_numpy()[ref])
        p2 = H.pct_ref(df['m2'].to_numpy()[ix], df['m2'].to_numpy()[ref])
        out[ix] = (p1 + p2) / 2
    return out


def sem_points(df, col, sems):
    a, r = [], []
    for s in sems:
        g = df[df['semester'] == s]
        a.append(H.auc_fail(g['y'].to_numpy(int), g[col].to_numpy(float)))
        r.append(H.pearson(g[col], g['final']))
    return np.array(a), np.array(r)


def main():
    log = open(os.path.join(OUT, 'run.log'), 'w', encoding='utf-8')
    H.log_versions(log)
    for p in (OOF, XLSX):
        print(f'sha256 {os.path.relpath(p, ROOT)} {sha(p)}', file=log)
    print(f'bootstrap: semester-stratified record-level, B={H.BOOT_B}, seed={H.BOOT_SEED}, stream [seed, semester rank]', file=log)

    oof = H.load_oof(OOF)
    f0 = oof['W6']
    sems = sorted(int(s) for s in f0['semester'].unique())
    sizes = {s: int((f0['semester'] == s).sum()) for s in sems}
    ns = np.array([sizes[s] for s in sems], float)
    N = ns.sum()
    sdraws = H.strat_draws(sizes)
    keys = [(W, c) for W in CANDS for c in ('m1', 'm2', 'm3')]
    boot = H.boot_strat_series(oof, keys, sdraws)

    # ------------------------------------------------------------------ item 1
    rows, summ = [], []
    metric = {}   # (variant, model, W) -> (auc_pt, r_pt, auc_b, r_b)
    for model, col in H.MODEL_COL.items():
        for W in CANDS:
            b = boot[(W, col)]
            df = oof[W]
            a_pt, r_pt = sem_points(df, col, sems)
            ses = []
            for j, s in enumerate(sems):
                lo, hi = H.percentile_ci(b['sem_auc'][:, j])
                se = float(np.nanstd(b['sem_auc'][:, j], ddof=1))
                ses.append(se)
                rows.append({'model': model, 'W': W, 'semester': s, 'n': sizes[s],
                             'n_fail': int(df.loc[df['semester'] == s, 'y'].sum()), 'auc': a_pt[j],
                             'auc_lo': lo, 'auc_hi': hi, 'auc_se_boot': se, 'r': r_pt[j]})
            wm_pt = H.weighted_mean(a_pt, ns)
            wm_b = b['sem_auc'] @ ns / N
            pooled_pt = H.auc_fail(df['y'].to_numpy(int), df[col].to_numpy(float))
            pooled_b = b['pooled_auc']
            diff_b = wm_b - pooled_b
            dl = H.dl_random_effects(a_pt, np.array(ses))
            wlo, whi = H.percentile_ci(wm_b)
            plo, phi = H.percentile_ci(pooled_b)
            dlo, dhi = H.percentile_ci(diff_b)
            summ.append({'model': model, 'W': W, 'pooled_auc': pooled_pt, 'pooled_lo': plo, 'pooled_hi': phi,
                         'weighted_mean_auc': wm_pt, 'weighted_lo': wlo, 'weighted_hi': whi,
                         'diff_weighted_minus_pooled': wm_pt - pooled_pt, 'diff_lo': dlo, 'diff_hi': dhi,
                         'dl_auc': dl['pooled'], 'dl_ci_lo': dl['ci_lo'], 'dl_ci_hi': dl['ci_hi'], 'dl_tau2': dl['tau2'],
                         'dl_i2': dl['i2'], 'dl_pi_lo': dl['pi_lo'], 'dl_pi_hi': dl['pi_hi'],
                         'sem_auc_min': float(a_pt.min()), 'sem_auc_max': float(a_pt.max())})
            r_all = H.pearson(df[col], df['final'])
            metric[('pooled', model, W)] = (pooled_pt, r_all, pooled_b, b['pooled_r'])
            metric[('semester_stratified', model, W)] = (wm_pt, float(H.fisher_weighted_r(r_pt, ns)), wm_b,
                                                         H.fisher_weighted_r(b['sem_r'], ns))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'H09_1_within_semester_auc.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(summ).to_csv(os.path.join(OUT, 'H09_1_summary.csv'), index=False, encoding='utf-8-sig')

    # ------------------------------------------------------------------ item 2
    sel_rows, ws_rows, mc_rows = [], [], []
    for variant in ('pooled', 'semester_stratified'):
        wstar = {}
        for model in H.MODEL_COL:
            ok = {'point': {}, 'ci95': {}, 'ci993': {}}
            a_ref, r_ref, ab_ref, rb_ref = metric[(variant, model, REF)]
            for W in CANDS:
                a_w, r_w, ab_w, rb_w = metric[(variant, model, W)]
                la_b, lr_b = ab_ref - ab_w, rb_ref - rb_w
                row = {'variant': variant, 'model': model, 'W': W, 'auc': a_w, 'r': r_w,
                       'auc_loss': a_ref - a_w, 'auc_loss_hi95': H.percentile_hi(la_b, 0.975),
                       'auc_loss_hi993': H.percentile_hi(la_b, 'bonf'), 'r_loss': r_ref - r_w,
                       'r_loss_hi95': H.percentile_hi(lr_b, 0.975), 'r_loss_hi993': H.percentile_hi(lr_b, 'bonf')}
                row['point_ok'] = H.qualifies(row['auc_loss'], row['r_loss'])
                row['ci95_ok'] = H.qualifies(row['auc_loss_hi95'], row['r_loss_hi95'])
                row['ci993_ok'] = H.qualifies(row['auc_loss_hi993'], row['r_loss_hi993'])
                sel_rows.append(row)
                ok['point'][int(W[1:])] = row['point_ok']
                ok['ci95'][int(W[1:])] = row['ci95_ok']
                ok['ci993'][int(W[1:])] = row['ci993_ok']
            for crit in ('point', 'ci95', 'ci993'):
                e = H.earliest_week(ok[crit])
                ws_rows.append({'variant': variant, 'model': model, 'criterion': crit, 'W_star': 'none' if e is None else e})
                if crit == 'ci95':
                    wstar[model] = e
        w_sel = min(wstar.values())
        feasible = [m for m in H.MODEL_COL if wstar[m] == w_sel]
        Wsel = f'W{w_sel}'
        auc_by = {m: metric[(variant, m, Wsel)][0] for m in feasible}
        ranked = sorted(feasible, key=lambda m: auc_by[m], reverse=True)
        if len(ranked) == 1:
            mc_rows.append({'variant': variant, 'W_sel': w_sel, 'feasible': ','.join(feasible), 'best': ranked[0],
                            'second': '', 'diff': np.nan, 'diff_lo': np.nan, 'diff_hi': np.nan,
                            'equivalent': False, 'chosen': ranked[0], 'M3_prime_passes_W8': wstar["M3'"] == 8})
        else:
            best, second = ranked[0], ranked[1]
            d_b = metric[(variant, best, Wsel)][2] - metric[(variant, second, Wsel)][2]
            lo, hi = H.percentile_ci(d_b)
            chosen = H.choose_model(auc_by, (lo, hi))
            mc_rows.append({'variant': variant, 'W_sel': w_sel, 'feasible': ','.join(feasible), 'best': best,
                            'second': second, 'diff': auc_by[best] - auc_by[second], 'diff_lo': lo, 'diff_hi': hi,
                            'equivalent': bool(lo >= -H.EQUIV_MARGIN and hi <= H.EQUIV_MARGIN), 'chosen': chosen,
                            'M3_prime_passes_W8': wstar["M3'"] == 8})
    pd.DataFrame(sel_rows).to_csv(os.path.join(OUT, 'H09_2_selection_rerun.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(ws_rows).to_csv(os.path.join(OUT, 'H09_2_wstar.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(mc_rows).to_csv(os.path.join(OUT, 'H09_2_model_choice.csv'), index=False, encoding='utf-8-sig')

    # ------------------------------------------------------------------ item 3
    idrows = []
    for W in H.W_ALL:
        df = oof[W]
        idrows.append({'W': W,
                       'max_abs_diff_vs_pooled_percentile_blend':
                           float(np.abs(df['m3_shipped'] - H.pooled_pct_blend(df['m1'].to_numpy(), df['m2'].to_numpy())).max()),
                       'max_abs_diff_vs_within_semester_blend':
                           float(np.abs(df['m3_shipped'].to_numpy() - H.within_semester_blend(df)).max())})
    pd.DataFrame(idrows).to_csv(os.path.join(OUT, 'H09_3a_identity_check.csv'), index=False, encoding='utf-8-sig')

    f7, f6 = {}, {}
    for W in CANDS:
        d = oof[W].copy()
        d['m3a'] = H.within_semester_blend(d)
        d['m3b'] = past_reference_blend(d, sems)
        f7[W] = d
        f6[W] = d[d['semester'] != sems[0]].reset_index(drop=True)
    b7 = H.boot_strat_series(f7, [(W, c) for W in CANDS for c in ('m3', 'm3a')], sdraws)
    b6 = H.boot_strat_series(f6, [(W, c) for W in CANDS for c in ('m3', 'm3a', 'm3b')], sdraws)
    ns6 = ns[1:]
    N6 = ns6.sum()
    out3, per_sem = [], []
    for subset, frames, bt, wts, ss in (('7 semesters', f7, b7, ns, sems),
                                         ('6 semesters (1112-1141)', f6, b6, ns6, sems[1:])):
        cols = ('m3', 'm3a') if subset == '7 semesters' else ('m3', 'm3a', 'm3b')
        for W in CANDS:
            df = frames[W]
            pts_a = {c: sem_points(df, c, ss)[0] for c in cols}
            for j, s in enumerate(ss):
                per_sem.append({'W': W, 'subset': subset, 'semester': s, 'auc_orig': pts_a['m3'][j],
                                'auc_a_within_semester': pts_a['m3a'][j],
                                'auc_b_past_reference': pts_a['m3b'][j] if 'm3b' in cols else np.nan})
            for metric_name in ('weighted_within_auc', 'pooled_auc'):
                pt, arr = {}, {}
                for c in cols:
                    if metric_name == 'weighted_within_auc':
                        pt[c] = H.weighted_mean(pts_a[c], wts)
                        arr[c] = bt[(W, c)]['sem_auc'] @ wts / wts.sum()
                    else:
                        pt[c] = H.auc_fail(df['y'].to_numpy(int), df[c].to_numpy(float))
                        arr[c] = bt[(W, c)]['pooled_auc']
                row = {'W': W, 'subset': subset, 'metric': metric_name, 'auc_orig': pt['m3'], 'auc_a': pt['m3a'],
                       'auc_b': pt.get('m3b', np.nan), 'd_a': pt['m3a'] - pt['m3'],
                       'd_a_lo': H.percentile_ci(arr['m3a'] - arr['m3'])[0], 'd_a_hi': H.percentile_ci(arr['m3a'] - arr['m3'])[1],
                       'd_b': np.nan, 'd_b_lo': np.nan, 'd_b_hi': np.nan}
                if 'm3b' in cols:
                    lo, hi = H.percentile_ci(arr['m3b'] - arr['m3'])
                    row.update({'d_b': pt['m3b'] - pt['m3'], 'd_b_lo': lo, 'd_b_hi': hi})
                out3.append(row)
    pd.DataFrame(out3).to_csv(os.path.join(OUT, 'H09_3b_percentile_reference.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(per_sem).to_csv(os.path.join(OUT, 'H09_3b_per_semester_auc.csv'), index=False, encoding='utf-8-sig')

    # memo tables generated from the output files (no hand transcription)
    def f3(x):
        return 'NA' if pd.isna(x) else f'{x:.3f}'.replace('-', '\u2212')
    sm = pd.DataFrame(summ)
    L = ['### Item 1. Week-8 AUC: pooled vs within-semester (95% CI, semester-stratified record-level bootstrap)', '',
         '| Predictor | Pooled AUC | Weighted within-semester AUC [CI] | Difference [CI] | Random-effects AUC [CI] | Prediction interval | Semester range |',
         '|---|---|---|---|---|---|---|']
    for _, r in sm[sm['W'] == 'W8'].iterrows():
        L.append(f"| {r['model']} | {f3(r['pooled_auc'])} | {f3(r['weighted_mean_auc'])} [{f3(r['weighted_lo'])}\u2013{f3(r['weighted_hi'])}] "
                 f"| {f3(r['diff_weighted_minus_pooled'])} [{f3(r['diff_lo'])}; {f3(r['diff_hi'])}] "
                 f"| {f3(r['dl_auc'])} [{f3(r['dl_ci_lo'])}\u2013{f3(r['dl_ci_hi'])}] | {f3(r['dl_pi_lo'])}\u2013{f3(r['dl_pi_hi'])} "
                 f"| {f3(r['sem_auc_min'])}\u2013{f3(r['sem_auc_max'])} |")
    L += ['', '### Item 1. Blend (M3\u2032) by week', '',
          '| Week | Pooled AUC | Weighted within-semester AUC | Difference [CI] | Tau^2 | I^2 |', '|---|---|---|---|---|---|']
    for _, r in sm[sm['model'] == "M3'"].iterrows():
        L.append(f"| {r['W']} | {f3(r['pooled_auc'])} | {f3(r['weighted_mean_auc'])} | {f3(r['diff_weighted_minus_pooled'])} "
                 f"[{f3(r['diff_lo'])}; {f3(r['diff_hi'])}] | {r['dl_tau2']:.5f} | {r['dl_i2']:.2f} |")
    sr = pd.DataFrame(sel_rows)
    L += ['', '### Item 2. Selection rule re-run, reference Week 12 (loss point [95% upper limit; 99.3% upper limit])', '',
          '| Metric | Predictor | Week | AUC loss | r loss | Point | 95% | 99.3% |', '|---|---|---|---|---|---|---|---|']
    for _, r in sr[sr['W'].isin(['W8', 'W10'])].iterrows():
        L.append(f"| {r['variant']} | {r['model']} | {r['W']} | {f3(r['auc_loss'])} [{f3(r['auc_loss_hi95'])}; {f3(r['auc_loss_hi993'])}] "
                 f"| {f3(r['r_loss'])} [{f3(r['r_loss_hi95'])}; {f3(r['r_loss_hi993'])}] "
                 f"| {'met' if r['point_ok'] else 'not met'} | {'met' if r['ci95_ok'] else 'not met'} | {'met' if r['ci993_ok'] else 'not met'} |")
    wsd = pd.DataFrame(ws_rows)
    L += ['', '### Item 2. Earliest qualifying week and model choice', '',
          '| Metric | Predictor | Point | 95% | 99.3% |', '|---|---|---|---|---|']
    for variant in ('pooled', 'semester_stratified'):
        for model in H.MODEL_COL:
            v = {c: str(wsd[(wsd['variant'] == variant) & (wsd['model'] == model) & (wsd['criterion'] == c)]['W_star'].iloc[0])
                 for c in ('point', 'ci95', 'ci993')}
            L.append(f"| {variant} | {model} | {v['point']} | {v['ci95']} | {v['ci993']} |")
    L += ['', '| Metric | W_sel | Feasible | Best | Second | Advantage [CI] | Equivalent | Chosen |', '|---|---|---|---|---|---|---|---|']
    for r in mc_rows:
        L.append(f"| {r['variant']} | {r['W_sel']} | {r['feasible']} | {r['best']} | {r['second']} | {f3(r['diff'])} "
                 f"[{f3(r['diff_lo'])}; {f3(r['diff_hi'])}] | {r['equivalent']} | {r['chosen']} |")
    o3 = pd.DataFrame(out3)
    L += ['', '### Item 3. M3\u2032 percentile reference, change in AUC versus shipped score (a = within-semester percentile, b = past-semester reference)', '',
          '| Week | Subset | Metric | Shipped AUC | a: change [CI] | b: change [CI] |', '|---|---|---|---|---|---|']
    for _, r in o3.iterrows():
        bcell = 'NA' if pd.isna(r['d_b']) else f"{f3(r['d_b'])} [{f3(r['d_b_lo'])}; {f3(r['d_b_hi'])}]"
        L.append(f"| {r['W']} | {r['subset']} | {r['metric']} | {f3(r['auc_orig'])} | {f3(r['d_a'])} [{f3(r['d_a_lo'])}; {f3(r['d_a_hi'])}] | {bcell} |")
    open(os.path.join(OUT, 'H09_summary_tables.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')

    print('done', file=log)
    log.close()
    print('done')


if __name__ == '__main__':
    main()
