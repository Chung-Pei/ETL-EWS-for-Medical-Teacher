# -*- coding: utf-8 -*-
"""H-04 and H-09 (items 1-3) rerun with student-cluster bootstrap. Plan: plan/H0409c_plan.md.
Run from the return root: python -I H-04_H-09_cluster_rerun/h0409_cluster_rerun.py"""
import hashlib
import os
import platform
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
import clusterboot as CB  # noqa: E402
import h_lib as H  # noqa: E402
import linkage as K  # noqa: E402

P2 = os.path.join(ROOT, 'materials', 'pack2')
OOF = os.path.join(ROOT, 'materials', 'pack1', '04_data', 'oof_scores_no_ids.csv')
R1 = os.path.join(ROOT, 'materials', 'round1_outputs')
OUT = os.path.join(ROOT, 'H-04_H-09_cluster_rerun')
MODELS = H.MODEL_COL
W4 = ['W6', 'W8', 'W10', 'W12', 'W14', 'W16']
W9 = ['W6', 'W8', 'W10', 'W12']
REFS = ['W12', 'W14', 'W16']
APPROX_TOL = 0.003


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def past_ref_blend(df, sems):
    out = np.full(len(df), np.nan)
    s_ = df['semester'].to_numpy(int)
    for s in sems[1:]:
        ix, ref = s_ == s, s_ < s
        p1 = H.pct_ref(df['m1'].to_numpy()[ix], df['m1'].to_numpy()[ref])
        p2 = H.pct_ref(df['m2'].to_numpy()[ix], df['m2'].to_numpy()[ref])
        out[ix] = (p1 + p2) / 2
    return out


def fz2d(r, n):
    """Weighted Fisher-z mean per row (weights n - 3); r, n of shape (B, k)."""
    w = n - 3.0
    return np.tanh((np.arctanh(np.clip(r, -0.999, 0.999)) * w).sum(axis=1) / w.sum(axis=1))


def main():
    log = open(os.path.join(OUT, 'run.log'), 'w', encoding='utf-8')
    print(f'python {platform.python_version()} numpy {np.__version__} pandas {pd.__version__}', file=log)
    print(f'sha256 {os.path.relpath(OOF, ROOT)} {sha(OOF)}', file=log)
    print(f'bootstrap: student-cluster (ews_lib.Boot draws), B={CB.B_DEFAULT}, seed={CB.SEED_DEV}', file=log)

    oofs = K.load_oof_with_ids(OOF, P2)
    base = oofs['W6']
    y = base['y'].to_numpy(int)
    sem = base['semester'].to_numpy(int)
    f = base['final'].to_numpy(float)
    sems = sorted(set(sem.tolist()))
    n_pt = np.array([int((sem == s).sum()) for s in sems], float)
    cb = CB.ClusterBoot(base['anon_id'].to_numpy())
    B = cb.B

    score = {}
    for W in W4:
        for col in ('m1', 'm2', 'm3'):
            score[(W, col)] = oofs[W][col].to_numpy(float)
    for W in W9:
        score[(W, 'm3a')] = H.within_semester_blend(oofs[W])
        score[(W, 'm3b')] = past_ref_blend(oofs[W], sems)
    keys4 = [(W, c) for W in W4 for c in ('m1', 'm2', 'm3')]
    keys9 = [(W, c) for W in W9 for c in ('m1', 'm2', 'm3', 'm3a')]
    keysb = [(W, 'm3b') for W in W9]
    keys6 = [(W, c) for W in W9 for c in ('m3', 'm3a', 'm3b')]
    Pa = {k: np.full(B, np.nan) for k in keys4 + [(W, 'm3a') for W in W9]}
    Pr = {k: np.full(B, np.nan) for k in keys4}
    Sa = {k: np.full((B, len(sems)), np.nan) for k in keys9 + keysb}
    Sr = {k: np.full((B, len(sems)), np.nan) for k in keys4 if k[0] in W9}
    P6 = {k: np.full(B, np.nan) for k in keys6}
    Ns = np.zeros((B, len(sems)))
    for b, ix in enumerate(cb.idx):
        yb, sb = y[ix], sem[ix]
        six = [ix[sb == s] for s in sems]
        ix6 = ix[sb != sems[0]]
        Ns[b] = [len(a) for a in six]
        for k in keys4:
            sc = score[k]
            Pa[k][b] = H.auc_fail(yb, sc[ix])
            Pr[k][b] = corr(sc[ix], f[ix])
        for W in W9:
            Pa[(W, 'm3a')][b] = H.auc_fail(yb, score[(W, 'm3a')][ix])
        for k in keys9:
            sc = score[k]
            for j, a in enumerate(six):
                Sa[k][b, j] = H.auc_fail(y[a], sc[a])
                if k in Sr:
                    Sr[k][b, j] = corr(sc[a], f[a])
        for k in keysb:
            sc = score[k]
            for j in range(1, len(sems)):
                Sa[k][b, j] = H.auc_fail(y[six[j]], sc[six[j]])
        for k in keys6:
            P6[k][b] = H.auc_fail(y[ix6], score[k][ix6])

    # ---------------------------------------------------------------- points
    def pa(k):
        return H.auc_fail(y, score[k])

    def pr(k):
        return corr(score[k], f)

    def sem_pts(k):
        a, r = [], []
        for s in sems:
            m = sem == s
            a.append(H.auc_fail(y[m], score[k][m]))
            r.append(corr(score[k][m], f[m]) if k in Sr else np.nan)
        return np.array(a), np.array(r)

    # ---------------------------------------------------------------- H-04
    rows = []
    for model, col in MODELS.items():
        for ref in REFS:
            for W in ['W6', 'W8', 'W10', 'W12']:
                la_b = Pa[(ref, col)] - Pa[(W, col)]
                lr_b = Pr[(ref, col)] - Pr[(W, col)]
                row = {'model': model, 'ref': ref, 'W': W, 'auc_ref': pa((ref, col)), 'auc_w': pa((W, col)),
                       'auc_loss': pa((ref, col)) - pa((W, col)),
                       'auc_loss_lo95': H.percentile_ci(la_b)[0], 'auc_loss_hi95': H.percentile_hi(la_b, 0.975),
                       'auc_loss_hi993': H.percentile_hi(la_b, 'bonf'),
                       'r_ref': pr((ref, col)), 'r_w': pr((W, col)), 'r_loss': pr((ref, col)) - pr((W, col)),
                       'r_loss_lo95': H.percentile_ci(lr_b)[0], 'r_loss_hi95': H.percentile_hi(lr_b, 0.975),
                       'r_loss_hi993': H.percentile_hi(lr_b, 'bonf')}
                row['point_ok'] = H.qualifies(row['auc_loss'], row['r_loss'])
                row['ci95_ok'] = H.qualifies(row['auc_loss_hi95'], row['r_loss_hi95'])
                row['ci993_ok'] = H.qualifies(row['auc_loss_hi993'], row['r_loss_hi993'])
                row['ci_method'] = 'student-cluster bootstrap (ews_lib.Boot draws)'
                rows.append(row)
    loss = pd.DataFrame(rows)
    loss.to_csv(os.path.join(OUT, 'H04c_loss_by_reference.csv'), index=False, encoding='utf-8-sig')
    er = []
    for model in MODELS:
        for ref in REFS:
            sub = loss[(loss['model'] == model) & (loss['ref'] == ref)]
            for crit, c in (('point', 'point_ok'), ('ci95', 'ci95_ok'), ('ci993', 'ci993_ok')):
                e = H.earliest_week({int(w[1:]): bool(v) for w, v in zip(sub['W'], sub[c])})
                er.append({'model': model, 'ref': ref, 'criterion': crit, 'earliest_W': 'none' if e is None else e})
    er = pd.DataFrame(er)
    er.to_csv(os.path.join(OUT, 'H04c_earliest_qualifying_week.csv'), index=False, encoding='utf-8-sig')

    # ---------------------------------------------------------------- H-09 item 1
    wrows, srows = [], []
    metric = {}
    for model, col in MODELS.items():
        for W in W9:
            k = (W, col)
            a_pt, r_pt = sem_pts(k)
            ses, wt_b = [], Ns
            for j, s in enumerate(sems):
                lo, hi = H.percentile_ci(Sa[k][:, j])
                se = float(np.nanstd(Sa[k][:, j], ddof=1))
                ses.append(se)
                wrows.append({'model': model, 'W': W, 'semester': s, 'n': int(n_pt[j]), 'n_fail': int(y[sem == s].sum()),
                              'auc': a_pt[j], 'auc_lo': lo, 'auc_hi': hi, 'auc_se_boot': se, 'r': r_pt[j]})
            wm_b = (Sa[k] * wt_b).sum(axis=1) / wt_b.sum(axis=1)
            wm_pt = H.weighted_mean(a_pt, n_pt)
            pooled_b = Pa[k]
            pooled_pt = pa(k)
            dl = H.dl_random_effects(a_pt, np.array(ses))
            wlo, whi = H.percentile_ci(wm_b)
            plo, phi = H.percentile_ci(pooled_b)
            dlo, dhi = H.percentile_ci(wm_b - pooled_b)
            srows.append({'model': model, 'W': W, 'pooled_auc': pooled_pt, 'pooled_lo': plo, 'pooled_hi': phi,
                          'weighted_mean_auc': wm_pt, 'weighted_lo': wlo, 'weighted_hi': whi,
                          'diff_weighted_minus_pooled': wm_pt - pooled_pt, 'diff_lo': dlo, 'diff_hi': dhi,
                          'dl_auc': dl['pooled'], 'dl_ci_lo': dl['ci_lo'], 'dl_ci_hi': dl['ci_hi'], 'dl_tau2': dl['tau2'],
                          'dl_i2': dl['i2'], 'dl_pi_lo': dl['pi_lo'], 'dl_pi_hi': dl['pi_hi'],
                          'sem_auc_min': float(a_pt.min()), 'sem_auc_max': float(a_pt.max())})
            metric[('pooled', model, W)] = (pooled_pt, pr(k), pooled_b, Pr[k])
            metric[('semester_stratified', model, W)] = (wm_pt, float(H.fisher_weighted_r(r_pt, n_pt)), wm_b, fz2d(Sr[k], Ns))
    pd.DataFrame(wrows).to_csv(os.path.join(OUT, 'H09c_1_within_semester_auc.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(srows).to_csv(os.path.join(OUT, 'H09c_1_summary.csv'), index=False, encoding='utf-8-sig')

    # ---------------------------------------------------------------- H-09 item 2
    sel, ws, mcr = [], [], []
    for variant in ('pooled', 'semester_stratified'):
        wstar = {}
        for model in MODELS:
            ok = {'point': {}, 'ci95': {}, 'ci993': {}}
            a_ref, r_ref, ab_ref, rb_ref = metric[(variant, model, 'W12')]
            for W in W9:
                a_w, r_w, ab_w, rb_w = metric[(variant, model, W)]
                la_b, lr_b = ab_ref - ab_w, rb_ref - rb_w
                row = {'variant': variant, 'model': model, 'W': W, 'auc': a_w, 'r': r_w, 'auc_loss': a_ref - a_w,
                       'auc_loss_hi95': H.percentile_hi(la_b, 0.975), 'auc_loss_hi993': H.percentile_hi(la_b, 'bonf'),
                       'r_loss': r_ref - r_w, 'r_loss_hi95': H.percentile_hi(lr_b, 0.975),
                       'r_loss_hi993': H.percentile_hi(lr_b, 'bonf')}
                row['point_ok'] = H.qualifies(row['auc_loss'], row['r_loss'])
                row['ci95_ok'] = H.qualifies(row['auc_loss_hi95'], row['r_loss_hi95'])
                row['ci993_ok'] = H.qualifies(row['auc_loss_hi993'], row['r_loss_hi993'])
                sel.append(row)
                for c in ('point', 'ci95', 'ci993'):
                    ok[c][int(W[1:])] = row[f'{c}_ok']
            for c in ('point', 'ci95', 'ci993'):
                e = H.earliest_week(ok[c])
                ws.append({'variant': variant, 'model': model, 'criterion': c, 'W_star': 'none' if e is None else e})
                if c == 'ci95':
                    wstar[model] = e
        w_sel = min(wstar.values())
        feas = [m for m in MODELS if wstar[m] == w_sel]
        Wsel = f'W{w_sel}'
        auc_by = {m: metric[(variant, m, Wsel)][0] for m in feas}
        ranked = sorted(feas, key=lambda m: auc_by[m], reverse=True)
        if len(ranked) == 1:
            mcr.append({'variant': variant, 'W_sel': w_sel, 'feasible': ','.join(feas), 'best': ranked[0], 'second': '',
                        'diff': np.nan, 'diff_lo': np.nan, 'diff_hi': np.nan, 'equivalent': False, 'chosen': ranked[0],
                        'M3_prime_passes_W8': wstar["M3'"] == 8})
        else:
            best, second = ranked[0], ranked[1]
            d_b = metric[(variant, best, Wsel)][2] - metric[(variant, second, Wsel)][2]
            lo, hi = H.percentile_ci(d_b)
            mcr.append({'variant': variant, 'W_sel': w_sel, 'feasible': ','.join(feas), 'best': best, 'second': second,
                        'diff': auc_by[best] - auc_by[second], 'diff_lo': lo, 'diff_hi': hi,
                        'equivalent': bool(lo >= -H.EQUIV_MARGIN and hi <= H.EQUIV_MARGIN),
                        'chosen': H.choose_model(auc_by, (lo, hi)), 'M3_prime_passes_W8': wstar["M3'"] == 8})
    sel = pd.DataFrame(sel)
    ws = pd.DataFrame(ws)
    mcr = pd.DataFrame(mcr)
    sel.to_csv(os.path.join(OUT, 'H09c_2_selection_rerun.csv'), index=False, encoding='utf-8-sig')
    ws.to_csv(os.path.join(OUT, 'H09c_2_wstar.csv'), index=False, encoding='utf-8-sig')
    mcr.to_csv(os.path.join(OUT, 'H09c_2_model_choice.csv'), index=False, encoding='utf-8-sig')

    # ---------------------------------------------------------------- H-09 item 3
    out3 = []
    mask6 = sem != sems[0]
    for subset, cols, js in (('7 semesters', ('m3', 'm3a'), list(range(len(sems)))),
                             ('6 semesters (1112-1141)', ('m3', 'm3a', 'm3b'), list(range(1, len(sems))))):
        for W in W9:
            pts, arrs = {}, {}
            for c in cols:
                k = (W, c)
                a_pt = np.array([H.auc_fail(y[sem == sems[j]], score[k][sem == sems[j]]) for j in js])
                w_pt = n_pt[js]
                wb = Ns[:, js]
                arrs[('w', c)] = (Sa[k][:, js] * wb).sum(axis=1) / wb.sum(axis=1)
                pts[('w', c)] = H.weighted_mean(a_pt, w_pt)
                if subset == '7 semesters':
                    arrs[('p', c)] = Pa[k]
                    pts[('p', c)] = pa(k)
                else:
                    arrs[('p', c)] = P6[k]
                    pts[('p', c)] = H.auc_fail(y[mask6], score[k][mask6])
            for mn, tag in (('weighted_within_auc', 'w'), ('pooled_auc', 'p')):
                row = {'W': W, 'subset': subset, 'metric': mn, 'auc_orig': pts[(tag, 'm3')], 'auc_a': pts[(tag, 'm3a')],
                       'auc_b': pts.get((tag, 'm3b'), np.nan), 'd_a': pts[(tag, 'm3a')] - pts[(tag, 'm3')],
                       'd_a_lo': H.percentile_ci(arrs[(tag, 'm3a')] - arrs[(tag, 'm3')])[0],
                       'd_a_hi': H.percentile_ci(arrs[(tag, 'm3a')] - arrs[(tag, 'm3')])[1],
                       'd_b': np.nan, 'd_b_lo': np.nan, 'd_b_hi': np.nan}
                if 'm3b' in cols:
                    lo, hi = H.percentile_ci(arrs[(tag, 'm3b')] - arrs[(tag, 'm3')])
                    row.update({'d_b': pts[(tag, 'm3b')] - pts[(tag, 'm3')], 'd_b_lo': lo, 'd_b_hi': hi})
                out3.append(row)
    pd.DataFrame(out3).to_csv(os.path.join(OUT, 'H09c_3b_percentile_reference.csv'), index=False, encoding='utf-8-sig')

    # ---------------------------------------------------------------- comparison with round 1
    r1_loss = pd.read_csv(os.path.join(R1, 'H04_loss_by_reference.csv'), encoding='utf-8-sig')
    r1_er = pd.read_csv(os.path.join(R1, 'H04_earliest_qualifying_week.csv'), encoding='utf-8-sig')
    r1_sel = pd.read_csv(os.path.join(R1, 'H09_2_selection_rerun.csv'), encoding='utf-8-sig')
    r1_ws = pd.read_csv(os.path.join(R1, 'H09_2_wstar.csv'), encoding='utf-8-sig')
    r1_mc = pd.read_csv(os.path.join(R1, 'H09_2_model_choice.csv'), encoding='utf-8-sig')
    cmp_rows = []
    m = loss.merge(r1_loss, on=['model', 'ref', 'W'], suffixes=('_c', '_r1'))
    for _, r in m.iterrows():
        for crit in ('ci95', 'ci993'):
            cmp_rows.append({'analysis': 'H-04 loss', 'item': f"{r['model']}|ref {r['ref']}|{r['W']}|{crit}",
                             'round1_auc_hi': r[f'auc_loss_hi{crit[2:] if crit == "ci993" else "95"}_r1'],
                             'cluster_auc_hi': r[f'auc_loss_hi{crit[2:] if crit == "ci993" else "95"}_c'],
                             'round1_r_hi': r[f'r_loss_hi{crit[2:] if crit == "ci993" else "95"}_r1'],
                             'cluster_r_hi': r[f'r_loss_hi{crit[2:] if crit == "ci993" else "95"}_c'],
                             'round1_ok': bool(r[f'{crit}_ok_r1']), 'cluster_ok': bool(r[f'{crit}_ok_c'])})
    m = er.merge(r1_er, on=['model', 'ref', 'criterion'], suffixes=('_c', '_r1'))
    for _, r in m.iterrows():
        cmp_rows.append({'analysis': 'H-04 earliest week', 'item': f"{r['model']}|ref {r['ref']}|{r['criterion']}",
                         'round1_value': str(r['earliest_W_r1']), 'cluster_value': str(r['earliest_W_c']),
                         'round1_ok': True, 'cluster_ok': str(r['earliest_W_r1']) == str(r['earliest_W_c'])})
    m = sel.merge(r1_sel, on=['variant', 'model', 'W'], suffixes=('_c', '_r1'))
    for _, r in m.iterrows():
        for crit in ('ci95', 'ci993'):
            sfx = '95' if crit == 'ci95' else '993'
            cmp_rows.append({'analysis': 'H-09 item 2', 'item': f"{r['variant']}|{r['model']}|{r['W']}|{crit}",
                             'round1_auc_hi': r[f'auc_loss_hi{sfx}_r1'], 'cluster_auc_hi': r[f'auc_loss_hi{sfx}_c'],
                             'round1_r_hi': r[f'r_loss_hi{sfx}_r1'], 'cluster_r_hi': r[f'r_loss_hi{sfx}_c'],
                             'round1_ok': bool(r[f'{crit}_ok_r1']), 'cluster_ok': bool(r[f'{crit}_ok_c'])})
    m = ws.merge(r1_ws, on=['variant', 'model', 'criterion'], suffixes=('_c', '_r1'))
    for _, r in m.iterrows():
        cmp_rows.append({'analysis': 'H-09 item 2 W*', 'item': f"{r['variant']}|{r['model']}|{r['criterion']}",
                         'round1_value': str(r['W_star_r1']), 'cluster_value': str(r['W_star_c']),
                         'round1_ok': True, 'cluster_ok': str(r['W_star_r1']) == str(r['W_star_c'])})
    m = mcr.merge(r1_mc, on='variant', suffixes=('_c', '_r1'))
    for _, r in m.iterrows():
        cmp_rows.append({'analysis': 'H-09 item 2 model choice', 'item': f"{r['variant']}|W_sel/chosen",
                         'round1_value': f"W{r['W_sel_r1']}/{r['chosen_r1']}", 'cluster_value': f"W{r['W_sel_c']}/{r['chosen_c']}",
                         'round1_ok': True, 'cluster_ok': (r['W_sel_r1'] == r['W_sel_c']) and (r['chosen_r1'] == r['chosen_c'])})
    cmp = pd.DataFrame(cmp_rows)
    cmp['decision_changed'] = cmp['round1_ok'] != cmp['cluster_ok']
    cmp.to_csv(os.path.join(OUT, 'H0409c_vs_round1_decisions.csv'), index=False, encoding='utf-8-sig')

    # ---------------------------------------------------------------- markdown tables
    def f3(x):
        return 'NA' if pd.isna(x) else f'{x:.3f}'.replace('-', '\u2212')
    L = [f"Decisions compared: {len(cmp)}; changed: {int(cmp['decision_changed'].sum())}", '', '### Changed decisions (round 1 record-level vs student-cluster)', '',
         '| Analysis | Item | Round 1 | Student-cluster |', '|---|---|---|---|']
    for _, r in cmp[cmp['decision_changed']].iterrows():
        if pd.notna(r.get('round1_auc_hi')):
            a = f"AUC hi {f3(r['round1_auc_hi'])}, r hi {f3(r['round1_r_hi'])}, ok={r['round1_ok']}"
            b = f"AUC hi {f3(r['cluster_auc_hi'])}, r hi {f3(r['cluster_r_hi'])}, ok={r['cluster_ok']}"
        else:
            a, b = r['round1_value'], r['cluster_value']
        L.append(f"| {r['analysis']} | {r['item']} | {a} | {b} |")
    L += ['', '### Boundary values (AUC upper limit within 0.002 of 0.020 or r upper limit within 0.002 of 0.030)', '',
          '| Analysis | Item | Round 1 AUC hi / r hi | Student-cluster AUC hi / r hi |', '|---|---|---|---|']
    bd = cmp[cmp['round1_auc_hi'].notna() & (((cmp['cluster_auc_hi'] - 0.02).abs() <= 0.002) | ((cmp['cluster_r_hi'] - 0.03).abs() <= 0.002) |
                                             ((cmp['round1_auc_hi'] - 0.02).abs() <= 0.002) | ((cmp['round1_r_hi'] - 0.03).abs() <= 0.002))]
    for _, r in bd.iterrows():
        L.append(f"| {r['analysis']} | {r['item']} | {f3(r['round1_auc_hi'])} / {f3(r['round1_r_hi'])} | {f3(r['cluster_auc_hi'])} / {f3(r['cluster_r_hi'])} |")
    L += ['', '### H-04 Week-8 rows (student-cluster)', '', '| Predictor | Ref | AUC loss [95% hi; 99.3% hi] | r loss [95% hi; 99.3% hi] | Point | 95% | 99.3% |', '|---|---|---|---|---|---|---|']
    for _, r in loss[loss['W'] == 'W8'].iterrows():
        L.append(f"| {r['model']} | {r['ref']} | {f3(r['auc_loss'])} [{f3(r['auc_loss_hi95'])}; {f3(r['auc_loss_hi993'])}] | {f3(r['r_loss'])} [{f3(r['r_loss_hi95'])}; {f3(r['r_loss_hi993'])}] | "
                 f"{'met' if r['point_ok'] else 'not met'} | {'met' if r['ci95_ok'] else 'not met'} | {'met' if r['ci993_ok'] else 'not met'} |")
    L += ['', '### H-04 earliest qualifying week (student-cluster)', '', '| Predictor | Criterion | Ref W12 | Ref W14 | Ref W16 |', '|---|---|---|---|---|']
    for model in MODELS:
        for crit in ('point', 'ci95', 'ci993'):
            v = [str(er[(er['model'] == model) & (er['ref'] == ref) & (er['criterion'] == crit)]['earliest_W'].iloc[0]) for ref in REFS]
            L.append(f"| {model} | {crit} | " + ' | '.join(v) + ' |')
    sm = pd.DataFrame(srows)
    L += ['', '### H-09 item 1, Week 8 (student-cluster)', '', '| Predictor | Pooled AUC [CI] | Weighted within-semester AUC [CI] | Difference [CI] | Random-effects AUC [CI] | Prediction interval |', '|---|---|---|---|---|---|']
    for _, r in sm[sm['W'] == 'W8'].iterrows():
        L.append(f"| {r['model']} | {f3(r['pooled_auc'])} [{f3(r['pooled_lo'])}\u2013{f3(r['pooled_hi'])}] | {f3(r['weighted_mean_auc'])} [{f3(r['weighted_lo'])}\u2013{f3(r['weighted_hi'])}] | "
                 f"{f3(r['diff_weighted_minus_pooled'])} [{f3(r['diff_lo'])}; {f3(r['diff_hi'])}] | {f3(r['dl_auc'])} [{f3(r['dl_ci_lo'])}\u2013{f3(r['dl_ci_hi'])}] | {f3(r['dl_pi_lo'])}\u2013{f3(r['dl_pi_hi'])} |")
    L += ['', '### H-09 item 2, semester-stratified metrics, Weeks 8 and 10 (student-cluster)', '', '| Predictor | Week | AUC loss [95% hi; 99.3% hi] | r loss [95% hi; 99.3% hi] | Point | 95% | 99.3% |', '|---|---|---|---|---|---|---|']
    for _, r in sel[(sel['variant'] == 'semester_stratified') & sel['W'].isin(['W8', 'W10'])].iterrows():
        L.append(f"| {r['model']} | {r['W']} | {f3(r['auc_loss'])} [{f3(r['auc_loss_hi95'])}; {f3(r['auc_loss_hi993'])}] | {f3(r['r_loss'])} [{f3(r['r_loss_hi95'])}; {f3(r['r_loss_hi993'])}] | "
                 f"{'met' if r['point_ok'] else 'not met'} | {'met' if r['ci95_ok'] else 'not met'} | {'met' if r['ci993_ok'] else 'not met'} |")
    L += ['', '| Metric | Predictor | Point | 95% | 99.3% |', '|---|---|---|---|---|']
    for variant in ('pooled', 'semester_stratified'):
        for model in MODELS:
            v = {c: str(ws[(ws['variant'] == variant) & (ws['model'] == model) & (ws['criterion'] == c)]['W_star'].iloc[0]) for c in ('point', 'ci95', 'ci993')}
            L.append(f"| {variant} | {model} | {v['point']} | {v['ci95']} | {v['ci993']} |")
    L += ['', '| Metric | W_sel | Best | Second | Advantage [CI] | Equivalent | Chosen |', '|---|---|---|---|---|---|---|']
    for _, r in mcr.iterrows():
        L.append(f"| {r['variant']} | {r['W_sel']} | {r['best']} | {r['second']} | {f3(r['diff'])} [{f3(r['diff_lo'])}; {f3(r['diff_hi'])}] | {r['equivalent']} | {r['chosen']} |")
    o3 = pd.DataFrame(out3)
    L += ['', '### H-09 item 3, change in AUC versus the shipped M3\u2032 (student-cluster)', '', '| Week | Subset | Metric | Shipped AUC | a: change [CI] | b: change [CI] |', '|---|---|---|---|---|---|']
    for _, r in o3.iterrows():
        bc = 'NA' if pd.isna(r['d_b']) else f"{f3(r['d_b'])} [{f3(r['d_b_lo'])}; {f3(r['d_b_hi'])}]"
        L.append(f"| {r['W']} | {r['subset']} | {r['metric']} | {f3(r['auc_orig'])} | {f3(r['d_a'])} [{f3(r['d_a_lo'])}; {f3(r['d_a_hi'])}] | {bc} |")
    open(os.path.join(OUT, 'H0409c_summary_tables.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    log.close()
    print('done')


if __name__ == '__main__':
    main()
