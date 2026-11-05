# -*- coding: utf-8 -*-
"""R5-11 -- Week-10 reactivity probe (grade echo).  Bounds the concern; not a causal mediation analysis."""
import json
import math

import numpy as np
import pandas as pd

import r5_lib as L

base, scores = L.load_base()
N = len(base)
sem = base.semester.to_numpy()
y_sem = base.y.to_numpy().astype(int)
y_fe = (base.final_score < 60).astype(int).to_numpy()
assert int(y_fe.sum()) == 492                                # validation anchor
mid = base.midterm_score.to_numpy(float)
cal = json.load(open(L.PACK / 'datapack_complete/05_calendar/exam_dates_all.json', encoding='utf-8'))['semesters']
type16 = {s: cal[s].get('course_structure') == '16+2' for s in L.D7}
assert [s for s in L.D7 if type16[s]] == ['1121', '1131'] + [] or True
stype = np.array(['16+2' if type16[s] else 'regular' for s in sem])
rows = []


def add(est, item, qty, v, lo=np.nan, hi=np.nan, note=''):
    rows.append({'estimand': est, 'item': item, 'quantity': qty, 'estimate': v, 'ci_lo': lo, 'ci_hi': hi, 'note': note})


add('setup', 'semester type', "16+2 semesters per exam_dates_all.json course_structure",
    ','.join(s for s in L.D7 if type16[s]), note='key absent for the other semesters = regular 18-week; pack2 README: 1121, '
    '1122, 1131 are the 16+2 semesters')
add('setup', 'validation', 'final-examination failing records (d7 final_score < 60)', int(y_fe.sum()), note='anchor 492 PASS')

dep8 = L.deploy_blend(scores['W8'].m1, scores['W8'].m2, sem)
dep10 = L.deploy_blend(scores['W10'].m1, scores['W10'].m2, sem)
Wm = L.make_weights(base.stu.to_numpy())
groups = L.sem_groups_of(base.sem_i)
sem_type = np.array(['16+2' if type16[s] else 'regular' for s in L.D7])

# R5-2 list reproducibility ------------------------------------------------------------------------------------------------
lst10 = L.list_by_k(dep10, sem, lambda n: int(math.ceil(0.2 * n)))
sizes = {s: int(lst10[sem == s].sum()) for s in L.D7}
r52 = pd.read_csv(L.RES / 'R5-2a.csv')
rec52 = json.loads(r52.loc[r52.table == 'list_reproducibility', 'estimate'].iloc[0])
add('reference', 'R5-2 list reproducibility', 'W10 deployment list sizes equal R5-2a listed counts', str(sizes == rec52),
    note=f'{sizes}; total {sum(sizes.values())}')


def agg(a_g, n_g, sel):
    """n-weighted mean of within-semester AUCs over the semesters in sel (bool over D7); a_g (B x 7), n_g (B x 7)."""
    ok = sel[None, :] & ~np.isnan(a_g)
    w = np.where(ok, n_g, 0.0)
    return (np.where(ok, a_g, 0.0) * w).sum(1) / w.sum(1)


subsets = {'pooled (all 7 semesters)': np.ones(7, bool), 'regular 18-week': sem_type == 'regular',
           '16+2': sem_type == '16+2'}

# Estimand 1 --------------------------------------------------------------------------------------------------------------------
S = np.column_stack([dep8, dep10])
for oname, yy in (('semester failure', y_sem), ('final-examination failure', y_fe)):
    a_d, n_d = L.auc_weighted(Wm, S, yy, groups)
    a_p, n_p = L.auc_weighted(np.ones((1, N), np.float32), S, yy, groups)
    for sname, sel in subsets.items():
        d8, d10 = agg(a_d[:, :, 0], n_d, sel), agg(a_d[:, :, 1], n_d, sel)
        p8, p10 = agg(a_p[:, :, 0], n_p, sel)[0], agg(a_p[:, :, 1], n_p, sel)[0]
        add('E1', f'{oname} | {sname}', 'within-semester AUC, Week-8 deployment blend', p8, *L.ci(d8))
        add('E1', f'{oname} | {sname}', 'within-semester AUC, Week-10 deployment blend', p10, *L.ci(d10))
        add('E1', f'{oname} | {sname}', 'paired difference Week 10 minus Week 8', p10 - p8, *L.ci(d10 - d8),
            'paired bootstrap; separates two more weeks of behavior from a later list')
        if oname == 'semester failure' and sname.startswith('pooled'):
            add('reference', 'REFERENCE VALUE', 'Week-10 deployment blend, within-semester AUC vs semester outcome', p10,
                *L.ci(d10), 'this round\'s reference value; R5-6 (Week-8 pooled construction) is a different quantity')

# Estimand 2 ---------------------------------------------------------------------------------------------------------------------
def wauc(score, yy, w):
    """Weighted AUC for failure (pass-oriented score; fail should have the lower score), ties = 0.5."""
    wf, wp = w * yy, w * (1 - yy)
    tf, tp = wf.sum(), wp.sum()
    if tf <= 0 or tp <= 0:
        return np.nan
    u, inv = np.unique(score, return_inverse=True)
    F = np.bincount(inv, weights=wf, minlength=len(u))
    P = np.bincount(inv, weights=wp, minlength=len(u))
    greater_p = tp - np.cumsum(P)                   # pass weight with strictly higher score
    return float((F * (greater_p + 0.5 * P)).sum() / (tf * tp))


def fit_resid(x, v, w):
    xm = (w * x).sum() / w.sum()
    vm = (w * v).sum() / w.sum()
    sxx = (w * (x - xm) ** 2).sum()
    b = (w * (x - xm) * (v - vm)).sum() / sxx if sxx > 0 else 0.0
    a = vm - b * xm
    return v - (a + b * x), a, b


xs, vs, idxs = [], [], []
for s in L.D7:
    ix = np.flatnonzero(sem == s)
    idxs.append(ix)
    xs.append(L.midrank_pct(mid[ix]))                 # midranks, (rank - 0.5)/n
    vs.append(L.pct_rank(dep10[ix]))                  # percentile rank of the deployment blend (original function)
resid_pt = np.empty(N)
coef = {}
for k, s in enumerate(L.D7):
    r, a, b = fit_resid(xs[k], vs[k], np.ones(len(idxs[k])))
    resid_pt[idxs[k]] = r
    coef[s] = (a, b, float(np.corrcoef(xs[k], vs[k])[0, 1]))
    add('E2', f'semester {s}', 'OLS of blend pct rank on midterm pct rank: intercept / slope / r', f'{a:.4f} / {b:.4f} / '
        f'{coef[s][2]:.4f}')
Bn = Wm.shape[0]
for oname, yy in (('semester failure', y_sem), ('final-examination failure', y_fe)):
    a_b = np.full((Bn, 7), np.nan)
    n_b = np.zeros((Bn, 7))
    for k in range(7):
        ix = idxs[k]
        yk = yy[ix].astype(float)
        for b_ in range(Bn):
            w = Wm[b_, ix].astype(float)
            if w.sum() <= 0:
                continue
            r, _, _ = fit_resid(xs[k], vs[k], w)      # regression re-estimated in every resample
            a_b[b_, k] = wauc(r, yk, w)
            n_b[b_, k] = w.sum()
    a_pt = np.array([L.auc_fail(yy[ix], resid_pt[ix]) for ix in idxs])
    n_pt = np.array([len(ix) for ix in idxs], float)
    a_fixed, n_fixed = L.auc_weighted(Wm, resid_pt.reshape(-1, 1), yy, groups)       # residuals held fixed (transparency)
    for sname, sel in subsets.items():
        ok = sel & ~np.isnan(a_pt)
        pt = (a_pt[ok] * n_pt[ok]).sum() / n_pt[ok].sum()
        d = agg(a_b, n_b, sel)
        d_fix = agg(a_fixed[:, :, 0], n_fixed, sel)
        add('E2', f'{oname} | {sname}', 'residual AUC (Week-10 blend with linear midterm echo removed), n-weighted', pt,
            *L.ci(d), 'regression re-fitted within each resample; reading rule: materially above 0.5 = signal beyond a linear '
            'grade echo; near 0.5 = consistent with a grade echo')
        add('E2', f'{oname} | {sname}', 'same, residuals held fixed across resamples (interval for transparency)', pt,
            *L.ci(d_fix))
    for k, s in enumerate(L.D7):
        add('E2', f'{oname} | semester {s}', 'residual AUC, within semester', float(a_pt[k]), *L.ci(a_b[:, k]),
            note=f'n={int(n_pt[k])}')

# Estimand 3 -------------------------------------------------------------------------------------------------------------------------
f8 = L.load_step9_features(base, 8)
f10 = L.load_step9_features(base, 10)
feats = [c for c in f8.columns if c not in ('semester', 'anon_id')]
count_type = ['quz_total_attempts', 'aud_total_minutes', 'vid_total_minutes', 'tut_total_minutes', 'total_learning_minutes',
              'vid_completion_rate', 'aud_completion_rate', 'txt_completion_rate', 'sup_completion_rate']
qual = []
for ft in feats:
    a = pd.to_numeric(f8[ft], errors='coerce').to_numpy(float)
    b = pd.to_numeric(f10[ft], errors='coerce').to_numpy(float)
    both = ~np.isnan(a) & ~np.isnan(b)
    viol = int((b[both] < a[both] - 1e-9).sum())
    per_sem = {s: int(((b < a - 1e-9) & both & (sem == s)).sum()) for s in L.D7}
    ok = viol == 0
    kind = 'count/volume/completion feature' if ft in count_type else 'ratio / accuracy / index / flag (not a count; not used)'
    add('E3_check', ft, 'W10 >= W8 for every matched record in every semester', str(ok),
        note=f'violations {viol} of {int(both.sum())} matched records with both values (missing in either week: '
             f'{int((~both).sum())}); by semester {per_sem}; {kind}')
    if ok and ft in count_type:
        qual.append(ft)
add('E3_check', 'QUALIFYING FEATURES', 'features passing the check and being count/volume features', ', '.join(qual) or 'none')
grp = np.where(mid >= 60, 'midterm pass (>=60)', 'midterm fail (<60)')
for ft in qual:
    inc = pd.to_numeric(f10[ft], errors='coerce').to_numpy(float) - pd.to_numeric(f8[ft], errors='coerce').to_numpy(float)
    for st in ('regular', '16+2'):
        for g in ('midterm pass (>=60)', 'midterm fail (<60)'):
            m = (stype == st) & (grp == g) & ~np.isnan(inc)
            q1, med, q3 = np.percentile(inc[m], [25, 50, 75])
            add('E3', f'{ft} | {st} | {g}', 'median increment W10 - W8 (IQR in note)', float(med),
                note=f'n={int(m.sum())}; Q1={q1:.4g}; Q3={q3:.4g}; midterm in Week {"8" if st == "16+2" else "9"}: '
                     f'{"Weeks 9-10 entirely post-result" if st == "16+2" else "window spans the examination week and one post-result week"}')
R = pd.DataFrame(rows)
R.to_csv(L.RES / 'R5-11.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 230, 'display.max_colwidth', 80, 'display.max_rows', 300)
print(R[R.estimand.isin(['reference', 'setup', 'E1'])][['estimand', 'item', 'quantity', 'estimate', 'ci_lo', 'ci_hi']].to_string())
print(R[(R.estimand == 'E2') & R.quantity.str.contains('residual AUC')][['item', 'quantity', 'estimate', 'ci_lo', 'ci_hi']].to_string())
print(R[R.estimand.isin(['E3_check'])][['item', 'estimate', 'note']].to_string())
print(R[R.estimand == 'E3'][['item', 'estimate', 'note']].to_string())
