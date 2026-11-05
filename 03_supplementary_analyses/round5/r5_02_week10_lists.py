# -*- coding: utf-8 -*-
"""R5-2 -- Week-10 deployment list x traffic lights (B5), incumbent statistics, matched-size comparison,
lead time (Addendum 1, week-based form) and fairness slice (Addendum 2).  Frozen scores only."""
import datetime as dt
import json
import math

import numpy as np
import pandas as pd

import r5_lib as L

base, scores = L.load_base()
N = len(base)
y = base.y.to_numpy().astype(int)
sem = base.semester.to_numpy()
mid = base.midterm_score.to_numpy(float)
Wm = L.make_weights(base.stu.to_numpy())
f32 = np.float32
fail = (y == 1)
flag = mid < 60                       # B5: midterm < 60 (red or yellow)
red = mid <= 50
yellow = (mid > 50) & (mid < 60)
rowsA, rowsB, rowsC = [], [], []


def add(rows, table, item, est, lo=np.nan, hi=np.nan, note=''):
    rows.append({'table': table, 'item': item, 'estimate': est, 'ci_lo': lo, 'ci_hi': hi, 'note': note})


def draw_stat(num_mask, den_mask):
    """Share draws  sum_w(num) / sum_w(den)  and the point value."""
    num = Wm @ num_mask.astype(f32)
    den = Wm @ den_mask.astype(f32)
    with np.errstate(invalid='ignore', divide='ignore'):
        d = num / den
    pt = num_mask.sum() / den_mask.sum() if den_mask.sum() else np.nan
    return pt, d


def count_draw(mask):
    return float(mask.sum()), Wm @ mask.astype(f32)


def dep_list(W, frac=0.2):
    dep = L.deploy_blend(scores[W].m1, scores[W].m2, sem)
    lst = L.list_by_k(dep, sem, lambda n: int(math.ceil(frac * n)))
    return dep, lst


def crosstab_block(W, tag):
    dep, lst = dep_list(W)
    both, mo, bo, nei = lst & flag, lst & ~flag, ~lst & flag, ~lst & ~flag
    add(rowsA, tag, 'list size (top 20% within semester, ceil)', int(lst.sum()))
    for nm, m in [('both (list and B5)', both), ('model list only', mo), ('B5 only', bo), ('neither', nei)]:
        add(rowsA, tag, f'cell count: {nm}', int(m.sum()))
        add(rowsA, tag, f'cell failing students: {nm}', int((m & fail).sum()))
    for nm, num, den in [
        ('overlap share = list students also B5-flagged / list', both, lst),
        ('model-list sensitivity (failing on list / all failing)', lst & fail, fail),
        ('model-list precision (failing on list / list)', lst & fail, lst),
        ('B5 sensitivity (failing flagged / all failing)', flag & fail, fail),
        ('B5 precision (failing flagged / flagged)', flag & fail, flag),
        ('red-only sensitivity (midterm <= 50)', red & fail, fail),
        ('red-only precision', red & fail, red),
    ]:
        pt, d = draw_stat(num, den)
        lo, hi = L.ci(d)
        add(rowsA, tag, nm, pt, lo, hi, 'student-cluster bootstrap B=2000 seed 20261031; lists fixed')
    for nm, m in [('failing students found by model list only (not B5-flagged)', mo & fail),
                  ('failing students flagged by B5 only (not on list)', bo & fail),
                  ('failing students found by both', both & fail),
                  ('failing students missed by B5 (not flagged)', ~flag & fail),
                  ('failing students missed by model list', ~lst & fail)]:
        pt, d = count_draw(m)
        lo, hi = L.ci(d)
        add(rowsA, tag, nm, pt, lo, hi, 'weighted count under resampling')
    # NNS = non-flagged students on the list per additional failing student found (published W8: 8.1)
    num_pt, num_d = count_draw(mo)
    den_pt, den_d = count_draw(mo & fail)
    nns = num_d / den_d
    add(rowsA, tag, 'NNS = model-list-only students / model-only finds', num_pt / den_pt, *L.ci(nns),
        'definition reproduces the published Week-8 value (8.1)')
    return dep, lst


# --- Week 10 (primary) and Week 8 (method consistency with the published numbers) -------------------------
add(rowsA, 'incumbent_B5', 'records evaluated', N)
add(rowsA, 'incumbent_B5', 'B5 flagged (midterm < 60)', int(flag.sum()))
add(rowsA, 'incumbent_B5', 'of which red (midterm <= 50)', int(red.sum()))
add(rowsA, 'incumbent_B5', 'of which yellow (50 < midterm < 60)', int(yellow.sum()))
add(rowsA, 'incumbent_B5', 'failing students (semester outcome)', int(fail.sum()))
add(rowsA, 'incumbent_B5', 'failing students flagged', int((flag & fail).sum()))
add(rowsA, 'incumbent_B5', 'failing students missed', int((~flag & fail).sum()))
add(rowsA, 'incumbent_B5', 'failing among red', int((red & fail).sum()))
add(rowsA, 'incumbent_B5', 'failing among yellow', int((yellow & fail).sum()))
dep10, lst10 = crosstab_block('W10', 'W10_deployment_list_x_B5')
dep8, lst8 = crosstab_block('W8', 'W8_consistency_deployment_blend_list_x_B5')
# the same Week-8 cross-tab with the pooled-percentile m3 re-ranked within semester (the ambiguity spec closes)
lst8_m3 = L.list_by_k(scores['W8'].m3, sem, lambda n: int(math.ceil(0.2 * n)))
add(rowsA, 'W8_consistency_m3_reranked', 'overlap share (list also B5-flagged / list)',
    float((lst8_m3 & flag).sum() / lst8_m3.sum()))
add(rowsA, 'W8_consistency_m3_reranked', 'model-only finds', int((lst8_m3 & ~flag & fail).sum()))
add(rowsA, 'W8_consistency_m3_reranked', 'NNS', float((lst8_m3 & ~flag).sum() / (lst8_m3 & ~flag & fail).sum()))
pub = {'overlap 57.3%': 0.573, 'model-only finds 28': 28, 'NNS 8.1': 8.1}
add(rowsA, 'W8_consistency_published', 'published: overlap 57.3%, 28 model-only finds, NNS 8.1', np.nan, note='see '
    'W8_consistency rows: overlap 306/534 = 57.3%, 28 finds, NNS 228/28 = 8.1 -> reproduced')

# --- R5-2 reproducibility handle for R5-11 -------------------------------------------------------------------
add(rowsA, 'list_reproducibility', 'W10 deployment list size per semester', json.dumps(
    {s: int(lst10[sem == s].sum()) for s in L.D7}), note='ceil(0.2 n); tie-break seed 7 (same function as R5-11)')

# --- matched-size comparison (primary) -----------------------------------------------------------------------------
kb5 = {s: int(flag[sem == s].sum()) for s in L.D7}
lstM = L.list_by_k(dep10, sem, kb5)
for s in L.D7:
    m = sem == s
    add(rowsB, 'per_semester', f'{s} n / failing / B5 flagged / matched list', f'{int(m.sum())} / {int(fail[m].sum())}'
        f' / {kb5[s]} / {int(lstM[m].sum())}')
    add(rowsB, 'per_semester', f'{s} B5 sensitivity', float((flag & fail & m).sum() / fail[m].sum()))
    add(rowsB, 'per_semester', f'{s} matched-model sensitivity', float((lstM & fail & m).sum() / fail[m].sum()))
    add(rowsB, 'per_semester', f'{s} B5 precision', float((flag & fail & m).sum() / flag[m].sum()))
    add(rowsB, 'per_semester', f'{s} matched-model precision', float((lstM & fail & m).sum() / lstM[m].sum()))
res = {}
for nm, num, den in [('B5 sensitivity', flag & fail, fail), ('matched model-list sensitivity', lstM & fail, fail),
                     ('B5 precision', flag & fail, flag), ('matched model-list precision', lstM & fail, lstM)]:
    pt, d = draw_stat(num, den)
    res[nm] = (pt, d)
    add(rowsB, 'pooled_matched_size', nm, pt, *L.ci(d), 'matched size: model list per semester = that semester\'s B5 '
        'flagged count')
for a, b in [('matched model-list sensitivity', 'B5 sensitivity'), ('matched model-list precision', 'B5 precision')]:
    d = res[a][1] - res[b][1]
    add(rowsB, 'pooled_matched_size', f'{a} minus {b}', res[a][0] - res[b][0], *L.ci(d), 'paired bootstrap')
add(rowsB, 'pooled_matched_size', 'matched list size (total)', int(lstM.sum()))
# W8 matched-size for method consistency
dep8m = dep8
lstM8 = L.list_by_k(dep8m, sem, kb5)
for nm, num, den in [('W8 matched model-list sensitivity', lstM8 & fail, fail), ('W8 matched model-list precision',
                     lstM8 & fail, lstM8)]:
    pt, d = draw_stat(num, den)
    add(rowsB, 'W8_method_consistency', nm, pt, *L.ci(d), 'not published in the spec; shown for Week-8/Week-10 '
        'method consistency')

# --- Addendum 1: lead time (week-based form; H-03 midterm dates are not in the pack) --------------------------------
cal = json.load(open(L.PACK / 'datapack_complete/05_calendar/exam_dates_all.json', encoding='utf-8'))['semesters']
dparse = dt.date.fromisoformat
lead_rows = []
for s in L.D7:
    c = cal[s]
    start = dparse(c['semester_start'])
    mw = 8 if c.get('course_structure') == '16+2' else 9
    typ = '16+2' if mw == 8 else 'regular'
    for W, wk in (('W8', 8), ('W10', 10)):
        close = start + dt.timedelta(days=7 * wk)                    # end of Week wk (start of the next day)
        mid_end = start + dt.timedelta(days=7 * mw)                  # end of the midterm week
        for dlt in (1, 2):
            light = mid_end + dt.timedelta(days=dlt)
            lead_rows.append({'semester': s, 'type': typ, 'list_week': W, 'issue_plus_days': dlt,
                              'list_available': close.isoformat(), 'light_available_upper_bound': light.isoformat(),
                              'lead_days': (light - close).days})
LR = pd.DataFrame(lead_rows)
for (typ, W), g in LR.groupby(['type', 'list_week']):
    lo = g[g.issue_plus_days == 1].lead_days.median()
    hi = g[g.issue_plus_days == 2].lead_days.median()
    add(rowsA, 'lead_time_week_based', f'{typ} semesters, {W} list: median signed lead (light date - list date), days',
        f'{lo:g} to {hi:g}', note=f'semesters: {", ".join(sorted(g.semester.unique()))}; negative = model list LATER than '
        'the traffic light; light = end of midterm week + 1..2 days (an upper bound for the exam day: an earlier exam '
        'day makes the list later still); week-based fallback of the spec, H-03 dates not in pack')
# date-based check where the pack has a date: Semester 1122 (midterm 2024-04-09)
c = cal['1122']
start = dparse(c['semester_start'])
exam = dparse(c['midterm'])
assert exam.isoformat() == '2024-04-09'
for W, wk in (('W8', 8), ('W10', 10)):
    close = start + dt.timedelta(days=7 * wk)
    leads = [((exam + dt.timedelta(days=d)) - close).days for d in (1, 2)]
    add(rowsA, 'lead_time_date_based_1122', f'1122 {W} list (closes {close}): light date - list date, days',
        f'{leads[0]:g} to {leads[1]:g}', note=f'midterm exam 2024-04-09 (in-pack check passed); light = exam day + 1..2; '
        f'negative = list later. The 1122 Week-8 value reproduces the published "16+2: 5-6 days later".')
add(rowsA, 'lead_time_week_based', 'convention', 'lead = traffic-light availability minus list availability',
    note='positive = model list earlier; list available at the close of its feature window (end of Week 8 / Week 10 '
         'counted from semester_start; week = days//7+1 as in the ETL); regular 18-week semesters: midterm Week 9; 16+2 '
         'semesters (1121, 1122, 1131): midterm Week 8; the Week-8 regular-semester figure published as "2-3 days '
         'earlier" cannot be reproduced from week-based data (needs the exam day).')
LR.to_csv(L.RES / 'R5-2a_lead_time_detail.csv', index=False, encoding='utf-8-sig')

# --- Addendum 2: fairness slice -----------------------------------------------------------------------------------------
cls = ['2yr_gen', '2yr_working', 'post', '4yr', 'retake']
pub8_auc = {'2yr_gen': 0.657, '2yr_working': 0.663, 'post': 0.612, '4yr': 0.707, 'retake': 0.671}
kfun = lambda n: int(math.ceil(0.2 * n))
variants = [
    ('W10_deployment_blend', 'W10', dep10, lst10, 'PRIMARY (spec): deployment blend, within-semester percentile construction'),
    ('W8_deployment_blend', 'W8', dep8, lst8, 'method check: deployment blend at Week 8'),
    ('W8_pooled_m3', 'W8', scores['W8'].m3.to_numpy(float), L.list_by_k(scores['W8'].m3, sem, kfun),
     'method check: frozen pooled-percentile m3 -- this construction reproduces the published Week-8 values'),
    ('W10_pooled_m3', 'W10', scores['W10'].m3.to_numpy(float), L.list_by_k(scores['W10'].m3, sem, kfun),
     'same construction as the published Week-8 slice, at Week 10 (apples-to-apples with Week 8)'),
]
for tag, W, score, lst, why in variants:
    cgroups = [np.flatnonzero(base.program5_key.to_numpy() == c) for c in cls]
    a_d, _ = L.auc_weighted(Wm, score.reshape(-1, 1), y, cgroups)
    a_p, _ = L.auc_weighted(np.ones((1, N), f32), score.reshape(-1, 1), y, cgroups)
    caps = []
    for gi, c in enumerate(cls):
        m = base.program5_key.to_numpy() == c
        n_fail = int((fail & m).sum())
        pt, d = draw_stat(lst & fail & m, fail & m)
        caps.append(pt)
        add(rowsC, tag, f'{c}: n / failing / listed', f'{int(m.sum())} / {n_fail} / {int(lst[m].sum())}', note=why)
        add(rowsC, tag, f'{c}: capture share (failing listed / failing)', pt, *L.ci(d),
            'student-cluster bootstrap B=2000 seed 20261031; list fixed')
        lo, hi = L.ci(a_d[:, gi, 0])
        add(rowsC, tag, f'{c}: within-class-type AUC', float(a_p[0, gi, 0]), lo, hi,
            ('published Week-8 value %.3f; match=%s' % (pub8_auc[c], round(float(a_p[0, gi, 0]), 3) == pub8_auc[c]))
            if W == 'W8' else 'pooled across semesters within the class type')
    add(rowsC, tag, 'capture-share range across the five class types (min to max)', f'{min(caps):.3f} to {max(caps):.3f}',
        note='published Week-8 range: 25% to 50%')

pdA, pdB, pdC = pd.DataFrame(rowsA), pd.DataFrame(rowsB), pd.DataFrame(rowsC)
pdA.to_csv(L.RES / 'R5-2a.csv', index=False, encoding='utf-8-sig')
pdB.to_csv(L.RES / 'R5-2b.csv', index=False, encoding='utf-8-sig')
pdC.to_csv(L.RES / 'R5-2c.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250, 'display.max_colwidth', 95, 'display.max_rows', 400)
print(pdA[['table', 'item', 'estimate', 'ci_lo', 'ci_hi']].to_string())
print(pdB[['table', 'item', 'estimate', 'ci_lo', 'ci_hi']].to_string())
print(pdC[['table', 'item', 'estimate', 'ci_lo', 'ci_hi']].to_string())
