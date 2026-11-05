# -*- coding: utf-8 -*-
"""R5-10 -- tabulation of separately recorded individual grade adjustments (all eight semesters).

Source: supplement gradebooks (S-code version), lecture sheets only (the sheets that carry a semester-score column;
lab ('實驗') sheets carry no semester score and are not part of the analysed outcome).
Operational definition (declared): an adjustment counts when the entry is numeric and HIGHER than the grade-book semester
score of the same row.  Entries at or below the score, fractional working annotations at or below the score, and text
entries do not count."""
import os
import re
import sys

import numpy as np
import pandas as pd

import r5_lib as L

GB = os.environ.get('R5_GB', str(L.PACK.parent / 'supplement' / 'gradebooks'))
SEMS = ['1111', '1112', '1121', '1122', '1131', '1132', '1141', '1142']

# ---- in-pack validation anchor (outcomes_1142.csv) -----------------------------------------------------------------------------
o = pd.read_csv(L.PACK / 'datapack_complete/08_class_membership_and_1142_outcomes/outcomes_1142.csv')
o['comp'] = (o.midterm_score + o.final_exam_score) / 2
adj = o[o.semester_score_adjusted.notna()]
n_adj = len(adj)
n_60 = int(((adj.semester_score_adjusted == 60) & (adj.comp < 60)).sum())
n_60_exact = int((adj.semester_score_adjusted == 60).sum())
anchor_ok = (n_adj == 35) and (n_60 == 27) and (n_60_exact == 27)
print(f'anchor outcomes_1142: non-empty {n_adj} (35), exactly 60 {n_60_exact}, exactly 60 with composite < 60 {n_60} (27) -> '
      f'{"PASS" if anchor_ok else "FAIL"}')
assert anchor_ok, 'R5-10 anchor failed: stop this item and say so in the memo'

# ---- parse gradebooks ------------------------------------------------------------------------------------------------------------
rec = []
sheets_used, sheets_skipped = {}, {}
for s in SEMS:
    xl = pd.ExcelFile(os.path.join(GB, f'gradebook_{s}.xls'), engine='calamine')
    sheets_used[s], sheets_skipped[s] = [], []
    for sh in xl.sheet_names:
        df = xl.parse(sh, header=None, dtype=object)
        hdr = [str(x).strip() if pd.notna(x) else '' for x in df.iloc[0].tolist()]
        if '學期成績' not in hdr:
            sheets_skipped[s].append(sh.strip())
            continue
        sheets_used[s].append(sh.strip())
        ci_ = {n: hdr.index(n) for n in ['學號', '期中考', '期末考', '學期成績', '調整']}
        for i in range(1, len(df)):
            v = df.iloc[i]
            code = str(v[ci_['學號']]).strip() if pd.notna(v[ci_['學號']]) else ''
            if not re.match(r'^S\d+$', code):
                continue                                   # class mean / SD / summary rows
            rec.append({'semester': s, 'sheet': sh.strip(), 'scode': code, 'mid': v[ci_['期中考']], 'fin': v[ci_['期末考']],
                        'sem_raw': v[ci_['學期成績']], 'adj_raw': v[ci_['調整']]})
R = pd.DataFrame(rec)
R['sem_score'] = pd.to_numeric(R['sem_raw'], errors='coerce')
R['adj_txt'] = R['adj_raw'].map(lambda x: '' if pd.isna(x) else str(x).strip())
R['adj_num'] = pd.to_numeric(R['adj_txt'].replace('', np.nan), errors='coerce')
R['nonempty'] = R['adj_txt'] != ''
R['evaluable'] = R['sem_score'].notna()
R['counted'] = R['nonempty'] & R['adj_num'].notna() & R['evaluable'] & (R['adj_num'] > R['sem_score'])
R['crossed'] = R['counted'] & (R['sem_score'] < 60) & (R['adj_num'] >= 60)
R['increase'] = np.where(R['counted'], R['adj_num'] - R['sem_score'], np.nan)
R['excl_text'] = R['nonempty'] & R['adj_num'].isna()
R['excl_le_score'] = R['nonempty'] & R['adj_num'].notna() & R['evaluable'] & (R['adj_num'] <= R['sem_score'])
R['excl_fractional_le'] = R['excl_le_score'] & (R['adj_num'] % 1 != 0)
R['excl_no_score'] = R['nonempty'] & R['adj_num'].notna() & ~R['evaluable']
R['counted_fractional'] = R['counted'] & (R['adj_num'] % 1 != 0)

# anchor cross-check: gradebook 1142 versus outcomes_1142.csv
g42 = R[(R.semester == '1142') & R.nonempty & R.adj_num.notna()]
x_ok = (len(g42) == n_adj) and (int(g42.counted.sum()) == int((adj.semester_score_adjusted > adj.semester_score).sum()))
print(f'grade book 1142 vs outcomes_1142: numeric non-empty {len(g42)} vs {n_adj}; counted {int(g42.counted.sum())} vs '
      f'{int((adj.semester_score_adjusted > adj.semester_score).sum())} -> {"consistent" if x_ok else "DIFFERENT"}')

analysis_n = pd.read_csv(L.PACK / 'oof_scores_keyed.csv', usecols=['semester', 'arm', 'feature_set', 'W'],
                         dtype={'semester': str}).query("arm=='P' and feature_set=='C' and W=='W8'").semester.value_counts().to_dict()
rows = []
for s in SEMS + ['D7 (1111-1141)', 'ALL EIGHT']:
    sub = R[R.semester.isin(SEMS[:7])] if s.startswith('D7') else (R if s == 'ALL EIGHT' else R[R.semester == s])
    c = sub[sub.counted]
    rows.append({
        'semester': s,
        'gradebook_records (lecture sheets, S-code rows)': len(sub),
        'unique_S_codes': sub.scode.nunique() if not s.startswith(('D7', 'ALL')) else np.nan,
        'records_without_numeric_semester_score': int((~sub.evaluable).sum()),
        'nonempty_adjustment_entries': int(sub.nonempty.sum()),
        'COUNTED_adjustments (entry > semester score)': len(c),
        'crossed_pass_boundary (score<60, adjusted>=60)': int(sub.crossed.sum()),
        'increase_min': float(c.increase.min()) if len(c) else np.nan,
        'increase_median': float(c.increase.median()) if len(c) else np.nan,
        'increase_max': float(c.increase.max()) if len(c) else np.nan,
        'counted_entries_that_are_fractional': int(sub.counted_fractional.sum()),
        'excluded_entries_total': int((sub.nonempty & ~sub.counted).sum()),
        'excluded: numeric <= semester score': int(sub.excl_le_score.sum()),
        'excluded: of which fractional working annotations': int(sub.excl_fractional_le.sum()),
        'excluded: text entries': int(sub.excl_text.sum()),
        'excluded: on rows with no numeric semester score': int(sub.excl_no_score.sum()),
        'analysis_set_records (keyed file)': (analysis_n.get(s, np.nan) if s in SEMS[:7] else
                                              (sum(analysis_n.values()) if s.startswith('D7') else np.nan)),
        'counts_restricted_to_analysis_set': 'not available (no S-code -> anon_id bridge in the pack)' if s != '1142' else
        'not applicable (sealed holdout, not in the 2,652-record analysis set)',
    })
T = pd.DataFrame(rows)
T.to_csv(L.RES / 'R5-10.csv', index=False, encoding='utf-8-sig')
notes = pd.DataFrame([
    {'item': 'operational definition', 'value': 'an adjustment counts when the entry is numeric and higher than the grade-book '
     'semester score of the same row'},
    {'item': 'anchor outcomes_1142.csv', 'value': f'non-empty {n_adj}; exactly 60: {n_60_exact}; exactly 60 with pure composite < 60: '
     f'{n_60} -> PASS'},
    {'item': 'grade book 1142 cross-check', 'value': f'numeric non-empty {len(g42)} vs outcomes {n_adj}; counted {int(g42.counted.sum())}'},
    {'item': 'sheets used (have a semester-score column)', 'value': str({k: v for k, v in sheets_used.items()})},
    {'item': 'sheets skipped (lab sheets, no semester-score column)', 'value': str({k: v for k, v in sheets_skipped.items()})},
])
notes.to_csv(L.RES / 'R5-10_notes.csv', index=False, encoding='utf-8-sig')
R.drop(columns=['mid', 'fin', 'sem_raw']).to_csv(L.RES / 'R5-10_record_level_no_names.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 260, 'display.max_columns', 30, 'display.max_colwidth', 60)
print(T.drop(columns=['counts_restricted_to_analysis_set']).T.to_string())
print('counted rows that are fractional:', R[R.counted_fractional][['semester', 'sem_score', 'adj_num']].to_string())
print('text entries:', R[R.excl_text].adj_txt.value_counts().to_dict())
