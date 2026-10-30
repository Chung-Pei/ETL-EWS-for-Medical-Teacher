# -*- coding: utf-8 -*-
"""H-01: assessment calendar and midterm timing audit. Plan: plan/H01_plan.md. Run from the return root:
python -I H-01/h01_calendar_audit.py"""
import hashlib
import importlib.util
import os
import re
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
import calendar_lib as C  # noqa: E402
import linkage as K  # noqa: E402

P2 = os.path.join(ROOT, 'materials', 'pack2')
OUT = os.path.join(ROOT, 'H-01')
MOD = os.path.join(P2, '07_etl_v7_15_34', 'lms_etl', 'modules')
TRACK5 = {'2yr_gen': '2yr', '2yr_working': '2yr', 'post': 'postbac', '4yr': '4yr', 'retake': 'retake'}


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def load_mod(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(MOD, fn))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fmt_dates(s):
    return ';'.join(sorted({d.strftime('%Y-%m-%d') for d in s})) if len(s) else ''


def main():
    log = open(os.path.join(OUT, 'run.log'), 'w', encoding='utf-8')
    import platform
    print(f'python {platform.python_version()} pandas {pd.__version__}', file=log)
    qpath = os.path.join(P2, '02_events', 'events_quiz.csv')
    rpath = os.path.join(P2, '02_events', 'events_reading.csv')
    for p in (qpath, rpath, os.path.join(P2, '05_calendar', 'exam_dates_all.json')):
        print(f'sha256 {os.path.relpath(p, ROOT)} {sha(p)}', file=log)
    cal = C.load_calendar(P2)
    parse_tag = load_mod('m03_tags', '03_parse_tags.py').parse_tag

    q = pd.read_csv(qpath, encoding='utf-8-sig', dtype=str)
    q['ts'] = pd.to_datetime(q['c4_測驗時間'], errors='coerce')
    q = q.rename(columns={'c3_測驗標題': 'title'})
    kk = q['title'].map(C.classify_exam_title)
    q['kind'] = kk.map(lambda x: x[0])
    q['sitting'] = kk.map(lambda x: x[1])
    starts = {(s, t): C.effective_start(cal, s, t) for s, t in q[['semester', 'class_track']].drop_duplicates().itertuples(index=False)}
    q['start'] = [starts[(s, t)] for s, t in zip(q['semester'], q['class_track'])]
    q['week'] = [C.etl_week(a, b) for a, b in zip(q['ts'], q['start'])]

    # quiz title ranges (observed attempts, not official open/close dates)
    rng = q.groupby(['semester', 'title']).agg(n_events=('anon_id', 'size'), n_students=('anon_id', 'nunique'),
                                               first_attempt=('ts', 'min'), last_attempt=('ts', 'max'),
                                               tracks=('class_track', lambda s: ';'.join(sorted(set(s))))).reset_index()
    rng.to_csv(os.path.join(OUT, 'H01_quiz_title_ranges.csv'), index=False, encoding='utf-8-sig')

    # exam titles in quiz records
    ex = q[q['kind'].notna()]
    tt = ex.groupby(['semester', 'title', 'kind', 'sitting']).agg(
        n_events=('anon_id', 'size'), n_students=('anon_id', 'nunique'), first_attempt=('ts', 'min'),
        last_attempt=('ts', 'max'), tracks=('class_track', lambda s: ';'.join(sorted(set(s)))),
        first_week_etl=('week', 'min'), last_week_etl=('week', 'max')).reset_index()
    tt['etl_category'] = tt['title'].map(parse_tag)
    tt['enters_QUZ_features'] = tt['etl_category'] == 'QUZ'
    tt['enters_LSA_Q_events'] = tt['title'].str.contains('題庫', na=False)
    tt.to_csv(os.path.join(OUT, 'H01_exam_titles_in_quiz_records.csv'), index=False, encoding='utf-8-sig')

    # reading titles related to exams
    r = pd.read_csv(rpath, encoding='utf-8-sig', dtype=str).rename(columns={'c3_教材標題': 'title'})
    rr = r[r['title'].str.contains('期中|期末|考試|考前|模擬', na=False)]
    rr = rr.groupby(['semester', 'title']).agg(n_events=('anon_id', 'size'), n_students=('anon_id', 'nunique')).reset_index()
    rr['etl_category'] = rr['title'].map(parse_tag)
    rr.to_csv(os.path.join(OUT, 'H01_reading_exam_related_titles.csv'), index=False, encoding='utf-8-sig')

    # analysis set by track
    rec = K.records_df(K.load_step9(P2, 8)).merge(K.load_class5(P2), on=['semester', 'anon_id'], how='left')
    rec['class_track'] = rec['program5_key'].map(TRACK5)

    # student-level exam events (analysis set)
    piv = ex.groupby(['semester', 'anon_id', 'kind', 'sitting'])['ts'].min().unstack(['kind', 'sitting'])
    piv.columns = [f'{k}_{s}_ts' for k, s in piv.columns]
    piv = piv.reset_index()
    st = rec[['semester', 'anon_id', 'program5_key', 'class_track']].merge(piv, on=['semester', 'anon_id'], how='left')
    st['effective_start'] = [C.effective_start(cal, s, t) for s, t in zip(st['semester'], st['class_track'])]
    for c in ('midterm_main_ts', 'midterm_makeup_ts', 'final_main_ts', 'final_makeup_ts'):
        if c not in st:
            st[c] = pd.NaT
    st['midterm_main_week'] = [C.etl_week(t, s) if pd.notna(t) else None for t, s in zip(st['midterm_main_ts'], st['effective_start'])]
    st.to_csv(os.path.join(OUT, 'H01_student_exam_events.csv'), index=False, encoding='utf-8-sig')

    # calendar by semester x track
    rows = []
    for (sem, tr), g in q.groupby(['semester', 'class_track']):
        start = C.effective_start(cal, sem, tr)
        mm = g[(g['kind'] == 'midterm') & (g['sitting'] == 'main')]
        fm = g[(g['kind'] == 'final') & (g['sitting'] == 'main')]
        mk = g[(g['kind'] == 'midterm') & (g['sitting'] == 'makeup')]
        fk = g[(g['kind'] == 'final') & (g['sitting'] == 'makeup')]
        d0, d1 = mm['ts'].min().normalize(), mm['ts'].max().normalize()
        f0, f1 = fm['ts'].min().normalize(), fm['ts'].max().normalize()
        w8, w7 = C.cutoff(start, 8), C.cutoff(start, 7)
        an = rec[(rec['semester'] == sem) & (rec['class_track'] == tr)]
        ids = set(an['anon_id'])
        rows.append({'semester': int(sem), 'class_track': tr, 'structure': C.structure(cal, sem),
                     'effective_start': start.date(), 'n_analysis': len(an),
                     'midterm_first_date': d0.date(), 'midterm_last_date': d1.date(),
                     'midterm_first_week': C.etl_week(d0, start), 'midterm_last_week': C.etl_week(d1, start),
                     'midterm_makeup_dates': fmt_dates(mk['ts'].dt.normalize()),
                     'final_first_date': f0.date(), 'final_last_date': f1.date(),
                     'final_first_week': C.etl_week(f0, start), 'final_last_week': C.etl_week(f1, start),
                     'final_makeup_dates': fmt_dates(fk['ts'].dt.normalize()),
                     'n_with_main_midterm_event': len(ids & set(mm['anon_id'])),
                     'n_with_main_final_event': len(ids & set(fm['anon_id'])),
                     'W7_cutoff': w7.date(), 'W8_cutoff': w8.date(),
                     'midterm_first_before_W8_cutoff': bool(d0 < w8), 'midterm_last_before_W8_cutoff': bool(d1 < w8),
                     'days_first_midterm_to_W8_cutoff': (w8 - d0).days,
                     'redyellow_earliest': (d0 + pd.Timedelta(days=1)).date(),
                     'redyellow_latest': (d1 + pd.Timedelta(days=2)).date(),
                     'lead_days_low': (d0 + pd.Timedelta(days=1) - w8).days,
                     'lead_days_high': (d1 + pd.Timedelta(days=2) - w8).days})
    cd = pd.DataFrame(rows)
    cd['midterm_coverage'] = (cd['n_with_main_midterm_event'] / cd['n_analysis']).round(4)
    cd['final_coverage'] = (cd['n_with_main_final_event'] / cd['n_analysis']).round(4)
    cd.to_csv(os.path.join(OUT, 'H01_calendar_by_semester_track.csv'), index=False, encoding='utf-8-sig')

    # documentary feature table with grep line references
    docs = [
        ('quz_first_attempt_accuracy', '04_feature_engineering.py', 'quz_first_attempt_accuracy', 'quiz events, category QUZ ([題庫]) only', 'no', 'exam sittings are [重要] -> UNTAGGED, excluded'),
        ('quz_final_accuracy', '04_feature_engineering.py', 'quz_final_accuracy', 'quiz events, category QUZ only', 'no', 'same'),
        ('quz_score_delta', '04_feature_engineering.py', 'quz_score_delta', 'quiz events, category QUZ only', 'no', 'same'),
        ('quz_total_attempts', '04_feature_engineering.py', 'quz_total_attempts', 'quiz events, category QUZ only', 'no', 'same'),
        ('quz_pass_rate', '04_feature_engineering.py', 'quz_pass_rate', 'quiz events, category QUZ only', 'no', 'same'),
        ('quz_cramming_ratio', '04_feature_engineering.py', 'quz_cramming_ratio', 'QUZ attempts within 7 days before configured midterm/final dates', 'yes (configured exam dates)', 'depends on which exam dates the frozen run had in its config'),
        ('active_weeks', '05_time_analysis.py', 'active_weeks', 'reading-log events by week from semester start (all reading titles)', 'no', 'window content changes with the cutoff'),
        ('consistency_score', '05_time_analysis.py', 'consistency_score', 'reading-log weekly rhythm', 'no', 'same'),
        ('weekly_minutes_std', '05_time_analysis.py', 'weekly_minutes_std', 'reading-log weekly minutes', 'no', 'same'),
        ('cram_pattern_score', '05_time_analysis.py', 'cram_pattern_score', 'reading-log minutes within 7 days before configured exam dates / total minutes', 'yes (configured exam dates)', 'same caveat as quz_cramming_ratio'),
        ('late_night_ratio', '05_time_analysis.py', 'late_night_ratio', 'reading-log hour-of-day distribution', 'no', ''),
        ('early_start_ratio', '05_time_analysis.py', 'early_start_ratio', 'reading-log minutes before (earliest exam date - 21 days) / total', 'yes (configured exam dates)', 'same caveat'),
        ('total_learning_minutes', '04_feature_engineering.py', 'total_learning_minutes', 'sum of reading minutes, all categories', 'no', ''),
        ('reading_inflation_ratio', '05_time_analysis.py', 'reading_inflation_ratio', 'reading integrity: raw vs capped minutes', 'no', ''),
        ('policy_gaming_flag', '05_time_analysis.py', 'policy_gaming_flag', 'reading integrity: raw meets, capped does not', 'no', ''),
        ('MM_ratio / MQ_ratio / QM_ratio / QQ_ratio', '09_lsa_event_log.py', 'BEHAVIORS', 'Lag-1 pairs of reading events (all titles) and quiz events whose title contains 題庫', 'no', 'exam sittings do not contain 題庫, so excluded'),
        ('s_cluster_encoded', '10_lsa_transition.py', 'S1', 'cluster from LSA transition matrix of the same event log', 'no', ''),
        ('is_4tech', '11_cross_analysis.py', 'is_4tech', 'class proxy (semester type)', 'no', ''),
        ('pre_exam_intensity (BAS passive term, not in FS_C)', '06_merge_grades.py', 'pre_exam_intensity', 'exam-proximity measure used by BAS', 'yes (configured exam dates)', ''),
    ]
    drows = []
    for feat, fn, pat, src, dep, note in docs:
        lines = open(os.path.join(MOD, fn), encoding='utf-8').read().splitlines()
        ln = next((i + 1 for i, s in enumerate(lines) if pat in s and not s.strip().startswith('#')), None)
        drows.append({'feature': feat, 'etl_file': f'lms_etl/modules/{fn}', 'first_reference_line': ln, 'source_events': src,
                      'depends_on_exam_dates': dep, 'note': note})
    pd.DataFrame(drows).to_csv(os.path.join(OUT, 'H01_feature_exam_dependence.csv'), index=False, encoding='utf-8-sig')

    # memo tables generated from the output files
    L = ['### Calendar by semester and class track (dates; weeks by the ETL rule)', '',
         '| Semester | Track | Structure | Start | Midterm dates (week) | Final dates (week) | W8 cutoff | Midterm before W8 cutoff | Red/yellow date range | Lead days (warning earlier = +) |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for _, r in cd.iterrows():
        mdt = f"{r['midterm_first_date']}" + (f" to {r['midterm_last_date']}" if r['midterm_last_date'] != r['midterm_first_date'] else '')
        fdt = f"{r['final_first_date']}" + (f" to {r['final_last_date']}" if r['final_last_date'] != r['final_first_date'] else '')
        mw = f"{r['midterm_first_week']}" if r['midterm_first_week'] == r['midterm_last_week'] else f"{r['midterm_first_week']}-{r['midterm_last_week']}"
        fw = f"{r['final_first_week']}" if r['final_first_week'] == r['final_last_week'] else f"{r['final_first_week']}-{r['final_last_week']}"
        lead = f"{r['lead_days_low']} to {r['lead_days_high']}"
        L.append(f"| {r['semester']} | {r['class_track']} | {r['structure']} | {r['effective_start']} | {mdt} (W{mw}) | {fdt} (W{fw}) | "
                 f"{r['W8_cutoff']} | {'yes' if r['midterm_first_before_W8_cutoff'] else 'no'} | {r['redyellow_earliest']} to {r['redyellow_latest']} | {lead} |")
    L += ['', f"Exam titles in quiz records: {len(tt)} titles, {int(tt['n_events'].sum())} events; ETL category UNTAGGED for {int((tt['etl_category'] == 'UNTAGGED').sum())}; "
          f"titles entering QUZ features: {int(tt['enters_QUZ_features'].sum())}; titles entering LSA Q events: {int(tt['enters_LSA_Q_events'].sum())}.",
          f"Share of all quiz events that are exam sittings: {tt['n_events'].sum() / len(q):.4f}."]
    open(os.path.join(OUT, 'H01_summary_tables.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')

    print(f'calendar rows={len(cd)} exam titles={len(tt)} students={len(st)}', file=log)
    log.close()
    print(f'calendar rows={len(cd)} exam titles={len(tt)} students={len(st)}')


if __name__ == '__main__':
    main()
