# -*- coding: utf-8 -*-
"""H-07 event rebuild of the event-derived features (post hoc). Spec: H07_addendum2_plan.md section 2.
Uses the ETL v7.15.34 module code unchanged; mirrors lms_etl.py lines ~1830-1925 (feature loop) and ~2282-2340 (LSA)."""
import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import pa_common as C  # noqa: E402

L = C.L
MOD_DIR = os.path.join(C.PACK, '02_pipeline', 'etl_v7_15_34', 'lms_etl', 'modules')
CAL = os.path.join(C.PACK, '03_calendar', 'H01_calendar_by_semester_track.csv')
TRACK = {'2yr_gen': '2yr', '2yr_working': '2yr', 'post': 'postbac', '4yr': '4yr', 'retake': 'retake'}
POLICY_HOURS = 24.0
FEATS = ['quz_first_attempt_accuracy', 'quz_final_accuracy', 'quz_score_delta', 'quz_total_attempts', 'quz_pass_rate',
         'quz_cramming_ratio', 'consistency_score', 'policy_gaming_flag', 'pre_exam_intensity',
         'reading_inflation_ratio', 'total_learning_minutes']
TIME5 = ['active_weeks', 'weekly_minutes_std', 'cram_pattern_score', 'late_night_ratio', 'early_start_ratio']


def mods():
    m = {}
    for key, fn in [('m03', '03_parse_tags.py'), ('m04', '04_feature_engineering.py'), ('m05', '05_time_analysis.py'),
                    ('m07c', '07c_stability.py'), ('m09', '09_lsa_event_log.py'), ('m10', '10_lsa_transition.py')]:
        m[key] = L.load_mod(os.path.join(MOD_DIR, fn), 'h07_' + key)
    return m


def load_calendar():
    cal = pd.read_csv(CAL, encoding='utf-8-sig', dtype={'semester': str})
    out = {}
    for r in cal.itertuples():
        out[(r.semester, r.class_track)] = {
            'semester_start': r.effective_start, 'midterm': r.midterm_first_date, 'final': r.final_first_date,
            'total_weeks': 18}
    return out


def load_events():
    q = pd.read_csv(os.path.join(C.IN, 'events_quiz.csv'), dtype=str)
    r = pd.read_csv(os.path.join(C.IN, 'events_reading.csv'), dtype=str)
    q = q.rename(columns={'c3_測驗標題': '測驗標題', 'c4_測驗時間': '測驗時間', 'c5_狀態': '狀態', 'semester': '_semester'})
    r = r.rename(columns={'c3_教材標題': '教材標題', 'c4_開始': '開始', 'c5_結束': '結束', 'c6_時間': '時間', 'semester': '_semester'})
    return q, r


def week_of(ts, start):
    return ((ts - start).dt.days // 7) + 1


def truncate(df, time_col, cal, week_limit=8, cut_days=None):
    """Keep events with week <= week_limit (lms_etl._truncate_by_week rule), or, if cut_days (dict (sem, track) -> days)
    is given, events strictly before start + cut_days (used by the strict-cutoff arms)."""
    ts = pd.to_datetime(df[time_col], errors='coerce')
    key = list(zip(df['_semester'], df['class_track']))
    start = pd.to_datetime(pd.Series([cal[k]['semester_start'] for k in key], index=df.index))
    if cut_days is None:
        keep = ts.isna() | (week_of(ts, start) <= week_limit)
    else:
        lim = pd.Series([cut_days[k] for k in key], index=df.index)
        keep = ts.isna() | ((ts - start).dt.days < lim)
    return df[keep].reset_index(drop=True)


def rebuild(week_limit=8, cut_days=None, drop_before=None, log=print):
    """Return (feat_df, time_df, lsa_type, ratios) keyed by (anon_id, semester).
    drop_before: optional dict (sem, track) -> (lo_days, hi_days) removing events with lo <= days-since-start < hi."""
    M = mods()
    cal = load_calendar()
    q, r = load_events()
    with contextlib.redirect_stdout(io.StringIO()):
        q = M['m03'].categorize_activity_df(q, '測驗標題')
    q = truncate(q, '測驗時間', cal, week_limit, cut_days)
    r = truncate(r, '開始', cal, week_limit, cut_days)
    q['anon_id'] = q['anon_id'].astype(str).str.strip()
    r['anon_id'] = r['anon_id'].astype(str).str.strip()
    bounds = M['m05'].compute_material_outlier_bounds(r)
    gq = {k: g for k, g in q.groupby(['anon_id', '_semester'])}
    gr = {k: g for k, g in r.groupby(['anon_id', '_semester'])}
    pairs = set(gq) | set(gr)
    ana = C.frozen_frame(8)
    keyset = set(zip(ana.anon_id, ana.semester))
    pairs |= keyset
    cm = C.load_class5()
    trk = {(a, s): TRACK[p] for a, s, p in zip(cm.anon_id, cm.semester, cm.program5_key)}
    empty_q, empty_r = q.iloc[0:0], r.iloc[0:0]
    students, feat_rows, time_rows = [], [], []
    for aid, sem in sorted(pairs):
        track = trk.get((aid, sem))
        if track is None:
            sub = gq.get((aid, sem), gr.get((aid, sem)))
            track = sub['class_track'].iloc[0]
        ed = dict(cal[(sem, track)])
        sq = gq.get((aid, sem), empty_q)
        sr = gr.get((aid, sem), empty_r)
        with contextlib.redirect_stdout(io.StringIO()):
            f = M['m04'].compute_quiz_features(sq, aid, ed)
            tp = M['m05'].compute_student_time_profile(aid, sr, ed, ed['semester_start'], int(ed['total_weeks']), bounds, POLICY_HOURS)
        f = dict(f)
        f['consistency_score'] = tp.get('consistency_score')
        f['total_learning_minutes'] = tp.get('capped_total_minutes')
        f['reading_inflation_ratio'] = tp.get('reading_inflation_ratio')
        f['policy_gaming_flag'] = tp.get('policy_gaming_flag')
        pre_mid = tp.get('pre_midterm_7d_minutes') or 0.0
        pre_fin = tp.get('pre_final_7d_minutes') or 0.0
        total_min = f.get('total_learning_minutes') or 0
        f['pre_exam_intensity'] = 0.0 if total_min == 0 else round((pre_mid + pre_fin) / ((total_min / int(ed['total_weeks'])) * 7), 4)
        students.append({'anon_id': aid, 'semester': sem, 'features': f, 'time_profile': tp})
        feat_rows.append({'anon_id': aid, 'semester': sem, 'track': track, **{k: f.get(k) for k in FEATS}})
        time_rows.append({'anon_id': aid, 'semester': sem, **{k: tp.get(k) for k in TIME5}})
    with contextlib.redirect_stdout(io.StringIO()):
        cv = M['m07c'].compute_stability_cv(students)
    for s in students:
        s['features']['stability_cv'] = cv.get((s['anon_id'], s['semester']), {}).get('cv')
    ids = {s['anon_id']: s['anon_id'] for s in students}
    with contextlib.redirect_stdout(io.StringIO()):
        ev = M['m09'].build_event_log({'reading_log': r, 'quiz': q}, anon_map=ids)
        _g, lsa_type, ratios = M['m10'].compute_lsa_by_lsa_type(ev, set(), students=students)
    feat_df = pd.DataFrame(feat_rows)
    time_df = pd.DataFrame(time_rows)
    return feat_df, time_df, lsa_type, ratios, {'n_quiz_events': len(q), 'n_reading_events': len(r), 'n_pairs': len(pairs)}


def encode_s(lsa_type, aid, sem):
    """s_cluster as stored in step9 lsa_type_map (composite key anon_id||semester)."""
    return lsa_type.get(f'{aid}||{sem}')


# ----------------------------------------------------------------------------------------------------------------
# injecting rebuilt features into the merged step9 structure and running the frozen pipeline (A1b flow)
# ----------------------------------------------------------------------------------------------------------------
import json  # noqa: E402
import tempfile  # noqa: E402

DERIVED_KEYS = FEATS


def merged_frozen_step9(W=8):
    """Merged frozen step9 structure for week W (same layout load_week reads)."""
    parts = [json.load(open(os.path.join(C.IN, 'step9_D7', f'out_W{W}', f'{s}_step9_inputs.json'), encoding='utf-8')) for s in C.D7]
    m = {'week_limit': parts[0]['week_limit'], 'scatter_data': [], 'lsa_type_map': {}, 'transition_ratios': {},
         'incomplete_semesters': [], 'time_features': {}}
    for p in parts:
        m['scatter_data'] += p['scatter_data']
        for k in ['lsa_type_map', 'transition_ratios', 'time_features']:
            m[k].update(p[k])
        m['incomplete_semesters'] = sorted(set(m['incomplete_semesters']) | set(p['incomplete_semesters']))
    return m


def frozen_as_rebuilt(m):
    """Extract the replaceable pieces from the frozen structure, in the same shape rebuild() returns (for test R5)."""
    rows, trows = [], []
    for rec in m['scatter_data']:
        f = rec.get('features') or {}
        rows.append({'anon_id': rec['anon_id'], 'semester': str(rec['semester']), **{k: f.get(k) for k in FEATS}})
        tf = m['time_features'].get(f"{rec['anon_id']}||{rec['semester']}", {})
        trows.append({'anon_id': rec['anon_id'], 'semester': str(rec['semester']), **{k: tf.get(k) for k in TIME5}})
    return pd.DataFrame(rows), pd.DataFrame(trows), dict(m['lsa_type_map']), dict(m['transition_ratios'])


def inject(m, feat_df, time_df, lsa_type, ratios):
    """Return a copy of m with the event-derived pieces replaced for every analysis-set record present in feat_df."""
    import copy
    out = copy.deepcopy(m)
    fd = {(r.anon_id, r.semester): r for r in feat_df.itertuples(index=False)}
    td = {(r.anon_id, r.semester): r for r in time_df.itertuples(index=False)}
    for rec in out['scatter_data']:
        k = (rec['anon_id'], str(rec['semester']))
        if k not in fd:
            continue
        for key in FEATS:
            v = getattr(fd[k], key)
            if isinstance(v, (np.floating, float)) and np.isnan(v):
                v = None
            rec['features'][key] = v.item() if hasattr(v, 'item') else v
        ck = f'{k[0]}||{k[1]}'
        out['time_features'][ck] = {t: (None if (getattr(td[k], t) is None or (isinstance(getattr(td[k], t), float) and np.isnan(getattr(td[k], t)))) else float(getattr(td[k], t))) for t in TIME5}
        if ck in lsa_type:
            out['lsa_type_map'][ck] = lsa_type[ck]
        if ck in ratios:
            out['transition_ratios'][ck] = ratios[ck]
    return out


def run_pipeline_w8(m, tmp_root=None):
    """A1b flow on a (possibly injected) merged step9 structure. Returns (df OOF canonical with m1, m2, m3)."""
    sys.path.insert(0, os.path.join(C.PACK, '02_pipeline'))
    import ews_experiment as E
    tmp_root = tmp_root or tempfile.mkdtemp()
    d = os.path.join(tmp_root, 'out_W8')
    os.makedirs(d, exist_ok=True)
    json.dump(m, open(os.path.join(d, 'step9_inputs.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    etl = L.Etl(os.path.join(C.PACK, '02_pipeline', 'etl_v7_15_34'))
    L.init_bas(etl)
    data = E.load_all(etl, tmp_root, [8])
    fr = C.frozen_frame(8)
    fin = {(r.semester, r.anon_id): float(r.final) for r in fr.itertuples()}
    for s in C.D7:
        for r in data[8][s]:
            r['final'] = fin[(s, r['anon_id'])]
            r['is_fail'] = int(r['final'] < 60)
    df = E.run_folds(etl, data[8], E.fold_defs('P'), etl.FS_C, False, {'m1': 1, 'm2': 1})
    df = df.sort_values(['semester', 'anon_id']).reset_index(drop=True)
    df['m3'] = np.asarray(C.recompute_m3(df))
    return df
