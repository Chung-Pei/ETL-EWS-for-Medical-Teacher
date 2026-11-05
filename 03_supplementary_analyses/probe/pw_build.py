# -*- coding: utf-8 -*-
"""Phase-Week Probe -- Week-W snapshot builder from the anonymized event layer (identifiers ASCII-only).

Purpose: rebuild the step9 inputs at an arbitrary cumulation endpoint W from events_quiz.csv / events_reading.csv by calling the
frozen ETL's OWN functions (lms_etl.py v7.15.34 helpers + modules 03/04/05/09/10), in the order run() calls them:
    per-student exam dates (BEFORE truncation) -> week truncation of quiz / reading_log -> material outlier bounds on the
    truncated batch -> per-student features (m04 quiz features, m05 time profile, pre_exam_intensity formula of run())
    -> LSA event log (m09) -> S type + transition ratios (m10).
Inputs the event layer does not hold (activity / reading_stats sheets: completion rates, aud/vid/tut minutes, cluster) and the grade
fields are WEEK-INVARIANT in the frozen step9 files (checked, see ledger F2); they are inherited from the frozen record shell.
No model, no outcome, no bootstrap in this file.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('PW_ROOT', HERE.parent))
PACK = ROOT / 'materials' / 'handoff_pack'                       # subset of the Round 5 pack (see README_package.md)
ETL_DIR = Path(os.environ.get('PW_ETL', PACK / 'datapack_complete' / '07_etl_v7_15_34'))
FROZEN = Path(os.environ.get('PW_FROZEN', PACK / 'datapack_complete' / '01_step9_inputs' / 'inputs_D7'))
EVENTS = Path(os.environ.get('PW_EVENTS', ROOT / 'materials' / 'events'))
CALENDAR = Path(os.environ.get('PW_CALENDAR', ROOT / 'materials' / 'handoff' / 'exam_dates_all.json'))
DERIVED = Path(os.environ.get('PW_DERIVED', ROOT / 'derived'))
RES = Path(os.environ.get('PW_RES', ROOT / 'results'))

D7 = ['1111', '1112', '1121', '1122', '1131', '1132', '1141']
TRACK_LABEL = {'2yr': '二技', 'postbac': '學後護一', 'retake': '重修班'}
WEEK_FEATS = ['consistency_score', 'pre_exam_intensity', 'quz_cramming_ratio', 'quz_final_accuracy',
              'quz_first_attempt_accuracy', 'quz_pass_rate', 'quz_score_delta', 'quz_total_attempts',
              'reading_inflation_ratio', 'total_learning_minutes', 'policy_gaming_flag']
STATIC_FEATS = ['aud_completion_rate', 'aud_total_minutes', 'sup_completion_rate', 'tut_total_minutes',
                'txt_completion_rate', 'vid_completion_rate', 'vid_total_minutes']
TIME5 = ['active_weeks', 'weekly_minutes_std', 'cram_pattern_score', 'late_night_ratio', 'early_start_ratio']
RATIOS = ['MM_ratio', 'MQ_ratio', 'QM_ratio', 'QQ_ratio']


def load_etl():
    """Import the frozen lms_etl.py by path (it loads its own modules m01..m12)."""
    try:
        import xlrd  # noqa: F401
    except ModuleNotFoundError:
        # etl.py imports xlrd at module level only for the grade-book .xls reader; this builder never reads a workbook.
        # A stub that raises on ANY use keeps the import working without changing any code path that runs here (ledger R-3).
        import types
        stub = types.ModuleType('xlrd')

        def _no(*a, **k):
            raise RuntimeError('xlrd stub: workbook reading is not available in the probe build')
        stub.__getattr__ = lambda name: _no
        sys.modules['xlrd'] = stub
    spec = importlib.util.spec_from_file_location('lms_etl_frozen', str(ETL_DIR / 'lms_etl.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['lms_etl_frozen'] = mod
    spec.loader.exec_module(mod)
    return mod


def section_map(cal):
    """(semester, class_track) -> _section_id of the calendar override whose label names that track (exam_dates_all.json)."""
    out = {}
    for sem, cfg in cal.items():
        for sid, ov in (cfg.get('calendar_overrides') or {}).items():
            # exclusive match, most specific label first: the 1141 retake label also contains the word of the 2yr track
            for trk in ('retake', 'postbac', '2yr'):
                if TRACK_LABEL[trk] in ov.get('_label', ''):
                    out[(sem, trk)] = sid
                    break
    return out


def load_events(cal):
    """Event CSVs -> DataFrames with the ETL's own column names (c3_/c4_/... prefixes removed)."""
    secmap = section_map(cal)
    q = pd.read_csv(EVENTS / 'events_quiz.csv', dtype=str, keep_default_na=True)
    q = q.rename(columns={'semester': '_semester', 'c3_測驗標題': '測驗標題', 'c4_測驗時間': '測驗時間', 'c5_狀態': '狀態'})
    r = pd.read_csv(EVENTS / 'events_reading.csv', dtype=str, keep_default_na=True)
    r = r.rename(columns={'semester': '_semester', 'c3_教材標題': '教材標題', 'c4_開始': '開始', 'c5_結束': '結束',
                          'c6_時間': '時間'})
    for df in (q, r):
        # every row carries a section id in the original export; tracks without a calendar override only need a DISTINCT id,
        # because the ETL's override rule fires on the number of distinct section ids among a student's exam rows
        df['_section_id'] = [secmap.get((s, t), f'{s}:{t}') for s, t in zip(df['_semester'], df['class_track'])]
    return q, r


def pairs_of(q, r, extra):
    s = set(zip(q['anon_id'], q['_semester'])) | set(zip(r['anon_id'], r['_semester'])) | set(extra)
    return s


def exam_dates_for(E, q, cal, pairs):
    """run() Step 2b: per-student dates from the [重要] exam titles, then same-semester representative fallback."""
    ped = E.build_per_student_exam_dates(quiz_df=q, exam_dates_all=cal)
    cache = {}

    def fallback(sem):
        if sem not in cache:
            anons = sorted({a for a, s in pairs if s == sem})
            rep = next((ped[(a, sem)] for a in anons if (a, sem) in ped), None)
            cache[sem] = rep if rep else dict(cal.get(sem, {}))
        return dict(cache[sem])

    for a, s in pairs:
        if (a, s) not in ped:
            ped[(a, s)] = fallback(s)
    return ped


def frozen_shell(W_shell=8):
    """Record shells (static fields) of the frozen step9 files at W_shell; returns {sem: [rec,...]}."""
    shell = {}
    for s in D7:
        d = json.load(open(FROZEN / f'out_W{W_shell}' / f'{s}_step9_inputs.json', encoding='utf-8'))
        shell[s] = d['scatter_data']
    return shell


def build_week(E, W, q_all, r_all, cal, ped, shell, log=print, bounds_extra_tracks=('2yr',)):
    """Rebuild the step9 inputs at endpoint W for the 7 semesters.  Returns merged step9-style dict.

    Population rules (all fixed by G3 at W8 and W10, ledger "Roster search"):
      * per-record features, time features and the LSA event log use the frozen record roster only;
      * the material outlier bounds are computed PER SEMESTER (one ETL run per semester) on the semester's truncated
        reading_log of the roster plus the event-only ids whose class_track is in `bounds_extra_tracks`.
    """
    m03, m04, m05, m09, m10 = E.m03, E.m04, E.m05, E.m09, E.m10
    roster = {(x['anon_id'], x['semester']) for v in shell.values() for x in v}
    # --- truncation (ETL _truncate_by_week; scope_semester='' = every semester) ---
    q = E._truncate_by_week(q_all, ('anon_id', '帳號'), ('測驗時間',), ped, W, scope_semester='')
    r = E._truncate_by_week(r_all, ('anon_id', '帳號'), ('開始',), ped, W, scope_semester='')
    log(f'  W{W}: quiz rows {len(q_all)} -> {len(q)}; reading rows {len(r_all)} -> {len(r)}')
    in_r = np.fromiter(((a, s_) in roster for a, s_ in zip(r['anon_id'], r['_semester'])), bool, len(r))
    in_q = np.fromiter(((a, s_) in roster for a, s_ in zip(q['anon_id'], q['_semester'])), bool, len(q))
    r_pop = r[in_r | r['class_track'].isin(list(bounds_extra_tracks)).to_numpy()] if bounds_extra_tracks else r[in_r]
    bounds_by_sem = {s_: m05.compute_material_outlier_bounds(r_pop[r_pop['_semester'] == s_].reset_index(drop=True))
                     for s_ in D7}
    r = r[in_r].reset_index(drop=True)
    q = q[in_q].reset_index(drop=True)
    qg = E._group_by_anon_semester(q)
    rg = E._group_by_anon_semester(r)
    empty = E._EMPTY_DF
    scatter, tfeat = [], {}
    n_events_pairs = set(qg) | set(rg)
    for s in D7:
        for rec in shell[s]:
            a = rec['anon_id']
            d = ped[(a, s)]
            sq = qg.get((a, s), empty)
            sr = rg.get((a, s), empty)
            f = m04.compute_quiz_features(sq, a, d, None)
            tp = m05.compute_student_time_profile(
                anon_id=a, reading_log_df=sr, exam_dates=d, semester_start=d.get('semester_start', ''),
                total_weeks=int(d.get('total_weeks', 18)),
                material_bounds=bounds_by_sem[s], policy_hours=24.0)
            total_min = tp.get('capped_total_minutes') or 0
            pre_mid = tp.get('pre_midterm_7d_minutes') or 0.0
            pre_fin = tp.get('pre_final_7d_minutes') or 0.0
            tw = int(d.get('total_weeks', 18))
            pei = 0.0 if total_min == 0 else round((pre_mid + pre_fin) / ((total_min / tw) * 7), 4)
            feats = dict(rec['features'])                       # static fields inherited, week fields overwritten below
            feats.update({
                'consistency_score': tp.get('consistency_score'),
                'pre_exam_intensity': pei,
                'quz_cramming_ratio': f.get('quz_cramming_ratio'),
                'quz_final_accuracy': f.get('quz_final_accuracy'),
                'quz_first_attempt_accuracy': f.get('quz_first_attempt_accuracy'),
                'quz_pass_rate': f.get('quz_pass_rate'),
                'quz_score_delta': f.get('quz_score_delta'),
                'quz_total_attempts': f.get('quz_total_attempts'),
                'reading_inflation_ratio': tp.get('reading_inflation_ratio'),
                'total_learning_minutes': tp.get('capped_total_minutes'),
                'policy_gaming_flag': tp.get('policy_gaming_flag'),
            })
            new = {k: v for k, v in rec.items() if k not in ('features', 'cluster')}
            new['cluster'] = ''                                  # not rebuilt (ledger R-2): not a model input
            new['features'] = feats
            scatter.append(new)
            tfeat[f'{a}||{s}'] = {k: tp.get(k) for k in TIME5}
    # --- LSA (m09 -> m10) on the truncated logs ---
    anon_map = {a: '' for a, _ in roster}
    ev = m09.build_event_log({'reading_log': r, 'quiz': q}, anon_map)
    _res, ltype, ratios = m10.compute_lsa_by_lsa_type(ev, set(), None)
    keys = {f"{x['anon_id']}||{x['semester']}" for x in scatter}
    return {'week_limit': W, 'scatter_data': scatter,
            'lsa_type_map': {k: v for k, v in ltype.items() if k in keys},
            'transition_ratios': {k: v for k, v in ratios.items() if k in keys},
            'incomplete_semesters': [], 'time_features': tfeat,
            '_n_event_pairs': len(n_events_pairs)}


def frozen_week(W):
    out = {'scatter_data': [], 'lsa_type_map': {}, 'transition_ratios': {}, 'time_features': {}}
    for s in D7:
        d = json.load(open(FROZEN / f'out_W{W}' / f'{s}_step9_inputs.json', encoding='utf-8'))
        out['scatter_data'] += d['scatter_data']
        for k in ('lsa_type_map', 'transition_ratios', 'time_features'):
            out[k].update(d[k])
    return out


def _eq(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    return abs(float(a) - float(b)) <= 1e-9


def compare(new, W):
    """G3 comparison: rebuilt vs frozen, field by field.  Returns (rows, summary)."""
    fz = frozen_week(W)
    fr = {(x['anon_id'], x['semester']): x for x in fz['scatter_data']}
    nw = {(x['anon_id'], x['semester']): x for x in new['scatter_data']}
    assert set(fr) == set(nw), 'record sets differ'
    rows = []
    def add(field, vals):
        diffs = [d for d in vals if d is not None]
        mism = sum(1 for d in vals if d is None)
        nums = [d for d in diffs if isinstance(d, float)]
        rows.append({'week': W, 'field': field, 'n': len(vals), 'n_mismatch': mism + sum(1 for d in diffs if d == 'type'),
                     'max_abs_diff': (max(nums) if nums else 0.0)})
    for fld in WEEK_FEATS:
        vals = []
        for k, rec in fr.items():
            a, b = nw[k]['features'].get(fld), rec['features'].get(fld)
            if _eq(a, b):
                vals.append(0.0)
            elif a is None or b is None or isinstance(a, bool) or isinstance(b, bool):
                vals.append(None)
            else:
                vals.append(abs(float(a) - float(b)))
        mism = sum(1 for d in vals if d is None or d > 1e-9)
        nums = [d for d in vals if d is not None]
        rows.append({'week': W, 'field': fld, 'n': len(vals), 'n_mismatch': mism,
                     'max_abs_diff': float(max(nums)) if nums else 0.0})
    for fld in TIME5:
        vals = []
        for (a, s) in fr:
            x, y = new['time_features'].get(f'{a}||{s}', {}).get(fld), fz['time_features'].get(f'{a}||{s}', {}).get(fld)
            vals.append(0.0 if _eq(x, y) else (abs(float(x) - float(y)) if x is not None and y is not None else None))
        nums = [d for d in vals if d is not None]
        rows.append({'week': W, 'field': fld, 'n': len(vals), 'n_mismatch': sum(1 for d in vals if d is None or d > 1e-9),
                     'max_abs_diff': float(max(nums)) if nums else 0.0})
    vals = [0.0 if new['lsa_type_map'].get(f'{a}||{s}') == fz['lsa_type_map'].get(f'{a}||{s}') else 1.0 for (a, s) in fr]
    rows.append({'week': W, 'field': 's_cluster(S type)', 'n': len(vals), 'n_mismatch': int(sum(vals)), 'max_abs_diff': max(vals)})
    for fld in RATIOS:
        vals = []
        for (a, s) in fr:
            x = new['transition_ratios'].get(f'{a}||{s}', {}).get(fld)
            y = fz['transition_ratios'].get(f'{a}||{s}', {}).get(fld)
            vals.append(0.0 if _eq(x, y) else (abs(float(x) - float(y)) if x is not None and y is not None else None))
        nums = [d for d in vals if d is not None]
        rows.append({'week': W, 'field': fld, 'n': len(vals), 'n_mismatch': sum(1 for d in vals if d is None or d > 1e-9),
                     'max_abs_diff': float(max(nums)) if nums else 0.0})
    return rows


def write_merged(new, W, tag):
    out = DERIVED / tag / f'out_W{W}'
    out.mkdir(parents=True, exist_ok=True)
    d = {k: v for k, v in new.items() if not k.startswith('_')}
    json.dump(d, open(out / 'step9_inputs.json', 'w', encoding='utf-8'), ensure_ascii=False)
    return out
