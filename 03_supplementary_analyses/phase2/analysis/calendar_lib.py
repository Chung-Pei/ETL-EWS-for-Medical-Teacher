# -*- coding: utf-8 -*-
"""Calendar helpers for round 2 (H-01 plan definitions)."""
import json
import os

import pandas as pd

TRACK_BY_LABEL = (('學後護', 'postbac'), ('重修', 'retake'), ('二技', '2yr'))


def etl_week(ts, start):
    """ETL rule (lms_etl._truncate_by_week): week = floor((ts - start).days / 7) + 1."""
    return int((pd.Timestamp(ts) - pd.Timestamp(start)).days // 7 + 1)


def cutoff(start, W):
    """Exclusive end of Week W: start + 7*W days at 00:00."""
    return pd.Timestamp(start) + pd.Timedelta(days=7 * W)


def classify_exam_title(title):
    t = str(title)
    if '正課期中' in t:
        kind = 'midterm'
    elif '正課期末' in t:
        kind = 'final'
    else:
        return None, None
    return kind, ('makeup' if '補考' in t else 'main')


def load_calendar(p2):
    return json.load(open(os.path.join(p2, '05_calendar', 'exam_dates_all.json'), encoding='utf-8'))['semesters']


def effective_start(cal, sem, track):
    entry = cal[str(sem)]
    start = entry['semester_start']
    for _cid, ov in (entry.get('calendar_overrides') or {}).items():
        label = ov.get('_label', '')
        for key, tr in TRACK_BY_LABEL:
            if key in label:
                if tr == track:
                    start = ov['semester_start']
                break
    return pd.Timestamp(start)


def structure(cal, sem):
    return cal[str(sem)].get('course_structure') or 'regular'
