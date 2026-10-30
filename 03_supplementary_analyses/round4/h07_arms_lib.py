# -*- coding: utf-8 -*-
"""Arm definitions for H-07 (spec: H07_addendum3_arms.md). Wraps h07_rebuild.rebuild without modifying it."""
import contextlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import h07_rebuild as R  # noqa: E402

C = R.C
L = R.L
_BASE_TRUNCATE = R.truncate


def midterm_weeks():
    cal = pd.read_csv(R.CAL, encoding='utf-8-sig', dtype={'semester': str})
    return {(r.semester, r.class_track): int(r.midterm_first_week) for r in cal.itertuples()}


MW = midterm_weeks()
ARMS = ['ORIG_W6', 'ORIG_W7', 'ORIG_W8', 'STRICT', 'EXCL_W6', 'EXCL_W7', 'EXCL_W8']
SENS = ['STRICT_R', 'EXCL_W8_R']
EXCL_WEEKS = {'EXCL_W6': 6, 'EXCL_W7': 7, 'EXCL_W8': 8, 'EXCL_W8_R': 8}
# Round 4 (H07_addendum4_placebo_plan.md): placebo arms remove FIXED weeks (inclusive), independent of the exam calendar.
PLCB_WEEKS = {'PLCB_W56': (5, 6), 'PLCB_W34': (3, 4)}
PLCB_ARMS = ['PLCB_W56', 'PLCB_W34']
PLCB_SENS = ['PLCB_W56_R', 'PLCB_W34_R']


def arm_spec(name):
    """Return (cut_days, excl) for an arm. cut_days: (sem, track) -> days (events strictly earlier than start + days).
    excl: (sem, track) -> (lo_week, hi_week) inclusive removal. Main arms keep retake at the standard window (S3 then
    overrides its features with ORIG); *_R arms define the retake window from its own calendar."""
    own = name.endswith('_R')
    base = name[:-2] if own else name
    kind = ('ORIG' if base.startswith('ORIG') else 'STRICT' if base.startswith('STRICT')
            else 'PLCB' if base.startswith('PLCB') else 'EXCL')
    k = {'ORIG_W6': 6, 'ORIG_W7': 7, 'ORIG_W8': 8, 'EXCL_W6': 6, 'EXCL_W7': 7, 'EXCL_W8': 8, 'STRICT': 8,
         'PLCB_W56': 8, 'PLCB_W34': 8}[base]
    cut, excl = {}, {}
    for key, mw in MW.items():
        is_retake = key[1] == 'retake'
        std = 7 * k
        if kind == 'ORIG' or (is_retake and not own):
            cut[key] = std
        elif kind == 'STRICT':
            cut[key] = 7 * (mw - 1)
        elif kind == 'PLCB':
            cut[key] = std
            excl[key] = PLCB_WEEKS[base]
        else:
            cut[key] = std
            excl[key] = (mw - 1, mw)
    return cut, excl


def _make_truncate(excl):
    def wrapped(df, time_col, cal, week_limit=8, cut_days=None):
        out = _BASE_TRUNCATE(df, time_col, cal, week_limit, cut_days)
        if not excl:
            return out
        ts = pd.to_datetime(out[time_col], errors='coerce')
        key = list(zip(out['_semester'], out['class_track']))
        start = pd.to_datetime(pd.Series([cal[k]['semester_start'] for k in key], index=out.index))
        week = ((ts - start).dt.days // 7) + 1
        lo = pd.Series([excl.get(k, (99, -1))[0] for k in key], index=out.index)
        hi = pd.Series([excl.get(k, (99, -1))[1] for k in key], index=out.index)
        drop = ts.notna() & (week >= lo) & (week <= hi)
        return out[~drop].reset_index(drop=True)
    return wrapped


def rebuild_arm(name):
    cut, excl = arm_spec(name)
    R.truncate = _make_truncate(excl)
    try:
        return R.rebuild(cut_days=cut)
    finally:
        R.truncate = _BASE_TRUNCATE


def fix_retake(rebuilt, orig):
    """S3: replace the 1141 retake records' features in `rebuilt` by those of `orig` (same structure as R.rebuild output)."""
    feat, time_, lsa, ratios, info = rebuilt
    ofeat, otime, olsa, oratios, _ = orig
    rk = set(zip(*[C.load_class5().query("program5_key == 'retake'")[c] for c in ['anon_id', 'semester']]))
    def swap(df, odf):
        df = df.copy()
        okey = odf.set_index(['anon_id', 'semester'])
        for i, (a, s) in enumerate(zip(df.anon_id, df.semester)):
            if (a, s) in rk:
                for c in df.columns:
                    if c not in ('anon_id', 'semester'):
                        df.iat[i, df.columns.get_loc(c)] = okey.loc[(a, s), c]
        return df
    lsa = dict(lsa)
    ratios = dict(ratios)
    for a, s in rk:
        ck = f'{a}||{s}'
        if ck in olsa:
            lsa[ck] = olsa[ck]
        if ck in oratios:
            ratios[ck] = oratios[ck]
    return swap(feat, ofeat), swap(time_, otime), lsa, ratios, info
