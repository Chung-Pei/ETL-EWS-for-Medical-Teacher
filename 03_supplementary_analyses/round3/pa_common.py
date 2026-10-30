# -*- coding: utf-8 -*-
"""Shared loaders for the round-3 supplementary rerun (post hoc). ASCII identifiers only.

Alignment rule (P0 addendum 1/2): step9 records are sorted by (semester, anon_id); this order equals the row order
of the frozen OOF file within each (arm, feature_set, W) block.
Outcome rule (rulings E): the outcome is the SEMESTER score = OOF `final` (step9 final_score is the final-EXAM score).
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

PACK = os.environ.get('PACK', '/home/claude/pack')
sys.path.insert(0, os.path.join(PACK, '02_pipeline'))
import ews_lib as L  # noqa: E402  (frozen pipeline library, unmodified)

D7 = L.D7
IN = os.path.join(PACK, '01_inputs')
CLASS5 = ['2yr_gen', '2yr_working', 'post', '4yr', 'retake']
CLASS_LABEL = {'2yr_gen': '2yr_gen', '2yr_working': '2yr_working', 'post': 'post', '4yr': '4yr', 'retake': 'retake'}
EXPECT_COUNTS = {'1111': 472, '1112': 221, '1121': 507, '1122': 234, '1131': 473, '1132': 236, '1141': 509}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_oof(arm='P', fs='C', W='W8'):
    oof = pd.read_csv(os.path.join(IN, 'oof_scores_no_ids.csv'), dtype={'semester': str})
    g = oof[(oof.arm == arm) & (oof.feature_set == fs) & (oof.W == W)].reset_index(drop=True)
    return g


def load_step9(W):
    """Canonical step9 table for week W (int): theory records with a final exam score, sorted by (semester, anon_id)."""
    rows = []
    for s in D7:
        path = os.path.join(IN, 'step9_D7', f'out_W{W}', f'{s}_step9_inputs.json')
        d = json.load(open(path, encoding='utf-8'))
        for r in d['scatter_data']:
            if r.get('edu_type') != 'theory' or r.get('final_score') is None:
                continue
            rows.append({'anon_id': r['anon_id'], 'semester': str(r['semester']), 'midterm': r.get('midterm_score'),
                         'final_exam': r['final_score'], 'is_4tech': r.get('is_4tech'),
                         'quz_pass_rate': (r.get('features') or {}).get('quz_pass_rate')})
    df = pd.DataFrame(rows).sort_values(['semester', 'anon_id']).reset_index(drop=True)
    return df


def load_class5():
    cm = pd.read_csv(os.path.join(IN, 'class_membership5.csv'), dtype={'anon_id': str, 'semester': str})
    return cm[['semester', 'anon_id', 'program5_key']]


def frozen_frame(W=8):
    """Frozen OOF (P arm, FS-C, week W) + anon_id (aligned) + class + step9 covariates. Row order = canonical."""
    o = load_oof('P', 'C', f'W{W}')
    s9 = load_step9(W)
    assert len(o) == len(s9), (len(o), len(s9))
    assert (o['semester'].to_numpy() == s9['semester'].to_numpy()).all(), 'semester order differs'
    df = o.copy()
    df['anon_id'] = s9['anon_id'].to_numpy()
    df['midterm'] = s9['midterm'].to_numpy(float)
    df['final_exam'] = s9['final_exam'].to_numpy(float)
    df['is_4tech'] = s9['is_4tech'].to_numpy()
    df['quz_pass_rate'] = pd.to_numeric(s9['quz_pass_rate'], errors='coerce').fillna(0.0).to_numpy()
    cm = load_class5()
    df = df.merge(cm, on=['semester', 'anon_id'], how='left')
    assert len(df) == len(o)
    # H-02 plan / P0 R3: M3' is re-derived as the pooled percentile blend. The file column differs from it only by
    # 1e-16 float noise, which changes the tie pattern (pooled AUC 0.727438 vs 0.727441); keep the file column aside.
    df['m3_file'] = df['m3']
    df['m3'] = np.asarray(recompute_m3(df))
    return df


def recompute_m3(df):
    """R3: pooled percentile blend of m1 and m2 (higher = safer)."""
    return (L.pct_rank(df['m1']) + L.pct_rank(df['m2'])) / 2

