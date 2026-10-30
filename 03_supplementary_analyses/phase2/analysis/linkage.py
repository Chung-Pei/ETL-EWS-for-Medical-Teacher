# -*- coding: utf-8 -*-
"""Data linkage for round 2: merge per-semester step9 files, attach student IDs to the frozen OOF, class5, 1142 outcomes."""
import json
import os

import numpy as np
import pandas as pd

D7 = ['1111', '1112', '1121', '1122', '1131', '1132', '1141']
FEATURES = ['aud_completion_rate', 'aud_total_minutes', 'consistency_score', 'policy_gaming_flag', 'pre_exam_intensity',
            'quz_cramming_ratio', 'quz_final_accuracy', 'quz_first_attempt_accuracy', 'quz_pass_rate', 'quz_score_delta',
            'quz_total_attempts', 'reading_inflation_ratio', 'sup_completion_rate', 'total_learning_minutes',
            'tut_total_minutes', 'txt_completion_rate', 'vid_completion_rate', 'vid_total_minutes']


def step9_path(p2, W, sem):
    return os.path.join(p2, '01_step9_inputs', 'inputs_D7', f'out_W{W}', f'{sem}_step9_inputs.json')


def load_step9(p2, W):
    """Merge the seven per-semester step9 files of one week into one dict (same layout as out_W{W}/step9_inputs.json)."""
    merged = {'week_limit': W, 'scatter_data': [], 'lsa_type_map': {}, 'transition_ratios': {},
              'incomplete_semesters': [], 'time_features': {}}
    for s in D7:
        d = json.load(open(step9_path(p2, W, s), encoding='utf-8'))
        assert d['week_limit'] == W, (W, s, d['week_limit'])
        merged['scatter_data'] += d['scatter_data']
        merged['lsa_type_map'].update(d['lsa_type_map'])
        merged['transition_ratios'].update(d['transition_ratios'])
        merged['time_features'].update(d['time_features'])
        merged['incomplete_semesters'] += list(d['incomplete_semesters'])
    return merged


def load_holdout_step9(p2):
    return json.load(open(os.path.join(p2, '01_step9_inputs', 'holdout_1142_W8', 'step9_inputs.json'), encoding='utf-8'))


def records_df(step9, require_final=True):
    rows = []
    for r in step9['scatter_data']:
        if require_final and r.get('final_score') is None:  # final_score is the final EXAM score
            continue
        f = r.get('features') or {}
        row = {'anon_id': r['anon_id'], 'semester': str(r['semester']), 'edu_type': r.get('edu_type'),
               'is_4tech': r.get('is_4tech'), 'midterm': r.get('midterm_score'), 'final_exam': r.get('final_score')}
        for k in FEATURES:
            row[k] = f.get(k)
        rows.append(row)
    df = pd.DataFrame(rows)
    df['_o'] = df['semester'].map({s: i for i, s in enumerate(D7 + ['1142'])})
    return df.sort_values(['_o', 'anon_id']).drop(columns='_o').reset_index(drop=True)


def check_alignment(oof_w, rec, c5):
    """P0 addendum 1 criteria 1, 2, 4 and addendum 2 criterion 3'. Raises ValueError on failure; returns diagnostics."""
    sem_o, sem_r = oof_w['semester'].to_numpy(int), rec['semester'].to_numpy(int)
    if not np.array_equal(sem_o, sem_r):
        raise ValueError('semester vectors differ')
    comp = 0.5 * rec['midterm'].to_numpy(float) + 0.5 * rec['final_exam'].to_numpy(float)
    fin = oof_w['final'].to_numpy(float)
    prog = rec[['semester', 'anon_id']].merge(c5, on=['semester', 'anon_id'], how='left')['program5_key'].to_numpy()
    diag = {}
    for s in sorted(set(sem_o)):
        ix = sem_o == s
        if s in (1111, 1112, 1122, 1132):
            if np.abs(fin[ix] - comp[ix]).max() > 1e-9:
                raise ValueError(f'semester {s}: OOF final != 0.5*midterm + 0.5*final exam')
            diag[s] = 'exact'
        else:
            r = float(np.corrcoef(fin[ix], comp[ix])[0, 1])
            plain = ix & np.isin(prog, ['2yr_gen', 'post'])
            share = float((np.abs(fin[plain] - comp[plain]) <= 1e-9).mean())
            if r < 0.90 or share < 0.95:
                raise ValueError(f'semester {s}: corr {r:.3f}, exact share in 2yr_gen/post {share:.3f}')
            diag[s] = {'corr': r, 'exact_share_2yr_gen_post': share}
    if int((fin < 60).sum()) != int(oof_w['y'].sum()):
        raise ValueError('y != (final < 60)')
    return diag


def load_oof_with_ids(oof_path, p2):
    """Frozen OOF (P arm, feature set C) with anon_id attached by (semester, anon_id) sort order, after the checks."""
    import h_lib as H
    oof = H.load_oof(oof_path)
    rec = records_df(load_step9(p2, 8))
    c5 = load_class5(p2)
    out = {}
    for W, df in oof.items():
        check_alignment(df, rec, c5)
        df = df.copy()
        df.insert(1, 'anon_id', rec['anon_id'].to_numpy())
        out[W] = df
    return out


def load_class5(p2):
    c = pd.read_csv(os.path.join(p2, '08_class_membership_and_1142_outcomes', 'class_membership5.csv'), dtype=str,
                    encoding='utf-8-sig')
    c = c[['semester', 'anon_id', 'program_key', 'program5_key', 'program5_label', 'retake_enrollment']]
    return c.drop_duplicates(['semester', 'anon_id'])


def load_outcomes_1142(p2):
    o = pd.read_csv(os.path.join(p2, '08_class_membership_and_1142_outcomes', 'outcomes_1142.csv'), dtype={'anon_id': str},
                    encoding='utf-8-sig')
    return o
