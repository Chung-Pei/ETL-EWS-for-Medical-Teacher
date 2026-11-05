# -*- coding: utf-8 -*-
"""Round 7 shared setup.  Imports the audit hook first.  Identifiers ASCII-only.
Differences from r5_lib.load_base: class_membership5.csv is NOT read (its directory path contains '1142',
forbidden by the Round 7 spec); retake status comes from 03_class_membership/class_membership.csv only."""
import r7_audit  # noqa: F401  (must be first)
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(os.environ.get('R7_WORK', str(Path(__file__).resolve().parent.parent)))          # folder that contains code/
STAGING = Path(os.environ.get('R7_STAGING', str(WORK.parent / 'round7_pack_staging')))     # unzipped handoff pack
PACK = STAGING / '05_materials' / 'handoff_pack'
RES = WORK / 'results'
RES.mkdir(parents=True, exist_ok=True)
os.environ['R5_PACK'] = str(PACK)
os.environ['R5_RES'] = str(RES)
sys.path.insert(0, str(WORK / 'code'))
import r5_lib as L  # noqa: E402  (unmodified Round 5 library, copied)

D7 = L.D7
SEED_TIERA = 20261006
SEED_POSTHOC = 20261031
PIPE = PACK / 'datapack_complete' / '04_pipeline_code'
ETL_DIR = PACK / 'datapack_complete' / '07_etl_v7_15_34'
MERGED = WORK / 'derived' / 'inputs_merged'
W_ALL = [6, 8, 10, 12]


def load_base():
    """base: 2,652 analysis records (keyed file P/C); scores[W]: frozen m1,m2,m3,pfail,qmi_none aligned to base rows.
    Adds midterm/final/semester components (d7) and retake (class_membership.csv, class_track == 'retake')."""
    k = pd.read_csv(PACK / 'oof_scores_keyed.csv', dtype={'anon_id': str, 'semester': str})
    pc = k[(k.arm == 'P') & (k.feature_set == 'C')]
    scores, ref_keys, base = {}, None, None
    for W in ['W6', 'W8', 'W10', 'W12', 'W14', 'W16']:
        g = pc[pc.W == W].reset_index(drop=True)
        keys = list(zip(g.semester, g.anon_id))
        if ref_keys is None:
            ref_keys = keys
            base = g[['semester', 'anon_id', 'stratum', 'final', 'y']].copy()
        assert keys == ref_keys, f'record order differs at {W}'
        scores[W] = g[['m1', 'm2', 'm3', 'pfail', 'qmi_none']].copy()
    d7 = pd.read_csv(PACK / 'd7_semester_scores.csv', dtype={'anon_id': str, 'semester': str})
    base = base.merge(d7, on=['anon_id', 'semester'], how='left', validate='one_to_one')
    base['final'] = base['final'].astype(float)
    cm = pd.read_csv(PACK / 'datapack_complete/03_class_membership/class_membership.csv', dtype=str)
    base = base.merge(cm[['semester', 'anon_id', 'class_track']], on=['semester', 'anon_id'], how='left',
                      validate='one_to_one')
    base['retake'] = (base['class_track'] == 'retake').astype(int)
    base['stu'] = pd.factorize(base['anon_id'])[0]
    base['stu_sorted'] = pd.Series(base['anon_id']).rank(method='dense').astype(int) - 1   # np.unique order (ews_lib.Boot)
    base['sem_i'] = base['semester'].map({s: i for i, s in enumerate(D7)})
    return base, scores


def counts_sorted_students(anon_ids, B=2000, seed=SEED_TIERA):
    """Student-cluster draw multiplicities replicating ews_lib.Boot's RNG stream exactly
    (np.unique-sorted student order, rng.integers(0, n_stu, size=(B, n_stu)))."""
    uniq, inv = np.unique(np.asarray(anon_ids), return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(B, len(uniq)))
    cnt = np.zeros((B, len(uniq)), np.float32)
    for b in range(B):
        cnt[b] = np.bincount(draws[b], minlength=len(uniq))
    return cnt[:, inv]            # (B x N) record weights


def import_pipeline():
    sys.path.insert(0, str(PIPE))
    import ews_lib as EL
    import ews_experiment as EE
    return EL, EE


def merge_step9(force=False):
    """Concatenate the per-semester step9 files of the seven development semesters into one file per week
    (same as r5_pipeline.merge_step9).  Only the D7 per-semester files are read."""
    import json
    MERGED.mkdir(parents=True, exist_ok=True)
    for W in W_ALL:
        out = MERGED / f'out_W{W}' / 'step9_inputs.json'
        if out.exists() and not force:
            continue
        merged = {'week_limit': W, 'scatter_data': [], 'lsa_type_map': {}, 'transition_ratios': {},
                  'incomplete_semesters': [], 'time_features': {}}
        for s in D7:
            d = json.load(open(PACK / f'datapack_complete/01_step9_inputs/inputs_D7/out_W{W}/{s}_step9_inputs.json',
                               encoding='utf-8'))
            assert d['week_limit'] == W
            merged['scatter_data'] += d['scatter_data']
            for kk in ('lsa_type_map', 'transition_ratios', 'time_features'):
                dup = set(merged[kk]) & set(d[kk])
                assert not dup, f'duplicate keys {kk}'
                merged[kk].update(d[kk])
            merged['incomplete_semesters'] = sorted(set(merged['incomplete_semesters']) | set(d['incomplete_semesters']))
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(merged, open(out, 'w', encoding='utf-8'), ensure_ascii=False)
    return MERGED


def apply_semester_outcome(data, base):
    """Replace the step9 final-examination outcome by the frozen semester-score outcome (keyed file final/y);
    features untouched (as r5_pipeline.apply_semester_outcome)."""
    key = {(s, a): v for s, a, v in zip(base.semester, base.anon_id, base.final)}
    n = 0
    for W, by in data.items():
        for sem, recs in by.items():
            for r in recs:
                v = key.get((r['semester'], r['anon_id']))
                if v is None:
                    raise KeyError((r['semester'], r['anon_id']))
                r['final'] = float(v)
                r['is_fail'] = int(v < 60)
                n += 1
    return n
