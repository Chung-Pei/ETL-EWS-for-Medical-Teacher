# -*- coding: utf-8 -*-
"""Frozen-pipeline access for R5-5 / R5-9: the pack's own ews_lib.py / ews_experiment.py and ETL v7.15.34 module 11,
unmodified.  Only the input layout is adapted: the pack stores one step9 file per semester (inputs_D7/out_W{W}/{sem}_step9_inputs.json);
the pipeline reads one file per week, so the per-semester files are concatenated (records, maps and feature dicts are keyed by
anon_id||semester, so concatenation is lossless)."""
import json
import os
import sys
import types
from pathlib import Path

import r5_lib as L

PIPE = L.PACK / 'datapack_complete' / '04_pipeline_code'
ETL_DIR = L.PACK / 'datapack_complete' / '07_etl_v7_15_34'
MERGED = L.HERE.parent / 'derived' / 'inputs_merged'
W_ALL = [6, 8, 10, 12]


def xgboost_available():
    try:
        import xgboost  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


def import_pipeline(allow_stub_for_non_xgb_checks=False):
    """Import the pack's ews_lib / ews_experiment.  If xgboost is missing and allow_stub is set, a stub module that raises
    on use is injected so the NON-XGBoost parts (data load, ETL features, BAS) can be checked; any XGBoost fit then fails."""
    if not xgboost_available():
        if not allow_stub_for_non_xgb_checks:
            raise RuntimeError('xgboost is not installed; R5-5 / R5-9 cannot run')
        stub = types.ModuleType('xgboost')

        class XGBClassifier:                                  # noqa: D401
            def __init__(self, *a, **k):
                raise RuntimeError('xgboost stub: real xgboost is required for any model fit')
        stub.XGBClassifier = XGBClassifier
        sys.modules['xgboost'] = stub
    sys.path.insert(0, str(PIPE))
    import ews_lib as EL
    import ews_experiment as EE
    return EL, EE


def merge_step9(force=False):
    MERGED.mkdir(parents=True, exist_ok=True)
    for W in W_ALL:
        out = MERGED / f'out_W{W}' / 'step9_inputs.json'
        if out.exists() and not force:
            continue
        merged = {'week_limit': W, 'scatter_data': [], 'lsa_type_map': {}, 'transition_ratios': {},
                  'incomplete_semesters': [], 'time_features': {}}
        for s in L.D7:
            d = json.load(open(L.PACK / f'datapack_complete/01_step9_inputs/inputs_D7/out_W{W}/{s}_step9_inputs.json',
                               encoding='utf-8'))
            assert d['week_limit'] == W
            merged['scatter_data'] += d['scatter_data']
            for k in ('lsa_type_map', 'transition_ratios', 'time_features'):
                dup = set(merged[k]) & set(d[k])
                assert not dup, f'duplicate keys {k}'
                merged[k].update(d[k])
            merged['incomplete_semesters'] = sorted(set(merged['incomplete_semesters']) | set(d['incomplete_semesters']))
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(merged, open(out, 'w', encoding='utf-8'), ensure_ascii=False)
    return MERGED


def apply_semester_outcome(data, base):
    """The packaged step9 files carry the FINAL-EXAMINATION score as final_score (v1 target, 492 failing); the frozen Rounds 1-4
    scores target the SEMESTER score (keyed file `final`/`y`, 560 failing).  Replace the outcome fields of the records
    accordingly (features untouched) so folds are trained and evaluated on the frozen target."""
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
