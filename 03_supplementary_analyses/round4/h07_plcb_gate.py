# -*- coding: utf-8 -*-
"""Round-4 gate G1 (post hoc; H07_addendum4_placebo_plan.md section 3).
Checks the freshly re-run ORIG_W8 arm (results/arms/ORIG_W8_rerun_oof.csv):
  (a) M3' pooled AUC reproduces 0.7291 at n = 2,600 (no retake) and 0.7272 at n = 2,652 (4 decimals);
  (b) the re-run scores agree with the round-3 ORIG_W8 arm OOF file (AUC difference <= 1e-9 for M1', M2', M3').
Exit code 0 = pass, 2 = stop and report. No parameter, seed or specification is adjusted to pass.
Usage: python h07_plcb_gate.py --out results"""
import argparse
import json
import os
import platform
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import h07_arms_lib as A  # noqa: E402

C = A.C
L = A.L
R3_ARMS = os.path.join(C.PACK, '07_round3_return_C_H07', 'results', 'arms')
ANCHOR_2600 = 0.7291
ANCHOR_2652 = 0.7272
FILE_TOL = 1e-9


def anchor_ok(value, expected, dp=4):
    """True when `value` rounds to `expected` at `dp` decimals."""
    return round(float(value), dp) == round(float(expected), dp)


def load(path):
    d = pd.read_csv(path, dtype={'semester': str})
    d['m3'] = np.asarray(C.recompute_m3(d))
    return d


def aucs(d, base, mask):
    y = base.y.to_numpy(int)[mask]
    return {k: L.auc_fail(y, d[c].to_numpy(float)[mask]) for k, c in [("M1'", 'm1'), ("M2'", 'm2'), ("M3'", 'm3')]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    a = ap.parse_args()
    base = C.frozen_frame(8)
    retake = (base.program5_key == 'retake').to_numpy()
    mask_main, mask_all = ~retake, np.ones(len(base), bool)
    new = load(os.path.join(a.out, 'arms', 'ORIG_W8_rerun_oof.csv'))
    old = load(os.path.join(R3_ARMS, 'ORIG_W8_oof.csv'))
    for d in (new, old):
        assert (d.semester.to_numpy() == base.semester.to_numpy()).all() and np.array_equal(d.final.to_numpy(), base.final.to_numpy())
    rn_main, rn_all = aucs(new, base, mask_main), aucs(new, base, mask_all)
    ro_main, ro_all = aucs(old, base, mask_main), aucs(old, base, mask_all)
    max_abs_score_diff = {c: float(np.max(np.abs(new[c].to_numpy(float) - old[c].to_numpy(float)))) for c in ['m1', 'm2']}
    res = {
        'env': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
        'anchor_m3_2600': {'expected': ANCHOR_2600, 'rerun': rn_main["M3'"], 'round3_file': ro_main["M3'"],
                           'pass': anchor_ok(rn_main["M3'"], ANCHOR_2600)},
        'anchor_m3_2652': {'expected': ANCHOR_2652, 'rerun': rn_all["M3'"], 'round3_file': ro_all["M3'"],
                           'pass': anchor_ok(rn_all["M3'"], ANCHOR_2652)},
        'rerun_vs_round3_file_auc_diff': {f'{m}@{n}': abs(r1[m] - r0[m]) for n, r1, r0 in
                                          [(2600, rn_main, ro_main), (2652, rn_all, ro_all)] for m in r1},
        'rerun_vs_round3_file_max_abs_score_diff': max_abs_score_diff,
        'tolerance_file_auc': FILE_TOL,
    }
    res['file_match'] = all(v <= FILE_TOL for v in res['rerun_vs_round3_file_auc_diff'].values())
    res['pass'] = bool(res['anchor_m3_2600']['pass'] and res['anchor_m3_2652']['pass'] and res['file_match'])
    json.dump(res, open(os.path.join(a.out, 'H07R4_gate_G1.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    with open(os.path.join(a.out, 'run_arms.log'), 'a', encoding='utf-8') as lg:
        lg.write('GATE G1 ' + ('PASS' if res['pass'] else 'FAIL') + ' ' + json.dumps(
            {'m3_2600': rn_main["M3'"], 'm3_2652': rn_all["M3'"], 'file_match': res['file_match']}) + '\n')
    sys.exit(0 if res['pass'] else 2)


if __name__ == '__main__':
    main()
