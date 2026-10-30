# -*- coding: utf-8 -*-
"""Round-4 H-07 placebo analysis (post hoc). Spec: H07_addendum4_placebo_plan.md.
ORIG_W8, EXCL_W8 and EXCL_W8_R scores are read from the round-3 arm OOF files and are never re-run or altered.
Helper functions (pct, boot_vectors, point, boot_recall, recall20) are imported unchanged from the round-3 analysis module.
Reports numbers and CIs only; the 'mechanical_match' column is a lookup against the rule the author pre-declared in 00_HANDOFF_round4.md.
Usage: python h07_plcb_analyze.py --out results"""
import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import h07_arms_lib as A  # noqa: E402
from h07_arms_analyze import W8_STRATUM, boot_recall, boot_vectors, load_arm, pct, point, recall20  # noqa: E402

C = A.C
L = A.L
R3_DIR = os.path.join(C.PACK, '07_round3_return_C_H07', 'results')
MODELS = ["M1'", "M2'", "M3'"]
COL = {"M1'": 'm1', "M2'": 'm2', "M3'": 'm3'}
CONTRAST_MODELS = ["M3'", "M2'"]          # M3' primary, M2' secondary (handoff)
R3_ARMS = ['ORIG_W8', 'EXCL_W8']
R3_SENS = ['EXCL_W8_R']
R4_ARMS = ['PLCB_W56', 'PLCB_W34']
R4_SENS = ['PLCB_W56_R', 'PLCB_W34_R']
STRATUM_LABEL = {'w8': 'midterm_in_W8 (1121,1122,1131)', 'w9': 'midterm_in_W9 (1111,1112,1132,1141)'}


# ------------------------------------------------------------------------------------------------ pure helpers (tested)
def paired_delta(pt_a, pt_b, dist_a, dist_b):
    """delta = a - b; percentile CI of the paired bootstrap difference (same resamples)."""
    d = float(pt_a - pt_b)
    dist = np.asarray(dist_a, float) - np.asarray(dist_b, float)
    lo, hi = pct(dist, 2.5), pct(dist, 97.5)
    return {'delta': d, 'ci_lo': lo, 'ci_hi': hi, 'ci_includes_0': bool(lo <= 0 <= hi)}


def rule_lookup(delta, lo, hi):
    """Mechanical lookup of the author's pre-declared rule on contrast 3 (delta AUC, EXCL_W8 - PLCB_W56)."""
    if lo <= 0 <= hi:
        return 'rule1: CI includes 0'
    if delta <= -0.02:
        return 'rule2: delta <= -0.02 and CI excludes 0'
    return 'neither (edge)'


def assert_no_ids(df):
    bad = [c for c in df.columns if 'anon' in str(c).lower()]
    assert not bad, f'result table must not contain identifier columns: {bad}'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def load_dir_arm(dirpath, name):
    return load_arm(dirpath, name)           # reads <dirpath>/arms/<name>_oof.csv and recomputes the pooled M3'


def parse_r3_log(path):
    out = {}
    for ln in open(path, encoding='utf-8'):
        m = re.match(r'^(\w+): events kept quiz=(\d+) reading=(\d+)', ln)
        if m:
            out[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return out


def save(df, out, name):
    assert_no_ids(df)
    df.to_csv(os.path.join(out, name), index=False, encoding='utf-8-sig')


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    args = ap.parse_args()
    out = args.out
    lg = open(os.path.join(out, 'run_analyze.log'), 'w', encoding='utf-8')

    def log(*x):
        msg = ' '.join(str(v) for v in x)
        print(msg, flush=True)
        lg.write(msg + '\n')
        lg.flush()

    import scipy
    import sklearn
    import xgboost
    log('# H-07 round-4 placebo analysis (post hoc) clock=%s' % time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    log('python', platform.python_version(), 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__, 'numpy', np.__version__,
        'pandas', pd.__version__, 'scipy', scipy.__version__, '| boot B=%d seed=%d' % (L.BOOT_B, L.BOOT_SEED))
    for f in ['H07_addendum4_placebo_plan.md', 'h07_arms_lib.py', 'h07_arms_run.py', 'h07_plcb_gate.py', 'h07_plcb_analyze.py']:
        log('sha256', f, sha256(os.path.join(HERE, f)))
    r3_inputs = [os.path.join(R3_DIR, 'arms', f) for f in
                 ['ORIG_W8_oof.csv', 'EXCL_W8_oof.csv', 'EXCL_W8_R_oof.csv', 'ORIG_W8_abl_quiz.csv', 'EXCL_W8_abl_quiz.csv']]
    h_before = {p: sha256(p) for p in r3_inputs}
    for p, h in h_before.items():
        log('round3 input sha256 (before)', os.path.basename(p), h)

    base = C.frozen_frame(8)
    n = len(base)
    retake = (base.program5_key == 'retake').to_numpy()
    main_mask, all_mask = ~retake, np.ones(n, bool)
    boot = L.Boot(base.anon_id.to_numpy())
    arms = {nm: load_dir_arm(R3_DIR, nm) for nm in R3_ARMS + R3_SENS}
    arms.update({nm: load_dir_arm(out, nm) for nm in R4_ARMS + R4_SENS})
    for nm, d in arms.items():
        assert len(d) == n
        assert (d.semester.to_numpy() == base.semester.to_numpy()).all() and np.array_equal(d.final.to_numpy(), base.final.to_numpy()), nm
        assert np.array_equal(d.y.to_numpy(int), base.y.to_numpy(int)), nm
    vec = {(nm, mo): arms[nm][COL[mo]].to_numpy(float) for nm in arms for mo in MODELS}
    t0 = time.time()

    # ---------------- main set (n = 2,600) --------------------------------------------------------------------------------
    main_arms = R3_ARMS + R4_ARMS
    V = {(nm, mo): vec[(nm, mo)] for nm in main_arms for mo in MODELS}
    BD = boot_vectors(boot, base, main_mask, V, strata=True)
    PT = {k: point(base, main_mask, v) for k, v in V.items()}
    rec_pt, rec_prec, rec_b = {}, {}, {}
    for nm in main_arms:
        for mo in CONTRAST_MODELS:
            fr = base[main_mask].assign(s=vec[(nm, mo)][main_mask])
            rec_pt[(nm, mo)], rec_prec[(nm, mo)] = recall20(fr, 's')
            rec_b[(nm, mo)] = boot_recall(boot, base, main_mask, vec[(nm, mo)])
    log(f'main bootstrap done {time.time() - t0:.0f}s')
    nfail = int(base.y[main_mask].sum())

    M = []
    for nm in main_arms:
        for mo in MODELS:
            k = (nm, mo)
            row = {'arm': nm, 'eval_set': 'main_no_retake', 'n': int(main_mask.sum()), 'n_fail': nfail, 'model': mo,
                   'auc': PT[k]['auc'], 'auc_lo': pct(BD[(k, 'auc')], 2.5), 'auc_hi': pct(BD[(k, 'auc')], 97.5),
                   'r': PT[k]['r'], 'r_lo': pct(BD[(k, 'r')], 2.5), 'r_hi': pct(BD[(k, 'r')], 97.5),
                   'source': 'round3_arm_oof_file' if nm in R3_ARMS else 'round4_run'}
            if mo in CONTRAST_MODELS:
                row['recall20'] = rec_pt[k]
                row['precision20'] = rec_prec[k]
                row['recall20_lo'], row['recall20_hi'] = pct(rec_b[k], 2.5), pct(rec_b[k], 97.5)
            M.append(row)
    save(pd.DataFrame(M), out, 'H07R4_metrics.csv')

    # ---------------- contrasts -----------------------------------------------------------------------------------------------
    CONTRASTS = [('C1', 'PLCB_W56 - ORIG_W8', 'PLCB_W56', 'ORIG_W8'),
                 ('C2', 'PLCB_W34 - ORIG_W8', 'PLCB_W34', 'ORIG_W8'),
                 ('C3', 'EXCL_W8 - PLCB_W56 (key contrast)', 'EXCL_W8', 'PLCB_W56')]
    CT = []

    def add_contrast(cid, label, a, b, mo, eval_set, n_eval, pt, bd, rp, rbd):
        ka, kb = (a, mo), (b, mo)
        d = paired_delta(pt[ka]['auc'], pt[kb]['auc'], bd[(ka, 'auc')], bd[(kb, 'auc')])
        CT.append({'contrast_id': cid, 'label': label, 'arm_a': a, 'arm_b': b, 'eval_set': eval_set, 'n': n_eval, 'model': mo,
                   'metric': 'auc', 'value_a': pt[ka]['auc'], 'value_b': pt[kb]['auc'], 'delta_a_minus_b': d['delta'],
                   'ci_lo': d['ci_lo'], 'ci_hi': d['ci_hi'], 'ci_includes_0': d['ci_includes_0']})
        d = paired_delta((rp[ka]) * 100, (rp[kb]) * 100, rbd[ka] * 100, rbd[kb] * 100)
        CT.append({'contrast_id': cid, 'label': label, 'arm_a': a, 'arm_b': b, 'eval_set': eval_set, 'n': n_eval, 'model': mo,
                   'metric': 'recall20_pp', 'value_a': rp[ka] * 100, 'value_b': rp[kb] * 100, 'delta_a_minus_b': d['delta'],
                   'ci_lo': d['ci_lo'], 'ci_hi': d['ci_hi'], 'ci_includes_0': d['ci_includes_0']})

    for cid, label, a, b in CONTRASTS:
        for mo in CONTRAST_MODELS:
            add_contrast(cid, label, a, b, mo, 'main_no_retake', int(main_mask.sum()), PT, BD, rec_pt, rec_b)

    # ---------------- stratified (C1 and C3 only) -----------------------------------------------------------------------------
    sem = base.semester.to_numpy()
    in8 = np.isin(sem, W8_STRATUM)
    ST = []
    for cid, label, a, b in [CONTRASTS[0], CONTRASTS[2]]:
        for mo in CONTRAST_MODELS:
            ka, kb = (a, mo), (b, mo)
            deltas = {}
            for st in ['w8', 'w9']:
                d = paired_delta(PT[ka]['auc_' + st], PT[kb]['auc_' + st], BD[(ka, 'auc_' + st)], BD[(kb, 'auc_' + st)])
                deltas[st] = d
                msk = main_mask & (in8 if st == 'w8' else ~in8)
                ST.append({'contrast_id': cid, 'label': label, 'model': mo, 'stratum': STRATUM_LABEL[st],
                           'n': int(msk.sum()), 'n_fail': int(base.y[msk].sum()),
                           'value_a': PT[ka]['auc_' + st], 'value_b': PT[kb]['auc_' + st], 'delta_a_minus_b': d['delta'],
                           'ci_lo': d['ci_lo'], 'ci_hi': d['ci_hi'], 'ci_includes_0': d['ci_includes_0']})
            dist_a = BD[(ka, 'auc_w8')] - BD[(ka, 'auc_w9')]
            dist_b = BD[(kb, 'auc_w8')] - BD[(kb, 'auc_w9')]
            pt_a = PT[ka]['auc_w8'] - PT[ka]['auc_w9']
            pt_b = PT[kb]['auc_w8'] - PT[kb]['auc_w9']
            did = paired_delta(pt_a, pt_b, dist_a, dist_b)
            assert abs(did['delta'] - (deltas['w8']['delta'] - deltas['w9']['delta'])) < 1e-12
            ST.append({'contrast_id': cid, 'label': label, 'model': mo, 'stratum': 'DiD = delta(W8-midterm stratum) - delta(W9-midterm stratum)',
                       'n': int(main_mask.sum()), 'delta_a_minus_b': did['delta'], 'ci_lo': did['ci_lo'], 'ci_hi': did['ci_hi'],
                       'ci_includes_0': did['ci_includes_0']})
    save(pd.DataFrame(ST), out, 'H07R4_stratified.csv')

    # ---------------- sensitivity (n = 2,652; retake own window) --------------------------------------------------------------
    sens_arms = ['ORIG_W8'] + R3_SENS + R4_SENS
    VS = {(nm, mo): vec[(nm, mo)] for nm in sens_arms for mo in MODELS}
    BS = boot_vectors(boot, base, all_mask, VS, strata=None)
    PS = {k: point(base, all_mask, v, with_strata=False) for k, v in VS.items()}
    rec_s, rec_s_b = {}, {}
    for nm in sens_arms:
        for mo in CONTRAST_MODELS:
            rec_s[(nm, mo)] = recall20(base.assign(s=VS[(nm, mo)]), 's')[0]
            rec_s_b[(nm, mo)] = boot_recall(boot, base, all_mask, VS[(nm, mo)])
    log(f'sensitivity bootstrap done {time.time() - t0:.0f}s')
    SE = []
    for nm in sens_arms:
        for mo in MODELS:
            k = (nm, mo)
            row = {'arm': nm, 'eval_set': 'all_2652_incl_retake', 'model': mo, 'auc': PS[k]['auc'],
                   'auc_lo': pct(BS[(k, 'auc')], 2.5), 'auc_hi': pct(BS[(k, 'auc')], 97.5), 'r': PS[k]['r'],
                   'source': 'round3_arm_oof_file' if nm in ['ORIG_W8'] + R3_SENS else 'round4_run'}
            if mo in CONTRAST_MODELS:
                row['recall20'] = rec_s[k]
                row['recall20_lo'], row['recall20_hi'] = pct(rec_s_b[k], 2.5), pct(rec_s_b[k], 97.5)
            if nm != 'ORIG_W8' and mo in CONTRAST_MODELS:
                k0 = ('ORIG_W8', mo)
                d = paired_delta(PS[k]['auc'], PS[k0]['auc'], BS[(k, 'auc')], BS[(k0, 'auc')])
                row.update({'delta_auc_vs_ORIG_W8': d['delta'], 'delta_lo': d['ci_lo'], 'delta_hi': d['ci_hi']})
                d = paired_delta(rec_s[k] * 100, rec_s[k0] * 100, rec_s_b[k] * 100, rec_s_b[k0] * 100)
                row.update({'recall20_delta_pp_vs_ORIG_W8': d['delta'], 'recall20_delta_lo': d['ci_lo'], 'recall20_delta_hi': d['ci_hi']})
            SE.append(row)
    yk = base.y.to_numpy(int)[retake]
    for nm in sens_arms:
        for mo in MODELS:
            SE.append({'arm': nm, 'eval_set': 'retake_only (descriptive, n=%d, fails=%d)' % (int(retake.sum()), int(yk.sum())),
                       'model': mo, 'auc': L.auc_fail(yk, VS[(nm, mo)][retake])})
    save(pd.DataFrame(SE), out, 'H07R4_sensitivity_retake.csv')
    add = [('C1_R', 'PLCB_W56_R - ORIG_W8', 'PLCB_W56_R', 'ORIG_W8'), ('C3_R', 'EXCL_W8_R - PLCB_W56_R', 'EXCL_W8_R', 'PLCB_W56_R')]
    for cid, label, a, b in add:
        for mo in CONTRAST_MODELS:
            add_contrast(cid, label, a, b, mo, 'all_2652_incl_retake', n, PS, BS, rec_s, rec_s_b)
    CTD = pd.DataFrame(CT)
    save(CTD, out, 'H07R4_contrasts.csv')

    # ---------------- mechanical lookup of the pre-declared rule (contrast 3; no executor judgement) -------------------------
    RL = []
    for cid, role in [('C3', 'main'), ('C3_R', 'sensitivity')]:
        for mo in CONTRAST_MODELS:
            r = CTD[(CTD.contrast_id == cid) & (CTD.model == mo) & (CTD.metric == 'auc')].iloc[0]
            RL.append({'contrast_id': cid, 'role': role, 'model': mo, 'delta_auc_a_minus_b': r.delta_a_minus_b, 'ci_lo': r.ci_lo,
                       'ci_hi': r.ci_hi, 'cond_ci_includes_0': bool(r.ci_includes_0),
                       'cond_delta_le_minus_0.02_and_ci_excludes_0': bool((not r.ci_includes_0) and r.delta_a_minus_b <= -0.02),
                       'mechanical_match': rule_lookup(r.delta_a_minus_b, r.ci_lo, r.ci_hi)})
    save(pd.DataFrame(RL), out, 'H07R4_rule_lookup.csv')

    # ---------------- event retention ------------------------------------------------------------------------------------------
    r3 = parse_r3_log(os.path.join(R3_DIR, 'run_arms.log'))
    r4 = {}
    ret_path = os.path.join(out, 'event_retention_round4_run.csv')
    if os.path.exists(ret_path):
        for r in pd.read_csv(ret_path).itertuples():
            r4[r.arm] = (int(r.n_quiz_events), int(r.n_reading_events))
    base_q, base_r = r3['ORIG_W8']
    RT = []
    for nm, src, vals in [('ORIG_W8', 'round3_run_arms.log', r3['ORIG_W8']), ('EXCL_W8', 'round3_run_arms.log', r3['EXCL_W8']),
                          ('EXCL_W8_R', 'round3_run_arms.log', r3['EXCL_W8_R'])] + \
                         [(nm, 'round4_run', r4[nm]) for nm in R4_ARMS + R4_SENS if nm in r4]:
        RT.append({'arm': nm, 'n_quiz_events': vals[0], 'n_reading_events': vals[1], 'quiz_share_of_ORIG_W8': vals[0] / base_q,
                   'reading_share_of_ORIG_W8': vals[1] / base_r, 'source': src})
    if 'ORIG_W8' in r4:
        RT.append({'arm': 'ORIG_W8 (round-4 re-run, gate check)', 'n_quiz_events': r4['ORIG_W8'][0], 'n_reading_events': r4['ORIG_W8'][1],
                   'quiz_share_of_ORIG_W8': r4['ORIG_W8'][0] / base_q, 'reading_share_of_ORIG_W8': r4['ORIG_W8'][1] / base_r,
                   'source': 'round4_run'})
    save(pd.DataFrame(RT), out, 'H07R4_event_retention.csv')

    # ---------------- optional: PLCB_W56 quiz-only ablation (M2') ---------------------------------------------------------------
    abl_path = os.path.join(out, 'arms', 'PLCB_W56_abl_quiz.csv')
    if os.path.exists(abl_path):
        AB = {}
        for nm, d in [('ORIG_W8', os.path.join(R3_DIR, 'arms', 'ORIG_W8_abl_quiz.csv')),
                      ('EXCL_W8', os.path.join(R3_DIR, 'arms', 'EXCL_W8_abl_quiz.csv')), ('PLCB_W56', abl_path)]:
            x = pd.read_csv(d, dtype={'semester': str})
            assert (x.semester.to_numpy() == base.semester.to_numpy()).all() and np.array_equal(x.final.to_numpy(), base.final.to_numpy()), nm
            AB[(nm, 'quiz_only')] = x['m2'].to_numpy(float)
        BA = boot_vectors(boot, base, main_mask, AB, strata=None, with_r=False)
        PA_ = {k: point(base, main_mask, v, with_strata=False) for k, v in AB.items()}
        rows = []
        for nm in ['ORIG_W8', 'EXCL_W8', 'PLCB_W56']:
            k = (nm, 'quiz_only')
            row = {'arm': nm, 'feature_group': 'quiz (M2\')', 'eval_set': 'main_no_retake', 'auc': PA_[k]['auc'],
                   'auc_lo': pct(BA[(k, 'auc')], 2.5), 'auc_hi': pct(BA[(k, 'auc')], 97.5)}
            if nm != 'ORIG_W8':
                k0 = ('ORIG_W8', 'quiz_only')
                d = paired_delta(PA_[k]['auc'], PA_[k0]['auc'], BA[(k, 'auc')], BA[(k0, 'auc')])
                row.update({'delta_vs_ORIG_W8': d['delta'], 'delta_lo': d['ci_lo'], 'delta_hi': d['ci_hi']})
            rows.append(row)
        save(pd.DataFrame(rows), out, 'H07R4_ablation_quiz_optional.csv')
        log(pd.DataFrame(rows).round(4).to_string())
    else:
        log('optional quiz-only ablation: file not found, skipped')

    # ---------------- integrity: round-3 inputs unchanged -------------------------------------------------------------------------
    for p, h in h_before.items():
        assert sha256(p) == h, p
    log('round-3 input files unchanged (sha256 re-verified after analysis)')

    pd.set_option('display.width', 250)
    log(pd.DataFrame(M).round(4).to_string())
    log(CTD.round(4).to_string())
    log(pd.DataFrame(ST).round(4).to_string())
    log(pd.DataFrame(SE).round(4).to_string())
    log(pd.DataFrame(RL).round(4).to_string())
    log(pd.DataFrame(RT).round(4).to_string())
    log('done')
    lg.close()


if __name__ == '__main__':
    main()
