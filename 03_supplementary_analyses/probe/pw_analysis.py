# -*- coding: utf-8 -*-
"""Phase-Week Probe -- estimands, gates and decision rule (identifiers ASCII-only).

Primary (paired, pooled over 1121 / 1122 / 1131, deployment construction):
    dAUC = AUC(W10) - AUC(W9),  dr = r(W10) - r(W9);   secondary dRecall@20% (within-semester lists).
Standing student-cluster bootstrap of Round 5: B = 2000, seed 20261031, per-student multiplicities over the full 2,652-record
base (r5_lib.make_weights), the same draws for every statistic, so differences are paired.
Decision rule (handoff s3): supported iff upper 95% CL of pooled dAUC <= 0.020 AND upper 95% CL of pooled dr <= 0.030.
Reads the lossless LOSO score files written by pw_loso.py; no model is fitted here."""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('PW_ROOT', HERE.parent))
os.environ.setdefault('R5_RES', str(ROOT / 'results' / '_r5lib_scratch'))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

import r5_lib as L

RES = ROOT / 'results'
EVAL = ['1121', '1122', '1131']                      # the three 16+2 semesters (midterm in Week 8)
AUC_BOUND, R_BOUND = 0.020, 0.030                    # locked tolerances (upper 95% CL of the paired loss)
EXPECT = {'1121': (507, 99), '1122': (234, 63), '1131': (473, 51)}       # frozen anchors (R5-6.csv)


def read_scores(W):
    d = pd.read_csv(RES / f'pw_loso_raw_W{W}.csv', dtype={'anon_id': str, 'semester': str}, float_precision='round_trip')
    return d


def main():
    base, _ = L.load_base()
    N = len(base)
    y = base.y.to_numpy().astype(int)
    f = base.final.to_numpy(float)
    sem = base.semester.to_numpy()
    raw = {W: read_scores(W) for W in (8, 9, 10)}
    for W, d in raw.items():
        assert list(zip(d.semester, d.anon_id)) == list(zip(base.semester, base.anon_id))
        assert np.array_equal(d.y.to_numpy(), y)

    # ---- scores: deployment blend = mean of within-semester percentile ranks of m1 and m2 (pass-oriented) ----
    dep = {W: L.deploy_blend(raw[W].m1, raw[W].m2, sem) for W in (8, 9, 10)}
    pooled8 = L.pooled_blend(raw[8].m1.to_numpy(), raw[8].m2.to_numpy())

    # ---- gate report ----
    Wm = L.make_weights(base.stu.to_numpy())                           # standing bootstrap, (2000 x N)
    one = np.ones((1, N), np.float32)
    all7 = L.sem_groups_of(base.sem_i)
    gates = []
    # G1 (same build): W8 pooled blend, full development set
    a8 = L.auc_fail(y, pooled8)
    r8 = L.pearson(pooled8, f)
    gates.append({'gate': 'G1 W8 pooled blend (frozen anchor)', 'AUC': 0.7274, 'r': 0.4484, 'source': 'Round 5 R5-0 (frozen scores)'})
    gates.append({'gate': 'G1 W8 pooled blend (same build rebuilt, from events)', 'AUC': a8, 'r': r8,
                  'AUC_delta_vs_frozen': a8 - 0.7274382853045616, 'r_delta_vs_frozen': r8 - 0.44837397612950636,
                  'source': 'pw_loso W8; Round 5 gate: -0.0019667 / -0.0045911'})
    # G2 (same build): W10 deployment blend, all 7 semesters, within-semester
    S10 = dep[10].reshape(-1, 1)
    ad, nd = L.auc_weighted(Wm, S10, y, all7)
    rd, _ = L.corr_weighted(Wm, S10, f, all7)
    auc_b, r_b = L.aggregate_deploy(ad, rd, nd)
    a1, n1 = L.auc_weighted(one, S10, y, all7)
    r1, _ = L.corr_weighted(one, S10, f, all7)
    auc_pt, r_pt = L.aggregate_deploy(a1, r1, n1)
    lo, hi = L.ci(auc_b[:, 0])
    gates.append({'gate': 'G2 W10 deployment blend, 7 semesters, within-semester AUC (frozen anchor)', 'AUC': 0.7351085520234132,
                  'AUC_lo': 0.7116910937467369, 'AUC_hi': 0.758380375723999, 'source': 'Round 5 R5-11'})
    gates.append({'gate': 'G2 W10 deployment blend, 7 semesters (same build rebuilt, from events)', 'AUC': float(auc_pt[0, 0]),
                  'AUC_lo': lo, 'AUC_hi': hi, 'r': float(r_pt[0, 0]),
                  'AUC_delta_vs_frozen': float(auc_pt[0, 0]) - 0.7351085520234132, 'source': 'pw_loso W10'})

    # ---- evaluation subset: records of 1121 / 1122 / 1131 ----
    gi = [np.flatnonzero(sem == s) for s in EVAL]
    info = []
    for s, ix in zip(EVAL, gi):
        n, nf = len(ix), int(y[ix].sum())
        assert (n, nf) == EXPECT[s], (s, n, nf)
        info.append((s, n, nf))
    n_retake = int(base.retake.to_numpy()[np.concatenate(gi)].sum())
    assert n_retake == 0, 'retake-cohort records in the evaluation semesters'
    S = np.column_stack([dep[9], dep[10]])                              # col 0 = W9, col 1 = W10 (pass-oriented)
    ad, nd = L.auc_weighted(Wm, S, y, gi)                               # (B x 3 x 2), (B x 3)
    rd, _ = L.corr_weighted(Wm, S, f, gi)
    a_pool, r_pool = L.aggregate_deploy(ad, rd, nd)                    # (B x 2)
    a1, n1 = L.auc_weighted(one, S, y, gi)
    r1, _ = L.corr_weighted(one, S, f, gi)
    a_pool_pt, r_pool_pt = L.aggregate_deploy(a1, r1, n1)              # (1 x 2)

    # Recall@20%: within-semester lists from the point-estimate scores, fixed across draws (Round 5 convention)
    ev = np.concatenate(gi)
    lists, rec_draw, rec_pt = {}, {}, {}
    for col, W in ((0, 9), (1, 10)):
        fr = pd.DataFrame({'semester': sem[ev], 'y': y[ev], 's': S[ev, col]}).reset_index(drop=True)
        rpt, _, lst = L.recall_at(fr, 's')                             # tie-break seed 7, ceil(0.2 n) per semester
        full = np.zeros(N, bool)
        full[ev] = lst
        lists[W] = full
        tp = (full & (y == 1)).astype(np.float32)
        yy = y.astype(np.float32)
        per_d, per_p = [], []
        for ix in gi:
            per_d.append((Wm[:, ix] @ tp[ix]) / (Wm[:, ix] @ yy[ix]))
            per_p.append(int(tp[ix].sum()) / int(yy[ix].sum()))
        tp_all = sum(Wm[:, ix] @ tp[ix] for ix in gi)
        fl_all = sum(Wm[:, ix] @ yy[ix] for ix in gi)
        rec_draw[W] = np.column_stack(per_d + [tp_all / fl_all])        # (B x 4): 3 semesters + pooled
        rec_pt[W] = np.array(per_p + [float(rpt)])
        assert abs(rec_pt[W][3] - sum(int(tp[ix].sum()) for ix in gi) / sum(int(yy[ix].sum()) for ix in gi)) < 1e-12

    rows = []
    labels = EVAL + ['pooled (3 semesters)']
    for k, lab in enumerate(labels):
        if k < 3:
            s, n, nf = info[k]
            auc9, auc10 = float(a1[0, k, 0]), float(a1[0, k, 1])
            r9, r10 = float(r1[0, k, 0]), float(r1[0, k, 1])
            d_auc, d_r = ad[:, k, 1] - ad[:, k, 0], rd[:, k, 1] - rd[:, k, 0]
            dr_a, dr_r = ad[:, k, :], rd[:, k, :]
        else:
            n, nf = sum(i[1] for i in info), sum(i[2] for i in info)
            auc9, auc10 = float(a_pool_pt[0, 0]), float(a_pool_pt[0, 1])
            r9, r10 = float(r_pool_pt[0, 0]), float(r_pool_pt[0, 1])
            d_auc, d_r = a_pool[:, 1] - a_pool[:, 0], r_pool[:, 1] - r_pool[:, 0]
            dr_a, dr_r = a_pool, r_pool
        rec9, rec10 = float(rec_pt[9][k]), float(rec_pt[10][k])
        d_rec = rec_draw[10][:, k] - rec_draw[9][:, k]
        row = {'row': lab, 'n': n, 'n_failing': nf}
        for nm, pt, dr in (('AUC_W9', auc9, dr_a[:, 0]), ('AUC_W10', auc10, dr_a[:, 1]),
                           ('r_W9', r9, dr_r[:, 0]), ('r_W10', r10, dr_r[:, 1]),
                           ('Recall20_W9', rec9, rec_draw[9][:, k]), ('Recall20_W10', rec10, rec_draw[10][:, k])):
            lo, hi = L.ci(dr)
            row.update({nm: pt, nm + '_lo': lo, nm + '_hi': hi})
        for nm, pt, dr in (('dAUC_W10_minus_W9', auc10 - auc9, d_auc), ('dr_W10_minus_W9', r10 - r9, d_r),
                           ('dRecall20_W10_minus_W9', rec10 - rec9, d_rec)):
            lo, hi = L.ci(dr)
            row.update({nm: pt, nm + '_lo': lo, nm + '_hi': hi})
        rows.append(row)
    R = pd.DataFrame(rows)
    pooled = R.iloc[3]
    ucl_auc, ucl_r = float(pooled['dAUC_W10_minus_W9_hi']), float(pooled['dr_W10_minus_W9_hi'])
    supported = bool(ucl_auc <= AUC_BOUND and ucl_r <= R_BOUND)
    R['decision_basis'] = ''
    R.loc[3, 'decision_basis'] = (f'upper 95% CL dAUC {ucl_auc:.4f} (bound {AUC_BOUND}); upper 95% CL dr {ucl_r:.4f} '
                                  f'(bound {R_BOUND}); phase week {"SUPPORTED" if supported else "NOT SUPPORTED"}')
    R['bootstrap'] = 'student-cluster, B=2000, seed 20261031, paired; Recall@20% lists fixed at the point-estimate scores'
    R.to_csv(RES / 'PW_probe_results.csv', index=False, encoding='utf-8-sig')
    G = pd.DataFrame(gates)
    G.to_csv(RES / 'PW_gates.csv', index=False, encoding='utf-8-sig')

    # ---- W9 OOF scores (record level, de-identified keys as in Round 5) ----
    out = pd.DataFrame({'semester': base.semester, 'anon_id': base.anon_id,
                        'evaluation_semester_16plus2': np.isin(sem, EVAL).astype(int),
                        'm1_BAS_W9': raw[9].m1, 'm2_XGBoost_W9': raw[9].m2,
                        'score_W9_deployment_blend_pass_oriented': dep[9],
                        'm1_BAS_W10_samebuild': raw[10].m1, 'm2_XGBoost_W10_samebuild': raw[10].m2,
                        'score_W10_deployment_blend_pass_oriented_samebuild': dep[10],
                        'outcome_fail_semester_score_lt60': y, 'semester_score': base.final})
    out.to_csv(RES / 'PW_W9_oof_scores.csv', index=False, encoding='utf-8-sig', float_format='%.17g')
    np.savez_compressed(RES / 'PW_boot_draws.npz', dAUC=a_pool[:, 1] - a_pool[:, 0], dr=r_pool[:, 1] - r_pool[:, 0],
                        dRecall=rec_draw[10][:, 3] - rec_draw[9][:, 3])
    json.dump({'supported': supported, 'ucl_dAUC': ucl_auc, 'ucl_dr': ucl_r, 'n_retake_in_eval': n_retake},
              open(RES / 'PW_decision.json', 'w'), indent=1)
    pd.set_option('display.width', 250)
    print(G.to_string())
    print(R.drop(columns=['decision_basis', 'bootstrap']).T.to_string())
    print(R.loc[3, 'decision_basis'])


if __name__ == '__main__':
    main()
