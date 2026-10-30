# -*- coding: utf-8 -*-
"""H-07 arm analysis (post hoc). Spec: H07_addendum3_arms.md.  Usage: python h07_arms_analyze.py --out results"""
import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import h07_arms_lib as A  # noqa: E402

C = A.C
L = A.L
W8_STRATUM = ['1121', '1122', '1131']      # midterm in week 8 (16+2)
MODELS = ["M1'", "M2'", "M3'"]
GROUPS = ['reading', 'quiz', 'sequence', 'class_proxy']
FROZEN_W6_M3 = 0.6694


def pct(a, q):
    return float(np.nanpercentile(np.asarray(a, float), q))


def load_arm(out, name):
    d = pd.read_csv(os.path.join(out, 'arms', f'{name}_oof.csv'), dtype={'semester': str})
    d['m3'] = np.asarray(C.recompute_m3(d))
    return d


def recall20(frame, col):
    rec, prec = L.recall_precision_at(frame, col)
    return rec, prec


def boot_vectors(boot, base, mask, vecs, strata=None, with_r=True):
    """vecs: name -> full-length score. Returns {(name, metric): array(B)}; metrics: auc, r, auc_w8, auc_w9."""
    y = base.y.to_numpy(int)
    f = base.final.to_numpy(float)
    sem = base.semester.to_numpy()
    in8 = np.isin(sem, W8_STRATUM)
    res = {}
    for n in vecs:
        res[(n, 'auc')] = np.full(boot.B, np.nan)
        if with_r:
            res[(n, 'r')] = np.full(boot.B, np.nan)
        if strata:
            res[(n, 'auc_w8')] = np.full(boot.B, np.nan)
            res[(n, 'auc_w9')] = np.full(boot.B, np.nan)
    for b, ix in enumerate(boot.idx):
        ix = ix[mask[ix]]
        yb, fb, s8 = y[ix], f[ix], in8[ix]
        for n, s in vecs.items():
            sb = s[ix]
            res[(n, 'auc')][b] = L.auc_fail(yb, sb)
            if with_r:
                res[(n, 'r')][b] = np.corrcoef(sb, fb)[0, 1]
            if strata:
                res[(n, 'auc_w8')][b] = L.auc_fail(yb[s8], sb[s8])
                res[(n, 'auc_w9')][b] = L.auc_fail(yb[~s8], sb[~s8])
    return res


def point(base, mask, s, with_strata=True):
    y = base.y.to_numpy(int)[mask]
    ss = s[mask]
    out = {'auc': L.auc_fail(y, ss), 'r': float(np.corrcoef(ss, base.final.to_numpy(float)[mask])[0, 1])}
    if with_strata:
        in8 = np.isin(base.semester.to_numpy()[mask], W8_STRATUM)
        out['auc_w8'] = L.auc_fail(y[in8], ss[in8])
        out['auc_w9'] = L.auc_fail(y[~in8], ss[~in8])
    return out


def boot_recall(boot, base, mask, s, B=None):
    y = base.y.to_numpy(int)
    sem = base.semester.to_numpy()
    out = np.empty(boot.B)
    for b, ix in enumerate(boot.idx):
        ix = ix[mask[ix]]
        fr = pd.DataFrame({'semester': sem[ix], 'y': y[ix], 's': s[ix]})
        out[b] = L.recall_precision_at(fr, 's')[0]
    return out


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
    log('# H-07 arms analysis (post hoc) clock=%s' % time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    log('python', platform.python_version(), 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__, 'numpy',
        np.__version__, 'pandas', pd.__version__, 'scipy', scipy.__version__, '| boot B=%d seed=%d' % (L.BOOT_B, L.BOOT_SEED))
    for f in ['H07_addendum2_plan.md', 'H07_addendum3_arms.md', 'h07_arms_lib.py', 'h07_arms_run.py', 'h07_arms_analyze.py']:
        log('sha256', f, C.sha256(os.path.join(HERE, f)))

    base = C.frozen_frame(8)
    n = len(base)
    retake = (base.program5_key == 'retake').to_numpy()
    main_mask = ~retake
    all_mask = np.ones(n, bool)
    boot = L.Boot(base.anon_id.to_numpy())
    arms = {nm: load_arm(out, nm) for nm in A.ARMS + A.SENS}
    for nm, d in arms.items():
        assert (d.semester.to_numpy() == base.semester.to_numpy()).all() and np.array_equal(d.final.to_numpy(), base.final.to_numpy())

    # S4: ORIG_W8 arm equals the A2 gate result
    g = json.load(open(os.path.join(out, 'H07_gate_A2.json'), encoding='utf-8'))
    y = base.y.to_numpy(int)
    for k, col in [("M1'", 'm1'), ("M2'", 'm2'), ("M3'", 'm3')]:
        a = L.auc_fail(y, arms['ORIG_W8'][col].to_numpy(float))
        assert abs(a - g['rebuilt_auc'][k]) < 1e-9, (k, a, g['rebuilt_auc'][k])
    log('S4 ok: ORIG_W8 arm == A2 gate AUCs (1e-9)')

    col = {"M1'": 'm1', "M2'": 'm2', "M3'": 'm3'}
    vec = {}
    for nm in A.ARMS:
        for mo in MODELS:
            vec[(nm, mo)] = arms[nm][col[mo]].to_numpy(float)
    # ablation vectors (M2' only)
    abl = {}
    for nm in ['ORIG_W8', 'STRICT', 'EXCL_W8']:
        for gname in GROUPS:
            d = pd.read_csv(os.path.join(out, 'arms', f'{nm}_abl_{gname}.csv'), dtype={'semester': str})
            assert (d.semester.to_numpy() == base.semester.to_numpy()).all()
            abl[(nm, gname)] = d['m2'].to_numpy(float)
        abl[(nm, 'all')] = vec[(nm, "M2'")]

    t0 = time.time()
    # ---------------- main bootstrap (eval = 2,600, no retake) ----------------------------------------------------
    keys = list(vec) + [('ABL', f'{nm}|{g_}') for (nm, g_) in abl if g_ != 'all']
    V = {k: v for k, v in vec.items()}
    for (nm, g_), v in abl.items():
        if g_ != 'all':
            V[('ABL', f'{nm}|{g_}')] = v
    BD = boot_vectors(boot, base, main_mask, V, strata=True)
    log(f'main bootstrap done {time.time() - t0:.0f}s')
    PT = {k: point(base, main_mask, v) for k, v in V.items()}
    rec_pt = {nm: recall20(base[main_mask].assign(s=vec[(nm, "M3'")][main_mask]), 's')[0] for nm in A.ARMS}
    rec_prec = {nm: recall20(base[main_mask].assign(s=vec[(nm, "M3'")][main_mask]), 's')[1] for nm in A.ARMS}
    rb = {nm: boot_recall(boot, base, main_mask, vec[(nm, "M3'")]) for nm in ['ORIG_W8', 'STRICT', 'EXCL_W8', 'ORIG_W6', 'ORIG_W7', 'EXCL_W7']}
    log(f'recall bootstrap done {time.time() - t0:.0f}s')
    nfail = int(base.y[main_mask].sum())

    # ---------------- metrics ---------------------------------------------------------------------------------------
    M = []
    for nm in A.ARMS:
        for mo in MODELS:
            k = (nm, mo)
            row = {'arm': nm, 'eval_set': 'main_no_retake', 'n': int(main_mask.sum()), 'n_fail': nfail, 'model': mo,
                   'auc': PT[k]['auc'], 'auc_lo': pct(BD[(k, 'auc')], 2.5), 'auc_hi': pct(BD[(k, 'auc')], 97.5),
                   'r': PT[k]['r'], 'r_lo': pct(BD[(k, 'r')], 2.5), 'r_hi': pct(BD[(k, 'r')], 97.5)}
            if mo == "M3'":
                row['recall20'] = rec_pt[nm]
                row['precision20'] = rec_prec[nm]
                if nm in rb:
                    row['recall20_lo'], row['recall20_hi'] = pct(rb[nm], 2.5), pct(rb[nm], 97.5)
            M.append(row)
    MET = pd.DataFrame(M)
    MET.to_csv(os.path.join(out, 'H07_arms_metrics.csv'), index=False, encoding='utf-8-sig')

    # ---------------- contrasts vs ORIG_W8 + verdicts -------------------------------------------------------------------
    CT = []
    for nm in ['STRICT', 'EXCL_W8']:
        for mo in MODELS:
            for metric in ['auc', 'r']:
                d = PT[(nm, mo)][metric] - PT[('ORIG_W8', mo)][metric]
                dist = BD[((nm, mo), metric)] - BD[(('ORIG_W8', mo), metric)]
                CT.append({'arm': nm, 'vs': 'ORIG_W8', 'model': mo, 'metric': metric, 'delta_arm_minus_orig': d,
                           'ci_lo': pct(dist, 2.5), 'ci_hi': pct(dist, 97.5), 'loss': -d, 'loss_ci_hi': -pct(dist, 2.5)})
        d = (rec_pt[nm] - rec_pt['ORIG_W8']) * 100
        dist = (rb[nm] - rb['ORIG_W8']) * 100
        CT.append({'arm': nm, 'vs': 'ORIG_W8', 'model': "M3'", 'metric': 'recall20_pp', 'delta_arm_minus_orig': d,
                   'ci_lo': pct(dist, 2.5), 'ci_hi': pct(dist, 97.5), 'loss': -d, 'loss_ci_hi': -pct(dist, 2.5)})
    CT = pd.DataFrame(CT)
    ver = []
    for nm in ['STRICT', 'EXCL_W8']:
        la = CT[(CT.arm == nm) & (CT.model == "M3'") & (CT.metric == 'auc')].iloc[0]
        lr = CT[(CT.arm == nm) & (CT.metric == 'recall20_pp')].iloc[0]
        ok_auc, ok_rec = la.loss <= 0.02, lr.loss <= 3.0
        verdict = ('criterion 1 met (AUC loss <= 0.02 and Recall loss <= 3 pp)' if ok_auc and ok_rec else
                   'criterion 1 NOT met (AUC loss > 0.02 and Recall loss > 3 pp)' if (not ok_auc and not ok_rec) else
                   'criterion 1 NOT met (only one of AUC loss <= 0.02 / Recall loss <= 3 pp holds)')
        ver.append({'arm': nm, 'M3_auc_loss': la.loss, 'M3_auc_loss_ci_hi': la.loss_ci_hi, 'M3_recall_loss_pp': lr.loss,
                    'M3_recall_loss_pp_ci_hi': lr.loss_ci_hi, 'arm_M3_auc': PT[(nm, "M3'")]['auc'],
                    'rebuilt_ORIG_W6_M3_auc': PT[('ORIG_W6', "M3'")]['auc'], 'frozen_W6_M3_auc': FROZEN_W6_M3,
                    'arm_minus_rebuilt_W6': PT[(nm, "M3'")]['auc'] - PT[('ORIG_W6', "M3'")]['auc'],
                    'arm_minus_frozen_W6': PT[(nm, "M3'")]['auc'] - FROZEN_W6_M3, 'criterion1': verdict})
    VER = pd.DataFrame(ver)
    CT.to_csv(os.path.join(out, 'H07_arms_contrasts.csv'), index=False, encoding='utf-8-sig')
    VER.to_csv(os.path.join(out, 'H07_arms_verdicts.csv'), index=False, encoding='utf-8-sig')

    # ---------------- stratified ------------------------------------------------------------------------------------------
    ST = []
    for mo in MODELS:
        for nm in A.ARMS:
            for st in ['w8', 'w9']:
                k = (nm, mo)
                row = {'arm': nm, 'model': mo, 'stratum': 'midterm_in_W8 (1121,1122,1131)' if st == 'w8' else 'midterm_in_W9 (1111,1112,1132,1141)',
                       'auc': PT[k]['auc_' + st], 'auc_lo': pct(BD[(k, 'auc_' + st)], 2.5), 'auc_hi': pct(BD[(k, 'auc_' + st)], 97.5)}
                if nm in ('STRICT', 'EXCL_W8'):
                    dist = BD[(k, 'auc_' + st)] - BD[(('ORIG_W8', mo), 'auc_' + st)]
                    row.update({'delta_vs_ORIG_W8': PT[k]['auc_' + st] - PT[('ORIG_W8', mo)]['auc_' + st],
                                'delta_lo': pct(dist, 2.5), 'delta_hi': pct(dist, 97.5)})
                ST.append(row)
        for nm in ['STRICT', 'EXCL_W8']:
            k = (nm, mo)
            d8 = PT[k]['auc_w8'] - PT[('ORIG_W8', mo)]['auc_w8']
            d9 = PT[k]['auc_w9'] - PT[('ORIG_W8', mo)]['auc_w9']
            dist = (BD[(k, 'auc_w8')] - BD[(('ORIG_W8', mo), 'auc_w8')]) - (BD[(k, 'auc_w9')] - BD[(('ORIG_W8', mo), 'auc_w9')])
            lo, hi = pct(dist, 2.5), pct(dist, 97.5)
            ST.append({'arm': nm, 'model': mo, 'stratum': 'DiD = delta(W8-midterm stratum) - delta(W9-midterm stratum)',
                       'delta_vs_ORIG_W8': d8 - d9, 'delta_lo': lo, 'delta_hi': hi,
                       'auc': np.nan, 'did_ci_includes_0': bool(lo <= 0 <= hi)})
    pd.DataFrame(ST).to_csv(os.path.join(out, 'H07_arms_stratified.csv'), index=False, encoding='utf-8-sig')

    # ---------------- trajectory -------------------------------------------------------------------------------------------
    TR = []
    for snap, (o, e) in {'W6': ('ORIG_W6', 'EXCL_W6'), 'W7': ('ORIG_W7', 'EXCL_W7'), 'W8': ('ORIG_W8', 'EXCL_W8')}.items():
        for mo in MODELS:
            for label, nm in [('ORIG', o), ('EXCL', e)]:
                k = (nm, mo)
                TR.append({'snapshot': snap, 'version': label, 'arm': nm, 'model': mo, 'auc': PT[k]['auc'],
                           'auc_lo': pct(BD[(k, 'auc')], 2.5), 'auc_hi': pct(BD[(k, 'auc')], 97.5), 'r': PT[k]['r']})
    pd.DataFrame(TR).to_csv(os.path.join(out, 'H07_arms_trajectory.csv'), index=False, encoding='utf-8-sig')

    # ---------------- ablation ---------------------------------------------------------------------------------------------
    AB = []
    for gname in GROUPS + ['all']:
        for nm in ['ORIG_W8', 'STRICT', 'EXCL_W8']:
            k = (nm, "M2'") if gname == 'all' else ('ABL', f'{nm}|{gname}')
            k0 = ('ORIG_W8', "M2'") if gname == 'all' else ('ABL', f'ORIG_W8|{gname}')
            row = {'feature_group': gname, 'arm': nm, 'auc': PT[k]['auc'], 'auc_lo': pct(BD[(k, 'auc')], 2.5),
                   'auc_hi': pct(BD[(k, 'auc')], 97.5)}
            if nm != 'ORIG_W8':
                dist = BD[(k, 'auc')] - BD[(k0, 'auc')]
                row.update({'delta_vs_ORIG_W8': PT[k]['auc'] - PT[k0]['auc'], 'delta_lo': pct(dist, 2.5), 'delta_hi': pct(dist, 97.5)})
            AB.append(row)
    pd.DataFrame(AB).to_csv(os.path.join(out, 'H07_ablation.csv'), index=False, encoding='utf-8-sig')

    # ---------------- sensitivity: retake own-calendar windows (eval = 2,652) ------------------------------------------------
    sv = {}
    for nm in ['ORIG_W8', 'STRICT_R', 'EXCL_W8_R']:
        for mo in MODELS:
            sv[(nm, mo)] = arms[nm][col[mo]].to_numpy(float)
    BS = boot_vectors(boot, base, all_mask, sv, strata=None)
    SE = []
    rr = {nm: boot_recall(boot, base, all_mask, sv[(nm, "M3'")]) for nm in ['ORIG_W8', 'STRICT_R', 'EXCL_W8_R']}
    for nm in ['ORIG_W8', 'STRICT_R', 'EXCL_W8_R']:
        for mo in MODELS:
            k = (nm, mo)
            pt = point(base, all_mask, sv[k], with_strata=False)
            row = {'arm': nm, 'eval_set': 'all_2652_incl_retake', 'model': mo, 'auc': pt['auc'], 'auc_lo': pct(BS[(k, 'auc')], 2.5),
                   'auc_hi': pct(BS[(k, 'auc')], 97.5), 'r': pt['r']}
            if nm != 'ORIG_W8':
                dist = BS[(k, 'auc')] - BS[(('ORIG_W8', mo), 'auc')]
                row.update({'delta_auc_vs_ORIG_W8': pt['auc'] - point(base, all_mask, sv[('ORIG_W8', mo)], False)['auc'],
                            'delta_lo': pct(dist, 2.5), 'delta_hi': pct(dist, 97.5)})
            if mo == "M3'":
                row['recall20'] = recall20(base.assign(s=sv[k]), 's')[0]
                if nm != 'ORIG_W8':
                    dd = (rr[nm] - rr['ORIG_W8']) * 100
                    row['recall20_delta_pp_vs_ORIG_W8'] = (row['recall20'] - recall20(base.assign(s=sv[('ORIG_W8', mo)]), 's')[0]) * 100
                    row['recall20_delta_lo'], row['recall20_delta_hi'] = pct(dd, 2.5), pct(dd, 97.5)
            SE.append(row)
    rk_mask = retake
    for nm in ['ORIG_W8', 'STRICT_R', 'EXCL_W8_R']:
        yk = base.y.to_numpy(int)[rk_mask]
        for mo in MODELS:
            SE.append({'arm': nm, 'eval_set': 'retake_only (descriptive, n=%d, fails=%d)' % (int(rk_mask.sum()), int(yk.sum())),
                       'model': mo, 'auc': L.auc_fail(yk, sv[(nm, mo)][rk_mask])})
    pd.DataFrame(SE).to_csv(os.path.join(out, 'H07_sensitivity_retake.csv'), index=False, encoding='utf-8-sig')
    log(MET.round(4).to_string())
    log(CT.round(4).to_string())
    log(VER.round(4).T.to_string())
    log('done')
    lg.close()


if __name__ == '__main__':
    main()
