# -*- coding: utf-8 -*-
"""H-09 point 4: selection rule rerun on FULL_RERUN vs NO4T retrained scores (post hoc). Plan: H09p4_plan.md.
Usage: python h09p4_select.py --out results   (needs results/H09p4_oof_full.csv and H09p4_oof_no4t.csv)"""
import argparse
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import h09p4_lib as H  # noqa: E402

C = H.C
L = H.L
B3_VALUES = {"M2'": -0.0274, "M3'": -0.0160}


def pct(a, q):
    return float(np.nanpercentile(np.asarray(a, float), q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    a = ap.parse_args()
    out = a.out
    logf = open(os.path.join(out, 'run_select.log'), 'w', encoding='utf-8')

    def log(*x):
        msg = ' '.join(str(v) for v in x)
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    import scipy
    import sklearn
    import xgboost
    log('# H-09p4 selection rerun (post hoc) clock=%s' % time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    log('python', platform.python_version(), 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__,
        'numpy', np.__version__, 'pandas', pd.__version__, 'scipy', scipy.__version__)
    log('boot B=%d seed=%d; q95=%.6f q993=%.6f' % (L.BOOT_B, L.BOOT_SEED, H.Q95, H.Q993))
    for f in ['H09p4_plan.md', 'h09p4_lib.py', 'h09p4_select.py']:
        log('sha256', f, C.sha256(os.path.join(HERE, f)))
    for f in ['H09p4_oof_full.csv', 'H09p4_oof_no4t.csv']:
        log('sha256', f, C.sha256(os.path.join(out, f)))

    frozen = {W: C.frozen_frame(W) for W in H.WEEKS}
    base = frozen[12]
    y = base.y.to_numpy(int)
    fin = base.final.to_numpy(float)
    semc = np.array([C.D7.index(s) for s in base.semester])
    boot = L.Boot(base.anon_id.to_numpy())

    frames = {}
    for cfg in ['full', 'no4t']:
        o = pd.read_csv(os.path.join(out, f'H09p4_oof_{cfg}.csv'), dtype={'semester': str})
        for W in H.WEEKS:
            g = o[o.W == f'W{W}'].reset_index(drop=True)
            assert len(g) == 2652
            assert (g.semester.to_numpy() == base.semester.to_numpy()).all()
            assert np.array_equal(g.final.to_numpy(), fin) and np.array_equal(g.y.to_numpy(), y)
            g['m3'] = np.asarray(C.recompute_m3(g))
            frames[(cfg, W)] = g

    # ---- curves ---------------------------------------------------------------------------------------------------
    t0 = time.time()
    curves = {'full': {}, 'no4t': {}}
    for W in H.WEEKS:
        c1 = H.curve(boot, y, frames[('full', W)].m1.to_numpy(float), fin, semc)  # M1' identical in both configs (TB4)
        assert np.array_equal(frames[('full', W)].m1.to_numpy(), frames[('no4t', W)].m1.to_numpy())
        for cfg in ['full', 'no4t']:
            curves[cfg][("M1'", W)] = c1
            for mn in ["M2'", "M3'"]:
                curves[cfg][(mn, W)] = H.curve(boot, y, frames[(cfg, W)][H.MCOL[mn]].to_numpy(float), fin, semc)
        log(f'curves W{W} done ({time.time() - t0:.0f}s)')
    fcur = {}
    for mn in ["M2'", "M3'"]:  # frozen W8 curves for the bridge only
        fcur[mn] = H.curve(boot, y, frozen[8][H.MCOL[mn]].to_numpy(float), fin, semc)

    # ---- 1. selection tables ----------------------------------------------------------------------------------------
    tabs = []
    for cfg, label in [('full', 'FULL_RERUN'), ('no4t', 'NO4T')]:
        t = H.selection_table(curves[cfg])
        t.insert(0, 'config', label)
        tabs.append(t)
    TAB = pd.concat(tabs, ignore_index=True)
    TAB.to_csv(os.path.join(out, 'H09p4_selection_table.csv'), index=False, encoding='utf-8-sig')
    WS = []
    for label in ['FULL_RERUN', 'NO4T']:
        w = H.wstar_summary(TAB[TAB.config == label])
        w.insert(0, 'config', label)
        WS.append(w)
    ref = pd.read_csv(os.path.join(C.PACK, '05_plans_and_prior', 'H09c_2_selection_rerun.csv'), encoding='utf-8-sig')
    w = H.wstar_summary(ref)
    w.insert(0, 'config', 'FROZEN_OOF (existing H09c table)')
    WS.append(w)
    WS = pd.concat(WS, ignore_index=True)
    WS.to_csv(os.path.join(out, 'H09p4_wstar_summary.csv'), index=False, encoding='utf-8-sig')
    log(WS.to_string())

    # ---- 2. W8 loss bounds ---------------------------------------------------------------------------------------------
    cols = ['config', 'variant', 'model', 'W', 'auc', 'r', 'auc_loss', 'auc_loss_hi95', 'auc_loss_hi993', 'r_loss',
            'r_loss_hi95', 'r_loss_hi993', 'point_ok', 'ci95_ok', 'ci993_ok']
    TAB[TAB.W == 'W8'][cols].to_csv(os.path.join(out, 'H09p4_w8_loss_bounds.csv'), index=False, encoding='utf-8-sig')

    # ---- 3. blend advantage ---------------------------------------------------------------------------------------------
    rows = []
    for cfg, label in [('full', 'FULL_RERUN'), ('no4t', 'NO4T')]:
        for var in ['semester_stratified', 'pooled']:
            for W in H.WEEKS:
                for other in ["M2'", "M1'"]:
                    for metric, k in [('auc', 1), ('r', 0)]:
                        d_pt = curves[cfg][("M3'", W)]['pt'][var][k] - curves[cfg][(other, W)]['pt'][var][k]
                        dist = curves[cfg][("M3'", W)]['dist'][var][k] - curves[cfg][(other, W)]['dist'][var][k]
                        lo, hi = pct(dist, 2.5), pct(dist, 97.5)
                        rows.append({'config': label, 'variant': var, 'W': f'W{W}', 'comparison': f"M3' - {other}",
                                     'metric': metric, 'delta': d_pt, 'ci_lo': lo, 'ci_hi': hi, 'ci_includes_0': bool(lo <= 0 <= hi)})
    BL = pd.DataFrame(rows)
    # descriptive "changed?" check (plan G-B2): (i) AUC rank order of models, (ii) CI-includes-0 flips
    chg = []
    for var in ['semester_stratified', 'pooled']:
        for W in H.WEEKS:
            order = {}
            for cfg in ['full', 'no4t']:
                order[cfg] = ' > '.join(sorted(H.MODELS, key=lambda m: -curves[cfg][(m, W)]['pt'][var][1]))
            for other in ["M2'", "M1'"]:
                g = BL[(BL.variant == var) & (BL.W == f'W{W}') & (BL.comparison == f"M3' - {other}") & (BL.metric == 'auc')].set_index('config')
                chg.append({'variant': var, 'W': f'W{W}', 'comparison': f"M3' - {other}",
                            'delta_full': g.loc['FULL_RERUN', 'delta'], 'delta_no4t': g.loc['NO4T', 'delta'],
                            'ci_includes_0_full': g.loc['FULL_RERUN', 'ci_includes_0'], 'ci_includes_0_no4t': g.loc['NO4T', 'ci_includes_0'],
                            'ci_flip': bool(g.loc['FULL_RERUN', 'ci_includes_0'] != g.loc['NO4T', 'ci_includes_0']),
                            'rank_full': order['full'], 'rank_no4t': order['no4t'], 'rank_changed': order['full'] != order['no4t']})
    CH = pd.DataFrame(chg)
    BL.to_csv(os.path.join(out, 'H09p4_blend_advantage.csv'), index=False, encoding='utf-8-sig')
    CH.to_csv(os.path.join(out, 'H09p4_blend_advantage_changed.csv'), index=False, encoding='utf-8-sig')
    log(CH.round(4).to_string())

    # ---- 4. B3 bridge at W8 -------------------------------------------------------------------------------------------
    br = []
    for mn in ["M2'", "M3'"]:
        for var in ['pooled', 'semester_stratified']:
            for basis, bcur in [('NO4T - FULL_RERUN', curves['full'][(mn, 8)]), ('NO4T - FROZEN_OOF', fcur[mn])]:
                d_pt = curves['no4t'][(mn, 8)]['pt'][var][1] - bcur['pt'][var][1]
                dist = curves['no4t'][(mn, 8)]['dist'][var][1] - bcur['dist'][var][1]
                br.append({'model': mn, 'variant': var, 'basis': basis, 'delta_auc': d_pt, 'ci_lo': pct(dist, 2.5),
                           'ci_hi': pct(dist, 97.5), 'existing_B3_value': B3_VALUES[mn],
                           'diff_from_B3': d_pt - B3_VALUES[mn]})
    BR = pd.DataFrame(br)
    BR.to_csv(os.path.join(out, 'H09p4_b3_bridge.csv'), index=False, encoding='utf-8-sig')
    log(BR.round(4).to_string())

    # ---- 5. reproduction diagnostics (no gate) -------------------------------------------------------------------------
    dg = []
    for W in H.WEEKS:
        for mn in H.MODELS:
            fz = L.auc_fail(y, frozen[W][H.MCOL[mn]].to_numpy(float))
            rr = L.auc_fail(y, frames[('full', W)][H.MCOL[mn]].to_numpy(float))
            nn = L.auc_fail(y, frames[('no4t', W)][H.MCOL[mn]].to_numpy(float))
            dg.append({'W': f'W{W}', 'model': mn, 'auc_frozen_oof': fz, 'auc_full_rerun': rr, 'auc_no4t': nn,
                       'rerun_minus_frozen': rr - fz, 'no4t_minus_rerun': nn - rr})
    DG = pd.DataFrame(dg)
    DG.to_csv(os.path.join(out, 'H09p4_repro_diagnostics.csv'), index=False, encoding='utf-8-sig')
    log(DG.round(4).to_string())
    log('done')
    logf.close()


if __name__ == '__main__':
    main()
