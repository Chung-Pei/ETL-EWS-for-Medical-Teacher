# -*- coding: utf-8 -*-
"""H-09 point 4 library (post hoc). Plan: H09p4_plan.md.  Selection-rule definitions were recovered bit-for-bit from the
existing H09c_2_selection_rerun.csv (see TB0)."""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import pa_common as C  # noqa: E402

L = C.L
D7 = C.D7
WEEKS = [6, 8, 10, 12]
MODELS = ["M1'", "M2'", "M3'"]
MCOL = {"M1'": 'm1', "M2'": 'm2', "M3'": 'm3'}
D_AUC, D_R = 0.02, 0.03
Q95 = 97.5
Q993 = 100 * (1 - 0.05 / 14)
ETL_DIR = os.path.join(C.PACK, '02_pipeline', 'etl_v7_15_34')


# ---------------------------------------------------------------------------------------------------------------
# selection-rule statistics
# ---------------------------------------------------------------------------------------------------------------
def _pearson(a, b):
    return np.corrcoef(a, b)[0, 1]


def _macro(y, s, f, semc):
    rs, aucs, ns = [], [], []
    for k in range(len(D7)):
        m = semc == k
        yk = y[m]
        if yk.size == 0 or yk.sum() == 0 or yk.sum() == yk.size:
            continue
        rs.append(_pearson(s[m], f[m]))
        aucs.append(L.auc_fail(yk, s[m]))
        ns.append(m.sum())
    ns = np.array(ns, float)
    z = np.arctanh(np.clip(rs, -0.999, 0.999))
    wz = ((ns - 3) * z).sum() / (ns - 3).sum()
    return float(np.tanh(wz)), float((np.array(aucs) * ns).sum() / ns.sum())


def curve(boot, y, s, f, semc):
    """Point estimates and bootstrap distributions of (r, AUC) for the pooled and semester_stratified variants."""
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    f = np.asarray(f, float)
    out = {'pt': {'pooled': (_pearson(s, f), L.auc_fail(y, s)), 'semester_stratified': _macro(y, s, f, semc)}}
    pr, pa, sr, sa = (np.empty(boot.B) for _ in range(4))
    for b, ix in enumerate(boot.idx):
        yb, sb, fb, cb = y[ix], s[ix], f[ix], semc[ix]
        pa[b] = L.auc_fail(yb, sb)
        pr[b] = _pearson(sb, fb)
        sr[b], sa[b] = _macro(yb, sb, fb, cb)
    out['dist'] = {'pooled': (pr, pa), 'semester_stratified': (sr, sa)}
    return out


def selection_table(curves, variants=('pooled', 'semester_stratified'), ref_w=12):
    """curves[(model, W)] -> curve(). Returns the H09c-schema table."""
    rows = []
    for var in variants:
        for mn in MODELS:
            for W in WEEKS:
                rW, aW = curves[(mn, W)]['dist'][var]
                r12, a12 = curves[(mn, ref_w)]['dist'][var]
                ptr, pta = curves[(mn, W)]['pt'][var]
                ptr12, pta12 = curves[(mn, ref_w)]['pt'][var]
                al, rl = a12 - aW, r12 - rW
                row = {'variant': var, 'model': mn, 'W': f'W{W}', 'auc': pta, 'r': ptr,
                       'auc_loss': pta12 - pta,
                       'auc_loss_hi95': np.nanpercentile(al, Q95), 'auc_loss_hi993': np.nanpercentile(al, Q993),
                       'r_loss': ptr12 - ptr,
                       'r_loss_hi95': np.nanpercentile(rl, Q95), 'r_loss_hi993': np.nanpercentile(rl, Q993)}
                row['point_ok'] = bool(row['auc_loss'] <= D_AUC and row['r_loss'] <= D_R)
                row['ci95_ok'] = bool(row['auc_loss_hi95'] <= D_AUC and row['r_loss_hi95'] <= D_R)
                row['ci993_ok'] = bool(row['auc_loss_hi993'] <= D_AUC and row['r_loss_hi993'] <= D_R)
                rows.append(row)
    return pd.DataFrame(rows)


def wstar_summary(tab):
    out = []
    for (var, mn), g in tab.groupby(['variant', 'model'], sort=False):
        g = g.set_index('W').loc[[f'W{w}' for w in WEEKS]]
        row = {'variant': var, 'model': mn}
        for flag in ['point_ok', 'ci95_ok', 'ci993_ok']:
            row['wstar_' + flag.replace('_ok', '')] = next((k for k in g.index if g.loc[k, flag]), None)
        out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------------------------
# data preparation for retraining
# ---------------------------------------------------------------------------------------------------------------
def merge_step9(W, dest_root):
    """Merge the per-semester step9 files of week W into dest_root/out_W{W}/step9_inputs.json (the layout load_week reads)."""
    parts = [json.load(open(os.path.join(C.IN, 'step9_D7', f'out_W{W}', f'{s}_step9_inputs.json'), encoding='utf-8')) for s in D7]
    m = {'week_limit': parts[0]['week_limit'], 'scatter_data': [], 'lsa_type_map': {}, 'transition_ratios': {},
         'incomplete_semesters': [], 'time_features': {}}
    for p in parts:
        assert p['week_limit'] == m['week_limit']
        m['scatter_data'] += p['scatter_data']
        for k in ['lsa_type_map', 'transition_ratios', 'time_features']:
            dup = set(m[k]) & set(p[k])
            assert not dup, (k, list(dup)[:3])
            m[k].update(p[k])
        m['incomplete_semesters'] = sorted(set(m['incomplete_semesters']) | set(p['incomplete_semesters']))
    d = os.path.join(dest_root, f'out_W{W}')
    os.makedirs(d, exist_ok=True)
    json.dump(m, open(os.path.join(d, 'step9_inputs.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    return d


def build_data(W, dest_root):
    """load_all-equivalent for D7 at week W, then OVERWRITE `final` with the OOF semester score and recompute is_fail."""
    import ews_experiment as E
    etl = L.Etl(ETL_DIR)
    L.init_bas(etl)  # sets the frozen BAS weights (ews_experiment.main does the same)
    merge_step9(W, dest_root)
    data = E.load_all(etl, dest_root, [W])
    frozen = C.frozen_frame(W)
    fin = {(r.semester, r.anon_id): float(r.final) for r in frozen.itertuples()}
    n = 0
    for s in D7:
        for r in data[W][s]:
            r['final'] = fin[(s, r['anon_id'])]
            r['is_fail'] = int(r['final'] < 60)
            n += 1
    return etl, data, n


def canonical(df):
    return df.sort_values(['semester', 'anon_id']).reset_index(drop=True)
