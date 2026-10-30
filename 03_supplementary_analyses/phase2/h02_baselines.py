# -*- coding: utf-8 -*-
"""H-02: baselines B3, B4, B5 versus M3' at Week 8 (B1/B2 not run, see plan). Run from the return root:
python -I H-02/h02_baselines.py"""
import hashlib
import math
import os
import platform
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
import clusterboot as CB  # noqa: E402
import flags as F  # noqa: E402
import h_lib as H  # noqa: E402
import linkage as K  # noqa: E402

P2 = os.path.join(ROOT, 'materials', 'pack2')
OOF = os.path.join(ROOT, 'materials', 'pack1', '04_data', 'oof_scores_no_ids.csv')
OUT = os.path.join(ROOT, 'H-02')
PRED = ["M3'", 'B3 quiz pass rate', 'B4 midterm score', 'B5 traffic light']
BASE = PRED[1:]
CLASSES = ['2yr_gen', '2yr_working', 'post', '4yr', 'retake']


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def wmean_auc(y, s, sem):
    a, w = [], []
    for v in np.unique(sem):
        m = sem == v
        if y[m].sum() in (0, m.sum()):
            continue
        a.append(H.auc_fail(y[m], s[m]))
        w.append(m.sum())
    return float(np.average(a, weights=w)) if a else np.nan


def group_auc(y, s, g, levels):
    out = []
    for v in levels:
        m = g == v
        out.append(H.auc_fail(y[m], s[m]) if (m.sum() and 0 < y[m].sum() < m.sum()) else np.nan)
    return np.array(out)


def top_lists(df, col, frac=None, k_by_sem=None, seed=7):
    """Indices (positions) of the lowest-score records per semester; same tie rule as H.recall_precision_at."""
    rng = np.random.default_rng(seed)
    chosen = []
    for sem, g in df.groupby('semester', sort=True):
        pos = np.where(df['semester'].to_numpy() == sem)[0]
        s = df[col].to_numpy(float)[pos]
        k = int(math.ceil(frac * len(pos))) if frac is not None else int(k_by_sem[sem])
        order = np.lexsort((rng.random(len(pos)), s))
        chosen.append(pos[order[:k]])
    return np.concatenate(chosen)


def main():
    log = open(os.path.join(OUT, 'run.log'), 'w', encoding='utf-8')
    print(f'python {platform.python_version()} numpy {np.__version__} pandas {pd.__version__}', file=log)
    for p in (OOF, os.path.join(P2, '01_step9_inputs', 'inputs_D7', 'out_W8', '1141_step9_inputs.json')):
        print(f'sha256 {os.path.relpath(p, ROOT)} {sha(p)}', file=log)
    print(f'bootstrap: student-cluster (ews_lib.Boot draws), B={CB.B_DEFAULT}, seed={CB.SEED_DEV}', file=log)

    oof = K.load_oof_with_ids(OOF, P2)['W8']
    rec = K.records_df(K.load_step9(P2, 8))
    assert (oof['anon_id'].to_numpy() == rec['anon_id'].to_numpy()).all()
    c5 = rec[['semester', 'anon_id']].merge(K.load_class5(P2), on=['semester', 'anon_id'], how='left')
    cls = c5['program5_key'].to_numpy()
    y = oof['y'].to_numpy(int)
    sem = oof['semester'].to_numpy(int)
    qpr = pd.to_numeric(rec['quz_pass_rate'], errors='coerce')
    n_missing_b3 = int(qpr.isna().sum())
    print(f'B3 missing quz_pass_rate set to 0.0: {n_missing_b3}', file=log)
    flag = rec['midterm'].map(F.traffic_light)
    score = {"M3'": oof['m3'].to_numpy(float), 'B3 quiz pass rate': qpr.fillna(0.0).to_numpy(float),
             'B4 midterm score': rec['midterm'].to_numpy(float),
             'B5 traffic light': (flag == 'green').astype(float).to_numpy()}
    base_rate = float(y.mean())
    frame = pd.DataFrame({'semester': sem, 'y': y, **{k: v for k, v in score.items()}})
    sems = sorted(np.unique(sem))
    not_retake = cls != 'retake'

    cb = CB.ClusterBoot(oof['anon_id'].to_numpy())
    # bootstrap arrays per predictor: pooled, weighted-within, class AUCs, same for excl-retake
    boot = {p: {'pooled': [], 'wmean': [], 'cls': [], 'pooled_x': [], 'wmean_x': [], 'sem': []} for p in PRED}
    for ix in cb.idx:
        yb, sb, cbv = y[ix], sem[ix], cls[ix]
        keep = cbv != 'retake'
        for p in PRED:
            s = score[p][ix]
            d = boot[p]
            d['pooled'].append(H.auc_fail(yb, s))
            d['wmean'].append(wmean_auc(yb, s, sb))
            d['cls'].append(group_auc(yb, s, cbv, CLASSES))
            d['pooled_x'].append(H.auc_fail(yb[keep], s[keep]))
            d['wmean_x'].append(wmean_auc(yb[keep], s[keep], sb[keep]))
            d['sem'].append(group_auc(yb, s, sb, sems))
    for p in PRED:
        for k in boot[p]:
            boot[p][k] = np.array(boot[p][k], float)

    def pt(p, subset):
        m = np.ones(len(y), bool) if subset == 'all' else not_retake
        return (H.auc_fail(y[m], score[p][m]), wmean_auc(y[m], score[p][m], sem[m]))

    rows = []
    for subset in ('all', 'excl_retake'):
        m = np.ones(len(y), bool) if subset == 'all' else not_retake
        for p in PRED:
            a, w = pt(p, subset)
            kp, kw = ('pooled', 'wmean') if subset == 'all' else ('pooled_x', 'wmean_x')
            sub = frame[m].reset_index(drop=True)
            rec20, prec20 = H.recall_precision_at(sub, p, 0.2)
            row = {'predictor': p, 'subset': subset, 'n': int(m.sum()), 'n_fail': int(y[m].sum()),
                   'auc_pooled': a, 'auc_pooled_lo': H.percentile_ci(boot[p][kp])[0], 'auc_pooled_hi': H.percentile_ci(boot[p][kp])[1],
                   'auc_wmean': w, 'auc_wmean_lo': H.percentile_ci(boot[p][kw])[0], 'auc_wmean_hi': H.percentile_ci(boot[p][kw])[1],
                   'recall20': rec20, 'precision20': prec20, 'lift20': prec20 / float(y[m].mean()),
                   'natural_list_n': np.nan, 'natural_recall': np.nan, 'natural_precision': np.nan, 'natural_lift': np.nan,
                   'matched_m3_recall': np.nan, 'matched_m3_precision': np.nan, 'status': 'run'}
            if p == 'B5 traffic light':
                fl = (score[p][m] == 0)
                tp = int((y[m][fl]).sum())
                row.update({'natural_list_n': int(fl.sum()), 'natural_recall': tp / y[m].sum(),
                            'natural_precision': tp / fl.sum(), 'natural_lift': tp / fl.sum() / float(y[m].mean())})
                ksem = {s_: int(((sem[m] == s_) & fl).sum()) for s_ in sems}
                pos = top_lists(sub, "M3'", k_by_sem=ksem)
                tpm = int(sub['y'].to_numpy()[pos].sum())
                row.update({'matched_m3_recall': tpm / y[m].sum(), 'matched_m3_precision': tpm / len(pos)})
            rows.append(row)
    for p in ('B1 class only', 'B2 class + prior failure rate'):
        rows.append({'predictor': p, 'subset': 'all', 'status': 'NOT RUN - README_pack2 section 2.3 truncated in the shipped file'})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(OUT, 'H02_baseline_metrics.csv'), index=False, encoding='utf-8-sig')

    # delta AUC
    drows = []
    for b in BASE:
        for subset, kp, kw in (('all', 'pooled', 'wmean'), ('excl_retake', 'pooled_x', 'wmean_x')):
            for metric, key, i in (('pooled', kp, 0), ('weighted_within_semester', kw, 1)):
                d = boot["M3'"][key] - boot[b][key]
                a = pt("M3'", subset)[i] - pt(b, subset)[i]
                lo, hi = H.percentile_ci(d)
                drows.append({'baseline': b, 'subset': subset, 'metric': metric, 'delta_auc': a, 'lo': lo, 'hi': hi})
        dcls = boot["M3'"]['cls'] - boot[b]['cls']
        for j, c in enumerate(CLASSES):
            m = cls == c
            a = H.auc_fail(y[m], score["M3'"][m]) - H.auc_fail(y[m], score[b][m]) if 0 < y[m].sum() < m.sum() else np.nan
            lo, hi = H.percentile_ci(dcls[:, j])
            drows.append({'baseline': b, 'subset': 'all', 'metric': f'class:{c}', 'delta_auc': a, 'lo': lo, 'hi': hi})
    dd = pd.DataFrame(drows)
    dd['ci_excludes_0'] = (dd['lo'] > 0) | (dd['hi'] < 0)
    dd.to_csv(os.path.join(OUT, 'H02_delta_auc.csv'), index=False, encoding='utf-8-sig')

    # AUC by class and by semester
    crow, srow = [], []
    for p in PRED:
        for j, c in enumerate(CLASSES):
            m = cls == c
            lo, hi = H.percentile_ci(boot[p]['cls'][:, j])
            crow.append({'predictor': p, 'class5': c, 'n': int(m.sum()), 'n_fail': int(y[m].sum()),
                         'auc': H.auc_fail(y[m], score[p][m]), 'lo': lo, 'hi': hi})
        for j, s_ in enumerate(sems):
            m = sem == s_
            lo, hi = H.percentile_ci(boot[p]['sem'][:, j])
            srow.append({'predictor': p, 'semester': s_, 'n': int(m.sum()), 'auc': H.auc_fail(y[m], score[p][m]), 'lo': lo, 'hi': hi})
    cdf = pd.DataFrame(crow)
    cdf.to_csv(os.path.join(OUT, 'H02_auc_by_class.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(srow).to_csv(os.path.join(OUT, 'H02_auc_by_semester.csv'), index=False, encoding='utf-8-sig')

    # list composition
    lrows = []
    pop = pd.Series(cls).value_counts()
    lists = {}
    for p in ("M3'", 'B3 quiz pass rate', 'B4 midterm score'):
        lists[p] = top_lists(frame, p, frac=0.2)
    lists['B5 traffic light (flagged)'] = np.where(score['B5 traffic light'] == 0)[0]
    for name, pos in lists.items():
        sub_cls = pd.Series(cls[pos])
        for c in CLASSES:
            m = sub_cls == c
            lrows.append({'list': name, 'class5': c, 'n_in_list': int(m.sum()), 'share_of_list': float(m.mean()),
                          'share_of_population': float(pop.get(c, 0) / len(cls)),
                          'fail_rate_in_list_class': float(y[pos][m.to_numpy()].mean()) if m.sum() else np.nan})
    ldf = pd.DataFrame(lrows)
    ldf.to_csv(os.path.join(OUT, 'H02_list_composition.csv'), index=False, encoding='utf-8-sig')

    # forest plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = ['#1f4e79', '#c55a11', '#548235', '#7f7f7f']
    k = 0
    ticks, labels = [], []
    for c in CLASSES:
        for p, col in zip(PRED, colors):
            r = cdf[(cdf['predictor'] == p) & (cdf['class5'] == c)].iloc[0]
            ax.errorbar(r['auc'], k, xerr=[[r['auc'] - r['lo']], [r['hi'] - r['auc']]], fmt='o', color=col, capsize=2,
                        label=p if c == CLASSES[0] else None)
            k += 1
        ticks.append(k - 2.5)
        labels.append(f"{c} (n={int(cdf[cdf['class5'] == c]['n'].iloc[0])})")
        k += 1
    ax.axvline(0.5, color='k', lw=0.8, ls='--')
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('AUC (95% student-cluster bootstrap CI)')
    ax.set_title('Week 8: AUC within program class')
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'H02_forest_auc_by_class.png'), dpi=150)
    plt.close(fig)

    # markdown tables from output files
    def f3(x):
        return 'NA' if pd.isna(x) else f'{x:.3f}'.replace('-', '\u2212')
    L = ['### Baselines at Week 8 (all 2,652 records)', '',
         '| Predictor | Pooled AUC [CI] | Weighted within-semester AUC [CI] | Recall@20% | Precision@20% | Lift@20% |', '|---|---|---|---|---|---|']
    for _, r in metrics[(metrics['subset'] == 'all') & (metrics['status'] == 'run')].iterrows():
        L.append(f"| {r['predictor']} | {f3(r['auc_pooled'])} [{f3(r['auc_pooled_lo'])}\u2013{f3(r['auc_pooled_hi'])}] | "
                 f"{f3(r['auc_wmean'])} [{f3(r['auc_wmean_lo'])}\u2013{f3(r['auc_wmean_hi'])}] | {f3(r['recall20'])} | {f3(r['precision20'])} | {r['lift20']:.2f} |")
    b5 = metrics[(metrics['predictor'] == 'B5 traffic light') & (metrics['subset'] == 'all')].iloc[0]
    L += ['', f"B5 flagged list (natural size {int(b5['natural_list_n'])}): recall {f3(b5['natural_recall'])}, precision {f3(b5['natural_precision'])}, lift {b5['natural_lift']:.2f}; "
          f"M3' list of the same size per semester: recall {f3(b5['matched_m3_recall'])}, precision {f3(b5['matched_m3_precision'])}.",
          '', '### Delta AUC (M3\u2032 minus baseline), 95% student-cluster CI', '',
          '| Baseline | Subset | Metric | Delta [CI] | CI excludes 0 |', '|---|---|---|---|---|']
    for _, r in dd.iterrows():
        L.append(f"| {r['baseline']} | {r['subset']} | {r['metric']} | {f3(r['delta_auc'])} [{f3(r['lo'])}; {f3(r['hi'])}] | {'yes' if r['ci_excludes_0'] else 'no'} |")
    L += ['', '### AUC within program class', '', '| Predictor | Class | n | Failing | AUC [CI] |', '|---|---|---|---|---|']
    for _, r in cdf.iterrows():
        L.append(f"| {r['predictor']} | {r['class5']} | {r['n']} | {r['n_fail']} | {f3(r['auc'])} [{f3(r['lo'])}\u2013{f3(r['hi'])}] |")
    L += ['', '### List composition by class (share of list vs share of population)', '',
          '| List | Class | In list | Share of list | Share of population | Fail rate in list |', '|---|---|---|---|---|---|']
    for _, r in ldf.iterrows():
        L.append(f"| {r['list']} | {r['class5']} | {r['n_in_list']} | {f3(r['share_of_list'])} | {f3(r['share_of_population'])} | {f3(r['fail_rate_in_list_class'])} |")
    open(os.path.join(OUT, 'H02_summary_tables.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    log.close()
    print('done')


if __name__ == '__main__':
    main()
