# -*- coding: utf-8 -*-
"""H-03: 2x2, lead time, capacity-constrained lists. Plan: plan/H03_plan.md. Run from the return root:
python -I H-03/h03_lead_capacity.py"""
import hashlib
import math
import os
import platform
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
import calendar_lib as C  # noqa: E402
import clusterboot as CB  # noqa: E402
import h_lib as H  # noqa: E402
import linkage as K  # noqa: E402

P2 = os.path.join(ROOT, 'materials', 'pack2')
OOF = os.path.join(ROOT, 'materials', 'pack1', '04_data', 'oof_scores_no_ids.csv')
OUT = os.path.join(ROOT, 'H-03')
SEMS_162 = {1121, 1122, 1131}
B1_REF = {'overlap_share': (0.573, 0.530, 0.617), 'model_only_fail_rate': (0.123, 0.083, 0.166), 'rr_model_only_vs_neither': (5.1, 3.2, 8.3)}
K_GRID = [10, 20, 30, 40, 60, 80]


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def top_pos(sem, score, frac=None, k=None, seed=7):
    """Positions of the lowest scores per semester (same tie rule as H.recall_precision_at); k may be a dict by semester."""
    rng = np.random.default_rng(seed)
    out = []
    for s in sorted(np.unique(sem)):
        pos = np.where(sem == s)[0]
        kk = int(math.ceil(frac * len(pos))) if frac is not None else int(k[s] if isinstance(k, dict) else k)
        kk = min(kk, len(pos))
        order = np.lexsort((rng.random(len(pos)), score[pos]))
        out.append(pos[order[:kk]])
    return np.concatenate(out)


def list_row(name, pos, y):
    tp = int(y[pos].sum())
    return {'list': name, 'n_flagged': int(len(pos)), 'tp': tp, 'recall': tp / y.sum(), 'precision': tp / len(pos),
            'lift': tp / len(pos) / y.mean()}


def main():
    log = open(os.path.join(OUT, 'run.log'), 'w', encoding='utf-8')
    print(f'python {platform.python_version()} numpy {np.__version__} pandas {pd.__version__}', file=log)
    ev_path = os.path.join(ROOT, 'H-01', 'H01_student_exam_events.csv')
    for p in (OOF, ev_path, os.path.join(ROOT, 'H-01', 'H01_calendar_by_semester_track.csv')):
        print(f'sha256 {os.path.relpath(p, ROOT)} {sha(p)}', file=log)
    print(f'bootstrap: student-cluster, B={CB.B_DEFAULT}, seed={CB.SEED_DEV}', file=log)

    oofs = K.load_oof_with_ids(OOF, P2)
    oof = oofs['W8']
    rec = K.records_df(K.load_step9(P2, 8))
    ev = pd.read_csv(ev_path, encoding='utf-8-sig', parse_dates=['midterm_main_ts', 'midterm_makeup_ts', 'final_main_ts', 'final_makeup_ts', 'effective_start'])
    assert (ev['anon_id'].to_numpy() == oof['anon_id'].to_numpy()).all() and (ev['semester'].to_numpy() == oof['semester'].to_numpy()).all()
    y = oof['y'].to_numpy(int)
    sem = oof['semester'].to_numpy(int)
    m3 = oof['m3'].to_numpy(float)
    flag = (rec['midterm'].to_numpy(float) < 60)
    track = ev['class_track'].to_numpy()
    n = len(y)

    in_list = np.zeros(n, bool)
    in_list[top_pos(sem, m3, frac=0.2)] = True

    # --- 2x2
    rows = []
    for ml in (True, False):
        for fl in (True, False):
            m = (in_list == ml) & (flag == fl)
            rows.append({'row_type': 'cell', 'model_list': 'yes' if ml else 'no', 'redyellow': 'yes' if fl else 'no',
                         'n': int(m.sum()), 'n_fail': int(y[m].sum()), 'fail_rate': float(y[m].mean()) if m.sum() else np.nan})
    cb = CB.ClusterBoot(oof['anon_id'].to_numpy())

    def stats(ix):
        il, fl, yy = in_list[ix], flag[ix], y[ix]
        lst = il.sum()
        both = (il & fl).sum()
        mo = il & ~fl
        nei = ~il & ~fl
        mo_fail = yy[mo].mean() if mo.sum() else np.nan
        ne_fail = yy[nei].mean() if nei.sum() else np.nan
        return np.array([both / lst, mo.sum(), mo_fail, ne_fail, mo_fail / ne_fail if ne_fail else np.nan, yy.mean()])

    pt = stats(np.arange(n))
    bt = cb.apply(stats)
    names = ['overlap_share', 'model_only_n', 'model_only_fail_rate', 'neither_fail_rate', 'rr_model_only_vs_neither', 'overall_fail_rate']
    for i, nm in enumerate(names):
        lo, hi = H.percentile_ci(bt[:, i])
        ref = B1_REF.get(nm, (np.nan, np.nan, np.nan))
        rows.append({'row_type': 'stat', 'stat': nm, 'value': pt[i], 'lo': lo, 'hi': hi,
                     'manuscript_B1_value': ref[0], 'manuscript_B1_lo': ref[1], 'manuscript_B1_hi': ref[2]})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'H03_2x2.csv'), index=False, encoding='utf-8-sig')

    # --- lead time
    mid = ev['midterm_main_ts'].where(ev['midterm_main_ts'].notna(), ev['midterm_makeup_ts'])
    src = np.where(ev['midterm_main_ts'].notna(), 'main', np.where(ev['midterm_makeup_ts'].notna(), 'makeup', 'none'))
    fin = ev['final_main_ts'].where(ev['final_main_ts'].notna(), ev['final_makeup_ts'])
    cut8 = pd.Series([C.cutoff(s, 8) for s in ev['effective_start']])
    lead_lo = (mid.dt.normalize() + pd.Timedelta(days=1) - cut8).dt.days.to_numpy(float)
    lead_hi = lead_lo + 1
    wk_final_model = (fin.dt.normalize() - cut8).dt.days.to_numpy(float) / 7
    structure = np.where(track == 'retake', 'retake', np.where(np.isin(sem, list(SEMS_162)), '16+2', 'regular'))
    both_fail = in_list & flag & (y == 1) & (src != 'none')
    stu = pd.DataFrame({'semester': sem, 'anon_id': oof['anon_id'].to_numpy(), 'class_track': track, 'structure': structure,
                        'midterm_source': src, 'lead_days_low': lead_lo, 'lead_days_high': lead_hi,
                        'lead_weeks_low': lead_lo / 7, 'lead_weeks_high': lead_hi / 7, 'weeks_to_final_from_model_flag': wk_final_model})[both_fail]
    stu.to_csv(os.path.join(OUT, 'H03_lead_time_students.csv'), index=False, encoding='utf-8-sig')

    def q(a, p):
        return float(np.nanpercentile(a, p)) if len(a) else np.nan

    lrows = []
    for grp in ('all', 'regular', '16+2', 'retake'):
        selm = both_fail if grp == 'all' else both_fail & (structure == grp)
        a, b = lead_lo[selm], lead_hi[selm]

        def med(ix, a=a):
            s = selm[ix]
            return np.nanmedian(lead_lo[ix][s]) if s.sum() else np.nan
        bm = cb.apply(med) if selm.sum() else np.array([np.nan])
        lo, hi = H.percentile_ci(bm)
        lrows.append({'group': grp, 'n_fail_model_and_redyellow': int(selm.sum()),
                      'low_days_median': q(a, 50), 'low_days_q1': q(a, 25), 'low_days_q3': q(a, 75),
                      'high_days_median': q(b, 50), 'high_days_q1': q(b, 25), 'high_days_q3': q(b, 75),
                      'low_weeks_median': q(a, 50) / 7, 'high_weeks_median': q(b, 50) / 7,
                      'low_days_median_ci_lo': lo, 'low_days_median_ci_hi': hi,
                      'weeks_to_final_from_model_flag_median': q(wk_final_model[selm], 50)})
    fail_in_list = in_list & (y == 1)
    cnt = {'n_fail_in_list': int(fail_in_list.sum()), 'n_fail_in_list_and_redyellow': int((fail_in_list & flag).sum()),
           'n_fail_in_list_not_redyellow(model_only)': int((fail_in_list & ~flag).sum()),
           'n_fail_redyellow_not_in_list': int((flag & ~in_list & (y == 1)).sum()),
           'n_fail_in_list_and_redyellow_without_midterm_event': int((fail_in_list & flag & (src == 'none')).sum()),
           'n_fail_in_list_and_redyellow_makeup_date_used': int((fail_in_list & flag & (src == 'makeup')).sum())}
    for k_, v in cnt.items():
        lrows.append({'group': 'counts', 'stat': k_, 'n_fail_model_and_redyellow': v})
    pd.DataFrame(lrows).to_csv(os.path.join(OUT, 'H03_lead_time.csv'), index=False, encoding='utf-8-sig')

    # --- capacity, guaranteed layer
    crows, kmin_by = [], {15: {}, 30: {}}
    for s in sorted(np.unique(sem)):
        mm = sem == s
        r_s = float(np.nanmedian(np.floor(wk_final_model[mm])))
        for c in (15, 30):
            kmin = int(60 * r_s // c)
            kmin_by[c][s] = kmin
            pos = top_pos(sem[mm], m3[mm], k=kmin)
            tp = int(y[mm][pos].sum())
            crows.append({'semester': int(s), 'contact_minutes': c, 'R_weeks': r_s, 'k_min': kmin,
                          'list_size_20pct': int(math.ceil(0.2 * mm.sum())), 'k_min_share_of_list': kmin / math.ceil(0.2 * mm.sum()),
                          'n_fail_semester': int(y[mm].sum()), 'tp_at_k_min': tp, 'recall_at_k_min': tp / y[mm].sum(),
                          'precision_at_k_min': tp / len(pos)})
    cap = pd.DataFrame(crows)
    cap.to_csv(os.path.join(OUT, 'H03_capacity_guaranteed_layer.csv'), index=False, encoding='utf-8-sig')

    # --- Recall@k and percentile lists
    rk, lists = [], {}
    for k_ in K_GRID:
        lists[f'k={k_}'] = top_pos(sem, m3, k=k_)
    for p_ in (10, 20, 30, 40):
        lists[f'pct={p_}'] = top_pos(sem, m3, frac=p_ / 100)
    for c in (15, 30):
        lists[f'k_min(c={c})'] = top_pos(sem, m3, k=kmin_by[c])
    for nm, pos in lists.items():
        rk.append(list_row(nm, pos, y))
    pd.DataFrame(rk).to_csv(os.path.join(OUT, 'H03_recall_at_k.csv'), index=False, encoding='utf-8-sig')

    # --- marginal cost
    mrows = []
    seq = [f'k={k_}' for k_ in K_GRID]
    seq2 = ['pct=10', 'pct=20', 'pct=30', 'pct=40']
    for chain in (seq, seq2):
        for a_, b_ in zip(chain[:-1], chain[1:]):
            da = list_row(a_, lists[a_], y)
            db = list_row(b_, lists[b_], y)
            dn, dt = db['n_flagged'] - da['n_flagged'], db['tp'] - da['tp']
            mrows.append({'from_list': a_, 'to_list': b_, 'extra_flagged': dn, 'extra_failing_found': dt,
                          'flagged_per_extra_failing': dn / dt if dt else np.nan})
    r20 = list_row('pct=20', lists['pct=20'], y)
    mrows.append({'from_list': 'none', 'to_list': 'pct=20 (average)', 'extra_flagged': r20['n_flagged'], 'extra_failing_found': r20['tp'],
                  'flagged_per_extra_failing': r20['n_flagged'] / r20['tp']})
    pd.DataFrame(mrows).to_csv(os.path.join(OUT, 'H03_marginal_cost.csv'), index=False, encoding='utf-8-sig')

    # --- remediation-window sensitivity
    avail = [6, 8, 10, 12, 14, 16]
    srows = []
    for struct, sems_s, fw in (('regular', {1111, 1112, 1132, 1141}, 18), ('16+2', SEMS_162, 16)):
        mask = np.isin(sem, list(sems_s)) & (track != 'retake')
        for r in (2, 4, 6):
            wf = max(w for w in avail if w <= fw - r)
            for tag, W in (('feasible_week', wf), ('W8_reference', 8)):
                d = oofs[f'W{W}']
                s_all = d['m3'].to_numpy(float)

                def auc_sub(ix, s_all=s_all):
                    ixs = ix[mask[ix]]
                    return H.auc_fail(y[ixs], s_all[ixs])
                bt2 = cb.apply(auc_sub)
                sub = pd.DataFrame({'semester': sem[mask], 'y': y[mask], 'm3': s_all[mask]})
                rec20, prec20 = H.recall_precision_at(sub, 'm3', 0.2)
                lo, hi = H.percentile_ci(bt2)
                srows.append({'structure': struct, 'remaining_weeks_required': r, 'row': tag, 'final_week': fw, 'warning_week': W,
                              'n': int(mask.sum()), 'auc_pooled': H.auc_fail(y[mask], s_all[mask]), 'auc_lo': lo, 'auc_hi': hi,
                              'recall20': rec20, 'precision20': prec20})
    pd.DataFrame(srows).to_csv(os.path.join(OUT, 'H03_remediation_sensitivity.csv'), index=False, encoding='utf-8-sig')

    # --- markdown tables from output files
    def f3(x):
        return 'NA' if pd.isna(x) else f'{x:.3f}'.replace('-', '\u2212')
    d2 = pd.read_csv(os.path.join(OUT, 'H03_2x2.csv'), encoding='utf-8-sig')
    L = ['### 2x2: Week-8 model list x red/yellow flag x outcome', '', '| Model list | Red/yellow | n | Failing | Fail rate |', '|---|---|---|---|---|']
    for _, r in d2[d2['row_type'] == 'cell'].iterrows():
        L.append(f"| {r['model_list']} | {r['redyellow']} | {int(r['n'])} | {int(r['n_fail'])} | {f3(r['fail_rate'])} |")
    L += ['', '| Statistic | Value [95% CI] | Manuscript B1 value [CI] |', '|---|---|---|']
    for _, r in d2[d2['row_type'] == 'stat'].iterrows():
        ref = '' if pd.isna(r['manuscript_B1_value']) else f"{r['manuscript_B1_value']} [{r['manuscript_B1_lo']}\u2013{r['manuscript_B1_hi']}]"
        L.append(f"| {r['stat']} | {f3(r['value'])} [{f3(r['lo'])}; {f3(r['hi'])}] | {ref} |")
    ld = pd.read_csv(os.path.join(OUT, 'H03_lead_time.csv'), encoding='utf-8-sig')
    L += ['', '### Lead time of the Week-8 warning before the red/yellow flag (failing students in both lists; days; positive = warning earlier)', '',
          '| Group | n | Low end median [Q1; Q3] | High end median [Q1; Q3] | Median in weeks (low; high) |', '|---|---|---|---|---|']
    for _, r in ld[ld['group'] != 'counts'].iterrows():
        L.append(f"| {r['group']} | {int(r['n_fail_model_and_redyellow'])} | {r['low_days_median']:.1f} [{r['low_days_q1']:.1f}; {r['low_days_q3']:.1f}] | "
                 f"{r['high_days_median']:.1f} [{r['high_days_q1']:.1f}; {r['high_days_q3']:.1f}] | {r['low_weeks_median']:.2f}; {r['high_weeks_median']:.2f} |")
    L += ['']
    for _, r in ld[ld['group'] == 'counts'].iterrows():
        L.append(f"- {r['stat']}: {int(r['n_fail_model_and_redyellow'])}")
    L += ['', '### Guaranteed-minimum layer (1 scheduled hour per week, one-to-one contacts of c minutes; assumption, not data)', '',
          '| Semester | c (min) | R (weeks) | k_min | 20% list | k_min / list | Recall@k_min | Precision@k_min |', '|---|---|---|---|---|---|---|---|']
    for _, r in cap.iterrows():
        L.append(f"| {int(r['semester'])} | {int(r['contact_minutes'])} | {r['R_weeks']:.0f} | {int(r['k_min'])} | {int(r['list_size_20pct'])} | {f3(r['k_min_share_of_list'])} | {f3(r['recall_at_k_min'])} | {f3(r['precision_at_k_min'])} |")
    rkd = pd.read_csv(os.path.join(OUT, 'H03_recall_at_k.csv'), encoding='utf-8-sig')
    L += ['', '### Recall@k and percentile lists (pooled over semesters)', '', '| List | Flagged | Failing found | Recall | Precision | Lift |', '|---|---|---|---|---|---|']
    for _, r in rkd.iterrows():
        L.append(f"| {r['list']} | {r['n_flagged']} | {r['tp']} | {f3(r['recall'])} | {f3(r['precision'])} | {r['lift']:.2f} |")
    md = pd.read_csv(os.path.join(OUT, 'H03_marginal_cost.csv'), encoding='utf-8-sig')
    L += ['', '### Extra students flagged per additional failing student found', '', '| From | To | Extra flagged | Extra failing found | Flagged per extra failing |', '|---|---|---|---|---|']
    for _, r in md.iterrows():
        L.append(f"| {r['from_list']} | {r['to_list']} | {r['extra_flagged']} | {r['extra_failing_found']} | {r['flagged_per_extra_failing']:.2f} |")
    rs = pd.read_csv(os.path.join(OUT, 'H03_remediation_sensitivity.csv'), encoding='utf-8-sig')
    L += ['', '### Remediation-window sensitivity (latest feasible warning week)', '', '| Structure | Weeks needed | Row | Warning week | n | Pooled AUC [CI] | Recall@20% |', '|---|---|---|---|---|---|---|']
    for _, r in rs.iterrows():
        L.append(f"| {r['structure']} | {r['remaining_weeks_required']} | {r['row']} | W{r['warning_week']} | {r['n']} | {f3(r['auc_pooled'])} [{f3(r['auc_lo'])}\u2013{f3(r['auc_hi'])}] | {f3(r['recall20'])} |")
    open(os.path.join(OUT, 'H03_summary_tables.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    log.close()
    print('done')


if __name__ == '__main__':
    main()
