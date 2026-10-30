# -*- coding: utf-8 -*-
"""H-15 (Phase 1 part): Table 1 audit (H-15-1) and holdout 40% recall arithmetic (H-15-3).

Run from the return root:  python -I H-15/h15_table1_audit.py
Plan: plan/supp_analysis_plan.md section 2.
"""
import hashlib
import importlib.util
import math
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
import h_lib as H  # noqa: E402

PACK = os.path.join(ROOT, 'materials', 'PaperA_supp_analyses_newchat_pack_20261026')
OOF = os.path.join(PACK, '04_data', 'oof_scores_no_ids.csv')
XLSX = os.path.join(PACK, '04_data', 'Table_A_data_era.xlsx')
TABLES_PY = os.path.join(PACK, '02_manuscript_current', 'tables_data_20261019.py')
MANUSCRIPT = os.path.join(PACK, '02_manuscript_current', 'PaperA_manuscript_NET_Draft_v2_20261019.md')
HOLDOUT = os.path.join(PACK, '03_specs', 'PaperA_holdout_REPORT.md')
OUT = os.path.join(ROOT, 'H-15')

WEEK = {'Week 6': 'W6', 'Week 8': 'W8', 'Week 10': 'W10', 'Week 12': 'W12', 'Week 14': 'W14',
        'Week 16': 'W16', 'Full semester': 'Full'}
PRED = {'BAS': ("M1'", 'm1'), 'XGBoost': ("M2'", 'm2'), 'BAS + XGBoost (blend)': ("M3'", 'm3')}
PRED_SHORT = {'BAS': 'BAS', 'XGBoost': 'XGBoost', 'BAS + XGBoost (blend)': 'blend'}
TOL = {'rpp': 0.05, 'other': 0.0005}


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def load_table1():
    spec = importlib.util.spec_from_file_location('tables_data', TABLES_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    t1 = [t for t in mod.TABLES if t['number'] == 'Table 1'][0]
    rows, week = [], None
    for r in t1['rows']:
        week = r[0] or week
        rows.append((week, r[1], r))
    return rows


def split_ci(cell):
    m = re.match(r'^\s*([\d.]+)\s*\(([\d.]+)\u2013([\d.]+)\)\s*$', cell)
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def main():
    log = open(os.path.join(OUT, 'run.log'), 'w', encoding='utf-8')
    H.log_versions(log)
    for p in (OOF, XLSX, TABLES_PY, MANUSCRIPT, HOLDOUT):
        print(f'sha256 {os.path.relpath(p, ROOT)} {sha(p)}', file=log)
    print('bootstrap: none (point-estimate audit; CI cells compared with frozen student-cluster values)', file=log)

    oof = H.load_oof(OOF)
    frozen = pd.ExcelFile(XLSX).parse('era_metric')
    frozen = frozen[frozen['era'] == 'ALL'].set_index(['W', 'model'])

    # recomputed metrics per (W, model)
    calc = {}
    for W in H.W_ALL:
        df = oof[W]
        for name, (mname, col) in PRED.items():
            y = df['y'].to_numpy(int)
            s = df[col].to_numpy(float)
            rec, prec = H.recall_precision_at(df, col, 0.2)
            calc[(W, mname)] = {'r': H.pearson(s, df['final']), 'auc': H.auc_fail(y, s),
                                'recall20': rec, 'precision20': prec}
    for mname in ("M1'", "M2'", "M3'"):
        r16, a16 = calc[('W16', mname)]['r'], calc[('W16', mname)]['auc']
        for W in H.W_ALL:
            c = calc[(W, mname)]
            c['rpp_r'] = 100 * c['r'] / r16
            c['rpp_auc'] = 100 * (c['auc'] - 0.5) / (a16 - 0.5)

    cells = []
    for week_label, pred_label, row in load_table1():
        W = WEEK[week_label]
        mname, _col = PRED[pred_label]
        c = calc[(W, mname)]
        fz = frozen.loc[(W, mname)]
        r_pt, r_lo, r_hi = split_ci(row[2])
        a_pt, a_lo, a_hi = split_ci(row[3])
        printed = {'r': (r_pt, c['r'], 'recomputed from OOF'), 'auc': (a_pt, c['auc'], 'recomputed from OOF'),
                   'recall20': (float(row[4]), c['recall20'], 'recomputed from OOF'),
                   'precision20': (float(row[5]), c['precision20'], 'recomputed from OOF'),
                   'rpp_r': (float(row[6]), c['rpp_r'], 'recomputed from OOF'),
                   'rpp_auc': (float(row[7]), c['rpp_auc'], 'recomputed from OOF'),
                   'r_ci_lo': (r_lo, float(fz['r_lo']), 'frozen era_metric ALL (student-cluster)'),
                   'r_ci_hi': (r_hi, float(fz['r_hi']), 'frozen era_metric ALL (student-cluster)'),
                   'auc_ci_lo': (a_lo, float(fz['auc_lo']), 'frozen era_metric ALL (student-cluster)'),
                   'auc_ci_hi': (a_hi, float(fz['auc_hi']), 'frozen era_metric ALL (student-cluster)')}
        for metric, (pv, ref, src) in printed.items():
            tol = (TOL['rpp'] if metric.startswith('rpp') else TOL['other']) + 1e-9
            diff = abs(pv - ref)
            cells.append({'week': week_label, 'predictor': PRED_SHORT[pred_label], 'metric': metric,
                          'printed': pv, 'reference_value': ref, 'reference_source': src,
                          'abs_diff': diff, 'match': bool(diff <= tol)})
    cells = pd.DataFrame(cells)
    cells.to_csv(os.path.join(OUT, 'H15_1_table1_cell_check.csv'), index=False, encoding='utf-8-sig')
    mism = cells[~cells['match']]
    mism.to_csv(os.path.join(OUT, 'H15_1_mismatches.csv'), index=False, encoding='utf-8-sig')

    # Week 16 detail at full precision, with true-positive counts
    det = []
    d16 = oof['W16']
    for pred_label, (mname, col) in PRED.items():
        c = calc[('W16', mname)]
        rng = np.random.default_rng(7)
        tp = flagged = 0
        for _s, g in d16.groupby('semester'):
            k = int(math.ceil(0.2 * len(g)))
            order = np.lexsort((rng.random(len(g)), g[col].to_numpy(float)))
            tp += int(g['y'].to_numpy(int)[order[:k]].sum())
            flagged += k
        det.append({'predictor': PRED_SHORT[pred_label], 'auc': c['auc'], 'r': c['r'], 'recall20': c['recall20'],
                    'precision20': c['precision20'], 'tp_top20': tp, 'flagged_top20': flagged,
                    'failing_total': int(d16['y'].sum())})
    det = pd.DataFrame(det)
    det.to_csv(os.path.join(OUT, 'H15_1_week16_detail.csv'), index=False, encoding='utf-8-sig')

    # manuscript sentences carrying mismatched printed numbers
    text = open(MANUSCRIPT, encoding='utf-8').read().splitlines()
    hits = []
    for _, m in mism.iterrows():
        token = f"{m['printed']:.3f}" if not m['metric'].startswith('rpp') else f"{m['printed']:.1f}"
        for i, line in enumerate(text, 1):
            if token in line:
                hits.append({'week': m['week'], 'predictor': m['predictor'], 'metric': m['metric'],
                             'printed_token': token, 'manuscript_line': i, 'snippet': line.strip()[:200]})
    pd.DataFrame(hits, columns=['week', 'predictor', 'metric', 'printed_token', 'manuscript_line', 'snippet']
                 ).to_csv(os.path.join(OUT, 'H15_1_manuscript_hits.csv'), index=False, encoding='utf-8-sig')

    # H-15-3 holdout arithmetic, parsed from the holdout report
    rep = open(HOLDOUT, encoding='utf-8').read()
    n_hold = int(re.search(r'n=(\d+)（成績可對接', rep).group(1))
    n_fail = int(re.search(r'不及格 (\d+) 人', rep).group(1))
    rows = []
    for q in (10, 20, 30, 40):
        m = re.search(rf'recall@{q}% / precision@{q}% \| ([\d.]+) / ([\d.]+) \|', rep)
        rec_p, prec_p = float(m.group(1)), float(m.group(2))
        list_n = int(math.ceil(q / 100 * n_hold))
        hits_n = int(round(prec_p * list_n))
        exact = hits_n / n_fail
        rows.append({'q': q / 100, 'n': n_hold, 'failing': n_fail, 'list_n': list_n,
                     'precision_printed': prec_p, 'hits': hits_n, 'hits_recomputes_precision':
                     bool(abs(hits_n / list_n - prec_p) < 0.0005 + 1e-9),
                     'recall_printed': rec_p, 'recall_exact': exact,
                     'recall_round3_half_up': math.floor(exact * 1000 + 0.5) / 1000,
                     'recall_printed_matches_round3': bool(abs(rec_p - (math.floor(exact * 1000 + 0.5) / 1000)) < 1e-9)})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'H15_3_holdout_arithmetic.csv'), index=False, encoding='utf-8-sig')
    ms = [{'line': i, 'snippet': ln.strip()[:200]} for i, ln in enumerate(text, 1) if '53.9' in ln or '53.8' in ln]
    pd.DataFrame(ms, columns=['line', 'snippet']).to_csv(os.path.join(OUT, 'H15_3_manuscript_hits.csv'),
                                                         index=False, encoding='utf-8-sig')
    print(f'cells={len(cells)} mismatches={len(mism)}', file=log)
    log.close()
    print(f'cells={len(cells)} mismatches={len(mism)}')


if __name__ == '__main__':
    main()
