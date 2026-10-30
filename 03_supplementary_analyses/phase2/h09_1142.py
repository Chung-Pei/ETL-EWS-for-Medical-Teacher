# -*- coding: utf-8 -*-
"""H-15-3 closure and the H-09 1142 part. Plan: plan/H153b_H09_1142_plan.md.
Run from the return root: python -I H-09_1142/h09_1142.py"""
import hashlib
import json
import math
import os
import platform
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
import h_lib as H  # noqa: E402
import linkage as K  # noqa: E402

P2 = os.path.join(ROOT, 'materials', 'pack2')
BLIND = os.path.join(ROOT, 'materials', 'pack4', 'blind_scores_1142', 'oof_1142_W8_scores_blind.csv')
OOF = os.path.join(ROOT, 'materials', 'pack1', '04_data', 'oof_scores_no_ids.csv')
REPORT = os.path.join(ROOT, 'materials', 'pack1', '03_specs', 'PaperA_holdout_REPORT.md')
CL = os.path.join(ROOT, 'H-04_H-09_cluster_rerun')
OUT = os.path.join(ROOT, 'H-09_1142')
STATED_SHA = '2dae508785fed2c275ee2505b96730321349cd8a67a2eb7633f1d0ab6ab223cd'
QS = (10, 20, 30, 40)
B, SEED = 2000, 20261017


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def hits_info(score, y, k, seed=7):
    n = len(score)
    order = np.lexsort((np.random.default_rng(seed).random(n), score))
    h_rand = int(y[order[:k]].sum())
    v = np.sort(score)[k - 1]
    low, tie = score < v, score == v
    nl, hl, nt, ht = int(low.sum()), int(y[low].sum()), int(tie.sum()), int(y[tie].sum())
    need = k - nl
    return h_rand, hl + max(0, need - (nt - ht)), hl + min(ht, need)


def half_up3(x):
    return math.floor(x * 1000 + 0.5) / 1000


def main():
    log = open(os.path.join(OUT, 'run.log'), 'w', encoding='utf-8')
    print(f'python {platform.python_version()} numpy {np.__version__} pandas {pd.__version__}', file=log)
    for p in (BLIND, OOF, REPORT):
        print(f'sha256 {os.path.relpath(p, ROOT)} {sha(p)}', file=log)
    print(f'bootstrap: record-level (one record per student in 1142), B={B}, seed={SEED}', file=log)

    b = pd.read_csv(BLIND, encoding='utf-8-sig', dtype={'anon_id': str})
    o = K.load_outcomes_1142(P2)
    h = K.records_df(K.load_holdout_step9(P2), require_final=False)
    m = b.merge(o, on='anon_id', how='inner').sort_values('anon_id').reset_index(drop=True)
    assert len(m) == 249 == len(b)
    y = m['fail'].to_numpy(int)
    fin = m['semester_score'].to_numpy(float)
    n, nfail = len(m), int(y.sum())

    within = (H.pct_rank(m['m1'].to_numpy()) + H.pct_rank(m['m2'].to_numpy())) / 2
    ident = float(np.abs(m['m3'].to_numpy() - within).max())
    oof = H.load_oof(OOF)['W8']
    ref1, ref2 = oof['m1'].to_numpy(float), oof['m2'].to_numpy(float)
    pooled_ref = (H.pct_ref(m['m1'].to_numpy(), ref1) + H.pct_ref(m['m2'].to_numpy(), ref2)) / 2
    scores = {"M3' within-cohort (file)": m['m3'].to_numpy(float), "M3' pooled-reference (D7 OOF)": pooled_ref,
              "M1' (file)": m['m1'].to_numpy(float), "M2' (file)": m['m2'].to_numpy(float)}

    rep = open(REPORT, encoding='utf-8').read()
    printed = {}
    for q in QS:
        mm = re.search(rf'recall@{q}% / precision@{q}% \| ([\d.]+) / ([\d.]+) \|', rep)
        printed[q] = (float(mm.group(1)), float(mm.group(2)))
    p_auc3 = float(re.search(r'\| AUC \| ([\d.]+) \|', rep).group(1))
    p_auc4 = float(re.search(r'AUC 點估計[^|]*\| ([\d.]+) \|', rep).group(1))
    p_r = float(re.search(r'\| r \| ([\d.]+) \|', rep).group(1))

    # ---------------------------------------------------------------- A. H-15-3
    auc_m3 = H.auc_fail(y, scores["M3' within-cohort (file)"])
    r_m3 = H.pearson(scores["M3' within-cohort (file)"], fin)
    rows, lists = [], [int(math.ceil(q / 100 * n)) for q in QS]
    for q, k in zip(QS, lists):
        hr, hmin, hmax = hits_info(scores["M3' within-cohort (file)"], y, k)
        rec, prec = hr / nfail, hr / k
        rows.append({'q': q / 100, 'list_n': k, 'hits_random_tie': hr, 'hits_min': hmin, 'hits_max': hmax,
                     'recall_exact': rec, 'precision_exact': prec, 'recall_printed': printed[q][0], 'precision_printed': printed[q][1],
                     'recall_matches_printed_3dp': bool(abs(half_up3(rec) - printed[q][0]) < 1e-9),
                     'precision_matches_printed_3dp': bool(abs(half_up3(prec) - printed[q][1]) < 1e-9),
                     'hits_invariant_to_ties': bool(hmin == hmax)})
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, 'H-15-3', 'H153_raw_hits.csv'), index=False, encoding='utf-8-sig')
    chk = {'sha256': sha(BLIND), 'sha256_stated': STATED_SHA, 'sha256_matches_stated': sha(BLIND) == STATED_SHA, 'rows': int(len(b)),
           'anon_ids_match_outcomes': bool(set(b['anon_id']) == set(o['anon_id'])),
           'anon_ids_match_holdout_step9': bool(set(b['anon_id']) == set(h['anon_id'])), 'failing': nfail,
           'm3_identity_max_abs_diff_vs_within_cohort_percentile_blend': ident,
           'm3_auc': auc_m3, 'm3_pearson_r_semester_score': r_m3, 'm1_auc': H.auc_fail(y, scores["M1' (file)"]),
           'm2_auc': H.auc_fail(y, scores["M2' (file)"]), 'report_auc_3dp': p_auc3, 'report_auc_4dp': p_auc4, 'report_r_3dp': p_r}
    json.dump(chk, open(os.path.join(ROOT, 'H-15-3', 'H153_blind_file_check.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    # ---------------------------------------------------------------- B. metrics and delta
    draws = np.random.default_rng(SEED).integers(0, n, size=(B, n))
    boot_auc = {k: np.array([H.auc_fail(y[d], v[d]) for d in draws]) for k, v in scores.items()}
    boot_r = {k: np.array([np.corrcoef(v[d], fin[d])[0, 1] for d in draws]) for k, v in scores.items()}
    mrows = []
    for name, v in scores.items():
        row = {'score': name, 'auc': H.auc_fail(y, v), 'auc_lo': H.percentile_ci(boot_auc[name])[0], 'auc_hi': H.percentile_ci(boot_auc[name])[1],
               'r': H.pearson(v, fin), 'r_lo': H.percentile_ci(boot_r[name])[0], 'r_hi': H.percentile_ci(boot_r[name])[1]}
        for q, k in zip(QS, lists):
            hr, hmin, hmax = hits_info(v, y, k)
            row.update({f'hits_{q}': hr, f'hits_min_{q}': hmin, f'hits_max_{q}': hmax, f'recall_{q}': hr / nfail, f'precision_{q}': hr / k})
        mrows.append(row)
    met = pd.DataFrame(mrows)
    met.to_csv(os.path.join(OUT, 'H09_1142_metrics.csv'), index=False, encoding='utf-8-sig')
    a_, b_ = "M3' pooled-reference (D7 OOF)", "M3' within-cohort (file)"
    drows = []
    for nm, pa, pb, ba, bb in (('auc', H.auc_fail(y, scores[a_]), H.auc_fail(y, scores[b_]), boot_auc[a_], boot_auc[b_]),
                               ('r', H.pearson(scores[a_], fin), H.pearson(scores[b_], fin), boot_r[a_], boot_r[b_])):
        lo, hi = H.percentile_ci(ba - bb)
        drows.append({'metric': f'{nm} (pooled-reference minus within-cohort)', 'delta': pa - pb, 'lo': lo, 'hi': hi})
    for q in QS:
        ra = met[met['score'] == a_].iloc[0][f'hits_{q}']
        rb = met[met['score'] == b_].iloc[0][f'hits_{q}']
        drows.append({'metric': f'hits@{q}% (pooled-reference minus within-cohort)', 'delta': ra - rb, 'lo': np.nan, 'hi': np.nan})
    dl = pd.DataFrame(drows)
    dl.to_csv(os.path.join(OUT, 'H09_1142_delta.csv'), index=False, encoding='utf-8-sig')

    # ---------------------------------------------------------------- vs development distribution (Week 8)
    summ = pd.read_csv(os.path.join(CL, 'H09c_1_summary.csv'), encoding='utf-8-sig')
    summ = summ[summ['W'] == 'W8'].set_index('model')
    vrows = []
    for model, nm in (("M1'", "M1' (file)"), ("M2'", "M2' (file)"), ("M3'", "M3' within-cohort (file)")):
        a = float(met[met['score'] == nm].iloc[0]['auc'])
        s = summ.loc[model]
        vrows.append({'predictor': model, 'auc_1142': a, 'dev_weighted_within_semester_auc': s['weighted_mean_auc'],
                      'dev_random_effects_auc': s['dl_auc'], 'pi_lo': s['dl_pi_lo'], 'pi_hi': s['dl_pi_hi'],
                      'dev_semester_min': s['sem_auc_min'], 'dev_semester_max': s['sem_auc_max'],
                      'inside_prediction_interval': bool(s['dl_pi_lo'] <= a <= s['dl_pi_hi']),
                      'inside_min_max': bool(s['sem_auc_min'] <= a <= s['sem_auc_max']),
                      'diff_vs_dev_weighted': a - s['weighted_mean_auc']})
    vs = pd.DataFrame(vrows)
    vs.to_csv(os.path.join(OUT, 'H09_1142_vs_dev.csv'), index=False, encoding='utf-8-sig')

    # ---------------------------------------------------------------- markdown tables from the output files
    def f3(x):
        return 'NA' if pd.isna(x) else f'{x:.3f}'.replace('-', '\u2212')

    def f4(x):
        return 'NA' if pd.isna(x) else f'{x:.4f}'.replace('-', '\u2212')
    L = ['### H-15-3: raw hits from the blind score file (within-cohort M3\u2032)', '',
         '| List | n | Hits (random ties) | Hits range under ties | Recall (exact) | Recall printed | Precision (exact) | Precision printed |', '|---|---|---|---|---|---|---|---|']
    for r in rows:
        L.append(f"| {int(r['q'] * 100)}% | {r['list_n']} | {r['hits_random_tie']} | {r['hits_min']}\u2013{r['hits_max']} | {f4(r['recall_exact'])} | {r['recall_printed']} | {f4(r['precision_exact'])} | {r['precision_printed']} |")
    L += ['', f"AUC {f4(auc_m3)} (report prints {p_auc3} and {p_auc4}); r {f4(r_m3)} (report {p_r}); file SHA-256 matches stated: {chk['sha256_matches_stated']}; m3 identity max diff {ident:.1e}.",
          '', '### H-09 1142 part: scores on the 249 students of 1142', '',
          '| Score | AUC [95% CI] | r [95% CI] | Hits@10/20/30/40% | Recall@20% | Precision@20% |', '|---|---|---|---|---|---|']
    for _, r in met.iterrows():
        L.append(f"| {r['score']} | {f3(r['auc'])} [{f3(r['auc_lo'])}\u2013{f3(r['auc_hi'])}] | {f3(r['r'])} [{f3(r['r_lo'])}\u2013{f3(r['r_hi'])}] | "
                 f"{r['hits_10']}/{r['hits_20']}/{r['hits_30']}/{r['hits_40']} | {f3(r['recall_20'])} | {f3(r['precision_20'])} |")
    L += ['', '| Difference | Value [95% CI] |', '|---|---|']
    for _, r in dl.iterrows():
        ci = '' if pd.isna(r['lo']) else f" [{f3(r['lo'])}; {f3(r['hi'])}]"
        L.append(f"| {r['metric']} | {f3(r['delta']) if 'hits' not in r['metric'] else int(r['delta'])}{ci} |")
    L += ['', '### 1142 versus the development distribution (Week 8, within-cohort scores)', '',
          '| Predictor | AUC 1142 | Dev weighted within-semester AUC | Prediction interval | Semester range | Inside interval | Inside range |', '|---|---|---|---|---|---|---|']
    for _, r in vs.iterrows():
        L.append(f"| {r['predictor']} | {f3(r['auc_1142'])} | {f3(r['dev_weighted_within_semester_auc'])} | {f3(r['pi_lo'])}\u2013{f3(r['pi_hi'])} | "
                 f"{f3(r['dev_semester_min'])}\u2013{f3(r['dev_semester_max'])} | {r['inside_prediction_interval']} | {r['inside_min_max']} |")
    open(os.path.join(OUT, 'H09_1142_summary_tables.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    log.close()
    print('done')


if __name__ == '__main__':
    main()
