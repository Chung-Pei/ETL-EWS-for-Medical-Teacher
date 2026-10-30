# -*- coding: utf-8 -*-
"""H-04: reference-week dependence of the earliest qualifying warning week.

Run from the return root:  python -I H-04/h04_reference_week.py
Plan: plan/supp_analysis_plan.md section 3. CI = record-level paired percentile bootstrap (approximate; see plan R2).
"""
import hashlib
import os
import shutil
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
import h_lib as H  # noqa: E402

PACK = os.path.join(ROOT, 'materials', 'PaperA_supp_analyses_newchat_pack_20261026')
OOF = os.path.join(PACK, '04_data', 'oof_scores_no_ids.csv')
XLSX = os.path.join(PACK, '04_data', 'Table_A_data_era.xlsx')
OUT = os.path.join(ROOT, 'H-04')
REFS = ['W12', 'W14', 'W16']
CANDS = ['W6', 'W8', 'W10', 'W12']
APPROX_TOL = 0.003


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def main():
    log = open(os.path.join(OUT, 'run.log'), 'w', encoding='utf-8')
    H.log_versions(log)
    for p in (OOF, XLSX):
        print(f'sha256 {os.path.relpath(p, ROOT)} {sha(p)}', file=log)
    print(f'bootstrap: record-level paired, B={H.BOOT_B}, seed={H.BOOT_SEED}, rng=numpy default_rng', file=log)

    oof = H.load_oof(OOF)
    n = len(oof['W6'])
    draws = H.record_draws(n)
    weeks = ['W6', 'W8', 'W10', 'W12', 'W14', 'W16']
    keys = [(W, c) for W in weeks for c in ('m1', 'm2', 'm3')]
    boot = H.boot_record_series(oof, keys, draws)
    pt = {}
    for W, c in keys:
        df = oof[W]
        pt[(W, c)] = (H.auc_fail(df['y'].to_numpy(int), df[c].to_numpy(float)), H.pearson(df[c], df['final']))

    fz = pd.ExcelFile(XLSX).parse('era_metric')
    fz = fz[fz['era'] == 'ALL'].set_index(['model', 'W'])

    rows = []
    for model, col in H.MODEL_COL.items():
        for ref in REFS:
            for W in CANDS:
                a_ref, r_ref = pt[(ref, col)]
                a_w, r_w = pt[(W, col)]
                la_b = boot[(ref, col)][0] - boot[(W, col)][0]
                lr_b = boot[(ref, col)][1] - boot[(W, col)][1]
                row = {'model': model, 'ref': ref, 'W': W, 'auc_ref': a_ref, 'auc_w': a_w, 'auc_loss': a_ref - a_w,
                       'auc_loss_hi95': H.percentile_hi(la_b, 0.975), 'auc_loss_hi993': H.percentile_hi(la_b, 'bonf'),
                       'r_ref': r_ref, 'r_w': r_w, 'r_loss': r_ref - r_w,
                       'r_loss_hi95': H.percentile_hi(lr_b, 0.975), 'r_loss_hi993': H.percentile_hi(lr_b, 'bonf')}
                row['point_ok'] = H.qualifies(row['auc_loss'], row['r_loss'])
                row['ci95_ok'] = H.qualifies(row['auc_loss_hi95'], row['r_loss_hi95'])
                row['ci993_ok'] = H.qualifies(row['auc_loss_hi993'], row['r_loss_hi993'])
                row['ci_method'] = 'record-level paired bootstrap (approximate; not student-cluster)'
                rows.append(row)
    loss = pd.DataFrame(rows)
    loss.to_csv(os.path.join(OUT, 'H04_loss_by_reference.csv'), index=False, encoding='utf-8-sig')

    er = []
    for model in H.MODEL_COL:
        for ref in REFS:
            sub = loss[(loss['model'] == model) & (loss['ref'] == ref)]
            for crit, colname in (('point', 'point_ok'), ('ci95', 'ci95_ok'), ('ci993', 'ci993_ok')):
                ok = {int(w[1:]): bool(v) for w, v in zip(sub['W'], sub[colname])}
                e = H.earliest_week(ok)
                er.append({'model': model, 'ref': ref, 'criterion': crit, 'earliest_W': 'none' if e is None else e})
    pd.DataFrame(er).to_csv(os.path.join(OUT, 'H04_earliest_qualifying_week.csv'), index=False, encoding='utf-8-sig')

    ap = []
    for model in H.MODEL_COL:
        for W in ('W6', 'W8', 'W10'):
            r = loss[(loss['model'] == model) & (loss['ref'] == 'W12') & (loss['W'] == W)].iloc[0]
            f = fz.loc[(model, W)]
            d_a = r['auc_loss_hi95'] - float(f['auc_loss_hi'])
            d_r = r['r_loss_hi95'] - float(f['r_loss_hi'])
            ap.append({'model': model, 'W': W, 'record_auc_hi': r['auc_loss_hi95'], 'frozen_cluster_auc_hi': float(f['auc_loss_hi']),
                       'diff_auc_hi': d_a, 'record_r_hi': r['r_loss_hi95'], 'frozen_cluster_r_hi': float(f['r_loss_hi']),
                       'diff_r_hi': d_r, 'approx_ok': bool(max(abs(d_a), abs(d_r)) <= APPROX_TOL + 1e-12)})
    pd.DataFrame(ap).to_csv(os.path.join(OUT, 'H04_ci_approx_check.csv'), index=False, encoding='utf-8-sig')


    # memo tables generated from the output files (no hand transcription)
    def fmt(x):
        return f'{x:.3f}'.replace('-', '\u2212')
    lines = ['### Week 8 losses by reference week (point estimate [95% upper limit; 99.3% upper limit])', '',
             '| Predictor | Reference | AUC loss | r loss | Point criterion | 95% CI criterion | 99.3% CI criterion |',
             '|---|---|---|---|---|---|---|']
    for _, r in loss[loss['W'] == 'W8'].iterrows():
        lines.append(f"| {r['model']} | {r['ref']} | {fmt(r['auc_loss'])} [{fmt(r['auc_loss_hi95'])}; {fmt(r['auc_loss_hi993'])}] "
                     f"| {fmt(r['r_loss'])} [{fmt(r['r_loss_hi95'])}; {fmt(r['r_loss_hi993'])}] "
                     f"| {'met' if r['point_ok'] else 'not met'} | {'met' if r['ci95_ok'] else 'not met'} "
                     f"| {'met' if r['ci993_ok'] else 'not met'} |")
    lines += ['', '### Earliest qualifying week among W6/8/10/12 (none = no candidate qualifies)', '',
              '| Predictor | Criterion | Ref W12 | Ref W14 | Ref W16 |', '|---|---|---|---|---|']
    ew = pd.DataFrame(er)
    for model in H.MODEL_COL:
        for crit in ('point', 'ci95', 'ci993'):
            vals = [str(ew[(ew['model'] == model) & (ew['ref'] == ref) & (ew['criterion'] == crit)]['earliest_W'].iloc[0])
                    for ref in REFS]
            lines.append(f'| {model} | {crit} | ' + ' | '.join(vals) + ' |')
    open(os.path.join(OUT, 'H04_summary_tables.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')

    shutil.copyfile(os.path.join(OUT, 'rule_version_history_source.csv'), os.path.join(OUT, 'H04_rule_version_history.csv'))
    print('done', file=log)
    log.close()
    print('done')


if __name__ == '__main__':
    main()
