# -*- coding: utf-8 -*-
"""H-15-3 denominator check. Plan: plan/H153_plan.md. Run from the return root: python -I H-15-3/h153_denominator_check.py"""
import json
import math
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
import linkage as K  # noqa: E402

P2 = os.path.join(ROOT, 'materials', 'pack2')
REPORT = os.path.join(ROOT, 'materials', 'pack1', '03_specs', 'PaperA_holdout_REPORT.md')


def main():
    out = K.load_outcomes_1142(P2)
    h = K.records_df(K.load_holdout_step9(P2), require_final=False)
    n, fails = len(out), int(out['fail'].sum())
    rep = open(REPORT, encoding='utf-8').read()
    lists = []
    for q in (10, 20, 30, 40):
        m = re.search(rf'recall@{q}% / precision@{q}% \| ([\d.]+) / ([\d.]+) \|', rep)
        rec_p, prec_p = float(m.group(1)), float(m.group(2))
        k = int(math.ceil(q / 100 * n))
        hits = int(round(prec_p * k))
        lists.append({'q': q / 100, 'list_n': k, 'precision_printed': prec_p, 'hits_from_report_precision': hits,
                      'recall_printed': rec_p, 'recall_exact': hits / fails})
    blind = [p for p in os.listdir(os.path.join(P2, '08_class_membership_and_1142_outcomes')) if 'blind' in p.lower()]
    res = {'n': n, 'failing': fails, 'fail_equals_semester_score_lt_60': bool(((out['semester_score'] < 60).astype(int) == out['fail']).all()),
           'anon_ids_match_holdout_step9': bool(set(out['anon_id']) == set(h['anon_id'])),
           'lists': lists, 'blind_score_file_present': bool(blind), 'recall40_exact_if_hits_63': 63 / fails}
    json.dump(res, open(os.path.join(ROOT, 'H-15-3', 'H153_denominator_check.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
