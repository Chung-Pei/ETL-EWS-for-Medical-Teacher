# -*- coding: utf-8 -*-
"""Pipeline-equivalence check that does NOT need xgboost: rebuild the frozen BAS score m1 (arm P, feature set C, LOSO) from the
packaged step9 inputs through the pack's own pipeline code and compare it with the frozen keyed file.  Two label variants:
(a) the step9 final-examination score as stored; (b) the semester-score outcome the frozen scores were built for."""
import numpy as np
import pandas as pd

import r5_lib as L
import r5_pipeline as P

EL, EE = P.import_pipeline(allow_stub_for_non_xgb_checks=True)
root = P.merge_step9()
etl = EL.Etl(str(P.ETL_DIR))
EL.init_bas(etl)
base, scores = L.load_base()
rows = []
for variant in ('a_step9_final_exam_label', 'b_semester_score_label'):
    for W in P.W_ALL:
        data = EE.load_all(etl, str(root), [W])
        EE.check_alignment(data)
        if variant.startswith('b'):
            P.apply_semester_outcome(data, base)
        df = EE.canon(EE.run_folds(etl, data[W], EE.fold_defs('P'), etl.FS_C, False, {'m1': 1}))
        assert list(zip(df.semester, df.anon_id)) == list(zip(base.semester, base.anon_id)), 'record order differs'
        d = np.abs(df.m1.to_numpy() - scores[f'W{W}'].m1.to_numpy())
        rows.append({'label_variant': variant, 'week': f'W{W}', 'records': len(df), 'max_abs_diff_m1': float(d.max()),
                     'n_diff_gt_1e-9': int((d > 1e-9).sum()), 'y_equal': bool((df.y.to_numpy() == base.y.to_numpy()).all()),
                     'final_equal': bool(np.allclose(df.final, base.final))})
T = pd.DataFrame(rows)
print(T.to_string())
T.to_csv(L.RES / 'R5-5_9_pipeline_check_BAS.csv', index=False, encoding='utf-8-sig')
