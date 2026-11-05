# -*- coding: utf-8 -*-
"""Self-check of the R5-9 fit stage: with every student count = 1 the duplicated-training path (data_train) must reproduce the
plain rebuilt LOSO of the reproduction gate (W8: AUC M1' 0.693305, M2' 0.704932, M3' 0.725472)."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import r5_lib as L
import r5_pipeline as P

EL, EE = P.import_pipeline()
root = P.merge_step9()
etl = EL.Etl(str(P.ETL_DIR)); EL.init_bas(etl)
base, scores = L.load_base()
stu_of = {(s, a): int(i) for s, a, i in zip(base.semester, base.anon_id, base.stu)}
c = np.ones(int(base.stu.max()) + 1)
d = EE.load_all(etl, str(root), [8]); P.apply_semester_outcome(d, base)
data = d[8]
dt = {s: [r for r in recs for _ in range(int(c[stu_of[(r['semester'], r['anon_id'])]]))] for s, recs in data.items()}
out = EE.canon(EE.add_m3(EE.run_folds(etl, data, EE.fold_defs('P'), etl.FS_C, False, {'m1': 1, 'm2': 1}, data_train=dt)))
y = base.y.to_numpy()
got = [round(L.auc_fail(y, out[k]), 6) for k in ('m1', 'm2', 'm3')]
print('counts=1 path AUC (m1, m2, m3):', got, '| expected [0.693305, 0.704932, 0.725472]', '| MATCH' if got == [0.693305, 0.704932, 0.725472] else '| MISMATCH')
