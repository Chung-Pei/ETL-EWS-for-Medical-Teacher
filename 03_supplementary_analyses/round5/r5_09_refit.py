# -*- coding: utf-8 -*-
"""R5-9 (fit stage) -- refitting-inclusive student-cluster bootstrap on the FROZEN-FEATURE path.
For each resample b (first B rows of the standing 2,000-draw stream, seed 20261031): students are drawn with replacement; a drawn
student's records enter every TRAINING fold with the draw multiplicity (duplicated records); the leave-one-semester-out fold
structure, features, FS-C, BAS and XGBoost 3.4.1 settings are those of the frozen pipeline (run_folds, arm P, no midterm).
The held-out semester's unique records are scored (evaluation weights = draw multiplicities are applied in the analysis stage).
Candidate weeks W6, W8, W10, W12 (step9 inputs).  Usage: python r5_09_refit.py START END OUTFILE"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
import time

import numpy as np
import pandas as pd

import r5_lib as L
import r5_pipeline as P

start, end, outfile = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
EL, EE = P.import_pipeline()
import xgboost
assert xgboost.__version__ == '3.4.1'
root = P.merge_step9()
etl = EL.Etl(str(P.ETL_DIR))
EL.init_bas(etl)
base, scores = L.load_base()
n_stu = int(base.stu.max()) + 1
stu_of = {(s, a): int(i) for s, a, i in zip(base.semester, base.anon_id, base.stu)}
counts = L.make_counts(n_stu, 2000, L.SEED)[start:end]            # standing stream; rows start..end-1
data = {}
for W in P.W_ALL:
    d = EE.load_all(etl, str(root), [W])
    P.apply_semester_outcome(d, base)
    data[W] = d[W]
folds = EE.fold_defs('P')
keys = list(zip(base.semester, base.anon_id))
res = {f'm{k}_W{W}': [] for k in (1, 2) for W in P.W_ALL}
idx_log, secs = [], []
t_all = time.time()
for j in range(end - start):
    t0 = time.time()
    c = counts[j]
    for W in P.W_ALL:
        dt = {s: [r for r in recs for _ in range(int(c[stu_of[(r['semester'], r['anon_id'])]]))]
              for s, recs in data[W].items()}
        out = EE.canon(EE.run_folds(etl, data[W], folds, etl.FS_C, False, {'m1': 1, 'm2': 1}, data_train=dt))
        o = out.set_index(['semester', 'anon_id']).loc[keys]
        res[f'm1_W{W}'].append(o.m1.to_numpy(np.float64))
        res[f'm2_W{W}'].append(o.m2.to_numpy(np.float64))
    idx_log.append(start + j)
    secs.append(time.time() - t0)
    if (j + 1) % 5 == 0 or j == end - start - 1:
        np.savez_compressed(outfile, idx=np.array(idx_log), secs=np.array(secs), **{k: np.array(v) for k, v in res.items()})
    print(f'resample {start + j}  {secs[-1]:.1f}s  (elapsed {time.time() - t_all:.0f}s)', flush=True)
print(f'done {end - start} resamples in {time.time() - t_all:.0f}s')
