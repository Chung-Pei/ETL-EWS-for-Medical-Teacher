# -*- coding: utf-8 -*-
"""R5-5 -- forward-chaining (expanding-window) validation with the frozen pipeline code, XGBoost 3.4.1 CPU, five seeds.
Targets 1121, 1122, 1131, 1132, 1141; training = all strictly earlier semesters (>= 2).  Two policies per target:
(i) Week-8 blend, pooled percentile construction (ranks pooled over the five target predictions);
(ii) Week-10 blend, within-semester construction (ranks and lists within the target semester)."""
import math
import time

import numpy as np
import pandas as pd

import r5_lib as L
import r5_pipeline as P

EL, EE = P.import_pipeline()
import xgboost
assert xgboost.__version__ == '3.4.1'
root = P.merge_step9()
etl = EL.Etl(str(P.ETL_DIR))
EL.init_bas(etl)
base, scores = L.load_base()
TARGETS = ['1121', '1122', '1131', '1132', '1141']
t0 = time.time()
raw = {}
for W in (8, 10):
    data = EE.load_all(etl, str(root), [W])
    P.apply_semester_outcome(data, base)
    fwd = EE.canon(EE.run_folds(etl, data[W], EE.fold_defs('FWD'), etl.FS_C, False, {'m1': 1, 'm2': 1}))
    loso = EE.canon(EE.run_folds(etl, data[W], EE.fold_defs('P'), etl.FS_C, False, {'m1': 1, 'm2': 1}))
    raw[W] = (fwd, loso)
    print(f'W{W} fits done {time.time() - t0:.0f}s', flush=True)

# ---- scores per variant ---------------------------------------------------------------------------------------------------------
def policy_scores(frame, W):
    if W == 8:
        return L.pooled_blend(frame.m1, frame.m2)                          # pooled over the frame's records
    return L.deploy_blend(frame.m1, frame.m2, frame.semester.to_numpy())    # within semester


fwd8, loso8 = raw[8]
fwd10, loso10 = raw[10]
key = lambda df: list(zip(df.semester, df.anon_id))
assert key(fwd8) == key(fwd10) and key(loso8) == key(loso10)
F = fwd8[['semester', 'anon_id', 'final', 'y']].copy().reset_index(drop=True)
F['fwd_W8policy'] = policy_scores(fwd8, 8)
F['fwd_W10policy'] = policy_scores(fwd10, 10)
F['fwd_m1_W8'], F['fwd_m2_W8'] = fwd8.m1.to_numpy(), fwd8.m2.to_numpy()
F['fwd_m1_W10'], F['fwd_m2_W10'] = fwd10.m1.to_numpy(), fwd10.m2.to_numpy()
LO = loso8[['semester', 'anon_id']].copy().reset_index(drop=True)
LO['loso_W8policy'] = policy_scores(loso8, 8)                               # pooled over all 7 semesters, as frozen
LO['loso_W10policy'] = policy_scores(loso10, 10)
frz = base[['semester', 'anon_id']].copy()
frz['frozen_W8policy'] = scores['W8'].m3.to_numpy()
frz['frozen_W10policy'] = L.deploy_blend(scores['W10'].m1, scores['W10'].m2, base.semester.to_numpy())
M = F.merge(LO, on=['semester', 'anon_id'], how='left', validate='one_to_one').merge(frz, on=['semester', 'anon_id'],
                                                                                 how='left', validate='one_to_one')
assert M[['loso_W8policy', 'frozen_W8policy']].notna().all().all() and len(M) == F.shape[0]
assert set(M.semester) == set(TARGETS)
# score file for independent re-verification
out = M[['semester', 'anon_id', 'fwd_W8policy', 'fwd_W10policy', 'y', 'final', 'fwd_m1_W8', 'fwd_m2_W8', 'fwd_m1_W10',
         'fwd_m2_W10']].rename(columns={'fwd_W8policy': 'score_W8_policy (pooled blend, pass-oriented)',
                                         'fwd_W10policy': 'score_W10_policy (within-semester blend, pass-oriented)',
                                         'y': 'outcome_fail_semester_score_lt60', 'final': 'semester_score'})
out.to_csv(L.RES / 'R5-5_scores.csv', index=False, encoding='utf-8-sig')

# ---- metrics with record-level bootstrap (target semester alone) ---------------------------------------------------------------------
rows = []
VAR = [('forward_chaining', 'fwd'), ('LOSO_rebuilt_same_code', 'loso'), ('LOSO_frozen_keyed_file', 'frozen')]
for pol, pcol, wlabel in (('(i) Week-8 blend, pooled construction', 'W8policy', 'W8'),
                          ('(ii) Week-10 blend, within-semester construction', 'W10policy', 'W10')):
    for tg in TARGETS + ['five-target aggregate']:
        sub = M if tg.startswith('five') else M[M.semester == tg]
        sub = sub.reset_index(drop=True)
        n = len(sub)
        y = sub.y.to_numpy().astype(int)
        f = sub.final.to_numpy(float)
        S = np.column_stack([sub[f'{v}_{pcol}'].to_numpy(float) for _, v in VAR])
        sem = sub.semester.to_numpy()
        if tg.startswith('five'):
            b2 = base[base.semester.isin(TARGETS)].reset_index(drop=True)
            idmap = {(s, a): i for i, (s, a) in enumerate(zip(b2.semester, b2.anon_id))}
            order = [idmap[(s, a)] for s, a in zip(sub.semester, sub.anon_id)]
            stu = pd.factorize(sub.anon_id)[0]
            Wm = L.make_weights(stu, B=L.BOOT_B, seed=L.SEED)
            groups = L.sem_groups_of(pd.Series(sem).map({s: i for i, s in enumerate(TARGETS)}).to_numpy())
            a_d, r_d = L.metrics_draws(Wm, S, y, f, groups, 'deploy')
            a_p, r_p = L.metrics_draws(np.ones((1, n), np.float32), S, y, f, groups, 'deploy')
            aucs, rs, a_dr, r_dr = a_p[0], r_p[0], a_d, r_d
            extra = 'n-weighted within-semester AUC / Fisher-z r over the five targets; student-cluster bootstrap'
        else:
            Wm = L.make_counts(n, L.BOOT_B, L.SEED)                  # record = cluster inside one semester
            allix = [np.arange(n)]
            a_d, _ = L.auc_weighted(Wm, S, y, allix)
            r_d, _ = L.corr_weighted(Wm, S, f, allix)
            a_p, _ = L.auc_weighted(np.ones((1, n), np.float32), S, y, allix)
            r_p, _ = L.corr_weighted(np.ones((1, n), np.float32), S, f, allix)
            aucs, rs, a_dr, r_dr = a_p[0, 0], r_p[0, 0], a_d[:, 0, :], r_d[:, 0, :]
            extra = 'record-level bootstrap B=2000, target semester alone'
        # Recall@20%: within-semester lists (ceil(0.2 n)), fixed, weights resampled
        rec_pt, rec_dr = [], []
        for c in range(S.shape[1]):
            lst = L.list_by_k(S[:, c], sem, lambda m: int(math.ceil(0.2 * m)))
            fl = Wm @ (y == 1).astype(np.float32)
            rec_dr.append((Wm @ (lst & (y == 1)).astype(np.float32)) / fl)
            rec_pt.append((lst & (y == 1)).sum() / y.sum())
        rec_dr = np.column_stack(rec_dr)
        for mname, pt, dr in (('AUC', aucs, a_dr), ('r (Pearson, semester score)', rs, r_dr), ('Recall@20%', np.array(rec_pt), rec_dr)):
            row = {'policy': pol, 'target': tg, 'n': n, 'n_fail': int(y.sum()), 'metric': mname, 'note': extra}
            for k, (nm, _) in enumerate(VAR):
                lo, hi = L.ci(dr[:, k])
                row.update({f'{nm}_est': float(pt[k]), f'{nm}_lo': lo, f'{nm}_hi': hi})
            for k2, nm2 in ((1, 'minus_LOSO_rebuilt'), (2, 'minus_LOSO_frozen')):
                d = dr[:, 0] - dr[:, k2]
                lo, hi = L.ci(d)
                row.update({f'forward_{nm2}_diff': float(pt[0] - pt[k2]), f'forward_{nm2}_lo': lo, f'forward_{nm2}_hi': hi})
            rows.append(row)
T = pd.DataFrame(rows)
T.to_csv(L.RES / 'R5-5.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 280, 'display.max_columns', 30)
print(T[['policy', 'target', 'n', 'n_fail', 'metric', 'forward_chaining_est', 'forward_chaining_lo', 'forward_chaining_hi',
         'LOSO_rebuilt_same_code_est', 'LOSO_frozen_keyed_file_est', 'forward_minus_LOSO_rebuilt_diff']].to_string())
print(f'total {time.time() - t0:.0f}s')
