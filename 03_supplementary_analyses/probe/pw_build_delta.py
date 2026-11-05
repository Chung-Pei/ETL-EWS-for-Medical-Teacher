# -*- coding: utf-8 -*-
"""Same-build vs frozen score differences at W8 and W10 (documents the build constant; not used by any estimand).
Frozen = Round 5 keyed OOF scores (arm P, FS-C). Rebuilt = pw_loso_raw_W{W}.csv (this probe, xgboost 3.4.1 CPU)."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('PW_ROOT', HERE.parent))
os.environ.setdefault('R5_RES', str(ROOT / 'results' / '_r5lib_scratch'))
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import r5_lib as L

RES = ROOT / 'results'
EVAL = ['1121', '1122', '1131']


def main():
    base, frozen = L.load_base()
    y, f, sem = base.y.to_numpy().astype(int), base.final.to_numpy(float), base.semester.to_numpy()
    one = np.ones((1, len(base)), np.float32)
    groups = {'7 semesters': L.sem_groups_of(base.sem_i), '1121/1122/1131': [np.flatnonzero(sem == s) for s in EVAL]}
    rows = []
    for W in (8, 10):
        raw = pd.read_csv(RES / f'pw_loso_raw_W{W}.csv', dtype={'anon_id': str, 'semester': str}, float_precision='round_trip')
        fz = frozen[f'W{W}']
        assert np.allclose(fz.m1.to_numpy(), fz.m1.to_numpy())
        sets = {'frozen': (fz.m1.to_numpy(), fz.m2.to_numpy()), 'rebuilt': (raw.m1.to_numpy(), raw.m2.to_numpy())}
        out = {}
        for nm, gi in groups.items():
            for tag, (m1, m2) in sets.items():
                S = L.deploy_blend(m1, m2, sem).reshape(-1, 1)
                a, n = L.auc_weighted(one, S, y, gi)
                r, _ = L.corr_weighted(one, S, f, gi)
                out[(nm, tag)] = tuple(float(v[0, 0]) for v in L.aggregate_deploy(a, r, n))
        for nm in groups:
            fa, fr = out[(nm, 'frozen')]
            ra, rr = out[(nm, 'rebuilt')]
            rows.append({'week': W, 'subset': nm, 'AUC_frozen': fa, 'AUC_rebuilt': ra, 'dAUC': ra - fa,
                         'r_frozen': fr, 'r_rebuilt': rr, 'dr': rr - fr})
        for k, nm in ((0, 'pooled, m1'), (1, 'pooled, m2')):
            fa, ra = L.auc_fail(y, sets['frozen'][k]), L.auc_fail(y, sets['rebuilt'][k])
            fr, rr = L.pearson(sets['frozen'][k], f), L.pearson(sets['rebuilt'][k], f)
            rows.append({'week': W, 'subset': nm, 'AUC_frozen': fa, 'AUC_rebuilt': ra, 'dAUC': ra - fa,
                         'r_frozen': fr, 'r_rebuilt': rr, 'dr': rr - fr})
    df = pd.DataFrame(rows)
    df.to_csv(RES / 'PW_build_delta_W8_W10.csv', index=False)
    print(df.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
