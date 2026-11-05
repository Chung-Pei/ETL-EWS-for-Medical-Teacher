# -*- coding: utf-8 -*-
"""LOSO out-of-fold scores (arm P, feature set C, no midterm, models m1 = BAS and m2 = XGBoost) with the frozen pipeline
(ews_lib / ews_experiment of the pack, unmodified), on the rebuilt snapshots.  Same build as Round 5 (xgboost 3.4.1 CPU).
Usage: python pw_loso.py 8 9 10      (W8 = pipeline check against the Round 5 rebuild; W9 = the probe; W10 = same-build comparator)
Holdout 1142 is not loaded: the snapshots hold the 7 development semesters only."""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('PW_ROOT', HERE.parent))
R5CODE = os.environ.get('PW_R5CODE', str(HERE))          # r5_lib.py / r5_pipeline.py (Round 5 helpers) live next to this file
os.environ.setdefault('R5_RES', str(ROOT / 'results' / '_r5lib_scratch'))
sys.path.insert(0, R5CODE)

import numpy as np
import pandas as pd

import r5_lib as L
import r5_pipeline as P

RES = ROOT / 'results'
SNAP = Path(os.environ.get('PW_SNAP', ROOT / 'derived' / 'rebuilt_C'))


def run(weeks):
    EL, EE = P.import_pipeline()
    import xgboost
    print('xgboost', xgboost.__version__, flush=True)
    assert xgboost.__version__ == '3.4.1'
    etl = EL.Etl(str(P.ETL_DIR))
    EL.init_bas(etl)
    base, _scores = L.load_base()
    out = {}
    t0 = time.time()
    for W in weeks:
        data = EE.load_all(etl, str(SNAP), [W])
        assert sorted(data[W].keys()) == L.D7 and '1142' not in data[W]
        P.apply_semester_outcome(data, base)
        df = EE.canon(EE.add_m3(EE.run_folds(etl, data[W], EE.fold_defs('P'), etl.FS_C, False, {'m1': 1, 'm2': 1})))
        assert list(zip(df.semester, df.anon_id)) == list(zip(base.semester, base.anon_id))
        df = df[['semester', 'anon_id', 'm1', 'm2', 'm3']].copy()
        df['final'] = base['final'].to_numpy()
        df['y'] = base['y'].to_numpy()
        df['stu'] = base['stu'].to_numpy()
        fn = RES / f'pw_loso_raw_W{W}.csv'
        df.to_csv(fn, index=False, float_format='%.17g')          # lossless: pandas' default float text drops the last bit
        back = pd.read_csv(fn, dtype={'anon_id': str, 'semester': str}, float_precision='round_trip')
        for c in ('m1', 'm2', 'm3', 'final'):
            assert np.array_equal(back[c].to_numpy(), df[c].to_numpy()), f'CSV round trip not exact for {c}'
        out[W] = df
        print(f'W{W} done {time.time() - t0:.0f}s', flush=True)
    return out


if __name__ == '__main__':
    run([int(w) for w in sys.argv[1:]] or [8, 9, 10])
