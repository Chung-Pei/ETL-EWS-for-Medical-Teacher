# -*- coding: utf-8 -*-
"""Score-level G3: LOSO on the FROZEN step9 inputs vs LOSO on the rebuilt snapshots (same build). Bitwise equality of m1, m2, m3
shows that the rebuilt features carry no information that the frozen features do not (the feature path is exact).
Usage: python -I pw_frozen_loso_check.py 8 10      (needs results/pw_loso_raw_W{W}.csv from pw_loso.py)"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('PW_ROOT', HERE.parent))
os.environ.setdefault('R5_RES', str(ROOT / 'results' / '_r5lib_scratch'))
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import xgboost

import r5_lib as L
import r5_pipeline as P

RES = ROOT / 'results'


def main(weeks):
    assert xgboost.__version__ == '3.4.1'
    P.merge_step9()                                         # concatenates the frozen per-semester files into derived/inputs_merged
    EL, EE = P.import_pipeline()
    etl = EL.Etl(str(P.ETL_DIR))
    EL.init_bas(etl)
    base, _ = L.load_base()
    rows = []
    for W in weeks:
        data = EE.load_all(etl, str(P.MERGED), [W])
        assert sorted(data[W].keys()) == L.D7 and '1142' not in data[W]
        P.apply_semester_outcome(data, base)
        df = EE.canon(EE.add_m3(EE.run_folds(etl, data[W], EE.fold_defs('P'), etl.FS_C, False, {'m1': 1, 'm2': 1})))
        rb = pd.read_csv(RES / f'pw_loso_raw_W{W}.csv', dtype={'anon_id': str, 'semester': str}, float_precision='round_trip')
        assert list(zip(df.semester, df.anon_id)) == list(zip(rb.semester, rb.anon_id))
        for m in ('m1', 'm2', 'm3'):
            d = float(np.max(np.abs(df[m].to_numpy() - rb[m].to_numpy())))
            rows.append({'week': W, 'score': m, 'n': len(df), 'max_abs_diff_frozen_inputs_vs_rebuilt_snapshot': d})
    out = pd.DataFrame(rows)
    out.to_csv(RES / 'PW_frozen_input_LOSO_check.csv', index=False)
    print(out.to_string(index=False))


if __name__ == '__main__':
    main([int(w) for w in sys.argv[1:]] or [8, 10])
