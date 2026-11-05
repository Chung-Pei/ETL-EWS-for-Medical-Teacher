# -*- coding: utf-8 -*-
"""Long-format row helper (columns as Round 6: item, basis, comparison, week, n, estimate, ci_lo, ci_hi, note)."""
import numpy as np
import pandas as pd
import r7_common as C

COLS = ['item', 'basis', 'comparison', 'week', 'n', 'estimate', 'ci_lo', 'ci_hi', 'note']


def add(rows, item, basis, comparison, week='', n=np.nan, estimate=np.nan, ci_lo=np.nan, ci_hi=np.nan, note=''):
    rows.append({'item': item, 'basis': basis, 'comparison': comparison, 'week': week, 'n': n,
                 'estimate': estimate, 'ci_lo': ci_lo, 'ci_hi': ci_hi, 'note': note})


def write(name, rows):
    d = C.RES / 'parts'
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLS).to_csv(d / f'{name}.csv', index=False, encoding='utf-8-sig')
