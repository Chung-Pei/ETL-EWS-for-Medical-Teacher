# -*- coding: utf-8 -*-
"""Helper (not part of the frozen analysis scripts): merge the per-week files written by h09p4_retrain.py into
H09p4_oof_full.csv / H09p4_oof_no4t.csv (rows ordered W6, W8, W10, W12 as in the delivered files).
Usage: python merge_oof.py results"""
import os
import sys

import pandas as pd

out = sys.argv[1] if len(sys.argv) > 1 else 'results'
for cfg in ['full', 'no4t']:
    parts = [pd.read_csv(os.path.join(out, f'H09p4_oof_{cfg}_W{w}.csv'), dtype={'semester': str}) for w in [6, 8, 10, 12]]
    pd.concat(parts, ignore_index=True).to_csv(os.path.join(out, f'H09p4_oof_{cfg}.csv'), index=False, encoding='utf-8-sig')
    print('merged', cfg)
