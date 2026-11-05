# -*- coding: utf-8 -*-
"""G3: rebuild W8 / W10 snapshots from events and compare with the frozen step9 files (pre-registered, see ledger)."""
import json, sys, time
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))      # python -I does not add the script directory
import pw_build as B

def main(weeks, tag='rebuilt_C', extra_tracks=('2yr',)):
    t0 = time.time()
    E = B.load_etl()
    cal = json.load(open(B.CALENDAR, encoding='utf-8'))['semesters']
    q, r = B.load_events(cal)
    shell = B.frozen_shell(8)
    extra = {(x['anon_id'], x['semester']) for v in shell.values() for x in v}
    pairs = B.pairs_of(q, r, extra)
    q = E.m03.categorize_activity_df(q, '測驗標題')
    ped = B.exam_dates_for(E, q, cal, pairs)
    print(f'setup {time.time()-t0:.0f}s; pairs={len(pairs)}')
    allrows = []
    for W in weeks:
        new = B.build_week(E, W, q, r, cal, ped, shell, bounds_extra_tracks=extra_tracks)
        B.write_merged(new, W, tag)
        rows = B.compare(new, W)
        allrows += rows
        print(f'W{W} done {time.time()-t0:.0f}s')
        print(pd.DataFrame(rows).to_string(index=False))
    pd.DataFrame(allrows).to_csv(B.RES / f'G3_feature_comparison_{tag}.csv', index=False)

if __name__ == '__main__':
    args = sys.argv[1:]
    tag = 'rebuilt_C'
    if '--tag' in args:                                   # optional: python -I pw_g3.py 6 12 --tag rebuilt_C_W6W12
        i = args.index('--tag')
        tag = args[i + 1]
        args = args[:i] + args[i + 2:]
    main([int(x) for x in args], tag=tag)
