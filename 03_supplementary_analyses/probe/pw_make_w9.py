# -*- coding: utf-8 -*-
"""Build the Week-9 snapshot (all 7 development semesters) with the construction that passed G3 at W8 and W10.
Evaluation semesters are 1121/1122/1131; the other four are needed as LOSO training folds. 1142 is NOT built."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))      # python -I does not add the script directory
import pw_build as B


def main(W=9, tag='rebuilt_C'):
    t0 = time.time()
    E = B.load_etl()
    cal = json.load(open(B.CALENDAR, encoding='utf-8'))['semesters']
    q, r = B.load_events(cal)
    shell = B.frozen_shell(8)
    extra = {(x['anon_id'], x['semester']) for v in shell.values() for x in v}
    pairs = B.pairs_of(q, r, extra)
    q = E.m03.categorize_activity_df(q, '測驗標題')
    ped = B.exam_dates_for(E, q, cal, pairs)
    new = B.build_week(E, W, q, r, cal, ped, shell)
    out = B.write_merged(new, W, tag)
    print(f'W{W} snapshot written to {out} in {time.time()-t0:.0f}s; records {len(new["scatter_data"])}')
    sems = sorted({x['semester'] for x in new['scatter_data']})
    assert sems == B.D7 and '1142' not in sems


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 9)
