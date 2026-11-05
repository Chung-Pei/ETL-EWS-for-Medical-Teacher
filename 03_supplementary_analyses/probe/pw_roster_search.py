# -*- coding: utf-8 -*-
"""G3 diagnostic: which event-only ids belong to the bounds population of each original per-semester run?
Enumerates inclusion subsets of the reading-active event-only ids; counts mismatches of total_learning_minutes at W8 / W10."""
import itertools, json, sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))      # python -I does not add the script directory
import pw_build as B


def main(weeks=(8, 10)):
    E = B.load_etl()
    m05 = E.m05
    cal = json.load(open(B.CALENDAR, encoding='utf-8'))['semesters']
    q, r = B.load_events(cal)
    shell = B.frozen_shell(8)
    rec_sets = {s: {x['anon_id'] for x in shell[s]} for s in B.D7}
    extra = {(x['anon_id'], x['semester']) for v in shell.values() for x in v}
    pairs = B.pairs_of(q, r, extra)
    qc = E.m03.categorize_activity_df(q, '測驗標題')
    ped = B.exam_dates_for(E, qc, cal, pairs)
    fz = {W: {(x['anon_id'], x['semester']): x['features']['total_learning_minutes'] for x in B.frozen_week(W)['scatter_data']}
          for W in weeks}
    out = []
    for W in weeks:
        rt = E._truncate_by_week(r, ('anon_id', '帳號'), ('開始',), ped, W, scope_semester='')
        rt = rt.assign(m=rt['時間'].map(m05._hms_to_minutes))
        for s in B.D7:
            rs = rt[rt['_semester'] == s].reset_index(drop=True)
            ex_ids = sorted(set(rs['anon_id']) - rec_sets[s])
            for k in range(len(ex_ids) + 1):
                for inc in itertools.combinations(ex_ids, k):
                    pop = rs[rs['anon_id'].isin(rec_sets[s] | set(inc))].reset_index(drop=True)
                    bounds = m05.compute_material_outlier_bounds(pop)
                    sub = rs[rs['anon_id'].isin(rec_sets[s])]
                    cap = np.minimum(sub['m'].to_numpy(), sub['教材標題'].map(bounds).fillna(90.0).to_numpy())
                    tot = pd.Series(cap).groupby(sub['anon_id'].to_numpy()).sum().round(2).to_dict()
                    mism = sum(1 for (a, s_), v in fz[W].items() if s_ == s and abs(tot.get(a, 0.0) - v) > 0.011)
                    out.append({'week': W, 'sem': s, 'included_extras': ','.join(inc) or '-', 'n_extras': len(ex_ids),
                                'mismatch': mism, 'n': len(rec_sets[s])})
    df = pd.DataFrame(out)
    df.to_csv(B.RES / 'G3_roster_search.csv', index=False)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
