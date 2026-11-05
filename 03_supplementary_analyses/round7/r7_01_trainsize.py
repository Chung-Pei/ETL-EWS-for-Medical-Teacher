# -*- coding: utf-8 -*-
"""R7-1: Tier B training-volume matching inside Era E1: all 20 ordered (train s, test t) single-semester pairs, s != t.
Same retraining pipeline as gate G-b (pack ews_lib / ews_experiment, grade-free FS-C features, BAS constants from the training
semester only, five XGBoost seeds).  Selection rule (Appendix B) applied inside t; intervals: student-cluster bootstrap,
B = 2000, seed 20261031, paired draws."""
import r7_retrain as R
import r7_common as C
import r7_rows as RW
import time
import numpy as np
import pandas as pd

L = C.L
t0 = time.time()
EL, EE, etl, base, scores, data = R.setup()
E1 = ['1111', '1112', '1121', '1122', '1131']
FTRACK = {'1112', '1122'}          # four-year technical semesters (entirely 4yr: ledger / Supplementary Table 3)
NAME = {'M1': 'BAS', 'M2': 'XGBoost', 'M3': 'blend'}


def run_pair(train_sem, test_sem):
    frames = {}
    removed = None
    for W in C.W_ALL:
        d, rm = R.fit_pair(EE, etl, data[W], [train_sem], test_sem)
        d = EE.add_m3(d)                      # blend percentile ranks computed within the test semester (single fold)
        frames[f'W{W}'] = d
        removed = rm
    ref = frames['W12']
    assert all((frames[w].anon_id.to_numpy() == ref.anon_id.to_numpy()).all() for w in frames)
    b = pd.DataFrame({'semester': test_sem, 'anon_id': ref.anon_id.to_numpy(), 'sem_i': 0})
    b['stu'] = pd.factorize(b.anon_id)[0]
    sc = {w: frames[w][['m1', 'm2', 'm3', 'pfail', 'qmi_none']].reset_index(drop=True) for w in frames}
    y = ref.y.to_numpy()
    f = ref.final.to_numpy(float)
    Wm = L.make_weights(b.stu.to_numpy())     # seed 20261031, B = 2000 (r5_lib defaults)
    res = L.run_rule(b, sc, y, f, 'pooled', Wm)
    return frames, res, removed, y


def summarize(train_sem, test_sem, res, y, removed):
    sel = res['sel']
    col = {n: i for i, n in enumerate(res['names'])}
    out = {'train': train_sem, 'test': test_sem, 'train_n_before_overlap_removal': len(data[8][train_sem]), 'train_overlap_removed': removed,
           'train_n_used': len(data[8][train_sem]) - removed, 'test_n': len(y), 'test_fail': int(y.sum()),
           'test_fail_rate_pct': 100 * y.mean(), 'train_track': 'F' if train_sem in FTRACK else 'S', 'test_track': 'F' if test_sem in FTRACK else 'S'}
    out['pair_type'] = 'same-track' if out['train_track'] == out['test_track'] else 'cross-track'
    for p in ('M1', 'M2', 'M3'):
        la = res['auc_d'][:, col[('W12', p)]] - res['auc_d'][:, col[('W8', p)]]
        lr = res['r_d'][:, col[('W12', p)]] - res['r_d'][:, col[('W8', p)]]
        tag = NAME[p]
        out[f'{tag}_W8_AUC'] = res['auc_pt'][('W8', p)]
        out[f'{tag}_W12_AUC'] = res['auc_pt'][('W12', p)]
        out[f'{tag}_W8_auc_loss_pt'] = res['auc_pt'][('W12', p)] - res['auc_pt'][('W8', p)]
        out[f'{tag}_W8_r_loss_pt'] = res['r_pt'][('W12', p)] - res['r_pt'][('W8', p)]
        out[f'{tag}_W8_auc_loss_lo'] = float(np.nanpercentile(la, 2.5))
        out[f'{tag}_W8_auc_loss_ucl'] = float(np.nanpercentile(la, 97.5))
        out[f'{tag}_W8_r_loss_ucl'] = float(np.nanpercentile(lr, 97.5))
        out[f'{tag}_Wstar'] = sel['wstar'][p]
    d10 = [d for d in sel['detail'] if d['pred'] == 'M3' and d['week'] == 'W10'][0]
    out['blend_W10_auc_loss_ucl'] = d10['auc_loss_ucl']
    out['blend_W10_r_loss_ucl'] = d10['r_loss_ucl']
    out['selected_policy'] = f"{sel['sel_week']}+{NAME.get(sel['sel_pred'], sel['sel_pred'])}"
    return out


# ---- E2 benchmark through the same pipeline (unrounded values of the gate G-b reading) ----------------------------------------
parts = {f'W{W}': [] for W in C.W_ALL}
for tr, te in (('1141', '1132'), ('1132', '1141')):
    for W in C.W_ALL:
        d, _ = R.fit_pair(EE, etl, data[W], [tr], te)
        parts[f'W{W}'].append(d)
e2 = {w: EE.canon(EE.add_m3(pd.concat(v, ignore_index=True))) for w, v in parts.items()}
E2B = {}
for col, tag in (('m1', 'BAS'), ('m2', 'XGBoost'), ('m3', 'blend')):
    p8, p12 = EL.point_metrics(e2['W8'], col), EL.point_metrics(e2['W12'], col)
    E2B[tag] = {'auc_loss': p12['auc'] - p8['auc'], 'r_loss': p12['r'] - p8['r']}
print('E2 benchmark (same pipeline, era-merged two folds): blend W8 point loss AUC %+.5f  r %+.5f' % (E2B['blend']['auc_loss'], E2B['blend']['r_loss']))

PAIR_FRAMES = {}
# ---- 20 pairs -------------------------------------------------------------------------------------------------------------
raw = []
for s in E1:
    for t in E1:
        if s == t:
            continue
        frames, res, removed, y = run_pair(s, t)
        PAIR_FRAMES[(s, t)] = frames
        r = summarize(s, t, res, y, removed)
        raw.append(r)
        print(f"[{time.time() - t0:4.0f}s] train {s} (n={r['train_n_used']}) -> test {t} (n={r['test_n']}, fail {r['test_fail']}) {r['pair_type']:11s} "
              f"blend W8 loss AUC {r['blend_W8_auc_loss_pt']:+.4f} (ucl {r['blend_W8_auc_loss_ucl']:.4f}) r {r['blend_W8_r_loss_pt']:+.4f}  W*={r['blend_Wstar']} | BAS loss {r['BAS_W8_auc_loss_pt']:+.4f}", flush=True)
D = pd.DataFrame(raw)
assert len(D) == 20
D.to_csv(C.RES / 'R7-1_pairs.csv', index=False, encoding='utf-8-sig')


# ---- additional (not in spec): reciprocal pairs merged, like-for-like with the two-fold Era E2 reading ----------------------------
REC = []
for i, a in enumerate(E1):
    for b in E1[i + 1:]:
        got = {}
        for w in ('W8', 'W12'):
            m = EE.canon(EE.add_m3(pd.concat([PAIR_FRAMES[(a, b)][w], PAIR_FRAMES[(b, a)][w]], ignore_index=True)))
            got[w] = EL.point_metrics(m, 'm3')
        REC.append({'pair': f'{a}<->{b}', 'type': 'same-track' if ((a in FTRACK) == (b in FTRACK)) else 'cross-track', 'n': got['W8']['n'], 'n_fail': got['W8']['n_fail'],
                    'blend_W8_auc_loss_pt': got['W12']['auc'] - got['W8']['auc'], 'blend_W8_r_loss_pt': got['W12']['r'] - got['W8']['r']})
RECD = pd.DataFrame(REC)
RECD.to_csv(C.RES / 'R7-1_reciprocal_pairs_additional.csv', index=False, encoding='utf-8-sig')


def dist_rows(sub, label):
    rows = []
    for key, nm, thr_a in (('blend_W8_auc_loss_pt', 'blend W8 AUC point loss', 0.020), ('blend_W8_r_loss_pt', 'blend W8 r point loss', 0.030)):
        v = sub[key].to_numpy()
        q1, q3 = np.percentile(v, 25), np.percentile(v, 75)
        RW.add(rows, 'R7-1', label, f'{nm}: median [IQR]', 'W8', len(v), float(np.median(v)), float(q1), float(q3), f'range {v.min():+.4f} to {v.max():+.4f}')
        RW.add(rows, 'R7-1', label, f'{nm}: range (min, max)', 'W8', len(v), float(v.min()), float(v.min()), float(v.max()), 'estimate = min; ci_lo = min; ci_hi = max')
        RW.add(rows, 'R7-1', label, f'{nm}: number of pairs beyond bound ({thr_a:.3f})', 'W8', len(v), int((v > thr_a).sum()), note='strictly greater than the bound')
    return rows


rows = []
for label, sub in (('all 20 pairs', D), ('same-track pairs (additional, not in spec)', D[D.pair_type == 'same-track']),
                   ('cross-track pairs (additional, not in spec)', D[D.pair_type == 'cross-track'])):
    rows += dist_rows(sub, label)
va, vr = D.blend_W8_auc_loss_pt.to_numpy(), D.blend_W8_r_loss_pt.to_numpy()
pub = {'auc': 0.051, 'r': 0.095}
RW.add(rows, 'R7-1', 'all 20 pairs', 'blend W8 AUC point loss: pairs >= +0.051 (published Era E2 value)', 'W8', 20, int((va >= 0.051).sum()), note='literal threshold')
RW.add(rows, 'R7-1', 'all 20 pairs', 'blend W8 AUC point loss: pairs >= E2 value reproduced in this build', 'W8', 20, int((va >= E2B['blend']['auc_loss']).sum()), note=f"E2 benchmark {E2B['blend']['auc_loss']:+.5f}")
RW.add(rows, 'R7-1', 'all 20 pairs', 'blend W8 r point loss: pairs >= +0.095 (published Era E2 value)', 'W8', 20, int((vr >= 0.095).sum()), note='literal threshold')
RW.add(rows, 'R7-1', 'all 20 pairs', 'blend W8 r point loss: pairs >= E2 value reproduced in this build', 'W8', 20, int((vr >= E2B['blend']['r_loss']).sum()), note=f"E2 benchmark {E2B['blend']['r_loss']:+.5f}")
for label, sub in (('reciprocal pairs merged, all 10 (additional, not in spec)', RECD), ('reciprocal pairs merged, cross-track 6 (additional, not in spec)', RECD[RECD.type == 'cross-track'])):
    for key, nm in (('blend_W8_auc_loss_pt', 'blend W8 AUC point loss'), ('blend_W8_r_loss_pt', 'blend W8 r point loss')):
        v = sub[key].to_numpy()
        E2v = E2B['blend']['auc_loss'] if 'auc' in key else E2B['blend']['r_loss']
        RW.add(rows, 'R7-1', label, f'{nm}: median [IQR]', 'W8', len(v), float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75)),
               f'range {v.min():+.4f} to {v.max():+.4f}; merged = both directions pooled, as the Era E2 reading')
        RW.add(rows, 'R7-1', label, f'{nm}: pairs >= Era E2 value of this build', 'W8', len(v), int((v >= E2v).sum()), note=f'E2 benchmark {E2v:+.5f}')
RW.add(rows, 'R7-1', 'Era E2 benchmark (two folds, same pipeline)', 'blend W8 AUC point loss', 'W8', 745, E2B['blend']['auc_loss'], note='published +0.051')
RW.add(rows, 'R7-1', 'Era E2 benchmark (two folds, same pipeline)', 'blend W8 r point loss', 'W8', 745, E2B['blend']['r_loss'], note='published +0.095')
for w in ('W6', 'W8', 'W10', 'W12'):
    RW.add(rows, 'R7-1', 'all 20 pairs', f'blend selected week = {w}: number of pairs', w, 20, int((D.blend_Wstar == w).sum()))
RW.add(rows, 'R7-1', 'all 20 pairs', 'training records used: min / max', '', 20, float(D.train_n_used.min()), float(D.train_n_used.min()), float(D.train_n_used.max()), 'estimate = min; ci_lo = min; ci_hi = max; Era E2 folds: 223 to 496')
RW.add(rows, 'R7-1', 'all 20 pairs', 'BAS-alone W8 AUC point loss: median [IQR]', 'W8', 20, float(D.BAS_W8_auc_loss_pt.median()), float(D.BAS_W8_auc_loss_pt.quantile(.25)), float(D.BAS_W8_auc_loss_pt.quantile(.75)),
       f"range {D.BAS_W8_auc_loss_pt.min():+.4f} to {D.BAS_W8_auc_loss_pt.max():+.4f}; Era E2 two folds {E2B['BAS']['auc_loss']:+.4f}")
for i, r in D.iterrows():
    RW.add(rows, 'R7-1 pair', f"train {r.train} -> test {r.test}", 'blend W8 AUC point loss', 'W8', int(r.test_n), r.blend_W8_auc_loss_pt, r.blend_W8_auc_loss_lo, r.blend_W8_auc_loss_ucl,
           f"train n {r.train_n_used}; test fail {r.test_fail}; {r.pair_type}; blend W* {r.blend_Wstar}; r loss {r.blend_W8_r_loss_pt:+.4f}; BAS AUC loss {r.BAS_W8_auc_loss_pt:+.4f}; ci = 2.5th/97.5th percentile of the loss")
RW.write('R7-1', rows)
pd.set_option('display.width', 250, 'display.max_columns', 40)
print()
print(D[['train', 'test', 'pair_type', 'train_n_used', 'test_n', 'test_fail', 'blend_W8_auc_loss_pt', 'blend_W8_auc_loss_ucl', 'blend_W8_r_loss_pt', 'blend_Wstar', 'BAS_W8_auc_loss_pt', 'selected_policy']].round(4).to_string(index=False))
for label, sub in (('all', D), ('same-track', D[D.pair_type == 'same-track']), ('cross-track', D[D.pair_type == 'cross-track'])):
    for key in ('blend_W8_auc_loss_pt', 'blend_W8_r_loss_pt'):
        v = sub[key].to_numpy()
        print(f'{label:12s} {key:22s} n={len(v)} median {np.median(v):+.4f} IQR [{np.percentile(v,25):+.4f}, {np.percentile(v,75):+.4f}] range [{v.min():+.4f}, {v.max():+.4f}]')
print()
print(RECD.round(4).to_string(index=False))
for label, sub in (('merged all', RECD), ('merged cross-track', RECD[RECD.type == 'cross-track'])):
    for key in ('blend_W8_auc_loss_pt', 'blend_W8_r_loss_pt'):
        v = sub[key].to_numpy()
        print(f'{label:20s} {key:22s} n={len(v)} median {np.median(v):+.4f} IQR [{np.percentile(v,25):+.4f}, {np.percentile(v,75):+.4f}] range [{v.min():+.4f}, {v.max():+.4f}]')
print(f'total {time.time() - t0:.0f}s')
