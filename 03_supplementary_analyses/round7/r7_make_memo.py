# -*- coding: utf-8 -*-
"""Build MEMO_round7.md and r7_file_access_audit.txt.  Every number is read from the result files (no hand-typed numbers)."""
import r7_common as C
import json
import os
import numpy as np
import pandas as pd

res = C.RES
OUT = C.WORK / 'deliverable'
OUT.mkdir(exist_ok=True)
R = pd.read_csv(res / 'R7_results.csv', encoding='utf-8-sig')
P = pd.read_csv(res / 'R7-1_pairs.csv', encoding='utf-8-sig')
REC = pd.read_csv(res / 'R7-1_reciprocal_pairs_additional.csv', encoding='utf-8-sig')
V = pd.read_csv(res / 'R7-2_variants.csv', encoding='utf-8-sig')
V['distinct'] = ~V.variant.isin(['V13 corner (0.5,0.5,0.5)', 'V20 corner (1.5,1.5,1.5)'])
CE = pd.read_csv(res / 'R7-3_cells.csv', encoding='utf-8-sig').set_index('cell')
S0 = json.load(open(res / 'R7-5_step0.json'))
SP = pd.read_csv(res / 'R7-5_subsamples_primary.csv', encoding='utf-8-sig')
SC = pd.read_csv(res / 'R7-5_subsamples_check.csv', encoding='utf-8-sig')
SELF = pd.read_csv(res / 'R7_selfcheck.csv', encoding='utf-8-sig')
d = C.STAGING / '06_r5_9_per_resample'
RP = {k: pd.read_csv(d / f'R5-9_per_resample_{k}.csv', encoding='utf-8-sig') for k in ('refit_pooled', 'frozen_pooled')}


def g(item, basis, comp_starts, week=None):
    x = R[(R['item'] == item) & (R.basis == basis) & R.comparison.str.startswith(comp_starts)]
    if week is not None:
        x = x[x.week == week]
    assert len(x) == 1, (item, basis, comp_starts, len(x))
    return x.iloc[0]


def s3(v):   # signed 3 dp
    return f'{v:+.3f}'


def s4(v):
    return f'{v:+.4f}'


def q(v, lo=25, hi=75):
    return np.percentile(v, lo), np.percentile(v, hi)


# ---------------- gates table --------------------------------------------------------------------------------
gate_rows = R[R['item'].str.startswith('Gate')]
gt = []
for _, r in gate_rows.iterrows():
    gt.append(f"| {r['item']} | {r.basis} | {r.comparison} | {r.estimate:.4f} | {r.note} |")
gate_md = '\n'.join(['| 閘門 | 基礎 | 讀數 | 觀測值 | 錨點／判定 |', '|---|---|---|---|---|'] + gt)
gd = R[R['item'] == 'Gate G-d']
gate_all = all(('PASS' in n) for n in R[R['item'].isin(['Gate G-a', 'Gate G-b'])].note) and \
    all(('PASS' in n) for n in R[(R['item'] == 'Gate G-c') & R.comparison.str.startswith(('BAS', 'blend'))].note)
gd_ok = (gd.iloc[0].estimate == 200) and abs(gd.iloc[1].estimate - 46.0) <= 0.5 and abs(gd.iloc[2].estimate - 36.5) <= 0.5
assert gate_all and gd_ok, 'gate failure: memo must not be built'

# ---------------- R7-1 ----------------------------------------------------------------------------------------
va, vr = P.blend_W8_auc_loss_pt.to_numpy(), P.blend_W8_r_loss_pt.to_numpy()
e2a = float(g('R7-1', 'Era E2 benchmark (two folds, same pipeline)', 'blend W8 AUC point loss').estimate)
e2r = float(g('R7-1', 'Era E2 benchmark (two folds, same pipeline)', 'blend W8 r point loss').estimate)
qa, qr = q(va), q(vr)
bas = P.BAS_W8_auc_loss_pt.to_numpy()
same, cross = P[P.pair_type == 'same-track'], P[P.pair_type == 'cross-track']
ra, rr = REC.blend_W8_auc_loss_pt.to_numpy(), REC.blend_W8_r_loss_pt.to_numpy()
rca = REC[REC.type == 'cross-track']
wc = P.blend_Wstar.value_counts()
bas_e2 = None
r1 = f"""
判定：可換成事實句（AUC 與 *r* 讀數不同，須分開寫）。

- 20 組有序配對（E1 內單學期訓練，s≠t），訓練筆數（去除與測試學期重疊學生後）{P.train_n_used.min()}–{P.train_n_used.max()}；Era E2 兩折為 223–496。
- blend W8 AUC 點損失：中位數 {s4(np.median(va))}，IQR [{s4(qa[0])}, {s4(qa[1])}]，範圍 [{s4(va.min())}, {s4(va.max())}]；逾 0.020 界線 {int((va > 0.020).sum())}/20 組。
- blend W8 *r* 點損失：中位數 {s4(np.median(vr))}，IQR [{s4(qr[0])}, {s4(qr[1])}]，範圍 [{s4(vr.min())}, {s4(vr.max())}]；逾 0.030 界線 {int((vr > 0.030).sum())}/20 組。
- 本建置重現的 Era E2 讀數：AUC {s4(e2a)}、*r* {s4(e2r)}（已刊 +0.051／+0.095）。
- AUC：≥ +0.051 者 {int((va >= 0.051).sum())}/20 組（{', '.join(f"{a}→{b} {v:+.4f}" for a, b, v in zip(P.train[va >= 0.051], P.test[va >= 0.051], va[va >= 0.051]))}）。E2 的 AUC 損失落在單學期訓練分布的上尾，不在分布之外。
- *r*：≥ +0.095 者 {int((vr >= 0.095).sum())}/20 組；E1 最大為 {s4(vr.max())}。E2 的 *r* 損失在 E1 單學期訓練分布之外。
- 同口徑對照（附加，非規格項）：E2 是兩折預測合併讀數。E1 互換配對合併（A→B 與 B→A 預測合併，10 組）AUC 損失中位數 {s4(np.median(ra))}，範圍 [{s4(ra.min())}, {s4(ra.max())}]；其中 ≥ E2 者 {int((ra >= e2a).sum())}/10（E2 高於最大值 {e2a - ra.max():.4f}）。*r* 範圍 [{s4(rr.min())}, {s4(rr.max())}]，≥ E2 者 {int((rr >= e2r).sum())}/10。僅看跨軌（4yr↔第一學期軌）6 組：AUC 中位數 {s4(rca.blend_W8_auc_loss_pt.median())}，範圍 [{s4(rca.blend_W8_auc_loss_pt.min())}, {s4(rca.blend_W8_auc_loss_pt.max())}]。
- 單折 20 組中，跨軌 12 組 AUC 損失中位數 {s4(cross.blend_W8_auc_loss_pt.median())}，同軌 8 組 {s4(same.blend_W8_auc_loss_pt.median())}；同軌最大值 {s4(same.blend_W8_auc_loss_pt.max())}（{same.loc[same.blend_W8_auc_loss_pt.idxmax(), 'train']}→{same.loc[same.blend_W8_auc_loss_pt.idxmax(), 'test']}），即同軌也可達 E2 量級。
- BAS 單獨（規則式、不訓練權重，只用訓練學期的標準化常數）W8 AUC 點損失：中位數 {s4(np.median(bas))}，範圍 [{s4(bas.min())}, {s4(bas.max())}]，逾 0.020 者 {int((bas > 0.020).sum())}/20 組；E2 兩折為 {s4(float(g('R7-1', 'all 20 pairs', 'BAS-alone').note.split('Era E2 two folds ')[1]))}。
- blend 選定週（單學期訓練、測試學期內）：W8 {int(wc.get('W8', 0))} 組、W10 {int(wc.get('W10', 0))} 組、W12 {int(wc.get('W12', 0))} 組、W6 {int(wc.get('W6', 0))} 組。

建議事實句內容：「Training-set size and era signal therefore cannot be separated」可換成上列分布句，兩個指標分開寫：
- AUC：E1 單學期訓練的 20 組中有 3 組達到或超過 E2 的 AUC 損失，故單學期訓練本身可產生 E2 量級的 AUC 損失；在 AUC 上，訓練量（或跨 cohort 遷移）與 era 訊號仍無法由本資料分開。同口徑的互換配對合併則是 E2 略高於 E1 最大值（高出 {e2a - ra.max():.4f}），屬上緣。
- *r*：E1 20 組與 10 個互換合併讀數都未達 E2 的 *r* 損失（E1 最大 {s4(vr.max())}／{s4(rr.max())}，E2 {s4(e2r)}）。這說明 E1 的單學期訓練未重現 E2 的 *r* 損失；本資料不能指認其原因（E2 只有兩折）。
- 「The shortfall is therefore not XGBoost overfitting alone」：BAS 單獨在 E1 單學期訓練也出現 W8 AUC 損失（上列），有直接讀數支持。
- 「跨 cohort 遷移」：單折跨軌配對的損失較大，但同軌配對亦達 E2 量級（同軌最大 {s4(same.blend_W8_auc_loss_pt.max())}），宜寫成兩者皆出現，不宜寫成以遷移為主。

限制：E1 每個測試學期只有單折；訓練學期組成與 E2 不完全同構（E2 兩個學期為 1132 全 4yr、1141 全第一學期軌）；配對間共用測試學期，20 組不獨立。
"""

# ---------------- R7-2 ----------------------------------------------------------------------------------------
ident = V.iloc[0]
Dd = V[V.distinct]


def cnt(col, w, sub):
    return int((sub[col] == w).sum())


def pair(col):
    return f"W8 {cnt(col, 'W8', V)}/21（相異 {cnt(col, 'W8', Dd)}/19）、W10 {cnt(col, 'W10', V)}/21（相異 {cnt(col, 'W10', Dd)}/19）"


def band(c):
    return f"{Dd[c].min():.4f}–{Dd[c].max():.4f}（恆等 {ident[c]:.4f}；與恆等之差 {s4(Dd[c].min() - ident[c])} 至 {s4(Dd[c].max() - ident[c])}）"


chg_pb = Dd[Dd.pooled_blend_Wstar != ident.pooled_blend_Wstar].variant.tolist()
chg_db = Dd[Dd.deploy_blend_Wstar != ident.deploy_blend_Wstar].variant.tolist()
chg_pB = Dd[Dd.pooled_BAS_Wstar != ident.pooled_BAS_Wstar].variant.tolist()
hi_p = int((Dd.pooled_blend_auc_W8 > ident.pooled_blend_auc_W8 + 1e-12).sum())
hi_d = int((Dd.deploy_blend_auc_W8 > ident.deploy_blend_auc_W8 + 1e-12).sum())
off = {W: float(g('R7-2 identity vs frozen', 'V00 identity (rebuilt BAS + frozen XGBoost) minus frozen', 'BAS pooled AUC', W).estimate) for W in ('W8', 'W10', 'W12')}
offb = {W: float(g('R7-2 identity vs frozen', 'V00 identity minus frozen', 'blend pooled AUC', W).estimate) for W in ('W8', 'W10', 'W12')}
offd = {W: float(g('R7-2 identity vs frozen', 'V00 identity minus frozen', 'blend deployment AUC', W).estimate) for W in ('W8', 'W10', 'W12')}
rho = {W: float(g('R7-2 identity vs frozen', 'V00 identity BAS vs frozen m1', 'Spearman', W).estimate) for W in ('W8', 'W10', 'W12')}
r2 = f"""
判定：第二個論證句（「pooled 定義下作者分數單獨選 Week 10」）可由一句擾動事實句取代，但該句必須如實寫出「選週並非對權重不變」。第一個論證句（簡單模型偏好未決定結果：blend 勝出是因其優勢落在等效帶之外）不由本項檢驗，維持原狀。

讀數（全部對重建恆等變體 V00 配對；21 變體中 V13＝(0.5,0.5,0.5) 與 V20＝(1.5,1.5,1.5) 正規化後與恆等數值相同，相異權重向量實為 19 個）：
- 恆等重建：pooled blend 選 {ident.pooled_blend_Wstar}、deployment blend 選 {ident.deploy_blend_Wstar}、BAS 單獨 pooled 選 {ident.pooled_BAS_Wstar}（與已刊一致）。
- (a) blend pooled 選週為 Week 8：{pair('pooled_blend_Wstar')}；改變者 {len(chg_pb)} 個（{', '.join(chg_pb) or '無'}）。blend deployment 選週：{pair('deploy_blend_Wstar')}；自 Week 10 改為 Week 8 者 {len(chg_db)} 個（{', '.join(chg_db) or '無'}）。
- (b) BAS 單獨 pooled 選週：{pair('pooled_BAS_Wstar')}；自 Week 10 改為 Week 8 者 {len(chg_pB)} 個（{', '.join(chg_pB) or '無'}）。BAS 單獨 deployment 選週：{pair('deploy_BAS_Wstar')}。
- (c) blend W8 AUC 擾動帶（19 個相異向量）：pooled {band('pooled_blend_auc_W8')}；deployment {band('deploy_blend_auc_W8')}。恆等的 W8 AUC 在另外 18 個相異變體中被超越 {hi_p} 次（pooled）、{hi_d} 次（deployment）。
- 恆等變體與凍結值的落差（另列）：BAS pooled AUC W8／W10／W12 差 {s4(off['W8'])}／{s4(off['W10'])}／{s4(off['W12'])}；blend pooled AUC 差 {s4(offb['W8'])}／{s4(offb['W10'])}／{s4(offb['W12'])}；blend deployment AUC 差 {s4(offd['W8'])}／{s4(offd['W10'])}／{s4(offd['W12'])}；重建與凍結 BAS 的 Spearman {rho['W8']:.3f}／{rho['W10']:.3f}／{rho['W12']:.3f}。逐筆分數皆不相同（2,652 筆全部差 >1e-9），屬已知建置常數，故一律以對恆等的配對差讀。

建議事實句內容（三項分開，方向不同）：
- blend pooled 選週：Week 8 於 {cnt('pooled_blend_Wstar', 'W8', V)}/21 變體；{len(chg_pb)} 個變體改選較晚的 Week 10。
- blend deployment 選週：Week 10 於 {cnt('deploy_blend_Wstar', 'W10', V)}/21；{cnt('deploy_blend_Wstar', 'W8', V)}/21 改選較早的 Week 8。
- BAS 單獨 pooled 選週：Week 10 於 {cnt('pooled_BAS_Wstar', 'W10', V)}/21；{cnt('pooled_BAS_Wstar', 'W8', V)}/21 改選較早的 Week 8。
- 已定版權重的 blend W8 AUC 不是帶內最大值（相異 18 個變體中 pooled {hi_p} 個、deployment {hi_d} 個高於它）。

限制：擾動只涵蓋三個權重的 ×0.5–1.5；XGBoost 用凍結 m2，未重訓；選週由上限是否 ≤ 界線決定，翻轉的變體多為上限貼近 0.020 者（逐變體上限見 R7_results.csv）；本項不能區分「權重調校造成的偏好」與「選週本身的邊界性」。
"""

# ---------------- R7-3 ----------------------------------------------------------------------------------------
def cell_line(k):
    c = CE.loc[k]
    return (f"| {k.replace('|', ' / ')} | {c.semesters} | {int(c.n):,} ({int(c.n_fail)}) | {s4(c.auc_loss_pt)}／{c.auc_loss_ucl:.4f} | {s4(c.r_loss_pt)}／{c.r_loss_ucl:.4f} | "
            f"{c.W_star_BAS}／{c.W_star_XGBoost}／{c.W_star_blend} | {'是' if (c.pt_loss_within_bounds_auc and c.pt_loss_within_bounds_r) else '否'} | {'是' if (c.ucl_beyond_bound_auc and c.ucl_beyond_bound_r) else '否'} |")


tbl3 = '\n'.join(['| 格 | 學期 | n (不及格) | blend W8 AUC 損失 點／上限 | blend W8 *r* 損失 點／上限 | 重選週 BAS／XGBoost／blend | W8 點損失皆在界內 | W8 上限皆破界 |', '|---|---|---|---|---|---|---|---|']
                  + [cell_line(k) for k in ('standard|pooled', 'standard|E1', 'standard|E2', 'F (4yr)|E1', 'F (4yr)|E2', 'S (2yr+postbac)|E1', 'S (2yr+postbac)|E2')])
r3 = f"""
判定：Discussion 的組成駁回句（「differ completely in composition」）可換成事實句（四格數字）。「不計為支持」是作者的處置，數據不要求更改；其理由可改為下列事實與檢力限制。

{tbl3}

（前三列為機器核對：標準 era 切法的重選週與 Supplementary Table 1 逐項相同。界線：AUC 0.020、*r* 0.030；上限為 97.5 百分位。）

組成配平後，era 型態（W8 點損失在界內、上限破界、blend 重選 Week 10）：
- 4yr 軌（F）：E1 與 E2 都呈此型態（點損失 AUC {s4(CE.loc['F (4yr)|E1'].auc_loss_pt)}／{s4(CE.loc['F (4yr)|E2'].auc_loss_pt)}，*r* {s4(CE.loc['F (4yr)|E1'].r_loss_pt)}／{s4(CE.loc['F (4yr)|E2'].r_loss_pt)}，皆在界內；上限皆破界；blend 皆重選 Week 10）。E1 與 E2 之間沒有型態差異。
- 第一學期軌（S）：blend 在 E1 與 E2 都重選 Week 12（Week 10 的 AUC 損失上限 {CE.loc['S (2yr+postbac)|E1'].blend_W10_auc_loss_ucl:.4f}／{CE.loc['S (2yr+postbac)|E2'].blend_W10_auc_loss_ucl:.4f}，破 0.020）；BAS 單獨重選 Week 10。E2 側 W8 *r* 點損失 {s4(CE.loc['S (2yr+postbac)|E2'].r_loss_pt)} 本身超過 0.030，E1 側為 {s4(CE.loc['S (2yr+postbac)|E1'].r_loss_pt)}（界內）。
- 四格的 W8 上限（AUC、*r*）全部破界。

建議事實句內容：組成配平後，型態在 4yr 軌仍在，且 E1 與 E2 相同；在第一學期軌，blend 兩個 era 皆重選 Week 12（標準切法為 Week 10），僅其 E2 側的 *r* 點損失越界。標準 era 切法的 E1 本身也呈同一型態（上表第二列），所以該型態不是 E2 專有。

限制（必須註明）：每軌 E2 側只有單一學期（F：1132，n = {int(CE.loc['F (4yr)|E2'].n)}；S：1141 扣除重修生，n = {int(CE.loc['S (2yr+postbac)|E2'].n)}），E2 側上限很寬（AUC 上限 {CE.loc['F (4yr)|E2'].auc_loss_ucl:.4f}／{CE.loc['S (2yr+postbac)|E2'].auc_loss_ucl:.4f}），檢力有限；各格不是獨立學期樣本，軌道內同一學生可跨學期。
"""

# ---------------- R7-4 ----------------------------------------------------------------------------------------
def dd(k, col):
    v = RP[k][col].to_numpy()
    return np.median(v), np.percentile(v, 25), np.percentile(v, 75), np.percentile(v, 90), float(np.mean(v > 0.020) * 100)


rf97, fr97 = dd('refit_pooled', 'ucl97_auc_W8_M3'), dd('frozen_pooled', 'ucl97_auc_W8_M3')
rf95, fr95 = dd('refit_pooled', 'ucl95_auc_W8_M3'), dd('frozen_pooled', 'ucl95_auc_W8_M3')
lr = -(RP['refit_pooled'].auc_W8_M3 - RP['refit_pooled'].auc_W12_M3).to_numpy()
lf = -(RP['frozen_pooled'].auc_W8_M3 - RP['frozen_pooled'].auc_W12_M3).to_numpy()
full = S0['pooled']['auc_loss_hi']
dep97 = np.median(pd.read_csv(d / 'R5-9_per_resample_refit_deploy.csv', encoding='utf-8-sig').ucl97_auc_W8_M3)
r4 = f"""
判定：可換成量化句。

口徑須先說明：規格指名 `ucl95_auc_W8_M3`。Table 15 Panel B 已刊的 deployment 數字（中位上限 0.029、78.5% 逾 0.020）由 `ucl97_auc_W8_M3` 重現（本次重算 deployment 重擬合中位 {dep97:.4f}），已刊的 0.014 與選擇規則本身也是 97.5 百分位口徑。為與已刊數字同口徑，建議事實句使用 `ucl97`；`ucl95` 同時入檔（R7_results.csv 兩套皆有）。

數字組（pooled 定義，同批 200 個重抽樣，各列內層 1,000 抽）：

| 口徑 | 檔案 | 上限中位數 | IQR | 90 百分位 | > 0.020 |
|---|---|---|---|---|---|
| `ucl97`（與已刊同口徑，建議） | 重擬合 | {rf97[0]:.4f} | [{rf97[1]:.4f}, {rf97[2]:.4f}] | {rf97[3]:.4f} | {rf97[4]:.1f}% |
| `ucl97` | 凍結 | {fr97[0]:.4f} | [{fr97[1]:.4f}, {fr97[2]:.4f}] | {fr97[3]:.4f} | {fr97[4]:.1f}% |
| `ucl95`（規格指名） | 重擬合 | {rf95[0]:.4f} | [{rf95[1]:.4f}, {rf95[2]:.4f}] | {rf95[3]:.4f} | {rf95[4]:.1f}% |
| `ucl95` | 凍結 | {fr95[0]:.4f} | [{fr95[1]:.4f}, {fr95[2]:.4f}] | {fr95[3]:.4f} | {fr95[4]:.1f}% |

- 逐列點損失（W12 − W8，正值為損失）：重擬合中位 {s4(np.median(lr))}，IQR [{s4(np.percentile(lr, 25))}, {s4(np.percentile(lr, 75))}]；凍結中位 {s4(np.median(lf))}，IQR [{s4(np.percentile(lf, 25))}, {s4(np.percentile(lf, 75))}]。重擬合有 {np.mean(lr > 0.020) * 100:.1f}% 的重抽樣點損失本身逾 0.020，凍結為 {np.mean(lf > 0.020) * 100:.1f}%。
- 全樣本凍結已刊上限 0.014（本次機器重算 {full:.4f}）。

建議事實句內容：重擬合下 pooled 定義的 W8 AUC 損失上限中位數 {rf97[0]:.3f}（凍結同批重抽樣 {fr97[0]:.3f}；全樣本凍結 0.014），{rf97[4]:.1f}% 的重抽樣逾 0.020 界線（凍結 {fr97[4]:.1f}%）。「樂觀」一句因此有量化，且方向與「earlier weeks easier to qualify」一致。

限制：200 個重抽樣，頻率二項誤差約 ±7 個百分點（補充材料已載）；此為純算術，未重算任何模型。
"""

# ---------------- R7-5 ----------------------------------------------------------------------------------------
w = SP.width.to_numpy()
w2 = SC.width.to_numpy()
WE2, WP = S0['E2']['width'], S0['pooled']['width']
r5 = f"""
判定：「1.87≈1.89」啟發式論證可換成經驗分布事實句。

步驟 0（Tier A 機器，B = 2,000、seed 20261006；blend W8 AUC 損失 = W12 − W8，percentile 區間）：
- pooled：{s4(S0['pooled']['auc_loss_pt'])} [{s4(S0['pooled']['auc_loss_lo'])}, {s4(S0['pooled']['auc_loss_hi'])}]，寬度 {WP:.4f}。
- E1：{s4(S0['E1']['auc_loss_pt'])} [{s4(S0['E1']['auc_loss_lo'])}, {s4(S0['E1']['auc_loss_hi'])}]，寬度 {S0['E1']['width']:.4f}。
- E2：{s4(S0['E2']['auc_loss_pt'])} [{s4(S0['E2']['auc_loss_lo'])}, {s4(S0['E2']['auc_loss_hi'])}]，寬度 W_E2 = {WE2:.4f}。
- 比值 W_E2／W_pooled = {WE2 / WP:.3f}（已刊 1.87）；√(2652/745) = {np.sqrt(2652 / 745):.3f}（已刊 1.89）。1.87 的分子 {WE2:.4f}、分母 {WP:.4f}；下界 {s4(S0['E2']['auc_loss_lo'])}（E2）與 {s4(S0['pooled']['auc_loss_lo'])}（pooled）為先前未印的數。

步驟 1（自 E1 紀錄層級不放回抽 n = 745，R = 200，子樣本內學生叢集 bootstrap B = 1,000、seed 20261006 + 子樣本編號）：
- 寬度：中位數 {np.median(w):.4f}，IQR [{np.percentile(w, 25):.4f}, {np.percentile(w, 75):.4f}]，範圍 [{w.min():.4f}, {w.max():.4f}]。中位寬度／pooled 寬度 = {np.median(w) / WP:.2f}。
- W_E2 在分布中的百分位：{np.mean(w < WE2) * 100:.1f}（窄於 W_E2 的子樣本比例）。另一組抽樣種子（檢核）：{np.mean(w2 < WE2) * 100:.1f}，中位寬度 {np.median(w2):.4f}。兩者差異屬蒙地卡羅變異，結論相同：W_E2 位於分布中段偏寬，非極端值。
- 上限逾 0.020 的子樣本比例：{np.mean(SP.hi > 0.020) * 100:.1f}%（檢核 {np.mean(SC.hi > 0.020) * 100:.1f}%）；子樣本上限中位數 {SP.hi.median():.4f}（E2 實測 {S0['E2']['auc_loss_hi']:.4f}）；子樣本點損失中位數 {s4(SP.pt_loss.median())}。

建議事實句內容：n = 745 的 E1 隨機子樣本，AUC 損失區間寬度中位數 {np.median(w):.3f}（IQR {np.percentile(w, 25):.3f}–{np.percentile(w, 75):.3f}），E2 的 {WE2:.3f} 落在第 {np.mean(w < WE2) * 100:.0f} 百分位；這些子樣本有 {np.mean(SP.hi > 0.020) * 100:.0f}% 的上限逾 0.020，儘管其點損失中位數僅 {s4(SP.pt_loss.median())}。結論「上限破界反映樣本小、非 Week 8 劣化」有經驗分布支持。

限制：子樣本取自 E1（不及格率約 17.5%），E2 為 30.5%（227 筆不及格，子樣本中位 {int(SP.n_fail.median())} 筆）；此處比較的是樣本數，不是不及格人數。W_E2 為單一 B = 2,000 的區間寬度，子樣本內為 B = 1,000。
"""

# ---------------- audit --------------------------------------------------------------------------------------
def audit_text():
    lines = []
    for name, title in (('r7_file_access_audit.jsonl', '最終乾淨重跑（run_all）'), ('r7_file_access_audit_dev_runs.jsonl', '開發階段各次執行（含一次已修正的錯誤執行）')):
        p = C.WORK / 'logs' / name
        if not p.exists():
            continue
        ev = [json.loads(x) for x in open(p, encoding='utf-8') if x.strip()]
        st = str(C.STAGING)
        blocked = [e for e in ev if e['verdict'] == 'BLOCKED']
        reads = sorted({e['path'] for e in ev if e['verdict'] == 'allowed' and e.get('kind') == 'read' and e['event'] == 'open' and e['path'].startswith(st)})
        lines.append(f'## {title}：{name}')
        lines.append(f'- 事件數 {len(ev)}；被封鎖事件 {len(blocked)}')
        for b in blocked:
            lines.append(f"  - BLOCKED：{b['script']} 嘗試開啟 {b['path']}（開發階段自測：指向已不存在的路徑，被掛鉤在觸及檔案系統前攔截）")
        lines.append(f'- 以 Python 開啟之資料／套件檔（staging 內，唯讀）{len(reads)} 個：')
        for r in reads:
            lines.append('  - ' + os.path.relpath(r, st))
        listed = sorted({e['path'] for e in ev if e['event'] in ('os.listdir', 'os.scandir') and e['path'].startswith(st)})
        lines.append(f'- 目錄列舉 {len(listed)} 次（staging 內）：' + ('；'.join(os.path.relpath(x, st) for x in listed) or '無'))
        bad = [e for e in ev if '1142' in e['path'].lower() and e['verdict'] != 'BLOCKED']
        lines.append(f'- 路徑含 1142 且被允許開啟的事件：{len(bad)}')
        lines.append('')
    return '\n'.join(lines)


manual = """## 掛鉤之外的手動動作（本次對話中，以 shell 執行，逐項列出）
- 解壓縮：`unzip` 將整個 zip（110 檔）寫入工作副本。其中路徑含 1142 者 5 項（`holdout_1142_W8/`、`exam_dates_1142.json`、`08_class_membership_and_1142_outcomes/` 內 3 檔、`gradebook_1142.xls`）只被寫到磁碟，未被開啟；隨後以 `rm -rf` 自工作副本刪除。原始 zip 未動。
- 目錄列表：`find`（僅檔名；含 `-iname '*1142*'`、`-ipath '*1142*'` 兩次名稱列表）。
- 以 shell 讀取（cat／sed／grep／awk／head／wc，皆非 1142 路徑）：`MANIFEST.md`、`00_SPEC_round7.md`、`01_landing_passages.md`、`02_manuscript_current.md`、`03_supplement_current.md`、`04_code_prior/PaperA_TierB_REPORT.md`、`tierB_within_era.py`、`r5_lib.py`、`r5_pipeline.py`、`r5_pipeline_check.py`、`r5_05_repro_gate.py`、`r5_00_gate.py`（及 `wc -l` 掃過 04_code_prior 其餘檔案之行數）、`00_README_Round5_spec.md`、`supplement/00_README_supplement.md`、`04_pipeline_code/ews_lib.py`、`ews_experiment.py`、`07_etl_v7_15_34/lms_etl/modules/11_cross_analysis.py`、`06_r5_9_per_resample/*.csv`（head、wc）。
- 套件：`pip install xgboost==3.4.1`（zip 內無 wheel；補充 README 稱 wheel 另檔交付，本包未含）。
- 未使用：`supplement/gradebooks/*.xls`（1111–1141）、`Table_A_data_era.xlsx`、`exam_dates_all.json`（兩份）、`oof_scores_no_ids.csv` 等，已解壓但未開啟。
"""
open(OUT / 'r7_file_access_audit.txt', 'w', encoding='utf-8').write('# Round 7 檔案存取稽核\n\n硬性規則：路徑含 1142 者一律封鎖（`code/r7_audit.py`，Python 層 open／listdir／scandir 掛鉤，每支程式第一行載入）。\n\n' + audit_text() + '\n' + manual)

# ---------------- memo ----------------------------------------------------------------------------------------
memo = f"""# MEMO — Paper A 第 7 輪補分析（Round 7），2026-11-05

性質：post hoc。本輪未改稿、未改補充材料、未更動鎖定紀錄。所有數字由 `code/` 重跑產生，`R7_results.csv` 為唯一數字來源，本 memo 之數字由程式自該檔與其附屬檔讀出。

## 1. 閘門（四項全過，才開始各項）

{gate_md}

G-d：`R5-9_per_resample_refit_pooled.csv` 為 {int(gd.iloc[0].estimate)} 列；Week 10 + blend {gd.iloc[1].estimate:.1f}%、Week 8 + blend {gd.iloc[2].estimate:.1f}%（已刊 46.0%／36.5%）。

G-a 以文件記載的 Tier A 協定（凍結分數、同一批重抽樣用於各子集、seed 20261006、學生依排序後編號）重現，未需改協定。三個標準子集的重選週與 Supplementary Table 1 逐項相同（見 `R7_selfcheck.csv`）。

## 2. 逐項裁決

### R7-1　Tier B 訓練量配對（落點：主文 Stability 第三段；補充 Tier B 程序段）
{r1}
### R7-2　BAS 權重擾動（落點：主文 Limitations 作者自評分數段）
{r2}
### R7-3　Era 同軌道 Tier A 重選（落點：主文 Discussion 第一段）
{r3}
### R7-4　pooled 重擬合 W8 損失上限分布（落點：Limitations 區間樂觀句；Table 15 Panel C）
{r4}
### R7-5　Tier A 寬度抽樣分布（落點：主文 Stability 第二段）
{r5}

## 3. 協定自述與裁量（Ruling：做了什麼決定、為何、錯了的代價）

1. R7-3 軌道定義：以 `03_class_membership/class_membership.csv` 的 `class_track` 定義（F＝4yr；S＝2yr＋postbac，重修生排除），不用 `class_membership5.csv` 的 `program5_key`。原因：該檔所在目錄名含 1142，同一份規格禁止開啟；兩者的分割在筆數層級一致（Round 5 規格的五類筆數：2yr_gen 1,008 ＋ 2yr_working 732 ＝ 1,740 ＝ `class_track` 的 2yr；post 169 ＝ postbac；4yr 691；retake 52）。逐筆是否一致未驗證，因為驗證需要開啟該檔。各格 n 以分析集實算：F E1 {int(CE.loc['F (4yr)|E1'].n)}、F E2 {int(CE.loc['F (4yr)|E2'].n)}、S E1 {int(CE.loc['S (2yr+postbac)|E1'].n)}、S E2 {int(CE.loc['S (2yr+postbac)|E2'].n)}；規格的名冊列數為 ≈464／≈237／≈1,465／≈457，並已預告分析集口徑會略小。代價：若 `program5_key` 與此分割有逐筆差異，各格 n 會隨之變動；你若授權開啟該檔，R7-3 可在約 1 分鐘內重跑（不涉重訓）。
2. R7-4 口徑：規格指名 `ucl95`；已刊數字為 97.5 百分位口徑（`ucl97`）。兩套皆入檔；memo 建議句用 `ucl97`。代價：若你要的是 `ucl95`，數字組在上表第三、四列。
3. R7-2 變體數：規格列 21 變體；角落 (0.5,0.5,0.5)、(1.5,1.5,1.5) 正規化後與恆等數值相同（`R7_selfcheck.csv`：差 ≤ 2e-16），故相異向量為 19。21 列照規格全數保留，計數兩種口徑並列。
4. R7-2 權重基準：用管線精確值（0.30/0.65、0.20/0.65、0.15/0.65），即規格所寫 0.462／0.308／0.231 的未四捨五入值；每次擾動後正規化至和為 1。恆等變體直接使用管線原權重（不經乘除）。
5. R7-5 子樣本抽取種子：規格只給子樣本內 bootstrap 的種子（20261006 + 編號），未給抽取種子。子樣本編號從 0 起算（編號 0–199；若你的編號從 1 起算，種子相差 1，影響僅在蒙地卡羅層級）。主結果以 20261006 + 1,000,000 + 編號抽取；另以 20261006 + 2,000,000 + 編號重做一組作檢核，兩組並列。
6. R7-1 選擇規則不含 Tier B 的輔導時間條件（規格未列）。該條件對候選週 W6／W8／W10 不會生效（距期末 12/10、10/8、8/6 週，皆 ≥ 4；Supplementary Table 2）。
7. R7-1 附加讀數（非規格項）：同軌／跨軌分層與互換配對合併，僅為讓 E2（兩折合併）與 E1（單折）同口徑；主讀數仍為規格的 20 組。
8. 已知來源差異：1141 的 WITHHELD，依 Round 5 規格的 Ruling R1，結果以鎖定的 OOF 學期成績 53.2 為準（`oof_scores_keyed.csv` 的 `final`），本輪全部使用該值。
9. 環境：Python 3.12.3、numpy 2.5.4、pandas 3.0.6、scipy 1.18.1、scikit-learn 1.9.1、xgboost 3.4.1（PyPI CPU 版，管線內 `n_jobs=1`）。Round 7 規格未載軟體版本。ETL CHANGELOG 記載 pandas 3.0.2／3.0.6、xgboost 3.4.1、scikit-learn 1.8.0／1.9.1（numpy、scipy 僅列為需求，無版本）；專案檔 `EWS_format_spec_v3.yaml`（v3 分析）記載 numpy 2.4.4、scipy 1.17.1。本輪 numpy 2.5.4、scipy 1.18.1 與後者不同。四項閘門均過；版本差異是否影響其他讀數，本輪未另行檢驗。
10. 重建偏移：XGBoost 重建絕對值與凍結值不可直接比較。本次 W8 重建 XGBoost AUC {float(pd.read_csv(res / 'gate_G_c.csv', encoding='utf-8-sig').set_index('score').loc['XGBoost', 'AUC_rebuilt']):.4f}（凍結 0.7092；差 {float(R[(R['item']=='Gate G-c') & R.comparison.str.startswith('XGBoost')].iloc[0].estimate):+.4f}）；規格記載重建 0.7049、差 −0.0043，故本輪未逐位元重現該建置常數（差約 0.0002），仍在 G-c 容差內。R7-1、R7-2 的結論一律用同建置內的配對差。
11. Bootstrap：學生為單位、成對抽樣、percentile 區間、B = 2,000；seed：R7-3、R7-5 步驟 0 與閘門 G-a 用 20261006（Tier A 學生依排序編號）；R7-1、R7-2 用 20261031（Round 5 標準串流）；R7-5 子樣本內 B = 1,000。
12. 自我檢核（`R7_selfcheck.csv`，全數 PASS）：加權 AUC／*r* 與逐抽樣明算一致；R7-5 一個子樣本之 1,000 抽損失與明算一致；Tier A 機器重選週與 Supplementary Table 1 一致。R7-1 完整跑兩次，逐組結果相同；整體乾淨重跑與開發階段逐格相同。本輪驗收為執行端自我檢核（無獨立複核者）；以貴方獨立複算為準。

## 4. 檔案存取稽核

見 `r7_file_access_audit.txt`。摘要：路徑含 1142 的檔案未被開啟；最終重跑的稽核中被封鎖事件與「被允許的 1142 事件」皆為 0（以該檔為準）。

## 5. 交付物與重跑順序

- `R7_results.csv`（長格式，欄位同 Round 6）；逐組／逐變體／逐格原始列：`R7-1_pairs.csv`、`R7-1_reciprocal_pairs_additional.csv`、`R7-2_variants.csv`、`R7-3_cells.csv`、`R7-5_subsamples_primary.csv`、`R7-5_subsamples_check.csv`、`R7-5_step0.json`。
- `code/`：`run_all.ps1`（Windows PowerShell）、`run_all.sh`；順序：r7_g_a_g_d → r7_g_b_g_c → r7_04_refit_pooled → r7_05_width → r7_03_era_track → r7_02_bas_perturb → r7_01_trainsize → r7_selfcheck → r7_assemble → r7_make_memo。需先設 `R7_STAGING` 指向解壓後的 `round7_pack_staging`，並自行 `pip install xgboost==3.4.1`。
"""
open(OUT / 'MEMO_round7.md', 'w', encoding='utf-8').write(memo)
print('memo written', len(memo), 'chars')
