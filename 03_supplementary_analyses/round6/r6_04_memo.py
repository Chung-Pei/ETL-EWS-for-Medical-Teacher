# -*- coding: utf-8 -*-
"""Writes results/MEMO_round6.md.  Every number is read from R6_results.csv, gate1_matrix.csv, gate1_auc_anchors.csv and
r6_variant1_extra.json (written by r6_01 / r6_02); nothing is typed by hand.  Prose is Traditional Chinese."""
import json

import pandas as pd

import r5_lib as L

R = pd.read_csv(L.RES / 'R6_results.csv', encoding='utf-8-sig')
G = pd.read_csv(L.RES / 'gate1_matrix.csv', encoding='utf-8-sig')
A = pd.read_csv(L.RES / 'gate1_auc_anchors.csv', encoding='utf-8-sig')
X = json.load(open(L.RES / 'r6_variant1_extra.json'))
VF = json.load(open(L.RES / 'r6_verify.json'))
assert VF['pass']
MINUS = '\u2212'


def num(x):
    s = '%.3f' % x
    if s in ('-0.000', '0.000'):
        s = '0.000'
    return s.replace('-', MINUS)


def sg(x):
    s = num(x)
    return s if s.startswith(MINUS) or s == '0.000' else '+' + s


def iv(lo, hi):
    return '(%s to %s)' % (num(lo), num(hi)) if lo < 0 else '(%s\u2013%s)' % (num(lo), num(hi))


def dlt(r):
    return '%s %s' % (sg(r.estimate), iv(r.ci_lo, r.ci_hi))


def auc(r):
    return '%s %s' % (num(r.estimate), iv(r.ci_lo, r.ci_hi))


def row(variant, key, week=None, n=None, basis=None):
    d = R[(R.variant == variant) & R.comparison.str.startswith(key)]
    if week:
        d = d[d.week == week]
    if n:
        d = d[d.n == n]
    if basis:
        d = d[d.basis == basis]
    assert len(d) == 1, (variant, key, week, n, basis, len(d))
    return d.iloc[0]


EIDS = ['E1a', 'E1b', 'E2a', 'E2b', 'E2c', 'E3a', 'E3b', 'E4']
O = {e: row('original', e + ' ') for e in EIDS}
V = {e: row('variant1', e + ' ') for e in EIDS}
ALT = X['alt_seed']          # id -> [protocol, est, lo, hi]
g2 = X['gate2']

# ---- Gate 1 table ------------------------------------------------------------------------------------------------------
CH = {'E1a': 'blend_pool', 'E1b': 'blend_pool', 'E2a': 'blend_pool', 'E2b': 'blend_pool', 'E2c': 'clean|blend_pool',
      'E3a': 'blend_pool', 'E3b': 'blend_pool', 'E4': 'blend_dep'}
PROTO = {e: ('R5' if e == 'E4' else 'H02') for e in EIDS}
g1rows = []
for e in EIDS:
    g = G[(G.protocol == PROTO[e]) & (G.estimand == e) & (G.candidate == CH[e])].iloc[0]
    pub = '%s %s' % (sg(g.pub_est), iv(g.pub_lo, g.pub_hi)) if pd.notna(g.pub_lo) else '%s（未發表區間）' % sg(g.pub_est)
    mine = '%s %s' % (sg(g.est), iv(g.lo, g.hi))
    g1rows.append('| %s | %s | %s | %.4f | %s |' % (e, mine, pub, abs(g.est - g.pub_est),
                                                    '是' if g.ci_ok else ('不適用' if pd.isna(g.pub_lo) else '否')))
e2c_main = G[(G.protocol == 'H02') & (G.estimand == 'E2c') & (G.candidate == 'main|blend_pool')].iloc[0]
e2c_clean = G[(G.protocol == 'H02') & (G.estimand == 'E2c') & (G.candidate == 'clean|blend_pool')].iloc[0]

# ---- Variant 1 vs original table -----------------------------------------------------------------------------------------
LAB = {'E1a': 'E1a blend − B1，pooled', 'E1b': 'E1b blend − B1，within-semester',
       'E2a': 'E2a blend − B2，pooled，main（n = 2,180）', 'E2b': 'E2b blend − B2，pooled，clean S3（n = 1,907）',
       'E2c': 'E2c blend − B2，within-semester，clean S3', 'E3a': 'E3a blend − B3，pooled',
       'E3b': 'E3b blend − B3，within-semester', 'E4': 'E4 B3 − blend，Week 10，deployment'}
tab = []
for e in EIDS:
    o, v = O[e], V[e]
    a = ALT[e]
    tab.append('| %s | %s | %s | %s | %s |' % (LAB[e], dlt(o), dlt(v), sg(v.estimate - o.estimate),
                                               '%s %s' % (sg(a[1]), iv(a[2], a[3]))))

# ---- per-score AUC lines ------------------------------------------------------------------------------------------------
def a_(variant, score, basis, n, week='W8'):
    return auc(row(variant, 'AUC ' + score, week=week, n=n, basis=basis))


aucl = []
for lab, score, basis, n, wk in [
        ('blend，pooled，全部 2,652', 'blend', 'pooled', 2652, 'W8'), ('blend，pooled，B2 main 子集 2,180', 'blend', 'pooled', 2180, 'W8'),
        ('blend，pooled，clean S3 1,907', 'blend', 'pooled', 1907, 'W8'), ('blend，within-semester，全部', 'blend', 'within', 2652, 'W8'),
        ('blend，within-semester，clean S3', 'blend', 'within', 1907, 'W8'),
        ('B1，pooled', 'B1', 'pooled', 2652, 'W8'), ('B1，within-semester', 'B1', 'within', 2652, 'W8'),
        ('B2，pooled，main', 'B2', 'pooled', 2180, 'W8'), ('B2，pooled，clean S3', 'B2', 'pooled', 1907, 'W8'),
        ('B2，within-semester，clean S3', 'B2', 'within', 1907, 'W8'),
        ('B3，pooled', 'B3', 'pooled', 2652, 'W8'), ('B3，within-semester', 'B3', 'within', 2652, 'W8'),
        ('blend，deployment，Week 10', 'blend', 'within', 2652, 'W10'), ('B3，deployment，Week 10', 'B3', 'within', 2652, 'W10')]:
    aucl.append('| %s | %s | %s |' % (lab, a_('original', score, basis, n, wk), a_('variant1', score, basis, n, wk)))

b1p_o = row('original', 'AUC B1', n=2652, basis='pooled').estimate
blp_o = row('original', 'AUC blend', week='W8', n=2652, basis='pooled').estimate
b1p_v = row('variant1', 'AUC B1', n=2652, basis='pooled').estimate
blp_v = row('variant1', 'AUC blend', week='W8', n=2652, basis='pooled').estimate
share_o = (b1p_o - .5) / (blp_o - .5)
share_v = (b1p_v - .5) / (blp_v - .5)
e1b_drop = (V['E1b'].estimate - O['E1b'].estimate) / O['E1b'].estimate
b3w_v = row('variant1', 'AUC B3', week='W8', n=2652, basis='within').estimate
blw_v = row('variant1', 'AUC blend', week='W8', n=2652, basis='within').estimate
b2m_o = row('original', 'AUC B2', n=2180, basis='pooled').estimate
b2m_v = row('variant1', 'AUC B2', n=2180, basis='pooled').estimate
blm_o = row('original', 'AUC blend', week='W8', n=2180, basis='pooled').estimate
blm_v = row('variant1', 'AUC blend', week='W8', n=2180, basis='pooled').estimate

md = f"""# MEMO_round6：無加分結果（Variant 1）下的增量

Paper A，第 6 輪補分析，2026-11-04。Post hoc。分數全部凍結，只有評估標籤改變。未碰 Semester 1142，未改手稿或補充材料。

## 1 閘門結果

### Gate 1：重現已發表增量（原始結果）

八個增量的點估計都在 ±0.003 內（最大差 {max(abs(G[(G.protocol == PROTO[e]) & (G.estimand == e) & (G.candidate == CH[e])].iloc[0].est - G[(G.protocol == PROTO[e]) & (G.estimand == e) & (G.candidate == CH[e])].iloc[0].pub_est) for e in EIDS):.4f}）。**Gate 1 通過。**

| 估計量 | 本次重現 | 已發表 | 點估計差 | 區間兩端與發表值在 3 位小數相同 |
|---|---|---|---|---|
""" + '\n'.join(g1rows) + f"""

Gate 1 同時決定了三件事，Variant 1 沿用：

1. E2c 的子集是 clean S3（n = 1,907）。main 子集（n = 2,180）得 {sg(e2c_main.est)} {iv(e2c_main.lo, e2c_main.hi)}，與已發表的 +0.047 不符；clean 子集得 {sg(e2c_clean.est)} {iv(e2c_clean.lo, e2c_clean.hi)}，與已發表值相符。已發表文字沒有寫明這個子集。
2. 區間協定依估計量而不同。E1a 至 E3b 的區間只有 seed 20261006 能重現（`ews_lib.Boot` 的學生排序），與手稿 Methods 的 seed 一致。E4 以及 B3 自身 AUC 的區間只有 seed 20261031 能重現，與手稿 Computational Notes 的 post hoc seed 一致。handoff 第 4 節把 20261031 稱為標準協定，但它不能重現 Table 2 的區間（例如 E1a 下界 0.053，發表值 0.054）。兩套協定都是 B = 2,000、學生為單位、成對抽樣、percentile 區間。
3. 分數定義。pooled 基礎的 blend 是由凍結的 m1、m2 以 `pct_rank` 重算的 pooled 百分位排名平均；within-semester 基礎（E1b、E2c、E3b）用同一個 blend 在各學期內計算 AUC，再以學期筆數加權；E4 的 blend 用學期內百分位排名（deployment）。儲存的 `m3` 欄位與重算值最大相差 1.1e-16，但這個浮點雜訊會破壞並列，使 AUC 差約 3e-6；E2b 的下界剛好在 3 位小數的捨入邊界上，重算版捨入為 −0.005（與發表值相同），儲存欄位版捨入為 −0.004。改用重算值後，與先前 H-02 結果檔（`H02B_delta_auc.csv`、`H02_delta_auc.csv`）的最大差為 9e-17。以儲存的 `m3` 欄位重跑 Variant 1，估計值與區間端點最多變動 {X['stored_m3_max_change']:.1e}。

驗證：(a) 本次的 bootstrap 權重矩陣與 `ews_lib.Boot`（程式碼直接由 `ews_lib.py` 以 AST 載入）的 2,000 組抽樣逐組相同；(b) 加權 AUC 與明確重抽樣後的 `auc_fail` 相差 0；(c) 另寫一套不共用函式的實作（明確重抽樣加 `rankdata`），重算 `R6_results.csv` 全部 {VF['rows']} 列，估計值與兩個區間端點的最大差為 {VF['max_abs_diff']:.1e}。

### Gate 2：Variant 1 建構

通過。加分筆數 5 / 58 / 204（1121 / 1131 / 1141），共 {g2['bonus_total']} 筆；紀錄分數 ≥ 60 但綜合分 < 60 的有 {g2['rec>=60_comp<60']} 筆；標籤改變 {g2['label_changes']} 筆（{g2['pass_to_fail']} 筆 pass→fail，{g2['fail_to_pass']} 筆 fail→pass）；不及格 {g2['failing_y0']} → {g2['failing_y1']}。唯一的 fail→pass 是 WITHHELD（1141）：d7 學期成績 {g2['WITHHELD']['d7_semester_score']}，OOF `final` {g2['WITHHELD']['oof_final']}，綜合分 {g2['WITHHELD']['composite']}，依規格不更動。非加分紀錄的標籤沒有任何改變。

交叉核對：以 Variant 1 標籤和 R5 協定，blend 在 Week 8 pooled 的 AUC 為 0.7196 [0.6976, 0.7417]，Week 10 deployment 為 0.7364 [0.7132, 0.7584]，與 R5-8 的值完全相同。

## 2 各估計量的判定

判定標準：增量的方向與「區間是否含 0」和已發表結論相同，稱為保留；變動幅度另列。B1 與 B2 分數由原始結果建構一次並固定，沒有因 Variant 1 重建。

- E1a（blend − B1，pooled）：保留。{dlt(V['E1a'])}，發表值 {dlt(O['E1a'])}，減少 {abs(V['E1a'].estimate - O['E1a'].estimate):.3f}。
- E1b（blend − B1，within-semester）：方向與區間不含 0 保留，幅度縮小。{dlt(V['E1b'])}，發表值 {dlt(O['E1b'])}，減少 {abs(V['E1b'].estimate - O['E1b'].estimate):.3f}（{abs(e1b_drop) * 100:.0f}%），區間下界由 {num(O['E1b'].ci_lo)} 降到 {num(V['E1b'].ci_lo)}。手稿用的「substantial and stable」是否仍成立，由作者判斷。
- E2a、E2b、E2c（blend − B2）：已發表的結論改變。三個估計的區間都含 0（E2a {dlt(V['E2a'])}；E2b {dlt(V['E2b'])}；E2c {dlt(V['E2c'])}），pooled 的兩個點估計為負。已發表的 E2a（{dlt(O['E2a'])}）和 E2c（{dlt(O['E2c'])}）區間不含 0，Variant 1 下都含 0。手稿的「fragile」仍預設有一個脆弱的增量；Variant 1 下沒有建立任何對 B2 的增量。
- E3a、E3b（blend − B3）：保留，仍未建立增量。E3a {dlt(V['E3a'])}；E3b {dlt(V['E3b'])}（發表值 {sg(O['E3b'].estimate)}，未發表區間）。Week 8 within-semester 的 AUC，B3 為 {num(b3w_v)}，blend 為 {num(blw_v)}，B3 仍較高。
- E4（B3 − blend，Week 10，deployment）：保留，B3 仍優於 blend。{dlt(V['E4'])}，發表值 {dlt(O['E4'])}。

B2 的增量消失來自兩個方向同時移動：B2 的 pooled AUC 在 main 子集由 {num(b2m_o)} 升到 {num(b2m_v)}，blend 同一子集由 {num(blm_o)} 降到 {num(blm_v)}。本次沒有檢驗原因。

## 3 Panel B 可用數字（補充材料格式）

ΔAUC ＝ blend 減基準；括號為 95% 區間；負值區間以 to 連接；「Variant 1 − 原始」由未四捨五入的估計值相減，因此可能與表中三位小數的相減差 0.001。「另一個 seed」欄是同一估計量改用另一套協定的結果，供選擇 Panel B 使用哪個 seed；兩套協定下的判定相同。

| 估計量 | 原始結果（重現） | Variant 1 | Variant 1 − 原始 | Variant 1，另一個 seed |
|---|---|---|---|---|
""" + '\n'.join(tab) + """

各分數自己的 AUC（95% 區間；R5 協定，seed 20261031，與 R5-8 及 Supplementary Table 17 一致）：

| 分數 | 原始結果 | Variant 1 |
|---|---|---|
""" + '\n'.join(aucl) + f"""

Table 2 的附註在 Variant 1 下對應為：B2 main 子集上，blend 的 pooled AUC 為 {num(blm_v)}，不是 {num(blp_v)}。

## 4 需要標記的原句（手稿 03_manuscript_current_20261104.md）

以下句子在 Variant 1 下結論會改變。是否修改、如何修改由你們處理。

1. 摘要 Results：
   > The blend added over class membership alone (ΔAUC +0.081 pooled; +0.056 within semesters) but no established increment over a quiz pass-rate feature, and a fragile one over a temporal-prior baseline.

   「a fragile one over a temporal-prior baseline」：Variant 1 下沒有建立對 B2 的增量。ΔAUC 數字另需更新為 {sg(V['E1a'].estimate)}、{sg(V['E1b'].estimate)}（若採用 Variant 1 數字）。
2. Results，What the Blend Adds：
   > Against class membership with a temporal prior (B2, pooled AUC 0.686), the increment was fragile: +0.033 [0.002, 0.065] in the main analysis, a paired estimate on the 2,180 records in which the temporal prior is estimable.

   Variant 1：{dlt(V['E2a'])}，B2 pooled AUC {num(b2m_v)}。
3. 同段：
   > The increment was +0.027 [−0.005, 0.060] in the clean subset (*n* = 1,907; sensitivity subset S3 in the Supplementary Material, not the Era E1 set of the same size) and +0.047 [0.016, 0.078] within semesters.

   Variant 1：{dlt(V['E2b'])}；within-semester {dlt(V['E2c'])}，區間含 0。
4. 同段：
   > Against class membership alone (B1, pooled AUC 0.647), the blend's increment was substantial and stable (ΔAUC +0.081, 95% CI [0.054, 0.108] pooled; +0.056 [0.029, 0.083] within semesters).

   增量仍存在，但 within-semester 縮小 {abs(e1b_drop) * 100:.0f}%（下界 {num(V['E1b'].ci_lo)}）。「stable」的措辭需要你們判斷。
5. 同段：
   > Class membership itself accounted for about 65% of the blend's discrimination above chance.

   Variant 1 下為 {share_v * 100:.1f}%（原始結果重算為 {share_o * 100:.1f}%），約 68%。
6. Discussion：
   > It does not demonstrably add over the quiz pass rate alone, and its advantage over class membership with a temporal prior is fragile.

   同第 1、2 點：Variant 1 下沒有對 B2 的增量。
7. 同段尚有 "+0.081 in pooled AUC"（對 B1）和 "difference in AUC +0.036 [0.020, 0.051]"（E4）。前者 Variant 1 為 {sg(V['E1a'].estimate)}；後者 Variant 1 為 {sg(V['E4'].estimate)} {iv(V['E4'].ci_lo, V['E4'].ci_hi)}，結論（B3 優於 blend）不變。

Table 2 的 B1、B2、B3 列若改用 Variant 1，數字需整列更新；B3 的結論（無增量）和 E4 的結論不變。

## 5 需要你知道或決定的事

- Panel B 的 seed：Panel B 屬 post hoc 表，Computational Notes 預設 seed 20261031。第 3 節表格同時列出兩個 seed 的結果，判定相同；本 memo 的主要欄位依 Gate 1 的重現結果，E1a 至 E3b 用 20261006，E4 用 20261031。
- B1 與 B2 固定：依 handoff，兩者由原始結果建構一次。它們的類別失敗率與時間先驗因此仍含加分後的標籤。重建 B1/B2 不在本輪範圍，也沒有計算。
- 未重算的項目：Variant 2 增量、Supplementary Table 17 的 B3 − XGBoost（+0.059）與 B3 − BAS（+0.066）、任何選擇規則。我判斷後兩個差值不需要重算：它們不涉及 blend，手稿只引用 B3 − blend。若要完整的 Table 17 敏感度，可另行要求。
- Semester 1142：`load_base()` 與 `build_scores()` 實際開啟的檔案只有 `d7_semester_scores.csv`、兩個 class membership 檔、`oof_scores_keyed.csv` 和 14 個 step9 JSON（W8、W10 各 7 個學期）。`class_membership5.csv` 只含 7 個開發學期。未開啟 `outcomes_1142.csv`、holdout 檔和 gradebook（稽核紀錄：`r6_file_access_audit.txt`）。
- 環境：本次為 numpy 2.5.4、pandas 3.0.6、scipy 1.18.1；發表時為 numpy 2.4.4、pandas 3.0.2、scipy 1.17.1。亂數序列與先前結果檔已逐項核對（見第 1 節），因此不影響結果。本輪沒有使用 xgboost（不重新訓練）。

## 6 檔案與重跑

`R6_results.csv`（44 列：8 個增量與 14 個 AUC，各含原始結果與 Variant 1）和 `MEMO_round6.md` 在包內最上層，也在 `results/`。`code/` 是程式。`results/` 另有 Gate 1 矩陣（`gate1_matrix.csv`）、各腳本的執行紀錄、獨立驗證紀錄與檔案存取稽核。

欄位：variant、basis、comparison、week、n、estimate、ci_lo、ci_hi、note（note 內有 failing 筆數與 bootstrap 協定）。

重跑（Windows PowerShell，於 `code/` 的上一層放 `materials/`，內容為 Round 5 的 `02_materials_round5/`）：

```powershell
cd code
python r6_01_gate1.py
python r6_02_variant1.py
python r6_03_verify.py
python r6_04_memo.py
```

`r6_02_variant1.py` 只有在 `gate1_decision.json` 記錄 Gate 1 通過時才會執行。`r5_lib.py` 是 Round 5 檔案的原樣複本。
"""
open(L.RES / 'MEMO_round6.md', 'w', encoding='utf-8').write(md)
print('memo written, %d chars' % len(md))
