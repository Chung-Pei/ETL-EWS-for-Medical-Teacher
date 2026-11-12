# 115-1 前瞻驗證鎖定規格（定稿，2026-11-12；待作者確認後連同 commit 清單執行鎖定）

目的：於 115-1 學期成績不存在之前，預先鎖定對「Week-10 部署政策」之前瞻評估規格。本規格不隸屬 Paper A 稿件（該稿之 holdout 為 1142）；本評估為獨立之後續驗證，結果不論方向皆依本規格報告。

本定稿與草稿（prospective_1151_lock_spec_DRAFT_20261112.md，md5 a65295524607c739e08b120228861fd8）的差異僅有四處：（1）第 4 節 Recall@20% 參照值已回源轉錄填入（草稿留空處）；（2）新增第 8 節「輸入資料（定稿時點事實）」；（3）新增第 9 節「已知限制」；（4）附錄已填入輸入檔案清單與 md5、參照值轉錄出處與定稿日期。第 1–7 節之政策、隊列、outcome、指標分區、區間與程序文字與草稿逐字相同，未作任何實質變更。

## 1. 待驗證政策（凍結，不調整）

- 政策：期中後第一份行為名單（本課程 115-1 為正規行事曆、期中考於 Week 9，故為 **Week 10** 名單）。
- 分數：percentile blend（M3′）＝ BAS（M1′）與 XGBoost（M2′）之百分位平均，**僅用行為特徵**（20 項行為特徵＋1 項學制指標；特徵定義、quiz 計數口徑、training-fold median imputation 與開發管線相同）。
- BAS 權重凍結：quiz-mastery 0.462、passive-learning (reversed) 0.308、log quiz attempts 0.231；標準化常數以開發七學期為訓練資料計算。
- XGBoost 凍結設定：500 trees、max depth 3、learning rate 0.1、subsample 0.9、column sampling 0.8、五種子 [42, 43, 44, 45, 46] 平均；**以開發七學期（1111–1141）全體為訓練集**擬合，一次適用於 115-1（部署口徑：新學期以既有模型計分，不重訓於 115-1）。
- 百分位：**115-1 學期內**計算（deployment 定義）。
- 特徵窗口：至 Week 10 結束之完整窗口（與開發評估同口徑）。
- 名單大小：top 10%／20%／30%／40% 全數報告；主要名單為 top 20%。

## 2. 隊列與排除（先於計分寫定）

- 納入：115-1 本課程全部可連結 LMS 且 outcome 可計算之紀錄（比照 1142 holdout 建構）。
- 正規行事曆班別為主要分析集；若 115-1 含壓縮行事曆班別（如重修班），該班別**另列**、不併入主要分析。
- 排除僅限 outcome 無法計算者，逐筆記錄原因。

## 3. Outcome

- 主要 outcome：成績簿「學期成績」欄 <60 為不及格（含該學期實際實施之平台加分，如有，比照主分析定義）。
- 預先承諾之敏感度分析：bonus-free outcome（以期中／期末原始分重建）重算主要指標一次。

## 4. 主要指標與事前分區（鎖定值）

- 主要指標：(i) AUC（115-1 學期內）；(ii) Recall@20%。
- 參照值（開發集部署口徑，凍結）：AUC 參照 **0.735**（Supplementary Table 13 之 within-semester 加權值）；Recall@20% 參照 **37.5%（0.375）**（定稿時自凍結之開發集部署口徑紀錄轉錄：Supplementary Table 6 Panel B 之 Week-10 deployment list sensitivity；轉錄出處詳附錄）。
  - AUC 出處複核註記：0.735 為部署構成（blend 百分位改於學期內計算）之學期內加權 AUC，載於 Supplementary Table 13 表註（files/PaperA_supplement_MedTeacher_v4_20261030.md，md5 f4e528ff3ce8f4c767ca8184b2aa2762，第 412 行：「The Week-10 deployment value used elsewhere in this supplement is 0.735 (95% CI 0.712–0.758)」），並與 Supplementary Table 18 Panel B 之「Blend, Week 10, deployment」列（0.735 (0.712–0.758)，同檔第 512 行）一致。注意 Table 13 本體 Blend Week 10 列之 0.737 係 pooled 百分位分數之 within-semester 指標，與部署構成之 0.735 為不同構成，執行端不得混用；本規格參照值為 **0.735**。
  - Recall@20% 構成對齊註記：37.5% 係開發集部署口徑 Week-10 名單（within-semester blend、每學期內 top 20%、名單合計 n = 534）所找回之不及格學生比例（210/560 = 0.375），與本規格對 115-1 之 Recall@20% 定義（單一學期內 top 20% 名單）構成相同。Supplementary Table 6 以「sensitivity」標示此量；依本專案定義，top-20% 名單之 sensitivity 即 Recall@20%，兩者為同一量。
- 分區（比照 1142 holdout 分區寬度）：AUC 下限＝參照 −0.05（＝**0.685**）；Recall@20% 下限＝參照 −10 個百分點（＝**27.5%**）。兩指標皆達下限＝「區內」；任一未達＝「區外」；落在下限 ±0.005／±0.5pp 之刀鋒帶（AUC [0.680, 0.690]；Recall@20% [27.0%, 28.0%]）＝「borderline」，依事實報告、不作裁決工具。
- 次要指標：Precision@20%、r（Spearman）、對 B1（班別不及格率，訓練＝開發七學期）之增量 AUC。全部報告，不挑選。

## 5. 區間與執行環境

- CI：學生層級 cluster bootstrap 於本學期內重抽（115-1 學生多為單筆紀錄時退化为紀錄層級）、B = 2,000、seed **20261112**（本規格日期，定稿寫死）。
- 環境：xgboost 3.4.1、scikit-learn／pandas 版本於執行報告中記錄；重建偏移比照 Computational Notes 慣例揭露。

## 6. 程序與時間鎖（三段）

1. **規格鎖（2026-11，Week 10 當週）**：本規格經作者確認後定稿，連同輸入檔案 md5 清單 commit 至 GitHub repo（僅規格與雜湊，不含學生資料），以 git 時間戳為鎖定證據；本地副本存 hidden_files 並記錄 md5。
2. **分數凍結（成績出現前）**：執行端依規格計算 115-1 分數與名單，輸出檔 md5 記錄於本檔附錄並回報作者端驗收；凍結後不得因任何中間結果修改規格或分數。
3. **結果銜接與評估（成績確定且 IRB 涵蓋確認後）**：115-1 不在現行輔大免審涵蓋（111(1)–114(2)）——outcome 銜接與分析須待涵蓋確認或修正案通過後執行。評估嚴格依第 4 節報告，含不利結果。

## 7. 偏離

任何偏離本規格之執行，於執行報告中逐項揭露（比照 lock manifest 偏離揭露慣例）；規格本身於鎖定後不修改，必要之修正以新版本並列保存並註明日期與理由，且不得於見到 outcome 後為之。

## 8. 輸入資料（定稿時點事實，據去識別化處理回報 2026-11-12）

- 來源：作者上傳之 115-1 原始包（115-1_wo_final_20261112.zip，md5 de311a2b3100b9ae250d536c4ac47b8f），經 IRB 去識別化程序處理後回傳；回傳包 ~/workspace/your_files/1151_behavior_deidentified_20261112.zip（md5 0d7f2175c066edf76fb9ae503c91daef），逐檔 md5 見附錄，與 Deidentification_Log_1151_20261112.txt 轉錄並經實檔複核相符。
- 隊列：共 **569 人**——二技一般 263、二技在職 210、二技重修班 40、學後護一 56；115-1 無四技檔。依第 2 節：主要分析集為正規行事曆班別（二技一般＋二技在職＋學後護一，529 人）；二技重修班（40 人，提早開課之壓縮行事曆）另列、不併入主要分析。
- 識別處理：anon_id 為隨機代碼（S＋6 位數），與 111–114 批次共用同一對照（569 人中 59 人沿用舊碼、510 人新配）；對照表僅存於去識別化處理端，不進入本規格、commit 或任何交付包。
- 成績檔完全未取用：原始包內之班級總成績檔僅經檢視，因名冊連結無必要（每一事件帳號皆在其自身 LMS 檔名冊內）而未取用任何欄位；無任何成績或分數欄位進入輸入資料。
- 事件涵蓋：全部事件迄 2026-11-12（匯出日）；學期自 2026-09-07 起算，匯出日落在 Week 10 內。學期前事件（最早 2026-07-22 起，含重修班提早開課紀錄）保留未截斷；週次切窗依本規格曆法於計分時執行，重修班提早開課之 override 另列處理。
- 欄位口徑：quiz 逐次及格狀態為 LMS 行為欄位，與歷史 pipeline 同口徑保留，非成績檔分數。

## 9. 已知限制（定稿時點寫定）

- 二技在職（210 人）之 LMS 檔為合併課殼（基礎醫學臨床應用），其測驗紀錄僅 318 列／49 人，遠低於其名冊人數；該班別之 quiz 類行為特徵涵蓋不全，計分照輸入資料執行，此限制於結果報告中必須揭露，不得於見到 outcome 後據此調整隊列或特徵。
- homework 與 video_qa 兩類紀錄於四個來源檔皆為空（僅表頭），相應特徵依開發管線口徑處理（缺失／零值口徑與開發集相同），不另設規則。
- 115-1 隊列組成與開發七學期不同（無四技、含合併課殼班別），此為前瞻驗證之既定條件，結果報告應併陳隊列組成，不得以組成差異為由事後修改分區或排除規則。

---
附錄（定稿填入，2026-11-12）：

- 輸入檔案清單與 md5（目錄 ~/workspace/your_files/1151_behavior_deidentified_20261112/；10 個 CSV，另附 README_1151.txt，md5 667e8ccaa2a1d0c6a64ff28ce2d707c2）：
  - activity.csv — a7f68195f954afe60bd14d773a13da35（125,692 列）
  - behavior_summary.csv — 4ee656bf5a196658c17633f3ea1c73c6（569 列）
  - discussion.csv — cfb9539d789e84301bff5caf1220ec0d（1 列）
  - homework.csv — 5cec72336f38244e9634b65c7db8445b（0 列，僅表頭）
  - quiz.csv — db14ba0d5fb2e02b1d2e8214a52d6985（20,283 列）
  - reading_log.csv — 5ac230a798fc2f2c8c4d899092283776（61,616 列）
  - reading_stats.csv — dcd43978d73752d1fdbd9aa8b1d085d3（29,578 列）
  - roster_labels.csv — b045d26c1353e02787be614ecc80f63d（569 列）
  - survey.csv — f15ba17a23b0d444b92fed8b25cf881b（551 列）
  - video_qa.csv — d411a2551c31675f03fa2eb443adf2b9（0 列，僅表頭）
- Recall@20% 參照轉錄值與出處：**37.5%（0.375；210/560）**。轉錄出處：files/PaperA_supplement_MedTeacher_v4_20261030.md（md5 f4e528ff3ce8f4c767ca8184b2aa2762），Supplementary Table 6，Panel B 表頭（第 201 行：「Week-10 deployment list (within-semester blend, top 20% within each semester; list n = 534)」），數值列（第 212 行：「Week-10 list sensitivity / precision | 37.5% / 39.3% |」）；同 Panel 交叉表（both 315 人〔187 不及格〕、list only 219 人〔23 不及格〕）勾稽：名單找回不及格者 187＋23＝210，210/560＝0.375，precision 210/534＝39.3% 與同列相符。
- AUC 參照轉錄值與出處：**0.735**（95% CI 0.712–0.758）。出處同上檔 Supplementary Table 13 表註（第 412 行）與 Table 18 Panel B 列（第 512 行），詳第 4 節複核註記。
- 規格定稿日期：2026-11-12。
- commit hash：待 commit 執行後回填（本定稿階段僅產出 commit 清單，不執行 commit）。
- 分數凍結檔 md5：待第二段（分數凍結）產出後回填。
