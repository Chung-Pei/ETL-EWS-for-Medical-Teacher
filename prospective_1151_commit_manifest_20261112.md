# 115-1 前瞻驗證規格鎖 — commit 清單（2026-11-12；僅清單，本階段不執行 commit）

用途：規格鎖（規格第 6 節第一段）之執行清單。作者確認定稿後，依本清單將下列檔案 commit 至 GitHub repo（https://github.com/Chung-Pei/ETL-EWS-for-Medical-Teacher），以 git 時間戳為鎖定證據。**commit 內容僅含規格與雜湊，不含任何學生資料**：輸入 CSV 本體、去識別對照表、原始上傳包一律不入 repo。

## 1. 擬 commit 檔案（2 件）

| 檔案 | md5 | 說明 |
|---|---|---|
| prospective_1151_lock_spec_FINAL_20261112.md | a4f878a8fdc517959aa0ac1017677e83 | 定稿規格本體（本地正本：hidden_files/） |
| prospective_1151_commit_manifest_20261112.md | （以本 commit 之 git blob 雜湊為準） | 本清單（含輸入雜湊與 commit 訊息草案） |

## 2. 輸入檔案雜湊（僅雜湊入 repo，資料本體不入）

來源目錄：~/workspace/your_files/1151_behavior_deidentified_20261112/（回傳包 1151_behavior_deidentified_20261112.zip，md5 0d7f2175c066edf76fb9ae503c91daef）。逐檔 md5 自 Deidentification_Log_1151_20261112.txt 轉錄，並已與實檔逐一複核相符（2026-11-12）。

| 檔案 | md5 | 列數 |
|---|---|---|
| activity.csv | a7f68195f954afe60bd14d773a13da35 | 125,692 |
| behavior_summary.csv | 4ee656bf5a196658c17633f3ea1c73c6 | 569 |
| discussion.csv | cfb9539d789e84301bff5caf1220ec0d | 1 |
| homework.csv | 5cec72336f38244e9634b65c7db8445b | 0（僅表頭） |
| quiz.csv | db14ba0d5fb2e02b1d2e8214a52d6985 | 20,283 |
| reading_log.csv | 5ac230a798fc2f2c8c4d899092283776 | 61,616 |
| reading_stats.csv | dcd43978d73752d1fdbd9aa8b1d085d3 | 29,578 |
| roster_labels.csv | b045d26c1353e02787be614ecc80f63d | 569 |
| survey.csv | f15ba17a23b0d444b92fed8b25cf881b | 551 |
| video_qa.csv | d411a2551c31675f03fa2eb443adf2b9 | 0（僅表頭） |
| README_1151.txt（附屬說明，非計分輸入） | 667e8ccaa2a1d0c6a64ff28ce2d707c2 | — |

## 3. 參照值轉錄（規格第 4 節鎖定值，附出處）

- AUC 參照 **0.735**（95% CI 0.712–0.758）——Supplementary Table 13 表註（supplement v4，md5 f4e528ff3ce8f4c767ca8184b2aa2762，第 412 行），與 Table 18 Panel B 列（第 512 行）一致。
- Recall@20% 參照 **37.5%（0.375；210/560）**——Supplementary Table 6 Panel B（同檔第 201 行表頭、第 212 行數值列；交叉表勾稽 187＋23＝210）。
- 分區下限：AUC 0.685；Recall@20% 27.5%。

## 4. commit 訊息草案

```
Lock prospective validation spec for Semester 115-1 (Week-10 deployment policy)

- Spec: prospective_1151_lock_spec_FINAL_20261112.md (md5 a4f878a8fdc517959aa0ac1017677e83)
- Locked before any 115-1 outcome exists: policy, feature window (through Week 10),
  primary metrics and zones (AUC ref 0.735, lower bound 0.685; Recall@20% ref 37.5%,
  lower bound 27.5%), CI protocol (student-cluster bootstrap, B=2000, seed 20261112).
- This commit contains the specification and input-file hashes ONLY.
  No student data are included: input CSVs remain in local storage, identified
  only by the md5 hashes listed in the manifest; the de-identification key is
  held separately and is never committed.
```

## 5. 執行後回填欄

- commit hash：（commit 後回填本清單與規格附錄）
- commit 日期時間：（同上）
- 本清單自身 md5：（commit 前計算回填第 1 節表格）
