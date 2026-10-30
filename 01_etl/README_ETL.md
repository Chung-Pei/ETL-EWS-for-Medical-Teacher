# LA DASH ETL

護理課程學習分析儀表板（LA DASH）的後端 ETL 套件。`etl.py` 處理歷年成績
檔（`raw_data/`），`lms_etl.py` 處理 LMS 學習行為匯出檔，兩者輸出供
`docs/` 前端儀表板讀取的 JSON 資料。

## 安裝

```
pip install -r requirements.txt
```

`requirements.txt` 含 `python-calamine`：這是 `.xlsx` 讀取的主引擎（Rust
實作），單檔讀取實測較 `openpyxl` 快約 5.6 倍，對 13 個真實歷史 LMS 檔案
的完整 pipeline 快約 26%，且輸出與 `openpyxl` 逐欄位完全一致。若環境中
未安裝，程式會自動降級使用 `openpyxl`（較慢但正確性相同），並在執行時
印出提示；不會因缺少這個套件而中斷執行。

## 使用方式

```
python etl.py --raw_dir raw_data/ --out docs/data.json

python lms_etl.py \
    --lms     lms_raw/ \
    --data    docs/data/data.json \
    --config  config/exam_dates.json \
    --salt    config/salt.txt \
    --out     docs/data/
```

完整參數說明見兩支腳本檔頭的 docstring。

## 資料夾結構

```
etl.py / lms_etl.py / lms_etl/modules/   ETL 程式本體
tests/                                    單元測試（pytest）
raw_data/                                 歷年成績 .xls/.xlsx（不隨此套件
                                           提供，需自行放入）
config/                                   exam_dates.json、salt.txt 等設定
                                           （不隨此套件提供，salt.txt 請自行
                                           產生一組隨機字串，勿沿用他人專案
                                           的值）
CHANGELOG_v7.md                           版本異動紀錄
```

執行測試：`pytest tests/`
