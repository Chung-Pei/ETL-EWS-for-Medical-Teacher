# config/

執行 `etl.py` / `lms_etl.py` 前，這個資料夾需要準備兩個檔案：

1. `salt.txt` — 去識別化用的隨機鹽值字串（用於雜湊學號產生 anon_id）。
   **請自行產生一組新的隨機字串**，勿沿用其他專案或範例中的值；一旦用於雜湊
   過真實學號，往後所有輸出都仰賴同一個值才能對應同一批 anon_id，請妥善保管、
   不要外流、不要與已雜湊資料一併散布。

2. `exam_dates.json` — 各學期校歷設定（`semester_start`、`total_weeks` 等）。
   可執行 `python lms_etl.py --generate-config --lms <你的LMS檔案路徑>` 自動掃描
   LMS資料補齊初版，仍請務必人工核對是否符合校方正式行事曆。

此目錄本身不隨模板套件提供真實鹽值或已核對過的正式行事曆設定。
