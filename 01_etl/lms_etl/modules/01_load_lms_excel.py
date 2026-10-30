"""
lms_etl/modules/01_load_lms_excel.py
讀取 LMS 匯出 Excel，回傳 dict of DataFrames。
支援 .xlsx（優先 calamine，未安裝時自動降級 openpyxl）與 .xls（xlrd）。
"""

import os
import importlib.util as _ilu
import pandas as pd

# [PERF-FIX 穿透式審查 0914] .xlsx 讀取效能瓶頸實測：對 111-1 二技學習紀錄.xlsx
# （10MB／9個sheet／約23萬列）用 cProfile 剖析全流程，pandas.read_excel()
# （openpyxl engine，經XML/pyexpat逐cell解析）占掉47%總執行時間，是單一最大
# 瓶頸。改用 python-calamine engine（Rust實作，非XML逐cell解析）後同一檔案
# 讀取時間 19.7s→3.5s（5.6倍）。正確性驗證：用全部13個真實歷史LMS檔案
# （111-1~114-2，1,288萬格cell、9個sheet逐一比對，dtype=str與production
# 呼叫方式完全一致）逐cell比對兩個engine的輸出，僅9格不同，且全部集中在
# 「討論記錄」sheet的自由文字留言欄（col=5），差異僅為該欄位開頭/結尾多餘的
# \n或\xa0空白字元（openpyxl保留、calamine裁掉），純屬未被任何下游計算引用
# 的自由文字欄位（全庫搜尋 sheets.get("discussion") 或 sheets["discussion"]
# 零命中），對任何成績/學習行為計算零影響。
# 因 python-calamine 是本專案原本沒有的新相依套件，改為「偵測到就用、
# 沒偵測到就自動退回原本的openpyxl」，避免使用者環境未安裝時直接壞掉；
# 若要吃到這次的效能改善，需另外 pip install python-calamine。
_HAS_CALAMINE = _ilu.find_spec("python_calamine") is not None

# ── Sheet 名稱映射（LMS 顯示名稱 → 內部 key）──
SHEET_MAP = {
    "學習紀錄統計":       "summary",
    "活動紀錄":           "activity",
    "教材閱讀紀錄":       "reading_log",
    "教材閱讀次數及時間":  "reading_stats",
    "作業繳交記錄":       "homework",
    "測驗紀錄":           "quiz",
    "問卷填寫紀錄":       "survey",
    "討論記錄":           "discussion",
    "影片問題答題記錄":    "video_quiz",
}

REQUIRED_SHEETS = {"summary", "activity", "reading_stats", "quiz"}


def _detect_header_row(df_raw: pd.DataFrame, scan_rows: int = 30) -> int:
    """
    在 raw DataFrame 中找含「帳號」的列索引作為 header row。
    找不到時回傳 0。

    v1.1 [PERF-01] 效能修正：原本對整個 df_raw（activity/reading_log等大型
    sheet實測可達10萬+列）做 `.astype(str).apply(..., axis=1)` 逐列掃描，
    實測單一sheet（活動紀錄，115,868列）耗時近10秒；header列在真實資料
    （8學期、9種sheet類型全數檢查）中實際上都落在前8列以內（LMS匯出格式
    固定為「標題列＋header列」），故改為只掃描前 scan_rows 列（預設30，
    已留3倍以上安全邊界）。若前 scan_rows 列真的找不到（極端邊界情況），
    才退回掃描全表──行為與修正前完全等價，只是絕大多數情況下大幅加速。
    """
    head = df_raw.head(scan_rows)
    mask = head.astype(str).apply(lambda row: row.str.contains("帳號").any(), axis=1)
    matched = mask[mask].index
    if len(matched):
        return int(matched[0])
    # 安全網：前 scan_rows 列都找不到時，退回掃描全表（維持與修正前相同的
    # 正確性保證，只是這個分支在真實資料中從未觸發過）
    mask_full = df_raw.astype(str).apply(lambda row: row.str.contains("帳號").any(), axis=1)
    matched_full = mask_full[mask_full].index
    return int(matched_full[0]) if len(matched_full) else 0


def _extract_course_header(df_raw: pd.DataFrame, sheet_name: str = "") -> "tuple[str, str]":
    """
    v6新增（22-7規劃書Phase 1，問題1）；v7.8修正：取 raw DataFrame 左上角
    儲存格文字（通常是「{課程代碼} {課程名稱} - {sheet名}」，如
    "1141112300055011R0 微生物及免疫學 - 測驗紀錄"），解析出：
      - course_id   : 第一個空白前的 token，作為該檔案的課程/班級代碼
                       （_section_id）。
      - course_name : 課程代碼之後、「 - {sheet名}」之前的課程名稱。

    v7.8根因修正：舊版只用 cell.split()[0] 取課程代碼，課程名稱部分被
    直接捨棄——這是 course_name 於 behavior.json / at_risk_profile.json
    恆為空字串的根因之一（另一根因見 lms_etl.py::_merge_lms_files() 的
    course_info_by_semester 收集邏輯）。改為同時解析並回傳兩者。

    這是校方系統自動產生的班級/場次代碼，同一學期若有多個班級（如正規班、
    重修班）各自匯出的xlsx，代碼會不同，可用來偵測「同學期但不同行事曆」
    的情況（見 lms_etl.py::validate_semester_calendar()）。

    找不到或格式不符時回傳 ("", "")——呼叫端須容忍空值，自動退回「無section
    區分／無課程名稱」的安全路徑，不會中斷主流程。
    """
    try:
        raw_cell = df_raw.iloc[0, 0]
    except (IndexError, KeyError):
        return "", ""
    # 用 pd.isna() 而非字串比對（如 cell != "nan"）判斷空值：儲存格若為
    # 真正的 None（而非 NaN float），str(None) 會變成字面字串 "None"，
    # 舊式字串比對會誤判為非空值而繼續解析，回傳無意義的 "None" 當作
    # course_id。pd.isna() 能正確涵蓋 None / NaN / NaT 等各種缺值型態。
    if pd.isna(raw_cell):
        return "", ""
    cell = str(raw_cell)
    if not cell:
        return "", ""

    # 優先用已知的「 - {sheet名}」結尾精確去除課程名稱之後的部分，避免
    # 課程名稱本身含括號（如「(合併班)」）時被誤切；sheet_name 未提供或
    # 與結尾不吻合時，退回用最後一個" - "切割（防呆，不中斷主流程）。
    suffix = f" - {sheet_name}"
    if sheet_name and cell.endswith(suffix):
        head = cell[: -len(suffix)]
    elif " - " in cell:
        head = cell.rsplit(" - ", 1)[0]
    else:
        head = cell

    parts = head.split(maxsplit=1)
    course_id   = parts[0] if parts else ""
    course_name = parts[1].strip() if len(parts) > 1 else ""
    return course_id, course_name


def load_lms_excel(fpath: str, verbose: bool = True) -> "tuple[dict, set, str, str]":
    """
    讀取 LMS Excel 檔。

    Parameters
    ----------
    fpath   : Excel 檔路徑（.xlsx 或 .xls）
    verbose : 是否印出讀取進度

    Returns
    -------
    (sheets, missing, course_id, course_name)
        sheets      : dict[str, pd.DataFrame]，key 為 SHEET_MAP 內部名稱
        missing     : set[str]，缺失的必要 sheet key 集合
        course_id   : str，該檔案的課程/班級代碼（見 _extract_course_header()），
                      取「第一個成功解析出非空值的sheet」；全部sheet都解析不到
                      時為空字串
        course_name : str，該檔案的課程名稱（v7.8新增，見 _extract_course_header()），
                      與 course_id 取自同一個儲存格、同一時機解析
    """
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"找不到檔案：{fpath}")

    ext = os.path.splitext(fpath)[1].lower()
    if ext == ".xls":
        engine = "xlrd"
    elif _HAS_CALAMINE:
        engine = "calamine"
    else:
        engine = "openpyxl"
        # [PERF-NOTICE 規格書 0919 3.2節] 原本退回 openpyxl 時完全靜默，使用者
        # 除非主動讀原始碼註解，否則不會知道自己正在跑效能較差的路徑。改為
        # 明確印出提示；不影響既有 graceful-fallback 行為本身（見規格書第4節
        # 「不要把_HAS_CALAMINE的graceful-fallback機制拿掉」）。
        print("   ⚠️  未偵測到 python-calamine，本次使用 openpyxl 讀取 Excel（較慢）。")
        print("       如需約 26% 的整體執行時間改善，請執行：pip install python-calamine")

    if verbose:
        print(f"[01] 讀取 LMS Excel：{os.path.basename(fpath)}  (engine={engine})")

    result: dict[str, pd.DataFrame] = {}
    missing_keys: set[str] = set(REQUIRED_SHEETS)
    course_id = ""
    course_name = ""

    with pd.ExcelFile(fpath, engine=engine) as xl:
        for sheet_name in xl.sheet_names:
            internal_key = SHEET_MAP.get(sheet_name, sheet_name)

            # 單次讀取 raw，偵測 header 後直接 slice，避免重複 IO
            df_raw = pd.read_excel(xl, sheet_name=sheet_name, header=None, dtype=str)
            header_row = _detect_header_row(df_raw)

            if not course_id:
                course_id, course_name = _extract_course_header(df_raw, sheet_name)

            df = df_raw.iloc[header_row + 1:].copy()
            df.columns = df_raw.iloc[header_row].tolist()

            # 清理欄位名稱
            df.columns = (
                pd.Index(df.columns)
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", "", regex=True)
            )

            # 清理儲存格：strip 字串，將 "nan" 還原為 NaN
            df = df.apply(
                lambda col: col.str.strip().replace("nan", pd.NA)
                if col.dtype == object else col
            )

            # 移除全空白列
            df = df.dropna(how="all").reset_index(drop=True)

            result[internal_key] = df
            missing_keys.discard(internal_key)

            if verbose:
                print(f"   [{sheet_name}] → key='{internal_key}', {len(df)} 列")

    # 警告缺失必要 Sheet，但不污染回傳 dict
    for key in missing_keys:
        print(f"   ⚠️  必要 Sheet 缺失：'{key}'（請確認 SHEET_MAP）")

    if verbose:
        print(f"[01] 完成，共載入 {len(result)} 個 Sheet"
              + (f"，課程代碼：{course_id}" if course_id else "")
              + (f"，課程名稱：{course_name}" if course_name else "") + "\n")

    return result, missing_keys, course_id, course_name


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法：python 01_load_lms_excel.py <xlsx路徑>")
        sys.exit(1)

    sheets, missing, course_id, course_name = load_lms_excel(sys.argv[1])
    print(f"課程代碼：{course_id}　課程名稱：{course_name}")
    for key, df in sheets.items():
        print(f"  {key}: {df.shape}  columns={list(df.columns[:5])}")
    if missing:
        print(f"缺失必要 Sheet：{missing}")
