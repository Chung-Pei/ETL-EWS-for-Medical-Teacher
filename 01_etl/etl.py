"""
etl.py — 學習數據分析儀表板前處理腳本
版本：v1.1
輸入：raw_data/ 資料夾下所有 .xls / .xlsx 檔案
輸出：docs/data.json

使用方式：
    python etl.py
    python etl.py --raw_dir 自訂資料夾 --out docs/data.json
"""

import os
import re
import sys
import json
import hashlib
import argparse
import unicodedata
from datetime import datetime
from collections import defaultdict
from functools import lru_cache

import xlrd

# [AUDIT-FIX 2026-09-10] 原本這裡是：
#   if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
#       sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# 跟lms_etl.py同一目的的_configure_utf8_stdio()比較後發現不一致（同一條
# 防護規則兩邊各自維護、卻不夠一致）：
#   1. 只保護stdout，沒有保護stderr
#   2. 沒有hasattr(stream, "reconfigure")防呆——若stream沒有這個方法會
#      直接AttributeError中斷
#   3. 沒有try/except包住reconfigure()本身——若呼叫失敗（少數stream
#      類型或環境下可能發生）會直接中斷，且是在還沒開始做任何實際工作
#      前就中斷
#   4. 條件式`if sys.stdout.encoding and ...`本身有漏洞：某些
#      被重新導向/管線接手的stream，`.encoding`會是None——`None and X`
#      短路為False，整段防護會靜默完全不執行，且不會有任何警告
# 改為比照lms_etl.py的寫法：不先檢查目前編碼是什麼，直接嘗試強制設為
# UTF-8，用hasattr+try/except把每一種可能失敗的情況都接住，失敗就靜默
# 放棄（維持原本可能有風險的編碼，不會比修正前更差），不會讓ETL在還沒
# 開始處理資料前就無故中斷。
def _configure_utf8_stdio() -> None:
    """Keep Windows CP950 consoles from crashing on Unicode output."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_configure_utf8_stdio()

# ── 路徑設定 ──────────────────────────────────────────────────
FAIL_THRESHOLD: float = 60.0   # 模組層級常數，與 06_merge_grades.PASS_THRESHOLD 對應
# v7.15.9 BUG-FIX: 排除小班 outlier 污染建議人數回歸
# 健四二F（健照系）等 4~9 人小班因 fallback 分類邏輯被錯誤歸入 2yr_gen，
# 導致拋物線頂點大幅左移至 ~14 人（正確值應為 ~40 人）。
# 護理系正常班級人數通常 ≥ 20 人，設 20 為下限可安全排除所有 outlier。
MIN_CLASS_SIZE: int = 20       # 建議人數回歸用班級人數下限
DEFAULT_RAW_DIR = "raw_data"
DEFAULT_OUT     = os.path.join("docs", "data.json")

# ── 異常標籤辭典（預編譯）──────────────────────────────────────
# 格式：(compiled_pattern, color, label)
# 長 pattern 在前，避免短 pattern 重複匹配（如「被檢舉考試作弊」須先於「考試作弊」）
EXCEPTION_DICT_COMPILED = [
    # 🔴 重大違規
    (re.compile(r"被檢舉考試作弊"),              "red",    "重大違規"),
    (re.compile(r"(?<!被檢舉)考試作弊"),         "red",    "重大違規"),
    # 🟡 學習態度異常
    (re.compile(r"未達\s*24\s*小時"),            "yellow", "學習態度異常"),
    (re.compile(r"閱讀\s*[小少於]*\s*24\s*小時"),"yellow", "學習態度異常"),
    (re.compile(r"前後測未做"),                  "yellow", "學習態度異常"),
    (re.compile(r"前測未做"),                    "yellow", "學習態度異常"),
    (re.compile(r"後測沒做"),                    "yellow", "學習態度異常"),
    (re.compile(r"筆試未達\s*21\s*分"),          "yellow", "學習態度異常"),
    (re.compile(r"筆試不及格"),                  "yellow", "學習態度異常"),
    (re.compile(r"最高\s*56\s*分計"),            "yellow", "學習態度異常"),
    (re.compile(r"期中考補考打八折"),             "yellow", "學習態度異常"),
    # 🔵 行政缺漏
    (re.compile(r"同意書未簽"),                  "blue",   "行政缺漏"),
    (re.compile(r"補同意書"),                    "blue",   "行政缺漏"),
    (re.compile(r"無簽同意書"),                  "blue",   "行政缺漏"),
    # ⚪ 特殊身份（非異常）
    (re.compile(r"小老師"),                      "gray",   "特殊身份"),
]

# 成績欄位的異常文字（缺考/重考）→ 轉為數值並標記
# 格式：(compiled_pattern, color, label, tag)
SCORE_EXCEPTIONS = [
    (re.compile(r"缺考"), "yellow", "學習態度異常", "缺考"),
    (re.compile(r"重考"), "yellow", "學習態度異常", "重考"),
]

# 學號格式：8 或 9 位數字
# v7.15.7（穿透式審查）：與 02_anonymize.py 同步「09開頭10位」手機格式學號
# （推廣中心進修教育組學生）；原規則只收 8-9 位數，會讓這類學生的成績記錄
# 整批被 is_valid_student_row() 靜默排除，不會出現在 data.json 中。
# 已用真實 raw_data 驗證：114(2)/111(2) 護四一班各有1名學生受影響。
RE_STUDENT_ID = re.compile(r"^(?:\d{8,9}|09\d{8})$")
# BUG-FIX（115(1)空白範本檔測試發現，見規劃書實作階段）：normalize_student_id("")
# 原本會 zfill(9) 補成 "000000000"，恰好符合上方 RE_STUDENT_ID 格式，導致空白
# 學號列被誤判為有效學生（已於 is_valid_student_row() 修正根因；穿透式審查
# 0913另移除了 normalize_student_id() 本身的 zfill，此路徑現在雙重不可能發生）。
# PHANTOM_STUDENT_ID 常數作為第二層獨立防線（防禦性重複檢查，比照
# 06b_clustering.py 對 edu_type=='practicum' 的既有慣例——上游已修正，這裡
# 仍額外檢查，避免日後其他程式路徑意外繞過 is_valid_student_row() 而重新
# 產生這個假學生）：
#   1) run_etl() 主流程於寫出 data.json 前，明確排除此ID的殘留記錄
#   2) 06_merge_grades.py::extract_grade_table() 於組裝 merged_df 前，
#      比照現有 practicum 排除模式再次防呆排除
PHANTOM_STUDENT_ID = "000000000"
# 末端班級代碼格式：如 "11121A 護二一Ａ"
RE_CLASS_CODE = re.compile(r"^(\d{5}[A-Za-z0-9])\s+(.+)$")
# 過濾用：時間戳（如「檢索:113.06.18 14:47」）
RE_TIMESTAMP  = re.compile(r"檢索[:：]?\d{2,3}\.\d{2}\.\d{2}")
# 純數字（實驗課尾端欄過濾用）
RE_PURE_NUM   = re.compile(r"^\d+\.?\d*$")
# 舊版 .xls mojibake 偵測
MOJIBAKE_RE   = re.compile(r"[¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿À-ÿ]")
# xlrd 浮點學號偵測（如 "312000011.0"）
RE_FLOAT_ID   = re.compile(r"^\d+\.\d+$")


# ══════════════════════════════════════════════════════════════
# 編碼修正工具
# ══════════════════════════════════════════════════════════════

def _looks_mojibake(text) -> bool:
    return bool(MOJIBAKE_RE.search(str(text or "")))


def open_workbook_with_encoding_fix(fpath: str):
    """
    開啟 Excel 檔，修正舊 .xls 中文工作表名稱/儲存格 mojibake 問題。
    回傳 (workbook, encoding_label)。
    """
    wb = xlrd.open_workbook(fpath)
    if not any(_looks_mojibake(name) for name in wb.sheet_names()):
        return wb, "default"

    for enc in ("cp950", "big5"):
        try:
            fixed = xlrd.open_workbook(fpath, encoding_override=enc)
        except Exception:
            continue
        if not any(_looks_mojibake(name) for name in fixed.sheet_names()):
            return fixed, enc

    return wb, "default-mojibake"


# ══════════════════════════════════════════════════════════════
# 工具函式
# ══════════════════════════════════════════════════════════════

def mask_student_id(sid: str) -> str:
    """學號遮蔽：前3碼 + **** + 後2碼。學號不足5碼時全部遮蔽。

    BUG-FIX（v7.15.15穿透式審查）：本函式與 lms_etl/modules/02_anonymize.py
    的同名函式本應行為一致（同一個「遮蔽學號」概念，只是歷史上各自獨立
    維護），但 02_anonymize.py 版本已於先前審查修正「先 normalize 再遮蔽，
    避免 float 格式殘留（如"141119002.0"）」，此處卻從未同步套用同一修正
    ——兩個同名函式因此實際行為不一致。目前 etl.py 內部呼叫處（
    build_student_map()）輸入皆已預先正規化，尚未實際觸發；但
    06_merge_grades.py::extract_grade_table() 明確以「相容尚未用修正後
    etl.py 重新產生的舊版 data.json」為設計前提呼叫本函式（見該處註解），
    舊版 data.json 若曾殘留未正規化學號字串當 dict key，即會重現
    02_anonymize.py 當初修正的同一問題。補上同一行 normalize，兩份實作
    恢復一致；normalize_student_id() 對已正規化字串為恆等操作，不影響
    任何現有呼叫處行為。
    """
    sid = normalize_student_id(sid)
    return (sid[:3] + "****" + sid[-2:]) if len(sid) >= 5 else "*" * len(sid)


def load_salt(config_path: str = "config/salt.txt") -> str:
    """
    讀取 salt 檔案。
    與 02_anonymize.py 邏輯完全一致：
      - 檔案不存在 → 拋出 FileNotFoundError
      - 內容為空或仍為預設值 → 拋出 ValueError

    [BUG-FIX 穿透式審查 0913] 原本這裡的嘗試編碼清單仍含 "utf-16"，但
    02_anonymize.py::load_salt() 已先修正移除（該處註解：「salt純文字
    不會用UTF-16，保留常用編碼即可」），本檔案這份是「同一段邏輯兩處
    維護、其中一處未同步」的重複實例（與下方 normalize_student_id()
    的zfill(9)問題同一類型），此處補上同步修正。
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"鹽值檔案不存在：{config_path}\n"
            "請確認 config/salt.txt 存在且路徑正確。"
        )
    for enc in ("utf-8-sig", "utf-8", "cp950"):
        try:
            with open(config_path, "r", encoding=enc) as f:
                salt = f.read().strip()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        raise ValueError(
            "config/salt.txt 編碼無法辨識（嘗試過：utf-8-sig, utf-8, cp950）。\n"
            "請以 UTF-8（無 BOM）重新儲存檔案。"
        )
    if not salt or "REPLACE_THIS" in salt:
        raise ValueError(
            "config/salt.txt 尚未設定有效 salt！\n"
            "請填入隨機字串（至少 32 字元），例如：\n"
            "  python -c \"import secrets; print(secrets.token_hex(32))\" > config/salt.txt"
        )
    return salt


def anonymize_student_id(sid: str, salt: str) -> str:
    """與 02_anonymize.py 完全一致：SHA-256(salt + sid)[:8]"""
    if not salt:
        return ""
    return hashlib.sha256((salt + sid.strip()).encode("utf-8")).hexdigest()[:8]


def normalize_student_id(raw) -> str:
    """
    統一學號格式為字串（8 或 9 位，去除 xlrd 浮點小數點）。

    [BUG-FIX 穿透式審查 0913] 原本 8 碼以下會 zfill(9) 補零，但
    02_anonymize.py::normalize_student_id() 已先移除這個 zfill（該處
    註解：避免 8 位合法學號被誤補成 9 位，導致 hash 結果跨學期不一致）
    ——本檔案這份未同步，屬於「同一段邏輯兩處維護、其中一處走鐘」。
    兩邊正規化結果必須逐字元相同，否則同一位學生在成績檔（本檔案）
    與LMS行為檔（02_anonymize.py）算出的 anon_id 會不一致，造成
    06_merge_grades.py 兩表用 (anon_id, semester) 對不上、成績欄位
    join 不到（"XX 名學生找不到對應成績"警告的可能成因之一）。
    已用全部31個歷史真實成績檔（100(1)~115(1)）逐列驗證：目前沒有任何
    一列學號在移除zfill後會落在1-7碼區間，故此修正對既有真實資料
    無任何學生被排除或改變判定的風險。
    """
    s = str(raw).strip()
    if RE_FLOAT_ID.match(s):
        try:
            s = str(int(float(s)))
        except ValueError:
            pass
    return s


def is_valid_student_row(row_vals: list) -> bool:
    """判斷是否為有效學生資料列（學號欄符合 8–9 位數字）。

    BUG-FIX（115(1)空白範本檔測試發現，見規劃書實作階段）：學號欄為空字串時，
    normalize_student_id("") 會被 zfill(9) 補成 "000000000"，恰好符合
    RE_STUDENT_ID 的 8-9 位數字格式，導致完全空白的列被誤判為有效學生列。
    歷史正式成績檔皆為已登錄資料，不會出現「已預留列但學號空白」的範本檔，
    故此缺陷從未被觸發；115(1)開課前的空白範本檔（僅表頭、學號姓名皆空）
    首次暴露此問題。修法：學號欄去除空白後為空字串時，直接判定無效，
    不進入補零正規化。
    """
    if not row_vals:
        return False
    if not str(row_vals[0]).strip():
        return False
    return bool(RE_STUDENT_ID.match(normalize_student_id(row_vals[0])))


def parse_score(val) -> tuple:
    """
    解析成績欄位值。
    回傳 (數值或None, 異常標籤列表)
    """
    if val is None or val == "":
        return None, []
    if isinstance(val, (int, float)):
        return float(val), []
    text = str(val).strip()
    if not text:
        return None, []
    try:
        return float(text), []
    except ValueError:
        pass
    for pattern, color, label, tag in SCORE_EXCEPTIONS:
        if pattern.search(text):
            return 0.0, [{"tag": tag, "color": color, "label": label}]
    return None, []


def parse_exceptions(text: str) -> list:
    """從備註文字解析異常標籤列表"""
    if not text or RE_TIMESTAMP.search(str(text)):
        return []
    return [
        {"tag": label, "color": color, "label": label}
        for pattern, color, label in EXCEPTION_DICT_COMPILED
        if pattern.search(str(text))
    ]


def _dedup_exceptions(exceptions: list) -> list:
    """異常標籤去重（保留首次出現順序）"""
    seen, result = set(), []
    for e in exceptions:
        if e["tag"] not in seen:
            seen.add(e["tag"])
            result.append(e)
    return result


def find_col(headers: list, *candidates) -> int:
    """在 header 列中找欄位索引，找不到回傳 -1。
    建立 O(N) 查找字典，消除 O(N×M) 雙層迴圈。"""
    header_map = {str(h).strip(): i for i, h in enumerate(headers)}
    for c in candidates:
        if c in header_map:
            return header_map[c]
    return -1


# 學期代碼解析 pattern（按優先序）
_RE_SEM_UNDERSCORE = re.compile(r"(\d{3})_(\d)")
_RE_SEM_PAREN      = re.compile(r"(\d{3})\((\d)\)")
_RE_SEM_PLAIN      = re.compile(r"(\d{3})")


def parse_semester_from_filename(fname: str) -> str:
    """
    從檔名解析學期代碼，支援多種格式：
      112_1__xxx.xls  → "1121"
      100(1)護二一.xls → "1001"
    """
    base = os.path.basename(fname)
    m = _RE_SEM_UNDERSCORE.match(base)
    if m:
        return m.group(1) + m.group(2)
    m = _RE_SEM_PAREN.match(base)
    if m:
        return m.group(1) + m.group(2)
    m = _RE_SEM_PLAIN.match(base)
    return (m.group(1) + "0") if m else "unknown"


def detect_sheet_type(sheet_name: str) -> str:
    """判斷分頁類型：practicum（實驗課） 或 theory（正課，含暑期重修班）"""
    return "practicum" if "實驗" in sheet_name else "theory"


def parse_class_from_cell(cell_text: str) -> tuple:
    """
    解析班級備註欄：
    "11121A 護二一Ａ" → ("11121A", "護二一Ａ")
    找不到格式則回傳 (None, None)
    """
    text = str(cell_text).strip()
    m = RE_CLASS_CODE.match(text)
    if m:
        return m.group(1), m.group(2)
    if text and not any(c.isdigit() for c in text[:3]):
        return None, text
    return None, None


def _get_trailing_cells(row_vals: list, start_col: int,
                        filter_pure_num: bool = False) -> list:
    """
    取 start_col 之後的所有非空白、非時間戳文字。
    filter_pure_num=True 時額外排除純數字（實驗課用）。
    提取為模組層級，消除 parse_theory_sheet / parse_practicum_sheet 的重複閉包。
    """
    results = []
    for c in range(start_col, len(row_vals)):
        v = str(row_vals[c]).strip()
        if not v or RE_TIMESTAMP.search(v):
            continue
        if filter_pure_num and RE_PURE_NUM.match(v):
            continue
        results.append(v)
    return results


def _col_val(row_vals: list, col: int):
    """安全取欄位值：col=-1（欄位不存在）時回傳空字串。"""
    return row_vals[col] if col != -1 else ""


# ══════════════════════════════════════════════════════════════
# 正課 / 暑期 分頁解析
# ══════════════════════════════════════════════════════════════

def parse_theory_sheet(sheet, semester: str, sheet_type: str) -> list:
    """解析正課或暑期學分分頁，回傳 list of dict（每位學生一筆記錄）"""
    if sheet.nrows < 2:
        return []

    headers = [str(sheet.cell_value(0, c)).strip() for c in range(sheet.ncols)]

    col_id      = find_col(headers, "學號")
    col_mid     = find_col(headers, "期中考")
    col_final   = find_col(headers, "期末考")
    # v7.15.7 穿透式審查（第二輪）：民國100-103年「健四二F」正課分頁用
    # 「學期平均」命名學期總分欄，非近年慣用的「學期成績」，find_col 找不到
    # 欄位時 col_sem==-1，導致整張分頁被提前 return [] 排除。
    col_sem     = find_col(headers, "學期成績", "學期平均")
    col_adj     = find_col(headers, "調整")
    col_read    = find_col(headers, "閱讀%")
    col_attend  = find_col(headers, "簽到%")
    col_readhr  = find_col(headers, "閱讀時數")

    if col_id == -1 or col_sem == -1:
        # v7.15.7 穿透式審查（第二輪）：col_id 有找到但 col_sem 找不到，代表
        # 這「看起來」是一張逐生成績表卻抓不到學期成績欄——與 BUG-1/BUG-2
        # 同一種「格式白名單覆蓋不全」問題，需要顯式警示而非靜默回傳空值，
        # 否則未來再出現新的欄名變體又會無聲無息漏記一整張分頁。
        # col_id == -1（連學號欄都沒有）通常代表這根本不是逐生資料表
        # （如「補考名單」等非制式分頁），維持靜默排除、不視為異常。
        if col_id != -1:
            print(f"     ⚠️  找到學號欄但找不到「學期成績」欄，本分頁整批排除（headers={headers}）")
        return []

    trailing_start = max(col_sem, col_adj if col_adj != -1 else 0) + 1

    records = []
    for r in range(1, sheet.nrows):
        row_vals = [sheet.cell_value(r, c) for c in range(sheet.ncols)]
        if not is_valid_student_row(row_vals):
            continue

        sid = normalize_student_id(row_vals[col_id])

        mid_score, mid_exc  = parse_score(_col_val(row_vals, col_mid))
        fin_score, fin_exc  = parse_score(_col_val(row_vals, col_final))
        sem_score, sem_exc  = parse_score(row_vals[col_sem])
        adj_val,   _        = parse_score(_col_val(row_vals, col_adj))
        read_val,  _        = parse_score(_col_val(row_vals, col_read))
        attend_val,_        = parse_score(_col_val(row_vals, col_attend))
        readhr_val,_        = parse_score(_col_val(row_vals, col_readhr))

        exceptions = mid_exc + fin_exc + sem_exc

        class_code_raw = class_name = None
        for cell_text in _get_trailing_cells(row_vals, trailing_start):
            code, name = parse_class_from_cell(cell_text)
            if name and class_name is None:
                class_code_raw, class_name = code, name
            exceptions.extend(parse_exceptions(cell_text))

        records.append({
            "student_id":     sid,
            "semester":       semester,
            "sheet_name":     sheet.name,
            "type":           sheet_type,
            "class_code_raw": class_code_raw,
            "class_name":     class_name or sheet.name,
            "midterm":        mid_score,
            "final":          fin_score,
            "semester_score": sem_score,
            "adjusted":       adj_val,
            "reading_pct":    read_val,
            "attend_pct":     attend_val,
            "reading_hours":  readhr_val,
            "exceptions":     _dedup_exceptions(exceptions),
            "is_retaker":     False,
            "delta":          None,
        })

    return records


# ══════════════════════════════════════════════════════════════
# 基礎醫學臨床應用(一)合併課微免部分分頁解析（115(1)起，見規劃書決策4／討論11）
# ══════════════════════════════════════════════════════════════

def parse_transition_micro_sheet(sheet, semester: str) -> list:
    """
    解析「基礎醫學臨床應用(一)」合併課微免部分成績分頁。與 parse_theory_sheet()
    分開獨立實作，因為欄位結構完全不同：無「學期成績」欄（改用「期末考」＝
    Test III raw分作為 semester_score，銜接歷史趨勢圖 Y 軸不中斷）、無「期中考」
    （微免部分全學期僅一次 Test III，無期中考——若沿用 parse_theory_sheet()，
    找不到「學期成績」欄會整批分頁排除，見該函式 col_sem==-1 分支）、多出
    「閱讀%」／「加權(4%)」／「最低閱讀(8h)」／「加權(30%)」等本課程特有欄位。

    欄位對應（115_1__護二一班級總成績.xls 實測欄名，逐字比對非固定index，
    因表格含視覺留白空欄，位置不保證每分頁一致）：
      閱讀%        → reading_pct（沿用既有欄位語意，與其他課程一致）
      加權(4%)     → weighted_completion_4pct
      最低閱讀(8h) → reading_min_8h_flag（校方系統無法自動產生，此欄位由
                     LA DASH 算出後供人工謄寫，見規劃書1.5節；讀取僅為
                     保留校方後續填入的既有值，非本函式的資料來源）
      期末考       → semester_score 與 final（Test III raw分，0-100）
      調整         → adjusted（沿用 parse_score() 既有數值/例外解析慣例，
                     與 parse_theory_sheet() 對「調整」欄的處理方式一致）
      加權(30%)    → weighted_examscore_30pct

    本格式所有欄位皆為具名表頭，無額外的「班級代碼備註」尾端留白區（不同於
    parse_theory_sheet() 的 trailing cells 掃描），故 class_code_raw 固定為
    None，class_name 直接採用分頁名稱（班級名稱即分頁名稱，如「日21甲」）。
    """
    if sheet.nrows < 2:
        return []

    headers = [str(sheet.cell_value(0, c)).strip() for c in range(sheet.ncols)]

    col_id     = find_col(headers, "學號")
    col_read   = find_col(headers, "閱讀%")
    col_read4  = find_col(headers, "加權(4%)")
    col_min8h  = find_col(headers, "最低閱讀(8h)")
    col_final  = find_col(headers, "期末考")
    col_adj    = find_col(headers, "調整")
    col_exam30 = find_col(headers, "加權(30%)")

    if col_id == -1 or col_final == -1:
        if col_id != -1:
            print(f"     ⚠️  找到學號欄但找不到「期末考」欄，本分頁整批排除（headers={headers}）")
        return []

    records = []
    for r in range(1, sheet.nrows):
        row_vals = [sheet.cell_value(r, c) for c in range(sheet.ncols)]
        if not is_valid_student_row(row_vals):
            continue

        sid = normalize_student_id(row_vals[col_id])

        fin_score,  fin_exc = parse_score(_col_val(row_vals, col_final))
        adj_val,    adj_exc = parse_score(_col_val(row_vals, col_adj))
        read_val,   _       = parse_score(_col_val(row_vals, col_read))
        read4_val,  _       = parse_score(_col_val(row_vals, col_read4))
        min8h_val,  _       = parse_score(_col_val(row_vals, col_min8h))
        exam30_val, _       = parse_score(_col_val(row_vals, col_exam30))

        records.append({
            "student_id":               sid,
            "semester":                 semester,
            "sheet_name":               sheet.name,
            # BUG-FIX（實作階段發現）：type 語意是「theory vs practicum」
            # 結構分類，不是課程變體標記——用它承載「transition_merged」
            # 會讓 11_cross_analysis.py／12_early_warning.py 等既有
            # edu_type=="theory" 閘門把這批學生整批擋在外面，包括討論10
            # 明確要求「不排除」的 bas_score/QMI 描述性計算。課程變體改由
            # course_track（06_merge_grades.build_scatter_data() 計算）
            # 承載，type 維持與其他正課記錄一致的 "theory"，讓這批學生
            # 正常流入所有既有 theory 閘門；12_early_warning.py 的
            # PENDING_INDEPENDENT_MODEL_COHORTS 機制才是實際排除點。
            "type":                     "theory",
            # course_track（見規劃書1.5節）：本函式只會被 run_etl() 依
            # course_track 路由呼叫（只處理合併課5班），故此處直接寫死字面
            # 值即可，不需重算——與 06_merge_grades.build_scatter_data()
            # 稍後另外計算的 course_track（供 scatter_data.json 全體記錄
            # 統一使用）是兩個獨立來源、但語意一致。這裡額外提供的目的是
            # 讓 07e_transition_micro_reading.py 這類「直接影響學生成績」
            # 的計算能在grade_records層級就近防呆比對，不必等到
            # scatter_data.json 組裝完才能篩，降低串接環節中途夾帶非本
            # 課程記錄的風險。
            "course_track":             "transition_merged_micro",
            "class_code_raw":           None,
            "class_name":               sheet.name,
            "midterm":                  None,
            "final":                    fin_score,
            "semester_score":           fin_score,
            "adjusted":                 adj_val,
            "reading_pct":              read_val,
            "attend_pct":               None,
            "reading_hours":            None,
            "weighted_completion_4pct": read4_val,
            "reading_min_8h_flag":      min8h_val,
            "weighted_examscore_30pct": exam30_val,
            "exceptions":               _dedup_exceptions(fin_exc + adj_exc),
            "is_retaker":               False,
            "delta":                    None,
        })

    return records


# ══════════════════════════════════════════════════════════════
# 實驗課分頁解析
# ══════════════════════════════════════════════════════════════

def parse_practicum_sheet(sheet, semester: str) -> list:
    """解析實驗課分頁，擷取：期中考15、期末考30、總成績100、調整"""
    if sheet.nrows < 2:
        return []

    headers = [str(sheet.cell_value(0, c)).strip() for c in range(sheet.ncols)]

    col_id    = find_col(headers, "學號")
    col_mid   = find_col(headers, "期中考15", "期中考")
    col_final = find_col(headers, "期末考30", "期末考")
    # v7.15.7 穿透式審查（第二輪）：民國100-108年（38個實驗課分頁）用
    # 「Total 110」/「Total 100」命名總分欄，非近年慣用的「總成績100」/
    # 「總成績」，find_col 找不到欄位時 col_total==-1，導致整張分頁被
    # parse_practicum_sheet() 提前 return [] 排除，8個學年的實驗課成績
    # 完全消失於 data.json（同一批學生的正課成績不受影響，因來自另一分頁）。
    col_total = find_col(headers, "總成績100", "總成績", "Total 110", "Total 100")
    col_adj   = find_col(headers, "調整")

    if col_id == -1 or col_total == -1:
        # 理由同 parse_theory_sheet：col_id 找到但 col_total 找不到時顯式警示，
        # col_id == -1（非逐生資料表）維持靜默排除。
        if col_id != -1:
            print(f"     ⚠️  找到學號欄但找不到「總成績」欄，本分頁整批排除（headers={headers}）")
        return []

    trailing_start = max(c for c in [col_total, col_adj] if c != -1) + 1

    records = []
    for r in range(1, sheet.nrows):
        row_vals = [sheet.cell_value(r, c) for c in range(sheet.ncols)]
        if not is_valid_student_row(row_vals):
            continue

        sid = normalize_student_id(row_vals[col_id])

        mid_score, mid_exc  = parse_score(_col_val(row_vals, col_mid))
        fin_score, fin_exc  = parse_score(_col_val(row_vals, col_final))
        tot_score, tot_exc  = parse_score(row_vals[col_total])
        adj_val,   _        = parse_score(_col_val(row_vals, col_adj))

        exceptions = mid_exc + fin_exc + tot_exc

        class_code_raw = class_name = None
        for cell_text in _get_trailing_cells(row_vals, trailing_start, filter_pure_num=True):
            code, name = parse_class_from_cell(cell_text)
            if name and class_name is None:
                class_code_raw, class_name = code, name
            exceptions.extend(parse_exceptions(cell_text))

        records.append({
            "student_id":     sid,
            "semester":       semester,
            "sheet_name":     sheet.name,
            "type":           "practicum",
            "class_code_raw": class_code_raw,
            "class_name":     class_name or sheet.name,
            "midterm":        mid_score,
            "final":          fin_score,
            "semester_score": tot_score,
            "adjusted":       adj_val,
            "reading_pct":    None,
            "attend_pct":     None,
            "reading_hours":  None,
            "exceptions":     _dedup_exceptions(exceptions),
            "is_retaker":     False,
            "delta":          None,
        })

    return records


# ══════════════════════════════════════════════════════════════
# 跨學期重修判定
# ══════════════════════════════════════════════════════════════

def detect_incomplete_grade_semesters(all_records: list) -> list:
    """
    Detect semesters whose final/semester grades are not settled yet.

    Some in-progress grade sheets contain a tiny placeholder score such as 7
    in the final/semester columns for a few rows. Those values must not be
    treated as real failing grades in cohort analysis.
    """
    by_sem: dict = defaultdict(list)
    for rec in all_records:
        if rec.get("type") == "theory" and rec.get("semester"):
            by_sem[str(rec["semester"])].append(rec)

    incomplete = []
    for sem, recs in by_sem.items():
        if len(recs) < 10:
            continue
        mid_count = sum(1 for r in recs if r.get("midterm") is not None)
        final_scores = [r.get("final") for r in recs if r.get("final") is not None]
        sem_scores = [r.get("semester_score") for r in recs if r.get("semester_score") is not None]
        settled_count = max(len(final_scores), len(sem_scores))
        settled_ratio = settled_count / len(recs) if recs else 0
        placeholder_like = (
            settled_count > 0 and
            all(float(v) <= 10 for v in final_scores + sem_scores)
        )

        if mid_count >= max(10, len(recs) * 0.5) and (
            settled_count == 0 or settled_ratio < 0.10 or placeholder_like
        ):
            incomplete.append(sem)

    return sorted(set(incomplete))


def nullify_incomplete_grade_scores(all_records: list, incomplete_semesters: list) -> int:
    """Clear unsettled final/semester scores so they do not count as failures."""
    incomplete_set = {str(s) for s in (incomplete_semesters or [])}
    changed = 0
    for rec in all_records:
        if rec.get("type") != "theory" or str(rec.get("semester", "")) not in incomplete_set:
            continue
        if rec.get("final") is not None or rec.get("semester_score") is not None:
            changed += 1
        rec["final"] = None
        rec["semester_score"] = None
        rec["grade_status"] = "incomplete"
    return changed


def compute_retaker_flags(all_records: list) -> list:
    """
    跨學期比對，標記重修生並計算 delta。

    修正說明（BUG-RETAKER）：
      舊邏輯以 (student_id, sheet_name, type) 為鍵，導致重修生換班後
      key 不同而無法觸發標記（實測僅覆蓋 ~8% 真實重修生）。
      另一缺陷：首修記錄也被標為 is_retaker=True，語義錯誤。

    正確邏輯：
      - 分組鍵：(student_id, type)，不含 sheet_name
        → 跨班重修仍能正確追蹤同一學生
      - 按學期升序排列後，逐筆檢查前一筆是否不及格（semester_score < 60）
        → 只有「前一學期不及格後再次選修」才標記 is_retaker=True
      - 首修記錄（i==0）恆為 False
      - delta：本次 semester_score 與首次 semester_score 之差（首修為 None）
    """
    groups: dict = defaultdict(list)
    for rec in all_records:
        groups[(rec["student_id"], rec["type"])].append(rec)

    for recs in groups.values():
        recs_sorted = sorted(recs, key=lambda x: x["semester"])
        
        # Filter out exact duplicate records in the same semester to avoid false retaker flags
        unique_recs = []
        seen_sems = set()
        for rec in recs_sorted:
            if rec["semester"] not in seen_sems:
                unique_recs.append(rec)
                seen_sems.add(rec["semester"])
        
        # Find the first valid score for delta calculation
        first_valid_score = None
        first_valid_rec   = None
        for rec in unique_recs:
            if rec["semester_score"] is not None:
                first_valid_score = rec["semester_score"]
                first_valid_rec   = rec
                break

        for i, rec in enumerate(recs_sorted):
            # To handle duplicates in same semester, we check if there's an EARLIER semester
            earlier_recs = [r for r in unique_recs if r["semester"] < rec["semester"]]
            
            if not earlier_recs:
                rec["is_retaker"] = False
                rec["delta"]      = None
            else:
                prev_rec = earlier_recs[-1]
                prev_score = prev_rec["semester_score"]
                # Retaker only when the previous settled score is a confirmed fail.
                # Unsettled/in-progress semesters have None and must not be counted as fail.
                rec["is_retaker"] = prev_score is not None and prev_score < FAIL_THRESHOLD
                # BUG-BASELINE-SELF-DELTA FIX（0916 穿透式審查，真實資料驗證命中22名學生）：
                # 當該生「時序上最早」的記錄成績為 None（休退、缺考等），first_valid_score
                # 會落在後面某一筆「earlier_recs 非空」的記錄上——若不特別排除，該筆會拿自己
                # 的分數跟自己比（rec is first_valid_rec），算出 delta=0.0 而非 None，被誤判成
                # 一筆「有效比較、持平」的重修紀錄，實際上它才是基準本身，不該被計入任何比較。
                # 與前端 main.js::_withMetricScores() 的 isFirst 判斷（BUG-SHEET-SCOPE-0916）
                # 對齊：基準記錄本身一律 delta=None，不論它是否恰好落在 earlier_recs 非空的分支。
                is_baseline_rec = rec is first_valid_rec
                rec["delta"]      = (
                    None if is_baseline_rec else
                    round(rec["semester_score"] - first_valid_score, 2)
                    if rec["semester_score"] is not None and first_valid_score is not None
                    else None
                )

    return all_records


# ══════════════════════════════════════════════════════════════
# 彙整 class_summary
# ══════════════════════════════════════════════════════════════

def compute_class_summary(all_records: list) -> dict:
    """
    計算各學期各班的統計摘要。
    單趟掃描同時累計「全體」與「排除重修生（_nr）」兩套統計，
    讓前端「包含/不含重修生」開關可直接讀取 ETL 預算欄位，
    無須在前端逐筆遍歷 DATA.students。
    _nr 欄位語義：is_retaker=False 的首修生記錄統計。
    """
    groups: dict = defaultdict(list)
    for rec in all_records:
        groups[f"{rec['semester']}_{rec['sheet_name']}_{rec['type']}"].append(rec)

    summary = {}
    for key, recs in groups.items():
        # ── 全體（含重修生）────────────────────────────────────
        scores, mids, fins = [], [], []
        dist = [0] * 11
        pass_count = retaker_count = 0
        # ── 排除重修生（_nr = no retaker）─────────────────────
        scores_nr, mids_nr, fins_nr = [], [], []
        dist_nr = [0] * 11
        pass_count_nr = 0
        count_nr_recs = 0           # 首修記錄數（含無成績），與 count 口徑一致

        for r in recs:
            ss  = r["semester_score"]
            mid = r["midterm"]
            fin = r["final"]
            is_ret = r["is_retaker"]

            # ── 全體累計 ──
            if ss is not None:
                scores.append(ss)
                dist[max(0, min(int(ss // 10), 10))] += 1
                if ss >= FAIL_THRESHOLD:
                    pass_count += 1
            if mid is not None:
                mids.append(mid)
            if fin is not None:
                fins.append(fin)
            if is_ret:
                retaker_count += 1

            # ── 排除重修生累計 ──
            if not is_ret:
                count_nr_recs += 1          # 首修記錄數（含無成績），與 count 口徑一致
                if ss is not None:
                    scores_nr.append(ss)
                    dist_nr[max(0, min(int(ss // 10), 10))] += 1
                    if ss >= FAIL_THRESHOLD:
                        pass_count_nr += 1
                if mid is not None:
                    mids_nr.append(mid)
                if fin is not None:
                    fins_nr.append(fin)

        n    = len(scores)
        n_nr = len(scores_nr)
        summary[key] = {
            # ── 識別欄位 ──────────────────────────────────────
            "semester":                recs[0]["semester"],
            "sheet_name":              recs[0]["sheet_name"],
            "type":                    recs[0]["type"],
            # ── 全體統計（含重修生）──────────────────────────
            "count":                   len(recs),
            "retaker_count":           retaker_count,           # 精確重修人次（整數）
            "avg_midterm":             round(sum(mids)/len(mids), 2)   if mids   else None,
            "avg_final":               round(sum(fins)/len(fins), 2)   if fins   else None,
            "avg_semester":            round(sum(scores)/n, 2)         if n      else None,
            "pass_rate":               round(pass_count/n, 4)          if n      else None,
            "fail_rate":               round((n-pass_count)/n, 4)      if n      else None,
            "retaker_ratio":           round(retaker_count/n, 4)       if n      else None,
            "score_distribution":      dist,
            # ── 排除重修生統計（_nr = no retaker）────────────
            # count_nr：全班首修記錄數（含無成績），口徑與 count 一致
            # 驗證：count - count_nr == retaker_count（精確重修人次）
            "count_nr":                count_nr_recs,
            "avg_midterm_nr":          round(sum(mids_nr)/len(mids_nr), 2) if mids_nr   else None,
            "avg_final_nr":            round(sum(fins_nr)/len(fins_nr), 2) if fins_nr   else None,
            "avg_semester_nr":         round(sum(scores_nr)/n_nr, 2)       if n_nr      else None,
            "pass_rate_nr":            round(pass_count_nr/n_nr, 4)        if n_nr      else None,
            "fail_rate_nr":            round((n_nr-pass_count_nr)/n_nr, 4) if n_nr      else None,
            "score_distribution_nr":   dist_nr,
        }

    return summary


# ══════════════════════════════════════════════════════════════
# 學制分類（與 main.js classInfo() 邏輯保持一致）
# ══════════════════════════════════════════════════════════════

_CN_NUM_MAP      = str.maketrans("零〇一ㄧ二兩三四五六七八九", "0011223456789")
_CLASS_WORK_SET  = set("甲乙戊己")
_CLASS_NIGHT_SET = set("丙丁")

PROGRAM_LABELS = {
    "2yr_gen":        "二技一般",
    "2yr_work":       "二技在職",
    "2yr_night":      "二技夜間",
    "4yr":            "四技一般",
    "post":           "學士後護",
    "retake_class":   "重修班",
    "retake_student": "重修生",
    # BUG-FIX（健四二F護理系統計污染修正，0806）：見 _classify_program() 說明
    "non_nursing":    "非護理系（歷史紀錄）",
}


# ── _clean_sheet_name 預編譯正規式 ────────────────────────────
_RE_INVIS   = re.compile(r"[\u00AD\u200B-\u200F\u202A-\u202E\u2060\uFE00-\uFE0F\uFEFF]")
_RE_DASH    = re.compile(r"[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]")
_RE_SPACES  = re.compile(r"[\s\u00A0\u1680\u180E\u2000-\u200A\u2028\u2029\u202F\u205F\u3000]+")


def _clean_sheet_name(s: str) -> str:
    """對應 main.js cleanSheetName：NFKC 正規化、控制字元清除、空白清除"""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = _RE_INVIS.sub("", s)
    s = _RE_DASH.sub("-", s)
    s = _RE_SPACES.sub("", s.strip())
    return s


def _class_code_text(s: str) -> str:
    """對應 main.js classCodeText：全形數字→半形、轉大寫、已→己"""
    return _clean_sheet_name(s).upper().translate(_CN_NUM_MAP).replace("已", "己")


@lru_cache(maxsize=512)
def _classify_program(sheet_name: str, semester: str = "") -> str:
    """
    依 main.js classInfo() 相同優先序回傳學制 key。
    優先序：
      1. 重修生
      2. 重修班（重修/暑期/學分/補修/微免/遠距/R數字）
      3. 學士後護（學後護/學士後/學後/後護）
      4. 護/日/N + 2X + A–E        → 二技一般
      5. 護/日/N + 2X + 甲乙戊己   → 二技在職 / 丙丁 → 二技夜間
      6. 護/N + 4X + A–D           → 四技一般
      7. 健 + 數字（非護理系特殊任務班級，如健四二F）→ non_nursing
      8. fallback（依學期末碼）

    BUG-FIX（健四二F護理系統計污染修正，0806）：對應 dashboard
    main.js classInfo() 0805第三次修正（NON-NURSING-EXCLUSION-0805）。
    健四二F（國際護理管理學士班，特殊任務班級）原本因班名格式無法匹配
    上方任一護理系學制正規表示式，落入第8步 fallback（僅憑學期末碼猜測
    2yr_gen/4yr），已知會污染 _collect_prog_pts() 的回歸資料集（見
    CHANGELOG v7.15.9「已知殘留問題」#2，該輪僅用 MIN_CLASS_SIZE=20
    門檻巧合擋住，非明確排除規則），也會透過 06_merge_grades.py 的
    is_4tech = 1 if _classify_program(...)=="4yr" else 0 影響 XGBoost
    訓練特徵與早期預警 QMI 分組。此處補上明確排除規則（比對「健」+
    中文數字/阿拉伯數字開頭，與 main.js 完全一致的判斷式），獨立回傳
    "non_nursing"，不落入任何護理系學制分類；下游 _collect_prog_pts()
    另有明確排除（見該函式修改處），確保不再僅依賴人數門檻的巧合防護。
    """
    raw  = _clean_sheet_name(sheet_name)
    code = _class_code_text(raw)

    if not raw:
        return "2yr_gen"
    if "重修生" in raw:
        return "retake_student"
    if re.search(r"重修|暑期|學分|補修|微免|遠距|R\d", raw, re.IGNORECASE) or re.search(r"R\d", code):
        return "retake_class"
    if re.search(r"學後護|學士後|學後|後護", raw):
        return "post"

    m = re.search(r"(?:護|日|N)?2([1-9])([A-E])", code)
    if m:
        return "2yr_gen"

    m = re.search(r"(?:護|日|N)?2([1-9])([甲乙丙丁戊己])", raw)
    if m:
        return "2yr_night" if m.group(2) in _CLASS_NIGHT_SET else "2yr_work"

    if re.search(r"(?:護|N)?4([1-9])([A-D])", code):
        return "4yr"

    if re.match(r"^健[〇零一二兩三四五六七八九ㄧ\d]", raw):
        return "non_nursing"

    return "4yr" if str(semester)[-1:] == "2" else "2yr_gen"


# TRANSITION_MERGED_MICRO_CLASSES（115(1)起「基礎醫學臨床應用(一)」課程異動，見規劃書討論11）：
# 依學期別列出該學期併入合併課、微免部分僅授課week13-18的班級名單。
# 成績檔（xls）沒有課程名稱欄位，sheet_name 只是班級名稱（見 _classify_program()
# docstring 同一結論），故 course_track 只能用班級名單比對，不能像 LMS 側
# （9-sheet xlsx「學習紀錄統計」列0）用課程名稱字串比對。
# 115(1)前所有班級（含護21A~E／學後護41／護2重修）確認皆為傳統18週結構的
# 「微生物及免疫學」，不在此名單內即維持 course_track=None，沿用既有解析邏輯。
# 未來學期若班級命名不同，於此新增一組學期鍵值即可，不需改動比對邏輯本身。
TRANSITION_MERGED_MICRO_CLASSES: dict = {
    "1151": {"日21甲", "日21乙", "護21丙", "護21戊", "護21己"},
}


def _classify_course_track(sheet_name: str, semester: str) -> str | None:
    """
    回傳 course_track：
      "transition_merged_micro"：該學期屬於「基礎醫學臨床應用(一)」合併課微免部分
      None：一般（含傳統18週結構的「微生物及免疫學」，即 course_track 對成績解析
            與下游模組而言無特殊意義，沿用既有邏輯）

    比對前經 _class_code_text() 正規化（全形轉半形、中文數字轉阿拉伯數字、
    已→己），避免「日二一甲」vs「日21甲」、「護21己」vs「護21已」等輸入
    差異造成誤判——教學計畫原始文件本身即出現「已」誤植，需正規化後才能
    正確比對。
    """
    classes = TRANSITION_MERGED_MICRO_CLASSES.get(str(semester))
    if not classes:
        return None
    code = _class_code_text(_clean_sheet_name(sheet_name))
    normalized = {_class_code_text(_clean_sheet_name(c)) for c in classes}
    return "transition_merged_micro" if code in normalized else None


def _collect_prog_pts(class_summary: dict) -> tuple:
    """
    從 class_summary 建立回歸用資料點。
    回傳 (all_pts, prog_pts)，消除兩個回歸函式的重複迴圈。
    """
    all_pts:  list          = []
    prog_pts: dict          = defaultdict(list)
    for v in class_summary.values():
        if not v["count"] or v["pass_rate"] is None:
            continue
        prog = _classify_program(v["sheet_name"], v.get("semester", ""))
        # BUG-FIX（健四二F護理系統計污染修正，0806）：明確排除非護理系
        # 班級（見 _classify_program() 說明），取代 v7.15.9 僅靠
        # MIN_CLASS_SIZE 人數門檻巧合排除的作法——後者若該班未來人數
        # 成長超過門檻即會失效重新污染回歸資料集，此處為根本、明確的
        # 排除規則，不受班級人數影響。
        if prog == "non_nursing":
            continue
        # v7.15.9 BUG-FIX: 排除小班 outlier（其餘非 non_nursing 但人數
        # 過少的班級，例如剛開班尚未滿班的正常護理系班級）
        # 這些班次的極少人數資料點會使二次回歸拋物線頂點大幅左移，
        # 造成建議人數嚴重偏低（全體合併 ~14 人，實際應為 ~40 人）。
        if v["count"] < MIN_CLASS_SIZE:
            continue
        pt   = (v["count"], round(v["pass_rate"] * 100, 4))
        all_pts.append(pt)
        prog_pts[prog].append(pt)
    return all_pts, prog_pts


# ══════════════════════════════════════════════════════════════
# 人數 vs 及格率：線性回歸
# ══════════════════════════════════════════════════════════════

def _linear_regression(pts: list) -> dict:
    """
    給定 [(x, y), ...] 計算線性回歸 y = a + bx 與 Pearson r。
    至少需 2 點；失敗時回傳 {"available": False}。
    單趟掃描同時累計所有統計量，避免 6 次獨立 sum() 迭代。
    """
    n = len(pts)
    if n < 2:
        return {"available": False, "n": n}

    # 單趟累計：Σx, Σy, Σx², Σxy
    sx = sy = sxx = sxy = 0.0
    x_min = x_max = pts[0][0]
    for x, y in pts:
        sx  += x;   sy  += y
        sxx += x*x; sxy += x*y
        if x < x_min: x_min = x
        if x > x_max: x_max = x

    denom = n * sxx - sx * sx
    if denom == 0:
        return {"available": False, "n": n}

    b      = (n * sxy - sx * sy) / denom
    a      = (sy - b * sx) / n

    # Pearson r：利用已有統計量推導，無需再次遍歷
    # r = (n·Σxy - Σx·Σy) / sqrt((n·Σx²-Σx²)·(n·Σy²-Σy²))
    # 需 Σy²，補一次輕量掃描（避免先前迭代記憶體佔用）
    syy = sum(y*y for _, y in pts)
    denom_r = (denom * (n * syy - sy * sy)) ** 0.5
    r = (n * sxy - sx * sy) / denom_r if denom_r else 0.0

    return {
        "available": True,
        "n":         n,
        "slope":     round(b, 4),
        "intercept": round(a, 2),
        "r":         round(r, 4),
        "line": [
            {"x": x_min, "y": round(a + b * x_min, 1)},
            {"x": x_max, "y": round(a + b * x_max, 1)},
        ],
    }


def compute_count_passrate_regression(class_summary: dict) -> dict:
    """
    以各班次 count vs pass_rate 計算線性回歸與 Pearson r（全體 + 各學制）。
    """
    all_pts, prog_pts = _collect_prog_pts(class_summary)

    reg_all          = _linear_regression(all_pts)
    reg_all["label"] = "全體合併"
    result           = {"all": reg_all}

    for prog, pts in prog_pts.items():
        reg          = _linear_regression(pts)
        reg["label"] = PROGRAM_LABELS.get(prog, prog)
        result[prog] = reg

    return result


# ══════════════════════════════════════════════════════════════
# 各學制建議修課人數（二次回歸求及格率最高頂點）
# ══════════════════════════════════════════════════════════════

def _quadratic_fit(xs: list, ys: list):
    """
    純 Python 最小平方二次回歸（含 partial pivoting）。
    解 3×3 正規方程，回傳 [a, b, c] 使 y ≈ ax²+bx+c。
    行列式趨近 0 時回傳 None。
    單趟掃描同時累計所有統計量，避免 6 次獨立 sum() 迭代。
    """
    n = len(xs)
    # 單趟累計 Σx, Σx², Σx³, Σx⁴, Σy, Σxy, Σx²y
    sx = sx2 = sx3 = sx4 = sy = sxy = sx2y = 0.0
    for x, y in zip(xs, ys):
        x2 = x * x
        sx   += x;    sx2  += x2;    sx3 += x2 * x
        sx4  += x2 * x2
        sy   += y;    sxy  += x * y; sx2y += x2 * y

    M = [
        [sx4, sx3, sx2, sx2y],
        [sx3, sx2, sx,  sxy ],
        [sx2, sx,  n,   sy  ],
    ]

    for col in range(3):
        max_row = max(range(col, 3), key=lambda r: abs(M[r][col]))
        M[col], M[max_row] = M[max_row], M[col]
        pivot = M[col][col]
        if abs(pivot) < 1e-10:
            return None
        for row in range(col + 1, 3):
            f = M[row][col] / pivot
            M[row] = [M[row][j] - f * M[col][j] for j in range(4)]

    coeffs = [0.0] * 3
    for i in range(2, -1, -1):
        coeffs[i] = (M[i][3] - sum(M[i][j] * coeffs[j] for j in range(i + 1, 3))) / M[i][i]

    return coeffs


def compute_optimal_enrollment(class_summary: dict) -> dict:
    """
    依學制分群，各自做二次回歸 y = ax²+bx+c（人數 vs 及格率%），
    求及格率最高時的建議修課人數（頂點 x = -b/2a，需 a < 0）。
    """
    all_pts, prog_pts = _collect_prog_pts(class_summary)
    prog_pts["all"]   = all_pts

    result = {}
    for prog, pts in prog_pts.items():
        label = "全體合併" if prog == "all" else PROGRAM_LABELS.get(prog, prog)
        n     = len(pts)

        if n < 3:
            result[prog] = {"available": False, "label": label, "reason": "insufficient_data", "n": n}
            continue

        xs, ys = [], []
        x_min = x_max = None
        for px, py in pts:
            xs.append(px); ys.append(py)
            if x_min is None or px < x_min: x_min = px
            if x_max is None or px > x_max: x_max = px
        coeffs = _quadratic_fit(xs, ys)
        if coeffs is None:
            result[prog] = {"available": False, "label": label, "reason": "singular_matrix", "n": n}
            continue

        a, b, c = coeffs
        if abs(a) < 1e-6:
            result[prog] = {"available": False, "label": label, "reason": "no_maximum", "shape": "linear", "n": n}
            continue
        if a > 0:
            result[prog] = {"available": False, "label": label, "reason": "no_maximum", "shape": "U_shaped", "n": n}
            continue

        optimal_x = -b / (2 * a)
        optimal_y = a * optimal_x**2 + b * optimal_x + c
        clamped   = max(x_min, min(x_max, optimal_x))

        result[prog] = {
            "available":             True,
            "label":                 label,
            "n":                     n,
            "optimal_count":         round(optimal_x, 1),
            "optimal_count_clamped": round(clamped, 1),
            "optimal_passrate":      round(min(optimal_y, 100.0), 1),
            "in_range":              x_min <= optimal_x <= x_max,
            "x_min":                 x_min,
            "x_max":                 x_max,
            "coef":                  [round(a, 6), round(b, 4), round(c, 2)],
        }

    return result


# ══════════════════════════════════════════════════════════════
# 彙整 students（以學號為 key）
# ══════════════════════════════════════════════════════════════

def build_student_map(all_records: list, salt: str = "") -> dict:
    """
    以原始學號為 key，整合所有修課記錄，並產生 anon_id。
    mask_student_id 與 anonymize_student_id（SHA-256）各只對每個唯一學號計算一次，
    不隨記錄數重複呼叫。
    """
    # 預先建立唯一學號 → (masked, anon) 對照表
    unique_sids = {rec["student_id"] for rec in all_records}
    sid_meta = {
        sid: (mask_student_id(sid), anonymize_student_id(sid, salt))
        for sid in unique_sids
    }

    students: dict = {}
    for rec in all_records:
        sid = rec["student_id"]
        masked, anon = sid_meta[sid]
        if sid not in students:
            students[sid] = {"name_masked": masked, "anon_id": anon, "records": []}
        students[sid]["records"].append({
            "semester":       rec["semester"],
            "sheet_name":     rec["sheet_name"],
            "type":           rec["type"],
            "class_code_raw": rec["class_code_raw"],
            "class_name":     rec["class_name"],
            "midterm":        rec["midterm"],
            "final":          rec["final"],
            "semester_score": rec["semester_score"],
            "adjusted":       rec["adjusted"],
            "reading_pct":    rec["reading_pct"],
            "attend_pct":     rec["attend_pct"],
            "reading_hours":  rec["reading_hours"],
            "is_retaker":     rec["is_retaker"],
            "delta":          rec["delta"],
            "exceptions":     rec["exceptions"],
        })

    for data in students.values():
        data["records"].sort(key=lambda x: (x["semester"], x["sheet_name"]))

    return students


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def load_exam_end_dates(config_path: str = "config/exam_dates.json") -> dict:
    """
    讀取期末考日期設定檔，回傳 {學期代碼: 期末考結束日ISO字串} 的扁平字典。

    36號規格書決議：exam_end_date 匯出位置為 data.json 的 meta.exam_end_dates，
    供前端 isWithinIdRevealWindow() 判斷8小時徽章的14天揭示窗口（自然日，
    期末考結束日隔天起算）。

    設定檔格式與 lms_etl.py --config 讀取的 config/exam_dates.json 共用同一份
    檔案、同一種 schema（{"semesters": {學期: {"final": "YYYY-MM-DD", ...}}}），
    但兩支腳本刻意不共用程式碼（見本檔開頭慣例：etl.py 是獨立腳本，不 import
    lms_etl.py），故此處另外實作一份最小讀取邏輯，只取需要的 "final" 值。

    找不到檔案或格式不符時，回傳空字典並印出警告，不中斷主流程——這與
    無 exam_end_date 時「預設顯示完整學號」的既定 fallback 語意一致：
    前端查無某學期的鍵值，就等同視為「還沒有考完試」。
    """
    if not os.path.exists(config_path):
        print(f"[ETL] ⚠️  找不到考試日期設定檔（{config_path}），"
              f"meta.exam_end_dates 將為空——8小時徽章會預設顯示完整學號")
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"[ETL] ⚠️  考試日期設定檔讀取失敗（{config_path}）：{e}"
              f"——meta.exam_end_dates 將為空")
        return {}

    # [AUDIT-FIX 穿透式審查 0907] 原本這裡直接假設 raw/semesters_cfg 一定是
    # dict，格式不符（例如整份檔案是list、或"semesters"鍵值本身是list）時
    # 會直接拋AttributeError讓整支ETL中斷——與本函式docstring自己承諾的
    # 「格式不符時回傳空字典、不中斷主流程」矛盾。這個設定檔是真人會手動
    # 編輯的檔案，手滑改壞格式不該讓整批30+學期的ETL全部失敗，補上明確的
    # 型別檢查，讓每一種文件承諾的降級路徑都真的有實作。
    if not isinstance(raw, dict):
        print(f"[ETL] ⚠️  考試日期設定檔格式不符（{config_path}）："
              f"頂層應為dict物件，實際為{type(raw).__name__}"
              f"——meta.exam_end_dates 將為空")
        return {}

    semesters_cfg = raw.get("semesters", {})
    if not isinstance(semesters_cfg, dict):
        print(f"[ETL] ⚠️  考試日期設定檔格式不符（{config_path}）："
              f"\"semesters\"鍵值應為dict物件，實際為{type(semesters_cfg).__name__}"
              f"——meta.exam_end_dates 將為空")
        return {}

    exam_end_dates = {}
    for sem, info in semesters_cfg.items():
        final_date = (info or {}).get("final") if isinstance(info, dict) else None
        if final_date:
            exam_end_dates[sem] = final_date
    return exam_end_dates


def run_etl(raw_dir: str, out_path: str, salt_path: str = "config/salt.txt",
            exam_config_path: str = "config/exam_dates.json"):
    print(f"[ETL] 掃描資料夾：{raw_dir}")
    salt = load_salt(salt_path)
    print(f"[ETL] 已載入 salt（{salt_path}），將產生 anon_id")

    xls_files = sorted([
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.lower().endswith((".xls", ".xlsx"))
    ])

    if not xls_files:
        print("[ETL] ⚠️  未找到任何 .xls 檔案，請確認 raw_data/ 資料夾")
        return

    all_records:   list = []
    semesters_seen: set = set()

    for fpath in xls_files:
        semester = parse_semester_from_filename(fpath)
        semesters_seen.add(semester)
        fname = os.path.basename(fpath)
        print(f"  → 處理：{fname}  (學期代碼: {semester})")

        try:
            wb, encoding_label = open_workbook_with_encoding_fix(fpath)
        except Exception as e:
            print(f"     ❌ 開啟失敗：{e}")
            continue

        if encoding_label != "default":
            print(f"     編碼修正：使用 {encoding_label} 重新讀取舊版 XLS 中文")

        for sh_name in wb.sheet_names():
            sh          = wb.sheet_by_name(sh_name)
            sheet_type  = detect_sheet_type(sh_name)
            course_track = _classify_course_track(sh_name, semester)
            if course_track == "transition_merged_micro":
                # 見規劃書決策1／討論11：成績檔無課程名稱欄位，course_track
                # 只能用班級名單比對（TRANSITION_MERGED_MICRO_CLASSES），命中
                # 時改走專用解析（欄位結構與一般正課完全不同，見該函式說明）。
                recs = parse_transition_micro_sheet(sh, semester)
            else:
                recs = (
                    parse_practicum_sheet(sh, semester)
                    if sheet_type == "practicum"
                    else parse_theory_sheet(sh, semester, sheet_type)
                )
            print(f"     分頁「{sh_name}」({sheet_type}): {len(recs)} 筆有效學生記錄")
            all_records.extend(recs)

    incomplete_grade_semesters = detect_incomplete_grade_semesters(all_records)
    if incomplete_grade_semesters:
        changed = nullify_incomplete_grade_scores(all_records, incomplete_grade_semesters)
        print(
            "[ETL] 偵測到尚未取得期末/學期成績的學期："
            f"{incomplete_grade_semesters}，已清除 {changed} 筆占位成績"
        )

    print(f"\n[ETL] 共 {len(all_records)} 筆原始記錄，開始跨學期重修判定...")

    # 見上方 PHANTOM_STUDENT_ID 定義處說明：第二層獨立防線，明確排除
    # student_id 為補零假學號的殘留記錄，寫出 data.json 前的最後把關。
    _n_before_phantom_filter = len(all_records)
    all_records = [r for r in all_records if r.get("student_id") != PHANTOM_STUDENT_ID]
    _n_phantom_removed = _n_before_phantom_filter - len(all_records)
    if _n_phantom_removed:
        print(f"[ETL] ⚠️  排除 {_n_phantom_removed} 筆學號為 {PHANTOM_STUDENT_ID!r} 的殘留記錄"
              f"（理論上 is_valid_student_row() 已從源頭排除，此為防禦性重複檢查，"
              f"若看到此訊息代表有其他路徑繞過了該檢查，建議追查根因）")

    all_records  = compute_retaker_flags(all_records)
    retaker_count = sum(1 for r in all_records if r["is_retaker"])
    print(f"[ETL] 重修記錄：{retaker_count} 筆")

    print("[ETL] 計算班級統計摘要...")
    class_summary = compute_class_summary(all_records)

    print("[ETL] 計算人數 vs 及格率回歸（全體 + 各學制）...")
    regression = compute_count_passrate_regression(class_summary)

    print("[ETL] 計算各學制建議修課人數（二次回歸）...")
    optimal_enrollment = compute_optimal_enrollment(class_summary)

    print("[ETL] 建立學生資料表...")
    students = build_student_map(all_records, salt=salt)

    print(f"[ETL] 讀取期末考日期設定：{exam_config_path}")
    _exam_end_dates_cfg = load_exam_end_dates(exam_config_path)
    # 只保留這次資料實際涵蓋到的學期，未收錄的學期不放進 meta，
    # 前端查無鍵值時會依既定規則 fallback 成「顯示完整學號」。
    exam_end_dates = {
        sem: _exam_end_dates_cfg[sem]
        for sem in sorted(semesters_seen)
        if sem in _exam_end_dates_cfg
    }
    print(f"[ETL]   {len(exam_end_dates)}/{len(semesters_seen)} 個學期有期末考日期資料")

    output = {
        "meta": {
            "generated_at":              datetime.now().isoformat(timespec="seconds"),
            "semesters":                 sorted(semesters_seen),
            "total_students":            len(students),
            "total_records":             len(all_records),
            "schema_version":            "3.0",
            "incomplete_grade_semesters": incomplete_grade_semesters,
            "incomplete_grade_note": (
                "These semesters have not settled final/semester grades yet; "
                "they are excluded from pass/fail rates and cohort score comparisons."
            ),
            "count_passrate_regression": regression,
            "optimal_enrollment":        optimal_enrollment,
            # ── 前端動態讀取用設定（避免前端硬編碼）────────────────
            # fail_threshold: 及格線，與 ETL FAIL_THRESHOLD 保持同步
            "fail_threshold":            FAIL_THRESHOLD,
            # covid_sem_range: 疫情期間學期範圍 [起, 迄]（固定不變）
            "covid_sem_range":           ["1082", "1112"],
            # curriculum_sem_range: 108課綱期間 [起, null]
            # 迄設為 null = 「持續更新中」，前端以 semesters[-1] 作為上界
            "curriculum_sem_range":      ["1111", None],
            # program_change_sem_range（見規劃書討論6）：「二技在職改制」
            # 教育架構大事件學期範圍 [起, 迄]，比照 covid_sem_range／
            # curriculum_sem_range 同一寫法。起始115(1)，暫無結束時間
            # （持續中，null，語意同 curriculum_sem_range）。
            "program_change_sem_range":  ["1151", None],
            # pass_color_high/mid: 及格率 UI 顏色判斷閾值（green/orange/red 邊界）
            "pass_color_high":           0.9,
            "pass_color_mid":            0.7,
            # exam_end_dates（36號規格書第八節）：{學期代碼: 期末考結束日ISO字串}，
            # 只收錄這次資料實際涵蓋到的學期。前端 isWithinIdRevealWindow() 用它
            # 判斷8小時徽章的14天完整學號揭示窗口；查無鍵值 = 預設顯示完整學號。
            "exam_end_dates":            exam_end_dates,
        },
        "students":      students,
        "class_summary": class_summary,
    }

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[ETL] ✅ 完成！輸出至：{out_path}")
    print(f"       學生數：{len(students)}")
    print(f"       班級摘要：{len(class_summary)} 個班次")
    print(f"       學期涵蓋：{sorted(semesters_seen)}")


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="學習數據分析儀表板 ETL 腳本")
    parser.add_argument("--raw_dir", default=DEFAULT_RAW_DIR,
                        help=f"原始 xls 資料夾（預設：{DEFAULT_RAW_DIR}）")
    parser.add_argument("--out",     default=DEFAULT_OUT,
                        help=f"輸出 JSON 路徑（預設：{DEFAULT_OUT}）")
    parser.add_argument("--salt",    default="config/salt.txt",
                        help="去識別化鹽值檔路徑（預設：config/salt.txt）")
    parser.add_argument("--exam_config", default="config/exam_dates.json",
                        help="期末考日期設定檔路徑，與 lms_etl.py --config 共用同一份"
                             "（預設：config/exam_dates.json）")
    args = parser.parse_args()
    run_etl(args.raw_dir, args.out, args.salt, args.exam_config)
