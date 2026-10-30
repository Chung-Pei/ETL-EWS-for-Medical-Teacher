"""
lms_etl.py — LMS 學習行為 ETL 主控腳本
用法：
  # 【推薦】指定目錄，自動讀取 lms_raw/ 下所有 .xlsx
  python lms_etl.py --lms lms_raw/

  # 搭配完整參數
  python lms_etl.py \
      --lms     lms_raw/ \
      --data    docs/data/data.json \
      --config  config/exam_dates.json \
      --salt    config/salt.txt \
      --out     docs/data/

  可選旗標：
    --course   <課程關鍵字>   data.json 中的課程名稱篩選（選填）
    --skip-grades             不合併成績（data.json 尚未準備時）
    --no-auto-dates           停用自動偵測，改用 config 固定日期
    --dry-run                 模擬執行，不寫入任何檔案
    --generate-config         只掃描並補齊 config，不繼續跑完整pipeline
                              （離線先準備好config用；一般情況不需要）

學期代碼（如 1141）由 xlsx 檔名前5碼自動解析（格式：YYY-S，例如 114-1）。
多個 xlsx 學期不一致時，取所有解析值中的最多數（主學期）。

批次合併：
  lms_raw/ 目錄下可放置多個 xlsx（一個學期可跨多班）。
  程式自動讀取目錄下所有 .xlsx，合併相同 sheet 後進行 ETL。
  summary sheet 依學號欄去重，保留最後一筆；其餘 sheet 縱向合併。

自動偵測考試日期（正確邏輯）：
  每個 xlsx 內涵蓋多個班級，學生可能跨班上課。
  因此期中考 / 期末考日期不具班級一致性，
  必須從每位學生自己的測驗紀錄中擷取。

  偵測規則：
    - 從「測驗記錄」Sheet 中，找測驗標題以 [重要] 開頭
      且含「期中考」/ 「期末考」的紀錄
    - 從該標題括號內解析民國年日期（支援半型 () 及全型（））
    - 每位學生（anon_id）各自取自己出現的那一筆日期
    - 同一學生若有多個期中考日期（極少見），取最早一筆；
      期末考取最晚一筆

  加上 --no-auto-dates 可停用此功能，改用 config 固定值（全體同一組日期）。

校歷驗證與config自動補齊（v7系列，見 CHANGELOG_v7.md 完整歷史）：
  執行時若偵測到 semester_start 缺漏，會依「該學期是否已有期末成績」
  （data.json）分流：
    - 已結束學期（已有期末成績）→ 直接用LMS實測值（練習題庫優先）自動
      套用，不詢問。
    - 最新／進行中學期（尚無期末成績）→ 互動詢問一次 (Y/N)：
      Y 採用建議值；N 手動輸入正確日期；直接 Enter 放棄則維持缺漏，
      交由 validate_semester_calendar() 硬中斷（不接受略過）。
  非互動環境（無 stdin，例如排程工作）一律視為未確認，安全退回硬中斷，
  不會誤判成使用者同意。
"""

import argparse
import json
import os
import re as _re
import sys
import traceback
import importlib.util
import pandas as pd
from collections import Counter

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

# 模組載入時立即執行（intentional side-effect）：確保後續所有 print() 以 UTF-8 輸出，
# 避免 Windows Store Python 在 CP950 環境下因中文字符崩潰。
_configure_utf8_stdio()

sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────────────────────────────
# 因 Python 模組名稱不得以數字開頭，使用 importlib 動態載入
# ─────────────────────────────────────────────────────────────

def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"無法載入模組：找不到檔案或路徑無效 → {path}")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        raise ImportError(f"模組 {name} 載入失敗（{path}）：{e}") from e
    return mod

_BASE = os.path.join(os.path.dirname(__file__), "lms_etl", "modules")

m01 = _load_module(os.path.join(_BASE, "01_load_lms_excel.py"),    "m01")
m02 = _load_module(os.path.join(_BASE, "02_anonymize.py"),         "m02")
m03 = _load_module(os.path.join(_BASE, "03_parse_tags.py"),        "m03")
m04 = _load_module(os.path.join(_BASE, "04_feature_engineering.py"),"m04")
m05 = _load_module(os.path.join(_BASE, "05_time_analysis.py"),     "m05")
m06 = _load_module(os.path.join(_BASE, "06_merge_grades.py"),      "m06")
m06b= _load_module(os.path.join(_BASE, "06b_clustering.py"),       "m06b")
m07 = _load_module(os.path.join(_BASE, "07_export_json.py"),       "m07")
m07c= _load_module(os.path.join(_BASE, "07c_stability.py"),        "m07c")
m07d= _load_module(os.path.join(_BASE, "07d_density_filter.py"),   "m07d")
m09 = _load_module(os.path.join(_BASE, "09_lsa_event_log.py"),     "m09")
m10 = _load_module(os.path.join(_BASE, "10_lsa_transition.py"),    "m10")
m11 = _load_module(os.path.join(_BASE, "11_cross_analysis.py"),    "m11")
m12 = _load_module(os.path.join(_BASE, "12_early_warning.py"),     "m12")

# ── CLI 引數 ──────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="教學成效儀表板 — LMS 學習行為 ETL（Phase 2）"
    )
    parser.add_argument("--lms",      required=True, nargs="+",
                        help="LMS Excel 路徑：多個檔案，或目錄（自動讀取目錄下所有 .xlsx）")
    parser.add_argument("--data",     default="docs/data/data.json",
                        help="成績 data.json 路徑（預設 docs/data/data.json）")
    parser.add_argument("--config",   default="config/exam_dates.json",
                        help="考試日期設定（預設 config/exam_dates.json）")
    parser.add_argument("--salt",     default="config/salt.txt",
                        help="去識別化鹽值檔（預設 config/salt.txt）")
    parser.add_argument("--out",      default="docs/data",
                        help="輸出目錄（預設 docs/data）")
    parser.add_argument("--course",   default="",
                        help="data.json 中 sheet_name 欄的篩選關鍵字"
                             "（通常對應班級/場次名稱，非課程科目名稱；選填，預設不篩選）")
    parser.add_argument("--skip-grades", action="store_true",
                        help="不合併成績（data.json 尚未準備時使用）")
    parser.add_argument("--no-auto-dates", action="store_true",
                        help="停用自動偵測考試日期（使用 config/exam_dates.json 中的固定值套用於所有學生）")
    parser.add_argument("--dry-run",  action="store_true",
                        help="執行所有步驟但不寫入檔案")
    parser.add_argument("--warning-target", default="",
                        help="提前預警系統的目標學期（如 1151）。"
                             "留空則自動使用 incomplete_semesters 中最新一個學期；"
                             "若 incomplete_semesters 為空，則不產生 warning_*.json。")
    parser.add_argument("--checkpoint-target", default="",
                        help="8小時閱讀門檻「學期中進度快照」目標學期（如 1151）。"
                             "留空則不產生checkpoint清單；僅對course_track="
                             "transition_merged_micro的學期有意義。")
    parser.add_argument("--week-limit", type=int, default=12,
                        help="早期預警特徵截斷週數（預設 12，僅作用於 Quiz/Time/LSA 來源）")
    parser.add_argument("--configure-weeks", default="",
                        help="v7.11新增：逗號分隔的學期代碼清單（如 '1121,1122,1131'），"
                             "強制對這些學期詢問週次結構（總週次18標準/16+2週特例、"
                             "自學週對應週次），即使該學期成績已結算也會詢問一次。"
                             "預設不設定時，只有尚無期末成績的最新/進行中學期才會詢問，"
                             "已結算成績的歷史學期維持自動套用、完全不打擾——這個旗標"
                             "是唯一能讓已結算歷史學期（如112-1/112-2/113-1因行政特例"
                             "為16+2週）主動觸發這組確認的方式，避免『每個沒設定"
                             "total_weeks的歷史學期都跳出來問』造成不必要的互動負擔。")
    parser.add_argument("--enable-xgb", action="store_true", dest="enable_xgb", default=True,
                        help="啟用 XGBoost 訓練/預測（預設啟用）")
    parser.add_argument("--disable-xgb", dest="enable_xgb", action="store_false",
                        help="停用 XGBoost，退回 BAS-only 模式")
    parser.add_argument("--generate-config", action="store_true",
                        help="v7.2新增：不跑完整pipeline，只掃描 --lms 底下的檔案，"
                             "把校歷驗證用得到的 semester_start / calendar_overrides "
                             "自動補進 --config 指定的檔案後即結束。已存在且已有 "
                             "semester_start 的學期不會被覆蓋（視為人工已確認），"
                             "只補缺漏的部分。仍需人工核對輸出結果是否符合校方正式"
                             "行事曆——這一步無法自動化。適合想離線先準備好config、"
                             "不想在跑pipeline途中被詢問Y/N的情境；一般情況下"
                             "不需要加這個參數，直接跑正常指令、遇到缺漏時會自動"
                             "互動詢問（見 --generate-config 說明）。")
    return parser.parse_args()

# ── 學期代碼解析 ──────────────────────────────────────────────

def parse_semester_from_filename(fname: str) -> str:
    """
    從 xlsx 檔名前5字元自動解析學期代碼。

    格式規則：檔名前5碼必須為 YYY-S（民國年3碼 + 連字號 + 學期碼1碼）。
    輸出統一4碼格式（移除連字號）。

    範例：
      "114-1班級A學習紀錄.xlsx" → "1141"
      "113-2班級B.xlsx"         → "1132"
      "other_file.xlsx"         → ""（無法解析，安全返回空字串）
    """
    m = _re.match(r'^(\d{3})-([12])', fname)
    return m.group(1) + m.group(2) if m else ""

def _infer_and_collect_semesters(
    file_paths: list,
) -> "tuple[str, list]":
    """
    單次迴圈同時計算：
    - 主學期（infer_semester_from_files 邏輯）
    - 全部學期列表（collect_semesters_from_files 邏輯）

    v6修正：拿掉 exam_dates 參數。per-semester keyed 後，檔名完全無法解析時
    已無「單一預設學期」概念可用，不再 fallback 至任何 exam_dates["semester"]。

    Returns
    -------
    (semester: str, behavior_semesters: list)
    """
    parsed = [parse_semester_from_filename(os.path.basename(p)) for p in file_paths]
    valid  = [s for s in parsed if s]
    behavior_semesters = sorted(set(valid))

    if not valid:
        return "", behavior_semesters

    counter    = Counter(valid)
    max_count  = max(counter.values())
    candidates = [s for s, c in counter.items() if c == max_count]
    chosen     = sorted(candidates)[-1]

    if len(set(valid)) > 1:
        print(f"  ⚠️  偵測到多個學期：{behavior_semesters}，以主學期 {chosen!r} 為準")
    else:
        print(f"  📅 自動偵測學期：{chosen}")

    return chosen, behavior_semesters

def format_semester_label(semester: str) -> str:
    """
    將 1141 / 114-1 轉為前端顯示用的 114(1)。
    """
    sem = str(semester or "").strip()
    m = _re.match(r"^(\d{3})-?([12])$", sem)
    return f"{m.group(1)}({m.group(2)})" if m else sem

def format_semester_range(semesters: list) -> str:
    """
    將多學期清單轉成 111(1)-114(2) 形式。
    """
    if not semesters:
        return ""
    labels = [format_semester_label(s) for s in semesters]
    return labels[0] if labels[0] == labels[-1] else f"{labels[0]}-{labels[-1]}"

# ── 讀取 config ───────────────────────────────────────────────
def load_config(config_path: str) -> dict:
    """
    v6修正：改讀取 per-semester keyed 結構，回傳 {"1141": {...}, "1142": {...}}。
    原本回傳單一扁平 dict（假設全體學期共用同一組 midterm/final/
    semester_start），這正是 Bug A 的根因——semester_start 全域殘留、
    未隨學期更新（見除錯報告v7 二～三節）。
    """
    if not os.path.exists(config_path):
        print(f"⚠️  找不到 config：{config_path}，使用預設空值")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if "semesters" not in raw:
        raise ValueError(
            f"config 格式不相容：{config_path} 未使用新版 per-semester keyed "
            f"結構（缺少頂層鍵 'semesters'）。請改為："
            f'\n  {{"semesters": {{"1141": {{...}}, "1142": {{...}}}}}}\n'
            f"人工確認後再重跑（刻意的硬中斷，避免誤用舊格式導致 Bug A 重演）。"
        )
    return raw["semesters"]

# ── 考試日期解析工具 ──────────────────────────────────────────

# 格式 1：含民國年（如 114.11.03 / 114-11-03）
_RE_DATE = _re.compile(
    r'[(\（]'
    r'(\d{2,3})[.\-/年]'       # 民國年（如 114）
    r'(\d{1,2})[.\-/月]'       # 月
    r'(\d{1,2})'               # 日
    r'(?:[日\s]*(\d{1,2}:\d{2})'           # 可選：開始時間 HH:MM
    r'(?:[~\-至](\d{1,2}:\d{2}))?)?'       # 可選：結束時間
    r'\s*[)\）]'               # 閉括號（半型或全型）
)

# 格式 2：只有 M/D（如舊學期 112-2 以前：(4/9 12:10-12:50)）
# 必須後接時間 HH:MM，避免誤抓「(前測)」「(2/12 - 2/29間)」等非日期括號
#
# v1.1（/systematic-debugging 全庫穿透式審查，2026-07）：真實資料（111-1
# 二技+學後護一）發現大量「(M/D HH:MM開始，限 XXX，已結束)」／
# 「(M/D HH:MM開始，限 XXX)(已結束)」／「(M/D HH:MM開始，限 XXX)
# (第1次補考)」等標題（22組真實標題、佔全部考試相關標題約28%），時間
# 後面接的是敘述性文字（開課班級限制、補考次數等）而非直接接結束時間
# 後閉括號，與原規則「必須緊接可選結束時間後立刻閉括號」的假設不符，
# 全數解析失敗（回傳None），造成這些學生查無期中/期末考日期、退回
# 學期預設值——與本輪其他3個bug同一種「格式假設過窄」問題。既然
# HH:MM時間本身已足以與「(前測)」「(2/12 - 2/29間)」等非日期括號區隔
# （這些都沒有HH:MM時間格式），不再要求「緊接閉括號」，只要求抓到
# 「(M/D HH:MM」即視為找到日期，括號內後續文字（無論是結束時間、班級
# 限制說明、或補考次數說明）一律略過不解析。
_RE_DATE_NO_YEAR = _re.compile(
    r'[(\（]'
    r'(\d{1,2})/(\d{1,2})'         # M/D
    r'\s+(\d{1,2}:\d{2})'           # 必須有時間 HH:MM
)

_RE_IMPORTANT = _re.compile(r'^\[重要\]')
# v1.1（/systematic-debugging 全庫穿透式審查，2026-07）：真實資料
# （111-1二技+學後護一）發現補考標題有兩種命名慣例並存：
#   1.「正課期中考試 補考-1」（112-2）── 含「期中考」子字串，原規則已涵蓋
#   2.「正課期中補考-1」（111-1）── 「補」插在「中」「考」之間，
#      原規則`期中考`（無?)不含此子字串，比對失敗
# `build_per_student_exam_dates()`docstring明載「同一學生同一學期若有
# 多筆期中考（複試/補考），取最早日期／期末取最晚日期」，即設計上
# 本應把補考記錄一併納入計算；原規則的「補考」命名比對缺口導致部分
# 學生（真實資料確認111-1二技至少3名學生的期中/期末考記錄僅有「期中
# 補考」/「期末補考」標題、無對應正課考試標題）完全查無期中/期末考
# 日期，退回學期預設值，以及至少1名學生的期末考日期本應取補考（較晚）
# 卻誤用正課（較早）日期──與RE_STUDENT_ID／TAG_RULES／題庫週次三個
# bug同一種「格式白名單覆蓋不全」問題。改用「補」字可有可無涵蓋兩種
# 命名慣例，不影響任何原本已比對成功的標題。
_RE_MIDTERM   = _re.compile(r'期中補?考')
_RE_FINAL     = _re.compile(r'期末補?考')

def _is_midterm_title(t) -> bool:
    return bool(_RE_MIDTERM.search(str(t)))

def _is_final_title(t) -> bool:
    return bool(_RE_FINAL.search(str(t)))

def _roc_to_ad(yy: str) -> int:
    """民國年 → 西元年（恆加 1911；民國年合理範圍 1–199）"""
    return int(yy) + 1911

def _parse_exam_date_from_title(title: str, semester: str = ""):
    """
    從測驗標題解析考試日期字串（YYYY-MM-DD）。
    找不到日期時回傳 None。

    支援兩種格式：
      1. 含民國年：(114.11.03 13:45-14:25) → 2025-11-03  [新學期]
      2. 只有 M/D：(4/9 12:10-12:50)       → 年份由 semester 推算 [舊學期 112-2 以前]
         - semester 格式 YYYS（如 "1122"）
         - 第2學期（S=2）：課程在「次一個」西元年的 2~7 月，
           ad_year = ROC+1911+1（整學期皆跨年，無邊界情況）
         - 第1學期（S=1）：課程在同年 9 月到次年 1 月，
           月份 ≤ 2 時 ad_year + 1（1 月尾段）
    """
    # 優先嘗試含年份格式
    m = _RE_DATE.search(title)
    if m:
        yy, mm, dd = m.group(1), m.group(2), m.group(3)
        return f"{_roc_to_ad(yy)}-{int(mm):02d}-{int(dd):02d}"

    # 嘗試無年份的 M/D 格式（需要 semester 推算年份）
    m2 = _RE_DATE_NO_YEAR.search(title)
    if m2 and semester and len(semester) == 4:
        mon  = int(m2.group(1))
        day  = int(m2.group(2))
        try:
            roc_year = int(semester[:3])
            sem_no   = int(semester[3])
        except ValueError:
            return None
        ad_year = roc_year + 1911
        # 第1學期跨年：9月開學，1~2月屬於隔年
        if sem_no == 1 and mon <= 2:
            ad_year += 1
        # 第2學期：2月開學，整學期（不僅邊界月份）都落在下一個西元年
        elif sem_no == 2:
            ad_year += 1
        return f"{ad_year}-{mon:02d}-{day:02d}"

    return None

def _is_exam_row(title) -> bool:
    """判斷測驗標題是否為 [重要] 期中/期末考記錄列。無 closure，可獨立測試。"""
    if not isinstance(title, str):
        return False
    t = title.strip()
    return bool(_RE_IMPORTANT.match(t) and (_RE_MIDTERM.search(t) or _RE_FINAL.search(t)))

def _apply_calendar_override(base_dates: dict, grp: "pd.DataFrame", anon_id: str, sem: str) -> dict:
    """
    v6新增（22-7規劃書Phase 4，問題1）：若該學生此學期的quiz列都來自同一個
    有 calendar_overrides 設定的 _section_id（如暑期壓縮重修班），用該
    override 覆寫 base_dates 的 semester_start/total_weeks 等欄位。

    base_dates 須為 dict(exam_dates_all.get(sem, {})) 的複本（呼叫端負責），
    本函式會就地修改並回傳同一個 dict。找不到 _section_id 欄、或該學生
    橫跨多個 section_id（無法判斷用哪組行事曆）時，原樣回傳 base_dates
    （退回學期預設值，這是安全的預設路徑）。
    """
    overrides = base_dates.pop("calendar_overrides", {})   # 管理用欄位不混入最終dict
    if not overrides or "_section_id" not in grp.columns:
        return base_dates
    sec_ids = grp["_section_id"].dropna().unique()
    if len(sec_ids) == 1 and sec_ids[0] in overrides:
        base_dates.update({
            k: v for k, v in overrides[sec_ids[0]].items() if not k.startswith("_")
        })
    elif len(sec_ids) > 1:
        print(f"  ⚠️  學生 {anon_id} 於學期 {sem} 橫跨多個 section_id"
              f"（{list(sec_ids)}），行事曆override無法判斷，使用學期預設值")
    return base_dates

def build_per_student_exam_dates(quiz_df, exam_dates_all: dict) -> dict:
    """
    【核心修正】從合併後的 quiz_df 中，依每位學生在每個學期
    （(anon_id, semester) 複合鍵）各自擷取其實際參加的期中考 / 期末考日期。

    v6修正：
    1. global_exam_dates（單一扁平dict）→ exam_dates_all（per-semester keyed，
       Phase 0產出），並改用每筆紀錄自己的 _semester 查對應學期設定，
       不再整批繼承同一份 semester_start（Bug A根因）。
    2. groupby("anon_id") → groupby(["anon_id","_semester"])，兩處
       （主路徑、寬鬆模式）都要改，避免同一學生跨學期的考試紀錄被混在
       一起算最早/最晚日期（Bug B同類模式：重修生跨學期混血）。
    3. 回傳鍵改為 (anon_id, semester) tuple。

    背景說明：
      - 每個 xlsx 包含一學期多班的資料
      - 學生可能跨班上課，考試時間不具班級一致性
      - quiz_df 中每列 = 一位學生的一筆測驗紀錄
      - 標題為「[重要] ... 期中考（114.11.13）」者為正式考試記錄
      - 同一考試，參加不同場次的學生，標題內日期可能不同

    跨學期修正（Fix: 111-1~113-1 標題無 [重要] 前綴時的 fallback 策略）：
      - quiz_df 含 _semester 欄時，先以「相同學期的所有測驗標題中最早/最晚日期」
        嘗試推算考試日期（放寬 [重要] 限制，改用「期中考」/「期末考」直接比對）
      - 此 fallback 僅在標準 [重要] 解析失敗（結果為空）時啟用

    規則：
      - 按 (anon_id, semester) 分組，篩選標題以 [重要] 開頭的期中/期末考記錄
      - 解析標題中的民國年日期
      - 同一學生同一學期若有多筆期中考（複試/補考），取最早日期
      - 同一學生同一學期若有多筆期末考，取最晚日期

    Parameters
    ----------
    quiz_df        : 合併後已匿名化（含 anon_id）的測驗紀錄 DataFrame
    exam_dates_all : Phase 0 產出的 per-semester keyed config { sem: {...} }

    Returns
    -------
    dict : { (anon_id, semester): { "midterm":..., "final":..., ...該學期config其他欄位 } }
    """
    result: dict = {}

    if quiz_df is None or quiz_df.empty:
        return result

    # 偵測標題欄位
    title_col = None
    for candidate in ("測驗標題", "活動名稱", "活動紀錄", "標題", "名稱"):
        if candidate in quiz_df.columns:
            title_col = candidate
            break
    if title_col is None or "anon_id" not in quiz_df.columns:
        return result
    if "_semester" not in quiz_df.columns:
        # 沒有 _semester 欄就無法做複合鍵分組，直接回退舊行為並示警
        print("  ⚠️  quiz_df 缺少 _semester 欄，無法建立複合鍵，回傳空 dict")
        return result

    # 只保留 [重要] 開頭、且含期中/期末的列（使用模組層級 _is_exam_row）
    exam_mask = quiz_df[title_col].apply(_is_exam_row)
    exam_df = quiz_df[exam_mask].copy()

    if exam_df.empty:
        # ── [跨學期 Fix] ──────────────────────────────────────────────────────────
        # exam_df 為空：[重要] 標題均已找到但日期格式無法解析
        # （112-2 以前：(M/D HH:MM) 格式，沒有民國年）
        # 也可能：標題完全沒有 [重要] 前綴（極少見）
        # 策略：放寬為只要含「期中考」/「期末考」，用新版 parse（含 semester 推算）再試一次
        # [BUG-FIX v7.15.17] 原本 `isinstance(t, str) and (_RE_MIDTERM.search(t) or
        # _RE_FINAL.search(t))` 在字串比對成功時，回傳的是 re.Match 物件（truthy但
        # 非bool），比對失敗時回傳 None——混雜 True/Match物件/None 的 object dtype
        # Series，pandas 布林遮罩會把 None 當NaN，直接丟出
        # 「Cannot mask with non-boolean array containing NA / NaN values」。
        # 這條 fallback 分支在既有30個歷史學期的真實資料裡從未真正執行到（因為
        # exam_mask 用嚴格版 _is_exam_row() 都能一次比對成功，不會落到這裡），
        # 所以這個bug過去沒被觸發過；本次因115(1)模擬測試資料而被實測發現，
        # 一併修正，改用 bool() 明確轉型，確保回傳值恆為 True/False。
        loose_mask = quiz_df[title_col].apply(
            lambda t: isinstance(t, str) and bool(_RE_MIDTERM.search(t) or _RE_FINAL.search(t))
        )
        loose_df = quiz_df[loose_mask].copy()
        if loose_df.empty:
            return result

        # 使用新版 parse：傳入每列自己的 _semester
        loose_df["_exam_date"] = loose_df.apply(
            lambda row: _parse_exam_date_from_title(
                str(row[title_col]), str(row.get("_semester", ""))
            ), axis=1
        )
        loose_df["_is_midterm"] = loose_df[title_col].apply(_is_midterm_title)
        loose_df["_is_final"]   = loose_df[title_col].apply(_is_final_title)
        loose_df = loose_df[loose_df["_exam_date"].notna()]
        if loose_df.empty:
            return result

        # 先建立 per-semester 代表日期（同學期全體最早期中/最晚期末）
        sem_dates: dict = {}
        for sem, sg in loose_df.groupby("_semester"):
            mid_d = sorted(sg.loc[sg["_is_midterm"], "_exam_date"].dropna().unique())
            fin_d = sorted(sg.loc[sg["_is_final"],   "_exam_date"].dropna().unique())
            sem_dates[sem] = dict(exam_dates_all.get(sem, {}))
            if mid_d:
                sem_dates[sem]["midterm"] = mid_d[0]
            if fin_d:
                sem_dates[sem]["final"] = fin_d[-1]
        print("  ⚠️  [寬鬆模式] 日期格式為 M/D，依學期推算年份")
        for sem, sd in sem_dates.items():
            print(f"     {sem}：期中 {sd.get('midterm','?')}　期末 {sd.get('final','?')}")

        # v6關鍵修正：改用 (anon_id, _semester) 複合鍵分組，
        # 同一學生跨學期的紀錄不再混算 mid_d/fin_d
        for (anon_id, sem), grp in loose_df.groupby(["anon_id", "_semester"]):
            sd = dict(sem_dates.get(sem, exam_dates_all.get(sem, {})))
            sd = _apply_calendar_override(sd, grp, anon_id, sem)
            mid_d = sorted(grp.loc[grp["_is_midterm"], "_exam_date"].dropna().unique())
            fin_d = sorted(grp.loc[grp["_is_final"],   "_exam_date"].dropna().unique())
            if mid_d: sd["midterm"] = mid_d[0]
            if fin_d: sd["final"]   = fin_d[-1]
            result[(anon_id, sem)] = sd

        print(f"  ✅ 逐人考試日期建立完成（寬鬆，複合鍵）：{len(result)} 組 (anon_id,semester)")
        return result

    # 標準路徑：解析每列的考試日期與類型（傳入 _semester 支援 M/D 格式推算年份）
    exam_df["_exam_date"] = exam_df.apply(
        lambda row: _parse_exam_date_from_title(
            str(row[title_col]), str(row.get("_semester", ""))
        ), axis=1
    )
    exam_df["_is_midterm"] = exam_df[title_col].apply(_is_midterm_title)
    exam_df["_is_final"]   = exam_df[title_col].apply(_is_final_title)

    # 過濾掉無法解析日期的列
    exam_df = exam_df[exam_df["_exam_date"].notna()]

    # v6關鍵修正：groupby("anon_id") → groupby(["anon_id","_semester"])
    for (anon_id, sem), grp in exam_df.groupby(["anon_id", "_semester"]):
        mid_dates = sorted(grp.loc[grp["_is_midterm"], "_exam_date"].unique())
        fin_dates = sorted(grp.loc[grp["_is_final"],   "_exam_date"].unique())

        # v6：改繼承該學期自己的config（Phase 0的exam_dates_all），
        # 不再是單一全域dict，semester_start不會再跨學期殘留
        student_dates = dict(exam_dates_all.get(sem, {}))
        student_dates = _apply_calendar_override(student_dates, grp, anon_id, sem)
        if mid_dates:
            student_dates["midterm"] = mid_dates[0]   # 最早（初試）
        if fin_dates:
            student_dates["final"] = fin_dates[-1]    # 最晚（正式）

        result[(anon_id, sem)] = student_dates

    found = len(result)
    mid_found = sum(1 for v in result.values() if v.get("midterm"))
    fin_found = sum(1 for v in result.values() if v.get("final"))
    print(f"  ✅ 逐人考試日期建立完成：{found} 組 (anon_id,semester)")
    print(f"     期中考有記錄：{mid_found} 組　期末考有記錄：{fin_found} 組")
    if mid_found < found or fin_found < found:
        mid_missing = found - mid_found
        fin_missing = found - fin_found
        print(f"  ⚠️  期中考缺少記錄：{mid_missing} 組，期末考缺少記錄：{fin_missing} 組，將 fallback 使用 config 該學期日期")

    return result

_CALENDAR_DATE_RE = _re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}")


def _looks_like_calendar_column(series: pd.Series) -> bool:
    """
    判斷欄位是否真的是「日曆時間戳」，而非欄名恰好含「時間」二字、
    實際卻是「時長」的欄位（例：教材閱讀紀錄的「時間」欄存的是
    累計閱讀時長 "08:13"、"00:07" 這種 分:秒 格式）。

    根因：validate_semester_calendar 原本只用「欄名是否包含"時間"或
    "日期"」這種寬鬆子字串比對來挑選候選欄位，這類時長欄位一旦被
    pd.to_datetime() 解析，"08:13" 會被當成「執行當下那一天的
    08:13」，混入一個跟任何學期都無關、卻長得像時間戳的假日期。
    目前因為後續只取 min()，且假日期恆等於「執行當日」（必然晚於
    任何歷史學期起始日），才剛好沒有影響到判斷結果——但這是僥倖，
    不是設計保證，欄位比對邏輯本身仍是錯的。

    強化作法：欄名初篩通過後，再抽樣該欄非空值，檢查字串中是否含
    完整日期樣式（YYYY-M-D 或 YYYY/M/D）。純「時:分」或「時:分:秒」
    的時長字串不含年份，一律判定為非日曆欄位而排除。
    """
    if series.empty:
        return False
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    sample = series.dropna().astype(str)
    if sample.empty:
        return False
    # 抽樣即可（欄位格式在同一欄內一致，無需全量掃描）
    sample = sample.head(30)
    return bool(sample.str.contains(_CALENDAR_DATE_RE).any())


_TITLE_CANDIDATES = ("測驗標題", "活動名稱", "活動紀錄", "標題", "名稱")


def _find_title_col(df):
    for c in _TITLE_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _compute_section_timestamps(sheets: dict, sem: str) -> tuple:
    """
    掃描 quiz/activity/reading_log/reading_stats 四個sheet，依 _section_id
    分組取得每組最早時間戳。回傳 (by_section, by_section_bank)：
      - by_section：全欄位掃描結果（驗證用的保守基準）
      - by_section_bank：僅「練習題庫」標題（不論第N週/M-D等命名）的
        最早作答時間（建議值用的精準信號，可能為空）
    v7.2抽出為獨立函式，供 validate_semester_calendar() 與
    generate_config_draft() 共用，避免同一份邏輯維護兩份。
    """
    by_section: dict = {}
    by_section_bank: dict = {}
    for key in ("quiz", "activity", "reading_log", "reading_stats"):
        df = sheets.get(key)
        if df is None or "_semester" not in df.columns:
            continue
        sub = df[df["_semester"].astype(str) == sem]
        if sub.empty:
            continue
        has_section_col = "_section_id" in sub.columns
        groups = sub.groupby("_section_id") if has_section_col else [("", sub)]
        title_col = _find_title_col(sub) if key == "quiz" else None
        for section_id, sec_df in groups:
            for col in sec_df.columns:
                if ("時間" in col) or ("日期" in col):
                    if not _looks_like_calendar_column(sec_df[col]):
                        continue
                    ts = pd.to_datetime(sec_df[col], errors="coerce").dropna()
                    if not ts.empty:
                        by_section.setdefault(section_id, []).append(ts.min())
                        if key == "quiz" and title_col:
                            bank_mask = sec_df[title_col].astype(str).str.contains(
                                "練習題庫", na=False
                            )
                            bank_ts = pd.to_datetime(
                                sec_df.loc[bank_mask, col], errors="coerce"
                            ).dropna()
                            if not bank_ts.empty:
                                by_section_bank.setdefault(section_id, []).append(bank_ts.min())
    return by_section, by_section_bank


def _suggest_section_start(section_id, by_section, by_section_bank):
    """優先採用該section「練習題庫」最早作答時間，沒有則退回全欄位掃描結果。"""
    if section_id in by_section_bank:
        return min(by_section_bank[section_id])
    return min(by_section[section_id])


def compute_calendar_suggestion(sheets: dict, sem: str):
    """
    針對單一學期，算出可直接寫進 config 的建議 entry：
      {"semester_start": "...", "calendar_overrides": {...}}（多section時才有）
    找不到任何可解析時間戳時回傳 None。
    """
    by_section, by_section_bank = _compute_section_timestamps(sheets, sem)
    if not by_section:
        return None

    def _suggest_start(section_id):
        return _suggest_section_start(section_id, by_section, by_section_bank)

    overall_earliest = min(_suggest_start(sid) for sid in by_section)
    labeled_sections = {sid: ts for sid, ts in by_section.items() if sid}
    entry = {"semester_start": str(overall_earliest.date())}
    if len(by_section) > 1 and labeled_sections:
        entry["calendar_overrides"] = {
            section_id: {
                "semester_start": str(_suggest_start(section_id).date()),
                "_label": "請填寫班級說明",
            }
            for section_id in labeled_sections
        }
    return entry


def _prompt_yes_no(question: str) -> bool:
    """簡單Y/N互動提示。非互動環境（無stdin，如排程工作）讀不到輸入時，
    安全預設為「未確認」，交由後續 validate_semester_calendar 硬中斷──
    「不確定就停下來」優先於「不確定就自動放行」，不會假設使用者同意。

    [ENCODING-FIX 2026-09-09] question參數與此函式自身的print()輸出改為
    ASCII/英文，避免Windows cmd/PowerShell在非UTF-8主控台編碼（例如
    chcp 437/850，或輸出被重新導向至記錄檔時）下，input()寫入提示字串
    時拋出UnicodeEncodeError（此例外不是EOFError/OSError，不會被下方
    既有的try/except攔截，會讓整支腳本中斷）。docstring本身維持中文
    （純註解，不會實際輸出到終端）。
    """
    try:
        ans = input(question).strip().lower()
    except (EOFError, OSError):
        print("  (Non-interactive environment detected; input unavailable; treated as not confirmed)")
        return False
    return ans in ("y", "yes")


def _prompt_manual_date(question: str):
    """互動輸入 YYYY-MM-DD 日期；非互動環境或使用者直接按 Enter 放棄則回傳 None。

    [ENCODING-FIX 2026-09-09] 同 _prompt_yes_no() 說明：question參數與此
    函式自身的print()輸出改為ASCII/英文，避免Windows終端非UTF-8編碼環境
    下input()寫入提示字串時崩潰。
    """
    while True:
        try:
            raw = input(question).strip()
        except (EOFError, OSError):
            return None
        if not raw:
            return None
        try:
            return pd.Timestamp(raw)
        except Exception:
            print("  [WARN] Invalid format. Use YYYY-MM-DD (e.g. 2026-09-07), or press Enter to skip")


def _semester_has_selfstudy_titles(sheets: dict, sem: str) -> bool:
    """
    v7.11新增：檢查該學期的測驗紀錄是否存在「自學週」字樣的標題。

    背景：112-1／112-2／113-1三學期因行政特例整學期只有16正課週，另外
    安排W17/W18作為2週補充內容（即「16+2週」結構），這類學期的補充題庫
    常用「自學週1」「自學週2」等非「第X週」格式命名，現行正規表示式
    `第(\\d+)週` 抓不到週次，這些提交紀錄會被整批捨棄（見
    06_merge_grades.py::_extract_quiz_week_num()）。

    只有偵測到這類非標準命名時才需要詢問 Q2（自學週對應週次），避免對
    不需要的學期造成不必要的互動打擾。
    """
    quiz_df = sheets.get("quiz")
    if quiz_df is None or quiz_df.empty or "_semester" not in quiz_df.columns:
        return False
    title_col = "測驗標題" if "測驗標題" in quiz_df.columns else None
    if not title_col:
        return False
    sem_rows = quiz_df[quiz_df["_semester"].astype(str) == str(sem)]
    if sem_rows.empty:
        return False
    return bool(sem_rows[title_col].astype(str).str.contains("自學週", na=False).any())


def _prompt_week_structure(sem: str, has_selfstudy_titles: bool) -> dict:
    """
    v7.11新增：詢問「本學期總週次結構」與（若偵測到「自學週」命名時）
    「自學週落在第幾週」，回傳可直接併入 exam_dates_all[sem] 的欄位 dict
    （total_weeks／course_structure／self_study_weeks）。

    背景：112-1／112-2／113-1三學期因行政特例只有16正課週、另加W17/W18
    補充2週（「16+2週」結構），需要以此類特例學期的方式硬性寫入設定，
    才能讓後續週次相關計算（截斷、題庫週別統計）取得正確資訊。這類特例
    無法從LMS實測資料自動判斷（行政政策事實，非資料本身可推導），因此
    需要人工確認一次；一旦確認並寫回 config，之後同一學期不會再問。

    非互動環境（無stdin）或使用者直接放棄時，安全預設為 18週標準結構、
    不設定 self_study_weeks（維持現行「找不到自學週映射就跳過該筆題庫」
    的安全預設，不會誤植錯誤週次；也與此欄位新增前的既有預設行為一致，
    對既有排程/批次執行沒有影響）。

    [ENCODING-FIX 2026-09-09] 全部print()/input()提示文字改為ASCII/英文，
    避免Windows終端非UTF-8編碼環境下崩潰（見_prompt_yes_no()說明）。
    docstring本身維持中文。
    """
    print(f"\n{'=' * 60}")
    print(f"[USER CONFIRMATION NEEDED] Week structure for semester {sem}")
    print("  Some semesters use a non-standard structure (e.g. 112-1/112-2/113-1")
    print("  have only 16 regular weeks + 2 supplementary weeks W17/W18, i.e. \"16+2\").")
    print(f"{'=' * 60}")
    print("  Q1. Total week structure for this semester?")
    print("      1) 18 weeks (standard)")
    print("      2) 16+2 weeks (16 regular + 2 supplementary)")
    print("      3) Other (manually enter total weeks)")
    try:
        choice1 = input("  Choose (1/2/3, Enter = default 1): ").strip()
    except (EOFError, OSError):
        print("  (Non-interactive environment detected; defaulting to 18-week standard structure)")
        return {"total_weeks": 18}

    if choice1 == "2":
        total_weeks, structure = 18, "16+2"
    elif choice1 == "3":
        manual_weeks = input("  Enter actual total weeks for this semester: ").strip()
        try:
            total_weeks, structure = int(manual_weeks), "custom"
        except ValueError:
            print("  [WARN] Invalid input, defaulting to 18-week standard structure")
            total_weeks, structure = 18, None
    else:
        total_weeks, structure = 18, None

    result: dict = {"total_weeks": total_weeks}
    if structure:
        result["course_structure"] = structure

    if not has_selfstudy_titles:
        return result

    print("\n  Detected non-standard quiz titles containing \"self-study week\" for this semester.")
    print("  Q2. Which weeks are the self-study weeks?")
    print("      1) Week 8 / Week 17")
    print("      2) Week 17 / Week 18")
    print("      3) Other (enter manually, comma-separated, e.g. 8,16)")
    print("      Enter: skip for now (keeps the existing safe default of skipping")
    print("             quiz rows that cannot be mapped to a week)")
    try:
        choice2 = input("  Choose (1/2/3, or Enter to skip): ").strip()
    except (EOFError, OSError):
        choice2 = ""

    if choice2 == "1":
        result["self_study_weeks"] = [8, 17]
    elif choice2 == "2":
        result["self_study_weeks"] = [17, 18]
    elif choice2 == "3":
        manual_raw = input("  Enter week numbers (comma-separated, e.g. 8,16): ").strip()
        try:
            result["self_study_weeks"] = [int(w.strip()) for w in manual_raw.split(",") if w.strip()]
        except ValueError:
            print("  [WARN] Invalid input, self-study week mapping not set for this semester")

    return result


def _semester_has_transition_merged_course(sheets: dict, sem: str) -> bool:
    """
    見規劃書1.2節：偵測該學期是否存在「基礎醫學臨床應用(一)」合併課資料
    （比照 _semester_has_selfstudy_titles() 的偵測模式），只有偵測到時才
    觸發 _prompt_transition_micro_window() 詢問，避免對不相關學期造成
    不必要的互動打擾。任一 sheet 出現 _course_track=="transition_merged_micro"
    即視為命中（_merge_lms_files() 已逐列標記，見該函式）。
    """
    for df in sheets.values():
        if df is None or df.empty or "_semester" not in df.columns or "_course_track" not in df.columns:
            continue
        sem_rows = df[df["_semester"].astype(str) == str(sem)]
        if sem_rows.empty:
            continue
        if (sem_rows["_course_track"] == "transition_merged_micro").any():
            return True
    return False


# TRANSITION_MICRO_WINDOW_DEFAULTS：見規劃書0.1節，取自實測教學計畫課表
# （1151二技進修遠距教學計畫_基礎醫學應用）115(1)課表日期，作為互動詢問的
# 預填建議值（可直接按Enter採用，或手動修改）。未來學期需另外設定，此處
# 僅為115(1)這一個學期的預設值，非通用公式（校歷每學期會變動）。
#
# week18_exam 取教學計畫課表「週18 1/4,5」兩天中「較晚」的一天
# （2027-01-05），因 _window_by_week() 以此值當窗口右界，若取較早的
# 1/4，第二天(1/5)的考試作答紀錄會被窗口過濾整批排除——即使
# _window_by_week() 已將右界正規化到當日23:59:59（見該函式BUG-FIX
# 說明），也只能涵蓋「所設定那一天」的整天，涵蓋不到「設定日之後」的
# 另一天，故此處直接設定為兩天中較晚者，確保兩天都落在窗口內。
TRANSITION_MICRO_WINDOW_DEFAULTS: dict = {
    "1151": {
        "week13_start": "2026-11-30",
        "week18_exam":  "2027-01-05",
    },
}


def _prompt_transition_micro_window(sem: str) -> dict:
    """
    見規劃書1.2節／討論11：詢問「基礎醫學臨床應用(一)」合併課微免部分
    在本學期的week13起始日／week18考試日，回傳可直接併入
    exam_dates_all[sem]["transition_micro_window"] 的欄位 dict。

    比照 _prompt_week_structure() 的既有慣例：非互動環境（無stdin）或
    使用者直接放棄時，安全預設為 None（不設定窗口）——此時下游
    _window_by_week() 呼叫端會因缺少窗口設定而略過本學期的窗口過濾，
    不會誤截斷（見該函式呼叫端邏輯），比硬套錯誤日期更安全。

    TRANSITION_MICRO_WINDOW_DEFAULTS 有該學期預設值時，於提示中顯示供
    使用者參考，直接按Enter即採用建議值。

    [ENCODING-FIX 2026-09-09] 全部print()/input()提示文字改為ASCII/英文，
    避免Windows終端非UTF-8編碼環境下崩潰（見_prompt_yes_no()說明）。
    docstring本身維持中文。

    BUG-FIX（v7.15.13穿透式審查）：原本此處還會多問一題「week17結束日」，
    且與week13_start／week18_exam並列為「三者缺一則整組窗口設定放棄」的
    必填項——但全庫搜尋確認 _window_by_week() 實際呼叫端（見該函式呼叫
    處）只讀取 week13_start／week18_exam 兩個鍵計算窗口，week17_end從未
    被任何下游程式碼消費（07e_transition_micro_reading.py 的checkpoint
    函式docstring明確記載已改版為不做日期切分，不需要任何邊界日）。這代表
    使用者若正確填答了唯二真正有作用的兩個日期，卻恰好漏填或填錯這個
    無人使用的第三題，會導致整組窗口過濾被靜默停用，且錯誤訊息完全不會
    提示問題出在這個不影響過濾邏輯的欄位。屬於前一版設計（教學計畫預留
    week17查核點、後未採用）的殘留欄位，本次移除，不影響任何現有窗口
    過濾行為（因為它從未真正參與過濾計算）。
    """
    defaults = TRANSITION_MICRO_WINDOW_DEFAULTS.get(str(sem), {})

    print(f"\n{'=' * 60}")
    print(f"[USER CONFIRMATION NEEDED] Semester {sem} \"Foundations of Clinical")
    print("  Medicine (I)\" merged course - microbiology/immunology week window")
    print("  (week13 start date / week18 exam date)")
    print(f"{'=' * 60}")

    def _ask_one(label: str, key: str):
        default_val = defaults.get(key)
        hint = f"(Enter = use suggested value {default_val})" if default_val else "(Enter = skip window setting)"
        try:
            raw = input(f"  {label} {hint}: ").strip()
        except (EOFError, OSError):
            print("  (Non-interactive environment detected; input unavailable)")
            return default_val
        if not raw:
            return default_val
        try:
            return str(pd.Timestamp(raw).date())
        except Exception:
            print("  [WARN] Invalid format, use YYYY-MM-DD. Keeping suggested value / unset for this item")
            return default_val

    week13_start = _ask_one("week13 start date", "week13_start")
    week18_exam  = _ask_one("week18 exam date (if exam spans multiple days, enter the latest day)", "week18_exam")

    if not (week13_start and week18_exam):
        print("  [WARN] Dates not fully set; window filtering not applied for this semester")
        return {}

    return {
        "transition_micro_window": {
            "week13_start": week13_start,
            "week18_exam":  week18_exam,
        }
    }


def _detect_incomplete_semesters(data_path: str, behavior_semesters: list) -> set:
    """
    依 data.json 判定「尚未有期末成績」的學期──與 Step 9 incomplete_semesters
    規則一致（該學期所有學生 final_score 皆為 null，見run()內Step 9附近定義：
    「某學期的所有學生 final_score 皆為 null → 該學期尚未獲得學末成績」）。

    只有這類「還在進行中／最新」的學期才需要互動式Y/N確認一次；已結束
    （已有期末成績）的學期直接用LMS實測值自動套用，不問──因為練習題庫
    等實測時間戳對已結束學期而言是完整、不會再變動的資料，逐一人工確認
    的邊際價值很低，且已結束學期不會被提前預警系統用來預測任何真實學生。

    data.json 不存在（例如尚未準備成績、只想先跑LMS behavior分析）時無法
    判斷，退而求其次：只把 behavior_semesters 中字典序最大的一個視為
    「最新/進行中」（4碼 YYYS 格式下字典序恰好等於時間序），其餘視為已
    結束。data.json 備妥後重跑，即會改用真正的期末成績判斷。
    """
    if not behavior_semesters:
        return set()
    if not data_path or not os.path.exists(data_path):
        return {max(behavior_semesters)}
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data_json = json.load(f)
    except Exception:
        return {max(behavior_semesters)}

    students = data_json.get("students", data_json)
    has_final = {sem: False for sem in behavior_semesters}
    for _sid, info in students.items():
        for rec in info.get("records", []):
            sem = rec.get("semester")
            if sem not in has_final or has_final[sem]:
                continue
            val = pd.to_numeric(pd.Series([rec.get("final")]), errors="coerce").iloc[0]
            if pd.notna(val):
                has_final[sem] = True

    return {sem for sem, done in has_final.items() if not done}


def _resolve_in_progress_semester(behavior_semesters: list, incomplete_semesters) -> "str | None":
    """
    v7.8新增：決定 _truncate_by_week() 的 scope_semester——「本次批次中，
    究竟哪個學期才是真正需要套用早期預警週次截斷的『進行中』學期」。

    BUG-FIX (2026-07-12)（交接規格書問題2 / H2 假設確認為根因）：
    舊版邏輯（run() 內原本的 `_in_progress_sem = max(behavior_semesters)`）
    恆取「本次批次中字典序最大的學期」，完全沒檢查該學期的成績是否已經
    結算。當一批次的最新學期其實已經有期末成績（例如114-2成績已結算，
    但仍是該批次behavior_semesters中字典序最大者），週次截斷仍會被誤套用
    在這個「看似最新、實則已結束」的學期上，導致 Week 13 起的
    reading_log／quiz 活動全數被砍成 0（behavior.json 的
    time_profile.weekly_minutes → at_risk_profile.json 的
    weekly_activity_series 逐層繼承這個被截斷的上游資料）。

    修正後：只有「同時滿足①是批次中字典序最大者、②確實被
    _detect_incomplete_semesters() 判定為尚未結算成績」的學期，才會被視為
    需要截斷的進行中學期；否則回傳 None，呼叫端須完全略過本次截斷
    （不可傳空字串——scope_semester="" 在 _truncate_by_week() 的既有語意
    是「沒指定scope，全體套用」，等同退回截斷所有學期的更早期bug，比
    不修正更糟，見該函式docstring的BUG-FIX(2026-07-03)說明）。

    Parameters
    ----------
    behavior_semesters   : 本次批次涵蓋的全部學期代碼列表
    incomplete_semesters : _detect_incomplete_semesters() 回傳的「尚未結算
                            成績」學期集合（set 或 list 皆可）

    Returns
    -------
    str | None : 需要截斷的進行中學期代碼；批次內沒有任何學期符合條件時為 None
    """
    incomplete_set = set(incomplete_semesters or [])
    candidates = [s for s in (behavior_semesters or []) if s in incomplete_set]
    return max(candidates) if candidates else None


def interactive_fill_missing_calendar(
    sheets: dict,
    behavior_semesters: list,
    exam_dates_all: dict,
    config_path: str,
    data_path: str = "",
    tolerance_days: int = 15,
    force_week_structure_sems: "set | frozenset" = frozenset(),
) -> set:
    """
    v7.4：在真正跑 validate_semester_calendar()（硬中斷版）之前，自動補齊
    缺漏的校歷設定──但依「這學期是否已結束」分兩種待遇：

      - 已結束學期：直接用 LMS 實測值（優先「練習題庫」信號，見v7.1）
        自動套用，不詢問。
      - 進行中／最新學期（data.json 尚無期末成績）：這是提前預警系統
        真正會用到、預測結果可能影響真實學生介入決策的學期，才需要互動
        詢問。詢問「僅第一次」──一旦這次執行確認過（Y或手動輸入）並寫回
        config，下次執行時 semester_start 已存在，不會再問。

    回傳本次執行中「已處理過」的 (sem) 或 (sem, section_id) 鍵集合，供
    validate_semester_calendar() 略過重複檢查（該檢查用的是同一份LMS
    實測信號，剛自動套用或剛經人工確認，沒有再檢查一次的必要）。

    force_week_structure_sems : v7.11新增。明確指定要強制詢問「週次結構」
        （總週次18標準/16+2週特例、自學週對應週次，見 _prompt_week_structure()）
        的學期集合，即使該學期成績已結算、原本會自動套用不詢問，仍會
        強制詢問一次（來自 --configure-weeks CLI 參數）。預設空集合時，
        只有尚無期末成績的最新/進行中學期才會詢問，已結算歷史學期完全
        不受影響、維持自動套用不打擾。
    """
    incomplete = _detect_incomplete_semesters(data_path, behavior_semesters)
    touched_keys = set()
    changed = False

    for sem in behavior_semesters:
        cfg = exam_dates_all.get(sem)
        is_incomplete = sem in incomplete

        if cfg is None or not cfg.get("semester_start"):
            suggestion = compute_calendar_suggestion(sheets, sem)
            if suggestion is None:
                continue  # 無可解析時間戳，無法建議，交由後面硬中斷

            # BUG-FIX (2026-07-13)：以下三處原本是 `exam_dates_all[sem] = suggestion`
            # （或全新dict）直接整個覆寫，若該學期在 semester_start 尚未確認前就已經
            # 預先寫入其他欄位（例如 total_weeks／course_structure／self_study_weeks，
            # 如112-2因行政特例需要提前手動填入16+2週設定，但semester_start當時
            # 還沒確認），這裡的覆寫會把那些預先寫入的欄位整組清空。改為與既有
            # cfg（若存在）合併，只讓 suggestion 疊加/覆蓋 semester_start 相關欄位，
            # 不影響其他既有欄位。
            if not is_incomplete:
                suggestion["_auto_generated"] = True
                suggestion["_label"] = (
                    "已結束學期，LMS實測值（練習題庫優先）自動套用，未經人工逐項確認"
                )
                exam_dates_all[sem] = {**(cfg or {}), **suggestion}
                touched_keys.add(sem)
                changed = True
                print(f"  [INFO] Semester {sem} (ended): auto-applied semester_start="
                      f"{suggestion['semester_start']}")
                continue

            print(f"\n{'=' * 60}")
            print(f"[USER CONFIRMATION NEEDED] Semester {sem} (no final grades yet, treated as latest/in-progress)")
            print("  semester_start not yet set. Based on LMS observed data (earliest quiz-bank")
            print("  submission preferred)")
            print(f"  Suggested value: {suggestion['semester_start']}")
            if suggestion.get("calendar_overrides"):
                for sid, ov in suggestion["calendar_overrides"].items():
                    print(f"    section \"{sid}\" suggestion: {ov['semester_start']}")
            print(f"{'=' * 60}")
            if _prompt_yes_no("  Use the suggested value above? (Y/N): "):
                suggestion["_confirmed_by_user"] = True
                exam_dates_all[sem] = {**(cfg or {}), **suggestion}
            else:
                manual = _prompt_manual_date("  Enter the correct semester start date (YYYY-MM-DD, Enter to skip): ")
                if manual is None:
                    continue
                exam_dates_all[sem] = {
                    **(cfg or {}),
                    "semester_start": str(manual.date()),
                    "_confirmed_by_user": True,
                }
            touched_keys.add(sem)
            changed = True
            continue

        # cfg 已存在且有 semester_start：檢查底下是否有新 section
        by_section, by_section_bank = _compute_section_timestamps(sheets, sem)
        base_start = pd.Timestamp(cfg["semester_start"])
        overrides = dict(cfg.get("calendar_overrides", {}))
        for section_id, timestamps in by_section.items():
            if not section_id or section_id in overrides:
                continue
            earliest = min(timestamps)
            delta = abs((earliest - base_start).days)
            if delta <= tolerance_days:
                continue
            suggested = _suggest_section_start(section_id, by_section, by_section_bank)

            if not is_incomplete:
                overrides[section_id] = {
                    "semester_start": str(suggested.date()),
                    "_label": "已結束學期新section，LMS實測值自動套用",
                    "_auto_generated": True,
                }
                print(f"  [INFO] Semester {sem} / section \"{section_id}\" (ended): "
                      f"auto-added override={suggested.date()}")
            else:
                print(f"\n{'=' * 60}")
                print(f"[USER CONFIRMATION NEEDED] Semester {sem} / section \"{section_id}\" (latest/in-progress)")
                print(f"  Differs from confirmed semester_start ({cfg['semester_start']}) by {delta} days, "
                      f"exceeding the {tolerance_days}-day tolerance")
                print(f"  Suggested independent start date for this section based on LMS observed data: {suggested.date()}")
                print(f"{'=' * 60}")
                if _prompt_yes_no("  Use the suggested value above? (Y/N): "):
                    overrides[section_id] = {
                        "semester_start": str(suggested.date()),
                        "_label": "請填寫班級說明",
                        "_confirmed_by_user": True,
                    }
                else:
                    manual = _prompt_manual_date(
                        "  Enter the correct start date for this section (YYYY-MM-DD, Enter to skip): "
                    )
                    if manual is None:
                        continue
                    overrides[section_id] = {
                        "semester_start": str(manual.date()),
                        "_label": "請填寫班級說明",
                        "_confirmed_by_user": True,
                    }
            cfg["calendar_overrides"] = overrides
            exam_dates_all[sem] = cfg
            touched_keys.add((sem, section_id))
            changed = True

    # v7.11新增：週次結構確認（總週次18標準 vs 16+2特例；自學週對應週次）。
    # 獨立於上面「semester_start是否已結束/進行中」的分支之外——這個結構性
    # 資訊（是否16+2週、自學週落在第幾週）是行政政策事實，跟該學期的
    # 成績是否已結算無關。但預設只對「進行中」學期詢問（維持與
    # semester_start確認相同的既有原則：已結算歷史學期自動套用、完全
    # 不打擾，見test_interactive_confirm.py測試2a的既有保證）；已結算的
    # 歷史學期（如112-1/112-2/113-1因行政特例為16+2週）需要透過
    # force_week_structure_sems（--configure-weeks CLI參數）明確指定才會
    # 詢問——這是唯一能讓已結算歷史學期主動觸發這組確認的方式，避免
    # 「每個沒設定total_weeks的歷史學期都跳出來問」造成不必要的互動負擔。
    # 只針對 exam_dates_all[sem] 尚未設定 total_weeks 的學期詢問一次；
    # 一旦這次執行確認過並寫回 config，下次執行時已存在，不會再問。
    for sem in behavior_semesters:
        cfg = exam_dates_all.get(sem)
        if cfg is None or cfg.get("total_weeks") is not None:
            continue
        is_incomplete = sem in incomplete
        if not (is_incomplete or sem in force_week_structure_sems):
            continue
        has_selfstudy = _semester_has_selfstudy_titles(sheets, sem)
        week_structure = _prompt_week_structure(sem, has_selfstudy)
        exam_dates_all[sem] = {**cfg, **week_structure}
        touched_keys.add(sem)
        changed = True

    # 見規劃書1.2節／討論11：基礎醫學臨床應用(一)合併課微免部分週次窗口
    # 確認（week13起始日／week18考試日）。獨立於上面的
    # total_weeks確認迴圈之外——這是「該學期是否存在合併課」的偵測結果，
    # 跟該學期是否為16+2週特例無關，兩者可能同時成立也可能互斥（雖然
    # 目前尚未出現過這種情況）。只針對 exam_dates_all[sem] 尚未設定
    # transition_micro_window、且該學期確實偵測到合併課LMS資料的學期
    # 詢問一次；一旦確認並寫回 config，下次執行時已存在，不會再問。
    for sem in behavior_semesters:
        cfg = exam_dates_all.get(sem, {})
        if cfg.get("transition_micro_window") is not None:
            continue
        if not _semester_has_transition_merged_course(sheets, sem):
            continue
        window_cfg = _prompt_transition_micro_window(sem)
        if window_cfg:
            exam_dates_all[sem] = {**cfg, **window_cfg}
            touched_keys.add(sem)
            changed = True

    if changed:
        raw = {"semesters": exam_dates_all}
        cfg_dir = os.path.dirname(config_path)
        if cfg_dir:
            os.makedirs(cfg_dir, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
        print(f"\n  ✅ 已將本次校歷確認結果寫回 {config_path}\n")

    return touched_keys


def validate_semester_calendar(
    sheets: dict,
    behavior_semesters: list,
    exam_dates_all: dict,
    tolerance_days: int = 15,
    skip_keys: set = None,
) -> None:
    """
    Phase 0 決議1：每個 behavior_semesters 中的學期，比對「LMS資料實測
    最早行為時間戳」與 config 的 semester_start，超過 tolerance_days 直接
    raise，強制人工確認，避免 Bug A 那種「config沒隨學期更新」的情境
    無聲流過。

    比較基準刻意用「最早行為時間戳」而非考試日期——期中/期末考通常在
    開學後1-3個月，拿來驗證「semester_start是否忘記更新」意義不大。

    v6修正（22-7規劃書Phase 2，問題1）：同一學期代碼下可能存在多套行事曆
    （例：正規班 vs 暑期壓縮重修班），改依 (semester, _section_id) 分組
    各自驗證，而非整學期只算一組最早時間戳。沒有 _section_id 欄（或
    值為空）的資料視為單一分組，行為與修改前完全相同（向下相容）。
    每個 (semester, section) 各自對照 config 的 calendar_overrides（若有）
    或該學期預設 semester_start；找不到對應 override 時，錯誤訊息會
    主動印出可直接貼進 config 的建議 JSON 片段。
    """
    skip_keys = skip_keys or set()
    for sem in behavior_semesters:
        cfg = exam_dates_all.get(sem)

        if sem in skip_keys:
            print(f"  ⏭️  學期 {sem}：本次執行已自動套用/人工確認過，略過重複檢查")
            continue

        # v7修正：把 by_section 時間戳計算搬到 cfg 檢查「之前」。
        # 根因：原本 cfg 整學期缺席時直接 raise，完全不看 sheets，
        # 導致錯誤訊息只會叫使用者「自己去補」，卻不像下面 per-section
        # 分支一樣主動算出建議值——兩種「缺設定」情境（整學期缺 vs 缺
        # 該學期底下某個 section 的 override）給的線索不對稱，這是原始
        # 設計遺漏，並非刻意行為。搬移後兩種分支都能用同一份 by_section
        # 產生可直接複製貼上的 JSON 建議。
        by_section, by_section_bank = _compute_section_timestamps(sheets, sem)

        if cfg is None or not cfg.get("semester_start"):
            if by_section:
                suggested_entry = compute_calendar_suggestion(sheets, sem)
                snippet_json = json.dumps(
                    {sem: suggested_entry}, ensure_ascii=False, indent=2
                )
                indented = "\n    ".join(snippet_json.splitlines())
                suggestion = (
                    '\n  （LMS資料實測建議值，優先取「練習題庫」最早作答日、'
                    '無題庫資料則退回全欄位最早時間戳，仍請人工核對校方正式'
                    '開學日再採用）'
                    '\n  建議於 config 的 "semesters" 下新增：\n    ' + indented
                    + '\n  或直接執行 `python lms_etl.py --generate-config ...`'
                    '自動補齊整份 config（已存在的學期不會被覆蓋）。'
                )
            else:
                suggestion = "\n  （此學期LMS資料中無可解析的時間戳欄位，需人工確認開學日）"
            raise ValueError(
                f"校歷驗證失敗：學期 {sem} 不在 config 中或缺少 semester_start。"
                f"請先於 config/exam_dates.json 的 \"semesters\" 下新增/補齊 "
                f"\"{sem}\" 項目，人工確認後再重跑。{suggestion}"
            )

        overrides = cfg.get("calendar_overrides", {})

        if not by_section:
            print(f"  ⚠️  學期 {sem}：LMS資料無可解析時間戳，略過校歷驗證")
            continue

        for section_id, timestamps in by_section.items():
            if (sem, section_id) in skip_keys:
                print(f"  ⏭️  學期 {sem} / section「{section_id}」：本次執行已自動套用/"
                      f"人工確認過，略過重複檢查")
                continue
            earliest_activity = min(timestamps)
            sec_cfg = overrides.get(section_id, {})
            effective_start = sec_cfg.get("semester_start") or cfg["semester_start"]
            configured_start = pd.Timestamp(effective_start)
            delta_days = abs((earliest_activity - configured_start).days)
            label = sec_cfg.get("_label", section_id or "（未標記section）")

            if delta_days > tolerance_days:
                suggestion = ""
                if section_id and not sec_cfg:
                    suggested_date = _suggest_section_start(section_id, by_section, by_section_bank).date()
                    suggestion = (
                        f'\n  建議於 config 的 "{sem}" 底下新增：\n'
                        f'    "calendar_overrides": {{"{section_id}": '
                        f'{{"semester_start": "{suggested_date}", '
                        f'"_label": "請填寫班級說明"}}}}'
                    )
                raise ValueError(
                    f"校歷驗證失敗：學期 {sem} / section「{label}」的有效 "
                    f"semester_start={effective_start!r}，但LMS資料最早時間戳為 "
                    f"{earliest_activity.date()}，相差 {delta_days} 天"
                    f"（門檻 {tolerance_days} 天）。這正是 Bug A 的根因模式，或"
                    f"該section屬於獨立行事曆（尚未設定 calendar_overrides），"
                    f"請確認後再重跑。（刻意硬中斷，不接受略過）{suggestion}"
                )
            print(f"  ✅ 學期 {sem} / section「{label}」校歷驗證通過"
                  f"（{effective_start}, 實測{earliest_activity.date()}, 差距{delta_days}天）")


def generate_config_draft(
    sheets: dict,
    behavior_semesters: list,
    config_path: str,
    tolerance_days: int = 15,
    data_path: str = "",
) -> dict:
    """
    v7.2新增；v7.3改為逐section檢查（原本只看學期層級，會漏掉「學期已
    確認、但底下多了一個新section」的情況——例如原本只有正規班，某天多
    了一個重修班，若只看學期層級會誤以為已經處理過而整學期跳過）；
    v7.4補上與 interactive_fill_missing_calendar() 一致的已結束/進行中
    分級標籤（原本無論哪個學期都套用同一句 "_label"，少了風險分級資訊
    ——這支是離線、非互動的批次工具，本來就不會問Y/N，但產生的草稿仍
    應該讓事後review的人一眼看出「哪些是已結束、風險低」vs「哪些是最新
    /進行中、請務必仔細核對」，兩支函式對同一件事的判斷標準需一致，
    否則容易造成「用--generate-config產生的草稿」跟「正常執行互動確認」
    對同一學期給出不同風險評估的困惑）。

    對應「config為何不能自己產生」的提問——生成本身可以完全自動化，
    人工確認的是「這個生成值是否等於校方正式開學日」，這兩件事要分開看：
      - 生成（把LMS實測值轉成合法 config JSON）：本函式做，全自動。
      - 確認（生成值是否等於官方行事曆）：無法自動化，因為程式沒有
        校方正式行事曆這個外部真相來源可以核對，只能拿行為資料當代理
        指標（這正是 validate_semester_calendar 存在的原因）。

    合併規則（安全、不覆蓋）：
      - 學期已存在且已有 semester_start：
          - 底下每個 section 若已有 calendar_overrides，或其實測時間戳
            與已確認的 semester_start 差距在容忍範圍內 → 不動。
          - 底下有「新 section」且差距超出容忍範圍、又沒有對應
            override → 只新增該 section 的 calendar_overrides（不動
            semester_start 本身，因為那是人工已確認過的值）。
      - 學期不存在或缺 semester_start → 用 compute_calendar_suggestion()
        算出來的值整個填入，並標註 "_auto_generated": true；已結束學期
        與最新/進行中學期給不同的 "_label" 文字，供事後review分辨風險。
    """
    incomplete = _detect_incomplete_semesters(data_path, behavior_semesters)

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if "semesters" not in raw:
            raise ValueError(
                f"config 格式不相容：{config_path} 未使用新版 per-semester keyed "
                f"結構（缺少頂層鍵 'semesters'），無法安全合併。"
            )
    else:
        raw = {"semesters": {}}

    semesters_cfg = raw["semesters"]
    added, kept, section_added, skipped = [], [], [], []

    for sem in behavior_semesters:
        existing = semesters_cfg.get(sem)

        if existing and existing.get("semester_start"):
            # 學期本身已人工確認，仍逐 section 檢查是否有「新班級」需要
            # 獨立的 calendar_overrides（不動 semester_start 本身）。
            by_section, by_section_bank = _compute_section_timestamps(sheets, sem)
            base_start = pd.Timestamp(existing["semester_start"])
            overrides = dict(existing.get("calendar_overrides", {}))
            new_sections = []
            for section_id, timestamps in by_section.items():
                if not section_id or section_id in overrides:
                    continue
                earliest = min(timestamps)
                delta = abs((earliest - base_start).days)
                if delta > tolerance_days:
                    suggested = _suggest_section_start(section_id, by_section, by_section_bank)
                    overrides[section_id] = {
                        "semester_start": str(suggested.date()),
                        "_label": "請填寫班級說明",
                        "_auto_generated": True,
                    }
                    new_sections.append(section_id)
            if new_sections:
                existing["calendar_overrides"] = overrides
                semesters_cfg[sem] = existing
                section_added.append(f'{sem}: {new_sections}')
            else:
                kept.append(sem)
            continue

        suggestion = compute_calendar_suggestion(sheets, sem)
        if suggestion is None:
            skipped.append(sem)
            continue
        suggestion["_auto_generated"] = True
        if existing and existing.get("_label"):
            suggestion["_label"] = existing["_label"]
        elif sem in incomplete:
            suggestion["_label"] = (
                "⚠️ 最新/進行中學期（尚無期末成績），由LMS實測值自動產生，"
                "提前預警系統會用到此學期，請務必核對校方正式開學日"
            )
        else:
            suggestion["_label"] = (
                "已結束學期，由LMS實測值（練習題庫優先）自動產生，"
                "風險較低但建議仍核對校方正式開學日"
            )
        semesters_cfg[sem] = suggestion
        added.append(sem)

    raw["semesters"] = semesters_cfg
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    print(f"\n  ✅ 已寫入 {config_path}")
    print(f"     新增學期：{added or '（無）'}")
    print(f"     既有學期新增section override：{section_added or '（無）'}")
    print(f"     維持不變（已完整確認）：{kept or '（無）'}")
    if skipped:
        print(f"     ⚠️ 無法產生建議（LMS無可解析時間戳，需人工填寫）：{skipped}")
    print("     請務必核對新增/補齊的項目是否符合校方正式行事曆，"
          "確認無誤後即可移除 \"_auto_generated\" 標記。")
    return raw


def save_exam_dates(
    detected: dict,
    existing: dict,        # v6：per-semester keyed 全體 dict {sem: {...}}
    output_path: str,
    semester: str = "",
    course_meta: dict = None,
) -> dict:
    """
    將偵測出的考試日期寫回 exam_dates.json（作為下次執行的參考）。
    detected 為該學期學生期中/期末日期的統計值（最早期中、最晚期末）。

    v6修正：只更新 semester 對應的那一個子項目，其餘學期的
    semester_start 完全不動（Bug A 根因：原本整份扁平 dict 共用，
    寫回時會把其他學期的 semester_start 一併覆蓋／殘留）。
    """
    if not semester:
        print("  ⚠️  save_exam_dates：semester 為空，無法定位要更新哪個學期，略過寫回")
        return existing

    merged_all = dict(existing)
    sem_entry  = dict(merged_all.get(semester, {}))

    updated_fields = []
    for key in ("midterm", "final"):
        if detected.get(key):
            old = sem_entry.get(key)
            sem_entry[key] = detected[key]
            if old != detected[key]:
                updated_fields.append(f"{semester}.{key}: {old!r} → {detected[key]!r}")

    if course_meta:
        sem_entry.setdefault("course_id",   course_meta.get("course_id", ""))
        sem_entry.setdefault("course_name", course_meta.get("course_name", ""))

    merged_all[semester] = sem_entry
    dir_part = os.path.dirname(output_path)
    os.makedirs(dir_part if dir_part else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"semesters": merged_all}, f, ensure_ascii=False, indent=2)

    if updated_fields:
        print(f"  📅 自動更新 {output_path}：")
        for msg in updated_fields:
            print(f"     {msg}")
    else:
        print(f"  📅 exam_dates.json 學期 {semester} 日期無變更（{output_path}）")
    return merged_all

# ── 主流程 ────────────────────────────────────────────────────
def run(args):
    print("=" * 60)
    print("  教學成效儀表板 — LMS ETL Phase 2")
    print("=" * 60)

    # ── Step 0：讀取設定 ──
    # v6：exam_dates_all 為 per-semester keyed 全體 dict {"1141": {...}, "1142": {...}}
    exam_dates_all = load_config(args.config)
    # 學期代碼不由使用者輸入，Step 1 後由 xlsx 檔名自動解析；
    # course_id/course_name 需等 semester 確定後才知道查哪個學期的值（v6後移）
    course_meta = {
        "semester":    "",
        "course_id":   "",
        "course_name": "",
    }

    # ── Step 1：讀取 LMS Excel ──
    print("\n[Step 1] 讀取 LMS Excel")
    lms_files = _resolve_lms_files(args.lms)
    if not lms_files:
        print("  ❌ 找不到任何 .xlsx 檔案，請確認 --lms 路徑")
        sys.exit(1)

    print(f"  📂 共 {len(lms_files)} 個檔案")
    sheets, course_info_by_semester = _merge_lms_files(lms_files)

    # 自動從 xlsx 檔名解析學期代碼（單次迴圈，同時取得主學期與全部學期列表）
    semester, behavior_semesters = _infer_and_collect_semesters(lms_files)
    _primary_cfg = exam_dates_all.get(semester, {})
    course_meta["semester"]      = semester
    # BUG-FIX (2026-07-12)：course_id/course_name 原本只查 exam_dates.json
    # 的 config（該檔案 schema 從未實際儲存這兩個欄位，只有 semester_start/
    # calendar_overrides，恆回傳空字串預設值）。真正的原始值其實存在於LMS
    # Excel左上角儲存格，已由 _merge_lms_files() 解析為 course_info_by_semester
    # （見該函式與 01_load_lms_excel.py::_extract_course_header() 的修正）。
    # 改為：config 若有人工填寫值則優先採用（尊重人工覆寫，例如課程改名後
    # 手動更正），查無才退回 LMS 實測自動解析值。
    _auto_course = course_info_by_semester.get(semester, {})
    course_meta["course_id"]     = _primary_cfg.get("course_id")   or _auto_course.get("course_id", "")
    course_meta["course_name"]   = _primary_cfg.get("course_name") or _auto_course.get("course_name", "")
    # v7.8新增：逐學期課程代碼/名稱對照表，供 07b_at_risk_profile.py 的
    # by_semester.{sem} 節點各自查表使用（不可全部 8 個學期共用同一個值，
    # 見交接規格書§1.5第3點——不同學期可能對應不同課程代碼/合併班課程名稱）。
    course_meta["by_semester"] = {
        sem: {
            "course_id":   exam_dates_all.get(sem, {}).get("course_id")
                            or course_info_by_semester.get(sem, {}).get("course_id", ""),
            "course_name": exam_dates_all.get(sem, {}).get("course_name")
                            or course_info_by_semester.get(sem, {}).get("course_name", ""),
        }
        for sem in behavior_semesters
    }
    course_meta["semesters"] = behavior_semesters
    course_meta["semester_range"] = format_semester_range(behavior_semesters)
    scope_label = f"學期：{semester}" if semester else "學期：全部（跨學期模式）"
    if behavior_semesters:
        scope_label += f"｜行為資料年度：{course_meta['semester_range']}"
    print(f"  {scope_label}  課程：{course_meta['course_name']}")

    # v7.2新增：--generate-config 模式，讀完檔案、算完 behavior_semesters
    # 後就地把 config 補齊，不繼續跑後面的去識別化/特徵工程等步驟。
    if getattr(args, "generate_config", False):
        print("\n[--generate-config] 開始掃描並補齊 config...")
        generate_config_draft(sheets, behavior_semesters, args.config, data_path=args.data)
        print("[--generate-config] 完成，請核對後重新執行（不加此參數）以跑完整pipeline。")
        return

    # v7.4新增：單一執行內建互動式確認（Y/N），依「已結束/進行中」分流：
    # 已結束學期直接用LMS實測值自動套用不問；只有尚無期末成績的最新學期
    # 才會詢問，且只問一次（下次執行時semester_start已存在，不會再問）。
    # 非互動環境（無stdin，例如排程工作）會自動退回「維持缺漏」，交由
    # 下面 validate_semester_calendar() 硬中斷——安全預設是「不確定就
    # 停下來」，不是「不確定就自動放行」。
    confirmed_keys = interactive_fill_missing_calendar(
        sheets, behavior_semesters, exam_dates_all, args.config, data_path=args.data,
        force_week_structure_sems=frozenset(
            s.strip() for s in args.configure_weeks.split(",") if s.strip()
        ),
    )

    # v6新增（決議1）：校歷驗證。LMS資料最早行為時間戳應落在 config 記載的
    # semester_start 附近（±15天＝加退選期14天+1天緩衝），避免 Bug A 那種
    # 「config 沒隨學期更新」的情境無聲流過。剛剛在本次執行中經人工當場
    # 確認（Y或手動輸入）的項目不再重複檢查——人已經看過建議值並做出決定，
    # 沒被動過的舊項目仍照原本規則檢查。
    validate_semester_calendar(
        sheets, behavior_semesters, exam_dates_all, skip_keys=confirmed_keys
    )

    # ── Step 2：去識別化 ──
    print("\n[Step 2] 去識別化")
    salt = m02.load_salt(args.salt)

    id_col = _detect_id_col(sheets)

    # NEW（0806 護理系統計/模型污染 預防性根除）：須在 _collect_student_ids()
    # 之前執行——一旦非護理系關聯學生的帳號進了 all_sids/lookup_table，
    # 後續 anonymize/merge 就已將其視為正式追蹤對象。見函式定義處完整說明。
    print("\n[Step 1b] 非護理系關聯學生過濾")
    sheets = _filter_non_nursing_students(sheets, id_col)

    all_sids = _collect_student_ids(sheets, id_col)
    print(f"  原始學號數：{len(all_sids)}")

    lookup_table = m02.build_lookup_table(all_sids, salt)
    if not args.dry_run:
        m02.save_lookup_table(lookup_table, "lookup_table.json")
    else:
        print("  [DRY RUN] lookup_table.json 未寫入")

    sheets = _anonymize_sheets(sheets, id_col, salt)

    # ── (anon_id, semester) pair 全集合（Step 2b 考試日期 fallback、
    #    Step 4/5 學生名冊複合鍵主迴圈共用；單一事實來源，避免同一
    #    bug 分散在兩處各自修一次）──
    # 用匿名化後、Step 3 分類/週數截斷「之前」的原始表計算：學期判定
    # 與教材分類、早期預警用的週數截斷無關，用截斷前的完整資料才不會
    # 漏抓學期。v6：不再「多學期取最新」收斂成單一學期（_build_semester_by_anon
    # 已淘汰），改由 _collect_anon_semester_pairs() 回傳完整 pair 集合，
    # 讓 Phase 3 主迴圈能為每個 (anon_id,semester) 各自產生一筆 record。
    _anon_semester_pairs = _collect_anon_semester_pairs(sheets)

    # ── Step 2b：自動偵測考試日期（匿名化後，依學生逐一解析）──
    # 必須在匿名化之後執行，因為需要 anon_id 作為索引
    print("\n[Step 2b] 自動偵測期中/期末考試日期（逐人）")

    # per_student_exam_dates: { (anon_id, semester): { midterm, final, ...該學期config其他欄位 } }
    per_student_exam_dates: dict = {}

    if args.no_auto_dates:
        print("  （已停用自動偵測，使用 config 固定值套用於所有學生）")
        for anon_id, sem in _anon_semester_pairs:
            _cfg = exam_dates_all.get(sem, {})
            if not _cfg.get("midterm") or not _cfg.get("final"):
                missing = [k for k in ("midterm", "final") if not _cfg.get(k)]
                print(f"  ⚠️  config 學期 {sem} 缺少考試日期：{missing}")
            per_student_exam_dates[(anon_id, sem)] = dict(_cfg)
    else:
        quiz_df_for_dates = _safe_get_df(sheets, "quiz")
        per_student_exam_dates = build_per_student_exam_dates(
            quiz_df=quiz_df_for_dates,
            exam_dates_all=exam_dates_all,
        )

        # 找不到任何記錄時的 fallback
        if not per_student_exam_dates:
            print("  ⚠️  quiz_df 中未找到任何 [重要] 期中/末考記錄，全體使用 config 各學期固定值")
            for anon_id, sem in _anon_semester_pairs:
                per_student_exam_dates[(anon_id, sem)] = dict(exam_dates_all.get(sem, {}))
        else:
            # 補齊：_anon_semester_pairs 中有但 quiz 未記錄的 (anon_id,semester)
            # （未參加考試），fallback 至同學期代表日期
            #
            # BUG-STUDENT-SEMESTER 修正：原本只查 activity_df 一張表，
            # activity_df 對「當前進行中學期」常缺完整匯出，導致本學期大多數
            # 沒有 quiz 記錄（未參加考試）的學生反查不到、fallback 到全域值，
            # 用錯學期的考試日期算出偏後的週次特徵。改用 Step 2 已建立的
            # _anon_semester_pairs（四表聯集）取代單查 activity_df。
            _sem_date_cache: dict = {}  # { semester_code: exam_dates_dict }

            def _get_sem_fallback(target_sem: str) -> dict:
                """同學期代表值 fallback：呼叫時 (anon_id, target_sem) 保證尚未存在
                於 per_student_exam_dates（見下方呼叫處的 if 判斷），故不做該生
                自己的直接查詢；改為借用同學期中任一位已解析出考試日期的學生
                之代表值，查無則退回 exam_dates_all 的學期預設值。"""
                if not target_sem:
                    return dict(exam_dates_all.get("", {}))
                if target_sem not in _sem_date_cache:
                    # sorted()：_anon_semester_pairs 是 set，Python 字串雜湊
                    # 預設逐process隨機化，若直接對 set 走 next() 取「第一個」，
                    # 同一批輸入資料在不同次執行下可能選到不同學生的代表值
                    # （當同學期不同班/section考試日期不同時，會造成同一批
                    # 輸入資料重跑產生不同結果，不可重現）。排序後固定取
                    # anon_id字典序最小者，確保同輸入恆定產生同輸出。
                    same_sem_anons = sorted({a for a, sem in _anon_semester_pairs if sem == target_sem})
                    rep_dates = next(
                        (per_student_exam_dates[(a, target_sem)] for a in same_sem_anons
                         if (a, target_sem) in per_student_exam_dates),
                        None
                    )
                    _sem_date_cache[target_sem] = rep_dates if rep_dates else dict(exam_dates_all.get(target_sem, {}))
                return dict(_sem_date_cache[target_sem])

            for anon_id, sem in _anon_semester_pairs:
                if (anon_id, sem) not in per_student_exam_dates:
                    per_student_exam_dates[(anon_id, sem)] = _get_sem_fallback(sem)

        # 寫回 config：取所有學生中最早期中 / 最晚期末（作為全域參考值，僅適用單一學期批次）
        all_mids = sorted(set(
            v["midterm"] for v in per_student_exam_dates.values() if v.get("midterm")
        ))
        all_fins = sorted(set(
            v["final"] for v in per_student_exam_dates.values() if v.get("final")
        ))
        global_summary = {
            "midterm": all_mids[0]  if all_mids else None,
            "final":   all_fins[-1] if all_fins else None,
        }

        if global_summary["midterm"]:
            print(f"  全體期中考範圍：{all_mids[0]} ～ {all_mids[-1]}")
        if global_summary["final"]:
            print(f"  全體期末考範圍：{all_fins[0]} ～ {all_fins[-1]}")

        if not args.dry_run and len(behavior_semesters) <= 1:
            exam_dates_all = save_exam_dates(
                detected=global_summary,
                existing=exam_dates_all,
                output_path=args.config,
                semester=semester,   # 已由 xlsx 檔名自動解析
                course_meta=course_meta,
            )
        elif not args.dry_run:
            print("  📅 跨學期 LMS 資料不寫回 exam_dates.json，避免以多年範圍覆蓋單一學期設定")
        else:
            sem_entry = dict(exam_dates_all.get(semester, {}))
            for key in ("midterm", "final"):
                if global_summary.get(key):
                    sem_entry[key] = global_summary[key]
            exam_dates_all[semester] = sem_entry
            print("  [DRY RUN] exam_dates.json 未寫入")

    # ── Step 3：標籤解析 ──
    print("\n[Step 3] 教材標籤解析")
    if sheets.get("activity") is not None:
        title_col = _detect_title_col(sheets["activity"])
        sheets["activity"] = m03.categorize_activity_df(sheets["activity"], title_col)
        material_counts = m03.count_materials_by_category(sheets["activity"], title_col)
        print(f"  教材件數統計：{material_counts}")

    else:
        material_counts = {cat: 0 for cat in m03.CATEGORIES}
        material_counts["UNTAGGED"] = 0

    if sheets.get("reading_stats") is not None:
        reading_title_col = _detect_title_col(sheets["reading_stats"])
        if not reading_title_col:
            # fallback：取第一個非帳號/學號/anon_id/_semester 的欄位作為教材標題欄
            skip_keywords = ("帳號", "學號", "anon_id", "_semester", "masked_id", "_section_id")
            candidates = [
                c for c in sheets["reading_stats"].columns
                if not any(k in c for k in skip_keywords)
            ]
            reading_title_col = candidates[0] if candidates else ""
            if reading_title_col:
                print(f"  ⚠️  reading_stats 標題欄 fallback 至：'{reading_title_col}'")
            else:
                print("  ⚠️  reading_stats 找不到標題欄，aud/vid/tut 時間將無法計算")
        if reading_title_col:
            sheets["reading_stats"] = m03.categorize_activity_df(
                sheets["reading_stats"], reading_title_col
            )
            rs_counts = sheets["reading_stats"]["category"].value_counts().to_dict()
            print(f"  reading_stats tag 結果：{rs_counts}")

    if sheets.get("quiz") is not None and "測驗標題" in sheets["quiz"].columns:
        sheets["quiz"] = m03.categorize_activity_df(sheets["quiz"], "測驗標題")

    # BUG-FIX (2026-07-12，交接規格書問題2 / H2)：進行中學期原本恆以
    # behavior_semesters 中字典序最大者判斷（例如 ["1141","1142"] → "1142"），
    # 未檢查該學期是否「真的」尚未結算成績——若批次中最新的學期其實已經
    # 有期末成績（如114-2成績已結算），仍會被誤判為進行中而套用截斷，
    # Week 13起活動全數歸零。改為同時比對 _detect_incomplete_semesters()
    # （與 interactive_fill_missing_calendar() 共用同一份「是否已結算成績」
    # 判斷依據，避免兩處邏輯不同步）；批次內沒有任何學期符合「尚未結算」
    # 才會回傳 None（見 _resolve_in_progress_semester() docstring）。
    # v6：Phase 3b（決議9）也需要這個判斷，且與 week_limit 是否啟用無關，
    # 故仍無條件執行，避免 NameError。
    _incomplete_semesters = _detect_incomplete_semesters(args.data, behavior_semesters)
    _in_progress_sem = _resolve_in_progress_semester(behavior_semesters, _incomplete_semesters)

    if args.week_limit and _in_progress_sem:
        # BUG-FIX (2026-07-03)：週數截斷只該套用在「目前仍在進行中」的學期，
        # 不該波及已結束、資料本來就完整的歷史學期（例如 114-1 被誤砍到只剩前 12 週）。
        # 留存一份未截斷的題庫紀錄，專供前端 UI 儀表板的「各週題庫作答強度」圖表使用。
        # 否則 114-2 的週次圖表會被截斷或因為沒有考期而完全消失。
        untruncated_quiz_df = sheets.get("quiz").copy() if sheets.get("quiz") is not None else None
        
        sheets["quiz"] = _truncate_by_week(
            sheets.get("quiz"), id_col_candidates=("anon_id", "帳號"),
            time_col_candidates=("測驗時間",),
            per_student_exam_dates=per_student_exam_dates, week_limit=args.week_limit,
            scope_semester=_in_progress_sem,
        )
        sheets["reading_log"] = _truncate_by_week(
            sheets.get("reading_log"), id_col_candidates=("anon_id", "帳號"),
            time_col_candidates=("開始",),
            per_student_exam_dates=per_student_exam_dates, week_limit=args.week_limit,
            scope_semester=_in_progress_sem,
        )
        print(f"  ℹ️  behavior_semesters={behavior_semesters} 中「{_in_progress_sem}」"
              f"被判定為尚未結算成績（進行中學期），quiz／reading_log 已套用"
              f"Week {args.week_limit}+ 截斷（--data 未提供或該學期查無期末成績時的"
              f"保守預設；若此學期實際已結束，請提供正確的 --data 成績檔重跑，"
              f"避免 Week {args.week_limit}+ 的真實活動被截斷為0）")
    elif args.week_limit:
        print(f"  ℹ️  behavior_semesters={behavior_semesters} 中沒有任何學期被判定"
              f"為「尚未結算成績」，略過週次截斷（避免誤砍已結束學期的Week 13+活動，"
              f"見 CHANGELOG_v7.md v7.8）")

    # 見規劃書1.2節／討論4：基礎醫學臨床應用(一)合併課微免部分週次窗口過濾
    # （week13起始日～week18考試日），9個sheet統一嘗試套用同一窗口。
    #
    # 與上面 _truncate_by_week() 的「進行中學期模擬截斷」是獨立的兩件事，
    # 範圍互不衝突、可能同時作用於同一列（例如1151剛好也是進行中學期）：
    # _truncate_by_week() 只影響 _in_progress_sem 整學期的列（早期預警
    # 模擬用途），本窗口只影響 _course_track=="transition_merged_micro"
    # 的列（課程結構隔離用途），兩者疊加是預期行為，非衝突。
    #
    # 實作階段附註：quiz的「測驗時間」／reading_log的「開始」欄已由既有
    # 程式碼（09_lsa_event_log.py／05_time_analysis.py／上方
    # _truncate_by_week() 呼叫端）驗證存在；其餘7個sheet的實際時間欄名
    # 尚未能以完整欄位的正式產出檔驗證（手上可核對的參考檔為精簡欄位
    # 版本，缺時間相關欄，見 _TRANSITION_WINDOW_TIME_COL_CANDIDATES 定義
    # 處說明）。_window_by_week() 找不到候選欄名時會安全略過該sheet並
    # 印出訊息，不會誤刪、也不會中斷主流程；115(1) week13+真實資料到齊
    # 後，建議依規劃書4.3節回歸測試準則，逐一核對這7個sheet是否真的被
    # 正確截斷（或確認其本來就沒有可截斷的時間粒度，例如 reading_stats
    # 為per-類別加總、非逐筆事件，不適用本窗口）。
    for _sem in behavior_semesters:
        _window_cfg = (exam_dates_all.get(_sem) or {}).get("transition_micro_window")
        if not _window_cfg:
            continue
        _w_start = _window_cfg.get("week13_start")
        _w_end = _window_cfg.get("week18_exam")
        if not (_w_start and _w_end):
            continue
        for _sheet_key in list(sheets.keys()):
            sheets[_sheet_key] = _window_by_week(
                sheets.get(_sheet_key),
                time_col_candidates=_TRANSITION_WINDOW_TIME_COL_CANDIDATES,
                window_start=_w_start, window_end=_w_end,
            )
        print(f"  ℹ️  [窗口過濾] 學期{_sem}「基礎醫學臨床應用(一)」合併課微免部分："
              f"9個sheet已嘗試套用 {_w_start}~{_w_end} 窗口過濾")

    # ── Step 4 & 5：特徵工程 + 時間分析 ──
    print("\n[Step 4+5] 特徵工程 + 時間分析")
    students = []

    activity_df      = _safe_get_df(sheets, "activity")
    reading_stats_df = _safe_get_df(sheets, "reading_stats")
    quiz_df          = _safe_get_df(sheets, "quiz")
    reading_log_df   = _safe_get_df(sheets, "reading_log")

    # v7.14新增（交接文件v10第1.4a節）：閱讀誠信度——用「全體學生」的
    # 教材閱讀紀錄，依教材分別算出離群值上界，供下方主迴圈每位學生
    # 呼叫 compute_reading_integrity_features() 時判斷是否為異常長時
    # （如掛自動播放）。必須用全體資料算，不能用單一學生的資料算
    # （單一學生的「離群值」定義不出來）。
    _material_bounds = m05.compute_material_outlier_bounds(reading_log_df)
    _POLICY_READING_HOURS = 24.0  # 課程規定累積閱讀時數門檻，見config或課綱調整

    # PERF-FIX: 預先按 (anon_id, semester) 複合鍵分組，消除主迴圈 O(N²) 全表掃描
    # 原模式：每位學生呼叫 df[df["anon_id"]==anon_id] → 2700人×300K行 = 8億次操作
    # v6修正（Phase 3，決議10相關）：_group_by_anon() 純anon_id分組 →
    # _group_by_anon_semester()。Bug B根因：純anon_id分組時，同一學生
    # 跨學期的行為資料混在同一個子集裡，即使主迴圈只跑一次也會把
    # 兩學期特徵混算成一組（看似「資料完整」，實為跨學期混血）。
    _activity_by_anon      = _group_by_anon_semester(activity_df)
    _reading_stats_by_anon = _group_by_anon_semester(reading_stats_df)
    _quiz_by_anon          = _group_by_anon_semester(quiz_df)
    _reading_log_by_anon   = _group_by_anon_semester(reading_log_df)

    # v6修正（Phase 3）：主迴圈改為對每個 (anon_id, semester) pair 各產生一筆
    # student record，取代原本「對每個 anon_id 產生一筆、多學期資料混血」的作法。
    # _anon_semester_pairs 已於 Step 2 建立（見該處註解），此處直接沿用同一份。
    for anon_id, student_semester in sorted(_anon_semester_pairs):
        masked_id = lookup_table.get(anon_id, "")
        # 每位學生在該學期使用自己實際參加的考試日期
        student_exam_dates = per_student_exam_dates.get((anon_id, student_semester), {})

        # 取各表「該學生＋該學期」的子集（O(1) dict lookup，取代 O(N) 全表掃描）
        s_activity      = _activity_by_anon.get((anon_id, student_semester),      _EMPTY_DF)
        s_reading_stats = _reading_stats_by_anon.get((anon_id, student_semester), _EMPTY_DF)
        s_quiz          = _quiz_by_anon.get((anon_id, student_semester),          _EMPTY_DF)
        s_reading_log   = _reading_log_by_anon.get((anon_id, student_semester),   _EMPTY_DF)

        feats = m04.compute_student_features(
            anon_id=anon_id,
            masked_id=masked_id,
            activity_df=s_activity,
            reading_stats_df=s_reading_stats,
            quiz_df=s_quiz,
            material_counts=material_counts,
            exam_dates=student_exam_dates,
        )
        time_profile = m05.compute_student_time_profile(
            anon_id=anon_id,
            reading_log_df=s_reading_log,
            exam_dates=student_exam_dates,
            semester_start=student_exam_dates.get("semester_start", ""),
            total_weeks=int(student_exam_dates.get("total_weeks", 18)),
            material_bounds=_material_bounds,
            policy_hours=_POLICY_READING_HOURS,
        )
        feats["features"]["consistency_score"] = time_profile.get("consistency_score")

        # v7.14新增（交接文件v10第1.4c節）：閱讀誠信度——用「清理後」
        # 時數覆蓋 total_learning_minutes，原始值另存為
        # raw_total_learning_minutes 供稽核／對照。
        # 決議（本輪與使用者確認）：覆蓋動作放在下方 pre_exam_intensity
        # 重新計算「之前」，讓清理後時數自然連動 pre_exam_intensity
        # （進而連動 11_cross_analysis.py 的 BAS neg_passive 分量），
        # 而非額外脫鉤處理——因為兩者本來就是用同一份 reading_log
        # 算出來的時數，理應一致，不應該一個乾淨一個髒。
        feats["features"]["raw_total_learning_minutes"] = time_profile.get("raw_total_minutes")
        feats["features"]["total_learning_minutes"]     = time_profile.get("capped_total_minutes")
        feats["features"]["reading_inflation_ratio"]     = time_profile.get("reading_inflation_ratio")
        feats["features"]["policy_gaming_flag"]          = time_profile.get("policy_gaming_flag")

        # pre_midterm_7d_minutes / pre_final_7d_minutes 由 m05 計算，
        # m04.compute_composite_features 在 m05 之前執行，故需回填並重新計算 pre_exam_intensity。
        pre_mid = time_profile.get("pre_midterm_7d_minutes") or 0.0
        pre_fin = time_profile.get("pre_final_7d_minutes")   or 0.0
        feats["features"]["pre_midterm_7d_minutes"] = pre_mid
        feats["features"]["pre_final_7d_minutes"]   = pre_fin

        total_min   = feats["features"].get("total_learning_minutes") or 0
        total_weeks = int(student_exam_dates.get("total_weeks", 18))
        # total_min=0 時 intensity 應為 0（學生無學習行為），不產生虛高值
        if total_min == 0:
            feats["features"]["pre_exam_intensity"] = 0.0
        else:
            weekly_avg = total_min / total_weeks
            feats["features"]["pre_exam_intensity"] = round(
                (pre_mid + pre_fin) / (weekly_avg * 7), 4
            )

        students.append({
            "anon_id":      anon_id,
            "masked_id":    masked_id,
            "semester":     student_semester,
            "features":     feats["features"],
            "time_profile": time_profile,
        })

    print(f"  處理學生數：{len(students)}")

    # ── ETL-1（16-1規劃書）：學習穩定度 CV（07c_stability.py）──
    # 掛載點選在此處（而非規格書原建議的「Step 5b/segment_stats之後」）：
    # stability_cv 只依賴 student["time_profile"]["weekly_minutes"]，
    # 與成績合併（Step 6）、分群（Step 5b）皆無資料相依性，越早算越單純，
    # 也不受「segment_stats 已改至 Step 6 之後執行」這類後續調整牽動。
    _stability_cv = m07c.compute_stability_cv(students)
    _cv_non_null = 0
    for s in students:
        cv = _stability_cv.get((s["anon_id"], s["semester"]), {}).get("cv")
        s["features"]["stability_cv"] = cv
        if cv is not None:
            _cv_non_null += 1
    print(f"  ✅ [ETL-1] stability_cv 計算完成：{_cv_non_null}/{len(students)} 位學生有效"
          f"（其餘為有效週數<2，無法計算CV）")

    # ── Step 5b：分群（提前，讓 quiz 建立時 cluster 欄位已存在）──
    # FIX-ORDER: assign_clusters 原在 Step 6 之後，導致 quiz_behavior /
    # segment_pearson 的 cluster 維度全為空。提前至此。
    print("\n[Step 5b] 學習行為分群")
    students = m06b.assign_clusters(students)

    # NOTE: segment_stats（Step 5c'）刻意移至 Step 6 之後執行
    # BUG-SCORE-ORDER: Step 5c 時 final_score 尚未回填，pass/fail 分組全為 None，
    # 導致 time_distribution.json 的 pass/fail segment 永遠空白。
    # 根治方式：在 Step 6 成績回填後、Step 6b 之前重算。
    segment_stats = {}

    # ── Step 6：合併成績 ──
    print("\n[Step 6] 合併成績 + 分群")
    grade_df         = None
    merged_df        = None
    pearson_matrix   = {}
    scatter_data     = []
    weeks_data       = []
    subgroup_pearson     = {}
    outlier_thresholds   = {}
    correlation_insights = {}
    lagged_pearson       = {}
    lsa_transition       = {}
    incomplete_semesters: list = []   # 尚未獲得學末成績的學期（預警用）
    incomplete_pearson:   dict = {}   # 未完成學期獨立相關性（midterm only）
    _complete_sems:       list = []   # behavior_semesters 扣除驗證學期後的近似訓練集學期，
                                       # 供教材離群偵測（compute_material_outlier_cats）使用；
                                       # 於 try 區塊外預先初始化，避免例外路徑下 NameError
    _excluded_material_detail: list = []   # v7.10新增：規模性變動排除明細（交接文件v2第4節），
                                            # 供 correlation_matrix.json 的 meta 輸出；
                                            # 同上，於 try 區塊外預先初始化避免 NameError

    if not args.skip_grades:
        try:
            data_json = m06.load_data_json(args.data)
            grade_df  = m06.extract_grade_table(
                data_json,
                target_semester=behavior_semesters,
                target_course_keyword=args.course,
                salt=salt,
            )

            # v6新增（決議9，Phase 3b）：僅補【目前進行中學期】官方選課但零LMS
            # 活動的學生（grade_df有名冊、students沒有對應record），歷史學期
            # 同類缺口不在本次範圍內、故意不處理（範圍已於決議9明確限定）。
            # features 留空 dict：下游 Phase 4a 之 QMI/BAS 計算讀取特徵時一律
            # 經由 .get(key, 0)/.get(key) or 0 等防禦寫法（已逐一確認，見本檔
            # 其餘 s["features"].get(...) 呼叫點），視同 total_attempts=0。
            if (
                _in_progress_sem
                and grade_df is not None and not grade_df.empty
                and "semester" in grade_df.columns
            ):
                grade_roster_target = grade_df[grade_df["semester"].astype(str) == _in_progress_sem]
                existing_pairs = {(s["anon_id"], s["semester"]) for s in students}
                missing_anons = [
                    aid for aid in grade_roster_target.get("anon_id", pd.Series(dtype=str)).dropna().unique()
                    if (aid, _in_progress_sem) not in existing_pairs
                ]
                if missing_anons:
                    print(f"  ⚠️  [Phase 3b] 發現 {len(missing_anons)} 位官方選課但零LMS活動的"
                          f"{_in_progress_sem}學生，補上零值列")
                for aid in missing_anons:
                    students.append({
                        "anon_id":      aid,
                        "masked_id":    lookup_table.get(aid, ""),
                        "semester":     _in_progress_sem,
                        "features":     {},   # 行為特徵全部缺席，下游視為 total_attempts=0
                        "time_profile": {},
                    })

            merged_df = m06.merge_behavior_grades(students, grade_df)

            # 建立 (anon_id, semester) → merged row 雙鍵查找（BUG-MERGED-INDEX 修正）
            # 重修學生在多學期模式下同一 anon_id 可能有多列 theory 記錄；
            # 以 (anon_id, semester) 對齊 behavior_students.semester，確保取到正確學期成績。
            # fallback：若雙鍵找不到，再嘗試單 anon_id（取 grade_df 中排序最後一列）。
            _merged_records = merged_df.to_dict("records")
            _merged_index_2key = {
                (str(r.get("anon_id", "")), str(r.get("semester", ""))): r
                for r in _merged_records
                if r.get("anon_id")
            }
            _merged_index_1key = {}
            for r in _merged_records:
                aid = str(r.get("anon_id", ""))
                if aid:
                    _merged_index_1key[aid] = r   # 最後一列覆蓋，作為 fallback

            _fallback_1key_count = 0
            for s in students:
                aid = str(s.get("anon_id", ""))
                sem = str(s.get("semester", ""))
                row_2key = _merged_index_2key.get((aid, sem))
                row = row_2key if row_2key is not None else _merged_index_1key.get(aid)
                if row_2key is None and row is not None:
                    _fallback_1key_count += 1
                if row:
                    s["midterm_score"]  = row.get("midterm_score")
                    s["final_score"]    = row.get("final_score")
                    s["semester_score"] = row.get("semester_score")
                    # edu_type 來自 grade_df.type；practicum 已於 extract_grade_table 排除，
                    # 此處值恆為 'theory' 或學制代碼，不會出現 'practicum'
                    s["edu_type"] = str(row.get("type") or "")
                    # course_track（見規劃書1.1節／討論1）：比照 edu_type 同一
                    # 模式，從 merged_df 該列的 sheet_name+semester 現算，
                    # 供 time_distribution.json 的 course_track 欄使用（
                    # tab-behavior-time.js 新增 _filterCourseTrack 篩選維度
                    # 依此欄判斷）。複用 m06 已載入的 _etl 參照，不在
                    # lms_etl.py 新增額外的頂層 import 依賴。
                    s["course_track"] = m06._etl._classify_course_track(
                        row.get("sheet_name", ""), sem
                    )

            # v6新增（Phase 5 相容性驗證）：雙鍵查找 fallback 至單鍵的次數應
            # 趨近於0——merge_behavior_grades 已改複合鍵join（見該函式
            # docstring），理論上每個 (anon_id,semester) 都能在 merged_df
            # 中找到唯一對應列。非零值代表仍有學生的 (anon_id,semester)
            # 組合在 grade_df 中找不到直接對應（例如該學期完全零成績記錄），
            # 值愈接近0愈健康。
            if _fallback_1key_count > 0:
                print(f"  ⚠️  [Phase 5] {_fallback_1key_count} 名學生成績查找 fallback 至單鍵"
                      f"（雙鍵查找失敗），請確認是否為預期情況（如零成績新生）")
            else:
                print("  ✅ [Phase 5] 成績雙鍵查找 100% 命中，無需 fallback 至單鍵")

            # BUG-FIX（ORDER-EDU-TYPE）：edu_type 欄必須在 grade_complete_df = merged_df.copy()
            # 之前寫入 merged_df，否則 grade_complete_df 是舊版拷貝，永遠缺少 edu_type，
            # 造成 compute_subgroup_pearson() 誤判欄位不存在而跳過分層相關性計算。
            merged_df["edu_type"] = merged_df["type"].fillna("").astype(str)

            # ── 未完成學期識別 ────────────────────────────────────────────────────
            # BUG-FIX (2026-07-12，v7.8穿透審查發現)：原本在此處重新用 merged_df
            # （behavior_df 與 grade_df 的 LEFT JOIN 結果，以 behavior_df 為主表）
            # 逐學期檢查 final_score 是否全為 null，這是與 _detect_incomplete_semesters()
            # （Step 3 已計算，見下方 _incomplete_semesters）語意相同、但完全獨立實作
            # 的「第三份」判斷邏輯——與 v7.6/v7.8 修復的 _in_progress_sem 是同一類
            # 「重複判斷未同步」風險：merged_df 只涵蓋「同時有LMS活動又有成績」的
            # 學生（LEFT JOIN 以 behavior_df 為主表；零LMS活動學生僅在 Phase 3b
            # 補零值列時才會補入 _in_progress_sem，settled學期不會補），若某學期
            # 剛好所有「有LMS活動」的學生成績都還沒登錄，但「零活動」的學生已經有
            # 成績，這裡會誤判為未完成，與 _detect_incomplete_semesters() 讀取
            # data.json 完整名冊的判斷結果不一致。
            # 改為直接沿用 Step 3 已計算好的 _incomplete_semesters（同一份 data.json、
            # 同一份判斷依據），確保全流程（週次截斷、Phase 3b零值列補齊、全體
            # 相關性排除、材料離群偵測的 _current_sem、Step 9 提前預警目標學期
            # 選擇）對「這個學期是否已結算成績」有唯一、一致的答案。
            incomplete_semesters: list = sorted(
                s for s in _incomplete_semesters if s in behavior_semesters
            )
            grade_complete_df = merged_df.copy()
            if incomplete_semesters and "semester" in merged_df.columns:
                sem_col = merged_df["semester"].astype(str)
                print(f"  ℹ️  [全體相關性] 排除尚未獲得學末成績的學期：{incomplete_semesters}")
                grade_complete_df = merged_df[~sem_col.isin(incomplete_semesters)].copy()

            # ── ETL-2（16-1規劃書）：資料密度過濾（07d_density_filter.py）──
            # corr_df 僅供「相關性計算」使用（pearson_matrix / subgroup_pearson /
            # outlier_thresholds），排除總學習時數過低（< mean-1σ）的無效樣本，
            # 避免幾乎零學習行為的學生稀釋/扭曲相關係數。
            # 明確不套用於：incomplete_pearson（當期未完課學期，資料本應完整
            # 保留，理由同 FIX-DEFECT-B）、scatter_data／weeks_data／
            # lagged_pearson／segment_pearson（Step 6b）／07b at-risk——
            # 這些沿用完整 grade_complete_df／merged_df，維持原有樣本範圍。
            corr_df = m07d.filter_for_correlation(grade_complete_df)

            # ── 新增教材類型特徵排除（Max-Multiple + MIN_DELTA 規模性變動偵測）──
            # 使用 m06.compute_material_outlier_cats()：
            # 當期增量 > max(歷史增量)×2 且 >= 10 件 → 排除對應特徵欄位於全體計算。
            # _current_sem 必須是「實際進入 grade_complete_df」的最新學期，
            # 否則 outlier 偵測的學期與 pearson_matrix 計算範圍不一致（BUG-FIX）
            _complete_sems = [s for s in behavior_semesters if s not in incomplete_semesters]
            _current_sem = str(_complete_sems[-1]) if _complete_sems else ""
            # v7.10新增（交接文件v2第4節）：return_detail=True 額外取得排除明細
            # （類別、涉及欄位、當期學期、當期/歷史最大增量），寫入
            # correlation_matrix.json 的 meta.excluded_material_detail，讓前端
            # 未來可以動態產生說明文字，不必再手動改程式碼同步（現況是
            # help-modal.js 把「114-2 新增 +35 件」這類文字寫死，ETL重跑換了
            # 學期或教材類別後不會自動更新）。
            _excluded_feats: list = []
            if _current_sem and len(behavior_semesters) >= 2:
                _excluded_feats, _excluded_material_detail = m06.compute_material_outlier_cats(
                    students, _current_sem, behavior_semesters, return_detail=True,
                )

            pearson_matrix = m06.compute_pearson_matrix(
                corr_df,
                excluded_features=_excluded_feats or None,
            )

            # 未完成學期獨立相關性（僅 midterm_score 可用）
            # FIX-DEFECT-B: incomplete_pearson 不繼承 _excluded_feats。
            # 未完成學期是當期學期，其 QUZ 行為品質指標資料完整有效，
            # 排除邏輯只適用於「跨學期合併」的全體相關性計算，
            # 不適用於單一學期的獨立計算。同理也不套用 ETL-2 密度過濾，
            # 沿用完整 merged_df（而非 corr_df）。
            incomplete_pearson: dict = {}
            for _sem in incomplete_semesters:
                _sem_df = merged_df[merged_df["semester"].astype(str) == _sem].copy()
                if len(_sem_df) >= 5:
                    incomplete_pearson[_sem] = m06.compute_pearson_matrix(
                        _sem_df,
                        target_cols=["midterm_score"],
                    )
                    print(f"  ℹ️  [未完成學期 {_sem}] 獨立相關性（midterm_score only）計算完成")

            # edu_type 欄已於上方（grade_complete_df 複製前）寫入，此處不再重複賦值。
            # SCALE-CHANGE-CONSISTENCY: subgroup/lagged 與 pearson_matrix 同源於
            # 跨學期合併資料，套用相同的 _excluded_feats，避免同一欄位在不同
            # 區塊呈現不一致（一邊標 scale_change、一邊顯示未標註數值）。
            subgroup_pearson = m06.compute_subgroup_pearson(
                corr_df,
                excluded_features=_excluded_feats or None,
            )

            outlier_thresholds = m06.compute_outlier_thresholds(corr_df)

            # 傳入未截斷的題庫紀錄，確保 114-2 各週作答強度圖表能完整呈現
            quiz_for_behavior = untruncated_quiz_df if 'untruncated_quiz_df' in locals() and untruncated_quiz_df is not None else quiz_df
            # v7.11新增：依 exam_dates_all 各學期的 self_study_weeks 設定
            # （由 interactive_fill_missing_calendar()::_prompt_week_structure()
            # 互動確認後寫入，見112-1/112-2/113-1「16+2週」特例）組裝
            # {semester: {"1": week_num, "2": week_num, ...}}，供
            # build_quiz_behavior_by_week() 逐學期查表「自學週N」對應
            # 週次；未設定的學期自動跳過，不影響既有行為。
            _self_study_week_map = {
                str(sem): {str(i + 1): wk for i, wk in enumerate(cfg["self_study_weeks"])}
                for sem, cfg in exam_dates_all.items()
                if cfg.get("self_study_weeks")
            }
            # v7.14.7新增（ETL交接：06_merge_grades.py 題庫週次解析殘留缺口）：
            # 從 per_student_exam_dates（(anon_id, semester) 複合鍵，已內建
            # 各學期/各section正確的semester_start，見 build_per_student_exam_dates()）
            # 篩出 semester_start，供 build_quiz_behavior_by_week() 第三階段
            # 對「醫用寄生蟲學 練習題庫」等不含週次資訊的標題，以測驗時間戳
            # 反推週次（見 _resolve_date_based_week() docstring）。查無
            # semester_start 的鍵值不納入，該列維持原行為（NaN，不列入統計）。
            _quiz_date_fallback_map = {
                key: cfg["semester_start"]
                for key, cfg in per_student_exam_dates.items()
                if cfg.get("semester_start")
            }
            # 見規劃書1.3節討論3（小幅修正版）：本批次資料中校歷為「16+2週」
            # 特例的學期集合，供 build_quiz_behavior_by_week() 補充頂層
            # pooled視圖的exam_weeks標記（週8/16），比照
            # tab-behavior-time.js::SIXTEEN_PLUS_TWO_SEMS 同一組學期判斷，
            # 從 exam_dates_all 的 course_structure 欄讀取（該欄先前只寫入
            # 未消費，見規劃書討論3發現，此為第一個實際讀取它的消費點）。
            _sixteen_plus_two_semesters = {
                str(sem) for sem, cfg in exam_dates_all.items()
                if cfg.get("course_structure") == "16+2"
            }
            weeks_data = m06.build_quiz_behavior_by_week(
                quiz_for_behavior, grade_df, behavior_students=students,
                self_study_week_map=_self_study_week_map or None,
                date_fallback_start_map=_quiz_date_fallback_map or None,
                sixteen_plus_two_semesters=_sixteen_plus_two_semesters or None,
            )

            correlation_insights = m06.compute_correlation_insights(
                pearson_matrix, weeks_data
            )

            # Phase C：時間滯後相關性（沿用主學期 semester_start，
            # 舊版 exam_dates 為單一扁平dict時語意相同；v6後改查
            # exam_dates_all 中「主學期」的子項目）
            semester_start = exam_dates_all.get(semester, {}).get("semester_start", "")
            lagged_pearson = m06.compute_lagged_pearson(
                merged_df, quiz_df,
                semester_start=semester_start,
                excluded_features=_excluded_feats or None,
            )

        except FileNotFoundError as e:
            print(f"  ⚠️  {e}\n  成績合併略過（可用 --skip-grades 抑制此警告）")

    # assign_clusters 已於 Step 5b 執行，此處只需建立 radar 資料
    radar_data = m06b.build_radar_chart_data(students)

    if merged_df is not None:
        # v4 修復：改用 (anon_id, semester) 複合鍵，避免重修生跨學期互相覆蓋
        # cluster 值（同一根因模式：舊版純 anon_id/masked_id 單鍵，字典推導式
        # 迭代到後面學期的 record 時，會把該 anon_id 的 cluster 覆蓋成「最後
        # 一筆」，導致較舊學期的 scatter 資料點被貼上錯誤的分群標籤）。
        cluster_map = {
            (s["anon_id"], str(s.get("semester", ""))): s.get("cluster", "")
            for s in students
        }
        scatter_data = m06.build_scatter_data(merged_df, cluster_map=cluster_map)

    # ── Step 5c'：時間分析分段預聚合（Step 6 成績回填後執行，pass/fail 分組有效）──
    # BUG-SCORE-ORDER 修正：必須在 final_score / semester_score 回填後執行，
    # 否則 pass/fail segment 永遠為空，time_distribution.json 輸出不完整。
    if args.skip_grades:
        print("  ⚠️  [Step 5c'] --skip-grades 模式：final_score 未回填，"
              "segment_stats 的 pass/fail 分組將為空（time tab pass/fail 視圖無資料）")
    print("\n[Step 5c'] 時間分析 segment 預聚合（含成績）")
    segment_stats = m05.compute_class_segment_stats(students)

    # ── Step 6b：相關性分段預聚合 ──
    segment_pearson = {}
    if merged_df is not None:
        # merged_df 的 "cluster" 欄已由 merge_behavior_grades()（依 (anon_id,
        # semester) 複合鍵）正確賦值，此處不再整批覆蓋。
        # v4 修復：舊版用純 anon_id 建 _cluster_map 整批覆蓋 merged_df["cluster"]，
        # 對重修生會把較舊學期的正確值覆蓋成「最後一筆學期」的分群結果
        # （與 scatter_data 的 cluster_map 同一根因，Bug B 在此處原地重演）。
        # 保留 _cluster_map（複合鍵）僅用於補上極少數 Step 6 名冊補零值學生
        # （Phase 3b，未經 assign_clusters 分群、cluster 欄本身缺漏）的情況，
        # 不覆蓋既有正確值。
        _cluster_map = {
            (str(st["anon_id"]), str(st.get("semester", ""))): str(st.get("cluster", ""))
            for st in students
        }
        _existing_cluster = (
            merged_df["cluster"].astype(str)
            if "cluster" in merged_df.columns
            else pd.Series("", index=merged_df.index)
        )
        _row_keys = list(zip(merged_df["anon_id"].astype(str), merged_df["semester"].astype(str)))
        _fallback_cluster = pd.Series(
            [_cluster_map.get(k, "") for k in _row_keys], index=merged_df.index
        )
        merged_df = merged_df.assign(
            cluster=_existing_cluster.where(_existing_cluster.str.len() > 0, _fallback_cluster)
        )
        print("\n[Step 6b] 相關性 segment 預聚合")
        segment_pearson = m06.compute_segment_pearson(
            merged_df,
            feature_cols=m06.FEATURE_COLS,
            target_cols=m06.TARGET_COLS,
        )

        # FIX-DEFECT-C: correlation_insights 改用 segment_pearson["all|all|all"]
        # 的完整矩陣（無 excluded_features 排除）作為洞察來源，
        # 確保 QUZ 品質指標被正確納入 highest_r / score_delta / cramming 計算。
        _full_pearson = segment_pearson.get("all|all|all", {}).get("pearson")
        if _full_pearson and correlation_insights is not None:
            correlation_insights = m06.compute_correlation_insights(
                pearson_matrix, weeks_data,
                full_pearson_matrix=_full_pearson,
            )
            print("  ℹ️  [Step 6b] correlation_insights 已用完整矩陣重算")

    # ── Step 8：LSA 行為序列轉移矩陣 ──
    print("\n[Step 8] LSA 行為序列轉移矩陣")
    try:
        event_log = m09.build_event_log(sheets, anon_map=lookup_table)

        # join cluster（09 不輸出此欄，從 students 補入）
        # v4 修復：獨立於 Step 6b 的 _cluster_map 重新建立複合鍵版本，改用
        # event_log 自己逐列的 semester 欄查表，而非單鍵 anon_id。舊版單鍵
        # 寫法對重修生會把較舊學期事件的 cluster 誤標成「最後一筆學期」的
        # 分群結果，進而污染 lsa_by_cluster 的分組統計（與 scatter_data /
        # segment_pearson 同一根因模式，Bug B 在此處原地重演）。
        _cluster_map_by_sem = {
            (str(s["anon_id"]), str(s.get("semester", ""))): str(s.get("cluster", ""))
            for s in students
        }
        if "semester" in event_log.columns:
            _event_keys = list(zip(
                event_log["anon_id"].astype(str), event_log["semester"].astype(str)
            ))
            _cluster_vals = pd.Series(
                [_cluster_map_by_sem.get(k, "") for k in _event_keys], index=event_log.index
            )
        else:
            print("  ⚠️  [Step 8] event_log 缺少 semester 欄，cluster 查表退回純 "
                  "anon_id（重修生可能不精確）")
            _cluster_map_legacy = {str(s["anon_id"]): str(s.get("cluster", "")) for s in students}
            _cluster_vals = event_log["anon_id"].astype(str).map(_cluster_map_legacy).fillna("")
        event_log = event_log.assign(cluster=_cluster_vals)
        #
        # BUG-LSA-SEMESTER 修正（歷史背景，semester 欄已由 m09.build_event_log
        # 逐列正確帶出，非本次改動範圍）：
        # 原本 semester 欄在此處用 _sem_map（來自 activity_df 的
        # anon_id → 單一學期 反查表）整批覆蓋，但 activity_df 對「當前
        # 進行中學期」常缺乏完整覆蓋（尚未有完整活動匯出），導致該學期
        # 大多數學生的 anon_id 找不到對應列，只能 fallback 到全域主學期
        # （依檔案數最多者選出，通常是較舊的大學期），而非其真正就讀的
        # 學期；只有「同時橫跨多學期、activity_df 有多筆紀錄」的重修生
        # 才會被 _semester_by_anon 正確標成最新學期。結果就是 by_semester
        # 篩選當前學期時，抓到的幾乎全是重修生，人數遠低於實際選課人數
        # （如 114(2) 只顯示 36 人）。
        #
        # 修正：semester 改用 09_lsa_event_log.build_event_log() 直接從
        # reading_log / quiz 來源列自帶的 _semester（_merge_lms_files() 依
        # 檔名寫入）傳遞出來的逐筆事件學期，不再整批用 anon_id 反查覆蓋。
        # 每筆事件用「自己實際發生的學期」歸類，重修生的各學期事件也能
        # 正確分開，不會被單一學期覆蓋。

        pass_set = set(
            str(s["anon_id"])
            for s in students
            if s.get("final_score") is not None
            if s.get("final_score") >= m11.PASS_THRESHOLD
        )
        if not pass_set:
            print("  ⚠️  [Step 8] pass_set 為空（成績未合併？），LSA pass/fail 分組將無意義（全數歸入 fail）")
        lsa_groups       = m10.compute_lsa_all_groups(event_log, pass_set)
        lsa_by_semester  = m10.compute_lsa_by_semester(event_log, pass_set)
        lsa_by_cluster   = m10.compute_lsa_by_cluster(event_log, pass_set)
        # FEAT-LSA-TYPE：真正的 S 序列行為分群（與 by_cluster 並存）
        # FIX-Q2：回傳 tuple (dict, student_lsa_type)，避免 null key 混入矩陣結構
        lsa_by_lsa_type, lsa_type_map, lsa_transition_ratios = m10.compute_lsa_by_lsa_type(
            event_log, pass_set, students=students
        )

        lsa_transition = {
            "session_gap_minutes": m09.LSA_SESSION_GAP_MINUTES,
            "behaviors":           m10.BEHAVIORS,
            "groups":              lsa_groups,
            "by_semester":         lsa_by_semester,
            "by_cluster":          lsa_by_cluster,
            "by_lsa_type":         lsa_by_lsa_type,   # FEAT-LSA-TYPE，key 僅含 S1–S5
            "lsa_type_map":        lsa_type_map,       # { anon_id: "S1"|...|"S5" }
            "transition_ratios":   lsa_transition_ratios,   # { anon_id: {MM/MQ/QM/QQ_ratio} }
        }
    except Exception as _lsa_e:
        print(f"  ⚠️  LSA 計算失敗（不影響主流程）：{_lsa_e}")
        lsa_transition = {}   # 明確重置，消除對 L803 初始化的隱性依賴

    # ── Step 9：跨模組分析（R×S）+ 提前預警系統 ──
    # 依賴：scatter_data（Step 6）+ lsa_transition.lsa_type_map（Step 8）+
    #       incomplete_semesters（驗證學期清單，如 1151）
    # 失敗不影響主流程（cross_analysis_data / warning_data 留空，
    # m07.export_all 將跳過對應檔案輸出）
    cross_analysis_data: dict = {}
    warning_data: dict = {}
    try:
        print("\n[Step 9] 跨模組分析（R×S）+ 提前預警系統")
        ltm = lsa_transition.get("lsa_type_map", {})

        cross_records, cross_filter_meta = m11.build_records_from_memory(
            scatter_data, ltm, incomplete_semesters,
            lsa_transition.get("transition_ratios", {}),
        )
        if not cross_records:
            print("  ⚠️  [Step 9] 訓練集（theory，扣除驗證學期）為空，跳過跨模組分析")
        else:
            cross_analysis_data = m11.compute_cross_analysis(
                cross_records, cross_filter_meta,
                enable_xgb=args.enable_xgb, week_limit=args.week_limit,
            )
            o = cross_analysis_data["overall"]

            # FEAT-META-SEMRANGE：訓練集學期範圍（如「111(1)-114(1)」）顯示字串。
            # 原始學期清單改由 11_cross_analysis.py::compute_cross_analysis() 直接從
            # with_final records 算出（權威來源，與 n_with_final 同源），
            # 此處僅呼叫既有的 format_semester_range() 做顯示格式化，
            # 取代先前於 tab-behavior-cross.js 中寫死的 "111-1–114-1" 字串。
            _training_sems = cross_analysis_data["meta"].get("training_semesters", [])
            cross_analysis_data["meta"]["training_semester_range"] = format_semester_range(_training_sems)

            print(f"  ✅ cross_analysis：n={o['n']}，"
                  f"期末不及格率={o['fail_rate_final']:.1%}，"
                  f"排除學期={cross_filter_meta['incomplete_semesters']}，"
                  f"訓練集範圍={cross_analysis_data['meta']['training_semester_range']}")
            print(f"  ℹ️  S群未分類（序列事件不足，非S5）：n="
                  f"{cross_analysis_data['meta'].get('n_s_cluster_unclassified', 0)}")
            print(f"  ℹ️  R群未分類（無LMS行為資料、未經assign_clusters分群，非R3）：n="
                  f"{cross_analysis_data['meta'].get('n_r_cluster_unclassified', 0)}")

            # ── 模型持久化（第7.4節）：xgb_validation 訓練成功時存檔 ──
            # 獨立 try/except，避免存檔失敗（如磁碟權限）連帶中斷提前預警產出
            xgb_val = cross_analysis_data.get("xgb_validation")
            if xgb_val and xgb_val.get("_model_artifacts"):
                try:
                    import joblib
                    _xgb_model, _xgb_imputer = xgb_val["_model_artifacts"]
                    _repo_root = os.path.dirname(os.path.dirname(_BASE))
                    _models_dir = os.path.join(_repo_root, "models")
                    os.makedirs(_models_dir, exist_ok=True)
                    _model_path = os.path.join(_models_dir, "xgb_week12.model")
                    _imputer_path = os.path.join(_models_dir, "xgb_week12_imputer.joblib")
                    _xgb_model.save_model(_model_path)
                    joblib.dump(_xgb_imputer, _imputer_path)
                    print(f"  ✅ 模型已存檔：{_model_path}")
                    print(f"  ✅ Imputer已存檔：{_imputer_path}")
                except Exception as _model_save_e:
                    print(f"  ⚠️  模型存檔失敗（不影響主流程）：{_model_save_e}")

            # ── 提前預警：目標學期 ──
            warning_target = args.warning_target.strip()
            if not warning_target:
                if incomplete_semesters:
                    warning_target = sorted(incomplete_semesters)[-1]
                else:
                    print("  ℹ️  [Step 9] incomplete_semesters 為空且未指定 --warning-target，"
                          "不產生 warning_*.json")

            if warning_target:
                target_records = m12.build_target_records_from_memory(
                    scatter_data, ltm, warning_target,
                    lsa_transition.get("transition_ratios", {}),
                )
                if not target_records:
                    print(f"  ⚠️  [Step 9] 目標學期 {warning_target} 無 theory 學生記錄，"
                          f"跳過提前預警")
                else:
                    warning_data = m12.compute_warning(
                        target_records, cross_analysis_data, warning_target
                    )
                    s = warning_data["summary"]
                    print(f"  ✅ warning_{warning_target}："
                          f"total={warning_data['meta']['total_students']}，"
                          f"HIGH={s['HIGH']['count']}，"
                          f"MEDIUM={s['MEDIUM']['count']}，"
                          f"LOW={s['LOW']['count']}")

            # 移除內部欄位（z-score 標準化用，不對外輸出）
            cross_analysis_data = {
                k: v for k, v in cross_analysis_data.items() if not k.startswith("_")
            }
    except Exception as _cross_e:
        print(f"  ⚠️  Step 9 跨模組分析/預警失敗（不影響主流程）：{_cross_e}")
        traceback.print_exc()
        cross_analysis_data = {}
        warning_data = {}

    # ── Step 7（輸出）：輸出 JSON ──
    if args.dry_run:
        print("\n[DRY RUN] 模擬輸出完成，未實際寫入檔案")
        print(f"  學生數：{len(students)}")
        print(f"  Pearson 矩陣目標變數：{list(pearson_matrix.keys())}")
        print(f"  分層學制：{list(subgroup_pearson.keys())}")
        print(f"  correlation_insights：{list(correlation_insights.keys())}")
        print(f"  lagged_pearson 特徵數：{len(lagged_pearson.get('results', {}))}")
        print(f"  segment_stats 組數：{len(segment_stats)}")
        print(f"  segment_pearson 組數：{len(segment_pearson)}")
        print(f"  lsa_transition 群組：{list(lsa_transition.get('groups', {}).keys())}")
        print(f"  lsa by_semester 學期數：{len(lsa_transition.get('by_semester', {}))}")
        print(f"  lsa by_cluster 分群數：{len(lsa_transition.get('by_cluster', {}))}")
        print(f"  lsa by_lsa_type S群數：{len(lsa_transition.get('by_lsa_type', {}))}")
        if cross_analysis_data:
            print(f"  cross_analysis n_with_final：{cross_analysis_data['meta']['n_with_final']}")
            print(f"  cross_analysis 排除學期：{cross_analysis_data['meta']['incomplete_semesters_excluded']}")
            print(f"  cross_analysis S群未分類：{cross_analysis_data['meta'].get('n_s_cluster_unclassified', 0)}")
        if warning_data:
            print(f"  warning 目標學期：{warning_data['meta']['target_semester']}")
            print(f"  warning 人數：{warning_data['meta']['total_students']}")
        return

    print("\n[Step 7] 輸出 JSON")
    m07.export_all(
        students=students,
        radar_data=radar_data,
        pearson_matrix=pearson_matrix,
        scatter_data=scatter_data,
        weeks_data=weeks_data,
        feature_cols=m06.FEATURE_COLS,
        target_cols=m06.TARGET_COLS,
        course_meta=course_meta,
        material_counts=material_counts,
        exam_dates=exam_dates_all.get(semester, {}),  # 主學期彙整值（顯示用）
        output_dir=args.out,
        subgroup_pearson=subgroup_pearson,
        correlation_insights=correlation_insights,
        outlier_thresholds=outlier_thresholds,
        lagged_pearson=lagged_pearson,           # Phase C
        segment_stats=segment_stats,             # Phase D
        segment_pearson=segment_pearson,         # Phase D
        lsa_transition=lsa_transition,           # Phase E
        incomplete_semesters=incomplete_semesters,   # 未完成學期清單
        incomplete_pearson=incomplete_pearson,       # 未完成學期獨立相關性
        cross_analysis_data=cross_analysis_data,     # Step 9：R×S跨模組分析
        warning_data=warning_data,                   # Step 9：提前預警系統
        excluded_material_detail=_excluded_material_detail,  # v7.10：規模性變動排除明細
    )

    # ── 問題A修正 v7.15.6：補寫 data.json meta.course_id / course_name ──
    # 與 behavior.json 的 course_meta 填入時機對齊（export_all 之後、Phase 4 之前）。
    # --skip-grades / --dry-run 皆已在此處之前 return，不受影響。
    if args.data and os.path.exists(args.data):
        try:
            with open(args.data, "r", encoding="utf-8") as _f:
                _data_json = json.load(_f)
            _meta = _data_json.setdefault("meta", {})
            _meta["course_id"]   = course_meta.get("course_id", "")
            _meta["course_name"] = course_meta.get("course_name", "")
            with open(args.data, "w", encoding="utf-8") as _f:
                json.dump(_data_json, _f, ensure_ascii=False, indent=2)
            print(f"  [A-FIX] data.json meta patched: {_meta['course_id']} / {_meta['course_name']}")
        except Exception as _e:
            print(f"  ⚠️  data.json meta patch failed (non-fatal): {_e}")
    else:
        print("  ⚠️  data.json not found, skipping meta patch (run etl.py first or provide --data)")

    # Phase 4：落後特徵輪廓報告（schema 3.0 由 m07b 統一輸出）
    # try/except 包覆：Phase 4 失敗不影響主 ETL 流程
    if grade_df is not None:
        try:
            _m07b = _load_module(
                os.path.join(_BASE, "07b_at_risk_profile.py"), "m07b"
            )
            _m07b.export_at_risk_profile_v2(
                students=students,
                course_meta=course_meta,
                exam_dates=exam_dates_all.get(semester, {}),  # 主學期考試日期（fallback 用）
                per_student_exam_dates=per_student_exam_dates,  # 逐人考試日期（(anon_id,semester)複合鍵）
                output_dir=args.out,
            )
        except Exception as _e:
            print(f"  ⚠️  Phase 4 報告產生失敗（不影響主流程）：{_e}")
            traceback.print_exc()

    # Phase 5：基礎醫學臨床應用(一)合併課微免部分8小時教材閱讀門檻清單
    # （見規劃書1.5節，實作階段依真實資料確認欄位後定案）。
    # try/except 包覆：非核心流程，失敗不影響主 ETL 流程。只有偵測到
    # course_track=="transition_merged_micro"學生的學期才會產出檔案，
    # 絕大多數學期（115(1)以前）不受影響、不產生任何檔案。
    try:
        m07e = _load_module(
            os.path.join(_BASE, "07e_transition_micro_reading.py"), "m07e"
        )
        _transition_semesters = sorted({
            s.get("semester") for s in students
            if s.get("course_track") == "transition_merged_micro" and s.get("semester")
        })
        for _sem in _transition_semesters:
            _grade_records = [
                s for s in students
                if s.get("semester") == _sem and s.get("course_track") == "transition_merged_micro"
            ]
            _reading_stats_df = sheets.get("reading_stats")
            # [AUDIT-NOTE 穿透式審查 0914] 這裡重用 m05._hms_to_minutes()（原本
            # 設計給 reading_log 分頁的「時間」欄位），套用在 reading_stats
            # 分頁的「閱讀時間」欄位上——同一份 reading_stats 欄位，
            # 04_feature_engineering.py::hms_to_minutes() 也會處理，但兩者
            # 對「非字串輸入」的處理方式不同（m05版：數字直接當分鐘數；
            # m04版：一律回傳0.0，見WARN-3 vs v1.2/v1.3註解），純字串輸入時
            # 兩者算法完全等價。已用全部14個真實歷史LMS檔案（111-1~114-2，
            # 324,317列）驗證「閱讀時間」欄位100%為字串、無一筆非字串值，
            # 故此差異目前不影響任何真實資料（見
            # test_hms_functions_cross_consistency_v7_15_29.py）。若未來LMS
            # 匯出格式改變導致此欄位出現數字型別，兩處計算結果將不再一致，
            # 需重新檢視。
            _minutes = m07e.compute_reading_minutes_by_student(_reading_stats_df, m05._hms_to_minutes)
            _shortfall_records = m07e.build_shortfall_records(_minutes, _grade_records)
            # [BUG-FIX 穿透式審查 0913] 見 07e_transition_micro_reading.py::
            # export_shortfall_json() 上方註解——重用 checkpoint_graded_ratio()
            # （而非另寫一份判斷式，避免重演同一種「各自維護」問題）算出這份
            # Phase 5輸出是否真的代表學期已結束，讓前端能正確判斷是否該讓
            # checkpoint版顯示，而不是只看檔案是否存在。
            _grading_complete, _graded_cnt, _ = m07e.checkpoint_graded_ratio(_grade_records)
            _shortfall_json = m07e.export_shortfall_json(
                _shortfall_records, _sem,
                grading_complete=_grading_complete, graded_count=_graded_cnt,
            )
            _out_json_path = os.path.join(args.out, f"micro_immuno_reading_shortfall_{_sem}.json")
            with open(_out_json_path, "w", encoding="utf-8") as _f:
                json.dump(_shortfall_json, _f, ensure_ascii=False, indent=2)
            print(f"  [07e] 學期{_sem}：{_shortfall_json['shortfall_count']}/"
                  f"{_shortfall_json['total_students']} 名學生8小時門檻未達標"
                  f"（成績登錄 {_graded_cnt}/{_shortfall_json['total_students']}，"
                  f"grading_complete={_grading_complete}） → {_out_json_path}")

            _csv_rows = m07e.export_shortfall_csv_rows(_shortfall_records)
            _out_csv_path = os.path.join(args.out, f"micro_immuno_reading_shortfall_{_sem}.csv")
            with open(_out_csv_path, "w", encoding="utf-8-sig", newline="") as _f:
                import csv as _csv
                _csv.writer(_f).writerows(_csv_rows)
            print(f"         已同步輸出可貼入Excel的清單 → {_out_csv_path}")
    except Exception as _e:
        print(f"  ⚠️  Phase 5（8小時教材閱讀門檻清單）產生失敗（不影響主流程）：{_e}")
        traceback.print_exc()

    # Phase 5b：8小時教材閱讀門檻「學期中進度提醒」（checkpoint版，見
    # handoff_checkpoint_feature.md 3a決策）。人工手動觸發，預設不啟用，
    # 未指定 --checkpoint-target 時完全靜默略過（不印任何訊息）。
    checkpoint_target = args.checkpoint_target.strip()
    if checkpoint_target:
        try:
            m07e = _load_module(
                os.path.join(_BASE, "07e_transition_micro_reading.py"), "m07e"
            )
            _cp_grade_records = [
                s for s in students
                if s.get("semester") == checkpoint_target
                and s.get("course_track") == "transition_merged_micro"
            ]
            if not _cp_grade_records:
                print(f"  ⚠️  [Phase 5b] --checkpoint-target={checkpoint_target} "
                      f"查無 transition_merged_micro 學生記錄，略過checkpoint清單產生")
            else:
                # 這門課全學期只考一次Test III、無期中考（見etl.py），
                # semester_score只要非null就代表Test III已出成績——超過
                # 半數學生已有分數，視為本學期已完結，checkpoint不再有
                # 意義，直接不產生檔案（使用者決策，2026-09-10）。
                # v7.15.24：判斷邏輯抽到m07e.checkpoint_graded_ratio()，
                # 跟tests/test_checkpoint_progress_v7_15_23.py共用同一個
                # 函式，避免各自維護導致跑掉（見該函式docstring說明）。
                _should_skip, _graded_count, _total_count = m07e.checkpoint_graded_ratio(
                    _cp_grade_records
                )
                if _should_skip:
                    print(f"  ℹ️  [Phase 5b] 學期{checkpoint_target}已有"
                          f"{_graded_count}/{_total_count}名學生Test III"
                          f"成績（>50%），判定已完結，略過checkpoint產生")
                else:
                    _reading_stats_df = sheets.get("reading_stats")
                    _cp_records = m07e.build_progress_checkpoint_records(
                        _reading_stats_df, m05._hms_to_minutes, _cp_grade_records
                    )
                    _cp_json = m07e.export_shortfall_json(_cp_records, checkpoint_target)
                    _cp_json["is_checkpoint"] = True  # 被動metadata，前端目前不讀取這個欄位
                    _cp_json_path = os.path.join(
                        args.out, f"micro_immuno_reading_checkpoint_{checkpoint_target}.json"
                    )
                    with open(_cp_json_path, "w", encoding="utf-8") as _f:
                        json.dump(_cp_json, _f, ensure_ascii=False, indent=2)
                    print(f"  [07e-checkpoint] 學期{checkpoint_target}進度快照："
                          f"{_cp_json['shortfall_count']}/{_cp_json['total_students']} "
                          f"名學生目前未達8小時門檻 → {_cp_json_path}")

                    _cp_csv_rows = m07e.export_shortfall_csv_rows(_cp_records)
                    _cp_csv_path = os.path.join(
                        args.out, f"micro_immuno_reading_checkpoint_{checkpoint_target}.csv"
                    )
                    with open(_cp_csv_path, "w", encoding="utf-8-sig", newline="") as _f:
                        import csv as _csv
                        _csv.writer(_f).writerows(_cp_csv_rows)
                    print(f"         已同步輸出可貼入Excel的清單 → {_cp_csv_path}")
        except Exception as _e:
            print(f"  ⚠️  Phase 5b（8小時教材閱讀門檻進度快照）產生失敗（不影響主流程）：{_e}")
            traceback.print_exc()

    print("\n✅ ETL 完成！")
    print(f"   輸出目錄：{args.out}")

# ── 工具函式 ──────────────────────────────────────────────────

def _detect_id_col(sheets: dict) -> str:
    for key in ("activity", "summary", "reading_stats"):
        df = sheets.get(key)
        if df is None:
            continue
        for col in df.columns:
            if "帳號" in col or "學號" in col:
                return col
    return "帳號"

def _detect_title_col(df) -> str:
    candidates = [
        "活動名稱", "教材名稱", "活動紀錄", "標題", "名稱", "測驗標題",
        "教材標題", "資源名稱", "教材",           # 補上 reading_stats 常見欄位名
    ]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        if "帳號" not in col and "學號" not in col:
            return col
    return ""

def _collect_student_ids(sheets: dict, id_col: str) -> list:
    sids = set()
    for key, df in sheets.items():
        if df is None or id_col not in df.columns:
            continue
        for val in df[id_col].dropna().unique():
            sid = m02.normalize_student_id(val)
            if m02.is_valid_student_id(sid):
                sids.add(sid)
    return list(sids)

def _anonymize_sheets(sheets: dict, id_col: str, salt: str) -> dict:
    result = {}
    for key, df in sheets.items():
        if df is None or id_col not in df.columns:
            result[key] = df
            continue
        result[key] = m02.anonymize_dataframe(df, id_col, salt)
    return result

def _group_by_anon_semester(df: pd.DataFrame) -> dict:
    """
    按 (anon_id, _semester) 複合鍵分組，取代 _group_by_anon() 純anon_id分組。

    Bug B根因：舊版只用anon_id分組，同一學生跨學期資料混在一起，
    重修生looked像「完整」但其實混血自兩學期。
    """
    if df is None or df.empty or "anon_id" not in df.columns:
        return {}
    if "_semester" not in df.columns:
        # 理論上不應發生（_merge_lms_files一律補_semester欄），保守fallback
        print("  ⚠️  DataFrame 缺少 _semester 欄，退回純 anon_id 分組（可能重演Bug B）")
        return {aid: sub.reset_index(drop=True) for aid, sub in df.groupby("anon_id", sort=False)}
    return {
        key: sub.reset_index(drop=True)
        for key, sub in df.groupby(["anon_id", "_semester"], sort=False)
    }


def _collect_anon_semester_pairs(sheets: dict) -> set:
    """
    取代 _build_semester_by_anon()。回傳四表聯集出現過的全部
    (anon_id, semester) pair，不做「多學期取最新」收斂——
    Phase 3 主迴圈需要每個 pair 各自產生一筆 record，收斂會
    重演 Bug B（重修生跨學期混血）。

    請用「Step 3 分類 / 週數截斷之前」的原始 sheets 呼叫本函式：學期
    判定跟教材分類、早期預警用的週數截斷無關，用截斷前的完整資料
    才不會漏抓學期。
    """
    pairs: set = set()
    for key in ("activity", "reading_stats", "quiz", "reading_log"):
        df = sheets.get(key)
        if df is None or df.empty or "anon_id" not in df.columns or "_semester" not in df.columns:
            continue
        sub = df[["anon_id", "_semester"]].dropna()
        sub = sub[sub["_semester"].astype(str) != ""]
        pairs |= set(zip(sub["anon_id"], sub["_semester"].astype(str)))
    return pairs

# 模組層級共用空 DataFrame sentinel，避免每次 run() 重複建立
_EMPTY_DF = pd.DataFrame()

def _safe_get_df(sheets: dict, key: str) -> pd.DataFrame:
    """
    安全取得 DataFrame。
    非空 DataFrame 進行 bool 判斷會拋出 ValueError: The truth value of a DataFrame is ambiguous。
    """
    df = sheets.get(key)
    if df is None or df.empty:
        return pd.DataFrame()
    return df

def _truncate_by_week(df: pd.DataFrame, id_col_candidates: tuple, time_col_candidates: tuple,
                       per_student_exam_dates: dict, week_limit: int,
                       scope_semester: str = "") -> pd.DataFrame:
    """
    依各生 (anon_id, semester) 的 exam_dates['semester_start'] 計算事件所在週次，
    過濾 df 至 week <= week_limit。
    時間欄/ID欄偵測沿用既有慣例（比照 09_lsa_event_log.py::_detect_id_col / _detect_time_col 的偵測邏輯）。
    semester_start 缺失或時間解析失敗的列保留（不誤刪）。

    BUG-FIX (2026-07-03)：原本不分學期，對整個 df 套用 week<=week_limit，
    導致「已結束的學期」（如 114-1）也被腰斬在第 week_limit 週之後，
    儀表板上該學期第 13 週起的柱狀圖全部消失。
    此參數的設計本意是「早期預警特徵截斷」──只該作用在尚未結束、
    需要模擬即時預測情境的『目前學期』，不該套用到已經結束、
    本來就該有完整週次資料的歷史學期。
    scope_semester 指定後，只有 "_semester" 欄等於該值的列會被週數過濾，
    其餘學期的列一律保留（不受 week_limit 影響）。
    scope_semester 留空時維持舊行為（全體套用），供未呼叫方明確指定範圍時的相容 fallback。

    v6修正：per_student_exam_dates 已改為 (anon_id, semester) 複合鍵
    （Phase 1），改用 df 本身逐列的 "_semester" 欄組成複合鍵查詢，
    不再只用 anon_id 單鍵查（單鍵查詢在複合鍵字典下會全數查空，
    等同於 semester_start 全數缺失，截斷功能會靜默失效）。
    """
    if df is None or df.empty:
        return df
    id_col = next((c for c in id_col_candidates if c in df.columns), None)
    time_col = next((c for c in time_col_candidates if c in df.columns), None)
    if not id_col or not time_col:
        return df  # 偵測不到欄位，原樣返回（防呆，不中斷主流程）

    ts = pd.to_datetime(df[time_col], errors="coerce")
    aid = df[id_col].astype(str).str.strip()
    if "_semester" in df.columns:
        sem_col = df["_semester"].astype(str)
        # 效能修正 (2026-07-11)：原本用 list comprehension 對「每一列」個別
        # 呼叫 pd.to_datetime()，在 quiz/reading_log 動輒 30~60萬列的規模下，
        # 等於把同一組（通常只有數千組）(anon_id, semester) 複合鍵的
        # semester_start 重複解析數十萬次——實測單這一步就佔了 Step 3~4
        # 之間 214 秒的絕大部分。改為先對「唯一」複合鍵建立查找表（只解析
        # per_student_exam_dates 內實際存在的鍵，通常僅一兩千筆），再用
        # Series.map() 向量化查回每一列，缺鍵時 map() 原生回傳 NaN，
        # 語意與原本的 NaT 完全相同（該列不受週數截斷、予以保留）。
        composite_key = pd.Series(list(zip(aid, sem_col)), index=df.index)
        unique_keys = set(composite_key)
        start_lookup = {
            key: pd.to_datetime(
                per_student_exam_dates.get(key, {}).get("semester_start"), errors="coerce"
            )
            for key in unique_keys
        }
        sem_start = pd.to_datetime(composite_key.map(start_lookup), errors="coerce")
    else:
        # 無 _semester 欄時無法組成複合鍵，保守回傳全 NaT（該列不受週數截斷）
        sem_start = pd.Series(pd.NaT, index=df.index)
    week = ((ts - sem_start).dt.days // 7) + 1
    in_scope = (
        (df["_semester"].astype(str) == scope_semester)
        if scope_semester and "_semester" in df.columns
        else pd.Series(True, index=df.index)  # 沒指定 scope 或無 _semester 欄 → 沿用舊行為
    )
    keep = (~in_scope) | week.isna() | (week <= week_limit)  # 解析失敗的列保留，不誤刪
    return df[keep].reset_index(drop=True)


# 各 sheet 可能的時間欄候選（見規劃書1.2節／討論4）：quiz 的「測驗時間」與
# reading_log 的「開始」已由既有 09_lsa_event_log.py／05_time_analysis.py／
# _truncate_by_week() 呼叫端驗證存在；其餘欄名取自 09_lsa_event_log.py
# ::_detect_time_col() 既有候選集合，尚未針對每個sheet逐一以完整欄位的
# 正式產出檔驗證過（見規劃書實作階段附註：手上可核對的參考檔為精簡欄位
# 版本，缺時間相關欄），115(1) week13+真實資料到齊後建議依4.3節回歸測試
# 準則覆核，確認各sheet實際有無被正確截斷。
_TRANSITION_WINDOW_TIME_COL_CANDIDATES: tuple = (
    "開始", "測驗時間", "提交時間", "時間", "登入時間", "日期", "存取時間",
)


def _window_by_week(df: pd.DataFrame, time_col_candidates: tuple,
                     window_start, window_end,
                     scope_col: str = "_course_track",
                     scope_value: str = "transition_merged_micro") -> pd.DataFrame:
    """
    依 [window_start, window_end] 絕對日曆日期區間過濾 df（雙邊窗口：
    window_start <= 事件日期 <= window_end），供「基礎醫學臨床應用(一)」
    合併課微免部分（僅week13-18）9個sheet窗口過濾使用（見規劃書1.2節／
    討論4）。

    與既有 _truncate_by_week() 的差異：
    - _truncate_by_week() 是單邊上限截斷（week <= week_limit，相對於各生
      個別 semester_start 計算的「第幾週」），設計給進行中學期的早期預警
      模擬使用，且截斷範圍以 scope_semester（整個學期）為單位。
    - 本函式是雙邊絕對日曆日期窗口，設計給「同一課程代碼混合3科18週
      資料、僅取week13-18」情境使用——合併課教學計畫已有明確課表日期，
      直接用絕對日曆日期比對即可，不需要像早期預警那樣先算「相對於
      學期起始的第幾週」；且截斷範圍以 scope_col/scope_value（課程/
      課程軌，而非整個學期）為單位，因為同一學期內合併課學生與傳統
      結構學生（見規劃書討論9、11）共存，不能整學期一併截斷。

    找不到時間欄、或 window_start／window_end 無法解析時，防呆回傳
    df 原樣（比照 _truncate_by_week() 既有慣例，不中斷主流程、不誤刪）。

    Parameters
    ----------
    df                  : 待過濾 DataFrame
    time_col_candidates : 依序嘗試的時間欄候選名稱
    window_start        : 窗口起始日（含）
    window_end          : 窗口結束日（含）
    scope_col           : 範圍限定欄名，預設 "_course_track"
    scope_value         : 範圍限定值，預設 "transition_merged_micro"

    Returns
    -------
    過濾後的 DataFrame（索引 reset）
    """
    if df is None or df.empty:
        return df
    time_col = next((c for c in time_col_candidates if c in df.columns), None)
    if not time_col:
        print(f"  ℹ️  [窗口過濾] 找不到時間欄（候選：{time_col_candidates}），略過此sheet")
        return df

    w_start = pd.to_datetime(window_start, errors="coerce")
    w_end = pd.to_datetime(window_end, errors="coerce")
    if pd.isna(w_start) or pd.isna(w_end):
        print(f"  ⚠️  [窗口過濾] window_start/window_end 無法解析（{window_start!r}/{window_end!r}），"
              f"略過窗口過濾，回傳原樣")
        return df
    # BUG-FIX（實作階段發現）：window_end 若只給日期（如"2027-01-05"），
    # pd.to_datetime 會解析成當日00:00:00，導致當天全部事件（正常應含在
    # 窗口內）因時分秒精度被排除在外。教學計畫課表week18考試日橫跨兩天
    # （如1/4,5），若只用其中一天當window_end更會整天漏算。改為將
    # window_end正規化到「當日23:59:59.999999」，確保比較以整日為粒度、
    # 不因時分秒精度誤刪本該納入的邊界日資料。
    w_end = w_end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    ts = pd.to_datetime(df[time_col], errors="coerce")
    in_scope = (
        (df[scope_col] == scope_value)
        if scope_col in df.columns
        else pd.Series(False, index=df.index)  # 找不到範圍欄時保守視為「不在範圍內」，不截斷任何列
    )
    keep = (~in_scope) | ts.isna() | ((ts >= w_start) & (ts <= w_end))
    filtered = df[keep].reset_index(drop=True)
    n_removed = int((~keep).sum())
    if n_removed:
        print(f"  [窗口過濾] {scope_value}：移除 {n_removed} 列（{time_col} 不在 "
              f"{w_start.date()}~{w_end.date()} 範圍內）")
    return filtered

# ── 批次多檔工具函式 ──────────────────────────────────────────

def _resolve_lms_files(lms_args: list) -> list:
    files = []
    for path in lms_args:
        if os.path.isdir(path):
            found = sorted([
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.lower().endswith(".xlsx") and not f.startswith("~$")
            ])
            if found:
                print(f"  📁 目錄 {path} 找到 {len(found)} 個 xlsx：")
                for f in found:
                    print(f"     {os.path.basename(f)}")
            else:
                print(f"  ⚠️  目錄 {path} 下未找到 .xlsx 檔案")
            files.extend(found)
        elif os.path.isfile(path):
            files.append(path)
        else:
            print(f"  ⚠️  路徑不存在：{path}")
    return files

# ══════════════════════════════════════════════════════════════
# 非護理系關聯學生過濾（0806 護理系統計/模型污染 預防性根除）
# ══════════════════════════════════════════════════════════════
# 背景：dashboard NON-NURSING-EXCLUSION-0805、etl.py v7.15.11 已分別從
# 前端顯示層與成績回歸統計層排除「健四二F」等非護理系班級。使用者指出
# 「健四二F」該特殊班級本身現實環境已不再招生，要求就實際 LMS 學習紀錄
# 檔案（111-1~114-2，共13個真實檔案）評估是否仍有「非護理系學生行為
# 資料流入護理系風險模型訓練」的根本性風險，並視評估結果一併根除。
#
# 穿透式檢查真實檔案發現：健四二F本身（學號42開頭）確實未出現在任何一份
# 學習紀錄檔案中，但「單位」欄位顯示同一批以「學後護一」「四技」等護理系
# 學制命名的檔案內，仍有其他4種非護理系單位的學生資料（因跨系共同必修課
# 共用同一堂課LMS匯出所致，非單一個案——健康事業管理系橫跨111-1/111-2/
# 112-2/113-1共4個學期出現）：
#   健康事業管理系Department of Health Care Management
#   嬰幼兒保育系Department of Infant and Child Care
#   語言治療與聽力學系Department of Speech Language Pathology and Audiology
# 另有「推廣中心進修教育組」——此為既有、已知的護理系在職專班/推廣教育
# 學生（02_anonymize.py 已有對應 09 開頭手機格式學號特殊處理），非本次
# 排除對象，與「護理系」一併列入白名單。
#
# 「單位」欄位只存在於 summary（學習紀錄統計）sheet，且經 _merge_lms_files()
# 既有註解確認該 sheet 全流程從未被任何模組讀取逐生數值——故本函式將其
# 首次啟用為「帳號→單位」查找表，過濾其餘 8 個 sheet 的學生列，而非改變
# summary sheet 原本「僅用於欄名/存在性檢查」的既有用途。
#
# 必須在 _merge_lms_files() 之後、Step 2 去識別化之前執行：去識別化後
# 帳號已雜湊為 anon_id，無法回頭比對原始「單位」欄位。
_NURSING_UNIT_WHITELIST_KEYWORDS = ("護理系", "推廣中心")


def _build_unit_lookup(sheets: dict, id_col: str) -> dict:
    """從 summary sheet 建立 {原始帳號: 單位} 對照表（原始、未匿名化帳號）。"""
    df = sheets.get("summary")
    if df is None or id_col not in df.columns:
        return {}
    unit_col = next((c for c in df.columns if "單位" in c), None)
    if unit_col is None:
        return {}
    lookup: dict = {}
    for _, row in df[[id_col, unit_col]].dropna(subset=[id_col]).iterrows():
        sid = m02.normalize_student_id(row[id_col])
        unit = str(row[unit_col]).strip() if pd.notna(row[unit_col]) else ""
        if sid and unit:
            lookup[sid] = unit  # 同一生跨學期單位理論上一致，後出現者覆蓋無妨
    return lookup


def _filter_non_nursing_students(sheets: dict, id_col: str) -> dict:
    """
    過濾掉「單位」查得到、且不屬護理系關聯白名單（_NURSING_UNIT_WHITELIST_
    KEYWORDS）的學生列，套用至全部 sheet（含 summary 自身，維持一致性）。

    查無單位資訊者（理論上不應發生——summary 為完整學生名冊，含未參與任何
    活動的學生）一律保留並警告計數，採「不確定就不排除」保守預設，避免
    因資料缺角誤刪真正的護理系學生。
    """
    unit_lookup = _build_unit_lookup(sheets, id_col)
    if not unit_lookup:
        print("  ⚠️  [非護理系過濾] summary sheet 查無「單位」欄位或無資料，略過此步驟")
        return sheets

    excluded_sids = {
        sid for sid, unit in unit_lookup.items()
        if not any(kw in unit for kw in _NURSING_UNIT_WHITELIST_KEYWORDS)
    }
    if not excluded_sids:
        print(f"  ✅ [非護理系過濾] {len(unit_lookup)} 名有單位紀錄的學生皆屬護理系關聯單位，無需排除")
        return sheets

    excluded_units = sorted({unit_lookup[s] for s in excluded_sids})
    print(f"  🚫 [非護理系過濾] 排除 {len(excluded_sids)} 名非護理系關聯學生")
    for u in excluded_units:
        n = sum(1 for s in excluded_sids if unit_lookup[s] == u)
        print(f"      單位「{u}」：{n} 人")

    unmapped_count = 0
    result: dict = {}
    for key, df in sheets.items():
        if df is None or id_col not in df.columns:
            result[key] = df
            continue
        normalized = df[id_col].map(m02.normalize_student_id)
        unmapped_count = max(unmapped_count, int((~normalized.isin(unit_lookup)).sum()))
        mask = ~normalized.isin(excluded_sids)
        removed = (~mask).sum()
        result[key] = df[mask].reset_index(drop=True)
        if removed:
            print(f"      {key}: 移除 {removed} 列")

    if unmapped_count:
        print(f"  ℹ️  [非護理系過濾] {unmapped_count} 筆列的帳號於 summary 查無單位紀錄，"
              f"依保守預設予以保留（未排除）")

    return result


def _classify_course_track_by_lms_name(course_name: str) -> "str | None":
    """
    LMS行為紀錄（9-sheet xlsx）側的 course_track 判斷（見規劃書1.1節／
    討論11）。與成績檔側 etl.py::_classify_course_track()（sheet_name比對
    班級名單）不同機制——LMS檔案的「學習紀錄統計」分頁列0確有課程名稱
    文字可用（實測如「1141112300055011R0 微生物及免疫學 - 學習紀錄統計」，
    見 01_load_lms_excel.py::_extract_course_header()），故此側改用課程
    名稱字串比對，不需要像成績檔側那樣仰賴班級名單。

    注意：既有「微生物及免疫學(合併班)」是行政上的班級/場次合併（多班
    共用同一場考試/題庫），與本次「基礎醫學臨床應用(一)」的3科課程內容
    合併是不同概念，比對字串刻意選「基礎醫學臨床應用」而非泛用的「合併」
    二字，避免誤判前者。
    """
    if not course_name:
        return None
    return "transition_merged_micro" if "基礎醫學臨床應用" in course_name else None


def _merge_lms_files(file_paths: list) -> "tuple[dict, dict]":
    """
    讀取多個 LMS xlsx 並合併同名 sheet 的 DataFrame。
    每個檔案的 DataFrame 會補上 _semester 欄（由檔名解析）與 _section_id 欄
    （由檔案內嵌課程代碼解析，v6新增，見 01_load_lms_excel.py::_extract_course_header()）。

    v7.8新增：同時回傳 course_info_by_semester（{semester: {course_id, course_name}}）。
    BUG-FIX (2026-07-12)：課程代碼/名稱先前雖然已由 01_load_lms_excel.py 解析出來，
    但只被拿來當本函式內的區域變數 file_section_id（僅用於逐列標記 _section_id），
    從未傳遞出這個函式——這是 course_name/course_id 於所有輸出 JSON 恆為空字串的
    根因之一（run() 組裝 course_meta 時只查 exam_dates.json config，而該 config
    的 schema 從未實際存過這兩個欄位）。改為依學期彙整（取每學期「第一個成功
    解析出非空值的檔案」，比照 01_load_lms_excel.py 內同一份「取第一筆非空值」
    慣例），回傳給呼叫端組裝 course_meta 使用。
    """
    all_sheets: dict[str, list] = {}
    course_info_by_semester: dict = {}

    for fpath in file_paths:
        print(f"\n  ── 讀取：{os.path.basename(fpath)}")
        try:
            _result = m01.load_lms_excel(fpath)
            # v7.8：load_lms_excel 現在回傳 (sheets, missing, course_id, course_name)
            # 4-tuple；保留舊格式相容分支，避免其他呼叫端或舊版模組尚未同步時整批中斷。
            if isinstance(_result, tuple) and len(_result) >= 4:
                sheets, file_section_id, file_course_name = _result[0], _result[2], _result[3]
            elif isinstance(_result, tuple) and len(_result) >= 3:
                sheets, file_section_id, file_course_name = _result[0], _result[2], ""
            elif isinstance(_result, tuple):
                sheets, file_section_id, file_course_name = _result[0], "", ""
            else:
                sheets, file_section_id, file_course_name = _result, "", ""
        except Exception as e:
            print(f"  ❌ 讀取失敗，略過：{e}")
            continue

        file_semester = parse_semester_from_filename(os.path.basename(fpath))

        if file_semester and file_semester not in course_info_by_semester:
            if file_section_id or file_course_name:
                course_info_by_semester[file_semester] = {
                    "course_id":   file_section_id,
                    "course_name": file_course_name,
                }

        for key, df in sheets.items():
            if df is not None and not df.empty:
                # 標記每筆資料的來源學期與課程/班級代碼（供後續 per-semester
                # 分析、行事曆驗證使用）
                df = df.copy()
                df["_semester"] = file_semester
                df["_section_id"] = file_section_id
                # course_track（見規劃書1.1節／討論11）：LMS側用課程名稱字串
                # 比對，逐列標記，供 _window_by_week() 的 scope_col 篩選使用。
                df["_course_track"] = _classify_course_track_by_lms_name(file_course_name)
                all_sheets.setdefault(key, []).append(df)

    merged: dict = {}
    for key, dfs in all_sheets.items():
        combined = pd.concat(dfs, ignore_index=True).dropna(how="all").reset_index(drop=True)

        if key == "summary":
            # 防禦性修正（v5）：summary（學習紀錄統計）目前全流程唯一用途是
            # 供 _detect_id_col() 偵測欄名、REQUIRED_SHEETS 存在性檢查，
            # 沒有任何模組讀取其逐生數值（已確認 04/05/06 均不引用），故此處
            # 的去重策略對現有輸出無實質影響。但原本僅用 id_col 去重，重修生
            # 橫跨兩學期的 summary 列會被去重掉一筆（保留 keep="last" 那學期）
            # ——與全專案「一律用 (id, semester) 複合鍵」的一致性原則不符，
            # 一旦未來有模組開始讀 summary 逐生數值就會是潛伏bug，故一併改
            # 用複合鍵，防患於未然。
            id_candidates = [c for c in combined.columns if "帳號" in c or "學號" in c]
            if id_candidates:
                id_col = id_candidates[0]
                dedup_subset = [id_col, "_semester"] if "_semester" in combined.columns else [id_col]
                combined = combined.drop_duplicates(subset=dedup_subset, keep="last").reset_index(drop=True)

        merged[key] = combined

    for req_key in m01.REQUIRED_SHEETS:
        if req_key not in merged:
            print(f"  ⚠️  所有檔案均缺少必要 Sheet：'{req_key}'")
            merged[req_key] = None

    print("\n  ✅ 合併完成：")
    for key, df in merged.items():
        if df is not None:
            print(f"     {key}: {len(df)} 列（合併自 {len(all_sheets.get(key, []))} 個檔案）")

    return merged, course_info_by_semester

if __name__ == "__main__":
    args = parse_args()
    run(args)
