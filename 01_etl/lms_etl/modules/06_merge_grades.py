"""
lms_etl/modules/06_merge_grades.py
讀取現有 data.json，將成績欄位合併至學習行為資料。
同時計算相關係數矩陣與散佈圖資料。
"""

import re
import sys
import json
import os
import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as _stats

# v7.15.10（/systematic-debugging 穿透式審查）：is_4tech 學制別特徵改由此處
# 統一計算，改用 etl.py 已審查過的 _classify_program()（依 sheet_name 班級
# 名稱識別實際學制），取代下游 11_cross_analysis.py／12_early_warning.py
# 原本各自獨立、僅憑「學期末碼」猜測學制的 fallback heuristic。
# 該 heuristic 對絕大多數正規班級恰好正確（因為本校 2 年制班級的正課記錄
# 恰好都歸檔在「-1」（第1學期）檔案、4 年制恰好都在「-2」（第2學期）檔案），
# 但對「同一學期檔案內混雜的重修班/遠距班」會誤判——例如 109學年度第2學期
# 檔案中的「護二二R0遠距」（重修班，非四技班），純學期末碼判斷會誤標為
# is_4tech=1，但 _classify_program() 依班級名稱正確識別為 retake_class。
# 詳見 CHANGELOG_v7.md v7.15.10。
#
# 為求穩健（無論此模組是由 lms_etl.py 的 importlib 動態載入，或日後被
# 獨立執行/測試），明確將專案根目錄（etl.py 所在目錄）加入 sys.path，
# 不依賴呼叫端是否已設定好。
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))  # lms_etl/modules → lms_etl → 專案根目錄
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import etl as _etl  # noqa: E402  （見上方 sys.path 設定說明，import 順序為刻意安排）

# 及格門檻：與 11_cross_analysis.PASS_THRESHOLD 保持一致（單一來源由 11 定義，此處鏡像）
PASS_THRESHOLD: float = 60.0

# v6新增（22-7規劃書Phase 0，問題2）：extract_grade_table() 的宣告schema單一來源。
# 找不到任何符合條件的record時，用這份欄位清單建構空DataFrame，確保回傳的一定是
# 「有正確欄位、只是0列」，而不是完全columnless的DataFrame——後者會讓下游
# merge_behavior_grades() 的 .merge(on=[...]) 直接 KeyError 當機。
GRADE_TABLE_COLUMNS = [
    "source_id", "anon_id", "masked_id", "semester", "sheet_name", "type",
    "midterm_score", "final_score", "semester_score", "reading_pct", "attend_pct",
]

# [AUDIT-FIX 穿透式審查 0914] 原本這裡是本檔案自己另外維護一份
# _RE_FLOAT_SID / _RE_VALID_SID / _normalize_student_id()，docstring
# 宣稱「與 etl.py / 02_anonymize.py 完全一致」，但 v7.15.26 移除
# etl.py／02_anonymize.py 兩邊 normalize_student_id() 的 zfill(9) 時，
# 這第三份複本沒有一併修正，仍保留 zfill(9)——docstring 的「完全一致」
# 因此已是假話。且本檔案開頭已經 `import etl as _etl`（供
# PHANTOM_STUDENT_ID／mask_student_id／_classify_program() 使用），
# 不像 etl.py 與 02_anonymize.py 兩者有「刻意不互相 import」的架構
# 理由，維護這第三份複本沒有正當性，屬於「同一段邏輯多處維護、其中
# 一處走鐘」。修正：直接呼叫 _etl.normalize_student_id() /
# _etl.RE_STUDENT_ID，砍掉本地複本，讓未來不可能再次各自走鐘（而非
# 只是重新同步一次、留著下次再度分岔的風險）。
# 真實資料驗證：此路徑僅在 data.json 的 anon_id 缺失時觸發（見下方
# _anon_from_source_id() 呼叫點），31 個真實歷史成績檔學號皆為 8-9碼
# 或 09 開頭 10 碼，移除 zfill 前後行為相同、零回歸風險。

def _anon_from_source_id(source_id, salt: str = "") -> str:
    """
    當 data.json 內 anon_id 為空時，以 salt 即時重算 SHA-256(salt+sid)[:8]。
    salt 為空時回傳空字串。
    """
    if not salt:
        return ""
    sid = _etl.normalize_student_id(source_id)
    if not _etl.RE_STUDENT_ID.match(sid):
        return ""
    raw = (salt + sid).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:8]

def _safe_float(val) -> "float | None":
    try:
        f = float(val)
        return None if f != f else f   # NaN self-inequality check，避免 np.isnan
    except (TypeError, ValueError):
        return None

def _is_nan(val) -> bool:
    """純 Python NaN 偵測，不依賴 numpy（用於 scalar 值）。"""
    try:
        return val != val
    except TypeError:
        return False


def _has_valid_anon(series_or_df) -> bool:
    """
    統一判斷 anon_id 欄位是否有至少一個非空值。
    接受 Series 或含 'anon_id' 欄的 DataFrame。
    """
    if hasattr(series_or_df, 'columns'):
        if 'anon_id' not in series_or_df.columns:
            return False
        col = series_or_df['anon_id']
    else:
        col = series_or_df
    return (col.astype(str).str.strip() != "").any()

# ── 讀取 data.json ────────────────────────────────────────────

def load_data_json(data_json_path: str = "docs/data/data.json") -> dict:
    """讀取現有成績 data.json，回傳原始 dict。"""
    if not os.path.exists(data_json_path):
        raise FileNotFoundError(f"找不到 data.json：{data_json_path}")
    with open(data_json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_grade_table(
    data_json: dict,
    target_semester=None,
    target_course_keyword: str = "",
    salt: str = "",
) -> pd.DataFrame:
    """
    從 data.json 提取成績表。

    Parameters
    ----------
    data_json             : data.json 的原始 dict
    target_semester       : 學期篩選，支援三種格式：
                            - None 或 "" 或 []：不篩選，取全部學期
                            - str（如 "1141"）：只取該學期
                            - list（如 ["1111","1112","1121"]）：只取清單內的學期
    target_course_keyword : data.json 中 sheet_name 欄的篩選關鍵字（選填，空白表示全部）。
                             注意：sheet_name 通常是班級/場次名稱（如「護21丙」），
                             不是課程科目名稱，見下方「找不到資料」時的診斷提示。
    salt                  : 用於補算空 anon_id

    Returns
    -------
    DataFrame with columns（見 GRADE_TABLE_COLUMNS；找不到資料時回傳同樣欄位、0列）：
        source_id, anon_id, masked_id, semester, sheet_name, type,
        midterm_score, final_score, semester_score, reading_pct, attend_pct
    """
    if not target_semester:
        semester_filter: set = set()
    elif isinstance(target_semester, str):
        semester_filter = {target_semester}
    else:
        semester_filter = set(target_semester)

    rows = []
    # v6新增（22-7 Phase 0）：記錄「符合學期篩選、但被課程關鍵字濾掉」的 sheet_name，
    # 供找不到任何資料時的診斷提示使用。
    skipped_by_course_filter = []
    students = data_json.get("students", data_json)
    recomputed = 0
    n_phantom_skipped = 0

    for source_id, student_info in students.items():
        # BUG-FIX 第二層防線（見 etl.py::PHANTOM_STUDENT_ID 定義處說明）：
        # source_id 為補零假學號時整個學生跳過，不進入任何後續聚合計算。
        # run_etl() 理論上已於寫出 data.json 前排除，這裡是防禦性重複
        # 檢查——尤其保護「尚未用修正後 etl.py 重新產生」的舊版 data.json
        # 檔案，避免其中殘留的假學生記錄流入 merged_df。
        if source_id == _etl.PHANTOM_STUDENT_ID:
            n_phantom_skipped += 1
            continue

        # BUG-FIX（v7.15.15穿透式審查）：原本「student_info.get("name_masked")
        # or source_id」在 name_masked 缺失時直接把「未遮蔽的原始學號」
        # source_id 填入 masked_id 欄，讓下游一律信任「masked_id欄=已遮蔽」
        # 的所有輸出（如07e/07b的CSV/JSON匯出）在此情境下實際外洩真實學號。
        # 對照下方 anon_id 的處理方式（缺失時用 _anon_from_source_id()
        # 重新「正確計算」出安全值，而非直接借用一個語意不符的原始值）——
        # masked_id 理應採用同一慣例：缺失時呼叫同一顆遮蔽函式重新計算，
        # 而不是走捷徑退回未遮蔽原始值。mask_student_id() 為純函式、成本
        # 極低，且此模組已於檔案開頭 import _etl，可直接呼叫。
        masked_id = student_info.get("name_masked") or _etl.mask_student_id(source_id)
        records = student_info.get("records", [])
        anon_id = str(student_info.get("anon_id", "")).strip()

        if not anon_id and salt:
            anon_id = _anon_from_source_id(source_id, salt)
            if anon_id:
                recomputed += 1

        for rec in records:
            if semester_filter and rec.get("semester") not in semester_filter:
                continue
            if target_course_keyword and target_course_keyword not in rec.get("sheet_name", ""):
                skipped_by_course_filter.append(rec.get("sheet_name", ""))
                continue
            # practicum 為獨立課程（60% 成績為報告，未記入 LMS），排除以防止污染分群計算
            if rec.get("type") == "practicum":
                continue

            rows.append({
                "source_id":      source_id,
                "anon_id":        anon_id,
                "masked_id":      masked_id,
                "semester":       rec.get("semester", ""),
                "sheet_name":     rec.get("sheet_name", ""),
                "type":           rec.get("type", ""),
                "midterm_score":  _safe_float(rec.get("midterm")),
                "final_score":    _safe_float(rec.get("final")),
                "semester_score": _safe_float(rec.get("semester_score")),
                "reading_pct":    _safe_float(rec.get("reading_pct")),
                "attend_pct":     _safe_float(rec.get("attend_pct")),
            })

    # v6修正（22-7 Phase 0）：明確指定 columns，rows=[] 時也會產生「有正確欄位、
    # 只是0列」的 DataFrame，而不是完全 columnless 的 pd.DataFrame([])——
    # 後者會讓下游 merge_behavior_grades() 的 .merge(on=[...]) 直接 KeyError 當機。
    df = pd.DataFrame(rows, columns=GRADE_TABLE_COLUMNS)
    if n_phantom_skipped:
        print(f"  ⚠️  [06] 排除 {n_phantom_skipped} 名學號為 "
              f"{_etl.PHANTOM_STUDENT_ID!r} 的殘留假學生（防禦性重複檢查，"
              f"若看到此訊息代表 data.json 是用修正前的 etl.py 產生的，"
              f"建議重新跑一次完整 etl.py 更新 data.json 本身）")
    if df.empty:
        scope = f"學期 {sorted(semester_filter)}" if semester_filter else "所有學期"
        print(f"  ⚠️  在 data.json 中找不到 {scope} 的成績資料")
        if target_course_keyword and skipped_by_course_filter:
            sample = sorted(set(skipped_by_course_filter))[:10]
            print(f"  ℹ️  --course \"{target_course_keyword}\" 沒有比對到任何資料，"
                  f"但該範圍內 data.json 實際的 sheet_name 值包含：{sample}")
            print("      提醒：--course 比對的是 data.json 的 sheet_name 欄"
                  "（通常是班級/場次名稱），不是課程科目名稱；"
                  "若不需要篩選特定班級，請改用空字串 --course \"\"")
        return df

    # ── 補考去重：同 (anon_id, semester) 有多筆 theory → 取 semester_score 最高者 ──
    # 成因：補考學生在同一學期有正課 + 補考兩筆記錄，應以較高分數計算，
    #        避免低分記錄覆蓋補考通過的成績。
    # 優先鍵：anon_id（有值時）；fallback：masked_id。
    join_key_dedup = "anon_id" if _has_valid_anon(df["anon_id"]) else "masked_id"
    dedup_keys = [join_key_dedup, "semester"]
    dup_mask = df.duplicated(subset=dedup_keys, keep=False)   # 計算一次，後面複用
    if dup_mask.any():
        dup_students = df.loc[dup_mask, join_key_dedup].nunique()
        print(f"  ℹ️  偵測到 {dup_students} 名學生同學期有多筆 theory 記錄（補考），取 semester_score 最高者")
        # NaN 視為 0 排序，確保有成績的列優先保留
        df = (
            df.sort_values("semester_score", ascending=False, na_position="last")
              .drop_duplicates(subset=dedup_keys, keep="first")
              .reset_index(drop=True)
        )

    semesters = df["semester"].unique().tolist()
    print(f"  [06] 成績表：{len(df)} 筆（{df['masked_id'].nunique()} 名學生，學期：{semesters}）（practicum 已排除）")
    if recomputed > 0:
        print(f"  ℹ️  {recomputed} 名學生的 anon_id 由 salt 即時補算（data.json 為舊版空值）")
        print("      建議重新執行 etl.py 以更新 data.json 中的 anon_id 欄位。")
    empty_anon = df["anon_id"].astype(str).str.strip().eq("").sum()
    if empty_anon > 0:
        print(f"  ⚠️  {empty_anon} 筆 anon_id 仍為空（source_id 格式非學號，或未提供 salt），將 fallback 至 masked_id join")
    return df

# ── 合併行為與成績 ────────────────────────────────────────────

def merge_behavior_grades(
    behavior_students: list,
    grade_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    將行為資料與成績表以 (anon_id, semester)（或 (masked_id, semester) 作
    fallback）進行 left join。

    Parameters
    ----------
    behavior_students : behavior.json 的 students 陣列
    grade_df          : extract_grade_table 產出的成績 DataFrame

    Returns
    -------
    合併後的 DataFrame（含行為 features + 成績欄位）

    v6新增發現與修正（Phase 5 相容性複查時發現，非v7原十項決議之一，
    但屬同一根因家族——直接關係到Phase 3的複合鍵修正是否真正生效）：
    施作前複查本函式時發現，原本的 .merge(on=join_key) 只用單一 anon_id
    （或masked_id）當join key，並未如22-1除錯報告v7決議5/Phase 5小節
    所假設的「本來就是複合鍵join」。Phase 3把 behavior_students 從
    「一anon_id一筆」改成「一(anon_id,semester)一筆」後，若grade_df中
    同一anon_id橫跨多學期各有一筆成績記錄（重修生常見情境），單鍵join
    會產生 M×N 筆cross-product輸出——同一anon_id的behavior列會被錯誤地
    同時配對到「別學期」的grade列。下游 lms_etl.py 的 _merged_index_2key
    是用merged_df每一列自己的(anon_id,semester)重新當key插入dict，
    cross-product情境下同一組(anon_id,semester) key會有兩筆候選紀錄
    互相覆蓋，最終保留哪一筆屬未定義行為（dict後寫入者覆蓋先寫入者），
    有實質機率讓某位重修生的某學期查到「另一學期」的behavior特徵，
    等同Bug B在merge層重演一次。改為複合鍵join直接消除cross-product。
    """
    records = []
    for s in behavior_students:
        row = {
            "anon_id":   s["anon_id"],
            "masked_id": s["masked_id"],
            "semester":  s.get("semester", ""),
            "cluster":   s.get("cluster", ""),
        }
        row.update(s.get("features", {}))
        records.append(row)

    behavior_df = pd.DataFrame(records)

    has_valid_anon = _has_valid_anon(grade_df)
    join_key = "anon_id" if has_valid_anon else "masked_id"

    # v6：只有當雙邊都有 semester 欄、且 behavior_df 的 semester 非全空時，
    # 才能用複合鍵join；否則退回單鍵（理論上不應發生，behavior_students
    # 自Phase 3起必定帶semester，保留此防呆避免中斷主流程）。
    use_composite = (
        "semester" in grade_df.columns
        and "semester" in behavior_df.columns
        and behavior_df["semester"].astype(str).ne("").any()
    )
    join_keys = [join_key, "semester"] if use_composite else [join_key]
    print(f"  [06] 成績合併 join key：{join_keys}"
          + ("" if use_composite else "（⚠️ 無法取得複合鍵，退回單鍵join，"
             "重修生若grade_df跨學期有多筆記錄可能錯誤配對）"))

    grade_cols = [c for c in [
        "anon_id", "masked_id", "semester", "midterm_score", "final_score",
        "semester_score", "reading_pct", "attend_pct", "type",
        # v7.15.10：新增 sheet_name，供 build_scatter_data() 正確計算
        # is_4tech（見檔案開頭 import etl 處說明），先前未選入此白名單，
        # 導致 sheet_name 雖存在於 grade_df 卻在此處被靜默丟棄。
        "sheet_name",
    ] if c in grade_df.columns]

    if use_composite:
        # semester 已是 join key，merge 時不需要 grade_df 端重複的 semester
        # 欄（否則 pandas 仍會照常合併，但確保雙邊 dtype 一致以策安全）
        behavior_df["semester"] = behavior_df["semester"].astype(str)
        grade_df = grade_df.copy()
        grade_df["semester"] = grade_df["semester"].astype(str)

    merged = behavior_df.merge(
        grade_df[grade_cols],
        on=join_keys,
        how="left",
        suffixes=("", "_grade"),
    )

    if "masked_id_grade" in merged.columns:
        merged["masked_id"] = merged["masked_id"].fillna(merged["masked_id_grade"])
        merged = merged.drop(columns=["masked_id_grade"])

    score_cols = [c for c in ["midterm_score", "final_score", "semester_score"] if c in merged.columns]
    truly_missing = (
        merged[score_cols].isna().all(axis=1).sum() if score_cols else len(merged)
    )
    if truly_missing > 0:
        print(f"  ⚠️  {truly_missing} 名學生找不到對應成績（join_key={join_keys} 不符，請確認 salt 與 data.json 來源一致）")

    if len(merged) != len(behavior_df):
        print(f"  ⚠️  [Phase 5] merge後列數（{len(merged)}）與behavior_df列數"
              f"（{len(behavior_df)}）不一致，可能仍有cross-product或重複"
              f"grade_df列，請檢查 extract_grade_table 的去重邏輯")

    print(f"  [06] 合併完成：{len(merged)} 筆，含成績欄位")
    return merged

# ── 相關係數計算 ──────────────────────────────────────────────

FEATURE_COLS = [
    "aud_total_minutes",
    "vid_total_minutes",
    "txt_completion_rate",
    "sup_completion_rate",
    "tut_total_minutes",
    "quz_total_attempts",
    "total_learning_minutes",
    "consistency_score",
    "pre_exam_intensity",
    # Ph2b 新增：題庫品質指標
    "quz_first_attempt_accuracy",
    "quz_final_accuracy",
    "quz_score_delta",
    "quz_cramming_ratio",
    # [ETL-1] 07c_stability.py 回寫，學習時數週間變異係數（std/mean）
    "stability_cv",
]

TARGET_COLS = ["midterm_score", "final_score"]

def compute_pearson_matrix(
    merged_df: pd.DataFrame,
    feature_cols: list = None,
    target_cols: list = None,
    excluded_features: list = None,
) -> dict:
    """
    計算 FEATURE_COLS × TARGET_COLS 的 Pearson r 矩陣。
    使用 scipy.stats.pearsonr 取得 p-value。
    p-value 下限 1e-9，避免前端顯示 0.0。

    Parameters
    ----------
    excluded_features : 從全體計算中排除的特徵欄位（如某學期才新增的教材類型，
                        該學期前資料必定近 0，納入全體計算會造成相關性偏差）。
                        排除的欄位仍出現在結果 dict 中，值為
                        {r: None, p: None, significant: False, reason: "excluded_new_material"}。

    Returns
    -------
    { target: { feature: {"r": float|None, "p": float|None, "significant": bool} } }
    """
    _feature_cols = feature_cols if feature_cols is not None else FEATURE_COLS
    _target_cols  = target_cols  if target_cols  is not None else TARGET_COLS
    _excluded     = set(excluded_features) if excluded_features else set()

    available_features = [c for c in _feature_cols if c in merged_df.columns]
    available_targets  = [c for c in _target_cols  if c in merged_df.columns]

    missing_features = set(_feature_cols) - set(available_features)
    if missing_features:
        print(f"  ⚠️  缺少特徵欄位（略過）：{missing_features}")
    if _excluded:
        print(f"  ℹ️  [全體相關性] 排除新增教材特徵（學期偏差防護）：{_excluded}")

    # 預先轉換所有特徵 Series，避免在 target × feature 雙層迴圈內重複 to_numeric
    feat_numeric: dict = {
        feat: pd.to_numeric(merged_df[feat], errors="coerce")
        for feat in available_features
        if feat not in _excluded
    }
    # 預先轉換所有目標 Series（同理）
    target_numeric: dict = {
        t: pd.to_numeric(merged_df[t], errors="coerce")
        for t in available_targets
    }

    result = {}
    for target in available_targets:
        result[target] = {}
        target_series = target_numeric[target]
        t_notna = target_series.notna()
        for feat in available_features:
            # 被排除的特徵：填入 None 並標記原因，前端可顯示說明而非空白
            if feat in _excluded:
                result[target][feat] = {
                    "r": None, "p": None, "significant": False,
                    "reason": "excluded_new_material",
                }
                continue
            feat_series = feat_numeric[feat]
            valid_mask  = t_notna & feat_series.notna()
            n_valid     = valid_mask.sum()
            fv = feat_series[valid_mask]
            tv = target_series[valid_mask]
            # std() 各算一次並快取，避免在條件判斷中重複計算
            f_std = fv.std()
            t_std = tv.std()
            if n_valid < 5 or f_std == 0 or t_std == 0:
                result[target][feat] = {"r": None, "p": None, "significant": False}
                continue
            r_val, p_val = _stats.pearsonr(fv.values, tv.values)
            r_val = float(r_val) if not np.isnan(r_val) else None
            p_val = float(p_val) if not np.isnan(p_val) else None
            result[target][feat] = {
                "r":           round(r_val, 4) if r_val is not None else None,
                "p":           round(max(p_val, 1e-9), 9) if p_val is not None else None,
                "significant": bool(p_val is not None and p_val < 0.05),
            }

    return result

def compute_subgroup_pearson(
    merged_df: pd.DataFrame,
    feature_cols: list = None,
    target_cols: list = None,
    subgroup_col: str = "edu_type",
    min_samples: int = 10,
    excluded_features: list = None,
) -> dict:
    """
    依學制（edu_type）分層計算 Pearson r 矩陣，校準辛普森悖論。

    excluded_features : 與 compute_pearson_matrix 一致，發生規模性變動
                        （scale_change）的欄位於各子群一併排除並標記，
                        維持與全體 pearson_matrix 口徑一致。

    Returns
    -------
    { "subgroup_value": { target: { feature: {r, p, significant} } } }
    """
    if subgroup_col not in merged_df.columns:
        print(f"  ⚠️  找不到分組欄位 {subgroup_col!r}，跳過分層相關性計算")
        return {}

    result = {}
    for group_val, sub_df in merged_df.groupby(subgroup_col, dropna=True):
        if len(sub_df) < min_samples:
            print(f"  ⚠️  學制 {group_val!r} 樣本數 {len(sub_df)} < {min_samples}，略過")
            continue
        result[str(group_val)] = compute_pearson_matrix(
            sub_df, feature_cols=feature_cols, target_cols=target_cols,
            excluded_features=excluded_features,
        )

    if result:
        print(f"  [06] 分層相關性：{list(result.keys())}")
    return result

def build_scatter_data(merged_df: pd.DataFrame, cluster_map: dict = None) -> list:
    """
    產出散佈圖資料點列表（每名學生一筆），僅保留有成績的學生。

    Parameters
    ----------
    merged_df   : merge_behavior_grades 產出的合併 DataFrame
    cluster_map : 可選，{ (anon_id, semester): cluster_code }，補入分群資訊。
                  注意：merged_df 本身的 "cluster" 欄（由 merge_behavior_grades
                  依 (anon_id, semester) 複合鍵正確賦值）優先採用；cluster_map
                  僅作為該欄缺漏/空值時的補值來源。
                  v4 修復：舊版 cluster_map 以純 anon_id/masked_id 為 key、且
                  優先於 merged_df 自帶欄位套用，導致重修生較舊學期的正確
                  cluster 值被較新學期的分群結果覆蓋（與 Bug B 同一根因模式）。
    """
    scatter_features = {
        "aud_total_minutes",
        "vid_total_minutes",
        "txt_completion_rate",
        "sup_completion_rate",
        "tut_total_minutes",
        "quz_total_attempts",
        "total_learning_minutes",
        "consistency_score",
        "pre_exam_intensity",
        "quz_first_attempt_accuracy",
        "quz_final_accuracy",
        "quz_score_delta",
        "quz_cramming_ratio",
        "quz_pass_rate",
        "aud_completion_rate",
        "vid_completion_rate",
        "active_weeks",
        "weekly_minutes_std",
        "cram_pattern_score",
        "late_night_ratio",
        "early_start_ratio",
        # v7.14新增（交接文件v10「第三階段」）：若不加進這個白名單，
        # XGB_FEATURES 即使列了這兩欄，也會因為 scatter_data 裡沒有
        # 這兩欄而永遠讀到 NaN（被 median imputer 補成常數，訓練上
        # 等於沒有這個特徵）。
        "reading_inflation_ratio",
        "policy_gaming_flag",
    }
    available_feats = sorted(scatter_features & set(merged_df.columns))

    # 至少一個成績欄位有值才納入
    has_score_mask = merged_df["midterm_score"].notna() | merged_df["final_score"].notna()
    sub = merged_df[has_score_mask].copy()

    # 向量化特徵 dict：先截出特徵子集，round → object，NaN→None，轉 records list
    feat_sub = sub[available_feats].apply(pd.to_numeric, errors="coerce").round(4)
    feat_records = feat_sub.where(feat_sub.notna(), other=None).to_dict("records")

    # cluster：優先採用 merged_df 自帶欄位（已依 (anon_id, semester) 複合鍵
    # 正確賦值，見 merge_behavior_grades）；僅在該欄缺漏/空字串時，才 fallback
    # 到呼叫端傳入的 cluster_map（複合鍵 (anon_id, semester) → cluster_code）。
    # v4 修復：避免重修生較舊學期的正確值被單鍵 cluster_map 覆蓋。
    own_cluster = sub.get("cluster", pd.Series("", index=sub.index)).astype(str)
    if cluster_map:
        composite_keys = list(zip(
            sub["anon_id"].astype(str),
            sub["semester"].astype(str) if "semester" in sub.columns else pd.Series("", index=sub.index),
        ))
        fallback_vals = pd.Series(
            [cluster_map.get(k, "") for k in composite_keys], index=sub.index
        )
        cluster_vals = own_cluster.where(own_cluster.str.len() > 0, fallback_vals)
    else:
        cluster_vals = own_cluster.values

    # 向量化取 midterm/final（已是 float 欄，直接用 where 轉 None）
    mid_vals  = sub["midterm_score"].where(sub["midterm_score"].notna(), other=None).tolist()
    fin_vals  = sub["final_score"].where(sub["final_score"].notna(),  other=None).tolist()

    # v7.15.10（/systematic-debugging）：is_4tech 改在此處統一、正確計算
    # （見檔案開頭 import etl 處完整說明），11_cross_analysis.py 與
    # 12_early_warning.py 之後直接讀取此欄位，不再各自重算。
    # sheet_name 理論上必存在（extract_grade_table 恆寫入該欄，見
    # GRADE_TABLE_COLUMNS），僅在極舊快取資料缺此欄時 fallback 至舊版
    # 純學期末碼判斷，並印出警告以利發現。
    if "sheet_name" in sub.columns:
        sheet_names_for_4tech = sub["sheet_name"].astype(str).tolist()
    else:
        print("  ⚠️  [06] merged_df 缺少 sheet_name 欄，is_4tech 退回舊版"
              "純學期末碼判斷（不精確，建議重新執行完整 pipeline 以更新快取）")
        sheet_names_for_4tech = [""] * len(sub)
    semesters_for_4tech = sub["semester"].astype(str).tolist() if "semester" in sub.columns else [""] * len(sub)
    is_4tech_vals = [
        1 if _etl._classify_program(sn, sm) == "4yr" else 0
        for sn, sm in zip(sheet_names_for_4tech, semesters_for_4tech)
    ]

    # course_track（見規劃書決策1／討論11）：比照 is_4tech 同一模式，統一在此
    # 計算一次，下游模組（11_cross_analysis.py／12_early_warning.py／
    # 07d_density_filter.py）直接讀取，不再各自重算。成績檔無課程名稱欄位，
    # 只能用 sheet_name 比對已知班級名單（_classify_course_track()），
    # 與 is_4tech 共用同一組 sheet_names_for_4tech／semesters_for_4tech。
    course_track_vals = [
        _etl._classify_course_track(sn, sm)
        for sn, sm in zip(sheet_names_for_4tech, semesters_for_4tech)
    ]

    rows = [
        {
            "anon_id":       str(r["anon_id"])   if not _is_nan(r.get("anon_id"))   else "",
            "masked_id":     str(r["masked_id"]) if not _is_nan(r.get("masked_id")) else "",
            "cluster":       str(cv),
            "semester":      str(r.get("semester") or ""),
            "edu_type":      str(r.get("type") or ""),
            "is_4tech":      is_4tech_vals[i],
            "course_track":  course_track_vals[i],
            "features":      feat_records[i],
            "midterm_score": mid_vals[i],
            "final_score":   fin_vals[i],
        }
        for i, (r, cv) in enumerate(zip(sub.to_dict("records"), cluster_vals))
    ]

    print(f"  [06] scatter_data：{len(rows)} 筆有效資料點")
    return rows

# ── 題庫週別行為彙整 ──────────────────────────────────────────

TOTAL_WEEKS   = 18
MIDTERM_WEEK  = 9
FINAL_WEEK    = 18

# BUG-FIX (2026-07-12，交接文件v3第1節，取代v2的結論)：v2曾推測1142的
# 「第15週」「第17週」完全沒有題庫、是真實無資料——經人工核對後這個推測
# 是錯的：這兩週的題庫確實存在，只是用了主題式命名「醫用寄生蟲學 練習題I」
# 「醫用寄生蟲學 練習題II」，沒有照「第X週」格式命名，導致現行正規表示式
# `第(\d+)週` 抓不到週次，這兩千多筆真實作答紀錄（練習題I：2,636筆，
# 練習題II：1,205筆）在 quz_only.dropna(subset=["_week_num"]) 這一步
# 被整批捨棄，segments 裡完全沒有任何 "1142|..." 開頭的 W15/W17 key。
#
# 這組映射已經過人工核對確認，可直接硬編碼套用。
#
# 順序很重要："醫用寄生蟲學 練習題II" 是 "醫用寄生蟲學 練習題I" 的父字串
# （"練習題II" = "練習題I" + "I"）。用子字串包含（in）判斷時，若沒有把
# 較長、較具體的字串排在前面比對，"練習題I" 的規則會誤判命中「練習題II」
# 的標題，因此下面刻意把 "練習題II" 排在 "練習題I" 之前，一旦命中就
# return，不會誤判。
_TITLE_WEEK_OVERRIDE_1142 = [
    ("醫用寄生蟲學 練習題II", 17),   # 順序必須在「練習題I」之前
    ("醫用寄生蟲學 練習題I",  15),
]

_WEEK_TITLE_RE      = re.compile(r"第(\d+)週")
_SELFSTUDY_TITLE_RE = re.compile(r"自學週(\d+)")


def _extract_quiz_week_num(title) -> float:
    """
    從題庫標題擷取週次（第一階段，語意上不含「自學週」，向量化
    `.map()` 呼叫，效能與原本相同）。優先比對「第X週」；找不到時比對
    已人工確認的1142硬編碼例外清單；全部都找不到則回傳 NaN。

    「自學週N」這類標題的週次對應改由第二階段
    `_resolve_selfstudy_week()` 處理（v7.11起，見下方），因為不同學期
    對應的實際週次可能不同（112-1／112-2／113-1因「16+2週」行政特例，
    需要逐學期設定，不能像1142一樣全域寫死一份映射）。
    """
    if not isinstance(title, str):
        return float("nan")
    m = _WEEK_TITLE_RE.search(title)
    if m:
        return float(m.group(1))
    for substr, wk in _TITLE_WEEK_OVERRIDE_1142:
        if substr in title:
            return float(wk)
    return float("nan")


def _resolve_selfstudy_week(title, semester: str, self_study_week_map: dict) -> float:
    """
    v7.11新增：第二階段，僅套用在第一階段（`_extract_quiz_week_num`）
    找不到週次、且標題比對得到「自學週N」的列，依
    `self_study_week_map[semester][N]` 查表得到實際週次。

    背景：112-1／112-2／113-1三學期因行政特例整學期只有16正課週，另外
    安排W17/W18作為2週補充內容（「16+2週」結構），這類學期的補充題庫
    常用「自學週1」「自學週2」等非「第X週」格式命名。與1142的
    「醫用寄生蟲學」不同，這幾個學期的自學週對應週次無法直接全域硬編碼
    （不同學期可能不同、且交接文件v2/v3皆指出僅為統計推論、尚待課程端
    確認），因此改為透過 `exam_dates.json` 各學期的 `self_study_weeks`
    設定（由 lms_etl.py::_prompt_week_structure() 互動確認後寫入）逐學期
    查表；找不到對應學期或對應index的設定時，回傳NaN（安全預設：不列入
    週別統計，不會誤植錯誤週次）。

    self_study_week_map 結構：{semester: {"1": week_num, "2": week_num, ...}}
    （由 lms_etl.py::run() 依 exam_dates_all 的 self_study_weeks 陣列
    轉換而來，索引位置對應「自學週1」「自學週2」……）。

    只在第一階段找不到週次的（通常很小的）子集合上以 `.apply(axis=1)`
    逐列處理，不影響 `_extract_quiz_week_num()` 對全量資料的向量化效能。
    """
    if not isinstance(title, str) or not self_study_week_map:
        return float("nan")
    m = _SELFSTUDY_TITLE_RE.search(title)
    if not m:
        return float("nan")
    sem_map = self_study_week_map.get(str(semester), {})
    mapped = sem_map.get(m.group(1))
    return float(mapped) if mapped is not None else float("nan")


def _resolve_date_based_week(
    subset: pd.DataFrame,
    time_col: str,
    date_fallback_start_map: dict,
    total_weeks: int,
) -> pd.Series:
    """
    v7.14.7新增：第三階段，僅套用在第一、二階段（標題文字比對——
    「第X週」正規表示式、1142硬編碼清單、self_study_week_map）皆
    找不到週次的（通常很小的）殘留子集合。

    背景（ETL交接：06_merge_grades.py 題庫週次解析殘留缺口）：真實資料
    顯示「醫用寄生蟲學」單元的題庫標題在不同學期/班級至少有三種命名
    慣例：
      1. 「第15週 醫用寄生蟲學 練習題I」（含週次）—— 第一階段涵蓋。
      2. 「醫用寄生蟲學 練習題I／II」（不含週次，1142專用命名）——
         `_TITLE_WEEK_OVERRIDE_1142` 硬編碼涵蓋。
      3. 「醫用寄生蟲學 練習題庫」（不含週次、不分I/II）—— 出現於
         111-1、111-2、112-1（學後護一重修班）、112-2，前兩階段完全
         不涵蓋，全數回傳NaN，被`quz_only.dropna(subset=["_week_num"])`
         整批捨棄，造成這幾個學期的「各週題庫作答強度」圖表在寄生蟲
         教學週次顯示為空白（112-2實測：W14/W15/W16共2,141筆真實作答
         紀錄，完全消失於週別統計）。

    採用「用測驗時間戳依 semester_start 反推週次」而非再新增一條標題
    字串規則，理由與hms bug／TAG_RULES覆蓋率缺口相同：標題比對是封閉式
    白名單，只能涵蓋已知措辭，無法保證涵蓋未來新學期可能出現的第4種
    命名慣例；時間戳反推不依賴標題措辭，可泛化到任何同類「無週次資訊」
    標題。公式與 lms_etl.py::_truncate_by_week() 完全一致：
    `week = (timestamp - semester_start).days // 7 + 1`，clip至
    `[1, total_weeks]`（與 05_time_analysis.py::compute_weekly_rhythm()
    的clip慣例一致，避免極端離群時間戳落到範圍外的無效週次）。

    date_fallback_start_map 結構：{(anon_id, semester): semester_start}
    （由 lms_etl.py::run() 從 per_student_exam_dates 篩出 semester_start
    轉換而來，已內建各學期/各section的正確 semester_start，包含
    111-1／112-1等有 calendar_overrides 的學期）。

    只在（通常很小的）殘留子集合上運算，不影響第一階段對全量資料的
    向量化效能；沿用 _truncate_by_week() 的「unique key 先建查找表、
    再 Series.map() 向量化查回」技巧，避免逐列 apply。
    """
    if subset.empty or not date_fallback_start_map or time_col not in subset.columns:
        return pd.Series(float("nan"), index=subset.index)
    if "anon_id" not in subset.columns or "_semester" not in subset.columns:
        return pd.Series(float("nan"), index=subset.index)

    anon_series = subset["anon_id"].astype(str)
    sem_series  = subset["_semester"].astype(str)
    composite_key = pd.Series(list(zip(anon_series, sem_series)), index=subset.index)
    unique_keys = set(composite_key)
    start_lookup = {
        key: pd.to_datetime(date_fallback_start_map.get(key), errors="coerce")
        for key in unique_keys
    }
    sem_start = pd.to_datetime(composite_key.map(start_lookup), errors="coerce")
    ts = pd.to_datetime(subset[time_col], errors="coerce")

    week = ((ts - sem_start).dt.days // 7).clip(0, total_weeks - 1) + 1
    return week.where(sem_start.notna() & ts.notna())

def build_quiz_behavior_by_week(
    quiz_df: pd.DataFrame,
    grade_df: pd.DataFrame,
    behavior_students: list = None,
    pass_threshold: float = PASS_THRESHOLD,
    total_weeks: int = TOTAL_WEEKS,
    midterm_week: int = MIDTERM_WEEK,
    final_week: int = FINAL_WEEK,
    self_study_week_map: dict = None,
    date_fallback_start_map: dict = None,
    sixteen_plus_two_semesters: set = None,
) -> list:
    """
    將測驗紀錄依週次彙整，產出 quiz_behavior.json 的 weeks 陣列。
    輸出固定 total_weeks 週，考試週保留佔位並標記 is_exam_week。

    Parameters
    ----------
    quiz_df        : 含 anon_id, masked_id, category, 測驗標題, 狀態 的 DataFrame
    grade_df       : 成績表（含 masked_id, semester_score）
    pass_threshold : 及格分數線（預設 60）
    total_weeks    : 學期總週數（預設 18）
    midterm_week   : 期中考週次（預設 9）
    final_week     : 期末考週次（預設 18）
    self_study_week_map : v7.11新增。{semester: {"1": week_num, "2": week_num, ...}}，
                          供「自學週N」命名的題庫標題查表對應到實際週次
                          （見 _resolve_selfstudy_week()）。由呼叫端依
                          exam_dates.json 各學期的 self_study_weeks 設定
                          轉換而來；未提供或某學期查無設定時，該學期的
                          「自學週」標題維持原行為（不列入週別統計）。
    date_fallback_start_map : v7.14.7新增。{(anon_id, semester): semester_start}，
                          供第一、二階段（標題文字比對）皆找不到週次的紀錄，
                          以「該生該學期的 semester_start」將測驗時間戳反推
                          週次（見 _resolve_date_based_week()）。由呼叫端從
                          per_student_exam_dates 篩出 semester_start 轉換而來；
                          未提供或查無對應 (anon_id, semester) 時，該列維持
                          原行為（NaN，不列入週別統計）。
    sixteen_plus_two_semesters : 見規劃書1.3節討論3（小幅修正版，非核心
                          重新分組）。本批次資料中，校歷為「16+2週」特例的
                          學期集合（歷史已知：1121/1122/1131，實際期中考/
                          期末考落在第8/16週而非第9/18週，比照
                          tab-behavior-time.js::SIXTEEN_PLUS_TWO_SEMS 同一組
                          學期與週次判斷）。若提供且非空，頂層（全部學期
                          合併）exam_weeks 除了預設 midterm_week/final_week
                          外，也會額外把第8/16週標記為 midterm/final。

                          範圍澄清（穿透式審查後確認，見規劃書實作階段）：
                          這只影響「全部學期」pooled 聚合視圖本身
                          （quiz_behavior.json 頂層 weeks[i].is_exam_week/
                          exam_type/title），不影響 segments（該dict已依
                          semester|cluster|pass 正確分群，本來就不受
                          midterm_week/final_week 這組全域參數影響，見該
                          dict 的組裝邏輯）。目前唯一已知消費端
                          tab-behavior-time.js 對考試週紅線標記完全採用
                          前端自算的 SIXTEEN_PLUS_TWO_SEMS 邏輯、不讀這幾個
                          頂層欄位（實測驗證見規劃書），此參數屬於「讓 raw
                          JSON本身語意正確、為未來可能的其他消費端預先打底」
                          的低風險小修，非當前畫面可見的行為變更——因此
                          刻意不做更大範圍的 week_groups 依 (semester, week)
                          重新分組（會動到 segments 依賴的核心聚合邏輯，
                          對現有唯一消費端沒有可驗證的效益，風險不對等）。
    """
    # ── 建立及格/不及格分群 ──────────────────────────────────
    pass_set: set = set()
    pass_by_anon_sem: dict = {}

    if grade_df is not None and not grade_df.empty:
        score_series = pd.to_numeric(grade_df["semester_score"], errors="coerce")
        midterm_series = pd.to_numeric(grade_df["midterm_score"], errors="coerce")
        effective_score = score_series.fillna(midterm_series)
        
        pass_rows = grade_df.loc[effective_score >= pass_threshold]

        if _has_valid_anon(grade_df):
            # NaN anon_id 過濾：排除空值，避免 NaN 混入 pass_set 造成誤判
            valid_anon = grade_df["anon_id"].astype(str).str.strip()
            pass_set = set(pass_rows.loc[valid_anon[pass_rows.index].str.len() > 0, "anon_id"])
            # FIX BUG-2: 用 column-exists 檢查取代 DataFrame.get()
            sem_series = (
                grade_df["semester"].astype(str)
                if "semester" in grade_df.columns
                else pd.Series("", index=grade_df.index)
            )
            anon_series  = grade_df["anon_id"].astype(str).str.strip()
            valid_mask   = (anon_series.str.len() > 0) & effective_score.notna()
            # 向量化建字典，避免 iterrows
            pass_by_anon_sem = {
                (a, s): sc >= pass_threshold
                for a, s, sc in zip(
                    anon_series[valid_mask],
                    sem_series[valid_mask],
                    effective_score[valid_mask],
                )
            }
        else:
            pass_set = set(pass_rows["masked_id"])

    student_meta = {
        str(s.get("anon_id", "")): {
            "cluster":            str(s.get("cluster", "")),
            "semester":           str(s.get("semester", "")),
            "quz_final_accuracy": s.get("features", {}).get("quz_final_accuracy"),
        }
        for s in (behavior_students or [])
        if str(s.get("anon_id", ""))
    }

    # ── 篩選題庫類型紀錄 ────────────────────────────────────
    if quiz_df is None or quiz_df.empty:
        quz_only = pd.DataFrame()
    elif "category" in quiz_df.columns:
        quz_only = quiz_df[quiz_df["category"] == "QUZ"].copy()
    else:
        quz_only = quiz_df.copy()

    title_col  = "測驗標題" if not quz_only.empty and "測驗標題" in quz_only.columns else None
    status_col = "狀態"     if not quz_only.empty and "狀態"     in quz_only.columns else None
    time_col   = "測驗時間" if not quz_only.empty and "測驗時間" in quz_only.columns else None

    # ── 從標題提取週次 ───────────────────────────────────────
    week_groups: dict = {}
    if title_col and not quz_only.empty:
        quz_only["_week_num"] = quz_only[title_col].map(_extract_quiz_week_num)

        # v7.11新增：第二階段，僅在（通常很小的）「第一階段找不到週次」
        # 子集合上查表「自學週N」對應的實際週次，不影響第一階段對全量
        # 資料的向量化效能。需要 _semester 欄位才能做到逐學期查表
        # （112-1／112-2／113-1的自學週對應週次可能不同，不能全域套用
        # 同一份映射）。
        if self_study_week_map and "_semester" in quz_only.columns:
            _need_selfstudy = quz_only["_week_num"].isna()
            if _need_selfstudy.any():
                _subset = quz_only.loc[_need_selfstudy]
                quz_only.loc[_need_selfstudy, "_week_num"] = _subset.apply(
                    lambda row: _resolve_selfstudy_week(
                        row[title_col], row["_semester"], self_study_week_map
                    ),
                    axis=1,
                )

        # v7.14.7新增：第三階段，僅在（通常很小的）前兩階段皆找不到週次的
        # 殘留子集合上，用測驗時間戳依 semester_start 反推週次（見
        # _resolve_date_based_week() docstring：涵蓋「醫用寄生蟲學 練習題庫」
        # 這類不含週次、不分I/II的第三種命名慣例，出現於111-1/111-2/
        # 112-1(學後護一重修班)/112-2）。同樣只處理殘留子集合，不影響
        # 第一階段對全量資料的向量化效能。
        if date_fallback_start_map and time_col and "_semester" in quz_only.columns:
            _need_date_fallback = quz_only["_week_num"].isna()
            if _need_date_fallback.any():
                quz_only.loc[_need_date_fallback, "_week_num"] = _resolve_date_based_week(
                    quz_only.loc[_need_date_fallback], time_col,
                    date_fallback_start_map, total_weeks,
                )

        for week_num, grp in quz_only.dropna(subset=["_week_num"]).groupby("_week_num"):
            week_groups[int(week_num)] = grp

    # ── 逐週建立輸出 ─────────────────────────────────────────
    exam_weeks = {midterm_week: "midterm", final_week: "final"}
    if sixteen_plus_two_semesters:
        # 見上方 docstring：只補充頂層pooled視圖的標記，不影響segments。
        exam_weeks.setdefault(8, "midterm")
        exam_weeks.setdefault(16, "final")
    weeks_data = []

    for w in range(1, total_weeks + 1):
        is_exam = w in exam_weeks
        grp = week_groups.get(w)
        has_quiz_data = grp is not None and not grp.empty

        # BUG-FIX (2026-07-13)：原本只要 is_exam 為 True 就直接 continue，
        # 完全不檢查 week_groups.get(w) ——若剛好有題庫資料被對應到這一週
        # （例如self_study_week_map把「自學週2」對應到與期末考同一週的
        # 情況，見112-1/112-2/113-1的「16+2週」特例），這些真實作答資料會
        # 被整批悄悄丟棄，不會出現在輸出裡。改為：只有「這週真的沒有題庫
        # 資料」時才走純佔位路徑；若有題庫資料，改走下方與非考試週相同的
        # 統計計算，並額外標記 is_exam_week/exam_type，讓前端仍能識別這是
        # 考試週，但不會平白遺失真實資料。
        if is_exam and not has_quiz_data:
            weeks_data.append({
                "week":         w,
                "title":        f"第{w}週 {'期中考' if exam_weeks[w] == 'midterm' else '期末考'}",
                "is_exam_week": True,
                "exam_type":    exam_weeks[w],
            })
            continue

        if grp is None or grp.empty:
            weeks_data.append({
                "week":                    w,
                "title":                   f"第{w}週 練習題庫",
                "pass_group_avg_attempts": 0.0,
                "fail_group_avg_attempts": 0.0,
                "overall_pass_rate":       0.0,
                "avg_final_score":         None,
                "active_students":         0,
            })
            continue

        if "anon_id" in grp.columns and pass_by_anon_sem:
            anon_series = grp["anon_id"].astype(str)
            _sem_lookup = {a: m.get("semester", "") for a, m in student_meta.items()}
            sem_series = grp["_semester"].astype(str) if "_semester" in grp.columns else anon_series.map(_sem_lookup).fillna("")
            is_pass = []
            is_fail = []
            for a, s in zip(anon_series, sem_series):
                if (a, s) in pass_by_anon_sem:
                    if pass_by_anon_sem[(a, s)]:
                        is_pass.append(True)
                        is_fail.append(False)
                    else:
                        is_pass.append(False)
                        is_fail.append(True)
                else:
                    is_pass.append(False)
                    is_fail.append(False)
            pass_grp = grp[is_pass]
            fail_grp = grp[is_fail]
        else:
            key_col   = "anon_id" if "anon_id" in grp.columns and pass_set else "masked_id"
            pass_grp  = grp[grp[key_col].isin(pass_set)]
            fail_grp  = grp[~grp[key_col].isin(pass_set)]
        pass_uniq = max(pass_grp["masked_id"].nunique(), 1)
        fail_uniq = max(fail_grp["masked_id"].nunique(), 1)
        active    = grp["masked_id"].nunique()

        entry: dict = {
            "week":                    w,
            "title":                   f"第{w}週 練習題庫",
            "pass_group_avg_attempts": round(len(pass_grp) / pass_uniq, 2),
            "fail_group_avg_attempts": round(len(fail_grp) / fail_uniq, 2),
            "active_students":         active,
        }

        # ── 分層 segments（anon_id 模式） ───────────────────
        if "anon_id" in grp.columns:
            # 向量化建 segment set：預建 lookup dict 用 .map(dict) 取代 lambda，
            # 並預計算 cluster_key → allowed-anon-set，避免逐 segment 重建。
            anon_series = grp["anon_id"].astype(str)

            # Lookup dicts（O(|student_meta|) 建一次，每列 O(1) 查詢）
            _sem_lookup     = {a: m.get("semester", "") for a, m in student_meta.items()}
            _cluster_lookup = {a: m.get("cluster", "")  for a, m in student_meta.items()}

            sem_series = (
                grp["_semester"].astype(str) if "_semester" in grp.columns
                else anon_series.map(_sem_lookup).fillna("")
            )
            cluster_series = anon_series.map(_cluster_lookup).fillna("")

            # 向量化建立 pass_status_series（取代 Python closure list-comp）
            # 優先查 pass_by_anon_sem（學期精確匹配）；fallback 查 pass_set（全局）
            if pass_by_anon_sem:
                pair_series = list(zip(anon_series, sem_series))
                pass_status_series = pd.Series(
                    [
                        ("pass" if pass_by_anon_sem[(a, s)] else "fail")
                        if (a, s) in pass_by_anon_sem
                        else "all"
                        for a, s in pair_series
                    ],
                    index=grp.index,
                )
            else:
                pass_status_series = pd.Series(
                    ["all" for _ in anon_series],
                    index=grp.index,
                )

            # 預建 cluster_key → allowed anon set（最多 5 個 cluster 值，建一次）
            unique_clusters = set(cluster_series.unique()) - {"", "all"}
            _allowed_by_cluster: dict[str, set] = {
                ck: {a for a, m in student_meta.items() if m.get("cluster") == ck}
                for ck in unique_clusters
            }

            # segment_set：從三個 Series 的唯一值組合直接建立，O(unique) 而非 O(n_rows)
            # 舊做法：逐列笛卡兒積，每列迭代 2×2×2=8 個 key → 500 學生×8=4000次
            # 新做法：取三個 Series 的 unique 值集合做笛卡兒積 → 實際 unique 組合數 << n_rows
            # BUG-FIX (2026-07-03)：原判斷式 "_semester" in grp.columns 恆為 False
            # （grp 從未真正擁有這個欄位，sem_series 一律走下方 anon_id→student_meta
            # 的 fallback 查表），導致 unique_sems 永遠只有 {"all"}，per-semester 的
            # segment（如 "1141|all|pass"）從未產生，前端依學期篩選時作答次數柱狀圖
            # 全部空白。sem_series 本身已正確計算，直接取其 unique 值即可。
            unique_sems     = {"all"} | (set(sem_series.unique())     - {""})
            unique_clusters = {"all"} | (set(cluster_series.unique()) - {""})
            unique_passes   = {"all"} | set(pass_status_series.unique())

            # BUG-FIX (2026-07-12，交接文件v2第3節)：原本排除 all|all|all
            # 這個組合（理由是「已由頂層 entry 計算」），但頂層 entry 的
            # overall_pass_rate/avg_final_score 並不包含「總作答次數/總不重複
            # 人數」這個 avg_attempts 數值——前端因此拿不到 ETL 算好的
            # all|all|all 真值，只能退而求其次用「及格組平均×人數+不及格組
            # 平均×人數，除以兩組人數和」去逼近，這個逼近公式會漏掉查無成績、
            # 既不屬於及格組也不屬於不及格組的學生（pass_status_series 的
            # "all" 類別）。sk=ck=pk="all" 時下面的遮罩本來就不做任何過濾
            # （sem_key/cluster_key/pass_key 皆為"all"時對應的if區塊都會跳過，
            # sub = grp 原封不動），跟其他 segment 走的是同一段迴圈、同一套
            # 算法，不需要額外邏輯，移除排除條件即可直接產生正確的全體真值。
            segment_set: set = {
                (sk, ck, pk)
                for sk in unique_sems
                for ck in unique_clusters
                for pk in unique_passes
                if sk and ck and pk
            }

            segments: dict = {}
            for sem_key, cluster_key, pass_key in segment_set:
                # 使用布林遮罩逐步過濾，不預先 copy 整個 grp
                mask = np.ones(len(grp), dtype=bool)
                # BUG-FIX (2026-07-03)：grp 沒有 "_semester" 欄位，改用已計算好的
                # sem_series（index 對齊 grp）進行遮罩，取代原本恆為 False 的判斷。
                if sem_key != "all":
                    mask &= (sem_series == sem_key).values
                if cluster_key != "all":
                    allowed = _allowed_by_cluster.get(cluster_key, set())
                    mask &= grp["anon_id"].astype(str).isin(allowed).values
                sub = grp[mask]
                if pass_key != "all":
                    if pass_by_anon_sem:
                        # pass_status_series 已在 segment 枚舉時向量化建立，直接用索引過濾
                        ps_sub = pass_status_series.loc[sub.index]
                        sub = sub[ps_sub == pass_key]
                    elif pass_set:
                        in_pass = sub["anon_id"].isin(pass_set)
                        sub = sub[in_pass] if pass_key == "pass" else sub[~in_pass]

                active_sub = (
                    sub["masked_id"].nunique() if "masked_id" in sub.columns
                    else sub["anon_id"].nunique()
                )
                # BUG-FIX (2026-07-03)：原本每個 segment 的 overall_pass_rate 都被
                # entry["overall_pass_rate"]（整週、跨學期合併後的及格率）覆蓋，
                # 導致依學期篩選時「週及格率」折線其實還是顯示全部學期合併的數字，
                # 與畫面上選定的學期不符。改為用該 segment 自己的 sub 重新計算。
                seg_pass_rate = (
                    round(sub[sub[status_col] == "及格"].shape[0] / len(sub), 4)
                    if status_col and len(sub) > 0 else 0.0
                )
                segments[f"{sem_key}|{cluster_key}|{pass_key}"] = {
                    "week":              w,
                    "title":             f"第{w}週 練習題庫",
                    "avg_attempts":      round(len(sub) / max(active_sub, 1), 2) if active_sub else 0.0,
                    "active_students":   int(active_sub),
                    "overall_pass_rate": seg_pass_rate,
                }
            entry["segments"] = segments

        # ── 計算該週及格率 ───────────────────────────────────
        if status_col:
            pass_count = grp[grp[status_col] == "及格"].shape[0]
            entry["overall_pass_rate"] = (
                round(pass_count / len(grp), 4) if len(grp) > 0 else 0.0
            )
        else:
            entry["overall_pass_rate"] = 0.0

        # avg_final_score = overall_pass_rate（以及格率代理題庫難度）
        entry["avg_final_score"] = entry["overall_pass_rate"]

        # BUG-FIX (2026-07-03)：移除原本會將全學期合併及格率覆蓋回每個 segment
        # 的迴圈——segments 現在已各自依自己的 sub 正確計算 overall_pass_rate，
        # 這裡不再需要（也不應該）覆蓋。

        # BUG-FIX (2026-07-13)：這一週同時是考試週（is_exam）又有真實題庫
        # 資料時（見上方說明），額外標記 is_exam_week/exam_type，讓前端仍
        # 能識別這是期中/期末考週，同時保留真實統計資料，不互相排斥。
        if is_exam:
            entry["is_exam_week"] = True
            entry["exam_type"] = exam_weeks[w]
            entry["title"] = (
                f"第{w}週 {'期中考' if exam_weeks[w] == 'midterm' else '期末考'}"
                f"（另有對應題庫資料）"
            )

        weeks_data.append(entry)

    print(f"  [06] quiz_behavior：{len(weeks_data)} 週（含考試週佔位）")
    return weeks_data

# ── 異常值閾值計算 ────────────────────────────────────────────

def compute_outlier_thresholds(
    merged_df: pd.DataFrame,
    feature_cols: list = None,
) -> dict:
    """
    計算各特徵的 IQR 和 Z-Score 異常值邊界，供前端靜態讀取。

    Returns
    -------
    { feature: { "iqr_lower", "iqr_upper", "mean", "std", "z_lower", "z_upper" } }
    """
    _feature_cols = feature_cols if feature_cols is not None else FEATURE_COLS
    available = [c for c in _feature_cols if c in merged_df.columns]

    result = {}
    for feat in available:
        series = pd.to_numeric(merged_df[feat], errors="coerce").dropna()
        if len(series) < 10:
            continue
        q       = series.quantile([0.25, 0.75])
        q1, q3  = float(q[0.25]), float(q[0.75])
        iqr      = q3 - q1
        mean, std = float(series.mean()), float(series.std())
        result[feat] = {
            "iqr_lower": round(q1 - 1.5 * iqr, 4),
            "iqr_upper": round(q3 + 1.5 * iqr, 4),
            "mean":      round(mean, 4),
            "std":       round(std, 4),
            "z_lower":   round(mean - 3 * std, 4),
            "z_upper":   round(mean + 3 * std, 4),
        }

    print(f"  [06] outlier_thresholds：{len(result)} 個特徵")
    return result

# ── 相關性洞察計算 ────────────────────────────────────────────

def compute_correlation_insights(
    pearson_matrix: dict,
    weeks_data: list,
    full_pearson_matrix: dict = None,
) -> dict:
    """
    從 Pearson 矩陣和週次資料萃取關鍵洞察，供前端直接讀取。

    Parameters
    ----------
    pearson_matrix      : 頂層（含 excluded_features 排除）的矩陣，用於一般 r 讀取。
    weeks_data          : 週次資料。
    full_pearson_matrix : 可選。無排除的完整矩陣（如 segment_pearson["all|all|all"]["pearson"]）。
                          FIX-DEFECT-C: highest_r / score_delta / cramming 等洞察指標
                          需從完整矩陣讀取，避免 excluded_new_material 造成洞察失真。
                          若未傳入則 fallback 使用 pearson_matrix。

    Returns
    -------
    {
        "highest_r_feature":      { "feature", "target", "r" },
        "score_delta_correlation": { "midterm", "final" },
        "cramming_correlation":    { "midterm", "final" },
        "highest_pass_rate_week":  { "week", "overall_pass_rate" },
        "lowest_pass_rate_week":   { "week", "overall_pass_rate" },
    }
    """
    # FIX-DEFECT-C: 洞察指標優先使用 full_pearson_matrix（無排除），
    # 確保 QUZ 品質指標等被頂層排除的欄位仍可被納入洞察計算。
    _insight_matrix = full_pearson_matrix if full_pearson_matrix else pearson_matrix

    insights: dict = {}

    _hr = _find_highest_r(_insight_matrix)
    insights["highest_r_feature"] = _hr if _hr else {"feature": None, "target": None, "r": None}

    insights["score_delta_correlation"] = {
        "midterm": _get_insight_r(_insight_matrix, "midterm_score", "quz_score_delta"),
        "final":   _get_insight_r(_insight_matrix, "final_score",   "quz_score_delta"),
    }
    insights["cramming_correlation"] = {
        "midterm": _get_insight_r(_insight_matrix, "midterm_score", "quz_cramming_ratio"),
        "final":   _get_insight_r(_insight_matrix, "final_score",   "quz_cramming_ratio"),
    }

    valid_weeks = [
        w for w in weeks_data
        if not w.get("is_exam_week")
        and w.get("overall_pass_rate") is not None
        and w.get("active_students", 0) > 0
    ]
    if valid_weeks:
        top_w    = max(valid_weeks, key=lambda x: x["overall_pass_rate"])
        bottom_w = min(valid_weeks, key=lambda x: x["overall_pass_rate"])
        insights["highest_pass_rate_week"] = {
            "week": top_w["week"], "overall_pass_rate": top_w["overall_pass_rate"]
        }
        insights["lowest_pass_rate_week"] = {
            "week": bottom_w["week"], "overall_pass_rate": bottom_w["overall_pass_rate"]
        }

    return insights

# ── 時間滯後相關性 ────────────────────────────────────────────

def compute_lagged_pearson(
    merged_df: pd.DataFrame,
    quiz_df: pd.DataFrame,          # 保留參數簽名相容性，目前未使用（簡化版）
    semester_start: str,            # 保留參數簽名相容性，目前未使用
    feature_cols: list = None,
    target_cols: list = None,
    front_weeks: tuple = (1, 2, 3),
    back_weeks:  tuple = (14, 15, 16),
    total_weeks: int = 18,
    min_samples: int = 20,
    excluded_features: list = None,
) -> dict:
    """
    時間滯後相關性（簡化版）：
      X  = 整學期學習行為特徵
      前 = midterm_score（前段代理）
      後 = final_score（後段代理）
      lag_delta = back.r - front.r（正值代表行為對後期預測力更強）

    注意：front_weeks / back_weeks / total_weeks 為介面保留參數，
    真正週次滯後需 per-student×per-week 資料，此版本採整學期特徵。

    excluded_features : 與 compute_pearson_matrix 一致，跨學期合併資料中
                        發生規模性變動（scale_change）的欄位，front/back/
                        lag_delta 一律填 None 並標記 reason="scale_change"，
                        避免顯示受規模偏差污染的滯後相關數值。

    Returns
    -------
    {
      "description", "front_target", "back_target",
      "front_weeks", "back_weeks",
      "results": { feature: { "front", "back", "lag_delta" } }
    }
    """
    _excluded     = set(excluded_features) if excluded_features else set()
    _feature_cols = feature_cols if feature_cols is not None else FEATURE_COLS
    _target_cols  = target_cols  if target_cols  is not None else TARGET_COLS

    available_features = [c for c in _feature_cols if c in merged_df.columns]

    front_target = "midterm_score" if "midterm_score" in merged_df.columns else (
        _target_cols[0] if _target_cols else None
    )
    back_target = "final_score" if "final_score" in merged_df.columns else (
        _target_cols[-1] if _target_cols else None
    )

    if not front_target or not back_target or front_target == back_target:
        print("  ⚠️  [lagged] 需要 midterm_score 與 final_score 兩個目標變數，略過")
        return {}

    front_series = pd.to_numeric(merged_df[front_target], errors="coerce")
    back_series  = pd.to_numeric(merged_df[back_target],  errors="coerce")

    # 預先轉換所有特徵 Series，避免在特徵迴圈內重複 to_numeric
    feat_numeric = {
        feat: pd.to_numeric(merged_df[feat], errors="coerce")
        for feat in available_features
        if feat not in _excluded
    }

    results = {}
    for feat in available_features:
        if feat in _excluded:
            results[feat] = {
                "front": None, "back": None, "lag_delta": None,
                "reason": "scale_change",
            }
            continue
        feat_series = feat_numeric[feat]
        front_stat  = _pearsonr_safe(feat_series, front_series, min_samples)
        back_stat   = _pearsonr_safe(feat_series, back_series,  min_samples)
        fr, br      = front_stat.get("r"), back_stat.get("r")
        results[feat] = {
            "front":     front_stat,
            "back":      back_stat,
            "lag_delta": round(br - fr, 4) if fr is not None and br is not None else None,
        }

    print(f"  [06] lagged_pearson：{len(results)} 個特徵（front={front_target}, back={back_target}）")
    return {
        "description":  f"學習行為 vs 前段成績（{front_target}）/ 後段成績（{back_target}）滯後對照",
        "front_target": front_target,
        "back_target":  back_target,
        "front_weeks":  list(front_weeks),
        "back_weeks":   list(back_weeks),
        "results":      results,
    }

# ── 分段預聚合（方案 B）────────────────────────────────────

def _get_insight_r(matrix: dict, target: str, feat: str):
    """從 Pearson 矩陣取出 r 值；支援 dict wrapper 與純數值兩種格式。"""
    val = matrix.get(target, {}).get(feat)
    return val.get("r") if isinstance(val, dict) else val


def _score_series(df: pd.DataFrame) -> pd.Series:
    """
    回傳合併後的成績數值 Series（NaN 表示無成績）。
    優先順序：final_score > semester_score（對齊 _get_score 規格）。
    提取為模組層級：_compute_for_subset 與其他函式可直接呼叫，無需閉包。
    """
    score = None
    for col in ("final_score", "semester_score"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            score = s if score is None else score.fillna(s)
    return score if score is not None else pd.Series(dtype=float, index=df.index)


def _pearsonr_safe(x_series: pd.Series, y_series: pd.Series, min_samples: int) -> dict:
    """
    安全 Pearson r 計算（帶樣本數檢查與常數列保護）。
    呼叫端保證傳入已 to_numeric 的 Series，此處不再重複轉換。
    """
    mask    = x_series.notna() & y_series.notna()
    n_valid = int(mask.sum())
    if n_valid < min_samples:
        return {"r": None, "p": None, "significant": False, "n": n_valid}
    xs = x_series[mask]
    ys = y_series[mask]
    if xs.std() == 0 or ys.std() == 0:
        return {"r": None, "p": None, "significant": False, "n": n_valid}
    r, p = _stats.pearsonr(xs.values, ys.values)
    r = float(r) if not np.isnan(r) else None
    p = float(max(p, 1e-9)) if p is not None and not np.isnan(p) else None
    return {
        "r":           round(r, 4) if r is not None else None,
        "p":           round(p, 9) if p is not None else None,
        "significant": bool(p is not None and p < 0.05),
        "n":           n_valid,
    }

def _find_highest_r(pearson_matrix: dict) -> dict:
    """
    從 Pearson 矩陣掃描最高 |r| 值並回傳摘要。
    複用 compute_correlation_insights() 的掃描邏輯，抽取為可重用輔助函式。

    Parameters
    ----------
    pearson_matrix : { target: { feature: {r, p, significant} | float } }

    Returns
    -------
    {"feature": str, "target": str, "r": float} 或 None（矩陣為空時）
    """
    best_r, best_feat, best_target = 0.0, None, None
    for target, feat_dict in pearson_matrix.items():
        for feat, val in feat_dict.items():
            r = val.get("r") if isinstance(val, dict) else val
            if r is not None and abs(r) > abs(best_r):
                best_r, best_feat, best_target = r, feat, target
    if best_feat is None:
        return None
    return {
        "feature": best_feat,
        "target":  best_target,
        "r":       round(best_r, 4),
    }

def compute_segment_pearson(
    merged_df: pd.DataFrame,
    feature_cols: list = None,
    target_cols: list = None,
    min_samples: int = 10,
    low_confidence_threshold: int = 30,
) -> dict:
    """
    依方案 B（線性展開）預算各分段的 Pearson r 矩陣，
    供前端篩選時直接讀取，取代即時重算。

    分組 key 格式："{semester}|{cluster}|{pass_status}"
    輸出組合（與 compute_class_segment_stats 一致）：
      "all|all|all"          <- 全量
      "<sem>|all|all"        <- 各學期（單維）
      "all|<cluster>|all"    <- 各分群（單維）
      "all|all|pass"         <- 及格學生（單維）
      "all|all|fail"         <- 不及格學生（單維）
      "all|<cluster>|pass"   <- 各分群 × 及格（交叉）
      "all|<cluster>|fail"   <- 各分群 × 不及格（交叉）
      "<sem>|all|pass"       <- 各學期 × 及格（交叉）
      "<sem>|all|fail"       <- 各學期 × 不及格（交叉）
      "<sem>|<cluster>|all"  <- 各學期 × 各分群（交叉）

    信心旗標：
      n < min_samples(10)           -> 跳過，不輸出
      min_samples <= n < 30         -> 輸出，附 low_confidence: true
      n >= low_confidence_threshold -> 正常輸出，low_confidence: false

    Parameters
    ----------
    merged_df                : merge_behavior_grades 產出的合併 DataFrame
    feature_cols             : 特徵欄位（預設 FEATURE_COLS）
    target_cols              : 目標欄位（預設 TARGET_COLS）
    min_samples              : 子集低於此值跳過（預設 10）
    low_confidence_threshold : 低於此值附 low_confidence 旗標（預設 30）

    Returns
    -------
    dict : {
        segment_key: {
            "pearson":        dict,
            "highest_r":      dict | None,
            "student_count":  int,
            "low_confidence": bool,
        }
    }
    """
    _feature_cols = feature_cols if feature_cols is not None else FEATURE_COLS
    _target_cols  = target_cols  if target_cols  is not None else TARGET_COLS

    def _compute_for_subset(sub):
        n = len(sub)
        if n < min_samples:
            return None
        pearson = compute_pearson_matrix(sub, feature_cols=_feature_cols, target_cols=_target_cols)
        return {
            "pearson":        pearson,
            "highest_r":      _find_highest_r(pearson),
            "student_count":  n,
            "low_confidence": n < low_confidence_threshold,
        }

    result = {}

    has_sem     = "semester" in merged_df.columns
    has_cluster = "cluster"  in merged_df.columns

    sem_series     = merged_df["semester"].astype(str).str.replace("-", "", regex=False) if has_sem     else None
    cluster_series = merged_df["cluster"].astype(str)                                    if has_cluster else None
    # FIX-C: has_score 必須從原始數值 Series 取 notna()，
    # 而非從 bool 比較結果取（bool Series 的 notna() 恆 True，
    # 會把「score=NaN → False」的無成績學生誤歸入 fail 組）
    _score = _score_series(merged_df)          # 模組層級函式，不再是閉包
    has_score = _score.notna()
    pass_flag = (_score >= PASS_THRESHOLD) & has_score
    fail_flag = (_score <  PASS_THRESHOLD) & has_score

    # 預先快取子集，避免後續交叉迴圈重複 groupby
    sem_groups     = {sem: sub for sem, sub in merged_df.groupby(sem_series,     dropna=True) if sem}     if has_sem     else {}
    cluster_groups = {c:   sub for c,   sub in merged_df.groupby(cluster_series, dropna=True) if c}       if has_cluster else {}

    # 全量
    entry = _compute_for_subset(merged_df)
    if entry:
        result["all|all|all"] = entry

    # 學期（單維）
    for sem, sub in sem_groups.items():
        entry = _compute_for_subset(sub)
        if entry:
            result[f"{sem}|all|all"] = entry

    # 分群（單維）
    for cluster, sub in cluster_groups.items():
        entry = _compute_for_subset(sub)
        if entry:
            result[f"all|{cluster}|all"] = entry

    # 及格狀態（單維）
    for pass_key, pflag in (("pass", pass_flag), ("fail", fail_flag)):
        entry = _compute_for_subset(merged_df[pflag])
        if entry:
            result[f"all|all|{pass_key}"] = entry

    # 分群 × 及格（交叉）
    for cluster, csub in cluster_groups.items():
        for pass_key, pflag in (("pass", pass_flag), ("fail", fail_flag)):
            entry = _compute_for_subset(csub[pflag.loc[csub.index]])
            if entry:
                result[f"all|{cluster}|{pass_key}"] = entry

    # 學期 × 及格（交叉）
    for sem, ssub in sem_groups.items():
        for pass_key, pflag in (("pass", pass_flag), ("fail", fail_flag)):
            entry = _compute_for_subset(ssub[pflag.loc[ssub.index]])
            if entry:
                result[f"{sem}|all|{pass_key}"] = entry

    # 學期 × 分群（交叉）
    for sem, ssub in sem_groups.items():
        for cluster, csub in cluster_groups.items():
            idx = ssub.index.intersection(csub.index)
            if idx.empty:
                continue
            entry = _compute_for_subset(ssub.loc[idx])
            if entry:
                result[f"{sem}|{cluster}|all"] = entry

    low_conf = sum(1 for v in result.values() if v.get("low_confidence"))
    print(
        f"  [06] segment_pearson：{len(result)} 組"
        f"（min_samples={min_samples}，low_confidence={low_conf} 組）"
    )
    return result


# ── 教材規模性變動排除（Max-Multiple + MIN_DELTA）──────────────────────────
#
# 演算法：當期增量 > max(歷史增量) × K  且  當期增量 >= MIN_DELTA
#
# 參數選擇依據（以 1111–1142 實際資料為準）：
#   K=2        : 確保「超過歷史單期最大增量的 2 倍」才觸發，過濾正常年度擴充
#   MIN_DELTA=10: 歷史最大正當增量為 QUZ+15；10 保留 QUZ 成長空間，
#                 排除 SUP+4 等小幅補充，物理語意為「單學期新增 10 件以上
#                 才屬規模性教學策略變動」

_CAT_FEATURE_MAP: dict[str, list[str]] = {
    # AUD-FIX (2026-06): aud_completion_rate 為 0-1 正規化比率，不隨 item_count
    # 規模變動系統性偏移，移出排除清單（僅保留 aud_total_minutes）。
    "AUD": ["aud_total_minutes"],
    "VID": ["vid_total_minutes"],
    # TXT/SUP 性質近 VID/TUT（教材規模變動屬真實偏差，非如 QUZ 常態年增），維持排除。
    "TXT": ["txt_completion_rate"],
    "SUP": ["sup_completion_rate"],
    "TUT": ["tut_total_minutes"],
    # REMOVED-QUZ-EXCLUDE (2026-06): 題庫題數每年常態性新增（期中/期末考題庫擴充），
    # 屬正常課程調整而非規模性偏差，故移除 quz_total_attempts 排除規則，
    # 回歸全體相關性正常計算。
}

# DERIVED-EXCLUDE: total_learning_minutes 為各類別 *_total_minutes 加總，
# 任一原始分鐘數類別（AUD/VID/TUT）觸發規模性排除時，此衍生欄位需連帶排除，
# 否則會繼承同樣的規模偏差卻不會顯示「Δ規模」提示。
_DERIVED_EXCLUDE_TRIGGER_COLS: set[str] = {
    "aud_total_minutes", "vid_total_minutes", "tut_total_minutes",
}
_DERIVED_EXCLUDE_FEATURE: str = "total_learning_minutes"

_OUTLIER_K: float = 2.0    # 倍數門檻：當期增量需超過歷史最大增量 × K
_OUTLIER_MIN_DELTA: int = 10  # 絕對下限：低於此件數的增量一律不排除

def compute_material_outlier_cats(
    students: list[dict],
    current_semester: str,
    semesters: list[str],
    return_detail: bool = False,
) -> "list[str] | tuple[list[str], list[dict]]":
    """
    偵測哪些教材類別在 current_semester 發生規模性變動，應排除於全體相關性計算。

    觸發條件（兩者同時成立）：
      1. curr_delta > max(historical_deltas) × _OUTLIER_K
      2. curr_delta >= _OUTLIER_MIN_DELTA

    防呆：
      - 歷史期數 < 2：無法建立基準，跳過（視為正常）
      - 歷史增量全為 0（max=0）：改以 curr_delta >= _OUTLIER_MIN_DELTA 單一條件判斷

    Parameters
    ----------
    students         : behavior.json 的 students 陣列（需含 semester + features）
    current_semester : 當期學期代碼（如 "1142"）
    semesters        : 所有學期清單（已排序）
    return_detail    : v7.10新增（交接文件v2第4節）。預設 False 維持向後相容
                        （僅回傳 excluded_cols，唯一既有呼叫端 lms_etl.py 沿用
                        舊行為不受影響）。設為 True 時額外回傳排除明細清單，
                        供 correlation_matrix.json 的 meta.excluded_material_detail
                        使用——目前前端「全體相關性計算說明」面板的排除類別
                        文字是寫死在 help-modal.js 的靜態字串（含具體學期與
                        增量數字），ETL 重跑後不會自動更新；補上這份明細後，
                        前端之後可以改成動態讀取，不必再手動改程式碼同步。

    Returns
    -------
    return_detail=False（預設）：list[str]，應排除的 FEATURE_COLS 欄位名稱
        （空清單 = 無規模性變動）
    return_detail=True：(excluded_cols, detail) 2-tuple，detail 為每個觸發
        排除的類別各一筆 dict：
        {"category", "feature_cols", "current_semester",
         "current_delta", "historical_max_delta"}
    """
    cats = list(_CAT_FEATURE_MAP.keys())

    # Step 1：各學期各類別教材目錄大小（學生 item_count 的 MAX 值）
    sem_max: dict[str, dict[str, int]] = defaultdict(lambda: {c: 0 for c in cats})
    for s in students:
        sem = str(s.get("semester", ""))
        feats = s.get("features", {})
        for cat in cats:
            val = int(feats.get(f"{cat.lower()}_item_count") or 0)
            if val > sem_max[sem][cat]:
                sem_max[sem][cat] = val

    ordered_sems = [str(s) for s in semesters if str(s) in sem_max]
    if current_semester not in ordered_sems:
        print(f"  ⚠️  [outlier_cats] current_semester={current_semester!r} 不在學期清單，跳過")
        return ([], []) if return_detail else []

    curr_idx = ordered_sems.index(current_semester)
    if curr_idx < 2:
        # 歷史增量序列需至少 1 筆（curr_idx >= 2 才有 ≥1 個歷史 delta）
        print("  ℹ️  [outlier_cats] 歷史期數不足，跳過規模性變動檢測")
        return ([], []) if return_detail else []

    # Step 2：歷史增量序列（不含 current_semester）
    hist_deltas: dict[str, list[int]] = {c: [] for c in cats}
    for i in range(1, curr_idx):
        for cat in cats:
            hist_deltas[cat].append(
                sem_max[ordered_sems[i]][cat] - sem_max[ordered_sems[i - 1]][cat]
            )

    # 當期增量
    prev_sem = ordered_sems[curr_idx - 1]
    curr_deltas = {
        cat: sem_max[current_semester][cat] - sem_max[prev_sem][cat]
        for cat in cats
    }

    # 預建每個類別對應的 FEATURE_COLS 欄位清單（避免在輸出迴圈重複 .get()）
    _cat_cols: dict[str, list[str]] = {
        cat: [c for c in _CAT_FEATURE_MAP.get(cat, []) if c in FEATURE_COLS]
        for cat in cats
    }

    # Step 3：Max-Multiple 判斷
    excluded_cols: list[str] = []
    detail: list[dict] = []
    for cat in cats:
        hist   = hist_deltas[cat]
        curr_d = curr_deltas[cat]
        hist_max = max(hist) if hist else 0

        if hist_max == 0:
            is_outlier = curr_d >= _OUTLIER_MIN_DELTA
            print(f"  ℹ️  [outlier_cats] {cat}: hist_max=0 → curr_delta={curr_d} >= {_OUTLIER_MIN_DELTA}? {is_outlier}")
        else:
            threshold  = hist_max * _OUTLIER_K
            is_outlier = (curr_d > threshold) and (curr_d >= _OUTLIER_MIN_DELTA)
            print(f"  ℹ️  [outlier_cats] {cat}: hist_max={hist_max} threshold={threshold:.0f} curr_delta={curr_d} → outlier={is_outlier}")

        if is_outlier:
            cols = _cat_cols[cat]
            excluded_cols.extend(cols)
            print(f"  ⚠️  [outlier_cats] {cat} 觸發規模性變動排除，欄位：{cols}")
            detail.append({
                "category":             cat,
                "feature_cols":         cols,
                "current_semester":     current_semester,
                "current_delta":        curr_d,
                "historical_max_delta": hist_max,
            })

    # DERIVED-EXCLUDE：若任一原始分鐘數類別被排除，total_learning_minutes
    # 為其加總，繼承同樣規模偏差，需連帶排除（避免漏標 Δ規模）。
    if (set(excluded_cols) & _DERIVED_EXCLUDE_TRIGGER_COLS
            and _DERIVED_EXCLUDE_FEATURE in FEATURE_COLS
            and _DERIVED_EXCLUDE_FEATURE not in excluded_cols):
        excluded_cols.append(_DERIVED_EXCLUDE_FEATURE)
        print(f"  ⚠️  [outlier_cats] {_DERIVED_EXCLUDE_FEATURE} 為衍生欄位，連帶排除")
        detail.append({
            "category":             _DERIVED_EXCLUDE_FEATURE,
            "feature_cols":         [_DERIVED_EXCLUDE_FEATURE],
            "current_semester":     current_semester,
            "current_delta":        None,
            "historical_max_delta": None,
        })

    return (excluded_cols, detail) if return_detail else excluded_cols

if __name__ == "__main__":
    print("merge_grades 模組載入成功")
    tests = [("82", 82.0), ("", None), (None, None), ("76.5", 76.5)]
    for val, expected in tests:
        result = _safe_float(val)
        status = "✓" if result == expected else "✗"
        print(f"  {status} _safe_float({val!r}) = {result}")
