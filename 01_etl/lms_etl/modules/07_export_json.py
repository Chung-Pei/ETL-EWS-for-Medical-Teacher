"""
lms_etl/modules/07_export_json.py
將所有計算結果序列化為 docs/data/ 下的靜態 JSON 檔案。
v2.1：export_time_distribution_json() 新增全班/分群預聚合欄位
      class_heatmap_matrix、class_avg_hourly_distribution、cohort_hourly
      students[] 每筆新增 heatmap_matrix、hourly_distribution
"""

import gzip
import json
import logging
import os
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── 輸出路徑設定 ──────────────────────────────────────────────
OUTPUT_DIR = "docs/data"

OUTPUT_FILES = {
    "behavior":           "behavior.json",
    "radar":              "radar_chart_data.json",
    "correlation":        "correlation_matrix.json",
    "quiz_behavior":      "quiz_behavior.json",
    "time_distribution":  "time_distribution.json",
    "cross_analysis":     "cross_analysis.json",
    "xgb_validation":         "xgb_validation.json",
    "xgb_feature_importance": "xgb_feature_importance.json",
    # warning 檔名含學期變數，實際輸出時動態組合為 warning_{semester}.json
    "warning_template":   "warning_{semester}.json",
}

# ── JSON Schema 版本常數（升版時只需改此處）─────────────────────
# 對應各 export_*() 輸出的 meta.schema_version 欄位。
# 前端 feature detection 依賴這些版本字串，請同步更新前端相容性判斷。
SCHEMA_VERSION_BEHAVIOR    = "3.0"   # export_behavior_json
SCHEMA_VERSION_RADAR       = "2.0"   # export_radar_chart_json
SCHEMA_VERSION_CORRELATION = "2.2"   # export_correlation_json / export_quiz_behavior_json
# v2.2（v7.15.10）：scatter_data 各筆新增 is_4tech 欄位（改由 06_merge_grades.
# build_scatter_data() 依 sheet_name 正確計算，取代下游各自重算的錯誤
# heuristic，見 CHANGELOG_v7.md v7.15.10）。向下相容之 additive minor bump。
SCHEMA_VERSION_TIME_DIST   = "2.1"   # export_time_distribution_json

def _ensure_dir(path: str) -> None:
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

def _write_json(data: Any, filepath: str, label: str = "") -> None:
    _ensure_dir(filepath)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            _sanitize_json_value(data),
            f,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
            allow_nan=False,
        )
    size_kb = os.path.getsize(filepath) / 1024
    logger.info("✅ %s  →  %s  (%.1f KB)", label or os.path.basename(filepath), filepath, size_kb)

def _write_json_gz(data: Any, filepath: str, label: str = "") -> None:
    """輸出 gzip 壓縮 JSON（主要用於大型檔案如 behavior.json）。
    同時輸出未壓縮 .json（相容）與 .json.gz（主要下載對象）。
    separators=(',', ':') 省略縮排，進一步減少原始體積。
    """
    _ensure_dir(filepath)
    sanitized = _sanitize_json_value(data)
    json_bytes = json.dumps(
        sanitized,
        ensure_ascii=False,
        separators=(",", ":"),   # 省略縮排，減少體積
        default=_json_default,
        allow_nan=False,
    ).encode("utf-8")

    # 壓縮版（.json.gz）
    gz_path = filepath + ".gz"
    with gzip.open(gz_path, "wb", compresslevel=9) as f:
        f.write(json_bytes)

    # 未壓縮版（保留相容性）：加縮排後重新編碼，複用 sanitized 避免重複序列化
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False)

    size_orig = len(json_bytes) / 1024
    size_gz   = os.path.getsize(gz_path) / 1024
    ratio     = (1 - size_gz / size_orig) * 100 if size_orig > 0 else 0
    logger.info(
        "✅ %s → %s (%.1f KB → %.1f KB gz, 壓縮率 %.0f%%)",
        label or os.path.basename(filepath), filepath, size_orig, size_gz, ratio
    )

def _sanitize_json_value(value: Any) -> Any:
    """Convert numpy/pandas NaN-like values to valid JSON nulls before export."""
    if isinstance(value, dict):
        return {k: _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _sanitize_json_value(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value

def _json_default(obj: Any) -> Any:
    """處理 numpy 型別序列化（兜底 fallback，_sanitize_json_value 為主）。"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"不支援的型別：{type(obj)}")

def _course_meta_fields(course_meta: dict) -> dict:
    """共用的 meta 欄位（不含 semesters — 各 export 函式依需求個別設定 effective_semesters）。"""
    fields = {
        "semester":             course_meta.get("semester", ""),
        "semester_range":       course_meta.get("semester_range", ""),
        "semester_range_label": course_meta.get("semester_range_label", ""),
    }
    return {k: v for k, v in fields.items() if v not in ("", None)}

# ── behavior.json ─────────────────────────────────────────────

def export_behavior_json(
    students: list,
    course_meta: dict,
    material_counts: dict,
    output_dir: str = OUTPUT_DIR,
    generated_at: str = "",
) -> str:
    """輸出 behavior.json + behavior.json.gz（策略 C：gzip 壓縮）。
    students 需已包含 anon_id, masked_id, cluster, features, time_profile, semester。
    gz 版體積約為原始的 25-30%（5.5MB → ~1.5MB），前端透過 _fetchWithGzFallback 自動取用。
    practicum 學生（edu_type='practicum'）已於 extract_grade_table 排除，此處為防禦性確認。
    """
    # 防禦性過濾：確保 practicum 學生不進入輸出（正常情況下此過濾不會移除任何學生）
    prac_filtered = [s for s in students if s.get("edu_type") == "practicum"]
    if prac_filtered:
        logger.warning("[07] export_behavior_json：偵測到 %d 名 edu_type='practicum'，已從輸出中移除", len(prac_filtered))
        students = [s for s in students if s.get("edu_type") != "practicum"]

    student_semesters = sorted(set(
        str(s.get("semester", "")) for s in students if s.get("semester")
    ))
    effective_semesters = course_meta.get("semesters") or student_semesters

    data = {
        "meta": {
            "schema_version":   SCHEMA_VERSION_BEHAVIOR,
            **_course_meta_fields(course_meta),
            "semesters":        effective_semesters,
            "course_id":        course_meta.get("course_id", ""),
            "course_name":      course_meta.get("course_name", ""),
            "generated_at":     generated_at or datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "student_count":    len(students),
            "material_counts":  material_counts,
        },
        "students": students,
    }
    fpath = os.path.join(output_dir, OUTPUT_FILES["behavior"])
    _write_json_gz(data, fpath, "behavior.json")  # ← 改用 gzip 版
    return fpath

# ── radar_chart_data.json ─────────────────────────────────────

def export_radar_chart_json(
    radar_data: dict,
    course_meta: dict,
    output_dir: str = OUTPUT_DIR,
    generated_at: str = "",
) -> str:
    """輸出 radar_chart_data.json。"""
    radar_meta = radar_data.get("meta", {}) if isinstance(radar_data, dict) else {}
    by_semester = radar_data.get("by_semester", {})
    semester_keys = sorted(by_semester.keys()) if by_semester else []
    effective_semesters = (
        radar_meta.get("semesters") or
        course_meta.get("semesters") or
        semester_keys
    )
    data = {
        **{k: v for k, v in radar_data.items() if k != "meta"},
        "meta": {
            **radar_meta,
            "schema_version": SCHEMA_VERSION_RADAR,
            **_course_meta_fields(course_meta),
            "semesters":      effective_semesters,
            "generated_at":   generated_at or datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "student_count":  radar_meta.get("student_count", 0),
        },
    }
    fpath = os.path.join(output_dir, OUTPUT_FILES["radar"])
    _write_json(data, fpath, "radar_chart_data.json")
    return fpath

# ── correlation_matrix.json ───────────────────────────────────

def export_correlation_json(
    pearson_matrix: dict,
    scatter_data: list,
    feature_cols: list,
    target_cols: list,
    course_meta: dict,
    output_dir: str = OUTPUT_DIR,
    subgroup_pearson: dict = None,
    correlation_insights: dict = None,
    outlier_thresholds: dict = None,
    lagged_pearson: dict = None,
    segment_pearson: dict = None,
    lsa_transition: dict = None,
    incomplete_semesters: list = None,
    incomplete_pearson: dict = None,
    excluded_material_detail: list = None,
    generated_at: str = "",
) -> str:
    """輸出 correlation_matrix.json（Pearson r 矩陣 + 散佈圖資料）。

    Ph2b 升級（schema_version 2.1）：
      pearson              : { target: { feature: {"r", "p", "significant"} } }
      subgroup_pearson     : 學制分層相關性（可選）
      correlation_insights : ETL 預算洞察（可選）
      outlier_thresholds   : 各特徵 IQR/Z 邊界（可選）
      lagged_pearson       : 時間滯後相關性（可選，Phase C）
      excluded_material_detail : v7.10新增（交接文件v2第4節）教材規模性變動
                                 排除明細（可選）——前端「全體相關性計算
                                 說明」面板原本把排除類別/學期/增量數字寫死
                                 在 help-modal.js，ETL 重跑換學期或教材類別
                                 後不會自動更新；補上這份明細後，前端後續
                                 可以改成動態讀取這裡的資料組字串。
    """
    data = {
        "meta": {
            "schema_version": SCHEMA_VERSION_CORRELATION,
            **_course_meta_fields(course_meta),
            "semesters":      course_meta.get("semesters", []),
            "generated_at":   generated_at or datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "student_count":  len(scatter_data),
            "pearson_note":   "r values now wrapped in {r, p, significant} objects",
        },
        "features":     feature_cols,
        "targets":      target_cols,
        "pearson":      pearson_matrix,
        "scatter_data": scatter_data,
    }
    if subgroup_pearson:
        data["subgroup_pearson"] = subgroup_pearson
    if correlation_insights:
        data["correlation_insights"] = correlation_insights
    if outlier_thresholds:
        data["outlier_thresholds"] = outlier_thresholds
    if lagged_pearson:
        data["lagged_pearson"] = lagged_pearson
    if segment_pearson:
        data["segment_pearson"] = segment_pearson
    if incomplete_semesters:
        data["meta"]["incomplete_semesters"] = incomplete_semesters
        data["meta"]["incomplete_semesters_note"] = (
            "以下學期尚未獲得學末成績，已排除於全體相關性計算；"
            "如需查看預警分析，請在前端選擇該學期單獨檢視。"
        )
    if incomplete_pearson:
        data["incomplete_pearson"] = incomplete_pearson
    if excluded_material_detail:
        data["meta"]["excluded_material_detail"] = excluded_material_detail
    if lsa_transition:
        # 透傳所有 public key（排除 _ 開頭的內部欄位），
        # 避免 11_cross_analysis 新增輸出 key 時需同步修改此處。
        data["lsa_transition"] = {
            k: v for k, v in lsa_transition.items() if not k.startswith("_")
        }

    fpath = os.path.join(output_dir, OUTPUT_FILES["correlation"])
    _write_json(data, fpath, "correlation_matrix.json")
    return fpath

# ── quiz_behavior.json ────────────────────────────────────────

def export_quiz_behavior_json(
    weeks_data: list,
    course_meta: dict,
    output_dir: str = OUTPUT_DIR,
    generated_at: str = "",
) -> str:
    """輸出 quiz_behavior.json（各週題庫行為）。
    by_semester：從 weeks[].segments 重組，供前端年度篩選直接讀取，無需前端重算。
    結構：{ "YYYS": { "weeks": [ {week, title, avg_attempts, active_students, overall_pass_rate}, ... ] } }
    """
    by_semester: dict = {}
    for week in weeks_data:
        if week.get("is_exam_week"):
            continue
        for seg_key, seg_val in week.get("segments", {}).items():
            # [Fix] 防禦 seg_key 格式異常（不足 3 段），避免 unpack ValueError
            parts = seg_key.split("|", 2)
            if len(parts) != 3:
                continue
            sem, cluster_key, pass_key = parts
            if not sem or sem == "all" or cluster_key != "all" or pass_key != "all":
                continue
            if sem not in by_semester:
                by_semester[sem] = {"weeks": []}
            by_semester[sem]["weeks"].append({
                "week":              seg_val.get("week"),
                "title":             seg_val.get("title"),
                "avg_attempts":      seg_val.get("avg_attempts", 0.0),
                "active_students":   seg_val.get("active_students", 0),
                "overall_pass_rate": seg_val.get("overall_pass_rate", 0.0),
            })

    data = {
        "meta": {
            "schema_version": SCHEMA_VERSION_CORRELATION,
            **_course_meta_fields(course_meta),
            "semesters":      course_meta.get("semesters", []),
            "generated_at":   generated_at or datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "total_weeks":    len(weeks_data),
        },
        "weeks":       weeks_data,
        "by_semester": by_semester,
    }
    fpath = os.path.join(output_dir, OUTPUT_FILES["quiz_behavior"])
    _write_json(data, fpath, "quiz_behavior.json")
    return fpath

# ── time_distribution 工具函式（模組層級）────────────────────

def _td_add_vectors(a: list, b: list) -> list:
    """截斷至 24，防止上游資料汙染。"""
    a, b = a[:24], b[:24]
    n = max(len(a), len(b))
    return [round((a[i] if i < len(a) else 0.0) + (b[i] if i < len(b) else 0.0), 4)
            for i in range(n)]

def _td_normalize_vector(v: list) -> list:
    total = sum(v)
    return [round(x / total, 6) for x in v] if total > 0 else [0.0] * len(v)

def _td_add_heatmap_raw(acc: dict, delta: dict) -> None:
    """In-place 累加，避免每次建立新 dict。"""
    for key, val in delta.items():
        acc[key] = round(acc.get(key, 0.0) + val, 2)

def _td_normalize_heatmap(raw: dict) -> dict:
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {k: round(v / total, 6) for k, v in raw.items()}

def _td_get_score(s: dict):
    # BUG-SCORE-DRIFT-0915 (穿透式審查)：與 05_time_analysis.py::_get_score() 屬
    # 同一段邏輯的兩份獨立維護副本，本副本漏了 midterm_score 備援（進行中學期
    # 暫時辨別依據）與 np.isfinite() 防護（pandas 合併常見的 float('nan') 誤判，
    # 同類問題已在 v7.15.23 修過一次，此處是漏改的第二處）。實測 1142（現行進行中
    # 學期）250 名學生：final_score/semester_score 皆為 None、僅 midterm_score
    # 有值，修正前本函式對全部 250 人回傳 None，與 _get_score() 回傳 60.0 等實際
    # 期中成績不一致，導致 time_distribution.json 內 cohort_hourly／pass_count／
    # fail_count 悄悄漏排這些學生，但 segment_stats（走 _get_score()）卻正常納入。
    # 修正：邏輯與 05_time_analysis.py::_get_score() 對齊（3 層 fallback + isfinite）。
    for key in ("final_score", "semester_score", "midterm_score"):
        v = s.get(key)
        if v is not None:
            try:
                f = float(v)
                return f if np.isfinite(f) else None
            except (TypeError, ValueError):
                pass
    return None

def _td_make_hourly(acc: list) -> dict:
    """24-element hourly accumulator → {raw, normalized} dict。無 closure，模組層級。"""
    return {"raw": [round(x, 2) for x in acc], "normalized": _td_normalize_vector(acc)}

def _td_student_row(s: dict) -> dict:
    """每位學生的 time_distribution 輸出列。無 closure，模組層級定義。"""
    tp = s.get("time_profile", {})
    return {
        "anon_id":                 s.get("anon_id"),
        "masked_id":               s.get("masked_id"),
        "cluster":                 s.get("cluster", ""),
        "edu_type":                s.get("edu_type", ""),
        # course_track（見規劃書1.1節／討論1）：tab-behavior-time.js的
        # _filterCourseTrack篩選維度依此欄判斷。None（一般課程）時輸出
        # null，前端已有處理null的既有慣例（比照edu_type的空字串處理）。
        "course_track":            s.get("course_track"),
        "active_weeks":            tp.get("active_weeks", 0),
        "zero_week_count":         tp.get("zero_week_count", 0),
        "first_login_week":        tp.get("first_login_week"),
        "consistency_score":       tp.get("consistency_score", 0),
        "cram_pattern_score":      tp.get("cram_pattern_score", 0),
        "total_learning_minutes":  (s.get("features", {}) or {}).get("total_learning_minutes", 0),
        "pre_midterm_7d_minutes":  tp.get("pre_midterm_7d_minutes", 0),
        "pre_final_7d_minutes":    tp.get("pre_final_7d_minutes", 0),
        "midterm_regular_minutes": tp.get("midterm_regular_minutes", 0),
        "midterm_period_minutes":  tp.get("midterm_period_minutes", 0),
        "final_regular_minutes":   tp.get("final_regular_minutes", 0),
        "final_period_minutes":    tp.get("final_period_minutes", 0),
        "semester":                s.get("semester", ""),
        "final_score":             s.get("final_score"),
        "semester_score":          s.get("semester_score"),
        "time_slot_distribution":  tp.get("time_slot_distribution", {}),
        "dominant_time_slot":      tp.get("dominant_time_slot", ""),
        "late_night_ratio":        tp.get("late_night_ratio", 0),
        # v2.1 新增欄位
        "heatmap_matrix":      tp.get("heatmap_matrix") or {"raw": {}, "normalized": {}},
        "hourly_distribution": tp.get("hourly_distribution") or {"raw": [0.0] * 24, "normalized": [0.0] * 24},
    }

# ── time_distribution.json (v2.1) ─────────────────────────────

def export_time_distribution_json(
    students: list,
    course_meta: dict,
    exam_dates: dict,
    output_dir: str = OUTPUT_DIR,
    generated_at: str = "",
    segment_stats: dict = None,
) -> str:
    """輸出 time_distribution.json（v2.1）。

    v2.1 新增輸出結構（ETL 預聚合，前端零運算）：
      class_avg_hourly_distribution : {"raw": [...24], "normalized": [...24]}
      class_heatmap_matrix          : {"raw": {...}, "normalized": {...}}
      cohort_hourly                 : {
          "pass":  {"raw": [...], "normalized": [...]},
          "fail":  {"raw": [...], "normalized": [...]},
          "top25": {"raw": [...], "normalized": [...]},
      }
      meta.cohort_counts            : {pass, fail, top25 人數}
      students[i] 新增：
          heatmap_matrix      : {"raw": {...}, "normalized": {...}}
          hourly_distribution : {"raw": [...], "normalized": [...]}

    注意：exam_dates 傳入的是全域彙整值（最早期中/最晚期末），
    僅作為 meta 的參考資訊。每位學生的實際考試日期已在
    lms_etl.py 的 per_student_exam_dates 中各自處理。
    分群依據：final_score 優先，fallback semester_score。
    """
    slot_keys = ["MORNING", "AFTERNOON", "EVENING", "LATE_NIGHT"]

    # 防禦性過濾：確保 practicum 學生不進入時間分析輸出
    prac_filtered = [s for s in students if s.get("edu_type") == "practicum"]
    if prac_filtered:
        logger.warning("[07] export_time_distribution_json：偵測到 %d 名 edu_type='practicum'，已從輸出中移除", len(prac_filtered))
        students = [s for s in students if s.get("edu_type") != "practicum"]

    # ── 單次遍歷：同時完成所有聚合計算 ──────────────────────────
    # 合併原本 11 次獨立遍歷（scored_valid/pass_group/fail_group/slot_sums/
    # class_hourly/heatmap/student_rows）為一次 O(N) 迴圈。
    slot_sums          = {k: 0.0 for k in slot_keys}
    class_hourly_acc   = [0.0] * 24
    class_heatmap_raw: dict = {}
    scored_list: list  = []   # [(score, student_row_dict)]
    student_time_rows: list = []
    slot_count = 0
    # cohort hourly accumulators（在主迴圈直接累積，避免後置 3 次再遍歷）
    _pass_hourly: list = [0.0] * 24
    _fail_hourly: list = [0.0] * 24
    pass_count = 0
    fail_count = 0

    for s in students:
        tp   = s.get("time_profile", {})
        row  = _td_student_row(s)
        student_time_rows.append(row)

        # slot 分布
        if tp:
            slot_count += 1
            dist = tp.get("time_slot_distribution", {})
            for k in slot_keys:
                slot_sums[k] += dist.get(k, 0)

        # hourly 向量
        raw_h = tp.get("hourly_distribution", {}).get("raw", [])
        if raw_h:
            class_hourly_acc = _td_add_vectors(class_hourly_acc, raw_h)

        # heatmap
        raw_hm = tp.get("heatmap_matrix", {}).get("raw", {})
        if raw_hm:
            _td_add_heatmap_raw(class_heatmap_raw, raw_hm)

        # 分群基礎 + cohort hourly 同步累積
        sc = _td_get_score(s)
        if sc is not None:
            scored_list.append((sc, s))
            if raw_h:
                if sc >= 60:
                    _pass_hourly = _td_add_vectors(_pass_hourly, raw_h)
                    pass_count += 1
                else:
                    _fail_hourly = _td_add_vectors(_fail_hourly, raw_h)
                    fail_count += 1
            elif sc >= 60:
                pass_count += 1
            else:
                fail_count += 1

    # ── 從 scored_list 建立分群 ──────────────────────────────────
    if scored_list:
        sorted_desc = sorted(scored_list, key=lambda x: x[0], reverse=True)
        top_n       = max(1, len(sorted_desc) // 4)
        top25_group = [s for _, s in sorted_desc[:top_n]]
    else:
        top25_group = []

    # top25 hourly 仍需一次遍歷（top25 在排序後才能確定，無法在主迴圈預知）
    _top25_hourly = [0.0] * 24
    for s in top25_group:
        raw = s.get("time_profile", {}).get("hourly_distribution", {}).get("raw", [])
        if raw:
            _top25_hourly = _td_add_vectors(_top25_hourly, raw)

    # ── 彙整聚合結果 ─────────────────────────────────────────────
    class_avg_dist = (
        {k: round(v / slot_count, 4) for k, v in slot_sums.items()}
        if slot_count > 0
        else {k: 0.0 for k in slot_keys}
    )
    class_avg_hourly = _td_make_hourly(class_hourly_acc)
    class_heatmap = {
        "raw":        class_heatmap_raw,
        "normalized": _td_normalize_heatmap(class_heatmap_raw),
    }
    cohort_hourly = {
        "pass":  _td_make_hourly(_pass_hourly),
        "fail":  _td_make_hourly(_fail_hourly),
        "top25": _td_make_hourly(_top25_hourly),
    }
    data = {
        "meta": {
            "schema_version":     SCHEMA_VERSION_TIME_DIST,
            **_course_meta_fields(course_meta),
            "semesters":          course_meta.get("semesters", []),
            "generated_at":       generated_at or datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "student_count":      len(student_time_rows),
            "midterm_date":       exam_dates.get("midterm", ""),
            "final_date":         exam_dates.get("final", ""),
            "exam_date_note":     "per-student dates are applied individually during ETL",
            "cohort_counts": {
                "pass":  pass_count,
                "fail":  fail_count,
                "top25": len(top25_group),
            },
        },
        "class_avg_time_distribution":    class_avg_dist,
        "class_avg_hourly_distribution":  class_avg_hourly,
        "class_heatmap_matrix":           class_heatmap,
        "cohort_hourly":                  cohort_hourly,
        "segment_stats":                  segment_stats or {},
        "students":                       student_time_rows,
    }

    fpath = os.path.join(output_dir, OUTPUT_FILES["time_distribution"])
    _write_json(data, fpath, "time_distribution.json")
    return fpath

def export_cross_analysis_json(
    cross_analysis_data: dict,
    output_dir: str = OUTPUT_DIR,
    generated_at: str | None = None,
) -> str | None:
    """輸出 cross_analysis.json（11_cross_analysis.py 的計算結果）。

    cross_analysis_data 可能含內部欄位（以 `_` 開頭，例如
    `_records_with_final`，僅供 12_early_warning.py 計算 z-score
    標準化參數），輸出前一律剔除。

    若 cross_analysis_data 為空 dict 或 None，視為該次 ETL 未產出
    跨模組分析（例如訓練集為空），不寫檔並回傳 None。
    """
    if not cross_analysis_data:
        logger.info("[07] cross_analysis_data 為空，跳過 cross_analysis.json 輸出")
        return None

    data = {k: v for k, v in cross_analysis_data.items() if not k.startswith("_")}

    # xgb_validation 巢狀在 cross_analysis_data 內，需額外剔除其內部的 _model_artifacts
    # （_model_artifacts 是 sklearn model 物件，不可 JSON 序列化）
    if "xgb_validation" in data and isinstance(data["xgb_validation"], dict):
        data["xgb_validation"] = {
            k: v for k, v in data["xgb_validation"].items() if not k.startswith("_")
        }

    # 補上統一的 generated_at — 用 dict spread 建新 dict 取代 setdefault，
    # 避免 setdefault 靜默 mutate 呼叫端的 cross_analysis_data["meta"] 物件。
    if generated_at and isinstance(data.get("meta"), dict) and "generated_at" not in data["meta"]:
        data["meta"] = {**data["meta"], "generated_at": generated_at}

    fpath = os.path.join(output_dir, OUTPUT_FILES["cross_analysis"])
    _write_json(data, fpath, "cross_analysis.json")
    return fpath

def export_xgb_validation_json(
    cross_analysis_data: dict,
    output_dir: str = OUTPUT_DIR,
    generated_at: str | None = None,
) -> str | None:
    """輸出 xgb_validation.json（11_cross_analysis.py::compute_xgb() 的驗證結果）。

    xgb_validation 為 None（--disable-xgb）或 status 為
    insufficient_data/missing_dependency 時不寫檔，回傳 None。
    """
    xgb = (cross_analysis_data or {}).get("xgb_validation")
    if not xgb or xgb.get("status") in ("insufficient_data", "missing_dependency"):
        logger.info("[07] xgb_validation 不存在/樣本不足，跳過輸出")
        return None
    data = {k: v for k, v in xgb.items() if not k.startswith("_")}
    if generated_at:
        data["generated_at"] = generated_at
    fpath = os.path.join(output_dir, OUTPUT_FILES["xgb_validation"])
    _write_json(data, fpath, "xgb_validation.json")
    return fpath

def export_xgb_feature_importance_json(
    cross_analysis_data: dict,
    output_dir: str = OUTPUT_DIR,
    generated_at: str | None = None,
) -> str | None:
    """輸出 xgb_feature_importance.json（11_cross_analysis.py 的 feature_importance）。

    feature_importance 為 list[dict]（每項含 feature/importance/rank）。
    輸出格式：{"features": [...], "generated_at": "..."}
    """
    fi = (cross_analysis_data or {}).get("feature_importance")
    if not fi:
        logger.info("[07] feature_importance 不存在，跳過輸出")
        return None
    # fi 是 list[dict]，不可用 dict(fi)（會 TypeError）
    data: dict = {"features": fi}
    if generated_at:
        data["generated_at"] = generated_at
    fpath = os.path.join(output_dir, OUTPUT_FILES["xgb_feature_importance"])
    _write_json(data, fpath, "xgb_feature_importance.json")
    return fpath

def export_warning_json(
    warning_data: dict,
    output_dir: str = OUTPUT_DIR,
    generated_at: str | None = None,
) -> str | None:
    """輸出 warning_{semester}.json（12_early_warning.py 的計算結果）。

    檔名中的 {semester} 取自 warning_data['meta']['target_semester']。
    若 warning_data 為空 dict 或 None，或缺少 target_semester，
    不寫檔並回傳 None。
    """
    if not warning_data:
        logger.info("[07] warning_data 為空，跳過 warning_*.json 輸出")
        return None

    meta = warning_data.get("meta", {})
    target_semester = meta.get("target_semester")
    if not target_semester:
        logger.warning("[07] warning_data 缺少 meta.target_semester，跳過輸出")
        return None

    data = {
        **warning_data,
        "meta": {**meta, **({"generated_at": generated_at} if generated_at and "generated_at" not in meta else {})},
    }

    filename = OUTPUT_FILES["warning_template"].format(semester=target_semester)
    fpath = os.path.join(output_dir, filename)
    _write_json(data, fpath, filename)
    return fpath

# ── 主輸出函式（整合呼叫）────────────────────────────────────

def export_all(
    students: list,
    radar_data: dict,
    pearson_matrix: dict,
    scatter_data: list,
    weeks_data: list,
    feature_cols: list,
    target_cols: list,
    course_meta: dict,
    material_counts: dict,
    exam_dates: dict,
    output_dir: str = OUTPUT_DIR,
    subgroup_pearson: dict = None,
    correlation_insights: dict = None,
    outlier_thresholds: dict = None,
    lagged_pearson: dict = None,
    segment_stats: dict = None,
    segment_pearson: dict = None,
    lsa_transition: dict = None,
    incomplete_semesters: list = None,
    incomplete_pearson: dict = None,
    cross_analysis_data: dict = None,
    warning_data: dict = None,
    excluded_material_detail: list = None,
) -> None:
    """一次輸出所有 JSON 檔案（不含 at_risk，需另行呼叫 export_at_risk_json）。
    統一 generated_at 時間戳，確保所有檔案一致。
    """
    generated_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    logger.info("[07] 開始輸出 JSON → %s/  (generated_at=%s)", output_dir, generated_at)

    export_behavior_json(students, course_meta, material_counts, output_dir, generated_at)
    export_radar_chart_json(radar_data, course_meta, output_dir, generated_at)
    export_correlation_json(
        pearson_matrix, scatter_data, feature_cols,
        target_cols, course_meta, output_dir,
        subgroup_pearson=subgroup_pearson,
        correlation_insights=correlation_insights,
        outlier_thresholds=outlier_thresholds,
        lagged_pearson=lagged_pearson,
        segment_pearson=segment_pearson,
        lsa_transition=lsa_transition,
        incomplete_semesters=incomplete_semesters,
        incomplete_pearson=incomplete_pearson,
        excluded_material_detail=excluded_material_detail,
        generated_at=generated_at,
    )
    export_quiz_behavior_json(weeks_data, course_meta, output_dir, generated_at)
    export_time_distribution_json(
        students, course_meta, exam_dates, output_dir, generated_at,
        segment_stats=segment_stats,
    )
    export_cross_analysis_json(cross_analysis_data, output_dir, generated_at)
    export_xgb_validation_json(cross_analysis_data, output_dir, generated_at)
    export_xgb_feature_importance_json(cross_analysis_data, output_dir, generated_at)
    export_warning_json(warning_data, output_dir, generated_at)
    logger.info("[07] 全部輸出完成")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("export_json 模組載入成功")
    logger.info("預設輸出目錄：%s", OUTPUT_DIR)
    logger.info("輸出檔案清單：%s", list(OUTPUT_FILES.values()))
