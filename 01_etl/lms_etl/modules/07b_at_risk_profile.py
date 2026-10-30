# ========================================================
# lms_etl/modules/07b_at_risk_profile.py
# Phase 4：學習落後特徵歸納與靜態報告產生器
#
# 呼叫方式（lms_etl.py Step 7 結束後）：
#   _m07b = _load_module(
#       os.path.join(_BASE, "07b_at_risk_profile.py"), "m07b"
#   )
#   _m07b.export_at_risk_profile_v2(
#       students=students,
#       course_meta=course_meta,
#       exam_dates=exam_dates,
#       output_dir=args.out,
#   )
# ========================================================

import json
import math
import os
import re
from datetime import date, datetime

import pandas as pd

SCHEMA_VERSION = "3.1"  # v7.14：新增 reading_integrity 區塊（向下相容之minor bump）

# ── 閾值設定（可依課程調整） ──────────────────────────────────
THRESHOLDS = {
    "pass_score":          60,      # semester_score 及格門檻
    "critical_gap_pct":   -65.0,    # 超過此落差 → critical
    "warning_gap_pct":    -40.0,    # 超過此落差 → warning
    "low_completion_pct":  20,      # 低完成率警示門檻（%）
    "decay_critical_pct": -35.0,    # 期中後衰退 → critical
}

# ── 雷達圖 6 維度（順序固定）→ features 欄位對應 ─────────────
# 已驗證自 04_feature_engineering.py + 03_parse_tags.py：
#   SUP 類包含：整理筆記／補充教材／動畫解析／教學動畫／動畫
#   完成率欄位：{cat_lower}_completion_rate（cat ∈ TXT/SUP/TUT）
#   時間欄位：aud_total_minutes（reading_stats sheet）
#   穩定性：consistency_score（m05 回填至 features，717 行）
RADAR_METRIC_MAP = [
    ("text_material_completion", "txt_completion_rate"),   # 維度 0：TXT
    ("supplementary_completion", "sup_completion_rate"),   # 維度 1：SUP（5 種標籤）
    ("tutoring_resource_rate",   "tut_completion_rate"),   # 維度 2：TUT
    ("pre_exam_intensity",       "pre_exam_intensity"),    # 維度 3：考前強度
    ("learning_stability",       "consistency_score"),     # 維度 4：穩定性
    ("audio_material_hours",     "aud_total_minutes"),     # 維度 5：AUD 分鐘
]


# ────────────────────────────────────────────────────────
# 分組
# ────────────────────────────────────────────────────────

def _as_valid_score(value):
    """將學期成績轉為 float；空值、NaN 與非數值一律視為未結算。"""
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        score = float(value)
    except (ValueError, TypeError):
        return None
    if pd.isna(score):
        return None
    return score


def _split_pass_fail(students: list, pass_score: int = 60):
    """依 semester_score 分割及格/不及格群組，排除無成績學生。

    排除條件（視為未結算，不計入任一群組）：
      - None：資料庫／JSON 的 null
      - 空字串 ""：Excel / CSV 匯入常見的空值型態
      - NaN：pandas 合併空成績後的缺值型態
      - 轉型失敗（非數值字串）：靜默略過並印出警告，避免整批 ETL 崩潰
    """
    pass_group, fail_group = [], []
    for s in students:
        score = _as_valid_score(s.get("semester_score"))
        # 排除 null、空字串、NaN 與非數值，視為未結算
        if score is None:
            continue
        (pass_group if score >= pass_score else fail_group).append(s)
    return pass_group, fail_group


# ────────────────────────────────────────────────────────
# 正規化
# ────────────────────────────────────────────────────────

def _minmax_normalize(values: list, all_values: list) -> float:
    """Min-Max 正規化至 0–100，以全體學生為基準"""
    if not all_values or not values:
        return 0.0
    v_min, v_max = min(all_values), max(all_values)
    if v_max == v_min:
        return 50.0  # 無變異時置中
    median_val = pd.Series(values).median()
    if pd.isna(median_val):
        return 0.0
    return round((median_val - v_min) / (v_max - v_min) * 100, 1)


# ────────────────────────────────────────────────────────
# 各節點建構函式
# ────────────────────────────────────────────────────────

def build_cohort_summary(
    pass_group: list,
    fail_group: list,
    total_enrolled: int | None = None,
) -> dict:
    """
    建立班級概況摘要。

    Parameters
    ----------
    pass_group      : 已通過 _split_pass_fail 的及格學生清單（semester_score >= 60）
    fail_group      : 已通過 _split_pass_fail 的不及格學生清單（semester_score < 60）
    total_enrolled  : 該範圍內「全體」學生人數（含 semester_score=null 的未結算學生）。
                      若為 None，退回舊行為（僅計算有成績者），適用於外部直接呼叫的相容場景。

    語義說明
    --------
    total_students  = total_enrolled（全體，含未結算）
    pass_count      = len(pass_group)       → 僅有效成績
    fail_count      = len(fail_group)       → 僅有效成績
    unsettled_count = total_students - pass_count - fail_count  → 未結算（null）人數
    fail_rate_pct   = fail_count / (pass_count + fail_count)    → 分母排除未結算，避免進行中學期拉高比率
    """
    pass_count   = len(pass_group)
    fail_count   = len(fail_group)
    graded_total = pass_count + fail_count  # 有效成績母體（分母）

    # 若未傳入 total_enrolled，退回舊行為以維持向後相容
    total_students  = total_enrolled if total_enrolled is not None else graded_total
    unsettled_count = max(total_students - graded_total, 0)  # 防呆：避免負值

    return {
        "total_students":  total_students,
        "pass_count":      pass_count,
        "fail_count":      fail_count,
        # 未結算人數：semester_score 為 null（進行中學期尚未期末考）
        "unsettled_count": unsettled_count,
        # fail_rate_pct 分母為「已結算」學生數，排除未結算干擾
        "fail_rate_pct":   round(fail_count / graded_total * 100, 1) if graded_total else 0.0,
    }


def build_metrics_comparison(pass_group: list, fail_group: list) -> dict:
    """
    計算雷達圖 6 維度的及格/不及格組中位數。
    欄位路徑：students[i].features.<field_name>
    正規化基準：全體學生（含及格+不及格），單次遍歷分別收集 all/pass/fail 數值。
    """
    result = {}

    for json_key, feat_field in RADAR_METRIC_MAP:
        all_vals, pass_vals, fail_vals = [], [], []
        for s in pass_group:
            v = float(s["features"].get(feat_field) or 0)
            all_vals.append(v)
            pass_vals.append(v)
        for s in fail_group:
            v = float(s["features"].get(feat_field) or 0)
            all_vals.append(v)
            fail_vals.append(v)

        pass_norm = _minmax_normalize(pass_vals, all_vals)
        fail_norm = _minmax_normalize(fail_vals, all_vals)
        gap       = round((fail_norm - pass_norm) / pass_norm * 100, 1) if pass_norm else 0.0

        pass_raw = round(pd.Series(pass_vals).median(), 4) if pass_vals else 0.0
        fail_raw = round(pd.Series(fail_vals).median(), 4) if fail_vals else 0.0

        result[json_key] = {
            "pass_median_normalized": pass_norm,
            "fail_median_normalized": fail_norm,
            "gap_percentage":         f"{gap:+.1f}%",
            "raw_pass_median":        pass_raw,
            "raw_fail_median":        fail_raw,
        }

    return result


def build_behavioral_markers(pass_group: list, fail_group: list, thresholds: dict) -> dict:
    """
    計算 zero_week_count 分布與低完成率紅旗指標。
    zero_week_count 來源：students[i].features.zero_week_count
    critical_weeks：不及格組各週零活動人數最高的前 2 週
    """
    fail_zero_weeks = [int(s["features"].get("zero_week_count") or 0) for s in fail_group]
    pass_zero_weeks = [int(s["features"].get("zero_week_count") or 0) for s in pass_group]

    # critical_weeks：各週零活動比率最高的週次
    week_zero_counts: dict = {}
    for s in fail_group:
        weekly = s.get("time_profile", {}).get("weekly_minutes", [])
        for week_idx, minutes in enumerate(weekly, start=1):
            if (minutes or 0) == 0:
                week_zero_counts[week_idx] = week_zero_counts.get(week_idx, 0) + 1

    critical_weeks = sorted(
        week_zero_counts,
        key=lambda w: week_zero_counts[w],
        reverse=True
    )[:2]

    # 低完成率紅旗：只對三個完成率維度（維度 0–2）計算
    completion_rate_dims = RADAR_METRIC_MAP[:3]
    low_threshold = thresholds["low_completion_pct"] / 100
    n_fail = len(fail_group) or 1
    n_pass = len(pass_group) or 1
    low_flags = []
    for json_key, feat_field in completion_rate_dims:
        fail_low = sum(1 for s in fail_group if (s["features"].get(feat_field) or 0) < low_threshold)
        pass_low = sum(1 for s in pass_group if (s["features"].get(feat_field) or 0) < low_threshold)
        if fail_low / n_fail >= 0.40:  # 不及格組 40% 以上才列入
            low_flags.append({
                "metric":              json_key,
                "threshold_pct":       thresholds["low_completion_pct"],
                "ratio_in_fail_group": round(fail_low / n_fail, 3),
                "ratio_in_pass_group": round(pass_low / n_pass, 3),
            })

    return {
        "zero_week_distribution": {
            # mode() 有多個眾數時取最小值（index=0），語義為「最常見的最小零活動週數」
            "fail_group_mode": int(pd.Series(fail_zero_weeks).mode().iloc[0]) if fail_zero_weeks else 0,
            "pass_group_mode": int(pd.Series(pass_zero_weeks).mode().iloc[0]) if pass_zero_weeks else 0,
            "critical_weeks":  critical_weeks,
        },
        "low_completion_flags": low_flags,
    }


def build_reading_integrity_markers(pass_group: list, fail_group: list) -> dict:
    """
    閱讀誠信度紅旗（v7.14，交接文件v10第2.2節）。

    只回傳「群組層級」統計（不及格組 vs 及格組的盛行率／平均灌水倍數
    對照），不列出個別學生 anon_id——與 build_behavioral_markers() 的
    low_completion_flags 慣例一致。若要查詢個別學生是否被標記，前端
    應直接讀取 behavior.json 裡每位學生 features.policy_gaming_flag
    （該欄位已隨 v7.14 資料清理一併輸出，屬全量欄位，非本函式範圍）。

    設計依據（交接文件v10第0.5節，倫理提醒見v7第10.7節）：本區塊定位
    為「提供客觀數據供人工複核」，不直接下判斷（如不使用「作弊」
    「造假」等定性字眼），避免單一指標誤傷少數合理情境（如受訪問題
    的視障學生使用螢幕報讀器導致單次時長異常偏長）。
    """
    def _group_stats(group: list) -> dict:
        n = len(group) or 1
        flags = [bool(s["features"].get("policy_gaming_flag")) for s in group]
        ratios = [
            float(s["features"].get("reading_inflation_ratio") or 1.0)
            for s in group
        ]
        return {
            "n_total":            len(group),
            "n_flagged":          sum(flags),
            "flagged_pct":        round(sum(flags) / n, 3),
            "avg_inflation_ratio": round(sum(ratios) / len(ratios), 3) if ratios else 1.0,
        }

    return {
        "fail_group": _group_stats(fail_group),
        "pass_group": _group_stats(pass_group),
    }


def build_temporal_decay(
    pass_group: list,
    fail_group: list,
    exam_dates: dict,
    is_aggregate: bool = False,  # v7.15.6 問題B修正：True 時 midterm 相關欄位設為 null
) -> dict:
    """
    計算時序性衰退指標。
    資料來源：students[i].time_profile.weekly_minutes（list[float]，index 0 = Week 1）

    exam_dates 需含欄位：semester_start, midterm, total_weeks
    midterm_week 由程式自動計算（exam_dates.json 無此欄位）：
      midterm_week = (midterm_date - semester_start_date).days // 7 + 1
      114-1 實例：(2025-11-13 - 2025-09-08) = 66 天 → Week 10

    FIX：移除重複的 per_student_exam_dates/semester_code 解析邏輯。
    上游 _build_profile_for_group 已透過 _resolve_semester_exam_dates()
    處理學期日期解析，此函式只需接收已解析完成的 exam_dates。
    """
    # ── 前置欄位存在性檢查 ────────────────────────────────────
    required_keys = ("semester_start", "midterm", "total_weeks")
    missing = [k for k in required_keys if not exam_dates.get(k)]
    if missing:
        print(f"  ⚠️  exam_dates 缺少欄位 {missing}，temporal_decay 將 fallback 至 Week 10")

    all_students = pass_group + fail_group
    has_weekly = any(
        bool(s.get("time_profile", {}).get("weekly_minutes"))
        for s in all_students
    )
    if not has_weekly:
        return {"available": False}

    # ── 計算期中考週次 ────────────────────────────────────────
    midterm_week_num = 10  # fallback（114-1 實際值）
    try:
        sem_start        = date.fromisoformat(exam_dates["semester_start"])
        midterm_dt       = date.fromisoformat(exam_dates["midterm"])
        midterm_week_num = max(1, (midterm_dt - sem_start).days // 7 + 1)
    except (KeyError, ValueError):
        pass

    midterm_label = f"Week {midterm_week_num}"

    def _group_weekly_series(group: list) -> dict:
        """各週學習分鐘的群組平均值（key 為 "Week N"）"""
        all_weeks: dict = {}
        for s in group:
            for week_idx, mins in enumerate(
                s.get("time_profile", {}).get("weekly_minutes", []), start=1
            ):
                wk = f"Week {week_idx}"
                all_weeks.setdefault(wk, []).append(float(mins or 0))
        return {wk: round(sum(vals) / len(vals), 1) for wk, vals in all_weeks.items()}

    fail_series = _group_weekly_series(fail_group)
    pass_series = _group_weekly_series(pass_group)

    def _decay_rate(series: dict, midterm_lbl: str) -> float:
        """期中考後均值相對期中考前均值的變化率（%）"""
        weeks   = sorted(series.keys(), key=lambda w: int(w.replace("Week ", "")))
        mid_idx = weeks.index(midterm_lbl) if midterm_lbl in weeks else len(weeks) // 2
        pre     = [series[w] for w in weeks[:mid_idx]]
        post    = [series[w] for w in weeks[mid_idx + 1:]]
        pre_avg  = sum(pre)  / len(pre)  if pre  else 0
        post_avg = sum(post) / len(post) if post else 0
        if pre_avg == 0:
            return 0.0
        return round((post_avg - pre_avg) / pre_avg * 100, 1)

    def _consecutive_zero_median(group: list) -> float:
        """各學生連續零活動週數的群組中位數"""
        czw_list = []
        for s in group:
            weekly = s.get("time_profile", {}).get("weekly_minutes", [])
            max_run = cur_run = 0
            for mins in weekly:
                if (mins or 0) == 0:
                    cur_run += 1
                    max_run  = max(max_run, cur_run)
                else:
                    cur_run = 0
            czw_list.append(max_run)
        return round(pd.Series(czw_list).median(), 1) if czw_list else 0.0

    # v7.15.6 問題B修正：跨學期聚合節點（is_aggregate=True）的 midterm_week_num
    # 是「隨機借用某一個學期曆法」算出的，對混合多學期樣本無明確語意。
    # 採方案B：僅將 midterm_week_num / post_midterm_decay_rate 設為 null，
    # consecutive_zero_weeks 與 weekly_activity_series 不依賴期中考日期，仍正常輸出。
    if is_aggregate:
        return {
            "available":        True,
            "midterm_week_num": None,          # 跨學期無明確定義
            "consecutive_zero_weeks": {
                "fail_group_median": _consecutive_zero_median(fail_group),
                "pass_group_median": _consecutive_zero_median(pass_group),
            },
            "post_midterm_decay_rate": None,   # 跨學期無明確定義，前端 optional-chain 可安全消費
            "weekly_activity_series": {
                "fail_group": fail_series,
                "pass_group": pass_series,
            },
        }

    return {
        "available":        True,
        "midterm_week_num": midterm_week_num,
        "consecutive_zero_weeks": {
            "fail_group_median": _consecutive_zero_median(fail_group),
            "pass_group_median": _consecutive_zero_median(pass_group),
        },
        "post_midterm_decay_rate": {
            "fail_group_median_pct": _decay_rate(fail_series, midterm_label),
            "pass_group_median_pct": _decay_rate(pass_series, midterm_label),
        },
        "weekly_activity_series": {
            "fail_group": fail_series,
            "pass_group": pass_series,
        },
    }


def build_prescriptive_summary(profile: dict, thresholds: dict) -> list:
    """
    依閾值規則自動產生處方性建議陣列。
    規則 1：輔導資源低完成率（TUT）
    規則 2：期中後衰退率過高
    規則 3：門檻解鎖教材（SUP）完成率落差
    規則 4：連續零活動週數 >= 2
    """
    summaries = []
    mc = profile.get("metrics_comparison", {})
    td = profile.get("temporal_decay", {})
    bm = profile.get("behavioral_markers", {})

    # ── 規則 1：輔導資源低完成率 ──
    for flag in bm.get("low_completion_flags", []):
        if flag["metric"] == "tutoring_resource_rate":
            ratio_pct = flag["ratio_in_fail_group"] * 100
            summaries.append({
                "severity":          "critical",
                "metric":            "tutoring_resource_rate",
                "trigger_condition": "ratio_in_fail_group >= 0.40",
                "finding":           f"{ratio_pct:.0f}% 的不及格學生輔導資源完成率低於 {flag['threshold_pct']}%，遠高於及格組的 {flag['ratio_in_pass_group']*100:.0f}%。",
                "action":            "建議下學期開學第一週即宣導輔導資源使用方式，並於第 4 週前設定強制完成里程碑。",
            })

    # ── 規則 2：期中後衰退 ──
    # v7.15.6 問題B修正：is_aggregate 時 post_midterm_decay_rate 為 None，需先守衛
    if td.get("available") and td.get("post_midterm_decay_rate") is not None:
        fail_decay = td.get("post_midterm_decay_rate", {}).get("fail_group_median_pct", 0)
        pass_decay = td.get("post_midterm_decay_rate", {}).get("pass_group_median_pct", 0)
        if fail_decay <= thresholds["decay_critical_pct"]:
            summaries.append({
                "severity":          "critical",
                "metric":            "post_midterm_decay_rate",
                "trigger_condition": f"fail_group_median_pct <= {thresholds['decay_critical_pct']}",
                "finding":           f"不及格學生期中考後學習活動量衰退 {abs(fail_decay):.1f}%，及格組僅衰退 {abs(pass_decay):.1f}%。",
                "action":            "建議期中考成績公布後一週內發送個別學習提醒，對分數低於 60 分學生啟動一對一輔導追蹤。",
            })

    # ── 規則 3：門檻解鎖教材（SUP）完成率落差 ──
    sup_gap_str = mc.get("supplementary_completion", {}).get("gap_percentage", "0%")
    try:
        sup_gap_raw = float(sup_gap_str.replace("%", "").replace("+", ""))
        sup_gap = 0.0 if math.isnan(sup_gap_raw) else sup_gap_raw
    except (ValueError, AttributeError):
        sup_gap = 0.0
    if sup_gap <= thresholds["warning_gap_pct"]:
        severity = "critical" if sup_gap <= thresholds["critical_gap_pct"] else "warning"
        summaries.append({
            "severity":          severity,
            "metric":            "supplementary_completion",
            "trigger_condition": f"gap_percentage <= {thresholds['warning_gap_pct']}%",
            "finding":           f"不及格群體門檻解鎖教材（整理筆記/補充教材/動畫）完成率比及格組低 {abs(sup_gap):.1f}%，顯示知識內化過程可能提早中斷。",
            "action":            "建議將 SUP 類教材設計為分段式（每 2 週一個里程碑），並在完成率低於 30% 時觸發自動提醒。",
        })

    # ── 規則 4：連續零活動週數 ──
    czw = td.get("consecutive_zero_weeks", {}) if td.get("available") else {}
    if czw.get("fail_group_median", 0) >= 2:
        summaries.append({
            "severity":          "warning",
            "metric":            "consecutive_zero_weeks",
            "trigger_condition": "fail_group_median >= 2",
            "finding":           f"不及格學生連續零活動週數中位數達 {czw['fail_group_median']:.1f} 週，及格組為 {czw.get('pass_group_median', 0):.1f} 週。",
            "action":            "建議設定「超過 2 週無學習活動」自動通知機制，並於關鍵週（如期中考後第 1 週）由教師主動聯繫學生。",
        })

    # 依嚴重性排序：critical → warning → info
    order = {"critical": 0, "warning": 1, "info": 2}
    return sorted(summaries, key=lambda x: order.get(x["severity"], 9))


# ────────────────────────────────────────────────────────
# 工具：從 per_student_exam_dates 取同學期代表性 exam_dates
# ────────────────────────────────────────────────────────

def _resolve_semester_exam_dates(
    group: list,
    per_student_exam_dates: dict,
    global_exam_dates: dict,
) -> dict:
    """
    從 per_student_exam_dates 中找同學期學生的代表性考試日期。
    取第一筆同時含 semester_start + midterm 的紀錄；
    若都找不到則 fallback 至 global_exam_dates。

    v6修正：per_student_exam_dates 已改為 (anon_id, semester) 複合鍵
    （lms_etl.py Phase 1），原本 per_student_exam_dates.get(anon_id, {})
    的單鍵查詢會全數查空（永遠 fallback 至 global_exam_dates，等同於
    Bug A 的「全域殘留值」問題原地重演）。改用每位學生自己的 s["semester"]
    欄（Phase 3 已保證每筆 student record 都有正確的所屬學期）組成
    複合鍵查詢。
    """
    for s in group:
        anon_id = str(s.get("anon_id") or s.get("masked_id", ""))
        sem     = str(s.get("semester", ""))
        candidate = per_student_exam_dates.get((anon_id, sem), {})
        if candidate.get("midterm") and candidate.get("semester_start"):
            return dict(candidate)
    return dict(global_exam_dates)


def _build_profile_for_group(
    *,
    semester_label: str,
    semester_code: str,  # v7.8：改為有實際用途，見下方 by_semester 查表
    pass_group: list,
    fail_group: list,
    total_enrolled: int,  # 全體學生人數（含未結算），由呼叫方計算後傳入
    course_meta: dict,
    thresholds: dict,
    exam_dates: dict,
    per_student_exam_dates: dict,
    is_aggregate: bool = False,  # v7.15.6 問題B修正：True 時跳過跨學期無效的 midterm 計算
) -> dict:
    """Build one at-risk profile block for either a single semester or all semesters."""
    sem_exam_dates = _resolve_semester_exam_dates(
        pass_group + fail_group, per_student_exam_dates, exam_dates
    )

    # BUG-FIX (2026-07-12)：course_id/course_name 原本恆用 course_meta 的
    # 單一 flat 值，導致 by_semester 的 8 個學期節點全部套用同一組值（若不同
    # 學期實際對應不同課程代碼/合併班課程名稱，會產生錯誤資訊）。改為優先
    # 查 course_meta["by_semester"][semester_code]（lms_etl.py run() 新增，
    # 逐學期彙整自 exam_dates.json config 或 LMS 實測值），查無（例如
    # semester_code="all" 的全學期聚合節點）才退回 flat 值代表「主學期」。
    _sem_course = course_meta.get("by_semester", {}).get(semester_code, {})
    profile = {
        "semester":           semester_label,
        "course_id":          _sem_course.get("course_id")   or course_meta.get("course_id", ""),
        "course_name":        _sem_course.get("course_name") or course_meta.get("course_name", ""),
        "generated_at":       datetime.now().isoformat(),
        # 傳入 total_enrolled：讓 cohort_summary 的 total_students 含未結算人數
        "cohort_summary":     build_cohort_summary(pass_group, fail_group, total_enrolled),
        "metrics_comparison": build_metrics_comparison(pass_group, fail_group),
        "behavioral_markers": build_behavioral_markers(pass_group, fail_group, thresholds),
        "reading_integrity":  build_reading_integrity_markers(pass_group, fail_group),
        "temporal_decay":     build_temporal_decay(
            pass_group,
            fail_group,
            exam_dates=sem_exam_dates,
            is_aggregate=is_aggregate,  # v7.15.6 問題B修正
        ),
    }
    profile["prescriptive_summary"] = build_prescriptive_summary(profile, thresholds)
    return profile


# ────────────────────────────────────────────────────────
# 學期格式工具（模組級，供函式共用）
# ────────────────────────────────────────────────────────

def _fmt(sem: str) -> str:
    """將學期代碼統一轉為顯示格式（1141 / 114-1 → 114(1)）"""
    m = re.match(r"^(\d{3})-?([12])$", sem.strip())
    return f"{m.group(1)}({m.group(2)})" if m else sem


def _sem_sort_key(label: str) -> str:
    """將 114(1) 轉為可排序字串 1141"""
    m = re.match(r"(\d+)\(([12])\)", label)
    return f"{m.group(1)}{m.group(2)}" if m else label


# ────────────────────────────────────────────────────────
# 主要輸出函式
# ────────────────────────────────────────────────────────

def export_at_risk_profile_v2(
    students: list,
    course_meta: dict,
    exam_dates: dict,
    output_dir: str = "docs/data",
    thresholds: dict = None,
    per_student_exam_dates: dict = None,
) -> None:
    """
    主要輸出函式，在 lms_etl.py Step 7 結尾呼叫。

    Schema 3.0：依 students[i]["semester"] 分學期計算，
    產出多學期結構的 at_risk_profile.json。
    若輸出檔案已存在（schema 3.0），讀取後累積更新 by_semester，
    保留其他學期歷史資料。

    Parameters
    ----------
    students               : lms_etl.py run() 中的 students 陣列
    course_meta            : { semester, semesters, course_id, course_name, ... }
    exam_dates             : 全域考試日期 dict（分學期 fallback 用）
    output_dir             : 輸出目錄（預設 docs/data）
    thresholds             : 可覆寫預設閾值（None 則使用 THRESHOLDS）
    per_student_exam_dates : { (anon_id, semester): { midterm, final, semester_start, ... } }
                              （v6：複合鍵，見 lms_etl.py Phase 1）
                             由 lms_etl.py 傳入，供分學期精準推算 midterm_week。
    """
    t    = thresholds or THRESHOLDS
    _pse = per_student_exam_dates or {}

    # ── 依 semester 欄位將 students 分組 ────────────────────────
    sem_groups: dict = {}
    for s in students:
        sem = str(s.get("semester") or course_meta.get("semester", "unknown"))
        sem_groups.setdefault(sem, []).append(s)

    if not sem_groups:
        print("  ℹ️  無有效學生資料，略過 at_risk_profile_v2 產生")
        return

    # ── 讀取既有 JSON（若存在且為 3.0 schema，累積更新）──────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "at_risk_profile.json")

    existing_by_semester: dict = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if str(existing.get("schema_version", "")).startswith("3"):
                existing_by_semester = existing.get("by_semester", {})
                print(f"  [INFO] 讀取既有 at_risk_profile.json（含學期：{list(existing_by_semester.keys())}）")
            else:
                print(f"  [WARN] 既有 at_risk_profile.json schema={existing.get('schema_version')}，將升級為 3.0（舊資料捨棄）")
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [WARN] 既有 at_risk_profile.json 讀取失敗，將重新產生：{e}")

    # ── 對每個學期各自計算 ───────────────────────────────────────
    updated_sems = []
    for sem_code, sem_students in sorted(sem_groups.items()):
        sem_label  = _fmt(sem_code)
        pass_group, fail_group = _split_pass_fail(sem_students, t["pass_score"])

        sem_total_enrolled = len(sem_students)
        sem_unsettled      = sem_total_enrolled - len(pass_group) - len(fail_group)
        print(f"\n  [Phase 4 · {sem_label}] 及格 {len(pass_group)} 人，不及格 {len(fail_group)} 人"
              + (f"，未結算 {sem_unsettled} 人" if sem_unsettled > 0 else ""))

        sem_profile = _build_profile_for_group(
            semester_label=sem_label,
            semester_code=sem_code,
            pass_group=pass_group,
            fail_group=fail_group,
            total_enrolled=sem_total_enrolled,  # 傳入全體人數
            course_meta=course_meta,
            thresholds=t,
            exam_dates=exam_dates,
            per_student_exam_dates=_pse,
        )

        existing_by_semester[sem_label] = sem_profile
        updated_sems.append(sem_label)

        n_critical = sum(1 for s in sem_profile["prescriptive_summary"] if s["severity"] == "critical")
        print(f"     高風險警示：{n_critical} 項 critical / {len(sem_profile['prescriptive_summary'])} 項總計")

    if not updated_sems and not existing_by_semester:
        print("  ℹ️  所有學期均無不及格學生，不輸出 at_risk_profile.json")
        return

    sorted_sems   = sorted(existing_by_semester.keys(), key=_sem_sort_key)
    default_sem   = sorted_sems[-1] if sorted_sems else ""

    # 「全部」聚合：_split_pass_fail 自動排除未結算學生，分母語義正確
    all_enrolled_count  = sum(len(v) for v in sem_groups.values())
    all_students_flat   = [s for group in sem_groups.values() for s in group]
    all_pass_group, all_fail_group = _split_pass_fail(all_students_flat, t["pass_score"])

    all_semesters_profile = None
    if all_enrolled_count:
        all_semesters_profile = _build_profile_for_group(
            semester_label="all",
            semester_code="all",
            pass_group=all_pass_group,
            fail_group=all_fail_group,
            total_enrolled=all_enrolled_count,  # 全體（含未結算）
            course_meta=course_meta,
            thresholds=t,
            exam_dates=exam_dates,
            per_student_exam_dates=_pse,
            is_aggregate=True,  # v7.15.6 問題B修正：跨學期聚合節點不借用單一學期曆法
        )

    output = {
        "schema_version":      SCHEMA_VERSION,
        "available_semesters": sorted_sems,
        "default_semester":    default_sem,
        "course_id":           course_meta.get("course_id", ""),
        "course_name":         course_meta.get("course_name", ""),
        "generated_at":        datetime.now().isoformat(),
        "all_semesters":       all_semesters_profile,
        "by_semester":         {k: existing_by_semester[k] for k in sorted_sems},
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  [OK] at_risk_profile.json 輸出完成（schema {SCHEMA_VERSION}）：{out_path}")
    print(f"     涵蓋學期：{sorted_sems}　本次更新：{updated_sems}")
