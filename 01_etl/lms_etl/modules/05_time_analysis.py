"""
lms_etl/modules/05_time_analysis.py
從 Sheet 3（教材閱讀紀錄）的時序資料提取時間維度特徵。
依規格書 V4.0 Section 6 定義。

v2.1：新增 compute_heatmap_matrix()、compute_hourly_distribution()
      compute_student_time_profile() 輸出新增 heatmap_matrix、hourly_distribution
v2.2（審查修正）：
  BUG-1  修正 early_start_ratio 重複累計導致高估
  BUG-2  修正期末計算起始點與期中重疊問題
  WARN-1 統一 _hour 欄位防禦性計算邏輯
  WARN-2 移除 TIME_SLOTS dead code dict
  WARN-3 _hms_to_minutes 新增整數/浮點輸入分支
  WARN-4 compute_student_time_profile 改用明確 key 組裝，避免 ** 展開衝突
v2.3（segment_stats 擴充 + 程式碼審查）：
  FEAT-1 compute_class_segment_stats 新增 cluster×pass、sem×pass、sem×cluster 交叉分組
  FIX-1  _get_score 從 closure 提升至模組層級，消除重複定義
  FIX-2  _add_hourly 移除累加階段中間 round，避免雙重捨入累積誤差
  FIX-3  修正 fail 分支 comment 誤標（cluster × pass → cluster × fail）
"""

import re
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional


# ── 學習時段分類 ──────────────────────────────────────────────

def classify_time_slot(hour: int) -> str:
    """將小時數（0~23）分類為四個學習時段。"""
    if 6 <= hour < 12:
        return "MORNING"
    elif 12 <= hour < 18:
        return "AFTERNOON"
    elif 18 <= hour < 23:
        return "EVENING"
    else:
        return "LATE_NIGHT"


# ── 閱讀紀錄解析 ──────────────────────────────────────────────

def parse_reading_log(reading_log_df: pd.DataFrame, anon_id: str) -> pd.DataFrame:
    """
    解析單一學生的時序閱讀紀錄。
    回傳含 _start_dt, _end_dt, _minutes, _hour, _slot 欄位的 DataFrame。
    anon_id 保留供向後相容；lms_etl 已傳學生子集，內部不再篩選。
    """
    student_df = reading_log_df.copy()

    if student_df.empty:
        return student_df

    start_col = "開始" if "開始" in student_df.columns else None
    end_col   = "結束" if "結束" in student_df.columns else None
    dur_col   = "時間" if "時間" in student_df.columns else None

    if start_col:
        student_df["_start_dt"] = pd.to_datetime(student_df[start_col], errors="coerce")
        student_df["_hour"]     = student_df["_start_dt"].dt.hour
        student_df["_slot"]     = student_df["_hour"].apply(
            lambda h: classify_time_slot(int(h)) if pd.notna(h) else "UNKNOWN"
        )

    if end_col:
        student_df["_end_dt"] = pd.to_datetime(student_df[end_col], errors="coerce")

    if dur_col:
        student_df["_minutes"] = student_df[dur_col].apply(_hms_to_minutes)
    elif start_col and end_col:
        student_df["_minutes"] = (
            (student_df["_end_dt"] - student_df["_start_dt"])
            .dt.total_seconds()
            .div(60)
            .clip(lower=0)
            .fillna(0)
        )
    else:
        student_df["_minutes"] = 0.0

    return student_df


def _hms_to_minutes(hms) -> float:
    """
    將時間字串轉換為分鐘數浮點值，支援 'HH:MM:SS' 與 'MM:SS' 兩種格式。
    WARN-3 修正：整數/浮點輸入（如 90 代表秒數或分鐘）靜默回傳 0.0 的問題，
    現改為直接當作「分鐘數」處理，避免上游型別不一致時的靜默丟失。

    BUG-4（v7.14真實資料驗證發現，根因追查至此）：原本只認得
    'HH:MM:SS'（2個冒號）格式，用真實8學期資料（reading_log的「時間」
    欄位，共646,963筆）驗證發現：89.65%的紀錄實際上是 'MM:SS'（僅1個
    冒號，如 "17:58" 代表17分58秒）格式，全數被這裡的正規表示式判定
    「無法解析」而靜默回傳0.0——其中75.18%並非真的是0分鐘，而是真實
    存在、卻被憑空歸零的閱讀時間。只有時長跨過1小時的紀錄（自然序列化
    為HH:MM:SS）才被正確解析，導致：越誇張/越像自動播放掛機的極端值
    反而被正確計入，正常的短時閱讀反而被歸零——資料方向恰好與「偵測
    灌水」的目的相反。新增 MM:SS 分支修正。
    """
    if hms is None:
        return 0.0
    if isinstance(hms, (int, float)):
        return max(0.0, float(hms))
    if not isinstance(hms, str):
        return 0.0
    hms = hms.strip()
    m3 = re.match(r"^(\d+):(\d{2}):(\d{2})$", hms)
    if m3:
        return int(m3.group(1)) * 60 + int(m3.group(2)) + int(m3.group(3)) / 60
    m2 = re.match(r"^(\d+):(\d{2})$", hms)
    if m2:
        return int(m2.group(1)) + int(m2.group(2)) / 60
    return 0.0


# ── 閱讀誠信度：教材離群值上界（v7.14，交接文件v10第1.1節）────────

def compute_material_outlier_bounds(
    reading_log_df: pd.DataFrame,
    title_col: str = "教材標題",
    min_samples: int = 5,
    fallback_cap: float = 90.0,
) -> dict:
    """
    對「全體學生」的教材閱讀紀錄，依教材分別計算離群值上界（Tukey fence）。
    用於 compute_reading_integrity_features() 判斷單次閱讀是否為異常
    （如放置自動播放未實際觀看）。

    方法依據：Rotelli & Monreale (2022, LAK'22) 建議的 IQR-based
    離群偵測；不用固定全域門檻，是因為不同教材（5分鐘動畫 vs 90分鐘
    上課錄影）合理閱讀時長天差地遠，見交接文件 v7 第10.2節。

    Parameters
    ----------
    reading_log_df : 全體學生（未依學生篩選）的教材閱讀紀錄 DataFrame，
                      需含 title_col、"開始"、"結束"（或已有 "_minutes"）欄位
    title_col      : 教材標題欄位名，預設 "教材標題"
    min_samples    : 單一教材至少要有幾筆紀錄才計算 IQR（樣本太少 IQR
                      不穩定，退回 fallback_cap）
    fallback_cap   : (a) 樣本不足教材的固定上限；(b) 母體本身這份教材
                      的95%以上紀錄都接近0分鐘（如僅是翻頁/開啟存取
                      事件，非真正閱讀行為）時，IQR算出的上界會退化到
                      趨近0，此時也改用這個值，見下方「退化保護」說明
                      （單位：分鐘）

    Returns
    -------
    dict：{教材標題: 上界分鐘數}，找不到的教材呼叫端應 fallback 至
    fallback_cap（本函式不內嵌 fallback 值進字典，避免呼叫端誤判
    「這個上界是算出來的還是預設的」）
    """
    if reading_log_df is None or reading_log_df.empty or title_col not in reading_log_df.columns:
        return {}

    df = reading_log_df.copy()
    if "_minutes" not in df.columns:
        start_col = "開始" if "開始" in df.columns else None
        end_col   = "結束" if "結束" in df.columns else None
        if not (start_col and end_col):
            return {}
        df["_minutes"] = (
            (pd.to_datetime(df[end_col], errors="coerce") -
             pd.to_datetime(df[start_col], errors="coerce"))
            .dt.total_seconds().div(60).clip(lower=0).fillna(0)
        )

    bounds = {}
    for material, grp in df.groupby(title_col):
        durations = grp["_minutes"].dropna()
        if len(durations) < min_samples:
            continue
        q1, q3 = durations.quantile([0.25, 0.75])
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        if upper < 1.0:
            # 退化保護：母體本身這份教材95%以上紀錄都接近0分鐘（如僅是
            # 翻頁/開啟的存取事件，多數學生的真正學習發生在其他教材，
            # 例如影片），IQR門檻在這種近乎全零的分布下不可靠——不能
            # 直接沿用算出來的極小值當上界，否則會把單一極端值（如
            # 掛著未關的41小時存取紀錄）幾乎整筆裁掉，反而壓縮過頭。
            # 改退回 fallback_cap，比照「教材不在母體統計裡」的處理方式。
            upper = fallback_cap
        bounds[material] = float(upper)

    return bounds


# ── 學習時段分布 ──────────────────────────────────────────────

_SLOT_KEYS = ("MORNING", "AFTERNOON", "EVENING", "LATE_NIGHT")


def compute_time_slot_distribution(log_df: pd.DataFrame) -> dict:
    """
    計算四時段各佔學習時間比例。
    回傳 { "MORNING": 0.12, "AFTERNOON": 0.18, "EVENING": 0.52, "LATE_NIGHT": 0.18 }
    """
    if log_df.empty or "_slot" not in log_df.columns or "_minutes" not in log_df.columns:
        return {k: 0.0 for k in _SLOT_KEYS}

    slot_minutes = log_df.groupby("_slot")["_minutes"].sum()
    total = slot_minutes.sum()

    return {
        slot: round(float(slot_minutes.get(slot, 0)) / total, 4) if total > 0 else 0.0
        for slot in _SLOT_KEYS
    }


def compute_dominant_slot(dist: dict) -> str:
    """回傳佔比最高的時段代碼。"""
    if not dist or all(v == 0 for v in dist.values()):
        return "UNKNOWN"
    return max(dist, key=dist.get)


# ── 距考試日期分析 ────────────────────────────────────────────

def compute_exam_proximity_features(
    log_df: pd.DataFrame,
    exam_dates: dict,
) -> dict:
    """
    計算學習行為與考試日期的時間關係特徵。
    """
    features = {
        "pre_midterm_7d_minutes":  0.0,
        "pre_final_7d_minutes":    0.0,
        "midterm_regular_minutes": 0.0,
        "midterm_period_minutes":  0.0,
        "final_regular_minutes":   0.0,
        "final_period_minutes":    0.0,
        "days_to_exam_avg":        None,
        "early_start_ratio":       0.0,
        "cram_pattern_score":      0.0,
    }

    if log_df.empty or "_start_dt" not in log_df.columns:
        return features

    valid_log = log_df.dropna(subset=["_start_dt"])
    if valid_log.empty:
        return features

    total_minutes = valid_log["_minutes"].sum()

    exam_configs = [
        ("midterm", "pre_midterm_7d_minutes"),
        ("final",   "pre_final_7d_minutes"),
    ]

    exam_dates_parsed = {}
    for key, feat_key in exam_configs:
        date_str = exam_dates.get(key)
        if not date_str:
            continue
        exam_dt = pd.Timestamp(date_str)
        exam_dates_parsed[key] = exam_dt

        window_start = exam_dt - pd.Timedelta(days=7)
        mask_7d = (valid_log["_start_dt"] >= window_start) & (valid_log["_start_dt"] <= exam_dt)
        features[feat_key] = round(float(valid_log.loc[mask_7d, "_minutes"].sum()), 2)

    first_log_dt = valid_log["_start_dt"].min()
    midterm_dt = exam_dates_parsed.get("midterm")
    final_dt   = exam_dates_parsed.get("final")

    def _period_minutes(start_dt, exam_dt):
        """計算 [start_dt, exam_dt] 期間的 regular（非考前 7 天）與 total 分鐘數。"""
        if pd.isna(start_dt) or exam_dt is None:
            return 0.0, 0.0
        pre_start    = exam_dt - pd.Timedelta(days=7)
        period_mask  = (valid_log["_start_dt"] >= start_dt) & (valid_log["_start_dt"] <= exam_dt)
        regular_mask = (valid_log["_start_dt"] >= start_dt) & (valid_log["_start_dt"] < pre_start)
        return (
            round(float(valid_log.loc[regular_mask, "_minutes"].sum()), 2),
            round(float(valid_log.loc[period_mask,  "_minutes"].sum()), 2),
        )

    mid_regular, mid_period = _period_minutes(first_log_dt, midterm_dt)

    # BUG-2 修正：期末起始點加一天，避免與期中計算範圍重疊
    final_start = (midterm_dt + pd.Timedelta(days=1)) if midterm_dt is not None else first_log_dt
    fin_regular, fin_period = _period_minutes(final_start, final_dt)

    features["midterm_regular_minutes"] = mid_regular
    features["midterm_period_minutes"]  = mid_period
    features["final_regular_minutes"]   = fin_regular
    features["final_period_minutes"]    = fin_period

    all_exam_dates = list(exam_dates_parsed.values())

    if all_exam_dates and not valid_log.empty:
        start_series = valid_log["_start_dt"]
        days_matrix  = np.column_stack([
            (exam - start_series).dt.days.abs().values
            for exam in all_exam_dates
        ])
        min_days = days_matrix.min(axis=1)
        features["days_to_exam_avg"] = round(float(min_days.mean()), 2)

    # BUG-1 修正：early_start_ratio 使用「所有考試日 cutoff 的最小值」作為唯一基準，
    # 避免同一筆學習記錄被多次考試的 cutoff 重複累加。
    if all_exam_dates and total_minutes > 0:
        earliest_cutoff = min(exam_dt - pd.Timedelta(days=21) for exam_dt in all_exam_dates)
        early_mask      = valid_log["_start_dt"] < earliest_cutoff
        early_minutes   = float(valid_log.loc[early_mask, "_minutes"].sum())
        features["early_start_ratio"] = round(early_minutes / total_minutes, 4)

    pre7d_total = features["pre_midterm_7d_minutes"] + features["pre_final_7d_minutes"]
    features["cram_pattern_score"] = round(
        pre7d_total / total_minutes, 4
    ) if total_minutes > 0 else 0.0

    return features


# ── 週學習節律分析 ────────────────────────────────────────────

def compute_weekly_rhythm(
    log_df: pd.DataFrame,
    semester_start: str,
    total_weeks: int = 18,
) -> dict:
    """
    分析學生的週學習節律穩定性。
    """
    features = {
        "active_weeks":       0,
        "weekly_minutes_std": 0.0,
        "zero_week_count":    0,
        "first_login_week":   None,
        "consistency_score":  0.0,
        # [ETL-1 依賴] 07c_stability.py::compute_stability_cv() 需要逐週
        # 原始時數（而非僅統計量 weekly_minutes_std）才能算 CV=std/mean。
        # 沿用 total_weeks 全長度（含 0 值週），由呼叫端自行過濾非0週。
        "weekly_minutes":     [],
    }

    if log_df.empty or "_start_dt" not in log_df.columns:
        return features

    # semester_start 為空字串或 None 時提早返回
    if not semester_start:
        return features
    sem_start = pd.Timestamp(semester_start)
    if pd.isna(sem_start):
        return features

    valid_log = log_df.dropna(subset=["_start_dt"]).copy()
    if valid_log.empty:
        return features

    valid_log["_week"] = (
        (valid_log["_start_dt"] - sem_start)
        .dt.days
        .floordiv(7)
        .clip(0, total_weeks - 1)
        + 1
    )

    weekly_minutes = valid_log.groupby("_week")["_minutes"].sum()
    all_weeks      = pd.Series(0.0, index=range(1, total_weeks + 1))
    all_weeks.update(weekly_minutes)

    active_weeks = int((all_weeks > 0).sum())
    features["active_weeks"]       = active_weeks
    features["zero_week_count"]    = int((all_weeks == 0).sum())
    features["weekly_minutes_std"] = round(float(all_weeks.std()), 2)
    features["first_login_week"]   = int(valid_log["_week"].min())
    features["weekly_minutes"]     = [round(float(v), 2) for v in all_weeks.tolist()]

    std = features["weekly_minutes_std"]
    features["consistency_score"] = (
        round(1 / (1 + std / 60), 4) if std > 0
        else (1.0 if active_weeks > 0 else 0.0)
    )

    return features


# ── 熱力圖矩陣 ────────────────────────────────────────────────

def compute_heatmap_matrix(log_df: pd.DataFrame) -> dict:
    """
    計算個人學習熱力圖矩陣（7×24）。

    Returns
    -------
    dict:
      "raw"        : { "weekday_hour": minutes }  weekday 0=Mon…6=Sun；省略 0 值
      "normalized" : 同 key，value 為佔總學習時間比例（加總 = 1.0）
    """
    def _empty(): return {"raw": {}, "normalized": {}}

    if log_df.empty or "_start_dt" not in log_df.columns or "_minutes" not in log_df.columns:
        return _empty()

    valid = log_df.dropna(subset=["_start_dt"])
    if valid.empty:
        return _empty()

    # WARN-1 修正：防禦性取用 _hour（parse_reading_log 已建立則複用）
    # 用 assign 避免 SettingWithCopyWarning（dropna 回傳的 view 不保證是 copy）
    valid = valid.assign(
        _weekday=valid["_start_dt"].dt.weekday,
        _hour=(valid["_hour"] if "_hour" in valid.columns else valid["_start_dt"].dt.hour),
    )
    grouped = valid.groupby(["_weekday", "_hour"], sort=True)["_minutes"].sum()

    raw: dict = {}
    for (wd, hr), mins in grouped.items():
        val = round(float(mins), 1)
        if val > 0:
            raw[f"{int(wd)}_{int(hr)}"] = val

    total = sum(raw.values())
    normalized = (
        {k: round(v / total, 6) for k, v in raw.items()}
        if total > 0 else {}
    )

    return {"raw": raw, "normalized": normalized}


# ── 24 小時分布向量 ───────────────────────────────────────────

def compute_hourly_distribution(log_df: pd.DataFrame) -> dict:
    """
    計算個人 24 小時學習分布向量。

    Returns
    -------
    dict:
      "raw"        : list[float], 長度 24，index = 小時（0~23）
      "normalized" : list[float], 長度 24，加總 = 1.0
    """
    _zero24 = [0.0] * 24

    if log_df.empty or "_start_dt" not in log_df.columns or "_minutes" not in log_df.columns:
        return {"raw": _zero24.copy(), "normalized": _zero24.copy()}

    valid = log_df.dropna(subset=["_start_dt"])
    if valid.empty:
        return {"raw": _zero24.copy(), "normalized": _zero24.copy()}

    # 防禦性取用 _hour，用 assign 避免 SettingWithCopyWarning
    hour_series = valid["_hour"] if "_hour" in valid.columns else valid["_start_dt"].dt.hour
    hourly = valid.assign(_hour=hour_series).groupby("_hour")["_minutes"].sum()

    raw   = [round(float(hourly.get(h, 0.0)), 2) for h in range(24)]
    total = sum(raw)
    normalized = (
        [round(v / total, 6) for v in raw]
        if total > 0 else _zero24.copy()
    )

    return {"raw": raw, "normalized": normalized}


# ── 閱讀誠信度：單一學生特徵（v7.14，交接文件v10第1.2節）────────

def compute_reading_integrity_features(
    log_df: pd.DataFrame,
    material_bounds: dict,
    title_col: str = "教材標題",
    fallback_cap: float = 90.0,
    policy_hours: float = 24.0,
) -> dict:
    """
    對單一學生「已經過 parse_reading_log() 解析」的 log_df，
    依教材離群上界（material_bounds，見 compute_material_outlier_bounds）
    計算清理後的真實閱讀時數，以及政策達標灌水旗標。

    驗證依據：交接文件 v8 第10.4節，用114-2四技247人實測，
    capped_total_minutes 與學期成績顯著正相關（r=0.15, p=0.019），
    而現行未清理的 raw 值無顯著關聯（p=0.96）。8學期合併驗證（n=2,675，
    見交接文件v10第0.5節）confirmed 現象跨學期一致存在（12.7%盛行率）。

    Parameters
    ----------
    log_df          : compute_student_time_profile() 內已呼叫過
                       parse_reading_log() 的單一學生 DataFrame
                       （已含 "_minutes" 欄位）
    material_bounds : compute_material_outlier_bounds() 的回傳值
    fallback_cap    : 教材不在 material_bounds 裡時的固定上限（分鐘）
    policy_hours    : 課程規定的累積閱讀時數門檻（小時），預設24，
                       建議設計成參數、不要寫死，因為不同課程/學制
                       規定可能不同

    Returns
    -------
    dict，新增以下欄位（將合併進學生 features）：
      - capped_total_minutes  : 清理後真實時數（分鐘）
      - raw_total_minutes     : 原始（未清理）時數，供稽核對照
      - reading_inflation_ratio : raw/capped，越高越可疑
      - policy_gaming_flag    : bool，raw達標但capped未達標
    """
    if log_df is None or log_df.empty or "_minutes" not in log_df.columns:
        return {
            "capped_total_minutes":     0.0,
            "raw_total_minutes":        0.0,
            "reading_inflation_ratio":  1.0,
            "policy_gaming_flag":       False,
        }

    if title_col in log_df.columns:
        # 向量化（取代逐列 .apply()）：Series.map() 查表 + np.minimum 比較，
        # 避免逐列呼叫 Python 函式的開銷。實測（模擬全體8學期規模、
        # 64萬列）：apply版5.87秒 → 向量化版1.72秒，約3.4倍。
        material_upper = log_df[title_col].map(material_bounds).fillna(fallback_cap)
        capped_series = np.minimum(log_df["_minutes"].to_numpy(), material_upper.to_numpy())
    else:
        capped_series = log_df["_minutes"].clip(upper=fallback_cap).to_numpy()

    raw_total    = float(log_df["_minutes"].sum())
    capped_total = float(capped_series.sum())

    raw_hours    = raw_total / 60.0
    capped_hours = capped_total / 60.0
    inflation_ratio = round(raw_total / capped_total, 3) if capped_total > 0 else 1.0
    gaming_flag = bool(raw_hours >= policy_hours and capped_hours < policy_hours)

    return {
        "capped_total_minutes":    round(capped_total, 2),
        "raw_total_minutes":       round(raw_total, 2),
        "reading_inflation_ratio": inflation_ratio,
        "policy_gaming_flag":      gaming_flag,
    }


# ── 主函式 ────────────────────────────────────────────────────

def compute_student_time_profile(
    anon_id: str,
    reading_log_df: pd.DataFrame,
    exam_dates: dict,
    semester_start: str,
    total_weeks: int = 18,
    material_bounds: Optional[dict] = None,   # ← 新增參數，預設 None 向後相容
    policy_hours: float = 24.0,                # ← 新增參數
) -> dict:
    """
    整合計算單一學生的所有時間維度特徵，回傳 time_profile dict。

    v2.1 新增欄位：heatmap_matrix、hourly_distribution
    v2.2 WARN-4 修正：明確 key 組裝取代 ** 展開，避免靜默覆蓋風險。
    """
    log_df        = parse_reading_log(reading_log_df, anon_id)
    dist          = compute_time_slot_distribution(log_df)
    exam_features = compute_exam_proximity_features(log_df, exam_dates)
    rhythm        = compute_weekly_rhythm(log_df, semester_start, total_weeks)
    heatmap       = compute_heatmap_matrix(log_df)
    hourly        = compute_hourly_distribution(log_df)
    # ── 新增（v7.14）：閱讀誠信度 ──
    integrity     = compute_reading_integrity_features(
        log_df, material_bounds or {}, policy_hours=policy_hours
    )

    return {
        # 時段
        "dominant_time_slot":     compute_dominant_slot(dist),
        "late_night_ratio":       dist.get("LATE_NIGHT", 0.0),
        "time_slot_distribution": dist,
        # 考試日期（供 07_export_json 使用）
        "midterm_date": exam_dates.get("midterm", ""),
        "final_date":   exam_dates.get("final",   ""),
        # 距考特徵（明確展開，避免 ** 衝突）
        "pre_midterm_7d_minutes":  exam_features["pre_midterm_7d_minutes"],
        "pre_final_7d_minutes":    exam_features["pre_final_7d_minutes"],
        "midterm_regular_minutes": exam_features["midterm_regular_minutes"],
        "midterm_period_minutes":  exam_features["midterm_period_minutes"],
        "final_regular_minutes":   exam_features["final_regular_minutes"],
        "final_period_minutes":    exam_features["final_period_minutes"],
        "days_to_exam_avg":        exam_features["days_to_exam_avg"],
        "early_start_ratio":       exam_features["early_start_ratio"],
        "cram_pattern_score":      exam_features["cram_pattern_score"],
        # 週節律（明確展開）
        "active_weeks":            rhythm["active_weeks"],
        "weekly_minutes_std":      rhythm["weekly_minutes_std"],
        "zero_week_count":         rhythm["zero_week_count"],
        "first_login_week":        rhythm["first_login_week"],
        "consistency_score":       rhythm["consistency_score"],
        # [ETL-1] 07c_stability.py 用此欄位計算 stability_cv（std/mean）
        "weekly_minutes":          rhythm["weekly_minutes"],
        # v2.1 新增
        "heatmap_matrix":          heatmap,
        "hourly_distribution":     hourly,
        # v7.14 新增：閱讀誠信度
        "capped_total_minutes":    integrity["capped_total_minutes"],
        "raw_total_minutes":       integrity["raw_total_minutes"],
        "reading_inflation_ratio": integrity["reading_inflation_ratio"],
        "policy_gaming_flag":      integrity["policy_gaming_flag"],
    }


# ── 分段預聚合（方案 B）────────────────────────────────────

def _get_score(s: dict):
    """final_score 優先，fallback semester_score，若皆無則 fallback midterm_score（進行中學期暫時辨別依據）。"""
    for key in ("final_score", "semester_score", "midterm_score"):
        v = s.get(key)
        if v is not None:
            try:
                f = float(v)
                return f if np.isfinite(f) else None
            except (TypeError, ValueError):
                pass
    return None


def compute_class_segment_stats(
    students: list,
    pass_threshold: float = 60.0,
) -> dict:
    """
    依方案 B（線性展開）預聚合時間分析所需的分段統計，
    供前端篩選時直接讀取，取代即時加總運算。

    分組 key 格式："{semester}|{cluster}|{pass_status}"
    輸出組合：
      "all|all|all"          ← 全量
      "<sem>|all|all"        ← 各學期（單維）
      "all|<cluster>|all"    ← 各分群 P1~P5（單維）
      "all|all|pass"         ← 及格學生（單維）
      "all|all|fail"         ← 不及格學生（單維）
      "all|<cluster>|pass"   ← 各分群 × 及格（交叉）
      "all|<cluster>|fail"   ← 各分群 × 不及格（交叉）
      "<sem>|all|pass"       ← 各學期 × 及格（quiz 篩選用）
      "<sem>|all|fail"       ← 各學期 × 不及格（quiz 篩選用）
      "<sem>|<cluster>|all"  ← 各學期 × 各分群（交叉）

    每組輸出：
      heatmap_raw          : dict  累加 time_profile.heatmap_matrix.raw
      hourly_raw           : list  累加 time_profile.hourly_distribution.raw（長度 24）
      late_night_ratio_mean: float 平均深夜比例
      avg_score            : float | None  平均成績（final_score 優先，fallback semester_score）
      p15_midterm          : float  midterm_period_minutes 第 15 百分位
      p15_final            : float  final_period_minutes 第 15 百分位
      student_count        : int

    信心旗標（由 compute_segment_pearson 使用，此處不附加）：
      n < 10  → 整個 key 跳過不輸出
      10 ≤ n  → 正常輸出（low_confidence 由相關性模組判斷）

    Parameters
    ----------
    students       : lms_etl.py 的 students list，每筆需含
                     anon_id, cluster, semester, final_score/semester_score,
                     time_profile（含 heatmap_matrix, hourly_distribution,
                     late_night_ratio, midterm_period_minutes, final_period_minutes）
    pass_threshold : 及格門檻（預設 60）

    Returns
    -------
    dict : { segment_key: { heatmap_raw, hourly_raw, late_night_ratio_mean,
                             avg_score, p15_midterm, p15_final, student_count } }
    """
    MIN_SAMPLES = 10

    def _add_hourly(acc: list, delta: list) -> list:
        """長度截斷至 24，in-place 累加"""
        for i, v in enumerate((delta or [])[:24]):
            acc[i] += v or 0.0
        return acc

    def _add_heatmap(acc: dict, delta: dict) -> None:
        """In-place 累加 heatmap raw dict"""
        for k, v in (delta or {}).items():
            acc[k] = round(acc.get(k, 0.0) + (v or 0.0), 2)

    # ── 建立分組索引 ─────────────────────────────────────────
    # group_map: key → list of students
    group_map: dict = defaultdict(list)

    for s in students:
        sem     = str(s.get("semester") or "").strip()
        cluster = str(s.get("cluster")  or "").strip()
        score   = _get_score(s)
        pass_status = (
            "pass" if score is not None and score >= pass_threshold
            else ("fail" if score is not None else None)
        )

        # 全量
        group_map["all|all|all"].append(s)
        # 學期（單維）
        if sem:
            group_map[f"{sem}|all|all"].append(s)
        # 分群（單維）
        if cluster:
            group_map[f"all|{cluster}|all"].append(s)
        # 及格狀態（單維） + cluster × pass 交叉 + sem × pass 交叉
        if pass_status == "pass":
            group_map["all|all|pass"].append(s)
            if cluster:
                group_map[f"all|{cluster}|pass"].append(s)   # cluster × pass
            if sem:
                group_map[f"{sem}|all|pass"].append(s)       # sem × pass（quiz 篩選用）
        elif pass_status == "fail":
            group_map["all|all|fail"].append(s)
            if cluster:
                group_map[f"all|{cluster}|fail"].append(s)   # cluster × fail
            if sem:
                group_map[f"{sem}|all|fail"].append(s)       # sem × fail（quiz 篩選用）
        # 學期 × 分群交叉
        if sem and cluster:
            group_map[f"{sem}|{cluster}|all"].append(s)

    # ── 逐組計算 ─────────────────────────────────────────────
    result: dict = {}

    for seg_key, group in group_map.items():
        n = len(group)
        if n < MIN_SAMPLES:
            continue

        heatmap_raw: dict = {}
        hourly_raw:  list = [0.0] * 24
        late_night_vals:    list = []
        score_vals:         list = []
        mid_period_vals:    list = []
        fin_period_vals:    list = []

        for s in group:
            tp = s.get("time_profile") or {}

            # heatmap 累加
            _add_heatmap(heatmap_raw, tp.get("heatmap_matrix", {}).get("raw", {}))

            # hourly 累加
            _add_hourly(hourly_raw, tp.get("hourly_distribution", {}).get("raw", []))

            # late_night_ratio
            lnr = tp.get("late_night_ratio")
            if lnr is not None:
                try:
                    late_night_vals.append(float(lnr))
                except (TypeError, ValueError):
                    pass

            # 成績
            sc = _get_score(s)
            if sc is not None:
                score_vals.append(sc)

            # P15 原始值
            for vals, key in (
                (mid_period_vals, "midterm_period_minutes"),
                (fin_period_vals, "final_period_minutes"),
            ):
                raw_v = tp.get(key)
                if raw_v is not None:
                    try:
                        fv = float(raw_v)
                        if fv > 0:
                            vals.append(fv)
                    except (TypeError, ValueError):
                        pass

        # 計算各項統計
        late_night_mean = (
            round(float(np.mean(late_night_vals)), 6)
            if late_night_vals else 0.0
        )
        avg_score = (
            round(float(np.mean(score_vals)), 4)
            if score_vals else None
        )
        p15_mid = (
            round(float(np.percentile(mid_period_vals, 15)), 2)
            if mid_period_vals else 0.0
        )
        p15_fin = (
            round(float(np.percentile(fin_period_vals, 15)), 2)
            if fin_period_vals else 0.0
        )

        result[seg_key] = {
            "heatmap_raw":           heatmap_raw,
            "hourly_raw":            [round(v, 2) for v in hourly_raw],
            "late_night_ratio_mean": late_night_mean,
            "avg_score":             avg_score,
            "p15_midterm":           p15_mid,
            "p15_final":             p15_fin,
            "student_count":         n,
        }

    print(f"  [05] segment_stats：{len(result)} 組（MIN_SAMPLES={MIN_SAMPLES}）")
    return result


if __name__ == "__main__":
    print("time_analysis 模組載入成功")
    print("classify_time_slot(22) =", classify_time_slot(22), "（應為 EVENING）")
    print("classify_time_slot(2)  =", classify_time_slot(2),  "（應為 LATE_NIGHT）")
    print("classify_time_slot(9)  =", classify_time_slot(9),  "（應為 MORNING）")
