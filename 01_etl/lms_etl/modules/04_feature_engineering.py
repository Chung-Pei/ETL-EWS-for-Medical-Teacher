"""
lms_etl/modules/04_feature_engineering.py
計算六大教材類別的量化學習行為指標。
依規格書 V4.0 Section 5 定義。
資料來源：Sheet 2（activity）+ Sheet 4（reading_stats）

Changelog
─────────
v1.1（程式碼審查）：
  FIX-A  compute_completion_features 移除內部 anon_id 篩選（lms_etl 已傳子集）
  FIX-B  compute_time_features 同上
  FIX-C  compute_quiz_features 同上
  FIX-D  compute_quiz_features 首答/末答計算從 iterrows 改為向量化（groupby+rank）
  FIX-E  aud_repeat_rate 改以教材標題去重後計算，修正多筆紀錄低估問題
  FIX-G  考前臨陣磨槍比例改用 union mask，避免考試窗口重疊時雙計
  FIX-H  compute_composite_features 移除 pre_exam_intensity 死碼
  FIX-I  completion_rate fallback 統一用 nunique() 避免量綱不一致

v1.2（/systematic-debugging，依交接文件《ETL交接：04_feature_engineering.py
     時間解析BUG》修正）：
  FIX-J  hms_to_minutes() 新增 MM:SS（1個冒號）解析分支。原正規表示式僅認
         HH:MM:SS，真實資料（8學期、2,913名學生）證實 reading_stats「閱讀
         時間」欄位76-78%的紀錄是MM:SS格式，全數被靜默歸零，影響
         aud_total_minutes/vid_total_minutes/tut_total_minutes等7個衍生
         欄位。修正邏輯比照 05_time_analysis.py::_hms_to_minutes()
         （v7.14.1已修正的姊妹函式，兩者為獨立實作、命名不同，需分別修正）。

v1.3（/systematic-debugging，穿透式審查，補強防呆，無回溯影響、無行為變化）：
  FIX-K  hms_to_minutes() 防呆判斷式由 `not hms or not isinstance(hms, str)`
         改為顯式 `hms is None` + `isinstance(hms, str)` 檢查，不再對原始
         輸入值直接做 `not x` 布林轉換。原寫法若輸入為 pd.NA，`not hms`
         會觸發 `TypeError: boolean value of NA is ambiguous` 而讓整支
         pipeline崩潰（05_time_analysis.py 的姊妹函式因為一律用 isinstance
         判斷、從未對原始值取布林值，不會有此問題）。
         【範圍界定：僅消除上述崩潰風險，對任何輸入的回傳值【完全不變】——
         非字串（含int/float/pd.NA/nan）一律回傳0.0，與既有測試
         test_feature_engineering_time_parse.py::測試0b的顯式斷言一致。
         已用真實資料驗證：目前 01_load_lms_excel.py 的清理流程
         （dtype=str 讀取）不會產生 pd.NA，故此修正對現有8學期真實資料
         【無任何數值變化】，純屬未來資料格式改變時的防護強化，不需
         重跑歷史資料。】
"""

import re
import pandas as pd
from typing import Optional


# ── 時間字串 → 分鐘轉換 ──────────────────────────────────────

def hms_to_minutes(hms) -> float:
    """
    將 "HH:MM:SS"、"H:MM:SS" 或 "MM:SS" 格式轉為浮點分鐘數。
    無法解析時（含 None、非字串型別如 int/float/pd.NA、空字串、格式不符）
    一律回傳 0.0。

    v1.2 [BUG-04-HMS] 修正：原本正規表示式只認得 HH:MM:SS（2個冒號），
    真實資料證實 reading_stats「閱讀時間」欄位76-78%的紀錄是 MM:SS
    （1個冒號，如"17:58"代表17分58秒）格式，全數被靜默解析成0.0分鐘，
    導致 aud_total_minutes/vid_total_minutes/tut_total_minutes 等7個
    欄位自v1起系統性低估（詳見交接文件《ETL交接：04_feature_engineering.py
    時間解析BUG》）。新增 MM:SS 分支修正，邏輯比照
    05_time_analysis.py::_hms_to_minutes()（v7.14.1已修正的姊妹函式）。

    v1.3 [FIX-K] 防呆判斷式改用顯式型別檢查（is None / isinstance），
    避免對 pd.NA 取布林值觸發例外。回傳值行為與v1.2完全相同，不影響
    任何既有真實資料計算結果（詳見上方 changelog v1.3 說明）。
    """
    if hms is None:
        return 0.0
    if not isinstance(hms, str):
        return 0.0
    hms = hms.strip()
    if not hms:
        return 0.0
    m3 = re.match(r"^(\d+):(\d{2}):(\d{2})$", hms)
    if m3:
        h, mi, s = int(m3.group(1)), int(m3.group(2)), int(m3.group(3))
        # [FIX#6] 避免浮點累積誤差：統一轉換為秒後再除
        return (h * 3600 + mi * 60 + s) / 60.0
    m2 = re.match(r"^(\d+):(\d{2})$", hms)
    if m2:
        mi, s = int(m2.group(1)), int(m2.group(2))
        return mi + s / 60.0
    return 0.0


def safe_int(val) -> int:
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return 0


# ── 活動紀錄（Sheet 2）特徵 ────────────────────────────────────

def compute_completion_features(
    activity_df: pd.DataFrame,
    anon_id: str,           # 保留參數供向後相容；lms_etl 已傳學生子集，內部不再篩選
    material_counts: dict,
) -> dict:
    """
    從活動紀錄計算各類別完成率。

    Parameters
    ----------
    activity_df     : 已含 category 欄位的活動紀錄 DataFrame（學生子集）
    anon_id         : 保留參數，供外部識別用途；函式內部不再做 anon_id 篩選
    material_counts : 各類別教材總件數（由 03_parse_tags.count_materials_by_category 產出）

    Returns
    -------
    dict of completion features
    """
    # PERF-FIX（0918 穿透式審查，使用者實測回報 lms_etl step4+5 較久）：
    # 原本對每位學生的小型 DataFrame（單一學生單一學期，通常數十至數百列）
    # 逐 category 做 6 次 boolean mask 篩選；pandas 每次篩選的固定開銷（建
    # 索引、對齊）在小型 DataFrame 上遠大於資料本身的處理量。micro-benchmark
    # （80列模擬資料，500次迭代取平均）：逐 category boolean filter
    # 3.65ms/call；改良為 pandas groupby 一次分組 3.03ms/call（僅約17%
    # 改善）；改為一次性轉 list-of-dict 後用純 Python dict/set 分桶與計數
    # 0.377ms/call（約9.7倍，此為採用方案）。數值語意（dropna/nunique/
    # 計數規則）逐行對齊原 pandas 寫法，並以真實 13 個歷史 LMS 檔全流程
    # 端到端比對新舊輸出 JSON 逐欄位完全一致（見
    # tests/test_completion_features_perf_v7_15_33.py）。
    student_df = activity_df
    features = {}
    categories = ["AUD", "VID", "TXT", "SUP", "TUT", "QUZ"]

    has_category = "category" in student_df.columns
    has_status   = "狀態" in student_df.columns
    has_title    = "活動紀錄" in student_df.columns
    has_bracket  = "bracket_tag" in student_df.columns

    # 一次性取值＋依 category 分桶，取代原本逐 category 篩選 student_df
    if has_category:
        cols = ["category"]
        if has_status:  cols.append("狀態")
        if has_title:   cols.append("活動紀錄")
        if has_bracket: cols.append("bracket_tag")
        by_cat = {}
        for r in student_df[cols].to_dict("records"):
            by_cat.setdefault(r["category"], []).append(r)
    else:
        by_cat = {}

    for cat in categories:
        cat_rows = by_cat.get(cat, [])
        total = material_counts.get(cat, 0)

        # FIX-I/K: 統一用「活動紀錄」欄 nunique()，與 material_counts（標題去重）量綱一致
        # 欄位缺失時 completed=0（安全降級），不 fallback 至任意欄以免語意錯誤
        if has_status and has_title:
            completed_titles = {
                r["活動紀錄"] for r in cat_rows
                if r.get("狀態") == "完成" and pd.notna(r.get("活動紀錄"))
            }
            completed = len(completed_titles)
        else:
            completed = 0
        cat_lower = cat.lower()
        completion_rate = (completed / total) if total > 0 else 0.0

        features[f"{cat_lower}_item_count"]       = completed
        features[f"{cat_lower}_completion_rate"]  = round(min(max(completion_rate, 0.0), 1.0), 4)

    # 延伸閱讀點閱件數（TXT 子類，單獨紀錄）
    # [FIX#1] 防護 bracket_tag 欄位缺失
    if has_bracket and has_status:
        ext_read = sum(
            1 for r in by_cat.get("TXT", [])
            if r.get("bracket_tag") == "延伸閱讀" and r.get("狀態") == "完成"
        )
    else:
        ext_read = 0
    features["extended_reading_count"] = ext_read

    return features


# ── 閱讀時間（Sheet 4）特徵 ───────────────────────────────────

def compute_time_features(
    reading_stats_df: pd.DataFrame,
    anon_id: str,           # 保留參數供向後相容；lms_etl 已傳學生子集，內部不再篩選
) -> dict:
    """
    從「教材閱讀次數及時間」計算各類別投入時間與重複閱讀指標。

    必要欄位：category, 閱讀次數, 閱讀時間
    """
    # FIX-B: lms_etl 已傳學生子集，移除內部全表篩選
    student_df = reading_stats_df

    if student_df.empty:
        return {}
    if "閱讀時間" not in student_df.columns or "閱讀次數" not in student_df.columns:
        return {}

    student_df = student_df.assign(
        _minutes=student_df["閱讀時間"].apply(hms_to_minutes),
        _reads=student_df["閱讀次數"].apply(safe_int),
    )

    # [FIX#1/#2] 統一防護 bracket_tag 欄位缺失
    has_bracket = "bracket_tag" in student_df.columns

    features = {}

    # ── AUD ──
    aud = student_df[student_df["category"] == "AUD"]
    features["aud_total_minutes"] = round(float(aud["_minutes"].sum()), 2)
    # FIX-E: 以教材標題去重後取各教材最大閱讀次數再平均，避免多筆紀錄低估重複率
    if not aud.empty and "活動紀錄" in aud.columns:
        features["aud_repeat_rate"] = round(
            float(aud.groupby("活動紀錄")["_reads"].max().mean()), 2
        )
    else:
        features["aud_repeat_rate"] = round(float(aud["_reads"].mean()), 2) if not aud.empty else 0.0

    # ── VID ──
    vid = student_df[student_df["category"] == "VID"]
    features["vid_total_minutes"] = round(float(vid["_minutes"].sum()), 2)
    # FIX: 對齊 aud_repeat_rate，以教材標題去重後取各教材最大閱讀次數再平均
    if not vid.empty and "活動紀錄" in vid.columns:
        features["vid_avg_repeat"] = round(
            float(vid.groupby("活動紀錄")["_reads"].max().mean()), 2
        )
    else:
        features["vid_avg_repeat"] = round(float(vid["_reads"].mean()), 2) if not vid.empty else 0.0

    # VID 子類：EverCam
    if has_bracket:
        evercam_mask = student_df["bracket_tag"].str.contains("EverCam", na=False)
        features["evercam_minutes"] = round(float(student_df.loc[evercam_mask, "_minutes"].sum()), 2)

        # VID 子類：動畫（含動畫、動畫解析、教學動畫）
        anim_mask = student_df["bracket_tag"].str.contains("動畫", na=False)
        features["animation_minutes"] = round(float(student_df.loc[anim_mask, "_minutes"].sum()), 2)
    else:
        features["evercam_minutes"]   = 0.0
        features["animation_minutes"] = 0.0

    # ── TXT（無時間加權，僅完成率，但可記錄閱讀時間）──
    txt = student_df[student_df["category"] == "TXT"]
    features["txt_total_minutes"] = round(float(txt["_minutes"].sum()), 2)

    # ── SUP ──
    sup = student_df[student_df["category"] == "SUP"]
    if has_bracket and not sup.empty:
        note_mask = sup["bracket_tag"].str.contains("整理筆記", na=False)
        features["sup_repeat_read"] = round(float(sup.loc[note_mask, "_reads"].mean()), 2) if note_mask.any() else 0.0
    else:
        features["sup_repeat_read"] = 0.0
    features["sup_total_minutes"] = round(float(sup["_minutes"].sum()), 2)

    # ── TUT ──
    tut = student_df[student_df["category"] == "TUT"]
    features["tut_total_minutes"] = round(float(tut["_minutes"].sum()), 2)

    # ── 總計 ──
    features["total_learning_minutes"] = round(float(student_df["_minutes"].sum()), 2)

    return features


# ── QUZ 指標（Sheet 6）────────────────────────────────────────

def compute_quiz_features(
    quiz_df: pd.DataFrame,
    anon_id: str,           # 保留參數供向後相容；lms_etl 已傳學生子集，內部不再篩選
    exam_dates: dict,
    material_counts: Optional[dict] = None,
) -> dict:
    """
    從測驗紀錄計算題庫刷題行為指標。

    必要欄位：測驗標題, 測驗時間, 狀態
    exam_dates: { "midterm": "YYYY-MM-DD", "final": "YYYY-MM-DD", ... }

    Ph2b 新增指標：
      quz_first_attempt_accuracy  : 各題庫首次作答及格率（平均）
      quz_final_accuracy          : 各題庫最終（最後一次）作答及格率（平均）
      quz_score_delta             : quz_final_accuracy - quz_first_attempt_accuracy
      quz_cramming_ratio          : 考前 7 天作答次數 / 總作答次數
                                    （與既有 quz_late_cram 考前 3 天並存，不取代）
    """
    # FIX-C: lms_etl 已傳學生子集，移除內部全表篩選
    if "category" in quiz_df.columns:
        quz_df = quiz_df[quiz_df["category"] == "QUZ"].copy()
    else:
        quz_df = quiz_df.iloc[0:0].copy()

    # ── 空值 fallback（含 Ph2b 新增指標）──────────────────────
    if quz_df.empty:
        return {
            "quz_total_attempts":          0,
            "quz_pass_rate":               0.0,
            "quz_first_pass_attempts":     None,
            "quz_coverage":                0.0,
            "quz_late_cram":               0.0,
            "quz_first_attempt_accuracy":  None,
            "quz_final_accuracy":          None,
            "quz_score_delta":             None,
            "quz_cramming_ratio":          0.0,
        }

    time_col   = "測驗時間" if "測驗時間" in quz_df.columns else None
    status_col = "狀態"     if "狀態"     in quz_df.columns else None
    title_col  = "測驗標題" if "測驗標題" in quz_df.columns else None

    if time_col:
        quz_df["_dt"] = pd.to_datetime(quz_df[time_col], errors="coerce")

    total_attempts = len(quz_df)

    # [FIX#4] status_col 為 None 時安全處理
    pass_count = quz_df[quz_df[status_col] == "及格"].shape[0] if status_col else 0

    features = {
        "quz_total_attempts": total_attempts,
        "quz_pass_rate": round(pass_count / total_attempts, 4) if total_attempts > 0 else 0.0,
    }

    # ── 首次及格所需次數（FIX-D：向量化，移除 iterrows）──────
    if title_col and status_col and time_col:
        grp_sorted = quz_df.sort_values("_dt").assign(
            _rank=lambda df: df.groupby(title_col).cumcount() + 1
        )
        is_pass = (grp_sorted[status_col] == "及格")

        # 首次及格所需次數：各題庫中 is_pass==True 的最小 rank
        pass_rows       = grp_sorted[is_pass]
        first_pass_rank = pass_rows.groupby(title_col)["_rank"].min()
        features["quz_first_pass_attempts"] = (
            round(float(first_pass_rank.mean()), 2) if not first_pass_rank.empty else None
        )

        # 首答 / 末答及格率：取各題庫 rank==1（first）與 rank==max（last）
        first_rows = grp_sorted[grp_sorted["_rank"] == 1]
        last_rows  = grp_sorted.loc[grp_sorted.groupby(title_col)["_rank"].idxmax()]

        n_titles = grp_sorted[title_col].nunique()
        if n_titles > 0:
            first_acc = round(float((first_rows[status_col] == "及格").sum() / n_titles), 4)
            final_acc = round(float((last_rows[status_col]  == "及格").sum() / n_titles), 4)
            features["quz_first_attempt_accuracy"] = first_acc
            features["quz_final_accuracy"]         = final_acc
            features["quz_score_delta"]            = round(final_acc - first_acc, 4)
        else:
            features["quz_first_attempt_accuracy"] = None
            features["quz_final_accuracy"]         = None
            features["quz_score_delta"]            = None

        # 題庫覆蓋率
        unique_titles    = quz_df[title_col].nunique()
        total_quz_titles = material_counts.get("QUZ", 0) if material_counts else unique_titles
        features["quz_coverage"] = (
            round(unique_titles / total_quz_titles, 4) if total_quz_titles > 0 else 0.0
        )
    else:
        features.update({
            "quz_first_pass_attempts":    None,
            "quz_coverage":               0.0,
            "quz_first_attempt_accuracy": None,
            "quz_final_accuracy":         None,
            "quz_score_delta":            None,
        })

    # ── 考前臨陣磨槍比例 ─────────────────────────────────────
    # FIX-G: 改用 union mask 合併各考試窗口，避免窗口重疊時同一筆作答被雙計
    if time_col and "_dt" in quz_df.columns:
        union_3d = pd.Series(False, index=quz_df.index)
        union_7d = pd.Series(False, index=quz_df.index)
        for exam_key in ("midterm", "final"):
            exam_date_str = exam_dates.get(exam_key)
            if not exam_date_str:
                continue
            exam_date = pd.Timestamp(exam_date_str)
            union_3d |= (quz_df["_dt"] >= exam_date - pd.Timedelta(days=3)) & (quz_df["_dt"] <= exam_date)
            union_7d |= (quz_df["_dt"] >= exam_date - pd.Timedelta(days=7)) & (quz_df["_dt"] <= exam_date)

        features["quz_late_cram"]      = round(int(union_3d.sum()) / total_attempts, 4) if total_attempts > 0 else 0.0
        features["quz_cramming_ratio"] = round(int(union_7d.sum()) / total_attempts, 4) if total_attempts > 0 else 0.0
    else:
        features["quz_late_cram"]      = 0.0
        features["quz_cramming_ratio"] = 0.0

    return features


# ── 綜合行為指標 ──────────────────────────────────────────────

def compute_composite_features(
    features: dict,
) -> dict:
    """
    計算綜合行為指標。

    Parameters
    ----------
    features : 已計算的各類別指標（由 compute_completion/time_features 產出）
    """
    categories = ["aud", "vid", "txt", "sup", "tut", "quz"]

    # material_diversity_score：有投入的類別數（completion_rate > 0）
    features["material_diversity_score"] = sum(
        1 for cat in categories
        if features.get(f"{cat}_completion_rate", 0) > 0
    )

    # FIX-H: pre_exam_intensity 移除（死碼）
    # 此值需 m05 的 pre_midterm_7d_minutes / pre_final_7d_minutes，
    # 但 m04 執行時 m05 尚未執行，features 中沒有這兩個欄位。
    # lms_etl 主迴圈在 m05 執行後重新計算並覆蓋，m04 版本從未被使用。
    # pre_exam_intensity 由 lms_etl 統一計算，此處不設定佔位值。

    # consistency_score 由 05_time_analysis 模組補填
    if "consistency_score" not in features:
        features["consistency_score"] = None

    return features


# ── 主函式：計算單一學生的完整特徵 ──────────────────────────────

def compute_student_features(
    anon_id: str,
    masked_id: str,
    activity_df: pd.DataFrame,
    reading_stats_df: pd.DataFrame,
    quiz_df: pd.DataFrame,
    material_counts: dict,
    exam_dates: dict,
) -> dict:
    """
    整合計算單一學生的全部特徵，回傳 features dict。
    """
    features = {}
    features.update(compute_completion_features(activity_df, anon_id, material_counts))
    features.update(compute_time_features(reading_stats_df, anon_id))
    features.update(compute_quiz_features(quiz_df, anon_id, exam_dates, material_counts))
    features = compute_composite_features(features)

    return {
        "anon_id":   anon_id,
        "masked_id": masked_id,
        "features":  features,
    }


if __name__ == "__main__":
    print("feature_engineering 模組載入成功")
    print("hms_to_minutes('02:30:00') =", hms_to_minutes("02:30:00"), "（應為 150.0）")
    print("hms_to_minutes('119:57:36') =", hms_to_minutes("119:57:36"), "（應為 7197.6）")
    # v1.2 [FIX-J] MM:SS格式回歸測試（修正前這3筆會被靜默歸零為0.0）
    print("hms_to_minutes('17:58') =", hms_to_minutes("17:58"), "（應為 17.9667）")
    print("hms_to_minutes('23:46') =", hms_to_minutes("23:46"), "（應為 23.7667）")
    print("hms_to_minutes('00:00') =", hms_to_minutes("00:00"), "（應為 0.0）")
