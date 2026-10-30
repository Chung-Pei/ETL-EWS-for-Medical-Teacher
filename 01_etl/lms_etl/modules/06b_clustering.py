"""
lms_etl/modules/06b_clustering.py
規則式學習行為分群（R1–R5，資源使用型態）。
依規格書 V4.0 Section 4 定義，不使用 ML，確保可解釋性。
R 前綴代表 Resource cluster；對應的 LSA 行為序列分群以 S 前綴（S1–S5）輸出。

Changelog
─────────
v1.1.0
  FIX-1 : P5 判斷加入 total_min == 0 的資料缺失防呆，避免空 features 誤分群
  FIX-2 : build_radar_chart_data 的 final_score 型別轉換加 try/except，
           防止非數值字串觸發 ValueError
  FIX-3 : 移除 build_radar_chart_data 未使用的 material_counts 參數
  OPT-1 : 抽出 _group_by_cluster_and_pass 共用函式，消除 by_semester 重複邏輯
  WARN-1: P1 使用 or 條件屬設計決策，保留並加強注釋

v1.1.2
  FIX-R5-THRESHOLD: p5_max_total_minutes 從 120 調整為 1440（24 小時），
    對齊課程規定「全學期最低閱讀時數需 ≥ 24 小時，否則必定人工不及格」。
    同步更新 __main__ 邊界測試案例。

v1.1.1
  FIX-PRACTICUM: assign_clusters 加入 practicum 學生防禦性警告，
    確認進入分群的學生均無 edu_type='practicum'（已由 extract_grade_table 排除）。
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ── 分群標籤定義 ──────────────────────────────────────────────
CLUSTER_LABELS: dict[str, str] = {
    "R1": "影音輔導型",
    "R2": "彈性聽覺型",
    "R3": "平均使用型",
    "R4": "題庫刷題型",
    "R5": "被動低參與型",
}

# ── 分群規則閾值（可依學期調整）──────────────────────────────
THRESHOLDS: dict[str, float] = {
    # P5：低參與，任何類別都不達標
    # 依課程規定：全學期最低閱讀時數 ≥ 24 小時，否則必定人工不及格
    "p5_max_total_minutes": 1440,        # 全學期學習時間 < 1440 分鐘（24 小時）

    # P4：題庫型，QUZ 次數遠高於影音
    "p4_min_quz_attempts":    20,        # 題庫作答 >= 20 次
    "p4_quz_dominance_ratio": 0.40,      # QUZ 佔總學習行為比例 >= 40%

    # P2：聽覺型，AUD 完成率高、VID 相對低
    "p2_min_aud_completion":  0.50,      # AUD 完成率 >= 50%
    "p2_max_vid_completion":  0.50,      # VID 完成率 < 50%

    # P1：影音輔導型，VID 或 TUT 任一達標即符合（設計如此，見 WARN-1）
    "p1_min_vid_completion":  0.50,      # VID 完成率 >= 50%
    "p1_min_tut_minutes":     30,        # TUT 投入 >= 30 分鐘
}

# 用於雷達圖的六大類教材
_CATEGORIES: tuple[str, ...] = ("aud", "vid", "txt", "sup", "tut", "quz")


# ─────────────────────────────────────────────────────────────
# 內部工具函式
# ─────────────────────────────────────────────────────────────

def _safe_final_score(student: dict) -> Optional[float]:
    """
    安全取得期末成績（float），處理 None / NaN / 非數值字串。
    回傳 None 表示成績不可用。
    """
    final = student.get("final_score")
    if final is None:
        return None
    # 排除 float NaN（NaN != NaN 為 True）
    if isinstance(final, float) and final != final:
        return None
    try:
        return float(final)
    except (TypeError, ValueError):
        return None


def _extract_completion_values(student: dict) -> list[float]:
    """取出六大類完成率向量，缺失補 0.0。"""
    feats = student.get("features", {})
    return [feats.get(f"{cat}_completion_rate") or 0.0 for cat in _CATEGORIES]


def _mean_values(group: list[list[float]]) -> list[float]:
    """計算向量群的逐維均值，空群回傳全零。"""
    if not group:
        return [0.0] * len(_CATEGORIES)
    return [round(float(v), 4) for v in np.mean(group, axis=0)]


def _group_students(
    students: list[dict],
) -> tuple[dict[str, list], list, list]:
    """
    將學生依分群與及格/不及格分組，回傳：
    - cluster_groups : {cluster_code: [values, ...]}
    - pass_group     : 及格學生的向量列表
    - fail_group     : 不及格學生的向量列表
    """
    cluster_groups: dict[str, list] = {k: [] for k in CLUSTER_LABELS}
    pass_group: list[list[float]] = []
    fail_group: list[list[float]] = []

    for s in students:
        cluster = s.get("cluster", "R3")
        values = _extract_completion_values(s)
        cluster_groups[cluster].append(values)

        score = _safe_final_score(s)
        if score is not None:
            (pass_group if score >= 60 else fail_group).append(values)

    return cluster_groups, pass_group, fail_group


def _build_clusters_summary(
    cluster_groups: dict[str, list],
) -> dict[str, dict]:
    """將分群向量群轉為輸出格式。"""
    return {
        code: {
            "label":  label,
            "count":  len(cluster_groups.get(code, [])),
            "values": _mean_values(cluster_groups.get(code, [])),
        }
        for code, label in CLUSTER_LABELS.items()
    }


def _build_pass_fail_summary(
    pass_group: list, fail_group: list
) -> dict[str, dict]:
    return {
        "pass": {
            "label":  "及格（≥60）",
            "count":  len(pass_group),
            "values": _mean_values(pass_group),
        },
        "fail": {
            "label":  "不及格（<60）",
            "count":  len(fail_group),
            "values": _mean_values(fail_group),
        },
    }


# ─────────────────────────────────────────────────────────────
# 公開 API
# ─────────────────────────────────────────────────────────────

def classify_student(features: dict) -> str:
    """
    依規則式邏輯將單一學生分群。
    優先順序：P5 > P4 > P2 > P1 > P3

    Parameters
    ----------
    features : behavior.json 中單一學生的 features dict

    Returns
    -------
    str : 資源使用分群代碼（"R1"–"R5"）
    """
    total_min      = features.get("total_learning_minutes") or 0
    vid_completion = features.get("vid_completion_rate")    or 0.0
    aud_completion = features.get("aud_completion_rate")    or 0.0
    tut_minutes    = features.get("tut_total_minutes")      or 0
    quz_attempts   = features.get("quz_total_attempts")     or 0
    diversity      = features.get("material_diversity_score") or 0

    # ── P5：被動低參與 ──
    # FIX-1：total_min == 0 代表資料缺失，與真正低參與同等處理（都分 P5）
    if (total_min < THRESHOLDS["p5_max_total_minutes"]) and (diversity <= 2):
        return "R5"

    # ── R4：題庫刷題型 ──
    if quz_attempts >= THRESHOLDS["p4_min_quz_attempts"]:
        estimated_quz_minutes = quz_attempts * 5
        quz_ratio = estimated_quz_minutes / max(total_min, 1)
        if quz_ratio >= THRESHOLDS["p4_quz_dominance_ratio"]:
            return "R4"

    # ── R2：彈性聽覺型 ──
    if (aud_completion >= THRESHOLDS["p2_min_aud_completion"] and
            vid_completion < THRESHOLDS["p2_max_vid_completion"]):
        return "R2"

    # ── R1：影音輔導型 ──
    # WARN-1：or 條件為規格設計（VID 或 TUT 任一高投入即視為影音導向學習者）
    if (vid_completion >= THRESHOLDS["p1_min_vid_completion"] or
            tut_minutes    >= THRESHOLDS["p1_min_tut_minutes"]):
        return "R1"

    # ── R3：平均使用型（預設）──
    return "R3"


def assign_clusters(students: list[dict]) -> list[dict]:
    """
    對所有學生進行分群，回傳更新後的 students 列表（新增 cluster 欄位）。

    Parameters
    ----------
    students : behavior.json 的 students 陣列（需含 features dict）

    Returns
    -------
    list（每個 student dict 新增 cluster 欄位）
    """
    # FIX-PRACTICUM：防禦性確認，practicum 學生應已於 extract_grade_table 排除
    prac_count = sum(1 for s in students if s.get("edu_type") == "practicum")
    if prac_count > 0:
        print(f"  ⚠️  [06b] 偵測到 {prac_count} 名 edu_type='practicum' 學生進入分群，"
              f"請確認 extract_grade_table 的 practicum 排除邏輯是否正確執行。")

    cluster_counts: dict[str, int] = {k: 0 for k in CLUSTER_LABELS}
    result: list[dict] = []

    for student in students:
        cluster = classify_student(student.get("features", {}))
        student_copy = {**student, "cluster": cluster}
        result.append(student_copy)
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

    print("  [分群結果]")
    for code, label in CLUSTER_LABELS.items():
        print(f"    {code} {label}: {cluster_counts.get(code, 0)} 名")

    return result


def build_radar_chart_data(students: list[dict]) -> dict:
    """
    計算各分群的六大類教材完成率均值，產出 radar_chart_data.json 結構。
    同時按學期分組計算 by_semester（供前端年度篩選使用）。

    Parameters
    ----------
    students : 已含 cluster 欄位的 students 陣列

    Returns
    -------
    dict : radar_chart_data 結構
    """
    # ── 全體統計 ──
    cluster_groups, pass_group, fail_group = _group_students(students)

    # ── by_semester ──
    all_semesters = sorted({
        str(s.get("semester", ""))
        for s in students
        if s.get("semester")
    })

    by_semester: dict[str, dict] = {}
    for sem in all_semesters:
        sem_students = [s for s in students if str(s.get("semester", "")) == sem]
        sem_cg, sem_pass, sem_fail = _group_students(sem_students)
        by_semester[sem] = {
            "student_count": len(sem_students),
            "clusters":      _build_clusters_summary(sem_cg),
            "pass_vs_fail":  _build_pass_fail_summary(sem_pass, sem_fail),
        }

    return {
        "meta": {
            "dimensions":     [c.upper() for c in _CATEGORIES],
            "cluster_labels": CLUSTER_LABELS,
            "student_count":  len(students),
        },
        "clusters":    _build_clusters_summary(cluster_groups),
        "pass_vs_fail": _build_pass_fail_summary(pass_group, fail_group),
        "by_semester": by_semester,
    }


# ─────────────────────────────────────────────────────────────
# 快速驗證
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases: list[tuple[dict, str, str]] = [
        ({"total_learning_minutes": 50,  "material_diversity_score": 1},               "R5", "低參與"),
        ({"total_learning_minutes": 500, "quz_total_attempts": 60,
          "vid_completion_rate": 0.3,   "aud_completion_rate": 0.4,
          "material_diversity_score": 4},                                               "R4", "刷題型"),
        ({"total_learning_minutes": 400, "aud_completion_rate": 0.80,
          "vid_completion_rate": 0.30,  "quz_total_attempts": 10,
          "material_diversity_score": 4},                                               "R2", "聽覺型"),
        ({"total_learning_minutes": 600, "vid_completion_rate": 0.85,
          "tut_total_minutes": 50,      "quz_total_attempts": 10,
          "material_diversity_score": 5},                                               "R1", "影音型"),
        ({"total_learning_minutes": 1500, "vid_completion_rate": 0.40,
          "aud_completion_rate": 0.35,  "quz_total_attempts": 12,
          "material_diversity_score": 5},                                               "R3", "平均使用"),
        # 邊界案例：完全空 features → 應為 R5
        ({},                                                                            "R5", "空資料"),
        # 邊界案例：total_min = 1440（恰好 24 小時，不觸發 R5，條件為嚴格 <）
        ({"total_learning_minutes": 1440, "material_diversity_score": 1},              "R3", "total=1440 不觸發R5"),
        # 邊界案例：total_min = 1439（< 24 小時，觸發 R5）
        ({"total_learning_minutes": 1439, "material_diversity_score": 1},              "R5", "total=1439 觸發R5"),
    ]

    print("分群規則測試：")
    all_pass = True
    for features, expected, desc in test_cases:
        result = classify_student(features)
        ok = result == expected
        if not ok:
            all_pass = False
        print(f"  {'✓' if ok else '✗'} [{result}] 預期 [{expected}] — {desc}")

    print(f"\n{'✅ 全部通過' if all_pass else '❌ 有測試失敗，請調整 THRESHOLDS'}")
