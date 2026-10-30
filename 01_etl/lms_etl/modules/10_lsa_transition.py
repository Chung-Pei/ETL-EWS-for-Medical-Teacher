"""
lms_etl/modules/10_lsa_transition.py
計算 Lag-1 行為序列轉移頻率矩陣、期望值、Adjusted Residuals Z-score，
分 all / pass / fail 三組輸出。

Z-score 公式（Adjusted Standardized Residuals）：
  E[A→B] = (row_total_A × col_total_B) / N
  Z[A→B] = (O[A→B] − E[A→B]) / sqrt(E[A→B] × (1 − row_prop_A) × (1 − col_prop_B))
顯著性：|Z| > 1.96（雙尾 p ≈ 0.05）

Changelog
─────────
v1.4.0
  FEAT-XGB：compute_lsa_by_lsa_type 回傳值擴充為 tuple(dict, student_lsa_type, transition_ratios)，
    新增 MM/MQ/QM/QQ_ratio（個人 Lag-1 轉移比例），供 XGBoost 雙模型早期預警系統特徵使用。

v1.1.0
  BUG-CLUSTER-1：compute_lsa_by_cluster 的 key 轉換原僅處理 R→S，
    若 event_log.cluster 為舊版 P1–P5（ETL 過渡期），by_cluster 全空。
    改用 _normalize_cluster_key()，同時相容 R/P → S。

v1.3.0
  THRESHOLD-ADJ：依實際資料分布（全體 p_mm 均值 ≈ 0.95）調整 S1/S2 閾值，
    使 S1/S2 能反映本課程高連續閱讀特性下的實際學習差異。
    S1 p_mm 上限：0.60 → 0.85
    S2 p_mm 範圍：0.60–0.85 → 0.85–0.95
    S2 max|Z| 條件：移除（全體 Z 值極大，此條件實際造成 S2 永遠為空）

v1.2.0
  FEAT-LSA-TYPE：新增真正的 S 分群邏輯（by_lsa_type），
    依每位學生自身的轉移特徵（p_mm、active_days、n_events、stability_cv）
    分類為 S1–S5（穩定高效 / 規律中效 / 波動中效 / 低頻低效 / 高風險）。
    與現有 by_cluster（R 資源分群切片）並存，不影響既有輸出。

  FIX-Q1：compute_lsa_by_cluster 移除 R→S 換牌，by_cluster key 保持 R1–R5，
    與 by_lsa_type 的 S1–S5 key 清楚分離；_normalize_cluster_key() 已移除。

  FIX-Q2：compute_lsa_by_lsa_type 改為回傳 tuple(dict, student_lsa_type)，
    student_lsa_type 不混入矩陣結構，避免 null key 問題。

    S 分群操作定義：
      p_mm            = p(M→M) = observed[M→M] / behavior_totals[M]（連續閱讀率，代理轉移熵）
      active_days     = 個人 timestamp 去重日期數（從 event_log 計算）
      median_days     = 同學期所有學生 active_days 的中位數
      n_events        = 個人序列對總數
      q1_events       = 同學期 n_events 的下四分位數
      stability_cv    = student["features"]["stability_cv"]（由 07c_stability.py 回寫）
      Z_MQ / Z_QM     = 個人 Z-score（當個人序列數 < 5 時視為 None）

    分群優先順序：S5 > S4 > S1 > S2 > S3

    S5（高風險）：|Z_MQ| > 3.0 且 |Z_QM| > 3.0，且 Z_MQ 與 Z_QM 異號
    S4（低頻低效）：active_days < median_days 且 n_events < q1_events
    S1（穩定高效）：active_days >= median_days 且 p_mm < _P_MM_S1S2_SPLIT 且任一 Z > +_Z_SIG_POS
    S2（規律中效）：active_days >= median_days 且 _P_MM_S1S2_SPLIT <= p_mm <= _P_MM_S2_MAX
    S3（波動中效）：其餘（含 p_mm > 0.95）
"""

import math
import numpy as np
import pandas as pd

BEHAVIORS = ["M", "Q"]
_KEYS = [f"{a}→{b}" for a in BEHAVIORS for b in BEHAVIORS]   # M→M, M→Q, Q→M, Q→Q

# ── S 分群閾值常數（v1.3.0 THRESHOLD-ADJ，依實際資料分布校正）──────────
# 單一來源：所有 classify_student_lsa 條件式引用此處，避免分散 magic number。
_Z_SIG_POS    = 1.96   # 雙尾 p ≈ 0.05 顯著性（S1 條件：任一正向 Z > 此值）
_Z_EXTREME    = 3.0    # 極端異常 Z 閾值（S5 條件：|Z_MQ| 與 |Z_QM| 均超過此值）
_P_MM_S1S2_SPLIT = 0.85  # S1/S2 分界：S1 用 p_mm < 此值，S2 用 p_mm >= 此值
_P_MM_S2_MAX  = 0.95   # S2 連續閱讀率上限（> 此值歸 S3 過度固著）

# S 分群標籤（by_lsa_type 用）
LSA_TYPE_LABELS: dict[str, str] = {
    "S1": "穩定高效",
    "S2": "規律中效",
    "S3": "波動中效",
    "S4": "低頻低效",
    "S5": "高風險",
}

# ── (anon_id, semester) 複合鍵的 JSON-safe 字串編碼（決議10）──────────
# lsa_type_map / transition_ratios 最終會被 07_export_json.py 直接寫入
# lsa_transition.json（見 lms_etl.py Step 8→Step 9→m07.export_all），
# Python dict 若用 tuple 當 key，json.dump 會直接 TypeError。改用
# 「anon_id||semester」字串鍵，consumer（11_cross_analysis.py /
# 12_early_warning.py）用 composite_key() 組出同樣字串來查表。
_KEY_SEP = "||"

def composite_key(anon_id: str, semester: str) -> str:
    """組出 (anon_id, semester) 的 JSON-safe 字串複合鍵，供跨模組共用同一套編碼。"""
    return f"{anon_id}{_KEY_SEP}{semester}"

def compute_transition_matrix(event_log: pd.DataFrame, already_filtered: bool = False) -> dict:
    """
    Parameters
    ----------
    event_log : build_event_log() 的輸出（含 behavior, next_behavior 欄）
    already_filtered : 呼叫端若能保證傳入的 event_log 每一列的
                behavior/next_behavior 都已經屬於 BEHAVIORS（例如上游已經
                做過同樣的 .isin() 篩選），可設為 True 跳過本函式內部重複
                的篩選＋.copy()。預設 False（維持所有既有呼叫端的行為
                完全不變，零回歸風險）。

                [效能優化 穿透式審查 0913，真實資料規模profiling實測發現]
                compute_lsa_by_lsa_type() 的Pass 2迴圈對每一位學生每一
                學期都呼叫一次 _compute_student_matrix()（進而呼叫這裡），
                但傳入的 grp 早已是從同函式最上方 valid_log（已篩選過
                behavior/next_behavior）groupby 出來的子集，於是这裡的
                篩選＋.copy()對每一次呼叫都是重複且結果必為no-op的白工。
                用真實LMS資料（1個學期，38,890列）拼接放大24倍模擬多學期
                全量規模（93萬列、5,976個「學生」）離線profiling量測：
                compute_lsa_by_lsa_type 單獨耗時 31.4 秒，明顯高於
                compute_lsa_all_groups/by_semester/by_cluster（同規模下
                皆 <1 秒）——是Step 8目前唯一的重負荷函式。已直接對照
                「micro_immuno_reading_shortfall_1151.json優化前後
                byte-for-byte diff」與全部29個既有測試檔案，確認此參數
                純粹跳過恆為no-op的重複運算，不改變任何輸出數值。
    """
    empty = _empty_matrix()

    if event_log is None or event_log.empty:
        return empty
    if "behavior" not in event_log.columns or "next_behavior" not in event_log.columns:
        print("  ⚠️  [10] event_log 缺少 behavior / next_behavior 欄")
        return empty

    # ── 觀察頻率 ──────────────────────────────────────────────────
    if already_filtered:
        valid = event_log
    else:
        valid = event_log[
            event_log["behavior"].isin(BEHAVIORS) &
            event_log["next_behavior"].isin(BEHAVIORS)
        ].copy()
    n_seq = len(valid)

    if n_seq == 0:
        return empty

    observed: dict = {k: 0 for k in _KEYS}
    counts = valid.groupby(["behavior", "next_behavior"]).size()
    for (a, b), cnt in counts.items():
        key = f"{a}→{b}"
        if key in observed:
            observed[key] = int(cnt)

    # ── 行/列邊際統計 ──────────────────────────────────────────────
    row_total: dict = {b: 0 for b in BEHAVIORS}
    col_total: dict = {b: 0 for b in BEHAVIORS}
    for a in BEHAVIORS:
        for b in BEHAVIORS:
            cnt = observed[f"{a}→{b}"]
            row_total[a] += cnt
            col_total[b] += cnt

    N = n_seq

    # ── 期望值 ──────────────────────────────────────────────────────
    expected: dict = {}
    for a in BEHAVIORS:
        for b in BEHAVIORS:
            key = f"{a}→{b}"
            e = (row_total[a] * col_total[b]) / N if N > 0 else 0.0
            expected[key] = round(e, 4)

    # ── Z-score（Adjusted Residuals）─────────────────────────────────
    z_score: dict = {}
    significant: dict = {}
    for a in BEHAVIORS:
        for b in BEHAVIORS:
            key = f"{a}→{b}"
            o = observed[key]
            e = expected[key]

            if e <= 0:
                z_score[key]     = None
                significant[key] = False
                continue

            row_prop = row_total[a] / N
            col_prop = col_total[b] / N

            denom_sq = e * (1.0 - row_prop) * (1.0 - col_prop)
            if denom_sq <= 0:
                z_score[key]     = None
                significant[key] = False
                continue

            z = (o - e) / math.sqrt(denom_sq)
            z_score[key]     = round(z, 4)
            significant[key] = abs(z) > _Z_SIG_POS

    behavior_totals = {b: row_total[b] for b in BEHAVIORS}

    return {
        "observed":        observed,
        "expected":        expected,
        "z_score":         z_score,
        "significant":     significant,
        "n_sequences":     n_seq,
        "behavior_totals": behavior_totals,
    }

def compute_lsa_all_groups(event_log: pd.DataFrame, pass_set: set) -> dict:
    """
    分三組（all / pass / fail）各自計算轉移矩陣。
    每組 groupData 附加 n_students（不重複 anon_id 數），供前端顯示人數資訊。
    """
    if event_log.empty or "anon_id" not in event_log.columns:
        empty = _empty_matrix()
        return {"all": empty, "pass": empty, "fail": empty}

    pass_mask = event_log["anon_id"].astype(str).isin(pass_set)
    groups = {
        "all":  event_log,
        "pass": event_log[pass_mask].reset_index(drop=True),
        "fail": event_log[~pass_mask].reset_index(drop=True),
    }
    # 全體不重複學生數（作為佔比分母）
    total_students = int(event_log["anon_id"].nunique())

    results = {}
    for name, grp in groups.items():
        mat = compute_transition_matrix(grp)
        n = mat.get("n_sequences", 0)
        n_stu = int(grp["anon_id"].nunique()) if not grp.empty else 0
        mat["n_students"]       = n_stu
        mat["total_students"]   = total_students   # 同組母群總數（all 的 n_students）
        print(f"  [10] LSA {name}：{n} 序列對")
        results[name] = mat

    return results

def compute_lsa_by_semester(event_log: pd.DataFrame, pass_set: set) -> dict:
    """
    依學期分組，每學期各出 all/pass/fail 三組。
    """
    if event_log is None or event_log.empty or "semester" not in event_log.columns:
        return {}
    result = {}
    for sem, grp in event_log.groupby("semester"):
        if not sem:
            continue
        result[str(sem)] = compute_lsa_all_groups(grp.reset_index(drop=True), pass_set)
    return result

def compute_lsa_by_cluster(event_log: pd.DataFrame, pass_set: set) -> dict:
    """
    依資源使用分群分組，每分群各出 all/pass/fail 三組。
    輸出 key 保留原始前綴（R1–R5 或舊版 P1–P5），不做 R→S 換牌。
    前端 CLUSTER_NAMES 以 R 前綴為 key 顯示資源分群標籤。

    v1.2.0 FIX-Q1：移除 _normalize_cluster_key() 換牌，by_cluster key 保持 R 前綴，
    與 by_lsa_type 的 S 前綴不再衝突，兩個 dropdown 語意清晰分離。
    """
    if event_log is None or event_log.empty or "cluster" not in event_log.columns:
        return {}

    result = {}
    for cluster, grp in event_log.groupby("cluster"):
        if not cluster:
            continue
        # FIX-Q1：直接使用原始 key（R1–R5），不換牌
        r_key = str(cluster)
        result[r_key] = compute_lsa_all_groups(grp.reset_index(drop=True), pass_set)
        print(f"  [10] LSA by_cluster {r_key}")

    return result

# ─────────────────────────────────────────────────────────────
# FEAT-LSA-TYPE：真正的 S 分群邏輯
# ─────────────────────────────────────────────────────────────

def _compute_student_matrix(student_log: pd.DataFrame) -> dict:
    """
    計算單一學生的轉移矩陣（結構同 compute_transition_matrix）。
    序列對數 < 5 時回傳空矩陣（樣本不足，Z 不可靠）。

    [效能優化 0913] already_filtered=True：見 compute_transition_matrix()
    參數說明——本函式唯一呼叫端（compute_lsa_by_lsa_type() Pass 2）保證
    傳入的 student_log 已是從 valid_log（behavior/next_behavior 皆已
    篩選過）groupby 出來的子集，可安全跳過內部重複篩選。
    """
    if len(student_log) < 5:
        return _empty_matrix()
    return compute_transition_matrix(student_log, already_filtered=True)

def classify_student_lsa(
    mat: dict,
    active_days: int,
    median_days: float,
    n_events: int,
    q1_events: float,
    stability_cv: "float | None",
) -> str:
    """
    依序列特徵將單一學生分為 S1–S5。

    優先順序：S5 > S4 > S1 > S2 > S3

    Parameters
    ----------
    mat           : _compute_student_matrix() 的輸出
    active_days   : 個人在 event_log 中出現的不重複日期數
    median_days   : 同學期所有學生的 active_days 中位數
    n_events      : 個人序列對總數（mat["n_sequences"]）
    q1_events     : 同學期 n_events 的下四分位數（Q1）
    stability_cv  : 週學習時間變異係數（None = 資料不足，視同高波動）

    Returns
    -------
    "S1" | "S2" | "S3" | "S4" | "S5"
    """
    z  = mat.get("z_score", {})
    bt = mat.get("behavior_totals", {})
    obs = mat.get("observed", {})

    z_mq = z.get("M→Q")
    z_qm = z.get("Q→M")

    # ── S5：異常序列（|Z| > _Z_EXTREME 且方向不一致）──
    # 方向不一致 = Z_MQ 與 Z_QM 異號（一個極端偏好，另一個極端迴避）
    if (z_mq is not None and z_qm is not None and
            abs(z_mq) > _Z_EXTREME and abs(z_qm) > _Z_EXTREME and
            z_mq * z_qm < 0):
        return "S5"

    # ── S4：低頻低效（活動天數 < 中位數 且 事件數 < Q1）──
    if active_days < median_days and n_events < q1_events:
        return "S4"

    # ── p(M→M)：連續閱讀率（代理「轉移熵」）──
    m_total = bt.get("M", 0)
    mm_obs  = obs.get("M→M", 0)
    p_mm    = (mm_obs / m_total) if m_total > 0 else 0.5  # 無 M 行為時視同中等

    any_sig_pos = any(v is not None and v > _Z_SIG_POS for v in [z.get("M→M"), z.get("Q→Q")])

    # ── S1：穩定高效 ──
    # v1.3.0 THRESHOLD-ADJ：p_mm 上限 0.60 → _P_MM_S1S2_SPLIT
    # 活動天數 >= 中位數 且 p_mm < _P_MM_S1S2_SPLIT 且 Z(M→M) 或 Z(Q→Q) > +_Z_SIG_POS
    if (active_days >= median_days and
            p_mm < _P_MM_S1S2_SPLIT and
            any_sig_pos):
        return "S1"

    # ── S2：規律中效 ──
    # v1.3.0 THRESHOLD-ADJ：範圍 → [_P_MM_S1S2_SPLIT, _P_MM_S2_MAX]，移除 max|Z| < 1.96 條件
    # 活動天數 >= 中位數 且 _P_MM_S1S2_SPLIT <= p_mm <= _P_MM_S2_MAX
    if (active_days >= median_days and
            _P_MM_S1S2_SPLIT <= p_mm <= _P_MM_S2_MAX):
        return "S2"

    # ── S3：波動中效（預設）──
    # 含：p_mm > 0.95（過度固著）或其餘未分類。
    # stability_cv 保留為介面參數，預留未來加入 CV > 0.5 精細分支之用。
    return "S3"

def compute_lsa_by_lsa_type(
    event_log: pd.DataFrame,
    pass_set: set,
    students: "list[dict] | None" = None,
) -> "tuple[dict, dict, dict]":
    """
    依真正的序列行為特徵將學生分為 S1–S5（by_lsa_type），
    每分群各出 all/pass/fail 三組轉移矩陣。

    此輸出與 by_cluster 並存，語意不同：
      by_cluster[Rx] = 「使用資源型態 Rx 的學生」之序列（key R1–R5）
      by_lsa_type[Sx] = 「序列行為符合 Sx 特徵」的學生之序列（key S1–S5）

    v6修正（決議10）：所有以 anon_id 為單一鍵的分組/查表，全部改為
    (anon_id, semester) 複合鍵。根因：純anon_id分組時，重修生跨學期的
    S1-S5分類/active_days/n_events會混血——即使兩學期序列行為模式截然
    不同，也會被合併計算成同一組統計量，算出的S分類/轉移比例對任一
    學期都不準確。此函式的輸出（student_lsa_type/transition_ratios）
    直接寫入 11_cross_analysis.py/12_early_warning.py 的核心 records，
    污染範圍不限於本模組。

    Parameters
    ----------
    event_log : build_event_log() 輸出，須含 anon_id / behavior /
                next_behavior / timestamp / semester 欄
    pass_set  : 及格學生 anon_id 集合
    students  : behavior.json 的 students 陣列（用於取 stability_cv）；
                為 None 時 stability_cv 全部視為 None

    Returns
    -------
    tuple[
        dict: { "S1": {"all":{...},"pass":{...},"fail":{...}}, ..., "S5": {...} },
        dict: { composite_key(anon_id,semester): "S1"|...|"S5" }  # student_lsa_type
        dict: { composite_key(anon_id,semester): {"MM_ratio","MQ_ratio","QM_ratio","QQ_ratio"} }
    ]
    鍵格式為 "anon_id||semester" 字串（見 composite_key()），非 tuple——
    這兩個 dict 會被直接寫入 lsa_transition.json，tuple key 無法 JSON 序列化。
    FIX-Q2：student_lsa_type 不混入矩陣 dict，避免 null key 問題。
    """
    if event_log is None or event_log.empty or "anon_id" not in event_log.columns:
        return {}, {}, {}
    if "semester" not in event_log.columns:
        print("  ⚠️  [10] event_log 缺少 semester 欄，無法建立複合鍵，回傳空結果")
        return {}, {}, {}

    # ── 建立 stability_cv 查找表（v6：(anon_id,semester) 複合鍵）──
    cv_map: dict = {}
    if students:
        for s in students:
            aid = str(s.get("anon_id", ""))
            sem = str(s.get("semester", ""))
            cv  = s.get("features", {}).get("stability_cv")
            if aid:
                cv_map[(aid, sem)] = cv

    # ── 逐學生逐學期統計：active_days（來自完整 event_log）──
    # n_events 將從 valid_log（M/Q 序列對）取得，確保 q1_events 基準與
    # classify_student_lsa 收到的 n_events（mat["n_sequences"]）語意一致。
    student_stats: dict = {}
    for (aid, sem), grp in event_log.groupby(["anon_id", "semester"]):
        aid, sem = str(aid), str(sem)
        n_active = grp["timestamp"].dt.date.nunique() if "timestamp" in grp.columns else 0
        student_stats[(aid, sem)] = {
            "semester":    sem,
            "active_days": n_active,
            "n_events":    0,          # 由下方 valid_log 迴圈回填（M/Q 序列對數）
            "cv":          cv_map.get((aid, sem)),
        }

    # ── 逐學生逐學期計算個人轉移矩陣並分群 ──
    # 先用 valid_log 回填 n_events，再計算學期基準（median_days/q1_events）
    student_lsa_type: dict = {}
    transition_ratios: dict = {}
    valid_log = event_log[
        event_log["behavior"].isin(BEHAVIORS) &
        event_log["next_behavior"].isin(BEHAVIORS)
    ]

    # Pass 1: 回填每位學生每學期的 n_events（M/Q 序列對數）
    for (aid, sem), grp in valid_log.groupby(["anon_id", "semester"]):
        key = (str(aid), str(sem))
        if key in student_stats:
            student_stats[key]["n_events"] = len(grp)

    # 依學期分組計算基準統計（active_days 中位數、n_events Q1）
    sem_groups: dict = {}
    for key, st in student_stats.items():
        sem_groups.setdefault(st["semester"], []).append(key)

    sem_median_days: dict = {}
    sem_q1_events:   dict = {}
    for sem, keys in sem_groups.items():
        days_vals   = [student_stats[k]["active_days"] for k in keys]
        events_vals = [student_stats[k]["n_events"]    for k in keys]
        sem_median_days[sem] = float(np.median(days_vals))           if days_vals   else 0.0
        sem_q1_events[sem]   = float(np.percentile(events_vals, 25)) if events_vals else 0.0

    # Pass 2: 計算個人轉移矩陣、transition_ratios、S 分群（逐 (anon_id,semester)）
    for (aid, sem), grp in valid_log.groupby(["anon_id", "semester"]):
        key  = (str(aid), str(sem))
        st   = student_stats.get(key, {})
        mat  = _compute_student_matrix(grp.reset_index(drop=True))

        bt, obs = mat.get("behavior_totals", {}), mat.get("observed", {})
        m_total, q_total = bt.get("M", 0), bt.get("Q", 0)
        transition_ratios[key] = {
            "MM_ratio": round(obs.get("M→M", 0) / m_total, 4) if m_total > 0 else 0.0,
            "MQ_ratio": round(obs.get("M→Q", 0) / m_total, 4) if m_total > 0 else 0.0,
            "QM_ratio": round(obs.get("Q→M", 0) / q_total, 4) if q_total > 0 else 0.0,
            "QQ_ratio": round(obs.get("Q→Q", 0) / q_total, 4) if q_total > 0 else 0.0,
        }

        lsa_type = classify_student_lsa(
            mat          = mat,
            active_days  = st.get("active_days", 0),
            median_days  = sem_median_days.get(st.get("semester", ""), 0.0),
            n_events     = mat.get("n_sequences", 0),
            q1_events    = sem_q1_events.get(st.get("semester", ""), 0.0),
            stability_cv = st.get("cv"),
        )
        student_lsa_type[key] = lsa_type

    # 序列對數不足（< 5）的學生歸入 S4
    for key in student_stats:
        if key not in student_lsa_type:
            student_lsa_type[key] = "S4"

    print("  [10] LSA by_lsa_type 分群結果：")
    for code, label in LSA_TYPE_LABELS.items():
        cnt = sum(1 for v in student_lsa_type.values() if v == code)
        print(f"    {code} {label}: {cnt} 名（人次，含重修生跨學期各自計）")

    # ── 依 S 分群切分 event_log，計算各群轉移矩陣 ──
    # 為 event_log 加上 lsa_type 欄（v6：逐列用 (anon_id,semester) 複合鍵查，
    # 不再只用 anon_id 單鍵——單鍵查詢在重修生身上會不分學期套用同一個S分類）
    _aid_col = event_log["anon_id"].astype(str)
    _sem_col = event_log["semester"].astype(str)
    lsa_type_series = pd.Series(
        [student_lsa_type.get((a, s), "S4") for a, s in zip(_aid_col, _sem_col)],
        index=event_log.index,
    )
    log_with_type = event_log.assign(lsa_type=lsa_type_series)

    result: dict = {}
    for code in LSA_TYPE_LABELS:
        grp = log_with_type[log_with_type["lsa_type"] == code].reset_index(drop=True)
        result[code] = compute_lsa_all_groups(grp, pass_set)
        n = result[code].get("all", {}).get("n_sequences", 0)
        print(f"  [10] LSA by_lsa_type {code}（{LSA_TYPE_LABELS[code]}）：{n} 序列對")

    # FIX-Q2：student_lsa_type 與 label 不寫入 by_lsa_type（避免 null key 混入矩陣結構）
    # lms_etl.py 可從 return value 拿 student_lsa_type 另存至 lsa_transition["lsa_type_map"]
    #
    # v6：輸出前把內部用的 (anon_id,semester) tuple key 轉成 JSON-safe 字串鍵
    # （composite_key()，見檔案開頭），因為這兩個 dict 會被 07_export_json.py
    # 直接寫入 lsa_transition.json；tuple key 會讓 json.dump 直接 TypeError。
    student_lsa_type_out  = {composite_key(a, s): v for (a, s), v in student_lsa_type.items()}
    transition_ratios_out = {composite_key(a, s): v for (a, s), v in transition_ratios.items()}
    return result, student_lsa_type_out, transition_ratios_out

# ── 工具函式 ─────────────────────────────────────────────────────

def _empty_matrix() -> dict:
    return {
        "observed":        {k: 0     for k in _KEYS},
        "expected":        {k: 0.0   for k in _KEYS},
        "z_score":         {k: None  for k in _KEYS},
        "significant":     {k: False for k in _KEYS},
        "n_sequences":     0,
        "behavior_totals": {b: 0 for b in BEHAVIORS},
    }
