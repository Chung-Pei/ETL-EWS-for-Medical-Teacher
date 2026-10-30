"""
lms_etl/modules/11_cross_analysis.py
================================================================
篩選訓練集記錄：edu_type='theory'，且排除尚未有 final 成績的驗證學期。
- practicum 理論上已於上游 06_merge_grades.extract_grade_table 排除，
  此處僅作二次防呆確認（見 build_records_from_memory 的 prac_count 統計）
- incomplete_semesters（如 1142）另由 v3 邏輯獨立驗證，不納入訓練集
- 輸入：correlation_matrix.json（scatter_data + lsa_transition.lsa_type_map）
- 輸出：cross_analysis.json

schema_version: "1.4"

v1.4（/systematic-debugging 穿透式審查，v7.15.10）：is_4tech 學制別特徵
改直接採用 06_merge_grades.build_scatter_data() 依 sheet_name 正確算好
的值（取代原本本檔案獨立、僅憑「學期末碼」猜測學制的 fallback
heuristic）。該 heuristic 對絕大多數正規班級恰好正確，但對同一學期
檔案內混雜的重修班/遠距班會誤判（真實資料驗證：1092學期「護二二R0
遠距」原本被誤標為 is_4tech=1）。詳見 CHANGELOG_v7.md v7.15.10。

v1.3（規格書 SPEC_學生學習行為分型查詢）：新增 student_classification 匯出鍵，
逐 (anon_id, semester) 複合鍵提供 trajectory / learning_approach（供 Panel C
個案分析用；R/S分群不重複匯出，前端已可由 behavior.json / lsa_type_map 取得）。
真實資料驗證時一併發現並修正 add_derived_features() 的trajectory區塊誤判
np.nan為「有值」的bug（詳見CHANGELOG_v7.md v7.15.3），除此修正外不影響其餘
既有欄位。
================================================================
"""

from __future__ import annotations
import json
import logging
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.4"
PASS_THRESHOLD = 60
BAS_WEIGHTS = {"midterm": 0.35, "qmi": 0.30, "neg_passive": 0.20, "log_attempts": 0.15}
R_ORDER = {"R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5}
S_ORDER = {"S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5}
# 說明：S_ORDER 用來排序 S1~S5 群組的顯示與 Spearman 序位計算；
# s_cluster 為 None（序列事件不足、未分類）時不當作 "S5" 處理。
LOW_SAMPLE_THRESHOLD = 5

QMI_WEIGHTS = {"first_acc": 0.55, "fin_acc": 0.45, "neg_delta": 0.30}
PASSIVE_WEIGHTS = {"cram": 0.70, "pre_exam": 0.30}

FEATURE_VERSION = "v4"
MODEL_VERSION = "xgb_v3"

XGB_FEATURES = [
    # ── Quiz 特徵 ──
    "quz_first_attempt_accuracy", "quz_final_accuracy", "quz_score_delta",
    "quz_total_attempts", "quz_pass_rate", "quz_cramming_ratio",
    # ── Time 特徵（來自 reading_log_df）──
    "active_weeks", "consistency_score", "weekly_minutes_std",
    "cram_pattern_score", "late_night_ratio", "early_start_ratio",
    # ── Learning 特徵（Phase 1 教材完成率，Phase 1.5 後新增）──
    "total_learning_minutes", "vid_completion_rate", "aud_completion_rate",
    "txt_completion_rate", "sup_completion_rate", "tut_total_minutes",
    # ── 閱讀誠信度（v7.14，交接文件v10第0.5節「第三階段」；8學期合併
    # 驗證 n=2,675：partial correlation r=-0.041~-0.057, p<0.05，效應量
    # 偏小但顯著，改用XGBoost（樹模型＋CV正則化）納入，而非手動指定
    # BAS線性權重，以避免對小效應量權重做武斷的手動校準）──
    "reading_inflation_ratio", "policy_gaming_flag",
    # ── LSA 特徵（來自 event_log 序列分析）──
    "s_cluster_encoded", "MM_ratio", "MQ_ratio", "QM_ratio", "QQ_ratio",
    # ── 早期預警用（截斷至 Week12 的期中成績）──
    "midterm",
    # v3 student/program features.
    "is_4tech",
    "student_type_fail_rate_hist",
]

# Features that should stay untruncated in xgb_validation.json.
UNTRUNCATED_FEATURES = [
    "total_learning_minutes", "vid_completion_rate", "aud_completion_rate",
    "txt_completion_rate", "sup_completion_rate", "tut_total_minutes",
    # reading_inflation_ratio / policy_gaming_flag 與 total_learning_minutes
    # 同樣源自 reading_log_df 的累積時數計算，週截斷（partial_truncation）
    # 的適用性與 total_learning_minutes 一致，故歸入同一清單。
    "reading_inflation_ratio", "policy_gaming_flag",
]

# v6新增（決議10）：(anon_id, semester) 複合鍵的 JSON-safe 字串編碼，
# 須與 10_lsa_transition.py 的 composite_key() 使用同一個分隔符與格式，
# 兩處各自定義（本檔案不依賴 numeric-prefixed module 的動態載入）而非
# import，只是純字串組合，重複成本低，維持一致即可。
_LSA_KEY_SEP = "||"

def _lsa_key(anon_id: str, semester: str) -> str:
    """比照 10_lsa_transition.py::composite_key()，組出同格式的字串複合鍵。"""
    return f"{anon_id}{_LSA_KEY_SEP}{semester}"


def _build_student_classification(records: list[dict]) -> dict:
    """
    v1.3 新增（規格書 SPEC_學生學習行為分型查詢 §3.1）。
    依 (anon_id, semester) 複合鍵輸出逐生逐學期的 trajectory / learning_approach，
    供 Panel C 學生表現個案分析使用。
    R/S分群不重複匯出於此（前端已可由 behavior.json / lsa_type_map 取得），
    僅補上此前經 add_derived_features() 算出、卻在 compute_cross_analysis()
    匯出前被過濾丟棄的兩個衍生欄位。key 格式沿用本檔既有 _lsa_key()，
    與 lsa_type_map 同一套編碼，前端可用同一組 anon_id||semester 字串查表。
    """
    return {
        _lsa_key(r["anon_id"], r["semester"]): {
            "trajectory": r.get("trajectory"),
            "learning_approach": r.get("learning_approach"),
        }
        for r in records
    }


# ────────────────────────────────────────────────────────────────
# 1. 資料準備與篩選
# ────────────────────────────────────────────────────────────────

def build_records_from_memory(scatter_data: list[dict],
                               lsa_type_map: dict,
                               incomplete_semesters: list[str] | None = None,
                               transition_ratios: dict | None = None,
                               ) -> tuple[list[dict], dict]:
    """
    供 lms_etl.py 主流程直接呼叫（無檔案 I/O）。
    取代舊版 load_and_filter() 讀檔方式，改為記憶體內直接傳遞
    scatter_data / lsa_type_map（皆沿用 correlation_matrix.json 產出的物件）。
    篩選規則：
      - edu_type == 'theory'
      - semester not in incomplete_semesters（如 1142，尚無 final 成績、另行驗證）
    回傳 (records, meta)

    [BUG-S5-DEFAULT FIX] s_cluster 不再預設補為 "S5"，改直接沿用
    lsa_type_map 缺值時的 None：lsa_type_map 對序列事件不足、未能分類的
    學生本來就沒有對應項目，若用 .get(key, "S5") 之類寫法補預設值，
    會把「未分類」誤植為「S5（已知且屬於特定風險分群）」，讓
    by_s_cluster 統計、Spearman 相關、XGBoost 訓練都把這批學生錯算進 S5 群
    （呼叫 add_derived_features 前的 s_cluster_encoded 一併適用同一原則）。
    """
    incomplete = set(incomplete_semesters or [])
    ltm = lsa_type_map or {}

    records = []
    prac_count = 0
    s_unclassified_count = 0
    for rec in scatter_data:
        if rec.get("edu_type") != "theory":
            prac_count += 1
            continue
        if rec["semester"] in incomplete:
            continue

        aid = rec["anon_id"]
        feats = rec.get("features", {}) or {}
        sem = str(rec["semester"])
        # v6修正（決議10）：原本純 anon_id 查表，重修生跨學期的兩筆 record 會查到同一個
        # 值（不分學期），改用 _lsa_key(aid, sem) 複合鍵查詢。
        s_cluster = ltm.get(_lsa_key(aid, sem))
        if s_cluster is None:
            s_unclassified_count += 1
        trans = (transition_ratios or {}).get(_lsa_key(aid, sem), {})

        records.append({
            "anon_id": aid,
            "semester": rec["semester"],
            "r_cluster": rec["cluster"],
            "s_cluster": s_cluster,
            "midterm": rec.get("midterm_score"),
            "final": rec.get("final_score"),
            # v7.15.10：直接承接 06_merge_grades.build_scatter_data() 依
            # sheet_name 正確算好的 is_4tech；None 表示讀到 v2.1（含）以前
            # 缺此欄的舊版 correlation_matrix.json，交給 add_derived_features()
            # 內的 fallback 分支處理並警告（見該函式）。
            "is_4tech_precomputed": rec.get("is_4tech"),
            **{k: (v if v is not None else 0.0) for k, v in feats.items()},
            **{k: trans.get(k, 0.0) for k in ("MM_ratio", "MQ_ratio", "QM_ratio", "QQ_ratio")},
        })

    if prac_count > 0:
        logger.warning("[11] 過濾掉 %d 筆 practicum 記錄"
                       "（理論上應已於上游 ETL 排除，若仍出現請確認資料來源）", prac_count)

    return records, {
        "incomplete_semesters": sorted(incomplete),
        "n_practicum_excluded": prac_count,
        "n_s_cluster_unclassified": s_unclassified_count,
    }

def load_and_filter(corr_path: str) -> tuple[list[dict], dict]:
    """
    獨立執行用（CLI / 測試用），讀檔後轉呼叫記憶體版
    build_records_from_memory()（lms_etl.py 主流程使用記憶體版本，
    不會呼叫 load_and_filter 這個讀檔版本）。
    """
    with open(corr_path, encoding="utf-8") as f:
        data = json.load(f)

    meta = data["meta"]
    incomplete = meta.get("incomplete_semesters", [])
    ltm = data.get("lsa_transition", {}).get("lsa_type_map", {})

    return build_records_from_memory(data["scatter_data"], ltm, incomplete)

# ── 統計工具（共用） ──────────────────────────────────────────────

def _zscore(x: float, mean: float, std: float) -> float:
    """Return a z-score; return 0 when std is zero."""
    return (x - mean) / std if std > 0 else 0.0

def _spearman(pairs, note):
    valid_pairs = [(x, y) for x, y in pairs if x is not None and y is not None and not np.isnan(x) and not np.isnan(y)]
    if len(valid_pairs) < 3:
        return {"rho": None, "p": None, "n": len(valid_pairs), "note": "insufficient sample"}
    xs, ys = zip(*valid_pairs)
    rho, p = stats.spearmanr(xs, ys)
    return {"rho": round(float(rho), 4), "p": float(p), "n": len(valid_pairs), "note": note}

# ────────────────────────────────────────────────────────────────
# 2. 共用統計與計算函式
# ────────────────────────────────────────────────────────────────

def _safe_qmi_passive_stats(with_final: list, include_log_attempts: bool = False) -> dict:
    """
    v6決議6：qmi=None混入with_final時，np.array()若不指定dtype會變成
    object dtype，導致.mean()/.std()直接TypeError當機（已於python互動環境
    實測驗證，見除錯報告v7 8.5節）。統一用這個函式做安全的統計量計算，
    取代所有裸建陣列的寫法（含本檔案 add_derived_features 與
    compute_bas_validation 兩處、以及 12_early_warning.py:compute_warning
    共三處，避免同一個bug分裂成多份獨立實作、各自修一次）。

    一併算 mid_mean/mid_std（雖然 midterm 本身已用 or 0.0 處理過 None，
    無崩潰風險，但統一放這裡可以少一個「呼叫端自己要記得另外算」的步驟）；
    include_log_attempts=True 時一併算 att_mean/att_std（訓練集4項BAS公式
    需要，只有 add_derived_features 這個呼叫端會用到）。

    ⚠️ 僅適用於「訓練集」（11_cross_analysis.py 的 with_final）。
    目標學期（12_early_warning.py:add_qmi_bas）的 mid_mean/mid_std 依
    決議8必須用 semester-local 基準，且目標學期 records 在呼叫當下還沒有
    自己的 qmi 值（要靠 compute_qmi_passive_bas() 這次呼叫才產生），
    不能也不該呼叫本函式，該處自行組裝對應的 stats dict。
    """
    qmi_raw = np.array([r.get("qmi") for r in with_final], dtype=float)
    pas_raw = np.array([r.get("passive_index") for r in with_final], dtype=float)
    mid_raw = np.array(
        [r.get("midterm") if r.get("midterm") is not None else 0.0 for r in with_final],
        dtype=float,
    )
    qmi_valid = ~np.isnan(qmi_raw)
    pas_valid = ~np.isnan(pas_raw)
    stats_out = {
        "qmi_mean": float(np.nanmean(qmi_raw)) if qmi_valid.any() else 0.0,
        "qmi_std":  float(np.nanstd(qmi_raw))  if qmi_valid.any() else 1.0,
        "pas_mean": float(np.nanmean(pas_raw)) if pas_valid.any() else 0.0,
        "pas_std":  float(np.nanstd(pas_raw))  if pas_valid.any() else 1.0,
        "mid_mean": float(mid_raw.mean()) if len(mid_raw) else 0.0,
        "mid_std":  float(mid_raw.std())  if len(mid_raw) else 1.0,
    }
    # std=0 會導致下游 _zscore 除以零；_zscore() 本身已對 std<=0 回傳0.0防呆，
    # 但這裡仍給 1.0 的保守 fallback，語意更清楚（僅在 with_final 長度<1或
    # 全體同值時才會踩到，理論上不常發生）
    if stats_out["qmi_std"] == 0.0:
        stats_out["qmi_std"] = 1.0
    if stats_out["pas_std"] == 0.0:
        stats_out["pas_std"] = 1.0
    if stats_out["mid_std"] == 0.0:
        stats_out["mid_std"] = 1.0
    if include_log_attempts:
        att_raw = np.array(
            [np.log1p(r.get("quz_total_attempts", 0.0) or 0.0) for r in with_final],
            dtype=float,
        )
        stats_out["att_mean"] = float(att_raw.mean()) if len(att_raw) else 0.0
        stats_out["att_std"]  = float(att_raw.std())  if len(att_raw) else 1.0
        if stats_out["att_std"] == 0.0:
            stats_out["att_std"] = 1.0
    return stats_out


def compute_qmi_passive_bas(
    record: dict,
    stats: dict,             # {"qmi_mean","qmi_std","pas_mean","pas_std","mid_mean","mid_std",
                              #  可選"att_mean","att_std"}
    qmi_weights: dict, passive_weights: dict, bas_weights: dict,
    include_log_attempts: bool,
) -> dict:
    """
    計算單一 record 的 quiz_mastery_index (QMI) / passive_index / bas_score。
    供 add_derived_features()（訓練集，include_log_attempts=True）與
    12_early_warning.py:add_qmi_bas()（目標學期，include_log_attempts=False）
    共用，取代原本兩處重複的迴圈邏輯。

    決議3+6+7整合：
    - QMI 公式沿用既有設計不變：
        QMI = first_acc*QMI_WEIGHTS["first_acc"] + fin_acc*QMI_WEIGHTS["fin_acc"]
              - max(delta,0)*QMI_WEIGHTS["neg_delta"]
    - total_attempts==0（真正沒作答，非0分）時 qmi=None，不再用 or 0.0
      硬湊出一個假的0分（原本的隱藏bug：total_attempts=0時first_acc/fin_acc/
      delta皆為None，經 or 0.0 後QMI算出0.0，跟「認真作答但考差」的學生
      混在同一個QMI值，是1142 HIGH=0異常的根因之一）。
    - qmi=None 時 bas_score 走 renormalize（拿掉 z_qmi 項，依
      include_log_attempts 決定剩餘2或3項依比例重算），標記
      W-INSUFFICIENT-QUIZ-DATA。
    - cram/pre_exam（passive_index 兩個分量）維持 or 0.0（既有設計不動，
      這兩個欄位在 total_attempts=0 時本來就有意義的 0，不是「缺值」）。
    """
    qmi_mean, qmi_std = stats["qmi_mean"], stats["qmi_std"]
    pas_mean, pas_std = stats["pas_mean"], stats["pas_std"]
    mid_mean, mid_std = stats["mid_mean"], stats["mid_std"]
    att_mean, att_std = stats.get("att_mean"), stats.get("att_std")

    first_acc = record.get("quz_first_attempt_accuracy")
    fin_acc   = record.get("quz_final_accuracy")
    delta     = record.get("quz_score_delta")
    cram      = record.get("quz_cramming_ratio", 0.0) or 0.0
    pre_exam  = record.get("pre_exam_intensity", 0.0) or 0.0
    total_attempts = record.get("quz_total_attempts", 0) or 0

    if total_attempts == 0:
        qmi = None
        data_flags = list(record.get("data_flags", [])) + ["W-INSUFFICIENT-QUIZ-DATA"]
    else:
        qmi = (
            (first_acc or 0.0) * qmi_weights.get("first_acc", 0)
            + (fin_acc or 0.0) * qmi_weights.get("fin_acc", 0)
            - max(delta or 0.0, 0) * qmi_weights.get("neg_delta", 0)
        )
        data_flags = list(record.get("data_flags", []))

    passive_index = (
        cram * passive_weights.get("cram", 0)
        + pre_exam * passive_weights.get("pre_exam", 0)
    )

    mid_val = record.get("midterm")
    z_mid = _zscore(mid_val if mid_val is not None else 0.0, mid_mean, mid_std)
    z_neg_passive = _zscore(-passive_index, -pas_mean, pas_std)

    if qmi is None:
        terms = {
            "midterm":     (bas_weights["midterm"], z_mid),
            "neg_passive": (bas_weights["neg_passive"], z_neg_passive),
        }
        if include_log_attempts and att_mean is not None:
            z_att = _zscore(np.log1p(total_attempts), att_mean, att_std)
            terms["log_attempts"] = (bas_weights.get("log_attempts", 0), z_att)
        w_sum = sum(w for w, _ in terms.values())
        bas_score = sum((w / w_sum) * z for w, z in terms.values()) if w_sum > 0 else 0.0
    else:
        z_qmi = _zscore(qmi, qmi_mean, qmi_std)
        terms = [
            (bas_weights["midterm"], z_mid),
            (bas_weights["qmi"], z_qmi),
            (bas_weights["neg_passive"], z_neg_passive),
        ]
        if include_log_attempts and att_mean is not None:
            z_att = _zscore(np.log1p(total_attempts), att_mean, att_std)
            terms.append((bas_weights.get("log_attempts", 0), z_att))
        bas_score = sum(w * z for w, z in terms)   # 4項/3項權重本身已合計1.0，不需renormalize

    record["qmi"] = qmi
    record["passive_index"] = passive_index
    record["bas_score"] = bas_score
    record["data_flags"] = data_flags
    return record


def add_derived_features(records: list[dict]) -> list[dict]:
    """
    衍生：quiz_mastery_index (QMI), passive_index, bas_score,
          learning_approach (DEEP/SURFACE/MODERATE),
          trajectory (SS/FS/SF/FF，需有 midterm 才能算)

    BAS v3 公式（data-driven grid search, r=0.479）：
      QMI          = first_acc*QMI_WEIGHTS["first_acc"] + final_acc*QMI_WEIGHTS["fin_acc"] - max(delta,0)*QMI_WEIGHTS["neg_delta"]
      passive_index = cram*PASSIVE_WEIGHTS["cram"] + pre_exam*PASSIVE_WEIGHTS["pre_exam"]
      BAS          = midterm*0.35 + z(QMI)*0.30 + z(-passive)*0.20 + z(log_att)*0.15
    BAS 依全體 records 中有 final 的子集算 z-score 基準（訓練集 pooled）。

    v6修正（決議3/6/7）：
    - QMI/passive/BAS 三者的核心計算改呼叫共用函式 compute_qmi_passive_bas()，
      取代原本本函式與 12_early_warning.py:add_qmi_bas() 各自一份的重複邏輯。
    - total_attempts==0 時 qmi=None（不再用 or 0.0 假裝成0分），bas_score
      走 renormalize 分支（決議3/7）。
    - z-score 基準統計量改呼叫 _safe_qmi_passive_stats()，內部強制
      dtype=float + isnan 過濾，修正 qmi=None 混入 with_final 時
      np.array().mean() 直接 TypeError 當機的問題（決議6，缺口A）。
    """
    # ── 第一輪：QMI / passive_index / learning_approach / trajectory 等
    #    不需要全體統計基準就能算的欄位 ──
    _is4tech_fallback_count = 0  # v7.15.10：統計退回舊heuristic的筆數，迴圈後警告
    for r in records:
        first_acc = r.get("quz_first_attempt_accuracy", 0.0) or 0.0
        delta = r.get("quz_score_delta", 0.0) or 0.0

        # learning approach（沿用既有設計不變：這裡刻意用 or 0.0 defaulted
        # 值分類，跟 QMI 本身是否為 None 是兩件事，不受決議3影響）
        if first_acc >= 0.6 and delta < 0.2:
            r["learning_approach"] = "DEEP"
        elif delta >= 0.3:
            r["learning_approach"] = "SURFACE"
        else:
            r["learning_approach"] = "MODERATE"

        # trajectory（需 midterm + final 都有值）
        # [v7.15.3修正] 原僅檢查 is not None，NaN在Python中 `np.nan is not None`
        # 為True，會被誤判為「有值」，導致fin=NaN時所有>=比較皆為False（NumPy
        # NaN比較語意），落入else分支被誤標為"FF"（而非正確的None/無法判定）。
        # 比照3行後 is_fail 既有寫法（is not None and not np.isnan(...)）修正。
        # 此bug先前完全無影響（trajectory此前只在記憶體內計算、從未匯出，見
        # student_classification新增前的compute_cross_analysis()），本次新增
        # 逐生匯出後才會被使用者實際看到，故一併修正。
        mid_valid = r["midterm"] is not None and not np.isnan(r["midterm"])
        fin_valid = r["final"] is not None and not np.isnan(r["final"])
        if mid_valid and fin_valid:
            mid, fin = r["midterm"], r["final"]
            if mid >= PASS_THRESHOLD and fin >= PASS_THRESHOLD:
                r["trajectory"] = "SS"
            elif mid < PASS_THRESHOLD and fin >= PASS_THRESHOLD:
                r["trajectory"] = "FS"
            elif mid >= PASS_THRESHOLD and fin < PASS_THRESHOLD:
                r["trajectory"] = "SF"
            else:
                r["trajectory"] = "FF"
        else:
            r["trajectory"] = None

        r["is_fail"] = int(r["final"] < PASS_THRESHOLD) if r["final"] is not None and not np.isnan(r["final"]) else None
        # Encode S cluster for model features; keep unclassified as NaN.
        r["s_cluster_encoded"] = S_ORDER.get(r.get("s_cluster"), np.nan)

        # [v3 Change A → v7.15.10 修正] 學制別特徵。
        # 原設計僅憑「semester 末碼」猜測學制（1='二技一般為主'/2='四技為主'），
        # 對本校絕大多數正規班級恰好成立（因歸檔慣例：2年制正課記錄恰好都在
        # 「-1」檔、4年制恰好都在「-2」檔），但對同一學期檔案內混雜的重修班/
        # 遠距班會誤判——例如 1092 學期的「護二二R0遠距」（重修班，非四技），
        # 純學期末碼判斷會誤標為 is_4tech=1。
        # 改為直接採用 06_merge_grades.build_scatter_data() 依 sheet_name
        # 正確算好的值（見 build_records_from_memory 的 is_4tech_precomputed）。
        # fallback 僅為向下相容：讀到 v2.1（含）以前缺此欄的舊版
        # correlation_matrix.json 時退回舊heuristic，並警告一次（見迴圈外的
        # _is4tech_fallback_count 計數器彙總，迴圈結束後統一印出警告）。
        _pre = r.pop("is_4tech_precomputed", None)
        if _pre is not None:
            r["is_4tech"] = int(_pre)
        else:
            r["is_4tech"] = 1 if r.get("semester", "")[-1] == "2" else 0
            _is4tech_fallback_count += 1

    if _is4tech_fallback_count > 0:
        logger.warning(
            "[11] %d 筆記錄的 correlation_matrix.json 缺少 is_4tech 欄"
            "（schema < 2.2 的舊版快取），已退回舊版純學期末碼判斷"
            "（對重修班/遠距班等特殊班級不精確）。建議重新執行完整 "
            "lms_etl.py pipeline 以更新快取。",
            _is4tech_fallback_count,
        )

    # ── [v3 Change C] student_type_fail_rate_hist：依學期分組計算歷史 ──
    # 重新命名自 semester_fail_rate_hist（v2），反映真正意義：反映
    # 「該學制歷史 fail rate」的 student-type fixed effect，非課程難度。
    # 108課綱改制後，1132/1142（四技）的 fail rate 顯著高於改制前，
    # 是學制固定效應而非課程本身變難。
    _sem_groups: dict[str, list] = defaultdict(list)
    for r in records:
        if r.get("final") is not None:
            _sem_groups[r["semester"]].append(int(r["final"] < PASS_THRESHOLD))
    _sem_fail_rate = {
        _sem: float(np.mean(_flags)) if _flags else np.nan
        for _sem, _flags in _sem_groups.items()
    }
    for r in records:
        r["student_type_fail_rate_hist"] = _sem_fail_rate.get(r["semester"], np.nan)

    # ── QMI / passive_index / bas_score：v6改呼叫共用函式 ──
    # 決議6修正：先算出 with_final 子集（含 z-score 基準用途，跟原本邏輯
    # 一致），但 qmi/passive_index 本身在決議3後可能是 None，先呼叫
    # compute_qmi_passive_bas() 算出全體 records 的 qmi/passive_index
    # （此時 bas_score 用暫時基準，下面再用訓練集基準重算一次）。
    #
    # 為了讓 with_final 篩選時能拿到 qmi/passive_index（part of 判斷
    # z-score 基準），採兩階段：
    #   1) 先算 qmi/passive_index（不需要全體統計量）
    #   2) 篩出 with_final 子集，用 _safe_qmi_passive_stats() 算基準
    #   3) 用該基準重新計算每筆 record 的 bas_score（z-score 依賴基準）
    _placeholder_stats = {
        "qmi_mean": 0.0, "qmi_std": 1.0, "pas_mean": 0.0, "pas_std": 1.0,
        "mid_mean": 0.0, "mid_std": 1.0, "att_mean": 0.0, "att_std": 1.0,
    }
    for r in records:
        compute_qmi_passive_bas(
            r, _placeholder_stats, QMI_WEIGHTS, PASSIVE_WEIGHTS, BAS_WEIGHTS,
            include_log_attempts=True,
        )

    with_final = [r for r in records if r["final"] is not None and not np.isnan(r["final"])]
    if len(with_final) >= 2:
        stats = _safe_qmi_passive_stats(with_final, include_log_attempts=True)
        for r in records:
            compute_qmi_passive_bas(
                r, stats, QMI_WEIGHTS, PASSIVE_WEIGHTS, BAS_WEIGHTS,
                include_log_attempts=True,
            )
    else:
        for r in records:
            r["bas_score"] = 0.0

    return records

# ────────────────────────────────────────────────────────────────
# 3. 分群統計
# ────────────────────────────────────────────────────────────────

def _trajectory_dist(sub: list[dict]) -> dict:
    traj = [r["trajectory"] for r in sub if r["trajectory"] is not None]
    total = len(traj)
    if total == 0:
        return {"SS": None, "FS": None, "SF": None, "FF": None, "n_with_midterm": 0}
    counts = defaultdict(int)
    for t in traj:
        counts[t] += 1
    return {
        "SS": round(counts["SS"] / total, 4),
        "FS": round(counts["FS"] / total, 4),
        "SF": round(counts["SF"] / total, 4),
        "FF": round(counts["FF"] / total, 4),
        "n_with_midterm": total,
    }

def _approach_dist(sub: list[dict]) -> dict:
    total = len(sub)
    if total == 0:
        return {"DEEP": None, "SURFACE": None, "MODERATE": None}
    counts = defaultdict(int)
    for r in sub:
        counts[r["learning_approach"]] += 1
    return {k: round(counts.get(k, 0) / total, 4) for k in ("DEEP", "SURFACE", "MODERATE")}

def _rescue_rate(sub: list[dict]) -> float | None:
    fs = sum(1 for r in sub if r["trajectory"] == "FS")
    ff = sum(1 for r in sub if r["trajectory"] == "FF")
    denom = fs + ff
    return round(fs / denom, 4) if denom > 0 else None

def compute_group_stats(records: list[dict], group_col: str,
                         overall_fail_rate: float, overall_n: int) -> "tuple[dict, int]":
    """
    依 group_col（'r_cluster' 或 's_cluster'）分組統計。
    n < LOW_SAMPLE_THRESHOLD 的分組標記 low_sample=True，
    不計算 fail_rate/mean 等統計量（樣本太少，數值不可靠）。
    可用於統計 R1-R5（group_col='r_cluster'）或 S1-S5（group_col='s_cluster'）
    各分群的不及格率、卡方檢定、拯救率等；該分群完全沒有樣本（n=0）時
    標記為 no_baseline（沒有任何歷史樣本可供比對）。

    修正 (2026-07-11)：原本用 `if code is not None` 過濾未分類記錄，但
    r_cluster 欄位的「未分類」sentinel 值實測是空字串 ""（見
    build_scatter_data / merge_behavior_grades：學生有成績但無任何LMS
    行為資料、未經 assign_clusters 分群時的空值），不是 None——空字串
    通過 `is not None` 檢查後被塞進 groups[""]，這個key不在 all_codes
    （R1-R5/S1-S5）中，從未被下面的 `for code in all_codes` 迴圈讀取，
    等於這些學生的記錄「靜默消失」：不出現在任何分群統計裡，也不像
    S-cluster 的 n_s_cluster_unclassified 一樣被計數，比by_semester明細
    洩漏出空字串鍵更隱蔽（後者至少會在JSON裡留下可疑痕跡）。
    改用truthy檢查（`if code:`）同時排除 None 與空字串，並回傳未分類筆數，
    供呼叫端比照 n_s_cluster_unclassified 的模式寫入 meta，不再靜默遺失。
    """
    all_codes = list(R_ORDER.keys()) if group_col == "r_cluster" else list(S_ORDER.keys())

    groups: dict[str, list[dict]] = defaultdict(list)
    n_unclassified = 0
    for r in records:
        # Skip unclassified groups (sentinel 可能是 None 或空字串，皆視為未分類).
        code = r[group_col]
        if code:
            groups[code].append(r)
        else:
            n_unclassified += 1

    result = {}
    for code in all_codes:
        sub = groups.get(code, [])
        with_final = [r for r in sub if r["final"] is not None and not np.isnan(r["final"])]
        n = len(with_final)

        if n < LOW_SAMPLE_THRESHOLD:
            result[code] = {
                "n": n,
                "n_total_records": len(sub),
                "low_sample": True,
                "no_baseline": (n == 0),
            }
            continue

        finals = np.array([r["final"] for r in with_final])
        fail_rate = float((finals < PASS_THRESHOLD).mean())

        with_mid = [r for r in with_final if r["midterm"] is not None]
        fail_rate_mid = (
            float((np.array([r["midterm"] for r in with_mid]) < PASS_THRESHOLD).mean())
            if with_mid else None
        )

        # chi-square vs overall
        n_fail = int((finals < PASS_THRESHOLD).sum())
        n_pass = n - n_fail
        overall_fail_n = round(overall_fail_rate * overall_n)
        overall_pass_n = overall_n - overall_fail_n
        rest_fail = max(overall_fail_n - n_fail, 0)
        rest_pass = max(overall_pass_n - n_pass, 0)
        try:
            chi2, p_chi2, _, _ = stats.chi2_contingency(
                [[n_fail, n_pass], [rest_fail, rest_pass]]
            )
        except ValueError:
            p_chi2 = None

        se = (overall_fail_rate * (1 - overall_fail_rate) / n) ** 0.5 if n > 0 else 0
        z = (fail_rate - overall_fail_rate) / se if se > 0 else None

        result[code] = {
            "n": n,
            "n_total_records": len(sub),
            "final_mean": round(float(finals.mean()), 2),
            "final_sd": round(float(finals.std()), 2),
            "fail_rate_final": round(fail_rate, 4),
            "fail_rate_midterm": round(fail_rate_mid, 4) if fail_rate_mid is not None else None,
            "z_vs_overall_final": round(z, 3) if z is not None else None,
            "p_chi2": round(float(p_chi2), 4) if p_chi2 is not None else None,
            "trajectory": _trajectory_dist(with_final),
            "approach": _approach_dist(with_final),
            "rescue_rate": _rescue_rate(with_final),
            "low_sample": False,
            "no_baseline": False,
        }
    return result, n_unclassified

# ────────────────────────────────────────────────────────────────
# 4. R×S 交叉分析
# ────────────────────────────────────────────────────────────────

def compute_cross_matrix(records: list[dict],
                          overall_fail_rate: float, overall_n: int,
                          with_final: "list[dict] | None" = None) -> dict:
    """R×S 交叉矩陣統計。
    with_final 由呼叫端預先過濾傳入以避免重複計算，未傳入時退回自行過濾 records。
    """
    if with_final is None:
        with_final = [r for r in records if r["final"] is not None and not np.isnan(r["final"])]
    matrix: dict[str, dict] = {}

    for rg in R_ORDER:
        matrix[rg] = {}
        for sg in ["S1", "S2", "S3", "S4", "S5"]:
            sub = [r for r in with_final if r["r_cluster"] == rg and r["s_cluster"] == sg]
            n = len(sub)

            # Mark low-sample R/S cells without running significance checks.
            if n < LOW_SAMPLE_THRESHOLD:
                matrix[rg][sg] = {"n": n, "low_sample": True}
                continue

            finals = np.array([r["final"] for r in sub])
            fail_rate = float((finals < PASS_THRESHOLD).mean())
            se = (overall_fail_rate * (1 - overall_fail_rate) / n) ** 0.5
            z = (fail_rate - overall_fail_rate) / se if se > 0 else None

            matrix[rg][sg] = {
                "n": n,
                "final_mean": round(float(finals.mean()), 2),
                "final_sd": round(float(finals.std()), 2),
                "fail_rate_final": round(fail_rate, 4),
                "z_vs_overall_final": round(z, 3) if z is not None else None,
                "low_sample": False,
            }
    return matrix

# ────────────────────────────────────────────────────────────────
# 5. Spearman
# ────────────────────────────────────────────────────────────────

def compute_spearman(records: list[dict],
                     with_final: "list[dict] | None" = None) -> dict:
    """Spearman r：R群/S群序位 vs final score。
    with_final 由呼叫端預先過濾傳入以避免重複計算，未傳入時退回自行過濾 records。
    """
    if with_final is None:
        with_final = [r for r in records if r["final"] is not None and not np.isnan(r["final"])]

    # Build Spearman pairs only from classified R/S clusters.
    r_pairs = [(R_ORDER[r["r_cluster"]], r["final"]) for r in with_final
               if r.get("r_cluster") in R_ORDER]
    s_pairs = [(S_ORDER[r["s_cluster"]], r["final"]) for r in with_final
               if r.get("s_cluster") in S_ORDER]

    return {
        "r_group_vs_final": _spearman(r_pairs, "R cluster ordinal vs final score"),
        "s_group_vs_final": _spearman(s_pairs, "S cluster ordinal vs final score"),
    }

# ────────────────────────────────────────────────────────────────
# 6. BAS / QMI 驗證
# ────────────────────────────────────────────────────────────────

def _pearson_stat(with_final: list, finals, arr_key: str):
    """Pearson r of records[arr_key] vs final scores."""
    arr = np.array([r.get(arr_key) for r in with_final], dtype=float)
    valid_mask = ~(np.isnan(arr) | np.isnan(finals))
    if valid_mask.sum() < 3:
        return {"r": None, "p": None}
    arr_valid = arr[valid_mask]
    fin_valid = finals[valid_mask]
    if arr_valid.std() == 0:
        return {"r": None, "p": None}
    rval, pval = stats.pearsonr(arr_valid, fin_valid)
    return {"r": round(float(rval), 4), "p": round(float(pval), 4)}

def _compute_per_type_qmi_thresholds(with_final: list[dict]) -> dict:
    """Compute per-student-type QMI percentile thresholds."""
    _MIN_N_TYPE = 10
    qmi_4_raw = np.array([r.get("qmi") for r in with_final if r.get("is_4tech") == 1], dtype=float)
    qmi_2_raw = np.array([r.get("qmi") for r in with_final if r.get("is_4tech") == 0], dtype=float)
    qmi_4 = qmi_4_raw[~np.isnan(qmi_4_raw)]
    qmi_2 = qmi_2_raw[~np.isnan(qmi_2_raw)]

    def _pct(arr: np.ndarray, p: int):
        if len(arr) < _MIN_N_TYPE:
            return None
        return round(float(np.percentile(arr, p)), 4)

    return {
        "qmi_q1_4tech": _pct(qmi_4, 20),
        "qmi_q2_4tech": _pct(qmi_4, 40),
        "qmi_q1_2tech": _pct(qmi_2, 20),
        "qmi_q2_2tech": _pct(qmi_2, 40),
    }


# v6複查（決議6）：此函式的 dtype=float + isnan 寫法本來就正確
# （見除錯報告v7 8.5節「對照組」），維持不動；qmi_arr 同時需要與
# finals 的聯合有效遮罩（qmi_valid），跟 _safe_qmi_passive_stats()
# 回傳的純聚合統計量（mean/std）用途不同，故不改呼叫共用函式。
def compute_bas_validation(records: list[dict],
                            with_final: "list[dict] | None" = None) -> dict:
    """BAS 驗證統計：Pearson r、QMI 分位對應不及格率等。
    with_final 由呼叫端預先過濾傳入以避免重複計算，未傳入時退回自行過濾 records。
    """
    if with_final is None:
        with_final = [r for r in records if r["final"] is not None and not np.isnan(r["final"])]
    finals = np.array([r["final"] for r in with_final])

    qmi_raw = np.array([r.get("qmi") for r in with_final], dtype=float)
    qmi_valid = ~(np.isnan(qmi_raw) | np.isnan(finals))
    qmi_arr = qmi_raw[qmi_valid]
    finals_valid = finals[qmi_valid]

    if len(qmi_arr) > 0:
        bounds = np.percentile(qmi_arr, [20, 40, 60, 80])
        edges = [qmi_arr.min()] + list(bounds) + [qmi_arr.max()]
    else:
        edges = []
        bounds = []

    quintiles = []
    if len(edges) >= 6:
        for i in range(5):
            lo, hi = edges[i], edges[i + 1]
            if i < 4:
                mask = (qmi_arr >= lo) & (qmi_arr < hi)
            else:
                mask = (qmi_arr >= lo) & (qmi_arr <= hi)
            n = int(mask.sum())
            fail_rate = float((finals_valid[mask] < PASS_THRESHOLD).mean()) if n > 0 else None
            quintiles.append({
                "q": i + 1,
                "range": [round(float(lo), 4), round(float(hi), 4)],
                "n": n,
                "fail_rate": round(fail_rate, 4) if fail_rate is not None else None,
            })
    else:
        quintiles = [{"q": i+1, "range": [0,0], "n": 0, "fail_rate": None} for i in range(5)]

    return {
        "quz_first_acc_r": _pearson_stat(with_final, finals, "quz_first_attempt_accuracy"),
        "qmi_r": _pearson_stat(with_final, finals, "qmi"),
        "bas_r": _pearson_stat(with_final, finals, "bas_score"),
        "bas_weights": BAS_WEIGHTS,
        "weight_basis": "data-driven grid search on theory training set (v3: +midterm +fin_acc +log_attempts)",
        "qmi_quintiles": quintiles,
        **_compute_per_type_qmi_thresholds(with_final),
    }

def compute_bas_auc(with_final: list[dict]) -> dict:
    """
    計算 BAS 的 AUC，使 BAS 與 XGBoost 可在同一指標框架下比較。
    BAS 分數高 = 學習狀況佳 = 低風險，故取負號後計算 AUC
    （高分預測不及格，取負使「高值 = 預測不及格」符合 roc_auc_score 慣例）。
    比照 compute_bas_validation() 傳入 with_final，避免重複過濾。
    """
    from sklearn.metrics import roc_auc_score as _roc_auc

    valid = [
        r for r in with_final
        if r.get("bas_score") is not None
        and not np.isnan(r.get("bas_score", float("nan")))
        and r.get("is_fail") is not None
    ]
    if len(valid) < 10:
        return {"status": "insufficient_data", "n": len(valid)}

    y_true  = [int(r["is_fail"]) for r in valid]
    y_score = [-r["bas_score"] for r in valid]  # 取負：BAS 低分 = 高風險

    try:
        auc = float(_roc_auc(y_true, y_score))
        return {"auc": round(auc, 4), "n": len(valid)}
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}

# ────────────────────────────────────────────────────────────────
# 6b. XGBoost 訓練與驗證校準
# ────────────────────────────────────────────────────────────────

def _compute_xgb_thresholds(records: list[dict], y: "np.ndarray",
                             proba: "np.ndarray") -> dict:
    """Compute per-type XGBoost risk thresholds from CV probabilities."""
    from sklearn.metrics import precision_score as _prec

    is_4tech = np.array([r.get("is_4tech", 0) for r in records])

    # 逐一嘗試不同閾值 t，取 precision 最接近目標值者，作為 HIGH 風險分界點
    # MEDIUM threshold is HIGH threshold minus a fixed offset.
    _MEDIUM_OFFSET = 0.15

    def _calibrate_type(mask: np.ndarray, target_precision: float) -> dict:
        y_sub = y[mask]
        p_sub = proba[mask]
        n = int(mask.sum())
        if n < 10 or y_sub.sum() < 2:
            return {"high": 0.80, "medium": 0.60, "high_precision": None, "n": n,
                    "note": "insufficient_samples_fallback"}

        best_t, best_diff = 0.80, float("inf")
        for t_int in range(30, 86):    # 0.30 ~ 0.85，間距 0.01
            t = t_int / 100.0
            flagged = (p_sub >= t).astype(int)
            if flagged.sum() == 0:
                break
            prec = float(_prec(y_sub, flagged, zero_division=0))
            diff = abs(prec - target_precision)
            if diff < best_diff:
                best_diff, best_t = diff, t

        high_t = round(best_t, 2)
        med_t  = round(max(0.30, high_t - _MEDIUM_OFFSET), 2)
        flagged_best = (p_sub >= high_t).astype(int)
        best_prec = float(_prec(y_sub, flagged_best, zero_division=0))
        return {
            "high": high_t,
            "medium": med_t,
            "high_precision": round(best_prec, 4),
            "n": n,
        }

    # Build per-type masks for threshold calibration.
    mask_4 = is_4tech == 1
    mask_2 = is_4tech == 0

    target_4 = float(y[mask_4].mean()) if mask_4.sum() >= 2 else 0.42
    target_2 = float(y[mask_2].mean()) if mask_2.sum() >= 2 else 0.25

    result_4 = _calibrate_type(mask_4, target_precision=target_4)
    result_2 = _calibrate_type(mask_2, target_precision=target_2)

    # Combined fallback when a per-type bucket is too small.
    combined = _calibrate_type(np.ones(len(y), dtype=bool),
                               target_precision=float(y.mean()))

    return {
        "4tech":  result_4,
        "2tech":  result_2,
        "combined": {"high": combined["high"], "medium": combined["medium"],
                     "high_precision": combined.get("high_precision")},
        "calibration_basis": "cv_proba_per_type",
        "target_precision_4tech": round(target_4, 4),
        "target_precision_2tech": round(target_2, 4),
    }


def compute_xgb(with_final: list[dict], enable_xgb: bool = True, week_limit: int = 12) -> dict:
    """Train/evaluate XGBoost and return validation metadata plus model artifacts."""
    if not enable_xgb:
        return {"status": "missing_dependency"}

    # Import optional XGBoost dependency lazily.
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return {"status": "missing_dependency"}

    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

    y = np.array([r["is_fail"] for r in with_final])
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos < 2 or n_neg < 2:
        return {"status": "insufficient_data", "n": len(with_final)}

    X_raw = np.array(
        [[r.get(f, np.nan) for f in XGB_FEATURES] for r in with_final], dtype=float
    )
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_raw)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # XGBoost parameters selected from prior validation.
    model = XGBClassifier(
        n_estimators=500, max_depth=3, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.8,
        scale_pos_weight=1.0,
        objective="binary:logistic", eval_metric="auc", random_state=42,
        verbosity=0,
    )

    try:
        proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    except ValueError:
        # covers: n_splits > n_samples-per-class, or any stratification failure
        return {"status": "insufficient_data", "n": len(with_final)}

    preds = (proba >= 0.5).astype(int)

    # 交叉驗證後用全量資料重新 fit，讓最終模型的 feature_importance 可用
    model.fit(X, y)

    # roc_auc_score can fail when only one class is present.
    try:
        auc = float(roc_auc_score(y, proba))
    except ValueError:
        auc = 0.0

    # [v3 Change E] CV probabilities 做 per-type threshold calibration
    # 作法：逐一嘗試閾值 t，取 precision(y[type], proba[type]>=t) 最接近目標值者
    # 作為 HIGH 風險分界點，結果存入 xgb_validation["thresholds"]，
    # 供 12_early_warning 讀取套用對應學制門檻
    thresholds = _compute_xgb_thresholds(with_final, y, proba)

    # ── XGB r：用 P(及格) = 1 - proba，方向對齊 BAS r ──
    # 注意：不可用 proba（P不及格），否則 r 為負值，與 BAS r 方向相反，
    #       無法直接並排比較（已實測 proba 版本 r = -0.3416）。
    finals_for_r = [r["final"] for r in with_final]
    try:
        _r_val, _p_val = stats.pearsonr(1 - proba, finals_for_r)
        xgb_r = {
            "r": round(float(_r_val), 4),
            "p": round(float(_p_val), 6),
            "n": len(with_final),
            "direction_note": "P(pass)=1-P(fail), aligned with BAS r direction",
        }
    except Exception as exc:
        xgb_r = {"status": "error", "message": str(exc)}

    return {
        "week_limit": week_limit,
        "auc": round(auc, 4),
        "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y, preds, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "feature_version": FEATURE_VERSION,
        "model_version": MODEL_VERSION,
        "training_semesters": sorted({r["semester"] for r in with_final}),
        "partial_truncation": True,
        "untruncated_features": UNTRUNCATED_FEATURES,
        "thresholds": thresholds,
        "xgb_r": xgb_r,
        "_model_artifacts": (model, imputer),
    }

# ────────────────────────────────────────────────────────────────
# 7. 按學期輸出整理
# ────────────────────────────────────────────────────────────────

def _simple_group_stats(group_dict: dict) -> dict:
    """Compute simple summary stats for each group."""
    out = {}
    for code, items in group_dict.items():
        if len(items) < LOW_SAMPLE_THRESHOLD:
            out[code] = {"n": len(items), "low_sample": True}
            continue
        fs = np.array([x["final"] for x in items])
        out[code] = {
            "n": len(items),
            "fail_rate_final": round(float((fs < PASS_THRESHOLD).mean()), 4),
            "final_mean": round(float(fs.mean()), 2),
        }
    return out

def compute_by_semester(records: list[dict]) -> dict:
    by_sem: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_sem[r["semester"]].append(r)

    result = {}
    for sem, sub in sorted(by_sem.items()):
        with_final = [r for r in sub if r["final"] is not None and not np.isnan(r["final"])]
        n = len(with_final)
        if n == 0:
            result[sem] = {"n": 0}
            continue

        finals = np.array([r["final"] for r in with_final])
        overall_fail = float((finals < PASS_THRESHOLD).mean())

        r_groups: dict[str, list[dict]] = defaultdict(list)
        s_groups: dict[str, list[dict]] = defaultdict(list)
        n_r_unclassified = 0
        n_s_unclassified = 0
        for r in with_final:
            # 修正 (2026-07-11)：r_cluster 的「未分類」sentinel 值實測是空
            # 字串 ""（學生有成績但無任何LMS行為資料、未經assign_clusters
            # 分群），不是 None。原本沒有任何判斷，"" 會直接變成
            # by_r_cluster 明細裡一個鍵名為空字串的雜訊節點；改為比照
            # 下面 s_cluster 的既有模式，未分類的排除在明細之外、另計數。
            if r["r_cluster"]:
                r_groups[r["r_cluster"]].append(r)
            else:
                n_r_unclassified += 1
            # Keep unclassified S-cluster records out of by_s_cluster.
            if r["s_cluster"] is None:
                n_s_unclassified += 1
            else:
                s_groups[r["s_cluster"]].append(r)

        result[sem] = {
            "n": n,
            "fail_rate_final": round(overall_fail, 4),
            "final_mean": round(float(finals.mean()), 2),
            "n_s_cluster_unclassified": n_s_unclassified,
            "n_r_cluster_unclassified": n_r_unclassified,
            "by_r_cluster": _simple_group_stats(r_groups),
            "by_s_cluster": _simple_group_stats(s_groups),
        }
    return result

# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def compute_cross_analysis(records: list[dict], filter_meta: dict,
                            enable_xgb: bool = True, week_limit: int = 12) -> dict:
    """Compute cross-analysis output from in-memory records."""
    # Derive features idempotently for both CLI and pipeline callers.
    records = add_derived_features(records)

    with_final = [r for r in records if r["final"] is not None and not np.isnan(r["final"])]
    n_total = len(records)
    n_with_final = len(with_final)

    # Authoritative training semester list comes from records with final scores.
    training_semesters = sorted(set(r["semester"] for r in with_final))

    finals = np.array([r["final"] for r in with_final])
    mids = np.array([r["midterm"] for r in with_final if r["midterm"] is not None])

    overall_fail_rate = float((finals < PASS_THRESHOLD).mean())
    overall_fail_mid = float((mids < PASS_THRESHOLD).mean()) if len(mids) else None

    # Count unclassified S-cluster records among records with final scores.
    n_s_unclassified_with_final = sum(1 for r in with_final if r["s_cluster"] is None)

    xgb_validation = compute_xgb(with_final, enable_xgb, week_limit) if enable_xgb else None

    feature_importance = []
    if xgb_validation and xgb_validation.get("_model_artifacts"):
        _model, _imputer = xgb_validation["_model_artifacts"]
        feature_importance = [
            {"feature": name, "importance": round(float(score), 4), "rank": i + 1}
            for i, (name, score) in enumerate(
                sorted(zip(XGB_FEATURES, _model.feature_importances_), key=lambda x: -x[1])[:20]
            )
        ]

    by_r_cluster, n_r_unclassified = compute_group_stats(
        records, "r_cluster", overall_fail_rate, n_with_final
    )
    by_s_cluster, _n_s_unclassified_check = compute_group_stats(
        records, "s_cluster", overall_fail_rate, n_with_final
    )
    # s_cluster 的未分類 sentinel 一律是 None（見 build_records_from_memory 的
    # ltm.get() 查找），理論上不會是空字串，這裡只是交叉驗證與既有
    # n_s_cluster_unclassified（來自 build_records_from_memory，範圍含所有
    # records、不限 with_final）是否數量級一致；兩者統計範圍不同（一個含
    # 全部records、一個只算with_final子集內的），本來就可能不完全相等，
    # 不作為硬性斷言，僅記錄供除錯參考。

    return {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "edu_type": "theory",
            "practicum_excluded": True,
            "incomplete_semesters_excluded": filter_meta["incomplete_semesters"],
            "n_total_records": n_total,
            "n_with_final": n_with_final,
            "n_s_cluster_unclassified": n_s_unclassified_with_final,
            "n_r_cluster_unclassified": n_r_unclassified,
            "training_semesters": training_semesters,
            "pass_threshold": PASS_THRESHOLD,
        },
        "overall": {
            "n": n_with_final,
            "mean_final": round(float(finals.mean()), 2),
            "sd_final": round(float(finals.std()), 2),
            "fail_rate_final": round(overall_fail_rate, 4),
            "fail_rate_midterm": round(overall_fail_mid, 4) if overall_fail_mid is not None else None,
            "trajectory": _trajectory_dist(with_final),
            "approach": _approach_dist(with_final),
        },
        "spearman": compute_spearman(records, with_final=with_final),
        "bas_validation": compute_bas_validation(records, with_final=with_final),
        "bas_auc": compute_bas_auc(with_final),
        "xgb_validation": xgb_validation,
        "feature_importance": feature_importance,
        "by_r_cluster": by_r_cluster,
        "by_s_cluster": by_s_cluster,
        "cross_matrix": compute_cross_matrix(records, overall_fail_rate, n_with_final, with_final=with_final),
        "by_semester": compute_by_semester(records),
        # v1.3 新增：逐生逐學期 trajectory/learning_approach，供 Panel C 個案分析查表。
        "student_classification": _build_student_classification(records),
        # Internal records for downstream warning z-score normalization.
        "_records_with_final": with_final,
    }

def run(corr_path: str, output_path: str) -> dict:
    """Run cross analysis from a correlation JSON path and write public output."""
    records, filter_meta = load_and_filter(corr_path)
    output = compute_cross_analysis(records, filter_meta)

    # Do not export internal fields.
    output_public = {k: v for k, v in output.items() if not k.startswith("_")}

    # Fix TypeError when serializing xgb_validation which contains non-serializable _model_artifacts
    if "xgb_validation" in output_public and isinstance(output_public["xgb_validation"], dict):
        output_public["xgb_validation"] = {
            k: v for k, v in output_public["xgb_validation"].items() if not k.startswith("_")
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_public, f, ensure_ascii=False, indent=2)

    return output

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    corr_path = sys.argv[1] if len(sys.argv) > 1 else "correlation_matrix.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "cross_analysis.json"

    result = run(corr_path, out_path)

    # ── 主控台摘要輸出 ──
    logger.info("=" * 60)
    logger.info("11_cross_analysis.py 執行完成")
    logger.info("=" * 60)
    m = result["meta"]
    logger.info("資料範圍：n_total=%d  n_with_final=%d", m["n_total_records"], m["n_with_final"])
    logger.info("排除學期：%s", m["incomplete_semesters_excluded"])
    logger.info("S群未分類（序列事件不足，非S5）：n=%d", m.get("n_s_cluster_unclassified", 0))
    logger.info("R群未分類（無LMS行為資料、未經assign_clusters分群，非R3）：n=%d",
                m.get("n_r_cluster_unclassified", 0))
    o = result["overall"]
    mid_str = f"{o['fail_rate_midterm']:.1%}" if o["fail_rate_midterm"] is not None else "N/A"
    logger.info("\n整體：mean=%s  fail_rate_final=%.1f%%  fail_rate_midterm=%s",
                o["mean_final"], o["fail_rate_final"] * 100, mid_str)
    logger.info("Trajectory: %s", o["trajectory"])
    logger.info("Approach: %s", o["approach"])

    logger.info("\n--- R群 ---")
    for code, stats_ in result["by_r_cluster"].items():
        if stats_.get("low_sample"):
            logger.info("  %s: n=%d (low_sample)", code, stats_["n"])
        else:
            logger.info("  %s: n=%d  fail=%.1f%%  mean=%s  rescue=%s",
                        code, stats_["n"], stats_["fail_rate_final"] * 100,
                        stats_["final_mean"], stats_["rescue_rate"])

    logger.info("\n--- S群 ---")
    for code, stats_ in result["by_s_cluster"].items():
        if stats_.get("low_sample"):
            logger.info("  %s: n=%d (low_sample/no_baseline)", code, stats_["n"])
        else:
            logger.info("  %s: n=%d  fail=%.1f%%  mean=%s",
                        code, stats_["n"], stats_["fail_rate_final"] * 100, stats_["final_mean"])

    logger.info("\n--- Spearman ---")
    for k, v in result["spearman"].items():
        logger.info("  %s: rho=%s  p=%s  n=%s", k, v["rho"], v["p"], v["n"])

    logger.info("\n--- BAS validation ---")
    bv = result["bas_validation"]
    logger.info("  quz_first_acc r=%s", bv["quz_first_acc_r"])
    logger.info("  qmi_r=%s", bv["qmi_r"])
    logger.info("  bas_r=%s", bv["bas_r"])
    logger.info("  QMI quintiles:")
    for q in bv["qmi_quintiles"]:
        fr = f"{q['fail_rate']:.1%}" if q["fail_rate"] is not None else "N/A"
        logger.info("    Q%d: range=%s  n=%d  fail=%s", q["q"], q["range"], q["n"], fr)

    logger.info("\n--- R×S 交叉矩陣（fail_rate_final）---")
    for rg, row in result["cross_matrix"].items():
        cells = []
        for sg in ["S1", "S2", "S3", "S4", "S5"]:
            cell = row.get(sg, {})
            if "fail_rate_final" in cell:
                cells.append(f"{sg}={cell['fail_rate_final']:.1%}(n={cell['n']})")
            else:
                cells.append(f"{sg}=N/A(n={cell.get('n', 0)})")
        logger.info("  %s: %s", rg, "  ".join(cells))

    logger.info("\n--- 按學期統計（fail_rate_final）---")
    for sem, s in result["by_semester"].items():
        if s["n"] > 0:
            logger.info("  %s: n=%d  fail=%.1f%%", sem, s["n"], s["fail_rate_final"] * 100)

    logger.info("\n✅ 輸出已存檔：%s", out_path)

