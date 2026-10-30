"""
lms_etl/modules/12_early_warning.py
================================================================
依 cross_analysis.json 的 QMI/BAS 五分位閾值，
對目標學期（僅有期中成績的前瞻學期）產生預警名單。

設計原則：
- 主規則使用 QMI/BAS 連續分數（樣本量大、穩健）
- R×S 矩陣規則作為「輔助標籤」，不作為主要觸發條件
  （因 R2/R5 等極端群在訓練集 low_sample/no_baseline）
- 目標學期（如 1151）：不含 final_score，14天後可回填驗證準確度

輸入：correlation_matrix.json（目標學期 records）+ cross_analysis.json（閾值來源）
輸出：warning_{semester}.json
schema_version: "1.6"

────────────────────────────────────────────────────────────
[v7.15.17 36號規格書「學號顯示及8小時功能修正」]
students[] 新增 anon_id 欄位（上游 records 早已算好這個值，只是組裝
最終輸出dict時漏放，一行補齊）；供前端 resolveRawStudentId() 反查
原始學號用——見36號規格書決議「提前預警不套用14天窗口，只要是目前
目標學期即永遠顯示完整學號」。

  1. schema_version 1.5 → 1.6
  2. compute_warning() 輸出的 students[] 新增 anon_id 欄位
────────────────────────────────────────────────────────────
[v7.15.10 /systematic-debugging 穿透式審查]
build_target_records_from_memory() 逐筆 is_4tech 改直接採用
06_merge_grades.build_scatter_data() 依 sheet_name 正確算好的值，取代
原本本檔案獨立、僅憑「target_semester 末碼」猜測學制的 fallback
heuristic（該 heuristic 對混雜重修班/遠距班的學期檔案會誤判，詳見
11_cross_analysis.py 同一修正與 CHANGELOG_v7.md v7.15.10 的完整說明）。

範圍界定（刻意不動）：assign_risk() 內 _get_xgb_thresholds() 與
assign_risk() 本身的 is_4tech_sem / _is_4tech_sem（依 target_semester
整批判斷要套用哪一組四技/二技校準閾值）維持原邏輯不變。這是「整批
目標學期檔案」層級的粗粒度判斷，非本次修正的逐筆記錄 is_4tech 特徵；
若目標學期檔案本身混雜多學制班級，此判斷仍可能不準確，但這是獨立的
既有設計問題（見該函式 docstring 原有 [v3 Change D] 說明），需要目標
學期記錄本身的 sheet_name 才能逐批細分，超出本次修正範圍，故不在此
一併變更，避免夾帶不相關改動。

  1. schema_version 1.4 → 1.5
  2. build_target_records_from_memory()：is_4tech 改讀取上游正確值
────────────────────────────────────────────────────────────
[2026-07-04 版本整併說明]
先前部署中的 warning_1142.json 曾出現 qmi/s_cluster 顯示 null，但用「本檔案
現行邏輯」重跑同一份 correlation_matrix.json 卻能正確算出數值——已確認是
docs/data/ 內各 json 分次、非同一次 pipeline run 疊加產生的版本不同步，
非本檔案邏輯缺陷。本次更新只變動兩處，其餘既有邏輯（Change A–D：is_4tech、
學制別 QMI 切點、student_type_fail_rate_hist、assign_risk 學制別分支）維持
不動：
  1. schema_version 1.3 → 1.4
  2. _combine_risk_levels()：BAS 與 XGB 兩極端分歧（一邊 HIGH、一邊 LOW）時，
     不再直接判定 HIGH，改標記為 MEDIUM + 待複核（見該函式 docstring）
重新產生 JSON 前，請確保 correlation_matrix.json / cross_analysis.json /
warning_{semester}.json 三者出自「同一次」lms_etl.py 執行，避免再次出現
版本不同步。
────────────────────────────────────────────────────────────
[基礎醫學臨床應用(一)課程異動，見規劃書1.3節／討論10]
新增 PENDING_INDEPENDENT_MODEL_COHORTS 通用清單機制：記錄「因學習樣態與
一般學制系統性不同，需累積足夠資料後另建獨立模型」而暫緩進入本模組正式
BAS+XGBoost預測輸出的學制清單。R分群（06b_clustering.py）／S分群
（10_lsa_transition.py）不受影響，正常分類；僅本模組的預測目標名單
（build_target_records_from_memory()）依此清單排除。
================================================================
"""

from __future__ import annotations
import importlib.util
import json
import logging
import os
import sys

import numpy as np

logger = logging.getLogger(__name__)


# PENDING_INDEPENDENT_MODEL_COHORTS：見上方 docstring 說明。每個 key 對應
# 一個判斷式（接受 scatter_data record dict，回傳 bool），命中即從早期
# 預警目標名單排除。之後不管哪個學制「累積夠資料、建好獨立模型」，直接從
# 此 dict 移除對應項目即可收尾，不需要另外改寫排除邏輯本身。
#
# 目前僅啟用 transition_merged_micro 一項（本次課程異動的實際範圍）。
# post_108_curriculum_4yr（108課綱後四技生，114(2)因「放棄式學習行為」
# 導致預警準確率大降的既定precedent）暫不在此啟用——啟用會實際改變現行
# 四技學生的預測輸出範圍，屬於既有系統的既定決議但尚未執行的另一件事，
# 超出本次課程異動的範圍，故先以註解記錄結構、不加入清單，避免夾帶
# 未經確認的行為變更。若要啟用，加入一行即可：
#   "post_108_curriculum_4yr": lambda rec: (
#       rec.get("is_4tech") == 1 and str(rec.get("semester", "")) >= "1111"
#   ),
PENDING_INDEPENDENT_MODEL_COHORTS: dict = {
    "transition_merged_micro": lambda rec: (
        rec.get("course_track") == "transition_merged_micro"
    ),
}


def _pending_cohort_name(rec: dict) -> str | None:
    """回傳 rec 命中的待建獨立模型學制名稱（PENDING_INDEPENDENT_MODEL_COHORTS
    任一判斷式命中），皆未命中回傳 None。"""
    for name, predicate in PENDING_INDEPENDENT_MODEL_COHORTS.items():
        if predicate(rec):
            return name
    return None

# PASS_THRESHOLD 與 11_cross_analysis 保持一致，從惰性快取取得（避免循環匯入副作用）。
# 若尚未快取則使用預設值 60（與 11_cross_analysis.PASS_THRESHOLD 相同）。
_PASS_THRESHOLD_DEFAULT = 60

def _get_pass_threshold() -> int:
    """從 11_cross_analysis 取 PASS_THRESHOLD，確保單一來源。"""
    try:
        mod = _get_cross_analysis_mod()
        return int(getattr(mod, "PASS_THRESHOLD", _PASS_THRESHOLD_DEFAULT))
    except Exception:
        return _PASS_THRESHOLD_DEFAULT

# XGB_FEATURES / PASS_THRESHOLD 均來自 11_cross_analysis.py。
# 使用共用模組快取（單例），避免多次 exec_module。
_XGB_FEATURES_CACHE: list | None = None
_CROSS_MOD_CACHE = None

def _get_cross_analysis_mod():
    """惰性載入 11_cross_analysis 模組（單例）。"""
    global _CROSS_MOD_CACHE
    if _CROSS_MOD_CACHE is None:
        _path = os.path.join(os.path.dirname(__file__), "11_cross_analysis.py")
        spec = importlib.util.spec_from_file_location("cross_analysis_for_warn", _path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CROSS_MOD_CACHE = mod
    return _CROSS_MOD_CACHE

def _get_xgb_features() -> list:
    global _XGB_FEATURES_CACHE
    if _XGB_FEATURES_CACHE is None:
        _XGB_FEATURES_CACHE = _get_cross_analysis_mod().XGB_FEATURES
    return _XGB_FEATURES_CACHE

def _get_qmi_weights() -> dict:
    """QMI_WEIGHTS 單一來源為 11_cross_analysis；惰性取得。"""
    return _get_cross_analysis_mod().QMI_WEIGHTS

def _get_passive_weights() -> dict:
    """PASSIVE_WEIGHTS 單一來源為 11_cross_analysis；惰性取得。"""
    return _get_cross_analysis_mod().PASSIVE_WEIGHTS

SCHEMA_VERSION = "1.6"   # warning_{semester}.json meta.schema_version（升版改此處）
                          # 1.6：students[] 新增 anon_id 欄位（36號規格書
                          # 決議「anon_id補強」，供前端於目前目標學期
                          # 永遠反查原始學號用，見 tab-behavior-warning.js）

# v6新增（決議10）：(anon_id, semester) 複合鍵的 JSON-safe 字串編碼，
# 須與 10_lsa_transition.py::composite_key() / 11_cross_analysis.py::_lsa_key()
# 使用同一格式（"anon_id||semester"）。
_LSA_KEY_SEP = "||"

def _lsa_key(anon_id: str, semester: str) -> str:
    """比照 10_lsa_transition.py::composite_key()，組出同格式的字串複合鍵。"""
    return f"{anon_id}{_LSA_KEY_SEP}{semester}"

# XGB 風險分層機率閾值（對應 QMI quintile 的 HIGH/MEDIUM/LOW 三級）
# HIGH：p >= 此值 → 與 Q1 不及格率（≈33%）對應
# MEDIUM：p >= 此值 → 與 Q2 不及格率（≈23%）對應
_XGB_HIGH_THRESHOLD   = 0.80   # 預設值（ETL v3 rerun 前的 fallback）
_XGB_MEDIUM_THRESHOLD = 0.60   # [v3 Change E] 已實作：ETL rerun 後自動從
                                # xgb_validation.thresholds 讀取學制別校準值，此常數僅作 fallback

# OOD 偵測：當目標學期期中不及格率 > 歷史均值 × 此倍率，視為高難度/OOD 學期
# 歷史訓練集全體期中不及格率均值 ≈ 0.37（固定於訓練集計算結果）
_OOD_THRESHOLD_MULTIPLIER = 1.20   # 超過歷史 120% 才觸發 OOD 切點校正
_OOD_MID_FAIL_RATE_DEFAULT = 0.37  # cross_analysis 未提供 fail_rate_midterm 時的保守預設

_RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

def _combine_risk_levels(bas_level: "str | None", xgb_level: "str | None") -> "tuple[str, str, bool]":
    """合併 BAS 與 XGB 兩個風險等級。

    Returns (final_level, source, model_disagreement)

    [2026-07-04 變更] 舊版無條件取兩者較嚴重者（max-override）。
    問題：任一模型只要判 HIGH，另一模型即使判 LOW 也會被覆蓋，實測 1142
    資料顯示 94% 學生因此被歸為單一等級 HIGH，喪失分級的鑑別度
    （尤其當 XGB 對整個 OOD/單一學制學期系統性偏高時）。

    新規則：
      - 雙方一致                     → 該等級，source="BAS+XGB"
      - 相差一級（如 MEDIUM vs HIGH） → 取較高者（維持原行為，分歧不算劇烈）
      - 相差兩級（HIGH vs LOW，最劇烈的分歧）→ 不直接判 HIGH，
        降級為 MEDIUM，並回傳 model_disagreement=True，
        供 compute_warning() 加註 W-MODEL-DISAGREE 標籤，交由人工複核
        而非自動判定為最高風險。
      - XGB 缺席（模型未啟用/樣本不足）→ 純用 BAS，source="BAS"
    """
    bas = bas_level if bas_level in _RISK_ORDER else "LOW"
    xgb = xgb_level if xgb_level in _RISK_ORDER else None
    if xgb is None:
        return bas, "BAS", False

    bas_v, xgb_v = _RISK_ORDER[bas], _RISK_ORDER[xgb]
    if bas_v == xgb_v:
        return bas, "BAS+XGB", False

    if {bas_v, xgb_v} == {_RISK_ORDER["HIGH"], _RISK_ORDER["LOW"]}:
        disagree_source = "XGB-ONLY" if xgb_v > bas_v else "BAS-ONLY"
        return "MEDIUM", disagree_source, True

    # 相差一級：維持原本「取較高者」的行為
    if xgb_v > bas_v:
        return xgb, "XGB", False
    return bas, "BAS", False

def build_target_records_from_memory(
    scatter_data: list[dict], lsa_type_map: dict,
    target_semester: str,
    transition_ratios: dict | None = None) -> list[dict]:
    """
    供 lms_etl.py 主流程直接呼叫（無檔案 I/O）。
    取出目標學期（理論上 final=None）的 theory 學生記錄。

    [BUG-S5-DEFAULT FIX] 與 11_cross_analysis.build_records_from_memory 同步修正：
    lsa_type_map 缺值（序列事件不足、未分類）改為 None，不再誤植為 "S5"。
    """
    ltm = lsa_type_map or {}
    records = []
    _is4tech_fallback_count = 0  # v7.15.10：統計退回舊heuristic的筆數，迴圈後警告
    _pending_cohort_excluded: dict = {}  # 見規劃書1.3節：統計各待建獨立模型學制被排除的筆數
    for rec in scatter_data:
        if rec.get("edu_type") != "theory":
            continue
        if rec["semester"] != target_semester:
            continue

        pending = _pending_cohort_name(rec)
        if pending is not None:
            _pending_cohort_excluded[pending] = _pending_cohort_excluded.get(pending, 0) + 1
            continue

        aid = rec["anon_id"]
        sem = str(rec["semester"])
        feats = rec.get("features", {}) or {}
        # v6修正（決議10）：原本純 anon_id 查表，重修生跨學期會查到同一個值。
        trans = (transition_ratios or {}).get(_lsa_key(aid, sem), {})
        records.append({
            "anon_id": aid,
            "masked_id": rec.get("masked_id"),
            "semester": rec["semester"],
            "r_cluster": rec["cluster"],
            "s_cluster": ltm.get(_lsa_key(aid, sem)),  # None＝序列事件不足，未分類（非 S5）
            "midterm": rec.get("midterm_score"),
            "final": rec.get("final_score"),  # 預期為 None（驗證學期）
            **{k: (v if v is not None else 0.0) for k, v in feats.items()},
            **{k: trans.get(k, 0.0) for k in ("MM_ratio", "MQ_ratio", "QM_ratio", "QQ_ratio")},
            # [v3 Change A → v7.15.10 修正] 學制別特徵。
            # 原僅憑 target_semester 末碼猜測學制，對混雜重修班/遠距班的
            # 學期檔案會誤判（詳見 CHANGELOG_v7.md v7.15.10、11_cross_analysis.py
            # 同一修正的完整說明）。改直接採用 06_merge_grades.build_scatter_data()
            # 依 sheet_name 正確算好的值；僅在讀到 schema<2.2 的舊版
            # correlation_matrix.json（缺此欄）時才 fallback 並警告一次。
            "is_4tech": (
                int(rec["is_4tech"]) if rec.get("is_4tech") is not None
                else (1 if rec.get("semester", "")[-1] == "2" else 0)
            ),
            # [v3 Change C] 前瞻學期 student_type_fail_rate_hist 不可得，
            # 設 NaN 由 XGBoost SimpleImputer(median) 以訓練集中位數填補。
            "student_type_fail_rate_hist": np.nan,
        })
        if rec.get("is_4tech") is None:
            _is4tech_fallback_count += 1

    if _is4tech_fallback_count > 0:
        logging.getLogger(__name__).warning(
            "[12] %d 筆目標學期記錄的 correlation_matrix.json 缺少 is_4tech 欄"
            "（schema < 2.2 的舊版快取），已退回舊版純學期末碼判斷。建議重新"
            "執行完整 lms_etl.py pipeline 以更新快取。", _is4tech_fallback_count,
        )
    if _pending_cohort_excluded:
        for cohort_name, n in _pending_cohort_excluded.items():
            print(
                f"  ℹ️  [12] {n} 筆「{cohort_name}」學制記錄依 "
                f"PENDING_INDEPENDENT_MODEL_COHORTS 排除於本次早期預警目標名單之外"
                f"（見規劃書1.3節／討論10：R分群/S分群/cross_analysis 不受影響）。"
            )
    return records

def load_target_semester(corr_path: str, target_semester: str) -> list[dict]:
    """獨立執行用（CLI）：讀檔後委派給 build_target_records_from_memory()。"""
    with open(corr_path, encoding="utf-8") as f:
        data = json.load(f)
    ltm = data.get("lsa_transition", {}).get("lsa_type_map", {})
    return build_target_records_from_memory(data["scatter_data"], ltm, target_semester)

def add_qmi_bas(records: list[dict], qmi_stats: dict) -> list[dict]:
    """
    計算 QMI / passive_index / BAS（v3.1 完整公式）。

    [v3.1 Fix A] 啟用 midterm 分量（semester-local z-score）：
      BAS = w_mid * z_mid_local + w_qmi * z_qmi_train + w_neg_passive * z_neg_passive_train
      - midterm z-score 使用目標學期自身 mean/std（semester-local calibration），
        避免 OOD 學期的期中成績分布偏移拉高或壓低 z 值。
      - qmi / neg_passive z 仍沿用訓練集參數（cross_analysis），確保跨學期可比較。
      - log_attempts 分量（w=0.15）在前瞻截斷時間點不穩定，維持省略。

    BAS 最佳權重（grid search on 1142 validated data）：
      w_mid=0.70, w_qmi=0.25, w_bas_qp=0.05
      但為保留公式可解釋性，沿用規格書 w_mid=0.35, w_qmi=0.30, w_neg_passive=0.20
      （三項合計 0.85，renormalize 邏輯已移入共用函式 compute_qmi_passive_bas()）

    v6修正（決議3/6/7/8）：核心計算改呼叫 11_cross_analysis.py 的共用函式
    compute_qmi_passive_bas()，取代本函式原本重複的迴圈邏輯：
    - qmi_mean/qmi_std/pas_mean/pas_std 沿用訓練集參數（不變）
    - mid_mean/mid_std 依決議8維持 semester-local 基準（本函式自算，
      不可呼叫 11 的 _safe_qmi_passive_stats()——那是訓練集 pooled 基準，
      跟這裡的語意不同，詳見該函式 docstring）
    - total_attempts==0 時 qmi=None，bas_score 走 renormalize（決議3/7，
      include_log_attempts=False，此處固定捨棄 log_attempts 分量，語意不變）
    """
    m11 = _get_cross_analysis_mod()

    qmi_mean = qmi_stats["qmi_mean"]
    qmi_std  = qmi_stats["qmi_std"]
    pas_mean = qmi_stats["passive_mean"]
    pas_std  = qmi_stats["passive_std"]
    bas_w    = qmi_stats["bas_weights"]  # {"midterm":0.35,"qmi":0.30,"neg_passive":0.20,...}

    # 迴圈前取一次：_get_*_weights() 觸發惰性模組載入（首次），避免 N 次重複呼叫
    _pt   = _get_pass_threshold()
    _qmiw = _get_qmi_weights()
    _pasw = _get_passive_weights()

    # [Fix A] semester-local midterm z-score 統計量（目標學期自身，非訓練集）
    # 決議8查證結果：與訓練集 pooled mean/std 確實不同，維持分開傳入不做統一。
    _mids = [r.get("midterm") for r in records if r.get("midterm") is not None]
    if len(_mids) >= 2:
        _mid_arr = np.array(_mids, dtype=float)
        _mid_mean_local = float(_mid_arr.mean())
        _mid_std_local  = float(_mid_arr.std())
    else:
        _mid_mean_local, _mid_std_local = 60.0, 15.0  # 保守預設

    stats = {
        "qmi_mean": qmi_mean, "qmi_std": qmi_std if qmi_std > 0 else 1.0,
        "pas_mean": pas_mean, "pas_std": pas_std if pas_std > 0 else 1.0,
        "mid_mean": _mid_mean_local, "mid_std": _mid_std_local if _mid_std_local > 0 else 1.0,
    }

    for r in records:
        m11.compute_qmi_passive_bas(
            r, stats, _qmiw, _pasw, bas_w, include_log_attempts=False,
        )
        # 沿用既有輸出精度（四捨五入至4位小數，warning_{semester}.json 既有格式）
        if r["qmi"] is not None:
            r["qmi"] = round(r["qmi"], 4)
        r["passive_index"] = round(r["passive_index"], 4)
        r["bas_score"]     = round(r["bas_score"], 4)

        # learning_approach 沿用既有設計不變（or 0.0 defaulted 值分類，
        # 跟 QMI 本身是否為 None 是兩件事，不受決議3影響）
        first_acc = r.get("quz_first_attempt_accuracy", 0.0) or 0.0
        delta     = r.get("quz_score_delta", 0.0) or 0.0
        if first_acc >= 0.6 and delta < 0.2:
            r["learning_approach"] = "DEEP"
        elif delta >= 0.3:
            r["learning_approach"] = "SURFACE"
        else:
            r["learning_approach"] = "MODERATE"

        r["midterm_status"] = (
            "PASS" if r["midterm"] is not None and r["midterm"] >= _pt
            else "FAIL" if r["midterm"] is not None
            else None
        )

    return records

def _sem_to_label(sem: str) -> str:
    """'1111' → '111-1'，'1142' → '114-2'（前端顯示格式）。"""
    return f"{sem[:3]}-{sem[3]}" if len(sem) == 4 else sem

def _training_range_label(cross_output: dict) -> str:
    """從 cross_analysis 自動推導訓練集範圍標籤，如 '111-1–114-2'。
    來源：xgb_validation.training_semesters（精確）或
          by_semester keys（fallback）或 meta.incomplete_semesters_excluded（輔助）。
    """
    # 優先：xgb_validation 的 training_semesters（最完整）
    xgb = cross_output.get("xgb_validation") or {}
    sems = xgb.get("training_semesters") or []

    # fallback：by_semester 的 keys（含 with_final 的學期）
    if not sems:
        sems = sorted(cross_output.get("by_semester", {}).keys())

    if not sems:
        return "unknown"

    sems = sorted(sems)
    return f"{_sem_to_label(sems[0])}–{_sem_to_label(sems[-1])}"


def _get_xgb_thresholds(xgb_validation: dict, target_semester: str) -> "tuple[float, float]":
    """[v3 Change E] 從 xgb_validation.thresholds 取得學制別校準閾值。

    優先順序：
      1. 學制別閾值（4tech / 2tech，依 target_semester[-1] 判斷）
      2. combined fallback（thresholds 存在但學制分組樣本不足時）
      3. 硬編碼預設值 0.80 / 0.60（thresholds 不存在時，ETL v3 rerun 前）
    """
    thresholds = (xgb_validation or {}).get("thresholds") or {}
    if not thresholds:
        # ETL v3 rerun 前：thresholds 尚不存在，使用歷史設定值
        return _XGB_HIGH_THRESHOLD, _XGB_MEDIUM_THRESHOLD

    is_4tech_sem = bool(target_semester) and target_semester[-1] == "2"
    type_key = "4tech" if is_4tech_sem else "2tech"
    type_data = thresholds.get(type_key, {})

    # 學制別閾值有效（n >= 10，note 無 fallback）
    if type_data.get("n", 0) >= 10 and "note" not in type_data:
        return float(type_data["high"]), float(type_data["medium"])

    # combined fallback
    combined = thresholds.get("combined", {})
    if combined.get("high") is not None:
        return float(combined["high"]), float(combined["medium"])

    return _XGB_HIGH_THRESHOLD, _XGB_MEDIUM_THRESHOLD


def add_xgb_probability(records: list[dict], xgb_validation: dict,
                        model_artifacts=None,
                        target_semester: str = "") -> list[dict]:
    if not xgb_validation or xgb_validation.get("status") in ("insufficient_data", "missing_dependency") \
       or model_artifacts is None:
        for r in records:
            r["xgb_probability"], r["risk_level_xgb"] = None, None
        return records

    # [v3 Change E] 依學制取校準閾值（fallback → 硬編碼預設）
    high_t, med_t = _get_xgb_thresholds(xgb_validation, target_semester)

    model, imputer = model_artifacts
    X = imputer.transform([[r.get(f, np.nan) for f in _get_xgb_features()] for r in records])
    probs = model.predict_proba(X)[:, 1]
    for r, p in zip(records, probs):
        r["xgb_probability"] = round(float(p), 4)
        r["risk_level_xgb"] = ("HIGH"   if p >= high_t else
                               "MEDIUM" if p >= med_t  else "LOW")
    return records

def assign_risk(records: list[dict], cross: dict,
                target_semester: str = "") -> list[dict]:
    """
    主規則（QMI 五分位）+ 輔助標籤（R/S群、Learning Approach、期中狀態）。

    風險等級對應 cross_analysis.bas_validation.qmi_quintiles：
      QMI Q1（最低20%）→ HIGH   （訓練集 fail_rate ≈ 33%）
      QMI Q2           → MEDIUM （訓練集 fail_rate ≈ 23%）
      QMI Q3+          → LOW

    [v2 Fix C] OOD semester-difficulty calibration：
      當目標學期期中不及格率 > 歷史均值 × 1.20，
      改用目標學期自身 QMI 20th/40th percentile（OOD 模式，最優先）。

    [v3 Change D] 學制別 QMI 切點（非 OOD 時）：
      依 target_semester[-1] 選擇四技/二技對應切點（來自 cross_analysis）。
      fallback：合併訓練集切點（qmi_q1_4tech/2tech 為 None 時）。
    """
    bv = cross["bas_validation"]
    q  = bv["qmi_quintiles"]
    q1_hi_hist = q[0]["range"][1]
    q2_hi_hist = q[1]["range"][1]

    # [Fix C] OOD detection（最優先，覆蓋學制別切點）
    hist_mid_fail = cross.get("overall", {}).get("fail_rate_midterm") or _OOD_MID_FAIL_RATE_DEFAULT
    _pt_local = _get_pass_threshold()
    _mids_valid = [r for r in records if r.get("midterm") is not None]
    target_mid_fail = (
        sum(1 for r in _mids_valid if r["midterm"] < _pt_local) / len(_mids_valid)
        if _mids_valid else 0.0
    )
    _ood = target_mid_fail > hist_mid_fail * _OOD_THRESHOLD_MULTIPLIER
    # [v3] 學制別判斷：hoisted 至 OOD check 之前，兩分支共用
    _is_4tech_sem = (target_semester[-1] == "2") if target_semester else False

    if _ood:
        # OOD：使用目標學期自身分位數
        # v6修正（決議6，缺口B）：qmi=None混入時，np.array()無dtype會變成
        # object dtype，np.percentile內部比較時TypeError當機；即使換用
        # np.nanpercentile若陣列仍是object dtype一樣會炸，兩個改動缺一不可
        # （建陣列當下就要dtype=float，讓None轉成np.nan）。
        _target_qmi = np.array([r["qmi"] for r in records], dtype=float)
        q1_hi = float(np.nanpercentile(_target_qmi, 20))
        q2_hi = float(np.nanpercentile(_target_qmi, 40))
    else:
        # [v3 Change D] 非 OOD：依學制選擇對應切點
        if _is_4tech_sem:
            q1_hi = bv.get("qmi_q1_4tech") or q1_hi_hist
            q2_hi = bv.get("qmi_q2_4tech") or q2_hi_hist
        else:
            q1_hi = bv.get("qmi_q1_2tech") or q1_hi_hist
            q2_hi = bv.get("qmi_q2_2tech") or q2_hi_hist

    r_groups = cross["by_r_cluster"]
    s_groups = cross["by_s_cluster"]

    for r in records:
        rules = []

        # 主規則：QMI 五分位
        # v6修正（決議3/6，缺口C）：qmi is None（total_attempts=0，真正沒
        # 作答）時不進入數值比較，直接標記 W-INSUFFICIENT-QUIZ-DATA，
        # risk_level 交由輔助規則（期中/R-S群）決定，不強制 HIGH/LOW。
        if r["qmi"] is None:
            level = None
            rules.append("W-INSUFFICIENT-QUIZ-DATA")
        elif r["qmi"] < q1_hi:
            level = "HIGH"
            rules.append("W-QMI-Q1")
        elif r["qmi"] < q2_hi:
            level = "MEDIUM"
            rules.append("W-QMI-Q2")
        else:
            level = "LOW"

        # 輔助規則：R1 × SURFACE × 期中不及格
        if (r["r_cluster"] == "R1" and r["learning_approach"] == "SURFACE"
                and r["midterm_status"] == "FAIL"):
            rules.append("W-R1-SURFACE-MIDFAIL")
            if level == "LOW":
                level = "MEDIUM"

        # 期中不及格本身列為輔助標籤
        if r["midterm_status"] == "FAIL":
            rules.append("W-MIDTERM-FAIL")

        # no_baseline 標記（R2/R5 等樣本不足群組）
        r_stat = r_groups.get(r["r_cluster"], {})
        s_stat = s_groups.get(r["s_cluster"], {}) if r["s_cluster"] is not None else {}
        if r_stat.get("low_sample") or r_stat.get("no_baseline"):
            rules.append(f"W-R-NO-BASELINE({r['r_cluster']})")
        # [BUG-S5-DEFAULT FIX] s_cluster is None 代表該生序列事件不足、未被 LSA
        # 分類，與「已分類但歷史樣本數不足」（W-S-NO-BASELINE）是不同情境，
        # 分開標記避免論文/前端誤讀為「已知高風險但樣本少」。
        if r["s_cluster"] is None:
            rules.append("W-S-UNCLASSIFIED")
        elif s_stat.get("low_sample") or s_stat.get("no_baseline"):
            rules.append(f"W-S-NO-BASELINE({r['s_cluster']})")

        r["risk_level"] = level
        r["triggered_rules"] = rules

        # historical reference（供前端顯示「歷史同組不及格率」）
        # v6修正（決議6，缺口D）：qmi is None時qmi_quintile直接給None，
        # 不進入數值比較鏈（q1_hi/q2_hi/q[2]/q[3]皆為float，None跟float
        # 比較在Python3會直接TypeError）。
        r["evidence"] = {
            "qmi_quintile": (
                None if r["qmi"] is None else
                1 if r["qmi"] < q1_hi else
                2 if r["qmi"] < q2_hi else
                3 if r["qmi"] < q[2]["range"][1] else
                4 if r["qmi"] < q[3]["range"][1] else 5
            ),
            "r_cluster_historical_fail_rate": r_stat.get("fail_rate_final"),
            "s_cluster_historical_fail_rate": s_stat.get("fail_rate_final"),
            "s_cluster_unclassified": r["s_cluster"] is None,
            "r_cluster_rescue_rate": r_stat.get("rescue_rate"),
            # [v3 Change D] 記錄本次採用的切點模式（供驗證/除錯用）
            "qmi_threshold_mode": ("ood" if _ood else "4tech" if _is_4tech_sem else "2tech"),
            "qmi_q1_hi_used": round(q1_hi, 4),
            "qmi_q2_hi_used": round(q2_hi, 4),
        }

    return records

def compute_warning(target_records: list[dict], cross_output: dict,
                     target_semester: str) -> dict:
    """
    核心計算（無檔案 I/O）。
    target_records：build_target_records_from_memory() 的輸出
    cross_output：compute_cross_analysis() 的完整回傳值
                  （含內部欄位 _records_with_final，用於 z-score 標準化參數）
    """
    train_records = cross_output["_records_with_final"]
    # v6修正（決議6，缺口E）：原本裸建陣列 np.array([r["qmi"] for r in
    # train_records]) 無dtype，qmi=None混入時.mean()/.std()直接TypeError
    # 當機（此函式是v5規格書逐檔清單完全未列到的第三處統計量計算，
    # 跟 11_cross_analysis.py 的缺口A/E 是同一個bug的不同副本）。
    # 改呼叫共用函式 _safe_qmi_passive_stats()，統一防呆邏輯。
    m11 = _get_cross_analysis_mod()
    _stats = m11._safe_qmi_passive_stats(train_records)
    qmi_stats = {
        "qmi_mean": _stats["qmi_mean"],
        "qmi_std": _stats["qmi_std"],
        "passive_mean": _stats["pas_mean"],
        "passive_std": _stats["pas_std"],
        "bas_weights": cross_output["bas_validation"]["bas_weights"],
    }

    records = add_qmi_bas(target_records, qmi_stats)
    records = add_xgb_probability(
        records, cross_output.get("xgb_validation"),
        (cross_output.get("xgb_validation") or {}).get("_model_artifacts"),
        target_semester=target_semester,
    )
    records = assign_risk(records, cross_output, target_semester=target_semester)

    for r in records:
        bas_level = r.get("risk_level")
        final_level, source, disagree = _combine_risk_levels(bas_level, r.get("risk_level_xgb"))
        r["risk_level_bas"] = bas_level
        r["risk_level_final"] = final_level
        r["risk_source"] = source
        r["model_disagreement"] = disagree
        if disagree:
            r["triggered_rules"] = r.get("triggered_rules", []) + ["W-MODEL-DISAGREE"]
        # Keep the legacy field as the dashboard-facing BAS+XGB level.
        r["risk_level"] = final_level

    summary = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    disagreement_count = 0
    for r in records:
        summary[r["risk_level"]] += 1
        if r.get("model_disagreement"):
            disagreement_count += 1

    q = cross_output["bas_validation"]["qmi_quintiles"]
    _low_q = [x for x in q[2:] if x["fail_rate"] is not None and x["n"] > 0]
    level_ref = {
        "HIGH": q[0]["fail_rate"],
        "MEDIUM": q[1]["fail_rate"],
        "LOW": round(sum(x["fail_rate"] * x["n"] for x in _low_q) /
                      sum(x["n"] for x in _low_q), 4) if _low_q else None,
    }

    return {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "target_semester": target_semester,
            "edu_type": "theory",
            "reference_data": (
                f"cross_analysis.json "
                f"({_training_range_label(cross_output)} training set, n={len(train_records)})"
            ),
            "data_cutoff": "期中考後（含期中成績，final 待 ~14 天後產出）",
            "total_students": len(records),
            "primary_rule": "QMI quintile (Q1=HIGH, Q2=MEDIUM, Q3-5=LOW)",
            "risk_level_policy": (
                "risk_level_final combines BAS/QMI and XGBoost: same level -> that level; "
                "one level apart -> the higher one (unchanged from v1.3); "
                "two levels apart (HIGH vs LOW, the sharpest disagreement) -> downgraded to "
                "MEDIUM with model_disagreement=true / W-MODEL-DISAGREE instead of auto-HIGH, "
                "flagged for manual review rather than auto-escalated. "
                "risk_level_bas preserves the BAS-only level."
            ),
            "secondary_rules": [
                "W-R1-SURFACE-MIDFAIL", "W-MIDTERM-FAIL",
                "W-R-NO-BASELINE", "W-S-NO-BASELINE", "W-S-UNCLASSIFIED",
                "W-MODEL-DISAGREE",
            ],
            "bas_formula": "v3.1: w_mid(local-z) + w_qmi(train-z) + w_neg_passive(train-z), renormalized",
        },
        "summary": {
            **{
                level: {"count": summary[level], "historical_fail_rate_ref": level_ref[level]}
                for level in ("HIGH", "MEDIUM", "LOW")
            },
            "model_disagreement_count": disagreement_count,
        },
        "students": [
            {
                "anon_id": r["anon_id"],  # 36號規格書決議：提前預警不套用14天窗口，
                                           # 只要是目前目標學期即永遠顯示完整學號，
                                           # 前端靠此欄位呼叫 resolveRawStudentId()
                "masked_id": r["masked_id"],
                "risk_level": r["risk_level"],
                "risk_level_final": r["risk_level_final"],
                "risk_level_bas": r["risk_level_bas"],
                "risk_source": r["risk_source"],
                "model_disagreement": r["model_disagreement"],
                "r_cluster": r["r_cluster"],
                "s_cluster": r["s_cluster"],
                "learning_approach": r["learning_approach"],
                "midterm_score": r["midterm"],
                "midterm_status": r["midterm_status"],
                "qmi": r["qmi"],
                "bas_score": r["bas_score"],
                "xgb_probability": r["xgb_probability"],
                "risk_level_xgb": r["risk_level_xgb"],
                "triggered_rules": r["triggered_rules"],
                "evidence": r["evidence"],
            }
            for r in sorted(
                records,
                key=lambda x: (
                    -_RISK_ORDER.get(x.get("risk_level_final"), 0),
                    -(x.get("xgb_probability") if x.get("xgb_probability") is not None else -1),
                    x["bas_score"],
                )
            )
        ],
    }

def _infer_target_semester(cross_output: dict) -> str:
    """從 cross_analysis 自動推導目標學期：取 incomplete_semesters_excluded 中最新一個。
    若無 incomplete_semesters，則嘗試用 by_semester 的最新學期 + 1。
    回傳空字串表示無法自動推導，呼叫端需明確指定。
    """
    incomplete = cross_output.get("meta", {}).get("incomplete_semesters_excluded", [])
    if incomplete:
        return sorted(incomplete)[-1]
    # fallback：訓練集最新學期 + 1（下一個學期）
    sems = sorted(cross_output.get("by_semester", {}).keys())
    if not sems:
        return ""
    last = sems[-1]
    # 學期末碼：1→2（同學年下一學期），2→下學年1
    if last[-1] == "1":
        return last[:-1] + "2"
    else:
        year = int(last[:3]) + 1
        return f"{year}1"


def run(corr_path: str, cross_path: str, output_path: str = "",
        target_semester: str = "") -> dict:
    """獨立執行用（CLI）：讀檔 → 計算 → 寫檔。

    target_semester 留空時，自動從 cross_analysis.json 推導
    （取 incomplete_semesters_excluded 最新值）。
    output_path 留空時，依 target_semester 自動命名為 warning_{semester}.json。
    """
    with open(cross_path, encoding="utf-8") as f:
        cross_output = json.load(f)

    # 自動推導 target_semester（若未指定）
    if not target_semester:
        target_semester = _infer_target_semester(cross_output)
    if not target_semester:
        raise ValueError(
            "無法自動推導 target_semester，請明確指定（如 '1151'）。\n"
            "確認 cross_analysis.json 含 meta.incomplete_semesters_excluded。"
        )

    # 自動命名 output_path（若未指定）
    if not output_path:
        output_path = f"warning_{target_semester}.json"

    # CLI 模式下 cross_analysis.json 已移除 _records_with_final（見 11_cross_analysis.run）
    # 故需重新從 correlation_matrix.json 計算訓練集 records 以取得 z-score 參數
    if "_records_with_final" not in cross_output:
        mod = _get_cross_analysis_mod()
        train_records, _ = mod.load_and_filter(corr_path)
        train_records = mod.add_derived_features(train_records)
        cross_output["_records_with_final"] = [r for r in train_records if r["final"] is not None]

    target_records = load_target_semester(corr_path, target_semester)
    output = compute_warning(target_records, cross_output, target_semester)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return output

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    corr_path  = sys.argv[1] if len(sys.argv) > 1 else "correlation_matrix.json"
    cross_path = sys.argv[2] if len(sys.argv) > 2 else "cross_analysis.json"
    # output_path 和 target_semester 留空 → run() 自動推導
    out_path   = sys.argv[3] if len(sys.argv) > 3 else ""
    target     = sys.argv[4] if len(sys.argv) > 4 else ""

    result = run(corr_path, cross_path, out_path, target)

    logger.info("=" * 60)
    logger.info("12_early_warning.py — 目標學期 %s 預警結果", target)
    logger.info("=" * 60)
    logger.info("總人數：%d", result["meta"]["total_students"])
    for level, s in result["summary"].items():
        logger.info("  %s: %d 人  (歷史參考不及格率 %.1f%%)",
                    level, s["count"], s["historical_fail_rate_ref"] * 100)

    logger.info("\n--- HIGH 風險學生（前5）---")
    high = [s for s in result["students"] if s["risk_level"] == "HIGH"][:5]
    for s in high:
        logger.info("  %s  R=%s S=%s  approach=%s  qmi=%.3f  bas=%.2f"
                    "  mid=%s(%s)  rules=%s",
                    s["masked_id"], s["r_cluster"], s["s_cluster"] or "未分類",
                    s["learning_approach"], s["qmi"], s["bas_score"],
                    s["midterm_score"], s["midterm_status"], s["triggered_rules"])

    logger.info("\n✅ 輸出已寫入：%s", out_path)
