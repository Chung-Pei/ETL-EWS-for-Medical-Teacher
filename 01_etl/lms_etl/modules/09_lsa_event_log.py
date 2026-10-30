"""
lms_etl/modules/09_lsa_event_log.py
合併 reading_log（教材閱讀）與 quiz（題庫）事件，
計算 Lag-1 序列對，供 LSA 轉移矩陣使用。

行為代碼：
  M — 教材閱讀（reading_log 逐筆事件）
  Q — 題庫作答（quiz sheet 篩選「題庫」標題）
"""

import pandas as pd

BEHAVIORS = ["M", "Q"]
LSA_SESSION_GAP_MINUTES = 120


def build_event_log(
    sheets: dict,
    anon_map: dict,
    session_gap_minutes: int = LSA_SESSION_GAP_MINUTES,
) -> pd.DataFrame:
    """
    合併 reading_log + quiz 事件，計算 Lag-1 有效序列對。

    Parameters
    ----------
    sheets             : _merge_lms_files() 輸出的 dict，含 reading_log / quiz key
    anon_map           : lookup_table = { anon_id: masked_id }（僅用於驗證 anon_id 有效）
    session_gap_minutes: 同一學習 session 的最大間隔（分鐘），超過視為新 session

    Returns
    -------
    DataFrame：anon_id | behavior | timestamp | next_behavior | gap_minutes | semester
    僅含 gap_minutes ∈ [0, session_gap_minutes] 的有效序列對。
    空 reading_log / quiz 時回傳空 DataFrame（graceful degradation）。

    semester 欄直接取自來源列的 _semester（由 _merge_lms_files() 依檔名寫入），
    確保「每一筆事件用自己實際發生的學期」歸類，而非事後用 anon_id 反查單一學期
    （後者對重修生／跨學期學生會整批誤植成同一學期，詳見 CHANGELOG）。
    """
    events = []  # list of {"anon_id", "behavior", "timestamp", "semester"}

    # ── M 事件：reading_log ──────────────────────────────────────
    reading_df = sheets.get("reading_log")
    if reading_df is None or (hasattr(reading_df, "empty") and reading_df.empty):
        print("  ⚠️  [09] reading_log 不存在或為空，M 行為事件將缺失（僅保留 Q 行為）")
    else:
        # 偵測 anon_id 欄（匿名化後應為 anon_id）
        id_col  = _detect_id_col(reading_df)
        # 偵測時間戳欄：優先「開始」，fallback「時間」「登入時間」「日期」
        time_col = _detect_time_col(reading_df, ("開始", "時間", "登入時間", "日期", "存取時間"))
        if id_col and time_col:
            valid_ids = reading_df[id_col].astype(str).str.strip()
            valid_mask = valid_ids.isin(anon_map)
            sub = reading_df[valid_mask]
            # 效能修正 (2026-07-11)：原本用 .map(_parse_ts) 對每一列個別呼叫
            # pd.to_datetime()，reading_log 動輒數十萬列規模下極慢（實測
            # build_event_log 單獨佔了Step 8前後382秒，是全流程最大瓶頸）。
            # pd.to_datetime(series, errors="coerce") 對整欄一次性向量化解析，
            # 語意與原本逐值 try/except 完全等價（失敗一律回傳 NaT，
            # 下游 .notna() 篩選對 NaT 與原本的 None 行為相同）。
            timestamps = pd.to_datetime(sub[time_col], errors="coerce")
            ts_mask = timestamps.notna()
            semesters = _extract_semester_col(sub)
            for aid, ts, sem in zip(
                valid_ids[valid_mask][ts_mask], timestamps[ts_mask], semesters[ts_mask]
            ):
                events.append({"anon_id": aid, "behavior": "M", "timestamp": ts, "semester": sem})
        else:
            print(f"  ⚠️  [09] reading_log 缺少 id 欄（{id_col!r}）或時間欄（{time_col!r}），M 行為略過")

    # ── Q 事件：quiz，篩選「題庫」標題 ─────────────────────────────
    quiz_df = sheets.get("quiz")
    if quiz_df is None or (hasattr(quiz_df, "empty") and quiz_df.empty):
        print("  ⚠️  [09] quiz sheet 不存在或為空，Q 行為事件將缺失")
    else:
        id_col_q   = _detect_id_col(quiz_df)
        title_col  = _detect_quiz_title_col(quiz_df)
        time_col_q = _detect_time_col(quiz_df, ("開始", "提交時間", "時間", "登入時間", "日期"))
        if id_col_q and title_col and time_col_q:
            # ⚠️ 不依賴 category 欄；直接用 str.contains 篩選「題庫」
            quiz_mask = quiz_df[title_col].astype(str).str.contains("題庫", na=False)
            sub_quiz  = quiz_df[quiz_mask]
            valid_ids = sub_quiz[id_col_q].astype(str).str.strip()
            id_mask   = valid_ids.isin(anon_map)
            sub_quiz  = sub_quiz[id_mask]
            valid_ids = valid_ids[id_mask]
            # 效能修正 (2026-07-11)：同上，改為向量化解析
            timestamps = pd.to_datetime(sub_quiz[time_col_q], errors="coerce")
            ts_mask    = timestamps.notna()
            semesters  = _extract_semester_col(sub_quiz)
            for aid, ts, sem in zip(valid_ids[ts_mask], timestamps[ts_mask], semesters[ts_mask]):
                events.append({"anon_id": aid, "behavior": "Q", "timestamp": ts, "semester": sem})
        else:
            print(f"  ⚠️  [09] quiz 缺少必要欄位（id={id_col_q!r}, title={title_col!r}, time={time_col_q!r}），Q 行為略過")

    if not events:
        print("  ⚠️  [09] 無任何事件，回傳空 DataFrame")
        return _empty_log()

    event_df = (
        pd.DataFrame(events)
        .sort_values(["anon_id", "timestamp"])
        .reset_index(drop=True)
    )
    n_events = len(event_df)
    n_students = event_df["anon_id"].nunique()

    # ── Lag-1 序列對計算 ─────────────────────────────────────────
    # shift(-1) 取得下一個事件（同一 anon_id 內）
    event_df["next_behavior"] = event_df.groupby("anon_id")["behavior"].shift(-1)
    event_df["next_ts"]       = event_df.groupby("anon_id")["timestamp"].shift(-1)
    event_df["gap_minutes"]   = (
        (event_df["next_ts"] - event_df["timestamp"]).dt.total_seconds() / 60
    )

    # 篩選有效序列對：同 anon_id + gap ∈ [0, session_gap_minutes]
    valid_mask = (
        event_df["next_behavior"].notna() &
        (event_df["gap_minutes"] >= 0) &
        (event_df["gap_minutes"] <= session_gap_minutes)
    )
    pairs = event_df[valid_mask][
        ["anon_id", "behavior", "timestamp", "next_behavior", "gap_minutes", "semester"]
    ].copy()
    pairs["gap_minutes"] = pairs["gap_minutes"].round(2)

    n_pairs = len(pairs)
    print(f"  [09] LSA event log：{n_events} 事件，{n_pairs} 有效序列對，{n_students} 位學生")
    return pairs.reset_index(drop=True)


# ── 工具函式 ─────────────────────────────────────────────────────

def _empty_log() -> pd.DataFrame:
    return pd.DataFrame(columns=["anon_id", "behavior", "timestamp", "next_behavior", "gap_minutes", "semester"])


def _extract_semester_col(df: pd.DataFrame) -> pd.Series:
    """
    取來源列自身的 _semester（_merge_lms_files() 依檔名解析寫入）。
    缺欄時安全回傳空字串序列，維持與既有 by_semester 分組的 graceful degradation
    （10_lsa_transition.compute_lsa_by_semester 會略過空字串學期）。
    """
    if "_semester" in df.columns:
        return df["_semester"].astype(str).fillna("")
    return pd.Series([""] * len(df), index=df.index)


def _detect_id_col(df: pd.DataFrame) -> str:
    """優先 anon_id，次選帳號/學號欄"""
    if "anon_id" in df.columns:
        return "anon_id"
    for col in df.columns:
        if "帳號" in col or "學號" in col:
            return col
    return ""


def _detect_time_col(df: pd.DataFrame, candidates: tuple) -> str:
    """依優先順序找時間欄"""
    for name in candidates:
        if name in df.columns:
            return name
    # fallback：欄名含「時間」「日期」「time」「date」（不分大小寫）
    for col in df.columns:
        lower = col.lower()
        if any(k in lower for k in ("時間", "日期", "time", "date")):
            return col
    return ""


def _detect_quiz_title_col(df: pd.DataFrame) -> str:
    """測驗標題欄優先，fallback 活動名稱等"""
    for name in ("測驗標題", "活動名稱", "標題", "名稱"):
        if name in df.columns:
            return name
    return ""
