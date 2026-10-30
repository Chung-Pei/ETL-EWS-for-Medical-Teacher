"""
lms_etl/modules/07e_transition_micro_reading.py
基礎醫學臨床應用(一)合併課微免部分8小時教材閱讀門檻計算（見規劃書1.5節／
討論8，實作階段依真實資料確認後定案）。

性質（重要）：這不是給早期預警參考用的輔助清單，而是校方成績作業會參考
的決策支援警示——但系統本身「不」自動修改或計算調整後成績，只提供
met_8hr_threshold 判定結果，由授課教師人工在送出成績前自行套用×0.56
（見下方「NO AUTOMATED GRADE MUTATION」說明）。

── 設計確認記錄（實作階段，穿透式審查後定案）──────────────
資料來源：「教材閱讀次數及時間」sheet（01_load_lms_excel.py 內部key為
reading_stats），非「教材閱讀紀錄」（reading_log）。經 01_load_lms_excel.py
（全程使用 pandas.read_excel()）實際解析真實LMS匯出檔確認，reading_stats
欄位為：項次/帳號/姓名/教材標題/閱讀次數/閱讀時間，其中「閱讀時間」為
逐教材加總後的累計時長（HH:MM:SS或MM:SS格式）。

（附註：實作過程中曾用獨立診斷腳本以 openpyxl read_only 模式讀取同一份
檔案，誤判為只有4欄、無時長資料——該模式對此類LMS平台匯出檔的
dimension中繼資料判讀有誤，與 01_load_lms_excel.py 實際使用的
pandas.read_excel() 讀取路徑不同，經比對真實截圖確認後修正。）

計算邏輯：教材閱讀不做週次切分，改用「教材標題類別白名單」（10個標準
分類前綴，對應授課教師多年來的教材命名慣例）篩選、直接加總reading_stats
「閱讀時間」欄。因合併課學生本來就只在最後6週修習這門課，不需要額外用
週次二次確認範圍——course_track欄本身已提供課程層級隔離（見1.1節），
類別前綴白名單提供教材層級隔離（排除非本課程性質的雜項，如公告、其他
科目教材）。
────────────────────────────────────────────────────────
"""

import re

import pandas as pd

READING_HOUR_THRESHOLD = 8.0
# 手動調整倍率（僅供UI提示教師參考，系統本身不套用，見下方
# build_shortfall_records() docstring「NO AUTOMATED GRADE MUTATION」）。
MANUAL_ADJUSTMENT_MULTIPLIER_HINT = 0.56

# 教材標題類別白名單（見規劃書1.5節最終規格，逐字比對，前綴需完全符合方括號格式）
#
# [BUG-FIX 穿透式審查 0912，真實資料測試發現] "[上課Evercam檔]"（小寫c）
# 原本無法比對到真實LMS匯出檔的實際標題 "[上課EverCam檔]"（大寫C）——
# 本白名單設計為逐字、區分大小寫的精確比對（見上方docstring「逐字比對」
# 說明），不是遺漏IGNORECASE，是字面拼寫本身就跟其他3處既有程式碼
# （03_parse_tags.py TAG_RULES、04_feature_engineering.py evercam_mask、
# test_parse_tags_coverage_v7_14_4.py）不一致——那3處全部正確拼成
# "EverCam"，只有本檔案這裡拼錯，屬於「同一份字面字串在多處各自維護、
# 其中一處拼寫走鐘」的同類問題，而非邏輯分支各自維護。
#
# 影響：真實114(2)資料實測，"[上課EverCam檔]" 類別有4,378筆閱讀紀錄
# （與白名單最大類別「上課PPT」5,728筆、「字幕影片檔」4,577筆同一量級，
# 非可忽略小類別）。比對不到會讓這部分閱讀時數完全不計入8小時門檻累計，
# 導致系統性低估每位學生的累計閱讀時數，可能誤判已達標學生為未達標。
# 已對13個真實歷史LMS檔案（111-1～114-2）逐一確認，標題一律固定為
# "EverCam"（大寫C），無例外，故直接修正字面拼寫，不改用IGNORECASE
# （維持既有「逐字比對」設計，避免意外比對到非預期的其他標題）。
READING_CATEGORY_PREFIXES = (
    "[上課PPT]", "[上課EverCam檔]", "[字幕影片檔]", "[課程Podcast]",
    "[延伸閱讀]", "[題庫]", "[整理筆記]", "[動畫]", "[動畫解析]",
    "[TA輔導]", "[教學動畫]",
)


def compute_reading_minutes_by_student(
    reading_stats_df: pd.DataFrame,
    hms_to_minutes_fn,
    course_track: str = "transition_merged_micro",
    category_prefixes: tuple = READING_CATEGORY_PREFIXES,
) -> dict:
    """
    計算每位學生「教材閱讀紀錄」中，標題符合 category_prefixes 白名單、
    且 course_track 符合指定值的教材累計閱讀分鐘數。不做週次切分（見上方
    模組docstring的設計變更記錄）。

    hms_to_minutes_fn：時間字串轉分鐘數的函式，呼叫端應傳入
    05_time_analysis.py::_hms_to_minutes()（v7.14穿透式審查修正過
    MM:SS格式被靜默歸零的bug，此處刻意不重新實作、避免同一類bug
    在兩處各自維護、日後修一邊漏一邊）。

    回傳：{(anon_id, semester): total_minutes} —— 複合鍵，不能只用
    anon_id單獨當key。同一學生若跨學期出現（例如重修生），reading_stats_df
    可能同時含多個學期的資料列，只用anon_id分組會把不同學期的閱讀時數
    誤加總在一起（比照 lms_etl.py::_merged_index_2key 既有的
    (anon_id, semester) 複合鍵慣例，見規劃書實作階段附註）。

    注意：真實匯出檔的「教材閱讀紀錄」sheet若無獨立時長欄位，本函式會
    回傳空結果並印出警告，不用估算值冒充真實計算結果（這類資料直接影響
    教師是否手動調整成績的判斷，寧可明確告知「算不出來」）。
    """
    empty_result: dict = {}
    if reading_stats_df is None or reading_stats_df.empty:
        return empty_result

    id_col = next((c for c in ("anon_id", "帳號") if c in reading_stats_df.columns), None)
    title_col = next((c for c in ("教材標題", "title") if c in reading_stats_df.columns), None)
    sem_col = "_semester" if "_semester" in reading_stats_df.columns else None
    if not id_col or not title_col or not sem_col:
        print("  ⚠️  [07e] reading_stats 缺少學號欄／教材標題欄／_semester欄，無法計算閱讀時數")
        return empty_result

    df = reading_stats_df
    if "_course_track" in df.columns:
        df = df[df["_course_track"] == course_track]
    if df.empty:
        return empty_result

    prefix_re = re.compile(
        "^(?:" + "|".join(re.escape(p) for p in category_prefixes) + ")"
    )
    titles = df[title_col].astype(str)
    matched = df[titles.str.match(prefix_re)]
    if matched.empty:
        return empty_result

    # 目標欄位為「閱讀時間」（見規劃書實作階段：01_load_lms_excel.py 以
    # pandas.read_excel() 正確解析出「教材閱讀次數及時間」sheet的6欄結構
    # 項次/帳號/姓名/教材標題/閱讀次數/閱讀時間，非本模組先前版本誤判的
    # 4欄——該誤判源自獨立診斷腳本用 openpyxl read_only 模式讀取，該模式
    # 對本檔案的 dimension 中繼資料失準，與實際production pipeline
    # （01_load_lms_excel.py，全程使用 pandas.read_excel）的行為不同）。
    dur_col = "閱讀時間" if "閱讀時間" in matched.columns else None
    if dur_col:
        minutes = matched[dur_col].apply(hms_to_minutes_fn)
    else:
        print(f"  ⚠️  [07e] reading_stats 沒有「閱讀時間」欄，無法計算精確閱讀分鐘數"
              f"（共 {len(matched)} 筆符合類別白名單但無法加總時長的記錄）。"
              f"請確認呼叫端傳入的是 01_load_lms_excel.py 解析出的 reading_stats"
              f"（'教材閱讀次數及時間' sheet），而非其他來源。")
        return empty_result

    grouped = matched.assign(_minutes=minutes.values).groupby([id_col, sem_col])["_minutes"].sum()
    return {(aid, str(sem)): mins for (aid, sem), mins in grouped.items()}


def build_shortfall_records(
    reading_minutes_by_student: dict,
    grade_records: list,
    threshold_hours: float = READING_HOUR_THRESHOLD,
) -> list:
    """
    合併「教材閱讀累計時數」與「成績檔Test III raw分」，產出每位學生的
    門檻判定結果。

    ── NO AUTOMATED GRADE MUTATION（見規劃書最終規格第3節）──
    本函式不計算、不建議、也不輸出任何「調整後成績」。raw_grade 原封不動
    保留自 grade_records，系統的角色僅止於「met_8hr_threshold 判定 +
    UI警示」，實際送出成績前的×0.56計算由授課教師人工執行——這是刻意的
    設計決策，不是遺漏。

    reading_minutes_by_student : dict
        由 compute_reading_minutes_by_student() 產出，key 為
        (anon_id, semester) 複合鍵（不是單一 anon_id，見該函式說明）。

    grade_records：list of dict，至少需含 anon_id、semester（複合鍵比對
    必需，見上方）、semester_score（Test III raw分，0-100）、姓名／
    masked_id（可選，供CSV輸出的人工識別欄用）。通常取自 lms_etl.py
    主流程的 students list（已由 (anon_id, semester) 正確合併成績檔資料，
    見規劃書1.1節）。

    回傳每筆：{student_id, anon_id, name, cumulative_reading_hours,
              met_8hr_threshold, raw_grade}
    （anon_id為36號規格書第八節新增：前端逐生清單於14天揭示窗口內
    需要用它反查原始學號，student_id欄位本身在窗口內外都維持顯示
    masked_id不變，改由前端動態決定要不要疊加顯示原始學號。）

    BUG-FIX（v7.15.13穿透式審查／真實資料測試發現）：原本 student_id
    欄位直接沿用查表用的 anon_id（8碼雜湊值，如"0757ce72"），對照下
    "name" 欄位卻因 lms_etl.py 主流程 students list 內只有 masked_id
    （如"990****58"）、無姓名欄，fallback 借用了 masked_id——導致輸出
    CSV/JSON「學號」欄放的是教師無法辨識的雜湊值，「姓名」欄放的卻是
    看起來像學號的遮蔽學號，兩欄語意對調。這份清單的存在目的正是要讓
    授課教師人工核對真實學生、於送出成績前手動調整（見上方docstring
    NO AUTOMATED GRADE MUTATION說明），student_id若是無法對照的雜湊值，
    教師拿到清單根本無法找到對應學生，這份警示清單就失去實際作用。
    改為：查表比對仍固定使用真正的 anon_id（reading_minutes_by_student
    的 key 格式，不可替換，否則全部查表會落空、時數全部算成0）；輸出
    欄位則優先採用人類可辨識的 masked_id 當 student_id 顯示值，
    grade_records 若未來真的提供 student_id（例如改接未去識別化的來源）
    則優先採用該值，最後才 fallback 回 anon_id（維持「查不到人類可讀
    識別碼時，至少不中斷、仍給一個值」的既有防呆慣例）。
    """
    records = []
    n_skipped_wrong_track = 0
    n_skipped_no_semester = 0
    for g in grade_records:
        track = g.get("course_track")
        if "course_track" in g and track != "transition_merged_micro":
            n_skipped_wrong_track += 1
            continue

        aid = g.get("anon_id")
        sem = g.get("semester")
        if not aid:
            continue
        if not sem:
            # 複合鍵比對必需學期資訊；缺學期時無法安全比對，寧可跳過
            # 這筆也不要誤用「只有anon_id」的舊式查表方式（那正是本次
            # 要修正的跨學期誤加總風險來源）。
            n_skipped_no_semester += 1
            continue
        total_minutes = reading_minutes_by_student.get((aid, str(sem)), 0.0)
        cumulative_reading_hours = round(total_minutes / 60.0, 2)
        met_8hr_threshold = cumulative_reading_hours >= threshold_hours

        # 顯示用識別碼：人類可辨識優先（masked_id），非查表用途，可與
        # 上面 aid（查表用，固定anon_id）不同值。
        display_id = g.get("student_id") or g.get("masked_id") or aid

        # v7.15.23修正：g.get("semester_score")在pandas合併後缺值常是
        # float('nan')而非None——若原封不動寫入raw_grade，json.dump()會
        # 輸出字面NaN token，這不是合法JSON，瀏覽器JSON.parse()/fetch().
        # json()會直接拋錯、整份清單完全載入失敗（不是顯示異常，是完全
        # 讀不到）。checkpoint版因為門檻設計本來就會有多數學生raw_grade
        # 缺值，這個bug幾乎每次checkpoint執行都會觸發，故在此源頭修正，
        # CSV/JSON兩邊消費端不需要各自處理。
        _raw_grade = g.get("semester_score")
        if pd.isna(_raw_grade):
            _raw_grade = None

        records.append({
            "student_id":               display_id,
            "anon_id":                  aid,  # 36號規格書第八節+37號規劃書3.3節：供前端
                                                # resolveRawStudentId(s.anon_id) 於14天窗口內
                                                # 反查原始學號用；查表鍵值固定用 aid，不可替換。
            "name":                     g.get("name") or g.get("姓名") or g.get("masked_id") or "",
            "cumulative_reading_hours": cumulative_reading_hours,
            "met_8hr_threshold":        met_8hr_threshold,
            "raw_grade":                _raw_grade,  # NaN已正規化為None，見上方註解
        })

    records.sort(key=lambda r: str(r["student_id"]))
    if n_skipped_wrong_track:
        print(f"  ⚠️  [07e] {n_skipped_wrong_track} 筆記錄的 course_track 不是 "
              f"transition_merged_micro，已排除於8小時門檻計算之外（防呆，"
              f"理論上呼叫端應已預先篩過，若看到此訊息請檢查上游篩選邏輯）")
    if n_skipped_no_semester:
        print(f"  ⚠️  [07e] {n_skipped_no_semester} 筆記錄缺少 semester 欄位，"
              f"無法安全比對 (anon_id, semester) 複合鍵，已跳過")
    return records


def export_shortfall_json(records: list, semester: str, threshold_hours: float = READING_HOUR_THRESHOLD,
                           grading_complete: bool = True, graded_count: int = None) -> dict:
    """
    組裝 micro_immuno_reading_shortfall_{semester}.json 的內容（見規劃書
    1.5節，給儀表板／at-risk-report.js顯示用）。呼叫端負責實際寫檔。

    manual_adjustment_multiplier_hint：僅供前端UI提示文字引用（例如
    「未滿8小時，送出成績前請手動×0.56」），系統本身不會用這個值計算
    任何欄位，見 build_shortfall_records() 的 NO AUTOMATED GRADE MUTATION
    說明。

    [BUG-FIX 穿透式審查 0913，真實pipeline+Playwright真瀏覽器測試發現]
    grading_complete / graded_count：新增欄位，修正一個「期末版遮蔽
    checkpoint版」的實際阻斷bug——本函式（Phase 5）呼叫端只要偵測到
    course_track=="transition_merged_micro"學生就會無條件產生這份檔案，
    不論Test III成績是否已登錄（這是Phase 5從36號規格書時代就有的既有
    設計，本身沒有問題）。但前端at-risk-report.js的
    「期末版存在時checkpoint版隱藏」判斷（決策2026-09-10）原本只檢查
    「這份檔案有沒有成功載入」，沒有檢查「這份檔案代表的是不是真的已
    結束的學期」——兩者的隱含假設對不上：前端以為「Phase 5檔案存在」
    等於「學期已結束」，但Phase 5其實從學期第一天、Test III成績都還是
    空的時候就會產生檔案。實測：用16.7%成績填寫率的模擬資料跑完整
    pipeline，Phase 5與Phase 5b（checkpoint）兩份檔案同時產生，再用
    Playwright載入真瀏覽器渲染，畫面固定顯示Phase 5的「逐生清單」標題、
    checkpoint的「期中進度快照」完全不會顯示——checkpoint功能因此在
    正常操作情境下形同無法觸達，不是罕見邊界案例。修正：Phase 5輸出時
    一併呼叫既有的 checkpoint_graded_ratio()（v7.15.24已抽成共用函式，
    這裡直接重用、不重寫判斷式）算出同一份「是否已完結」訊號寫入這裡，
    前端的hasFinal判斷改成同時檢查這個欄位（見at-risk-report.js對應
    修正），而不是只看檔案是否成功fetch。
    """
    shortfall = [r for r in records if not r["met_8hr_threshold"]]
    result = {
        "semester":                          semester,
        "threshold_hours":                   threshold_hours,
        "manual_adjustment_multiplier_hint":  MANUAL_ADJUSTMENT_MULTIPLIER_HINT,
        "generated_scope":                   "教材標題類別白名單（見 READING_CATEGORY_PREFIXES），不做週次切分",
        "total_students":                    len(records),
        "shortfall_count":                   len(shortfall),
        "students":                          records,
        "grading_complete":                  grading_complete,
    }
    if graded_count is not None:
        result["graded_students"] = graded_count
    return result


def export_shortfall_csv_rows(records: list, threshold_hours: float = READING_HOUR_THRESHOLD) -> list:
    """
    組裝可直接複製貼上到Excel的清單列（依學號排序，欄位：學號、姓名、
    累計閱讀時數、是否達標、Test III原始成績）。呼叫端負責實際寫檔。

    刻意不含「建議調整後分數」欄——調整計算由教師人工執行，系統不提供
    計算結果（見 build_shortfall_records() 的 NO AUTOMATED GRADE MUTATION
    說明），避免CSV欄位本身造成「系統已經算好、照抄即可」的誤導。

    [AUDIT-FIX 穿透式審查 0908第三輪] threshold_hours原本沒有這個參數，
    表頭「是否達標8小時」的「8」是寫死的文字——跟同檔案內
    build_shortfall_records()／export_shortfall_json()這兩個姊妹函式
    都已經是「threshold_hours: float = READING_HOUR_THRESHOLD」的預設
    參數寫法不一致，是本專案這輪已經修過數次的同一類「同一條規則各自
    維護導致跑掉」問題（READING_HOUR_THRESHOLD若調整，這裡的表頭文字
    不會跟著變）。改為比照姊妹函式的預設參數寫法；目前唯一呼叫點
    （lms_etl.py Phase 5）沒有明確傳入threshold_hours，行為不變（依然
    是READING_HOUR_THRESHOLD＝8.0），純粹補上一致性與未來調整時的
    正確性保障。
    """
    header = ["學號", "姓名", "累計閱讀時數", f"是否達標{threshold_hours:g}小時", "Test III原始成績"]
    rows = [header]
    for r in records:
        rows.append([
            r["student_id"],
            r["name"],
            r["cumulative_reading_hours"],
            "是" if r["met_8hr_threshold"] else "否",
            r["raw_grade"] if pd.notna(r["raw_grade"]) else "Unfilled",
        ])
    return rows


def checkpoint_graded_ratio(cp_grade_records: list, threshold_ratio: float = 0.5) -> tuple:
    """
    判斷checkpoint是否應該略過產生（見規格書2026-09-10/12決策）。

    這門課全學期只考一次Test III、無期中考（見etl.py
    parse_transition_micro_sheet()），semester_score欄只要非null就代表
    Test III已出成績、不會提早/漸進式被填。超過threshold_ratio比例的
    學生已有分數，視為本學期已完結，checkpoint不再有意義。

    抽成獨立函式（v7.15.24）而不是留在lms_etl.py run()裡的inline判斷式，
    是因為原本的寫法只能靠測試檔另外「照抄」同一段公式來驗證——那份
    抄本若跟這裡日後改動不同步（例如改門檻、改pd.notna()判斷方式），
    測試會繼續通過但測到的其實不是真正在跑的邏輯。抽成函式後測試檔
    直接import這個函式，兩邊不會分開維護走鐘。

    g.get("semester_score")在pandas合併後缺值常是float('nan')而非
    None，用pd.notna()正確處理兩種缺值表示法。

    回傳 (should_skip, graded_count, total_count)。
    """
    total_count = len(cp_grade_records)
    graded_count = sum(1 for s in cp_grade_records if pd.notna(s.get("semester_score")))
    ratio = (graded_count / total_count) if total_count else 0.0
    return (ratio > threshold_ratio, graded_count, total_count)


def build_progress_checkpoint_records(
    reading_stats_df: pd.DataFrame,
    hms_to_minutes_fn,
    grade_records: list,
    threshold_hours: float = READING_HOUR_THRESHOLD,
) -> list:
    """
    見規劃書1.5節「已確認需要」的提前預警（highlight）版本：學期進行中
    （建議於week17結束、進入week18考試週前）先跑一次進度提醒，讓時數
    明顯落後的學生還有機會在最後一週補課。

    設計變更（配合上方模組docstring的重新設計）：原版本靠「week16_end」
    日期窗口區分「進度快照」與「期末最終」兩種計算。新設計不做週次/日期
    切分，教材累計時數的計算邏輯與 build_shortfall_records() 完全相同；
    「進度快照」與「期末最終」的差異純粹在於「何時執行」——LMS匯出檔本身
    只會包含「執行當下已發生」的存取紀錄，故學期中執行本函式自然只會
    看到當時為止的累計進度，不需要額外的日期篩選邏輯。

    本函式實質上是 compute_reading_minutes_by_student() +
    build_shortfall_records() 的薄包裝，差異只在於：
    1) 額外標記 is_checkpoint=True，供前端／CSV輸出區分這不是期末最終清單
    2) 額外提供 hours_remaining_needed（距離門檻還差幾小時，達標者為0）

    刻意不做「數學上絕對不可能達標」的判定：是否「不可能」達標，取決於
    學生剩餘時間內最多能讀多少小時，這沒有一個可以直接算出的硬性上限
    （不像「總分數上限100分」那樣有明確邊界）——非常投入的學生仍可能在
    最後幾天大量閱讀追上進度。故只回報事實性進度資訊，交由使用者（授課
    教師）自行判斷是否已經來不及、要不要主動提醒特定學生。
    """
    minutes_by_student = compute_reading_minutes_by_student(reading_stats_df, hms_to_minutes_fn)
    records = build_shortfall_records(minutes_by_student, grade_records, threshold_hours=threshold_hours)
    for r in records:
        r["is_checkpoint"] = True
        r["hours_remaining_needed"] = max(
            0.0, round(threshold_hours - r["cumulative_reading_hours"], 2)
        )
    return records
