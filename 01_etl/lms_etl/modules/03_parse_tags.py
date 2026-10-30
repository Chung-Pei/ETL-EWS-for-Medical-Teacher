"""
lms_etl/modules/03_parse_tags.py
解析教材標題中的方括號標籤，歸類為六大類別代碼。
依規格書 V4.0 Section 3 定義。
"""

import re
from typing import Optional

import pandas as pd


# ── 六大類別代碼 ──
CATEGORIES = ("AUD", "VID", "TXT", "SUP", "TUT", "QUZ")

# ── 標籤映射（依優先順序排列，先長後短）──
#
# v1.1（/systematic-debugging，依課程負責人確認的語意套用，見26號交接文件
#      《ETL交接：03_parse_tags.py 標籤覆蓋率缺口》決議）：
#   FIX-L  TAG含「筆記」（不限定「整理筆記」四字）一律歸SUP，改用子字串
#          比對取代原本的精確比對（原[整理筆記]精確規則已被此規則涵蓋）。
#   FIX-M  TAG含「補充」（不限定「補充教材」四字）一律歸SUP，改用子字串
#          比對取代原本的精確比對（原[補充教材]精確規則已被此規則涵蓋）。
#          【注意：未採用同樣手法處理「影片」子字串，因為[影片問題]（QUZ）
#          與[字幕影片檔]（VID）皆含「影片」二字，若比照做成「含影片即
#          SUP」的子字串規則，會誤把這兩條既有規則的分類結果搶走，
#          因此「教學影片」僅新增精確比對，不做子字串擴展。】
#   FIX-N  新增[教學影片]精確比對 → SUP（與既有[教學動畫]/[動畫解析]同組）。
#   FIX-O  新增「[寄生蟲] N.1 醫學寄生蟲學-」（N為數字）→ VID：此為醫學
#          寄生蟲學單元的影音教材，標題不遵循一般「[媒體類型]」標籤慣例，
#          而是用科目名稱當標籤，故需要專屬規則（比對範圍延伸到方括號
#          之後的章節編號與科目名稱文字，不只比對方括號本身）。
#   FIX-P  新增「N.1 醫學寄生蟲學-」（無方括號前綴、N為數字）→ TXT：
#          同一寄生蟲單元下，沒有方括號前綴的版本是文字教材（與FIX-O的
#          影音版本並存，兩者標題僅差在有無「[寄生蟲]」前綴）。此規則
#          是全表唯一不以「\[」開頭的規則，因為它專門比對「無方括號」
#          的情況，前置的digit要求（^\d+）已足以與其他規則區隔，不會
#          誤比對到任何以「[」開頭的既有標籤。
#
# v1.2（/systematic-debugging 全庫穿透式審查，2026-07，FIX-O/FIX-P廣義化）：
#   真實資料（8學期activity+reading_stats全量掃描）發現FIX-O/FIX-P原本
#   要求的章節編號格式「\d+\.1」（單一數字+.1）沒有涵蓋合併章節的命名
#   慣例：「2.1 3.1 醫學寄生蟲學-原蟲 114rev」（合併第2、3章修訂版，
#   481筆）與「4.5.6.1 醫學寄生蟲學-蠕蟲」（合併第4-6章，374筆），
#   兩者章節編號前綴含多組數字/多個句點，與「\d+\.1」不匹配，全數落入
#   UNTAGGED（結構上與其他已正確歸類的「N.1 醫學寄生蟲學-...」文字教材
#   完全相同，只是合併了多章內容，不是新的媒體類型，故沿用FIX-P已由
#   課程負責人確認的TXT歸屬，非新增語意決策）。改用`[\d.\s]+`（一段
#   由數字、句點、空白組成的字元）取代`\d+\.1`，涵蓋任意章節編號組合，
#   不再需要為未來可能出現的第3、4種章節編號寫法逐一新增規則──與
#   本輪同時發現的題庫週次死區bug（06_merge_grades.py）採用「日期反推
#   取代標題字串規則」是同一種「不再對標題格式做封閉式假設」的修正
#   思路。已用真實資料驗證：廣義化後，這2組合計855筆真實記錄改為TXT，
#   且已確認接觸過這2個合併檔的全部學生（114-2共249人、114-1學後護一
#   共51人）皆同時有其他TXT類別教材紀錄，material_diversity_score不受
#   影響，R群分類無變動（見CHANGELOG_v7.md v7.14.8章節驗證明細）。
TAG_RULES: list[tuple[str, str]] = [
    (r"\[課程podcast\]",    "AUD"),
    (r"\[上課錄音檔\]",      "AUD"),
    (r"\[字幕影片檔\]",      "VID"),
    (r"\[上課EverCam檔\]",  "VID"),
    (r"\[上課Powercam檔\]", "VID"),
    (r"\[寄生蟲\]\s*[\d.\s]+醫學寄生蟲學-", "VID"),  # FIX-O（v1.2廣義化）
    (r"\[上課PPT\]",        "TXT"),
    (r"\[參考教材\]",        "TXT"),
    (r"\[延伸閱讀\]",        "TXT"),
    (r"^[\d.\s]+醫學寄生蟲學-", "TXT"),               # FIX-P（唯一不以\[開頭的規則；v1.2廣義化）
    (r"\[[^\]]*筆記[^\]]*\]", "SUP"),                  # FIX-L（含子字串「筆記」）
    (r"\[[^\]]*補充[^\]]*\]", "SUP"),                  # FIX-M（含子字串「補充」）
    (r"\[動畫解析\]",        "SUP"),  # 長標籤優先於 [動畫]
    (r"\[教學動畫\]",        "SUP"),
    (r"\[動畫\]",           "SUP"),
    (r"\[教學影片\]",        "SUP"),  # FIX-N
    (r"\[TA輔導\]",         "TUT"),
    (r"\[題庫\]",           "QUZ"),
    (r"\[影片問題\]",        "QUZ"),
]

# ── 行政/排除類標籤（歸入 UNTAGGED，不計入六大類統計）──
ADMIN_TAG_RULES: list[str] = [
    r"\[重要\]",
]

# ── 預編譯正規表示式 ──
_COMPILED_RULES = [(re.compile(pat, re.IGNORECASE), cat) for pat, cat in TAG_RULES]
_COMPILED_ADMIN = [re.compile(pat, re.IGNORECASE) for pat in ADMIN_TAG_RULES]

# 提取首個方括號標籤（供 extract_bracket_tag 使用）
_RE_ANY_BRACKET = re.compile(r"^\[([^\]]+)\]")


def parse_tag(title: str) -> str:
    """
    解析教材標題，回傳類別代碼。

    規則（依優先順序）：
      1. 非字串或空值           → "UNTAGGED"
      2. 行政標籤（如 [重要]）   → "UNTAGGED"
      3. 比對 TAG_RULES         → 對應類別代碼
      4. 無匹配                 → "UNTAGGED"

    Bug fix：多標籤標題（如 "[重要][上課PPT] xxx"）原本會因 ADMIN 匹配而
    提早回傳 UNTAGGED；此版本行政標籤仍保持優先排除語意，符合規格書設計。
    若需「跳過行政標籤並繼續比對」，請修改此處邏輯。

    Parameters
    ----------
    title : 教材標題字串

    Returns
    -------
    str : "AUD"/"VID"/"TXT"/"SUP"/"TUT"/"QUZ"/"UNTAGGED"
    """
    if not isinstance(title, str) or not title.strip():
        return "UNTAGGED"

    title = title.strip()

    for pat in _COMPILED_ADMIN:
        if pat.match(title):
            return "UNTAGGED"

    for pat, cat in _COMPILED_RULES:
        if pat.match(title):
            return cat

    return "UNTAGGED"


def extract_bracket_tag(title: str) -> Optional[str]:
    """提取首個方括號標籤內容（如 "[上課PPT] Ch1" → "上課PPT"）"""
    if not isinstance(title, str) or not title.strip():
        return None
    m = _RE_ANY_BRACKET.match(title.strip())
    return m.group(1) if m else None


def categorize_activity_df(df: pd.DataFrame, title_col: str = "活動紀錄") -> pd.DataFrame:
    """
    對活動紀錄 DataFrame 新增 category、bracket_tag 欄位。

    最佳化：合併為單次 apply，減少重複迭代。

    Parameters
    ----------
    df        : 活動紀錄 DataFrame（需含 title_col 欄位）
    title_col : 教材標題欄位名稱（預設 "活動紀錄"）

    Returns
    -------
    DataFrame（新增 category、bracket_tag 兩欄，不修改原始 df）
    """
    df = df.copy()

    def _parse_both(title: str) -> tuple[str, Optional[str]]:
        return parse_tag(title), extract_bracket_tag(title)

    parsed = df[title_col].apply(_parse_both)
    df["category"]    = parsed.map(lambda x: x[0])
    df["bracket_tag"] = parsed.map(lambda x: x[1])
    return df


def count_materials_by_category(
    activity_df: pd.DataFrame,
    title_col: str = "活動紀錄",
) -> dict[str, int]:
    """
    統計各類別教材總件數（以標題去重）。

    Bug fix：原版在 activity_df 缺少 category 欄時，直接對傳入物件賦值
    導致 side-effect；此版本改為建立本地副本，保持純函式語意。

    Returns
    -------
    dict : { "AUD": 12, "VID": 38, ..., "UNTAGGED": 3 }
    """
    # 純函式：使用本地副本，不修改呼叫方的 DataFrame
    local_df = categorize_activity_df(activity_df, title_col)

    unique_materials = local_df[[title_col, "category"]].drop_duplicates(subset=title_col)
    counts = unique_materials["category"].value_counts().to_dict()

    result: dict[str, int] = {cat: 0 for cat in CATEGORIES}
    result["UNTAGGED"] = 0
    result.update(counts)
    return result


# ── 快速驗證 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases: list[tuple[str, str]] = [
        ("[上課PPT] Ch 1. 微生物學緒論",           "TXT"),
        ("[上課EverCam檔] Ch 1. 微生物學緒論",     "VID"),
        ("[動畫] 工作細胞 1- 肺炎鏈球菌",           "SUP"),
        ("[動畫解析] 工作細胞 3- 巨噬與T細胞活化",  "SUP"),
        ("[教學動畫] 單株抗體 與 抗癌PD-1",        "SUP"),
        ("[上課Powercam檔] Ch 1. 微生物學緒論",    "VID"),
        ("[題庫] 第1週 練習題庫",                  "QUZ"),
        ("[TA輔導] 微生物期中考範圍",               "TUT"),
        ("[整理筆記] Ch2. 參與免疫反應的細胞",      "SUP"),
        ("[上課錄音檔] 免疫Ch 1",                  "AUD"),
        ("[課程podcast] 第一週",                   "AUD"),
        ("[延伸閱讀] 疫苗簡史 by BBC",             "TXT"),
        ("[重要] 微免學正課 修課同意書",             "UNTAGGED"),
        ("沒有標籤的教材",                         "UNTAGGED"),
    ]

    print("標籤解析測試：")
    all_pass = True
    for title, expected in test_cases:
        result = parse_tag(title)
        ok = result == expected
        if not ok:
            all_pass = False
        print(f"  {'✓' if ok else '✗'}  [{result:8s}] {title[:45]}")

    print(f"\n{'✅ 全部通過' if all_pass else '❌ 有測試失敗，請檢查 TAG_RULES'}")
