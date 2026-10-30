"""
lms_etl/modules/02_anonymize.py
學生去識別化：學號 → SHA-256(salt + sid)[:8] → anon_id
同時產出 lookup_table（本地保留，不上傳 git）。
"""

import re
import hashlib
import json
import os
import pandas as pd


# 學號格式：8 或 9 位純數字（一般學制），或 09 開頭 10 位純數字
# （推廣中心進修教育組學生的帳號採用手機號碼格式）。
#
# v1.2（/systematic-debugging 全庫穿透式審查，2026-07）：真實資料（8學期
# summary sheet全量掃描，2,702個不重複帳號）發現5筆帳號被原規則
# （僅8-9位）判定為無效、遭`anonymize_dataframe()`整批過濾捨棄——並非
# 髒資料，而是「推廣中心進修教育組」這個獨立招生管道的在學學生，其
# 帳號欄位登記的是手機號碼（09開頭10位數），與「護理系」等一般學制
# 8-9位學號並存於同一份 summary/activity/reading_stats/quiz 資料表。
# 這5名學生每人皆有完整一學期的真實學習紀錄（每人100-800+列不等），
# 原規則會讓這些學生完全消失於整個pipeline（不會被納入anon_id/students
# 清單，也就不會出現在任何R群分類、相關係數矩陣、早期預警模型輸出中）
# ——與hms bug／TAG_RULES／題庫週次三個bug本質上是同一種「格式白名單
# 覆蓋不全，真實資料靜默漏計」問題，只是這次影響範圍是「整個學生」
# 而非「部分紀錄列」，嚴重度更高。
#
# 已用真實資料確認：全部5筆不合規帳號皆為10位數、開頭皆為"09"、
# 且對應的「單位」欄位皆為「推廣中心進修教育組」，其餘所有跨系選修
# 學生（健康事業管理系／嬰幼兒保育系／語言治療與聽力學系）皆為標準
# 9位學號、不受影響。改用「09開頭10位」而非「放寬為任意10位」，
# 是為了不放寬過頭意外收下未來可能出現的其他格式髒資料。
RE_STUDENT_ID = re.compile(r"^(?:\d{8,9}|09\d{8})$")


def load_salt(config_path: str = "config/salt.txt") -> str:
    """
    讀取固定鹽值。
    檔案不存在或仍為預設值時拋出 ValueError，防止誤用。
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"鹽值檔案不存在：{config_path}\n"
            "請建立 config/salt.txt 並填入隨機字串（至少 32 字元）。"
        )

    # [FIX] 移除 utf-16：salt 純文字不會使用 UTF-16；保留常用編碼即可
    for enc in ("utf-8-sig", "utf-8", "cp950"):
        try:
            with open(config_path, "r", encoding=enc) as f:
                salt = f.read().strip()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        raise ValueError(
            "config/salt.txt 編碼無法辨識（嘗試過：utf-8-sig / utf-8 / cp950）。\n"
            "請以 UTF-8（無 BOM）重新儲存檔案。"
        )

    if not salt or "REPLACE_THIS" in salt:
        raise ValueError(
            "config/salt.txt 尚未設定！\n"
            "請將內容替換為隨機字串（至少 32 字元），例如：\n"
            "  python -c \"import secrets; print(secrets.token_hex(32))\" > config/salt.txt"
        )
    return salt


def anonymize(student_id: str, salt: str) -> str:
    """
    將學號轉為 8 碼十六進位匿名代碼。
    同一 salt + 同一 student_id 永遠產出相同結果（跨學期可比對）。
    """
    raw = (salt + student_id.strip()).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:8]


def mask_student_id(sid: str) -> str:
    """
    產生遮蔽學號：前 3 碼 + **** + 後 2 碼，不足 5 碼時全部遮蔽。
    [FIX] 先 normalize 再遮蔽，避免 float 格式殘留（如 "141119002.0"）。
    """
    sid = normalize_student_id(sid)
    return (sid[:3] + "****" + sid[-2:]) if len(sid) >= 5 else "*" * len(sid)


def normalize_student_id(raw) -> str:
    """
    統一學號格式：去掉小數點（xlrd 可能讀成 float）。
    [FIX] 移除錯誤的 zfill(9)：學號本身為 8 或 9 位，不應強制補零至 9 位，
          以免 8 位合法學號被填成 9 位，導致 hash 結果跨學期不一致。
    """
    s = str(raw).strip()
    if re.match(r"^\d+\.\d+$", s):
        s = str(int(float(s)))
    return s


def is_valid_student_id(sid: str) -> bool:
    return bool(RE_STUDENT_ID.match(sid.strip()))


def build_lookup_table(student_ids: list, salt: str) -> dict:
    """
    建立 { anon_id: masked_id } 對照表。
    輸出至本地檔案，不上傳 git。
    """
    table = {}
    for raw_id in student_ids:
        sid = normalize_student_id(raw_id)
        if not is_valid_student_id(sid):
            continue
        table[anonymize(sid, salt)] = mask_student_id(sid)
    return table


def save_lookup_table(table: dict, out_path: str = "lookup_table.json"):
    """
    將對照表儲存至本地。
    ⚠️  此檔案包含 anon_id ↔ masked_id 對應，不得上傳至 git。
    """
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    print(f"[02] lookup_table 已儲存：{out_path}（{len(table)} 筆，本地保留勿上傳）")


def anonymize_dataframe(df: pd.DataFrame, id_col: str, salt: str) -> pd.DataFrame:
    """
    將 DataFrame 的學號欄位替換為 anon_id，新增 masked_id 欄位，
    並 drop 原始學號欄（去識別化目的）。
    [FIX] 移除未使用的 build_anonymizer 閉包呼叫（死碼）。
    [FIX] Drop 原始 id_col，避免 PII 殘留於輸出 DataFrame。
    """
    df = df.copy()
    df["_sid_norm"] = df[id_col].apply(normalize_student_id)
    df["_valid"] = df["_sid_norm"].apply(is_valid_student_id)

    invalid_count = (~df["_valid"]).sum()
    if invalid_count > 0:
        print(f"   ⚠️  過濾無效學號列：{invalid_count} 筆")

    df = df[df["_valid"]].copy()
    df["anon_id"] = df["_sid_norm"].apply(lambda s: anonymize(s, salt))
    df["masked_id"] = df["_sid_norm"].apply(mask_student_id)

    # [FIX] Drop 原始學號欄（PII）及工作欄
    df = df.drop(columns=["_sid_norm", "_valid", id_col])

    return df


if __name__ == "__main__":
    import sys
    salt_path = sys.argv[1] if len(sys.argv) > 1 else "config/salt.txt"
    try:
        salt = load_salt(salt_path)
        test_ids = ["141119002", "141119003", "111114144"]
        table = build_lookup_table(test_ids, salt)
        print("測試對照表：")
        for anon, masked in table.items():
            print(f"  {anon} → {masked}")
    except (FileNotFoundError, ValueError) as e:
        print(f"錯誤：{e}")
