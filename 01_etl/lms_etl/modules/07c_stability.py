"""
lms_etl/modules/07c_stability.py
計算每位學生每週學習時間的變異係數（CV），回寫至 student["features"]["stability_cv"]。
"""


def compute_stability_cv(students: list) -> dict:
    """
    回傳：{ (anon_id, semester): { "cv": float | None, "mean_weekly_min": float | None } }
    有效週：weekly_minutes 中非 None 且 > 0 的值
    有效週數 < 2 時，cv 與 mean_weekly_min 均為 None

    [複合鍵修正] 原版以 anon_id 單鍵存放結果：重修生同一 anon_id 橫跨
    多學期會各有一筆 student record（見 lms_etl.py Phase 3 複合鍵重構），
    單鍵寫入會讓後一學期的 CV 覆蓋前一學期，兩學期最終讀到同一個值——
    與本專案其餘模組已修復的「Bug B：重修生跨學期混血」屬同一根因模式。
    此處統一改用 (anon_id, semester) tuple 作為 key（此 dict 僅供 lms_etl.py
    內部即時查表用，不直接 JSON 輸出，故不需要 composite_key() 那種
    JSON-safe 字串鍵）。
    """
    result = {}
    for s in students:
        key = (s.get("anon_id", ""), s.get("semester", ""))
        weekly = s.get("time_profile", {}).get("weekly_minutes", [])
        arr = [v for v in weekly if v is not None and v > 0]
        if len(arr) < 2:
            result[key] = {"cv": None, "mean_weekly_min": None}
            continue
        mean = sum(arr) / len(arr)
        # 樣本標準差（除以 n-1）
        variance = sum((x - mean) ** 2 for x in arr) / (len(arr) - 1)
        std = variance ** 0.5
        result[key] = {
            "cv":               round(std / mean, 4) if mean > 0 else None,
            "mean_weekly_min":  round(mean, 2),
        }
    valid_count = sum(1 for v in result.values() if v.get("cv") is not None)
    print(f"  [07c] stability_cv 計算完成：{valid_count}/{len(result)} 筆 (anon_id,semester) 記錄有有效 CV 值")
    return result
