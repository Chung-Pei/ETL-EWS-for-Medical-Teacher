"""
lms_etl/modules/07d_density_filter.py
排除 total_learning_minutes 最低分位（預設後5%）的低參與樣本，產出 corr_df
（僅供相關性計算）。
【重要】僅用於相關性計算輸入，不影響 scatter_data、at-risk 或其他輸出。

【設計說明，非bug，見ETL交接文件v5第6.1節複查結論】
以 total_learning_minutes（跨特徵的複合／加總指標）篩選「低參與」樣本、
並套用到全部特徵欄位的相關性計算（包括跟總時數本身無關的比率型指標，例如
quz_cramming_ratio、consistency_score），是「16-1規劃書」既有的刻意設計，
用意是避免幾乎零學習行為的學生稀釋/扭曲相關係數。下游
compute_pearson_matrix()／compute_subgroup_pearson()／
compute_outlier_thresholds() 本身已對每個 (feature, target) 各自獨立判斷
NaN 有效性，此處是額外疊加的「先剔除整體低參與樣本」前處理，屬於刻意的
整體性排除設計，不是誤用的連坐排除 bug。
"""

import os
import sys

import pandas as pd

# 見 06_merge_grades.py 同一段落說明：明確將專案根目錄加入 sys.path，
# 不依賴呼叫端是否已設定好。
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))  # lms_etl/modules → lms_etl → 專案根目錄
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import etl as _etl  # noqa: E402  （import 順序為刻意安排，理由同上）


def _filter_one_group(group_df: pd.DataFrame, col: str, quantile: float, group_label: str) -> pd.DataFrame:
    """對單一 course_track 分組計算門檻並篩選，供 filter_for_correlation() 分組後逐組呼叫。"""
    series = pd.to_numeric(group_df[col], errors="coerce")

    if len(group_df) < 2 or series.notna().sum() < 2:
        print(f"  ⚠️  [07d] （{group_label}）樣本數不足或 {col!r} 有效值不足，略過此分組密度過濾，保留完整分組")
        return group_df

    threshold = series.quantile(quantile)
    if pd.isna(threshold):
        print(f"  ⚠️  [07d] （{group_label}）{col!r} 分位數計算失敗，略過此分組密度過濾，保留完整分組")
        return group_df

    filtered = group_df[series >= threshold]
    print(
        f"  [07d] 密度過濾（{group_label}）：{len(filtered)}/{len(group_df)} 筆"
        f"（閾值={threshold:.1f} 分鐘，第{quantile:.0%}分位）"
    )
    return filtered


def filter_for_correlation(
    merged_df: pd.DataFrame,
    col: str = "total_learning_minutes",
    quantile: float = 0.05,
) -> pd.DataFrame:
    """
    閾值：merged_df[col] 的第 quantile 分位數（預設後5%視為低參與）。
    印出：[07d] 密度過濾：N/M 筆（閾值=X.X 分鐘，第X%分位）

    v2（ETL交接文件v5第6.1節修正）：閾值由 mean - 1σ 改為百分位數。
    mean-1σ 在 col 分布極度右偏時（少數學生總時數異常龐大，把 std 撐大）
    會讓 threshold 變成負值，導致篩選在該資料下完全失效（0人被排除）——
    分布只要之後變得不那麼極端右偏（例如清掉少數超大離群值），threshold
    就會突然變正、開始真的排除學生，篩選行為因此不可預期。百分位數不受
    少數極端離群值影響，篩選比例維持穩定、可預期。

    v3（基礎醫學臨床應用(一)課程異動，見規劃書1.3節／討論10）：閾值改為
    依 course_track 分組各自計算。course_track=="transition_merged_micro"
    的學生僅有week13-18（5-6週）資料，total_learning_minutes 天生低於一般
    18週學生，若與其他學生合併計算同一個全域分位數門檻，會被系統性誤判為
    低參與離群值而被排除（並非真的參與度低，只是觀察窗口本來就短）。
    分組後，5-6週學生只跟5-6週學生比、18週學生只跟18週學生比，維持篩選
    「相對參與度」的原始設計意圖，不因觀察窗口長度不同而失真。

    Parameters
    ----------
    merged_df : 全量合併 DataFrame
    col       : 過濾依據欄位，預設 total_learning_minutes
    quantile  : 低參與門檻分位數，預設 0.05（後5%）

    Returns
    -------
    過濾後的 DataFrame（索引 reset）
    """
    if col not in merged_df.columns:
        print(f"  ⚠️  [07d] 欄位 {col!r} 不存在，略過密度過濾，回傳完整 merged_df")
        return merged_df

    if len(merged_df) < 2 or pd.to_numeric(merged_df[col], errors="coerce").notna().sum() < 2:
        print(f"  ⚠️  [07d] 樣本數不足或 {col!r} 有效值不足，略過密度過濾，回傳完整 merged_df")
        return merged_df

    if "sheet_name" in merged_df.columns and "semester" in merged_df.columns:
        course_track_col = pd.Series(
            [
                _etl._classify_course_track(sn, sm)
                for sn, sm in zip(
                    merged_df["sheet_name"].astype(str), merged_df["semester"].astype(str)
                )
            ],
            index=merged_df.index,
        )
    else:
        course_track_col = pd.Series([None] * len(merged_df), index=merged_df.index)

    filtered_parts = [
        _filter_one_group(
            group_df, col, quantile,
            group_label=("一般" if pd.isna(track_val) else str(track_val)),
        )
        for track_val, group_df in merged_df.groupby(course_track_col, dropna=False)
    ]
    filtered = pd.concat(filtered_parts).reset_index(drop=True) if filtered_parts else merged_df.iloc[0:0].reset_index(drop=True)
    return filtered
