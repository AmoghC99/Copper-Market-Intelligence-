from typing import Dict, List

import pandas as pd


def check_date_ordering(
    df: pd.DataFrame, date_col: str = "timestamp"
) -> Dict[str, int]:
    s = pd.to_datetime(df[date_col], errors="coerce")
    return {
        "null_dates": int(s.isna().sum()),
        "out_of_order": int((s.dropna() != s.dropna().sort_values()).sum()),
    }


def check_duplicates(df: pd.DataFrame, date_col: str = "timestamp") -> int:
    return int(df[date_col].duplicated().sum())


def check_numeric_fields(df: pd.DataFrame, fields: List[str]) -> Dict[str, int]:
    res = {}
    for f in fields:
        if f in df.columns:
            res[f] = int(pd.to_numeric(df[f], errors="coerce").isna().sum())
        else:
            res[f] = -1
    return res


def check_ohlc_relationships(df: pd.DataFrame) -> Dict[str, int]:
    # count rows where low > high or open/close out of bounds
    issues = 0
    if not all(c in df.columns for c in ("high", "low")):
        return {"issues": issues}
    issues += int((df["low"] > df["high"]).sum())
    return {"issues": int(issues)}


def check_negative_volume(df: pd.DataFrame) -> int:
    if "volume" not in df.columns:
        return 0
    return int((df["volume"] < 0).sum())
