from __future__ import annotations

import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.to_datetime(out["timestamp"]) if "timestamp" in out.columns else out.index
    out["dayofweek"] = pd.DatetimeIndex(idx).dayofweek
    out["month"] = pd.DatetimeIndex(idx).month
    out["quarter"] = pd.DatetimeIndex(idx).quarter
    # trading-day position in month (1..n)
    out["day_in_month"] = pd.DatetimeIndex(idx).day
    return out
