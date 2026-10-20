"""External feature utilities for Stage 6 experiments.

Functions are modular and designed to be leakage-safe by using explicit
availability/publication timestamps (an "as-of" join).

Note: data retrieval from remote sources (FRED/LME) is left to the
caller or an optional fetch helper; this module focuses on feature
construction and alignment.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def as_of_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str = "date",
    right_available_col: str = "available_date",
    right_value_col: str = "value",
) -> pd.DataFrame:
    """Perform an as-of join: for each row in `left` pick the most recent
    row in `right` whose `right_available_col` <= `left[left_on]`.

    left: DataFrame with a date-like column `left_on` (e.g. daily timestamps).
    right: DataFrame with at least `right_available_col` and `right_value_col`.

    Returns a copy of `left` with `right_value_col` merged in (NaN when no
    prior observation is available).
    """
    left_df = left.copy()
    right_df = right.copy()
    left_df[left_on] = pd.to_datetime(left_df[left_on])
    right_df[right_available_col] = pd.to_datetime(
        right_df[right_available_col]
    )

    left_sorted = left_df.sort_values(left_on).reset_index(drop=True)
    right_sorted = (
        right_df.sort_values(right_available_col).reset_index(drop=True)
    )

    merged = pd.merge_asof(
        left_sorted,
        right_sorted,
        left_on=left_on,
        right_on=right_available_col,
        direction="backward",
    )
    return merged


def usd_features(usd_series: pd.Series, prefix: str = "usd") -> pd.DataFrame:
    """Create leakage-safe USD features from a daily USD index series.

    Features produced:
    - pct_change_1, pct_change_5, pct_change_20 (simple returns)
    - log_change_1, log_change_5, log_change_20
    - rolling_vol_20 (20-day rolling std of daily returns)

    `usd_series` must be a pandas Series indexed by date (daily business days
    recommended). Returns a DataFrame indexed by date with new columns.
    """
    s = usd_series.copy().sort_index()
    df = pd.DataFrame(index=s.index)
    df[f"{prefix}_pct_1"] = s.pct_change(1)
    df[f"{prefix}_pct_5"] = s.pct_change(5)
    df[f"{prefix}_pct_20"] = s.pct_change(20)
    import numpy as _np

    df[f"{prefix}_log_1"] = _np.log(s / s.shift(1))
    df[f"{prefix}_log_5"] = _np.log(s / s.shift(5))
    df[f"{prefix}_log_20"] = _np.log(s / s.shift(20))
    df[f"{prefix}_vol_20"] = df[f"{prefix}_pct_1"].rolling(20).std()
    return df


def expand_monthly_as_of(
    monthly_df: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    obs_date_col: str = "obs_date",
    value_col: str = "value",
    publication_lag_months: int = 1,
) -> pd.DataFrame:
    """Convert monthly observations into a daily series using a conservative
    as-of policy.

    - `monthly_df` must contain an observation date (`obs_date_col`) and a
      value column (`value_col`). If `available_date` is not present, it
      is computed as `obs_date + publication_lag_months`.
    - `daily_index` is the desired daily index (e.g. the model's forecast
      timestamps).

    Returns a DataFrame with index `daily_index` and column `value_col` that
    contains the latest monthly observation available by each daily timestamp.
    """
    m = monthly_df.copy()
    m[obs_date_col] = pd.to_datetime(m[obs_date_col])
    if "available_date" not in m.columns:
        m["available_date"] = (
            m[obs_date_col] + pd.DateOffset(months=publication_lag_months)
        )

    left = pd.DataFrame({"date": pd.to_datetime(daily_index)})

    right = m.rename(
        columns={
            "available_date": "available_date",
            obs_date_col: "obs_date",
            value_col: value_col,
        }
    )
    right = right[["available_date", "obs_date", value_col]].sort_values(
        "available_date"
    )

    merged = as_of_join(
        left,
        right,
        left_on="date",
        right_available_col="available_date",
        right_value_col=value_col,
    )
    out = merged.set_index("date")[[value_col]]
    out.index = pd.DatetimeIndex(out.index)
    return out


def inventory_features(
    inventory_df: Optional[pd.DataFrame], prefix: str = "lme"
) -> Optional[pd.DataFrame]:
    """Compute inventory features if a public inventory DataFrame is provided.

    `inventory_df` is expected to have a date index or a `date` column and a
    `stock` column. If `None` or empty, return None and document that inventory
    data was not available.
    """
    if inventory_df is None or inventory_df.empty:
        return None
    df = inventory_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    df = df.sort_index()
    if "stock" not in df.columns:
        raise ValueError("inventory_df must contain a 'stock' column")
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_stock"] = df["stock"]
    out[f"{prefix}_pct_1"] = df["stock"].pct_change(1)
    out[f"{prefix}_pct_5"] = df["stock"].pct_change(5)
    out[f"{prefix}_pct_20"] = df["stock"].pct_change(20)
    return out
