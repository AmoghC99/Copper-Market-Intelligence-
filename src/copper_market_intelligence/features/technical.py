from __future__ import annotations

import pandas as pd


def add_lagged_close(
    df: pd.DataFrame, close_col: str = "close", lags=(1, 2, 5, 10, 20)
) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"close_lag_{lag}"] = out[close_col].shift(lag)
    return out


def add_rolling_stats(
    df: pd.DataFrame, close_col: str = "close", windows=(5, 20, 60)
) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        out[f"ma_{w}"] = out[close_col].rolling(window=w, min_periods=1).mean()
        out[f"std_{w}"] = out[close_col].rolling(window=w, min_periods=1).std()
        out[f"min_{w}"] = out[close_col].rolling(window=w, min_periods=1).min()
        out[f"max_{w}"] = out[close_col].rolling(window=w, min_periods=1).max()
    return out


def add_momentum(df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
    out = df.copy()
    out["rel_to_ma_5"] = out[close_col] / out["ma_5"]
    out["rel_to_ma_20"] = out[close_col] / out["ma_20"]
    out["rel_to_ma_60"] = out[close_col] / out["ma_60"]
    out["mom_5"] = out[close_col] / out[close_col].shift(5) - 1
    out["mom_20"] = out[close_col] / out[close_col].shift(20) - 1
    return out


def add_volatility_measures(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hl_range"] = out["high"] - out["low"]
    # ATR-like using high-low and previous close
    out["tr1"] = out["high"] - out["low"]
    out["tr2"] = (out["high"] - out["close"].shift(1)).abs()
    out["tr3"] = (out["low"] - out["close"].shift(1)).abs()
    out["true_range"] = out[["tr1", "tr2", "tr3"]].max(axis=1)
    out["atr_14"] = out["true_range"].rolling(window=14, min_periods=1).mean()
    out["ret_1"] = out["close"] / out["close"].shift(1) - 1
    out["vol_ret_5"] = out["ret_1"].rolling(window=5, min_periods=1).std()
    out["vol_ret_20"] = out["ret_1"].rolling(window=20, min_periods=1).std()
    out["vol_ret_60"] = out["ret_1"].rolling(window=60, min_periods=1).std()
    return out
