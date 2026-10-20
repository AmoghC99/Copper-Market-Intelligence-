from __future__ import annotations

import pandas as pd


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["vol_lag_1"] = out["volume"].shift(1)
    out["vol_lag_5"] = out["volume"].shift(5)
    out["vol_ma_20"] = out["volume"].rolling(window=20, min_periods=1).mean()
    out["vol_vs_20"] = out["volume"] / out["vol_ma_20"]
    return out
