from __future__ import annotations

import numpy as np
import pandas as pd

from copper_market_intelligence.features.pipeline import build_feature_dataset


def test_no_future_leakage():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=30, freq="B"),
            "open": np.arange(30.0) + 100.0,
            "high": np.arange(30.0) + 101.0,
            "low": np.arange(30.0) + 99.0,
            "close": np.arange(30.0) + 100.0,
            "volume": np.arange(30.0) * 10 + 100,
        }
    )
    out = build_feature_dataset(df)
    # Check that no engineered column contains values that come from future
    # close. For example, log_ret_1 must equal log(fut_close_1 / close)
    assert "log_ret_1" in out.columns
    assert "fut_close_1" in out.columns
    # ensure that for rows where fut_close_1 is not NaN, the relationship holds
    mask = out["fut_close_1"].notna()
    left = out.loc[mask, "log_ret_1"]
    right = np.log(out.loc[mask, "fut_close_1"] / out.loc[mask, "close"])
    assert all(np.isclose(left, right))
