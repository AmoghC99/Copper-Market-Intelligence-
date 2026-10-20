from __future__ import annotations

import numpy as np
import pandas as pd

from copper_market_intelligence.forecasting.backtest import (
    expanding_mean_baseline,
    expanding_walk_forward,
    momentum_baseline,
    zero_return_baseline,
)


def make_sample_df(n=300):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.linspace(100.0, 120.0, n)
    df = pd.DataFrame({
        "timestamp": dates,
        "open": close - 0.5,
        "high": close + 0.5,
        "low": close - 1.0,
        "close": close,
        "volume": np.random.randint(100, 200, size=n),
    })
    # add simple returns
    df["log_ret_1"] = np.log(df["close"].shift(-1) / df["close"])
    df["log_ret_5"] = np.log(df["close"].shift(-5) / df["close"])
    df["log_ret_20"] = np.log(df["close"].shift(-20) / df["close"])
    df["close_lag_5"] = df["close"].shift(5)
    return df


def test_expanding_walk_forward_basic():
    df = make_sample_df(300)
    models = {
        "zero": zero_return_baseline,
        "mean": expanding_mean_baseline,
        "mom5": lambda tr, h: momentum_baseline(tr, h, lookback=5),
    }
    res = expanding_walk_forward(
        df, min_train_periods=50, horizons=(1, 5), retrain_frequency=10, models=models
    )
    # basic asserts
    assert not res.empty
    assert set(res["model_name"].unique()) == set(models.keys())
    # ensure horizons present
    assert set(res["target_horizon"].unique()).issuperset({1, 5})
