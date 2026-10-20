from __future__ import annotations

from typing import Optional

import pandas as pd

from copper_market_intelligence.features.calendar import add_calendar_features
from copper_market_intelligence.features.macro import fred_join_asof
from copper_market_intelligence.features.returns import add_returns
from copper_market_intelligence.features.targets import generate_log_return_targets
from copper_market_intelligence.features.technical import (
    add_lagged_close,
    add_momentum,
    add_rolling_stats,
    add_volatility_measures,
)
from copper_market_intelligence.features.volume import add_volume_features


def build_feature_dataset(
    market_daily: pd.DataFrame,
    fred_monthly: Optional[pd.DataFrame] = None,
    fred_policy: str = "lagged",
) -> pd.DataFrame:
    """Build a leakage-safe feature dataset from daily market data.

    Returns a DataFrame including `timestamp`, original market columns,
    engineered features, and targets.

    Notes:
    - Rows lost are only due to lookback/lagging and are represented as NaNs in
      leading rows.
    - The function does not drop rows arbitrarily.
    """
    df = market_daily.copy()
    if "timestamp" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or 0: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    df = add_lagged_close(df)
    df = add_rolling_stats(df)
    df = add_momentum(df)
    df = add_volatility_measures(df)
    df = add_volume_features(df)
    df = add_calendar_features(df)

    # returns and targets
    df = add_returns(df)
    df = generate_log_return_targets(df)

    # fred join (optional)
    if fred_monthly is not None:
        df = fred_join_asof(df, fred_monthly, policy=fred_policy)

    return df
