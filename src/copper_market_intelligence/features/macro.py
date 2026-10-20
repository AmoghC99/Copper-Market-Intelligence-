from __future__ import annotations

import pandas as pd


def fred_join_asof(
    daily: pd.DataFrame, fred_monthly: pd.DataFrame, policy: str = "lagged"
) -> pd.DataFrame:
    """Join monthly FRED series to daily dataframe using a conservative as-of policy.

    Policies:
    - "lagged": use the previous month's value for all days in the current
      month (conservative; avoids peeking at current month value)
    - "asof": forward-fill monthly values to daily as-of the month date (uses
      value when available at month-end)

    ``fred_monthly`` should be a DataFrame indexed by month or include a
    ``date`` column at month granularity.
    """

    if fred_monthly is None:
        return daily.copy()

    d = daily.copy()
    f = fred_monthly.copy()
    if "date" in f.columns:
        f["date"] = pd.to_datetime(f["date"]).dt.to_period("M").dt.to_timestamp()
        f = f.set_index("date")
    else:
        f.index = pd.to_datetime(f.index).to_period("M").to_timestamp()

    if policy == "lagged":
        # shift fred series forward by 1 month so that daily rows in month M
        # see value from month M-1
        f_shift = f.shift(1)
        d["fred_value_lagged"] = (
            d["timestamp"].dt.to_period("M").dt.to_timestamp().map(f_shift["value"])
        )
    else:
        # asof join: map month timestamp to the month's value
        d["fred_value_asof"] = (
            d["timestamp"].dt.to_period("M").dt.to_timestamp().map(f["value"])
        )

    return d
