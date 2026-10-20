from __future__ import annotations

import numpy as np
import pandas as pd

from copper_market_intelligence.features.targets import generate_log_return_targets


def test_generate_log_return_targets():
    # synthetic close series
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    close = pd.Series(np.arange(10.0) + 100.0, index=dates)
    df = pd.DataFrame({"timestamp": dates, "close": close}).reset_index(drop=True)

    out = generate_log_return_targets(df, horizons=(1, 3))
    # check columns exist
    assert "log_ret_1" in out.columns
    assert "log_ret_3" in out.columns
    # compute expected manually for row 0
    expected_1 = (
        np.log(out.loc[0, "fut_close_1"] / out.loc[0, "close"])
        if not pd.isna(out.loc[0, "fut_close_1"])
        else None
    )
    if expected_1 is not None:
        assert np.isclose(out.loc[0, "log_ret_1"], expected_1)
