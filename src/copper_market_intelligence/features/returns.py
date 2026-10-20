from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def add_returns(
    df: pd.DataFrame, close_col: str = "close", periods: Iterable[int] = (1, 5, 20)
) -> pd.DataFrame:
    """Add simple log returns for given periods as `ret_{p}` where p is period."""
    out = df.copy()
    for p in periods:
        out[f"ret_{p}"] = np.log(out[close_col] / out[close_col].shift(p))
    return out
