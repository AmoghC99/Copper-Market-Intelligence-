from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def generate_log_return_targets(
  df: pd.DataFrame, close_col: str = "close", horizons: Iterable[int] = (1, 5, 20)
) -> pd.DataFrame:
  """Add log-return targets to a copy of `df`.

  Definitions:
  - t: the current timestamp index/row where a forecast is produced. Information
    available at t includes all market fields at t and earlier (no future data).
  - forecast horizon h: number of trading days ahead.
  - log_return_t+h = log(close[t+h] / close[t])

  The function creates columns ``log_ret_{h}`` for each horizon h. It also
  creates ``fut_close_{h}`` which is the raw future close at t+h (useful for
  reconstructing future price forecasts from predicted returns).

  Missing future targets are left as NaN (we do not fill or forward-fill).
  """
  out = df.copy()
  for h in horizons:
    out[f"log_ret_{h}"] = np.log(out[close_col].shift(-h) / out[close_col])
    out[f"fut_close_{h}"] = out[close_col].shift(-h)
  return out
