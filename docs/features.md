# Feature Engineering and Forecast Targets

This document describes Stage 4: leakage-safe feature engineering and forecasting target design.

## Target definitions

Primary targets (all use log returns):

- `log_ret_1`: 1-trading-day log return. For timestamp `t`, using close at `t` and close at `t+1`:

  log_return_{t+1} = log(close[t+1] / close[t])

  - `t` meaning: the row/timestamp where the forecast is produced. All features must be computed using data available at or before `t`.
  - Information available at `t`: `open, high, low, close, volume` for `t` and all prior timestamps.
  - Horizon: `h = 1` trading day.
  - Missing future values: Not filled; targets are NaN when future data is unavailable.

- `log_ret_5`: 5-trading-day log return.

- `log_ret_20`: 20-trading-day log return.

For each horizon `h`, we also store `fut_close_{h}` = `close[t+h]` to enable reconstruction of future price forecasts from predicted returns.

## Feature groups

- Returns: short-term log returns (`ret_1`, `ret_5`, `ret_20`).
- Lagged close: `close_lag_1, close_lag_2, close_lag_5, close_lag_10, close_lag_20`.
- Rolling stats: `ma_{5,20,60}`, `std_{5,20,60}`, `min_{5,20,60}`, `max_{5,20,60}`.
- Momentum: `rel_to_ma_{5,20,60}`, `mom_{5,20}`.
- Volatility: `hl_range`, `true_range`, `atr_14`, `vol_ret_{5,20,60}`.
- Volume: `vol_lag_1`, `vol_lag_5`, `vol_ma_20`, `vol_vs_20`.
- Calendar: `dayofweek`, `month`, `quarter`, `day_in_month`.
- Optional macro (FRED) joins: conservative `fred_value_lagged` or `fred_value_asof` via explicit policy.

## FRED handling policy

- `lagged` policy (default): For days in month M, use the FRED value from month M-1 (conservative; avoids using contemporaneous month value that may be published later).
- `asof` policy: map month values to that month's days (uses monthly value when available).

We do NOT forward-fill the monthly FRED into daily rows as if it were known on each day without an explicit policy.

## Leakage checks

- All rolling windows use past data only (pandas `rolling` and `shift` with appropriate directions).
- Targets are computed using future shifted closes (`.shift(-h)`) and left as NaN when missing.
- Tests assert no feature uses future information and that target columns use future observations only.

## Why predict returns?

- Returns are stationary relative to price and support consistent loss functions.
- Predicted return r can be converted to a future price: fut_close = close[t] * exp(r).

## Continuous-futures notes

- HG=F is a continuous series; contract rolls can introduce discontinuities.
- We do not stitch contracts yet. Features are computed from the continuous series; document potential roll-related artifacts.

## Files

- `src/copper_market_intelligence/features/` contains modular builders.
- `data/processed/features_daily.parquet` will be written by the pipeline.

