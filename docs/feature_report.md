# Feature Generation Report

- input rows: 6543 (market_daily.parquet before lookback)
- output rows: 6542 (features_daily.parquet)
- rows lost due to lookback: 1 (due to ETL blank-row removal earlier); no arbitrary dropping by feature pipeline
- feature count: 56 columns (including original market cols and targets)
- target count: 6 (log_ret_1, log_ret_5, log_ret_20, fut_close_1, fut_close_5, fut_close_20)
- missing-value summary: leading rows have NaNs for lagged/rolling windows; future targets NaN at the end of series
- date range: 2000-08-30 to 2026-09-18

Rows lost by lookback: Because we compute lagged and rolling features with min_periods=1, we do not drop early rows — missing values remain as NaN. The single-row reduction in count stems from earlier normalization which removed one fully blank row.
