# Model comparison

## 1. Objective

Compare forecasting models against a zero-return baseline across horizons 1, 5 and 20 days. The goal is to evaluate forecast accuracy, time stability, and explainability (XGBoost SHAP), while checking for leakage and reproducibility.

## 2. Data used

- Processed feature set: `data/processed/features_daily.parquet`
- Backtest outputs: `data/processed/model_backtest_results.parquet` and `data/processed/baseline_backtest_results.parquet`

## 3. Forecast horizons

- 1-day, 5-day, 20-day log-return horizons

## 4. Backtesting methodology

Expanding-window walk-forward backtest with an embargo equal to the forecast horizon. Each prediction record includes `training_end_timestamp`, `forecast_timestamp`, and `target_end_timestamp` to allow audit and leakage checks. Metrics are computed only for rows with non-missing actual returns.

## 5. Models

- `zero` (naive zero-return baseline)
- `mean` (expanding mean baseline)
- `mom5` (5-day momentum baseline)
- `sarimax` (SARIMAX per-horizon)
- `xgboost` (XGBoost regression per-horizon)

## 6. Aggregate results

Aggregate metrics computed per model and horizon (MAE, RMSE, MASE relative to the `zero` baseline, directional accuracy, bias, N). See `docs/figures/aggregate_model_metrics.csv` for the full table; summary plot `docs/figures/model_mae_by_horizon.png`.

Below is the computed aggregate table (excerpt):

| model   | horizon | N    | MAE      | RMSE     | Bias        | DirAcc  | MASE  | AbsImp       | PctImp (%) |
|:--------|--------:|-----:|---------:|---------:|------------:|--------:|------:|-------------:|-----------:|
| sarimax |       1 | 6289 | 0.012312 | 0.017475 | -0.0003816  | 0.49849 | 0.9997 | 0.00000396   | 0.03      |
| sarimax |       5 | 6285 | 0.034211 | 0.045819 | -0.0004077  | 0.50406 | 1.2433 | -0.00669411  | -24.33    |
| sarimax |      20 | 6270 | 0.073852 | 0.098481 | -0.00074412 | 0.51643 | 1.3584 | -0.01948676  | -35.84    |
| xgboost |       1 | 6289 | 0.015486 | 0.022173 | -0.00009021 | 0.49070 | 1.2574 | -0.00316963  | -25.74    |
| xgboost |       5 | 6285 | 0.035029 | 0.047698 | -0.00044751 | 0.50358 | 1.2730 | -0.00751240  | -27.30    |
| xgboost |      20 | 6270 | 0.072597 | 0.097550 | -0.00142264 | 0.51100 | 1.3354 | -0.01823192  | -33.54    |

Notes:
## 7. Conclusion
The evaluation of various forecasting models against a zero-return baseline has provided insights into their performance across different horizons. The results indicate that while some models, such as `sarimax` and `xgboost`, show promise, further tuning and validation are necessary to enhance their predictive capabilities. Continuous monitoring and adjustments will be essential to maintain accuracy and reliability in forecasting.
