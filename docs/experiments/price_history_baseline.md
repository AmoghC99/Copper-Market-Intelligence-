# Stage 5 — Price-History-Only Baseline Experiment

This document records the Stage 5 experimental results using only price-history features and the existing model specifications. It is the official experiment record for the price-history-only baseline.

## Dataset

- Source: processed price-history feature set stored in `data/processed/features_daily.parquet`.
- Backtest results: `data/processed/model_backtest_results.parquet` and baseline file `data/processed/baseline_backtest_results.parquet`.
- Time range and sample sizes: see `docs/backtest_audit.md` for per-horizon sample counts and QC (summary table included below).

## Forecast horizons

- 1 day (1d)
- 5 days (5d)
- 20 days (20d)

## Walk-forward methodology

- Expanding-window walk-forward backtest.
- Embargo equals forecast horizon to avoid leakage from nearby observations.
- Per-prediction metadata recorded: `training_end_timestamp`, `forecast_timestamp`, `target_end_timestamp`, training indices, and target indices for auditability.

## Naive baseline

- `zero` (naive hold/zero forecast) — predicts zero return for each horizon. This baseline is used to compute MASE and to evaluate whether models beat a simple price-history benchmark.

## Models evaluated

- `SARIMAX`: per-horizon SARIMAX fitted on numeric price-history features.
- `XGBoost`: per-horizon XGBoost using the same numeric feature set and careful feature selection to avoid leakage (target-like columns dropped).

No other models or additional features were introduced in this experiment.

## Metrics

- MAE (mean absolute error)
- RMSE (root mean squared error)
- MASE (mean absolute scaled error) with the `zero` baseline as the scaling reference
- Directional accuracy (fraction of predictions where sign(predicted_return) == sign(actual_return)); note directional accuracy excludes cases where both actual and predicted are zero or otherwise excluded by QC rules.

## Results (summary)

Per the audit and aggregated metrics (see `docs/backtest_audit.md` and `docs/figures/aggregate_model_metrics.csv`), the Stage 5 results are:

- SARIMAX vs Zero:
  - 1d: SARIMAX MAE improved negligibly vs Zero (~0.03% relative improvement in MAE).
  - 5d: SARIMAX MAE is worse than Zero.
  - 20d: SARIMAX MAE is worse than Zero.

- XGBoost vs Zero:
  - 1d: XGBoost MAE is worse than Zero.
  - 5d: XGBoost MAE is worse than Zero.
  - 20d: XGBoost MAE is worse than Zero.

- Directional accuracy:
  - Approximately 50% across models and horizons; see `docs/backtest_audit.md` for per-horizon DirAcc values. Directional accuracy is close to random and should not be treated as evidence of a reliable directional forecasting edge.

For reference, the audit table excerpt:

| model | horizon | N_preds | MAE | RMSE | MASE | DirAcc |
|:------|--------:|--------:|-----:|-----:|-----:|------:|
| mean  | 1 | 1258 | 0.012321 | 0.018323 | 1.0004 | 0.4960 |
| zero  | 1 | 1258 | 0.012316 | 0.018306 | 1.0000 | nan |
| SARIMAX | 1 | 1258 | 0.012312 | 0.017475 | 0.9997 | 0.4985 |
| XGBoost | 1 | 1258 | 0.015486 | 0.022173 | 1.2574 | 0.4907 |

(See `docs/backtest_audit.md` lines 1–14 for the full audit table and per-horizon numbers.)

## Limitations

- Experiment uses only the current price-history feature set; no fundamental, alternative, or external covariates were added.
- Model specifications used default or previously agreed hyperparameters; no extensive hyperparameter search was conducted for this experiment.
- Results are conditioned on the data preprocessing, embargo policy, and walk-forward design described above.
- Directional accuracy near 50% does not imply market unpredictability — it only indicates that these models and features did not produce a robust directional edge in this backtest.

## Conclusion

- The Stage 5 price-history-only baseline experiment shows that, with the current feature set and model specifications:
  1. `SARIMAX` gives a negligible MAE improvement at 1d (~0.03% relative to the naive baseline) but performs worse at 5d and 20d.
  2. `XGBoost` performs worse than the naive baseline at 1d, 5d, and 20d.
  3. Directional accuracy is approximately 50% and is not evidence of a reliable directional forecasting edge.

- These conclusions do not assert that the market is inherently unpredictable. They explicitly state that this experiment establishes that the current price-history feature set and model specifications did not materially outperform the naive benchmark in this backtest.

## Artifacts and reproducibility

- Aggregated metrics CSV: `docs/figures/aggregate_model_metrics.csv`
- Backtest results: `data/processed/model_backtest_results.parquet`
- Baseline results: `data/processed/baseline_backtest_results.parquet`
- SHAP artifacts for `XGBoost`: `docs/figures/xgb_shap_*` (per-horizon summaries and plots)
- Validation script: `scripts/validate_model_backtest.py`
- Aggregation and report scripts: `scripts/final_model_report.py`, `scripts/fill_zero_improvements.py`

## Next steps (recommended)

- Run controlled hyperparameter searches or cross-validation for SARIMAX and XGBoost, monitoring leakage carefully.
- Add additional non-price features (fundamentals, macro, sentiment) and re-evaluate in the same walk-forward framework.
- Consider regime-based analyses and stability checks before making production decisions.

---

Record created and committed as the Stage 5 official price-history-only baseline experiment.
