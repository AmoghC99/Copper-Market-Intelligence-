# External Market Information Experiment

## 1. Objective

Does adding external copper-market information provide incremental
out-of-sample predictive value beyond the existing price-history-only
feature set?

## 2. Baseline

Model A (baseline):
- Existing price-history features only.
- Existing target definitions and preprocessing.
- The same expanding-window walk-forward methodology used in Stage 5.
- Baseline models and benchmarks: `zero`, historical mean, momentum,
  `SARIMAX`, and `XGBoost` as implemented in Stage 5.

Stage 5 established the price-history-only benchmark used here.

## 3. Incremental Models

- Model A: Price history only (baseline).
- Model B: Model A + USD features.
- Model C: Model B + China demand/activity features.
- Model D: Model C + copper inventory features (only if reliable
  historical inventory data is available).

The experiment will change only the information set; forecasting
methodology and cross-validation/backtest folds remain identical.

## 4. External Information Sources

| Information group | Example variable | Frequency | Source | Availability policy | Status |
|:------------------|:-----------------|:---------:|:-------|:-------------------|:------:|
| USD / FX | Broad U.S. Dollar Index (nominal) | Daily | FRED | as-of daily index; compute changes with no lookahead | Planned |
| China demand / activity | China imports, industrial output | Monthly | FRED / official releases | Conservative as-of policy; monthly obs become usable after publication lag | Planned (data sourcing required) |
| Copper inventory | LME warehouse stocks | Daily (reporting delay) | LME public reports | Use only genuinely public historical coverage with explicit reporting delay; omit if adequate history unavailable | Investigate / conditional |

Notes:
- Sources marked *Planned* require verification of historical coverage and public availability. Do not assume availability dates.

## 5. Information Availability Policy

- Observation date: the date the economic activity or reading refers to (e.g., month end for monthly series).
- Publication / availability date: the date the data was publicly released or conservatively assumed available.
- Forecast timestamp: the model's prediction date for which features must be available.
- As-of join logic: features are joined using an explicit as-of join that only attaches external observations whose `available_date` <= `forecast_timestamp`.
- Conservative lag assumptions: when publication dates are unknown, assume a one-month conservative lag for monthly series unless a specific release schedule is known.
- Revisions: only the information that was available as-of the forecast timestamp may be used; do not use revised future data.
- If publication dates are unavailable for a series, document the assumption and use a conservative lag or exclude the series.

Explicit rule: a feature may be used for forecast timestamp `t` only if it was publicly available by `t`.

## 6. Feature Groups

USD (Group A):
- Daily percent change (1d), 5-day change, 20-day change
- Log changes for robustness checks
- Rolling volatility (e.g., 20-day)

China demand / activity (Group B):
- Month-over-month change
- Year-over-year change
- Use conservative as-of availability (do not forward-fill monthly values without availability)

Inventory (Group C):
- Inventory level
- Percentage change
- 5-day change and 20-day change
- Document reporting delay (e.g., two-day delay for LME reports) and only use values available by forecast timestamp

These are feature descriptions only; no predictive claims are made here.

## 7. Backtesting Methodology

- Use the same expanding-window walk-forward backtest as Stage 5.
- Evaluate horizons: 1-day, 5-day, 20-day.
- Track per-prediction metadata: `forecast_timestamp`, `training_end_timestamp`, `target_end_timestamp`, indices, and model/horizon tags.

Metrics to compute:
- MAE, RMSE, MASE (scaled to Model A), directional accuracy, bias, prediction count.

Use identical folds and seed settings to ensure apples-to-apples comparisons.

## 8. Incremental Evaluation

Compare:
- Model B vs Model A
- Model C vs Model A
- Model D vs Model A

Report incremental changes per horizon and per metric:
- Absolute MAE change
- Percentage MAE change
- RMSE change
- Directional accuracy change

Do not declare a universal winner; report per-horizon and per-period results.

## 9. Robustness

Evaluate persistence and stability:
- Annual performance breakdowns
- Recent-period (e.g., last N months/years) performance
- Rolling MAE windows
- Regime-specific analyses (when feasible)

## 10. Explainability

For XGBoost runs, produce SHAP analyses comparing incremental feature sets:
- Price-only vs Price+USD vs Price+USD+China vs Price+USD+China+Inventory

Document that SHAP describes the model's behavior and is not causal proof.

## 11. Limitations

- COMEX HG=F and other futures-series are proxies and may not perfectly represent spot-market dynamics.
- Monthly external data have lower frequency and require conservative as-of handling.
- Publication/availability dates may be imperfect or unavailable; conservative lags are used when necessary.
- Historical LME inventory data coverage may be limited or proprietary; do not ingest restricted sources.
- Data revisions can affect backtest fidelity; only use data that would have been available at forecast time.

## 12. Results

Placeholder — results will be added after running the live-data experiment and analysis. Do not populate this section until the full incremental backtests are completed.

## 13. Decision Criteria

We will consider external information useful if it meets these criteria across multiple checks:
1. Improves out-of-sample error relative to Model A on one or more horizons.
2. Produces improvements that are consistent (not restricted to a single short period).
3. Improves more than one evaluation metric (MAE and RMSE for example).
4. Remains useful in recent periods.
5. Shows reasonable feature importance patterns (not dominated by one noisy variable).

No numeric thresholds are prescribed in advance; decisions will be based on the totality of evidence.

---

Next steps (after scaffolding):
1. Implement FRED / LME ingestion adapters and verify historical coverage (data sourcing step).
2. Build incremental feature datasets (`data/processed/external_features_daily.parquet`).
3. Run incremental backtests (Model A–D) for horizons 1/5/20d and save results to `data/processed/incremental_model_results.parquet`.
4. Produce SHAP analyses and the final experiment report.
