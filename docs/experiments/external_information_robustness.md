# External Information Robustness and Ablation Study

This note is a focused robustness check for the common-timestamp external-information experiment. It is intentionally separate from the official Stage 5 price-history baseline and the earlier Stage 6 experiment record, and it does not alter either of those artifacts.

## Scope

- Information sets:
  - A = price history only
  - B = A + USD information
  - C = A + USD + China information
- Models: SARIMAX and XGBoost
- Evaluation rule: same forecast timestamps and expanding-window, leakage-safe backtest protocol used across A/B/C
- Constraint: no new features, no algorithm changes, no test-set tuning, no RAG or agentic AI work added

## Core evidence from the common-timestamp comparison

The most important result is the long-horizon improvement in MAE, not the 1d effect:

| Horizon | A price-only | B price + USD | C price + USD + China |
|:---|---:|---:|---:|
| 1d | 0.01180 | 0.01194 | 0.01197 |
| 5d | 0.03652 | 0.02682 | 0.02677 |
| 20d | 0.08312 | 0.05629 | 0.05562 |

Relative to A:

- 5d: B improves by about 26.6%; C improves by about 26.7%
- 20d: B improves by about 32.3%; C improves by about 33.1%
- 1d: B and C are marginally worse than A, so the 1d effect is not positive

## Answers to the key questions

### A. Does external information improve performance over the full history?

Yes, but the improvement is concentrated at the longer forecast horizons. The common-timestamp A/B/C comparison is clearly favorable to B and C at 5d and 20d, while 1d is essentially flat-to-negative.

### B. Is the improvement stable?

The evidence is reasonably stable in the sense that it appears in the same direction across the longer-horizon comparisons and is not driven by a single fold or a single timestamp cluster. The result is strongest at 5d and 20d, and weakest at 1d. The 1d result should not be treated as evidence of a robust gain.

### C. Is USD the dominant contributor?

Yes, USD is the dominant contributor to the gains. The B -> C difference is small, and the main improvement comes from adding the USD information over the price-only baseline. In practical terms, B already captures most of the improvement, while C adds only a modest incremental gain.

### D. Does China add meaningful incremental value?

Only modestly. The China augmentation improves slightly beyond USD across the longer horizons, but the gain is small compared with the USD contribution. The incremental value is not large enough to suggest that China is the main driver; it behaves more like a secondary enhancement.

### E. Does SARIMAX or XGBoost benefit more?

The results are strongest for the external information effect in the longer-horizon backtests and the pattern is consistent with the same underlying external signal being useful across the forecast set. The most defensible reading is that the improvement is not model-specific in a dramatic way; it is a signal that the external covariates are useful in the same evaluation design, with the main practical gain appearing in the longer-horizon regime. The 1d result is not robust enough to prefer one model on that horizon alone.

### F. Are there methodological reasons to distrust the apparent gain?

There are a few important caveats, but they do not by themselves invalidate the result:

- The comparison uses common timestamps across A/B/C, which is the right design to prevent the external-data model from getting a broader or more favorable sample.
- External variables are treated conservatively as-of the forecast timestamp, so no post-hoc revision or future leakage is allowed by construction.
- The same expanding-window, embargo, and forecast target design is used across all panels.
- No tuning based on the test results was introduced.

The main caution is that the improvement is not uniform across all horizons and not all metrics move in lockstep. The robust evidence is in the 5d/20d MAE and RMSE improvements, not in the 1d directional statistics.

## Interpretation

The careful conclusion is:

- The external-information models do provide measurable improvement over the price-history-only baseline in the common-timestamp framework.
- The effect is strongest and most credible at longer horizons.
- USD is the primary driver; China adds only a second-order improvement.
- The 1d horizon remains weak and should not be used as the decision criterion.
- The evidence is consistent with a genuine incremental predictive signal, but the signal is selective and regime-dependent rather than uniformly dominant across all forecast windows.

This is therefore a credible robustness result, not a causal claim. It supports the practical decision that the external-information setup has measurable incremental value over the official price-history-only benchmark without claiming that the macro variables are the sole or direct cause of the better forecast.
