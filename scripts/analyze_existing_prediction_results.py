from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PRED_PATH = PROC / "model_backtest_results.parquet"
REPORT_PATH = ROOT / "docs" / "experiments" / "external_information_robustness_full.md"


def diracc(actual: pd.Series, pred: pd.Series) -> float:
    mask = (~pd.isna(actual)) & (~pd.isna(pred)) & (actual != 0) & (pred != 0)
    if not mask.any():
        return np.nan
    return float(np.mean(np.sign(actual[mask].to_numpy()) == np.sign(pred[mask].to_numpy())))


def mae(actual: pd.Series, pred: pd.Series) -> float:
    mask = (~pd.isna(actual)) & (~pd.isna(pred))
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs(actual[mask].to_numpy() - pred[mask].to_numpy())))


def rmse(actual: pd.Series, pred: pd.Series) -> float:
    mask = (~pd.isna(actual)) & (~pd.isna(pred))
    if not mask.any():
        return np.nan
    errors = actual[mask].to_numpy() - pred[mask].to_numpy()
    return float(np.sqrt(np.mean(np.square(errors))))


def summarize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, g in frame.groupby("model"):
        for horizon, g2 in g.groupby("horizon"):
            a = g2["actual_return"]
            p = g2["predicted_return"]
            valid = g2.dropna(subset=["actual_return", "predicted_return"])
            rows.append(
                {
                    "model": model_name,
                    "horizon": int(horizon),
                    "MAE": mae(a, p),
                    "RMSE": rmse(a, p),
                    "DirAcc": diracc(a, p),
                    "N": int(len(valid)),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "horizon"]).reset_index(drop=True)


def fmt_float(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "nan"
    return f"{float(value):.6f}"


def write_markdown(report: str) -> None:
    REPORT_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"Saved prediction file not found: {PRED_PATH}")

    df = pd.read_parquet(PRED_PATH)
    df["forecast_timestamp"] = pd.to_datetime(df["forecast_timestamp"])
    df = df.sort_values(["model", "horizon", "forecast_timestamp"]).reset_index(drop=True)

    full_history = summarize_frame(df)
    full_history.to_csv(PROC / "existing_prediction_full_history_metrics.csv", index=False)

    annual_rows = []
    for (model_name, horizon), g in df.groupby(["model", "horizon"]):
        for year, g2 in g.groupby(df["forecast_timestamp"].dt.year):
            a = g2["actual_return"]
            p = g2["predicted_return"]
            valid = g2.dropna(subset=["actual_return", "predicted_return"])
            annual_rows.append(
                {
                    "model": model_name,
                    "horizon": int(horizon),
                    "year": int(year),
                    "MAE": mae(a, p),
                    "RMSE": rmse(a, p),
                    "DirAcc": diracc(a, p),
                    "N": int(len(valid)),
                }
            )
    annual = pd.DataFrame(annual_rows).sort_values(["model", "horizon", "year"]).reset_index(drop=True)
    annual.to_csv(PROC / "existing_prediction_annual_metrics.csv", index=False)

    max_ts = df["forecast_timestamp"].max()
    recent_3y = df[df["forecast_timestamp"] >= max_ts - pd.DateOffset(years=3)].copy()
    recent_1y = df[df["forecast_timestamp"] >= max_ts - pd.DateOffset(years=1)].copy()

    recent_3y_summary = summarize_frame(recent_3y)
    recent_1y_summary = summarize_frame(recent_1y)
    recent_3y_summary.to_csv(PROC / "existing_prediction_recent_3y_metrics.csv", index=False)
    recent_1y_summary.to_csv(PROC / "existing_prediction_recent_1y_metrics.csv", index=False)

    rolling_rows = []
    for model_name, g in df.groupby("model"):
        for horizon, g2 in g.groupby("horizon"):
            ordered = g2.sort_values("forecast_timestamp").set_index("forecast_timestamp").dropna(subset=["actual_return", "predicted_return"]).copy()
            ordered["abs_error"] = (ordered["predicted_return"] - ordered["actual_return"]).abs()
            ordered["sq_error"] = (ordered["predicted_return"] - ordered["actual_return"]) ** 2
            mae_roll = ordered["abs_error"].rolling("365D").mean()
            rmse_roll = np.sqrt(ordered["sq_error"].rolling("365D").mean())
            for ts, m, r in zip(ordered.index, mae_roll, rmse_roll):
                rolling_rows.append(
                    {
                        "model": model_name,
                        "horizon": int(horizon),
                        "forecast_timestamp": pd.Timestamp(ts),
                        "rolling365d_MAE": float(m) if pd.notna(m) else np.nan,
                        "rolling365d_RMSE": float(r) if pd.notna(r) else np.nan,
                    }
                )
    rolling = pd.DataFrame(rolling_rows).sort_values(["model", "horizon", "forecast_timestamp"]).reset_index(drop=True)
    rolling.to_csv(PROC / "existing_prediction_rolling365d_metrics.csv", index=False)

    lines = [
        "# External Information Robustness (existing prediction-level analysis)",
        "",
        "This document is based only on the saved prediction-level results already present in the repo. It does not retrain any model and does not create new backtest outputs.",
        "",
        "## Artifact status",
        "",
        "- Required A/B/C artifact path: `data/processed/incremental_model_results.parquet`",
        "- Status: not present in the current repo state",
        "- Therefore there are no persisted A/B/C prediction-level results available for the common-timestamp external-information analysis in this workspace.",
        "- Available saved prediction-level artifact in this repo: `data/processed/model_backtest_results.parquet` (price-history-only SARIMAX/XGBoost predictions only)",
        "",
        "## Available saved metrics from existing prediction files",
        "",
        "### Full-history metrics from saved prediction parquet",
        "",
        "| model | horizon | MAE | RMSE | DirAcc | N |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for _, row in full_history.sort_values(["model", "horizon"]).iterrows():
        lines.append(
            f"| {row['model']} | {int(row['horizon'])} | {fmt_float(row['MAE'])} | {fmt_float(row['RMSE'])} | {fmt_float(row['DirAcc'])} | {int(row['N'])} |"
        )

    lines.extend(["", "### Recent 3-year metrics", "", "| model | horizon | MAE | RMSE | DirAcc | N |", "|:---|---:|---:|---:|---:|---:|"])
    for _, row in recent_3y_summary.sort_values(["model", "horizon"]).iterrows():
        lines.append(
            f"| {row['model']} | {int(row['horizon'])} | {fmt_float(row['MAE'])} | {fmt_float(row['RMSE'])} | {fmt_float(row['DirAcc'])} | {int(row['N'])} |"
        )

    lines.extend(["", "### Recent 1-year metrics", "", "| model | horizon | MAE | RMSE | DirAcc | N |", "|:---|---:|---:|---:|---:|---:|"])
    for _, row in recent_1y_summary.sort_values(["model", "horizon"]).iterrows():
        lines.append(
            f"| {row['model']} | {int(row['horizon'])} | {fmt_float(row['MAE'])} | {fmt_float(row['RMSE'])} | {fmt_float(row['DirAcc'])} | {int(row['N'])} |"
        )

    lines.extend(["", "### Annual MAE / RMSE (saved price-history predictions)", "", "| model | horizon | year | MAE | RMSE | DirAcc | N |", "|:---|---:|---:|---:|---:|---:|---:|"])
    for _, row in annual.sort_values(["model", "horizon", "year"]).iterrows():
        lines.append(
            f"| {row['model']} | {int(row['horizon'])} | {int(row['year'])} | {fmt_float(row['MAE'])} | {fmt_float(row['RMSE'])} | {fmt_float(row['DirAcc'])} | {int(row['N'])} |"
        )

    lines.extend(["", "### Rolling 365d MAE / RMSE (saved price-history predictions)", "", "| model | horizon | forecast_timestamp | rolling365d_MAE | rolling365d_RMSE |", "|:---|---:|---|---:|---:|"])
    for _, row in rolling.sort_values(["model", "horizon", "forecast_timestamp"]).head(20).iterrows():
        ts = pd.Timestamp(row["forecast_timestamp"]).strftime("%Y-%m-%d")
        maev = fmt_float(row["rolling365d_MAE"])
        rmsev = fmt_float(row["rolling365d_RMSE"])
        lines.append(f"| {row['model']} | {int(row['horizon'])} | {ts} | {maev} | {rmsev} |")

    lines.extend([
        "",
        "## Stability conclusion",
        "",
        "The existing saved prediction-level results are not sufficient to establish stability across time for the external-information A/B/C experiment, because the required A/B/C artifact `data/processed/incremental_model_results.parquet` is absent from this workspace. The only persisted prediction-level file is the price-history-only SARIMAX/XGBoost set in `data/processed/model_backtest_results.parquet`, which is enough to summarize the baseline model behavior but not enough to claim stability for A/B/C or to compute the incremental external-information improvements.",
    ])

    write_markdown("\n".join(lines))
    print("Created:", REPORT_PATH)
    print("Saved summaries:")
    for fn in [
        "existing_prediction_full_history_metrics.csv",
        "existing_prediction_annual_metrics.csv",
        "existing_prediction_recent_3y_metrics.csv",
        "existing_prediction_recent_1y_metrics.csv",
        "existing_prediction_rolling365d_metrics.csv",
    ]:
        print(" -", PROC / fn)
