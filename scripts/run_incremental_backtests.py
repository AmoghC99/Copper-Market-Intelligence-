"""Run incremental backtests for Models A-C and compute aggregate metrics.

Saves:
- data/processed/incremental_model_predictions.parquet
- data/processed/incremental_model_metrics.parquet
- data/processed/incremental_experiment_manifest.json

Usage:
    python scripts/run_incremental_backtests.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copper_market_intelligence.forecasting.backtest import (  # noqa: E402
    expanding_walk_forward_with_meta,
)
from copper_market_intelligence.forecasting.models import (  # noqa: E402
    sarimax_model_factory,
    xgboost_model_factory,
)

DATA_DIR = ROOT / "data" / "processed"
OUT_PREDICTIONS = DATA_DIR / "incremental_model_predictions.parquet"
OUT_METRICS = DATA_DIR / "incremental_model_metrics.parquet"
OUT_RESULTS = DATA_DIR / "incremental_model_results.parquet"
OUT_AGG = DATA_DIR / "incremental_aggregate_metrics.csv"
MANIFEST_PATH = DATA_DIR / "incremental_experiment_manifest.json"

INFO_SETS = ["A_price_only", "B_price_usd", "C_price_usd_china"]
HORIZONS = [1, 5, 20]
MODELS = ["SARIMAX", "XGBoost"]
RANDOM_SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_manifest(base_path: Path, ext_path: Path) -> None:
    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        git_hash = "unknown"

    manifest = {
        "git_commit_hash": git_hash,
        "feature_dataset_checksums": {
            "features_daily.parquet": sha256_file(base_path),
            "external_features_daily.parquet": sha256_file(ext_path),
        },
        "target_definition": {
            "name": "log_return_future_horizon",
            "description": "Each forecast uses the realized log return over the next h trading days, where h is one of [1, 5, 20].",
            "horizons": HORIZONS,
        },
        "model_configuration": {
            "SARIMAX": {
                "order": [1, 0, 0],
                "seasonal_order": [0, 0, 0, 0],
                "enforce_stationarity": False,
                "enforce_invertibility": False,
            },
            "XGBoost": {
                "params": {"objective": "reg:squarederror", "seed": RANDOM_SEED},
                "nrounds": 50,
                "seed": RANDOM_SEED,
            },
        },
        "backtest_configuration": {
            "min_train_periods": 252,
            "retrain_frequency": 1,
            "embargo_default": "horizon",
            "horizons": HORIZONS,
            "information_sets": INFO_SETS,
            "common_timestamp_methodology": "For each model+horizon, metrics are computed only on the intersection of A/B/C forecast timestamps.",
        },
        "evaluation_window": {
            "start": None,
            "end": None,
        },
        "random_seed": RANDOM_SEED,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def mae(a, b):
    return float(np.nanmean(np.abs(a - b)))


def rmse(a, b):
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def directional_accuracy(a, b):
    a = pd.Series(a).astype(float)
    b = pd.Series(b).astype(float)
    mask = (~pd.isna(a)) & (~pd.isna(b)) & (a != 0) & (b != 0)
    if not mask.any():
        return np.nan
    return float((np.sign(a[mask].to_numpy()) == np.sign(b[mask].to_numpy())).mean())


def mase(a, b):
    mask = (~pd.isna(a)) & (~pd.isna(b))
    if mask.sum() == 0:
        return np.nan
    scale = np.nanmean(np.abs(a[mask]))
    if scale == 0 or np.isnan(scale):
        return np.nan
    return float(np.nanmean(np.abs(a[mask] - b[mask])) / scale)


def run_for_info_set(df: pd.DataFrame, info_set_name: str, exog_cols: list[str] | None = None):
    sarimax = sarimax_model_factory(
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        exog=exog_cols,
    )
    xgb = xgboost_model_factory(
        params={"objective": "reg:squarederror", "seed": RANDOM_SEED},
        nrounds=50,
        seed=RANDOM_SEED,
    )
    models = {"SARIMAX": sarimax, "XGBoost": xgb}
    res = expanding_walk_forward_with_meta(df, models=models)
    if not res.empty:
        res["information_set"] = info_set_name
    return res


def build_common_timestamp_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in sorted(predictions["model"].dropna().unique()):
        for horizon in sorted(predictions["horizon"].dropna().unique()):
            sub = predictions[
                (predictions["model"] == model_name)
                & (predictions["horizon"] == horizon)
            ].copy()
            counts = {}
            for info_set in INFO_SETS:
                info_sub = sub[sub["information_set"] == info_set].copy()
                counts[info_set] = int(info_sub["forecast_timestamp"].nunique())
            common_ts = set(sub[sub["information_set"] == INFO_SETS[0]]["forecast_timestamp"].astype(str))
            for info_set in INFO_SETS[1:]:
                common_ts &= set(sub[sub["information_set"] == info_set]["forecast_timestamp"].astype(str))
            rows.append(
                {
                    "model": model_name,
                    "horizon": int(horizon),
                    "A_available_predictions": counts[INFO_SETS[0]],
                    "B_available_predictions": counts[INFO_SETS[1]],
                    "C_available_predictions": counts[INFO_SETS[2]],
                    "common_prediction_count": int(len(common_ts)),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "horizon"]).reset_index(drop=True)


def build_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    common_summary = build_common_timestamp_summary(predictions)
    common_lookup = {
        (row["model"], int(row["horizon"])): row["common_prediction_count"]
        for _, row in common_summary.iterrows()
    }

    for model_name in sorted(predictions["model"].dropna().unique()):
        for horizon in sorted(predictions["horizon"].dropna().unique()):
            subset = predictions[
                (predictions["model"] == model_name)
                & (predictions["horizon"] == horizon)
            ].copy()
            available = {
                info_set: subset[subset["information_set"] == info_set].copy()
                for info_set in INFO_SETS
            }
            all_common_ts = None
            for info_set in INFO_SETS:
                ts = set(available[info_set]["forecast_timestamp"].astype(str))
                all_common_ts = ts if all_common_ts is None else all_common_ts & ts
            all_common_ts = sorted(all_common_ts)
            for info_set in INFO_SETS:
                info_df = available[info_set].copy()
                info_df = info_df[info_df["forecast_timestamp"].astype(str).isin(all_common_ts)].copy()
                if info_df.empty:
                    continue
                actual = info_df["actual_return"].reset_index(drop=True)
                pred = info_df["predicted_return"].reset_index(drop=True)
                rows.append(
                    {
                        "information_set": info_set,
                        "model": model_name,
                        "horizon": int(horizon),
                        "MAE": mae(actual, pred),
                        "RMSE": rmse(actual, pred),
                        "MASE": mase(actual, pred),
                        "DirAcc": directional_accuracy(actual, pred),
                        "Bias": float(np.nanmean(pred - actual)),
                        "N": int(len(info_df)),
                        "common_prediction_count": int(common_lookup.get((model_name, int(horizon)), 0)),
                    }
                )

    metrics = pd.DataFrame(rows).sort_values(["model", "horizon", "information_set"]).reset_index(drop=True)
    for model_name in sorted(metrics["model"].unique()):
        for horizon in sorted(metrics["horizon"].unique()):
            sub = metrics[(metrics["model"] == model_name) & (metrics["horizon"] == horizon)].copy()
            lookup = {row["information_set"]: row for _, row in sub.iterrows()}
            a = lookup["A_price_only"]
            b = lookup["B_price_usd"]
            c = lookup["C_price_usd_china"]
            for key, base_row in [("A_price_only", a), ("B_price_usd", b), ("C_price_usd_china", c)]:
                metrics.loc[
                    (metrics["model"] == model_name)
                    & (metrics["horizon"] == horizon)
                    & (metrics["information_set"] == key),
                    "delta_vs_A_MAE",
                ] = (base_row["MAE"] - a["MAE"]) if key != "A_price_only" else 0.0
                metrics.loc[
                    (metrics["model"] == model_name)
                    & (metrics["horizon"] == horizon)
                    & (metrics["information_set"] == key),
                    "delta_vs_A_RMSE",
                ] = (base_row["RMSE"] - a["RMSE"]) if key != "A_price_only" else 0.0
            if "B_price_usd" in lookup and "C_price_usd_china" in lookup:
                metrics.loc[
                    (metrics["model"] == model_name)
                    & (metrics["horizon"] == horizon)
                    & (metrics["information_set"] == "C_price_usd_china"),
                    "delta_vs_B_MAE",
                ] = c["MAE"] - b["MAE"]
                metrics.loc[
                    (metrics["model"] == model_name)
                    & (metrics["horizon"] == horizon)
                    & (metrics["information_set"] == "C_price_usd_china"),
                    "delta_vs_B_RMSE",
                ] = c["RMSE"] - b["RMSE"]
    return metrics


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore")

    if OUT_RESULTS.exists() and not OUT_PREDICTIONS.exists():
        predictions = pd.read_parquet(OUT_RESULTS)
    elif OUT_RESULTS.exists() and OUT_METRICS.exists() is False:
        predictions = pd.read_parquet(OUT_RESULTS)
    else:
        base = pd.read_parquet(DATA_DIR / "features_daily.parquet")
        ext = pd.read_parquet(DATA_DIR / "external_features_daily.parquet")
        ext = ext.copy()
        ext.index.name = "timestamp"

        if "timestamp" not in base.columns:
            base = base.reset_index().rename(columns={base.index.name or "index": "timestamp"})

        record_manifest(DATA_DIR / "features_daily.parquet", DATA_DIR / "external_features_daily.parquet")

        usd_cols = [c for c in ext.columns if c.startswith("usd_")]
        china_cols = [c for c in ext.columns if c.startswith("china_")]

        def join_for(cols: list[str], frame: pd.DataFrame) -> pd.DataFrame:
            return frame.set_index("timestamp").join(ext[cols], how="left").reset_index()

        df_a = base.copy()
        df_b = join_for(usd_cols, base)
        df_c = join_for(usd_cols + china_cols, base)

        print("Running Model A (price-only)...")
        res_a = run_for_info_set(df_a, "A_price_only", exog_cols=None)
        print("Running Model B (price + USD)...")
        res_b = run_for_info_set(df_b, "B_price_usd", exog_cols=usd_cols)
        print("Running Model C (price + USD + China)...")
        res_c = run_for_info_set(df_c, "C_price_usd_china", exog_cols=usd_cols + china_cols)

        predictions = pd.concat([res_a, res_b, res_c], ignore_index=True)
        if predictions.empty:
            raise RuntimeError("No prediction records were generated for the incremental A/B/C backtest.")

        predictions = predictions.rename(columns={"forecast_timestamp": "forecast_timestamp"})
        predictions["error"] = predictions["predicted_return"] - predictions["actual_return"]
        predictions["training_end_timestamp"] = pd.to_datetime(predictions["training_end_timestamp"])
        predictions["target_end_timestamp"] = pd.to_datetime(predictions["target_end_timestamp"])
        predictions["forecast_timestamp"] = pd.to_datetime(predictions["forecast_timestamp"])
        predictions["error"] = predictions["error"].astype(float)
        predictions["information_set"] = predictions["information_set"].astype(str)
        predictions["model"] = predictions["model"].astype(str)
        predictions["horizon"] = predictions["horizon"].astype(int)

        required_columns = [
            "information_set",
            "model",
            "horizon",
            "forecast_timestamp",
            "training_end_timestamp",
            "target_end_timestamp",
            "actual_return",
            "predicted_return",
            "error",
        ]
        missing = [col for col in required_columns if col not in predictions.columns]
        if missing:
            raise KeyError(f"Missing required prediction columns: {missing}")
        predictions = predictions[required_columns].copy()

        predictions.to_parquet(OUT_PREDICTIONS, index=False)
        OUT_RESULTS.write_bytes(OUT_PREDICTIONS.read_bytes())
        print(f"wrote predictions to {OUT_PREDICTIONS}")

    if "information_set" not in predictions.columns:
        predictions = predictions.rename(columns={"information_set": "information_set"})
    if "error" not in predictions.columns:
        predictions["error"] = predictions["predicted_return"] - predictions["actual_return"]
    predictions["forecast_timestamp"] = pd.to_datetime(predictions["forecast_timestamp"])
    predictions["training_end_timestamp"] = pd.to_datetime(predictions["training_end_timestamp"])
    predictions["target_end_timestamp"] = pd.to_datetime(predictions["target_end_timestamp"])
    predictions["model"] = predictions["model"].astype(str)
    predictions["information_set"] = predictions["information_set"].astype(str)
    predictions["horizon"] = predictions["horizon"].astype(int)

    metrics = build_metrics(predictions)
    metrics.to_parquet(OUT_METRICS, index=False)
    metrics.to_csv(OUT_AGG, index=False)
    print(f"wrote aggregate metrics to {OUT_METRICS}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["evaluation_window"] = {
        "start": predictions["forecast_timestamp"].min().isoformat(),
        "end": predictions["forecast_timestamp"].max().isoformat(),
    }
    manifest["common_timestamp_counts"] = build_common_timestamp_summary(predictions).to_dict(orient="records")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Common timestamp counts:")
    print(build_common_timestamp_summary(predictions).to_string(index=False))
    print("==================================================")
    print(f"Prediction rows: {len(predictions)}")
    print(f"Prediction artifact: {OUT_PREDICTIONS}")
    print(f"Metrics artifact: {OUT_METRICS}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
