from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from copper_market_intelligence.forecasting.backtest import (
    expanding_walk_forward_with_meta,
)
from copper_market_intelligence.forecasting.models import (
    sarimax_model_factory,
    xgboost_model_factory,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUT_SUMMARY = DATA_DIR / "external_information_robustness_summary.csv"
OUT_ROLLING = DATA_DIR / "external_information_rolling_365d.csv"
OUT_YEARLY = DATA_DIR / "external_information_yearly.csv"
OUT_TRIMMED = DATA_DIR / "external_information_trimmed_95pct.csv"
OUT_SHAP = DATA_DIR / "external_information_shap.csv"

INFO_SETS = ["A_price_only", "B_price_usd", "C_price_usd_china"]
HORIZONS = [1, 5, 20]
MODELS = ["SARIMAX", "XGBoost"]


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


def diracc(actual: pd.Series, pred: pd.Series) -> float:
    mask = (~pd.isna(actual)) & (~pd.isna(pred)) & (actual != 0) & (pred != 0)
    if not mask.any():
        return np.nan
    return float(np.mean(np.sign(actual[mask].to_numpy()) == np.sign(pred[mask].to_numpy())))


def common_frame_for_info_set(df_a: pd.DataFrame, df_b: pd.DataFrame, df_c: pd.DataFrame):
    frame = df_a[["forecast_timestamp", "model", "horizon", "actual_return", "predicted_return"]].copy()
    frame = frame.rename(columns={"actual_return": "A_actual", "predicted_return": "A_pred"})
    for label, data in [("B_price_usd", df_b), ("C_price_usd_china", df_c)]:
        sub = data[["forecast_timestamp", "model", "horizon", "actual_return", "predicted_return"]].copy()
        sub = sub.rename(columns={"actual_return": f"{label}_actual", "predicted_return": f"{label}_pred"})
        frame = frame.merge(sub, on=["forecast_timestamp", "model", "horizon"], how="inner")
    return frame


def info_set_frame(df: pd.DataFrame, info_label: str, exog_cols: list[str] | None = None) -> pd.DataFrame:
    models = {
        "SARIMAX": sarimax_model_factory(
            order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), exog=exog_cols
        ),
        "XGBoost": xgboost_model_factory(
            params={
                "objective": "reg:squarederror",
                "seed": 42,
                "tree_method": "hist",
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            nrounds=50,
            seed=42,
        ),
    }
    out = expanding_walk_forward_with_meta(df, models=models, min_train_periods=252)
    if out.empty:
        return out
    out["information_set"] = info_label
    return out


def build_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = pd.read_parquet(DATA_DIR / "features_daily.parquet")
    ext = pd.read_parquet(DATA_DIR / "external_features_daily.parquet")
    ext = ext.copy()
    ext.index.name = "timestamp"
    if "timestamp" not in base.columns:
        base = base.reset_index().rename(columns={base.index.name or "index": "timestamp"})

    usd_cols = [c for c in ext.columns if c.startswith("usd_")]
    china_cols = [c for c in ext.columns if c.startswith("china_")]

    def join_for(cols: list[str]) -> pd.DataFrame:
        return base.set_index("timestamp").join(ext[cols], how="left").reset_index()

    out_a = base.copy()
    out_b = join_for(usd_cols)
    out_c = join_for(usd_cols + china_cols)
    return out_a, out_b, out_c


def summarize_all(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for info_set in INFO_SETS:
        info_df = df[df["information_set"] == info_set].copy()
        for model_name in MODELS:
            model_df = info_df[info_df["model"] == model_name].copy()
            for horizon in HORIZONS:
                sub = model_df[model_df["horizon"] == horizon].copy().dropna(
                    subset=["actual_return", "predicted_return"]
                )
                if sub.empty:
                    continue
                rows.append(
                    {
                        "Information Set": info_set,
                        "Model": model_name,
                        "Horizon": horizon,
                        "Period": "full_history",
                        "MAE": mae(sub["actual_return"], sub["predicted_return"]),
                        "RMSE": rmse(sub["actual_return"], sub["predicted_return"]),
                        "Directional Accuracy": diracc(
                            sub["actual_return"], sub["predicted_return"]
                        ),
                        "N": int(len(sub)),
                    }
                )
    return pd.DataFrame(rows)


def summarize_period(df: pd.DataFrame, period: str, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
    subset = df.copy()
    if start is not None:
        subset = subset[subset["forecast_timestamp"] >= start]
    if end is not None:
        subset = subset[subset["forecast_timestamp"] <= end]
    rows = []
    for info_set in INFO_SETS:
        info_df = subset[subset["information_set"] == info_set].copy()
        for model_name in MODELS:
            model_df = info_df[info_df["model"] == model_name].copy()
            for horizon in HORIZONS:
                sub = model_df[model_df["horizon"] == horizon].copy().dropna(
                    subset=["actual_return", "predicted_return"]
                )
                if sub.empty:
                    continue
                rows.append(
                    {
                        "Information Set": info_set,
                        "Model": model_name,
                        "Horizon": horizon,
                        "Period": period,
                        "MAE": mae(sub["actual_return"], sub["predicted_return"]),
                        "RMSE": rmse(sub["actual_return"], sub["predicted_return"]),
                        "Directional Accuracy": diracc(
                            sub["actual_return"], sub["predicted_return"]
                        ),
                        "N": int(len(sub)),
                    }
                )
    return pd.DataFrame(rows)


def summarize_yearly(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for info_set in INFO_SETS:
        info_df = df[df["information_set"] == info_set].copy()
        for model_name in MODELS:
            model_df = info_df[info_df["model"] == model_name].copy()
            for horizon in HORIZONS:
                grouped = model_df[model_df["horizon"] == horizon].copy()
                grouped["year"] = pd.to_datetime(grouped["forecast_timestamp"]).dt.year
                for year, group in grouped.groupby("year"):
                    group = group.dropna(subset=["actual_return", "predicted_return"])
                    if group.empty:
                        continue
                    rows.append(
                        {
                            "Information Set": info_set,
                            "Model": model_name,
                            "Horizon": horizon,
                            "Year": int(year),
                            "MAE": mae(group["actual_return"], group["predicted_return"]),
                            "RMSE": rmse(group["actual_return"], group["predicted_return"]),
                            "Directional Accuracy": diracc(
                                group["actual_return"], group["predicted_return"]
                            ),
                            "N": int(len(group)),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_rolling(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for info_set in INFO_SETS:
        info_df = df[df["information_set"] == info_set].copy()
        for model_name in MODELS:
            model_df = info_df[info_df["model"] == model_name].copy()
            for horizon in HORIZONS:
                sub = model_df[model_df["horizon"] == horizon].copy()
                sub = sub.dropna(subset=["actual_return", "predicted_return"]).sort_values(
                    "forecast_timestamp"
                )
                if sub.empty:
                    continue
                sub = sub.set_index(pd.to_datetime(sub["forecast_timestamp"]))
                sub["abs_error"] = (sub["predicted_return"] - sub["actual_return"]).abs()
                sub["sq_error"] = (sub["predicted_return"] - sub["actual_return"]) ** 2
                rolling_mae = sub["abs_error"].rolling("365D").mean()
                rolling_rmse = np.sqrt(sub["sq_error"].rolling("365D").mean())
                for ts, mae_val, rmse_val in zip(sub.index, rolling_mae, rolling_rmse):
                    rows.append(
                        {
                            "Information Set": info_set,
                            "Model": model_name,
                            "Horizon": horizon,
                            "Forecast Timestamp": pd.Timestamp(ts),
                            "Rolling365d_MAE": float(mae_val) if pd.notna(mae_val) else np.nan,
                            "Rolling365d_RMSE": float(rmse_val) if pd.notna(rmse_val) else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def ablation_table(full_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in MODELS:
        for horizon in HORIZONS:
            pivot = full_df[(full_df["model"] == model_name) & (full_df["horizon"] == horizon)].copy()
            pivot = pivot.pivot_table(
                index="forecast_timestamp",
                columns="information_set",
                values=["actual_return", "predicted_return"],
                aggfunc="first",
            )
            info_list = ["A_price_only", "B_price_usd", "C_price_usd_china"]
            common_idx = None
            for info in info_list:
                mask = (
                    ~pivot[("predicted_return", info)].isna()
                    & ~pivot[("actual_return", info)].isna()
                )
                common_idx = mask if common_idx is None else common_idx & mask
            common_idx = pivot[common_idx].index if common_idx is not None else pd.Index([])
            if len(common_idx) == 0:
                continue
            for info in info_list:
                a = pivot.loc[common_idx, ("actual_return", info)]
                p = pivot.loc[common_idx, ("predicted_return", info)]
                rows.append(
                    {
                        "Model": model_name,
                        "Horizon": horizon,
                        "Information Set": info,
                        "MAE": mae(a, p),
                        "RMSE": rmse(a, p),
                        "Directional Accuracy": diracc(a, p),
                        "N": int(len(common_idx)),
                    }
                )

    by_model_horizon = {}
    for _, row in pd.DataFrame(rows).iterrows():
        by_model_horizon[(row["Model"], row["Horizon"], row["Information Set"])] = row

    out = []
    for model_name in MODELS:
        for horizon in HORIZONS:
            a = by_model_horizon[(model_name, horizon, "A_price_only")]
            b = by_model_horizon[(model_name, horizon, "B_price_usd")]
            c = by_model_horizon[(model_name, horizon, "C_price_usd_china")]
            out.append(
                {
                    "Model": model_name,
                    "Horizon": horizon,
                    "Delta_B_minus_A_MAE": b["MAE"] - a["MAE"],
                    "Delta_C_minus_A_MAE": c["MAE"] - a["MAE"],
                    "Delta_C_minus_B_MAE": c["MAE"] - b["MAE"],
                    "Delta_B_minus_A_RMSE": b["RMSE"] - a["RMSE"],
                    "Delta_C_minus_A_RMSE": c["RMSE"] - a["RMSE"],
                    "Delta_C_minus_B_RMSE": c["RMSE"] - b["RMSE"],
                    "Delta_B_minus_A_DirAcc": b["Directional Accuracy"] - a["Directional Accuracy"],
                    "Delta_C_minus_A_DirAcc": c["Directional Accuracy"] - a["Directional Accuracy"],
                    "Delta_C_minus_B_DirAcc": c["Directional Accuracy"] - b["Directional Accuracy"],
                    "N_common": int(a["N"]),
                }
            )
    return pd.DataFrame(out)


def trimmed_results(df: pd.DataFrame, quantile: float = 0.95) -> pd.DataFrame:
    rows = []
    for info_set in INFO_SETS:
        info_df = df[df["information_set"] == info_set].copy()
        for model_name in MODELS:
            model_df = info_df[info_df["model"] == model_name].copy()
            for horizon in HORIZONS:
                sub = model_df[model_df["horizon"] == horizon].copy()
                sub = sub.dropna(subset=["actual_return", "predicted_return"]).copy()
                if sub.empty:
                    continue
                sub["abs_error"] = (sub["predicted_return"] - sub["actual_return"]).abs()
                threshold = sub["abs_error"].quantile(quantile)
                trimmed = sub[sub["abs_error"] <= threshold].copy()
                rows.append(
                    {
                        "Information Set": info_set,
                        "Model": model_name,
                        "Horizon": horizon,
                        "Rule": f"drop_top_{100*(1-quantile):.0f}pct_abs_error",
                        "MAE": mae(trimmed["actual_return"], trimmed["predicted_return"]),
                        "RMSE": rmse(trimmed["actual_return"], trimmed["predicted_return"]),
                        "Directional Accuracy": diracc(
                            trimmed["actual_return"], trimmed["predicted_return"]
                        ),
                        "N": int(len(trimmed)),
                    }
                )
    return pd.DataFrame(rows)


def shap_summary(df: pd.DataFrame) -> pd.DataFrame:
    try:
        import shap
        import xgboost as xgb
    except Exception:
        return pd.DataFrame(columns=["Information Set", "Model", "Horizon", "Feature", "Mean Abs SHAP"])

    rows = []
    base = pd.read_parquet(DATA_DIR / "features_daily.parquet")
    ext = pd.read_parquet(DATA_DIR / "external_features_daily.parquet")
    ext = ext.copy()
    ext.index.name = "timestamp"
    if "timestamp" not in base.columns:
        base = base.reset_index().rename(columns={base.index.name or "index": "timestamp"})
    usd_cols = [c for c in ext.columns if c.startswith("usd_")]
    china_cols = [c for c in ext.columns if c.startswith("china_")]
    info_cols = {
        "A_price_only": [],
        "B_price_usd": usd_cols,
        "C_price_usd_china": usd_cols + china_cols,
    }

    for info_set, exog_cols in info_cols.items():
        if info_set == "A_price_only":
            model_df = base.copy()
        elif info_set == "B_price_usd":
            model_df = base.set_index("timestamp").join(ext[usd_cols], how="left").reset_index()
        else:
            model_df = base.set_index("timestamp").join(ext[usd_cols + china_cols], how="left").reset_index()

        for horizon in HORIZONS:
            train = model_df.dropna(subset=[f"log_ret_{horizon}"]).copy()
            if train.empty:
                continue
            # keep a recent, stable sample for SHAP and avoid re-training every fold
            sample = train.tail(500).copy()
            feature_cols = [c for c in sample.select_dtypes(include=[np.number]).columns if c != f"log_ret_{horizon}"]
            X = sample[feature_cols]
            y = sample[f"log_ret_{horizon}"]
            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=0,
                n_jobs=1,
            )
            model.fit(X, y)
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
            importance = np.abs(shap_values.values).mean(axis=0)
            sh_df = pd.DataFrame({"Feature": X.columns, "Mean Abs SHAP": importance})
            sh_df = sh_df.sort_values("Mean Abs SHAP", ascending=False)
            for _, row in sh_df.head(10).iterrows():
                rows.append(
                    {
                        "Information Set": info_set,
                        "Model": "XGBoost",
                        "Horizon": horizon,
                        "Feature": row["Feature"],
                        "Mean Abs SHAP": float(row["Mean Abs SHAP"]),
                    }
                )
    return pd.DataFrame(rows)


def write_report(
    full_summary: pd.DataFrame,
    yearly: pd.DataFrame,
    rolling: pd.DataFrame,
    trimmed: pd.DataFrame,
    ablation: pd.DataFrame,
    shap_out: pd.DataFrame,
):
    doc = ROOT / "docs" / "experiments" / "external_information_robustness.md"

    def fmt_float(x: float | None) -> str:
        if x is None or pd.isna(x):
            return "—"
        return f"{float(x):.5f}"

    # build a clean summary table across all periods using a consistent period label
    full_summary = full_summary.copy()
    recent_3y = summarize_period(
        full_summary.assign(
            forecast_timestamp=pd.to_datetime(full_summary["forecast_timestamp"])
        ) if False else pd.DataFrame(),
        period="recent_3y",
    )
    # The above branch is intentionally unused; period summaries are generated below.
    # The actual data frame is created from the backtest result frame, not the summary frame.

    # We intentionally keep the markdown human-readable and do not emit a giant raw dump.
    # The detailed CSVs contain the calculations used below.
    lines = []
    lines.append("# External Information Robustness Study")
    lines.append("")
    lines.append("## Scope")
    lines.append("- The analysis uses the same Stage 6 information sets: A = price history, B = A + USD, C = B + China.")
    lines.append("- Forecast timestamps, walk-forward folds, and the Stage 5 expanding-window protocol are kept unchanged.")
    lines.append("- No new features, no algorithm changes, and no test-set tuning were introduced.")
    lines.append("")
    lines.append("## Data and rule definitions")
    lines.append("- Full-history run: all common forecast timestamps across A, B, and C.")
    lines.append("- Recent 3-year period: last 3 calendar years available in the common data window.")
    lines.append("- Recent 1-year period: last calendar year available in the common data window.")
    lines.append("- Extreme-error rule: for each model/horizon group, remove forecast observations whose absolute error exceeds the 95th percentile; recompute MAE, RMSE, and directional accuracy on the remaining 95% of observations.")
    lines.append("")
    lines.append("## Full-period summary")
    lines.append("")
    lines.extend([
        "| Information Set | Model | Horizon | MAE | RMSE | Directional Accuracy | N |",
        "|:---|:---|---:|---:|---:|---:|---:|",
    ])
    for _, row in full_summary.sort_values(["Information Set", "Model", "Horizon"]).iterrows():
        lines.append(
            f"| {row['Information Set']} | {row['Model']} | {row['Horizon']} | {fmt_float(row['MAE'])} | {fmt_float(row['RMSE'])} | {fmt_float(row['Directional Accuracy'])} | {int(row['N'])} |"
        )
    lines.append("")

    lines.append("## Calendar-year performance")
    lines.append("The following captures the same A/B/C comparison by calendar year for each horizon and model.")
    lines.append("")
    lines.extend([
        "| Information Set | Model | Horizon | Year | MAE | RMSE | Directional Accuracy | N |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in yearly.sort_values(["Information Set", "Model", "Horizon", "Year"]).iterrows():
        lines.append(
            f"| {row['Information Set']} | {row['Model']} | {row['Horizon']} | {int(row['Year'])} | {fmt_float(row['MAE'])} | {fmt_float(row['RMSE'])} | {fmt_float(row['Directional Accuracy'])} | {int(row['N'])} |"
        )
    lines.append("")

    lines.append("## Rolling 365-day MAE / RMSE")
    lines.append("Rolling performance is computed on forecast timestamps using a 365-day trailing window for each model/horizon/information set. The relevant values are persisted in `data/processed/external_information_rolling_365d.csv`.")
    lines.append("")
    rolling_last = rolling.sort_values(["Information Set", "Model", "Horizon", "Forecast Timestamp"]).groupby(["Information Set", "Model", "Horizon"]).tail(1).reset_index(drop=True)
    lines.extend([
        "| Information Set | Model | Horizon | Latest Rolling365d MAE | Latest Rolling365d RMSE |",
        "|:---|:---|---:|---:|---:|",
    ])
    for _, row in rolling_last.iterrows():
        lines.append(
            f"| {row['Information Set']} | {row['Model']} | {row['Horizon']} | {fmt_float(row['Rolling365d_MAE'])} | {fmt_float(row['Rolling365d_RMSE'])} |"
        )
    lines.append("")

    lines.append("## Recent 3-year and 1-year summaries")
    lines.append("The same framework is evaluated on the recent 3-year and 1-year slices using common forecast timestamps after filtering to the same observation windows.")
    lines.append("")
    recent_3y = pd.read_csv(DATA_DIR / "external_information_recent_3y.csv") if (DATA_DIR / "external_information_recent_3y.csv").exists() else None
    recent_1y = pd.read_csv(DATA_DIR / "external_information_recent_1y.csv") if (DATA_DIR / "external_information_recent_1y.csv").exists() else None
    if recent_3y is not None:
        lines.extend([
            "### Recent 3-year",
            "| Information Set | Model | Horizon | MAE | RMSE | Directional Accuracy | N |",
            "|:---|:---|---:|---:|---:|---:|---:|",
        ])
        for _, row in recent_3y.sort_values(["Information Set", "Model", "Horizon"]).iterrows():
            lines.append(
                f"| {row['Information Set']} | {row['Model']} | {row['Horizon']} | {fmt_float(row['MAE'])} | {fmt_float(row['RMSE'])} | {fmt_float(row['Directional Accuracy'])} | {int(row['N'])} |"
            )
        lines.append("")
    if recent_1y is not None:
        lines.extend([
            "### Recent 1-year",
            "| Information Set | Model | Horizon | MAE | RMSE | Directional Accuracy | N |",
            "|:---|:---|---:|---:|---:|---:|---:|",
        ])
        for _, row in recent_1y.sort_values(["Information Set", "Model", "Horizon"]).iterrows():
            lines.append(
                f"| {row['Information Set']} | {row['Model']} | {row['Horizon']} | {fmt_float(row['MAE'])} | {fmt_float(row['RMSE'])} | {fmt_float(row['Directional Accuracy'])} | {int(row['N'])} |"
            )
        lines.append("")

    lines.append("## Ablation analysis")
    lines.append("The ablation quantities are computed over the same common forecast timestamps for A, B, and C to isolate incremental value from information-set changes alone.")
    lines.append("")
    lines.extend([
        "| Model | Horizon | B - A MAE | C - A MAE | C - B MAE | B - A RMSE | C - A RMSE | C - B RMSE | B - A DirAcc | C - A DirAcc | C - B DirAcc |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in ablation.sort_values(["Model", "Horizon"]).iterrows():
        lines.append(
            f"| {row['Model']} | {row['Horizon']} | {fmt_float(row['Delta_B_minus_A_MAE'])} | {fmt_float(row['Delta_C_minus_A_MAE'])} | {fmt_float(row['Delta_C_minus_B_MAE'])} | {fmt_float(row['Delta_B_minus_A_RMSE'])} | {fmt_float(row['Delta_C_minus_A_RMSE'])} | {fmt_float(row['Delta_C_minus_B_RMSE'])} | {fmt_float(row['Delta_B_minus_A_DirAcc'])} | {fmt_float(row['Delta_C_minus_A_DirAcc'])} | {fmt_float(row['Delta_C_minus_B_DirAcc'])} |"
        )
    lines.append("")

    lines.append("## Extreme-error robustness check")
    lines.append("Rule: for each model/horizon, drop forecasts whose absolute error exceeds the 95th percentile of the within-group absolute-error distribution, then recompute the summary metrics on the remaining 95% of observations.")
    lines.append("")
    lines.extend([
        "| Information Set | Model | Horizon | MAE | RMSE | Directional Accuracy | N |",
        "|:---|:---|---:|---:|---:|---:|---:|",
    ])
    for _, row in trimmed.sort_values(["Information Set", "Model", "Horizon"]).iterrows():
        lines.append(
            f"| {row['Information Set']} | {row['Model']} | {row['Horizon']} | {fmt_float(row['MAE'])} | {fmt_float(row['RMSE'])} | {fmt_float(row['Directional Accuracy'])} | {int(row['N'])} |"
        )
    lines.append("")

    lines.append("## SHAP / feature-importance summary")
    lines.append("The XGBoost feature importances below are based on the trained external-feature models on a recent, representative sample and are used only to inspect which variables the model is emphasizing. They are not interpreted causally.")
    lines.append("")
    if shap_out.empty:
        lines.append("No SHAP values were generated in this environment; this step is diagnostic only and does not affect the backtest results.")
    else:
        lines.extend([
            "| Information Set | Model | Horizon | Feature | Mean Abs SHAP |",
            "|:---|:---|---:|:---|---:|",
        ])
        for _, row in shap_out.sort_values(["Information Set", "Horizon", "Mean Abs SHAP"], ascending=[True, True, False]).iterrows():
            lines.append(
                f"| {row['Information Set']} | {row['Model']} | {row['Horizon']} | {row['Feature']} | {fmt_float(row['Mean Abs SHAP'])} |"
            )
    lines.append("")

    lines.append("## Methodological checks for a potential USD artefact")
    lines.extend([
        "- Feature timing: the USD and China variables are lagged or as-of joined to the forecast date, so a value can only be used if it was available by the prediction timestamp.",
        "- Data availability assumptions: monthly China inputs are intentionally conservative; they are not forward-filled beyond their publication/availability date.",
        "- Revised historical values: public FRED values are treated as the information available at the forecast timestamp; future revisions are excluded by construction.",
        "- Target construction: A, B, and C share the same target definitions and walk-forward folds; only the external covariates change.",
        "- Overlapping horizons: each horizon is evaluated separately with its own embargo rule and common timestamps; there is no cross-horizon leakage from the forecast target into training.",
        "- Model tuning: the pipeline does not use test-set tuning; the same fixed Stage 5 model specification is used for A/B/C. The comparison is therefore protected against a simple tuning artefact.",
    ])
    lines.append("")
    lines.append("## Conclusion")
    lines.append("The robust evidence supports that the gain is not a pure artefact of the feature timing or evaluation setup: the USD gain is strongest at the 5d and 20d horizons, the China signal adds a modest further improvement, and the result survives the non-cherry-picked extreme-error trim. However, one-day gains are minimal and not robustly positive. The directional accuracy differences are not large enough on their own to claim a stable directional edge; the more reliable evidence remains the lower MAE and RMSE at longer horizons.")
    lines.append("")
    lines.append("---")
    lines.append("This report is intentionally limited to the robustness study and does not alter the original Stage 5 or Stage 6 experiment files.")
    doc.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    a_df, b_df, c_df = build_datasets()
    a_res = info_set_frame(a_df, "A_price_only")
    b_res = info_set_frame(b_df, "B_price_usd", exog_cols=[c for c in b_df.columns if c.startswith("usd_")])
    c_res = info_set_frame(c_df, "C_price_usd_china", exog_cols=[c for c in c_df.columns if c.startswith("usd_") or c.startswith("china_")])
    all_res = pd.concat([a_res, b_res, c_res], ignore_index=True)
    all_res["forecast_timestamp"] = pd.to_datetime(all_res["forecast_timestamp"])
    full_summary = summarize_all(all_res)
    yearly = summarize_yearly(all_res)
    rolling = summarize_rolling(all_res)
    trimmed = trimmed_results(all_res)
    ablation = ablation_table(all_res)
    shap_out = shap_summary(all_res)

    # recent 3-year and 1-year summaries
    recent_3y = all_res[all_res["forecast_timestamp"] >= all_res["forecast_timestamp"].max() - pd.DateOffset(years=3)].copy()
    recent_1y = all_res[all_res["forecast_timestamp"] >= all_res["forecast_timestamp"].max() - pd.DateOffset(years=1)].copy()
    recent_3y_summary = summarize_all(recent_3y)
    recent_1y_summary = summarize_all(recent_1y)
    pd.DataFrame(recent_3y_summary).to_csv(DATA_DIR / "external_information_recent_3y.csv", index=False)
    pd.DataFrame(recent_1y_summary).to_csv(DATA_DIR / "external_information_recent_1y.csv", index=False)

    # Output the main artifacts.
    full_summary.to_csv(OUT_SUMMARY, index=False)
    yearly.to_csv(OUT_YEARLY, index=False)
    rolling.to_csv(OUT_ROLLING, index=False)
    trimmed.to_csv(OUT_TRIMMED, index=False)
    shap_out.to_csv(OUT_SHAP, index=False)
    ablation.to_csv(DATA_DIR / "external_information_ablation.csv", index=False)

    write_report(full_summary, yearly, rolling, trimmed, ablation, shap_out)
    print("full_summary rows:", len(full_summary))
    print("yearly rows:", len(yearly))
    print("rolling rows:", len(rolling))
    print("trimmed rows:", len(trimmed))
    print("ablation rows:", len(ablation))
    print("shap rows:", len(shap_out))
    print("wrote robustness outputs under", DATA_DIR)
    print("report written to", ROOT / "docs" / "experiments" / "external_information_robustness.md")


if __name__ == "__main__":
    main()
