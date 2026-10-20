from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def directional_accuracy_policy(preds, actuals):
    # preds, actuals are pd.Series aligned
    df = pd.DataFrame({"pred": preds, "act": actuals}).dropna()
    if df.empty:
        return {
            "n": 0,
            "n_zero_pred": 0,
            "n_zero_act": 0,
            "n_excluded": 0,
            "dir_acc": float("nan"),
        }
    # count zeros in predictions
    n_zero_pred = (df["pred"] == 0).sum()
    # count zeros in actual
    n_zero_act = (df["act"] == 0).sum()
    # exclude rows where pred == 0 unless both 0
    mask_nonzero = (df["pred"] != 0)
    df_dir = df[mask_nonzero]
    n_excluded = len(df) - len(df_dir)
    if df_dir.empty:
        dir_acc = float("nan")
    else:
        dir_acc = (np.sign(df_dir["pred"]) == np.sign(df_dir["act"])) .mean()
    return {
        "n": len(df),
        "n_zero_pred": int(n_zero_pred),
        "n_zero_act": int(n_zero_act),
        "n_excluded": int(n_excluded),
        "dir_acc": float(dir_acc),
    }


def compute_mase(errors_abs, naive_mae):
    if naive_mae == 0:
        return float("nan")
    return float(errors_abs.mean() / naive_mae)


def audit_backtest(res_meta: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # normalize column names from different backtest outputs
    if "model" not in res_meta.columns and "model_name" in res_meta.columns:
        res_meta = res_meta.rename(columns={"model_name": "model"})
    if "horizon" not in res_meta.columns and "target_horizon" in res_meta.columns:
        res_meta = res_meta.rename(columns={"target_horizon": "horizon"})
    # count predictions
    total_preds = len(res_meta)

    models = sorted(res_meta["model"].unique())
    horizons = sorted(res_meta["horizon"].unique())

    lines = ["# Backtest Audit Report", ""]
    lines.append(f"Total predictions: {total_preds}")
    lines.append("")

    # naive MAE per horizon (zero model) based on training period? Here we follow earlier: naive MAE computed from zero model predictions errors in the test records
    naive = res_meta[res_meta["model"] == "zero"].dropna(subset=["actual_return", "predicted_return"]).copy()
    naive_mae = naive.assign(err=lambda d: (d["predicted_return"] - d["actual_return"]).abs()).groupby("horizon")["err"].mean().to_dict()

    rows = []
    for model in models:
        for h in horizons:
            df = res_meta[(res_meta["model"] == model) & (res_meta["horizon"] == h)].copy()
            n = len(df)
            valid = df.dropna(subset=["actual_return", "predicted_return"])
            n_valid = len(valid)
            n_zero_actual = int((valid["actual_return"] == 0).sum())
            n_zero_pred = int((valid["predicted_return"] == 0).sum())
            # errors
            errors = valid["predicted_return"] - valid["actual_return"]
            mae = float(np.mean(np.abs(errors))) if n_valid else float("nan")
            rmse = float(np.sqrt(np.mean(errors ** 2))) if n_valid else float("nan")
            mase = compute_mase(np.abs(errors), naive_mae.get(h, float("nan")))
            dir_stats = directional_accuracy_policy(valid["predicted_return"], valid["actual_return"])
            rows.append({
                "model": model,
                "horizon": h,
                "N_preds": n,
                "N_valid": n_valid,
                "N_zero_actual": n_zero_actual,
                "N_zero_pred": n_zero_pred,
                "MAE": mae,
                "RMSE": rmse,
                "MASE": mase,
                "DirAcc": dir_stats["dir_acc"],
                "Dir_n": dir_stats["n"] - dir_stats["n_excluded"],
                "Dir_excluded": dir_stats["n_excluded"],
            })
    df_report = pd.DataFrame(rows)
    txt = ["# Audit summary", "", df_report.to_markdown(index=False), ""]
    (out_dir / "backtest_audit.md").write_text("\n".join(txt))
    print('wrote audit to', out_dir / 'backtest_audit.md')


if __name__ == '__main__':
    res = pd.read_parquet('data/processed/baseline_backtest_results.parquet')
    audit_backtest(res, Path('docs'))
