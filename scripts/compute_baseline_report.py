from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def mase(series, naive_mae):
    if naive_mae == 0:
        return float("nan")
    return series.mean() / naive_mae


def summarize_backtest(res: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    models = sorted(res["model_name"].unique())
    horizons = sorted(res["target_horizon"].unique())

    lines = []
    lines.append("# Baseline Backtest Results")
    lines.append("")

    # compute baseline naive MAE per horizon using zero model
    naive = res[res["model_name"] == "zero"].copy()
    naive_mae = (
        naive.dropna(subset=["actual_return", "predicted_return"])
        .assign(err=lambda df: (df["predicted_return"] - df["actual_return"]).abs())
        .groupby("target_horizon")["err"]
        .mean()
        .to_dict()
    )

    # overall table
    rows = []
    for model in models:
        for h in horizons:
            df = res[(res["model_name"] == model) & (res["target_horizon"] == h)].copy()
            valid = df.dropna(subset=["actual_return", "predicted_return"])
            n = len(valid)
            if n == 0:
                continue
            errors = valid["predicted_return"] - valid["actual_return"]
            mae = float(np.mean(np.abs(errors)))
            rmse = float(math.sqrt(np.mean(errors ** 2)))
            bias = float(np.mean(errors))
            dir_acc = float((np.sign(valid["predicted_return"]) == np.sign(valid["actual_return"])).mean())
            mase_v = mase(np.abs(errors), naive_mae.get(h, float("nan")))
            rows.append((model, h, mae, rmse, mase_v, dir_acc, bias, n))

    df_table = pd.DataFrame(rows, columns=["model", "horizon", "MAE", "RMSE", "MASE", "DirAcc", "Bias", "N"])
    lines.append("## Summary table")
    lines.append("")
    lines.append(df_table.to_markdown(index=False))
    lines.append("")

    # performance by year
    res["year"] = res["timestamp"].dt.year
    lines.append("## Performance by Year (MAE)")
    lines.append("")
    for model in models:
        for h in horizons:
            df = res[(res["model_name"] == model) & (res["target_horizon"] == h)].dropna(subset=["actual_return", "predicted_return"])
            if df.empty:
                continue
            per_year = (
                (df.assign(err=lambda d: (d["predicted_return"] - d["actual_return"]).abs()) )
                .groupby("year")
                ["err"].mean()
                .reset_index()
            )
            lines.append(f"### {model} horizon {h}")
            lines.append("")
            lines.append(per_year.to_markdown(index=False))
            lines.append("")

    # plots: cumulative actual vs predicted (for each model/horizon)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    for model in models:
        for h in horizons:
            df = res[(res["model_name"] == model) & (res["target_horizon"] == h)].copy()
            df = df.sort_values("timestamp")
            df2 = df.dropna(subset=["actual_return", "predicted_return"]).copy()
            if df2.empty:
                continue
            # cumulative returns
            df2["cum_actual"] = np.cumsum(df2["actual_return"].fillna(0))
            df2["cum_pred"] = np.cumsum(df2["predicted_return"].fillna(0))
            plt.figure(figsize=(8, 4))
            plt.plot(df2["timestamp"], df2["cum_actual"], label="actual")
            plt.plot(df2["timestamp"], df2["cum_pred"], label="predicted")
            plt.title(f"Cumulative returns: {model} horizon {h}")
            plt.legend()
            pfile = plot_dir / f"cum_{model}_h{h}.png"
            plt.tight_layout()
            plt.savefig(pfile)
            plt.close()

            # error histogram
            plt.figure(figsize=(6, 3))
            errors = df2["predicted_return"] - df2["actual_return"]
            plt.hist(errors.dropna(), bins=50)
            plt.title(f"Error distribution: {model} h{h}")
            pfile2 = plot_dir / f"errhist_{model}_h{h}.png"
            plt.tight_layout()
            plt.savefig(pfile2)
            plt.close()

    md_path = out_dir / "baseline_results.md"
    md_text = "\n".join(lines)
    md_path.write_text(md_text)
    print('wrote', md_path)


if __name__ == '__main__':
    out_dir = Path('docs')
    res = pd.read_parquet('data/processed/baseline_backtest_results.parquet')
    summarize_backtest(res, out_dir)
