from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mase(errors_abs, naive_mae):
    if naive_mae == 0 or np.isnan(naive_mae):
        return np.nan
    return errors_abs.mean() / naive_mae


def directional_accuracy(preds, actuals):
    df = pd.DataFrame({'pred': preds, 'act': actuals}).dropna()
    if df.empty:
        return np.nan, 0, 0
    mask = df['pred'] != 0
    n_excluded = (~mask).sum()
    df2 = df[mask]
    if df2.empty:
        return np.nan, int(n_excluded), len(df)
    return (np.sign(df2['pred']) == np.sign(df2['act'])).mean(), int(n_excluded), len(df)


def summarize(path: str = 'data/processed/model_backtest_results.parquet', out_dir: str = 'docs'):
    df = pd.read_parquet(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # naive per-horizon MAE from zero model
    naive = df[df['model'] == 'zero'].copy()
    naive['err_abs'] = (naive['predicted_return'] - naive['actual_return']).abs()
    naive_mae = naive.groupby('horizon')['err_abs'].mean().to_dict()

    rows = []
    for (model, h), g in df.groupby(['model', 'horizon']):
        g = g.dropna(subset=['predicted_return', 'actual_return'])
        n = len(g)
        if n == 0:
            rows.append({'model': model, 'horizon': h, 'N': 0})
            continue
        errors = g['predicted_return'] - g['actual_return']
        mae = mean_absolute_error(g['actual_return'], g['predicted_return'])
        rmse = float(np.sqrt(mean_squared_error(g['actual_return'], g['predicted_return'])))
        mase_v = mase(errors.abs(), naive_mae.get(h, np.nan))
        dir_acc, n_excl, n_total = directional_accuracy(g['predicted_return'], g['actual_return'])
        bias = errors.mean()
        rows.append({
            'model': model,
            'horizon': h,
            'MAE': mae,
            'RMSE': rmse,
            'MASE': mase_v,
            'DirAcc': dir_acc,
            'Bias': bias,
            'N': n,
            'Dir_excluded': n_excl,
        })
    out = pd.DataFrame(rows)
    (out_dir / 'model_comparison.md').write_text('# Model comparison\n\n' + out.to_markdown(index=False))
    print('wrote', out_dir / 'model_comparison.md')


if __name__ == '__main__':
    summarize()
