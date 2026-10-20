from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

OUT_DIR = Path('docs/figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_backtest(path: Path):
    df = pd.read_parquet(path)
    return df


def validate_backtest(df: pd.DataFrame):
    required = [
        'model', 'horizon', 'forecast_timestamp', 'training_end_timestamp',
        'target_end_timestamp', 'actual_return', 'predicted_return', 'error'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise AssertionError(f'missing required columns: {missing}')

    issues = []
    # duplicates
    dup = df.duplicated(subset=['model', 'horizon', 'forecast_timestamp']).sum()
    if dup:
        issues.append(f'duplicate forecast records: {dup}')

    # chronological and timestamp relations
    for (m, h), g in df.groupby(['model', 'horizon']):
        g2 = g.sort_values('forecast_timestamp')
        if not g2['forecast_timestamp'].is_monotonic_increasing:
            issues.append(f'forecast_timestamp not chronological for {m} h{h}')
        # check training_end <= forecast
        bad = (g2['training_end_timestamp'] > g2['forecast_timestamp']).sum()
        if bad:
            issues.append(f'{bad} rows where training_end_timestamp > forecast for {m} h{h}')
        # target_end > forecast
        bad2 = (g2['target_end_timestamp'] <= g2['forecast_timestamp']).sum()
        if bad2:
            issues.append(f'{bad2} rows where target_end_timestamp <= forecast for {m} h{h}')

    # null predictions
    null_preds = df['predicted_return'].isna().sum()
    if null_preds:
        issues.append(f'null predicted_return: {null_preds}')

    # numeric types
    for c in ['actual_return', 'predicted_return', 'error']:
        if not pd.api.types.is_numeric_dtype(df[c]):
            issues.append(f'column {c} not numeric')

    # error consistency
    diff = np.abs(df['error'] - (df['predicted_return'] - df['actual_return']))
    bad_err = (diff > 1e-8).sum()
    if bad_err:
        issues.append(f'{bad_err} rows with inconsistent error values')

    return issues


def aggregate_metrics(df: pd.DataFrame):
    # compute per model,horizon metrics
    rows = []
    zero_mae = {}
    for (m, h), g in df.groupby(['model', 'horizon']):
        g2 = g.dropna(subset=['actual_return', 'predicted_return'])
        n = len(g2)
        mae = mean_absolute_error(g2['actual_return'], g2['predicted_return'])
        rmse = math.sqrt(mean_squared_error(g2['actual_return'], g2['predicted_return']))
        bias = (g2['predicted_return'] - g2['actual_return']).mean()
        # directional accuracy
        dir_acc = np.mean(np.sign(g2['predicted_return']) == np.sign(g2['actual_return']))
        rows.append({'model': m, 'horizon': h, 'N': n, 'MAE': mae, 'RMSE': rmse, 'Bias': bias, 'DirAcc': dir_acc})
        if m == 'zero':
            zero_mae[h] = mae

    out = pd.DataFrame(rows)
    # compute MASE relative to zero baseline
    def mase(row):
        z = zero_mae.get(row['horizon'], np.nan)
        return row['MAE'] / z if z and not np.isnan(z) else np.nan

    out['MASE'] = out.apply(mase, axis=1)

    # improvements vs zero
    def improv(row):
        z = zero_mae.get(row['horizon'], np.nan)
        if np.isnan(z):
            return pd.Series({'AbsImp': np.nan, 'PctImp': np.nan})
        absimp = z - row['MAE']
        pct = absimp / z * 100 if z else np.nan
        return pd.Series({'AbsImp': absimp, 'PctImp': pct})

    out[['AbsImp', 'PctImp']] = out.apply(improv, axis=1)
    return out


def time_stability(df: pd.DataFrame):
    df = df.dropna(subset=['actual_return', 'predicted_return']).copy()
    df['year'] = pd.to_datetime(df['forecast_timestamp']).dt.year
    annual = df.groupby(['model', 'horizon', 'year']).apply(
        lambda g: mean_absolute_error(g['actual_return'], g['predicted_return'])
    ).reset_index(name='MAE')

    # rolling MAE by forecast date
    rolls = {}
    for (m, h), g in df.groupby(['model', 'horizon']):
        g2 = g.sort_values('forecast_timestamp')
        g2 = g2.set_index(pd.to_datetime(g2['forecast_timestamp']))
        roll = g2['error'].abs().rolling('365D').mean()
        rolls[(m, h)] = roll
    return annual, rolls


def plots_and_save(df: pd.DataFrame, agg: pd.DataFrame, annual, rolls):
    # metric comparison plot (MAE)
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in agg['model'].unique():
        sub = agg[agg['model'] == m]
        ax.plot(sub['horizon'], sub['MAE'], marker='o', label=m)
    ax.set_xlabel('Horizon')
    ax.set_ylabel('MAE')
    ax.set_title('Model MAE by Horizon')
    ax.legend()
    fig.savefig(OUT_DIR / 'model_mae_by_horizon.png', dpi=150)
    plt.close(fig)

    # rolling MAE example for XGBoost and SARIMAX h=1
    for key, series in rolls.items():
        m, h = key
        series.plot(title=f'Rolling MAE (365D) {m} h{h}')
        plt.ylabel('Rolling MAE')
        plt.xlabel('forecast_timestamp')
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'rolling_mae_{m}_h{h}.png', dpi=150)
        plt.close()

    # annual MAE heatmap-like table plots
    for (m, h), grp in annual.groupby(['model', 'horizon']):
        fig, ax = plt.subplots(figsize=(6, 3))
        years = grp['year'].astype(str)
        ax.bar(years, grp['MAE'])
        ax.set_title(f'Annual MAE {m} h{h}')
        ax.set_ylabel('MAE')
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig.savefig(OUT_DIR / f'annual_mae_{m}_h{h}.png', dpi=150)
        plt.close()


def update_docs(agg: pd.DataFrame):
    md = []
    md.append('# Model comparison')
    md.append('\n## Aggregate results\n')
    md.append(agg.to_markdown(index=False))
    Path('docs/model_comparison.md').write_text('\n'.join(md))


def main():
    path = Path('data/processed/model_backtest_results.parquet')
    df = load_backtest(path)
    # accept alternate column name used in backtest outputs
    if 'error' not in df.columns and 'prediction_error' in df.columns:
        df = df.rename(columns={'prediction_error': 'error'})

    issues = validate_backtest(df)
    if issues:
        print('Validation issues found:')
        for it in issues:
            print('-', it)
    else:
        print('No validation issues found')

    agg = aggregate_metrics(df)
    agg.to_csv('docs/figures/aggregate_model_metrics.csv', index=False)
    annual, rolls = time_stability(df)
    plots_and_save(df, agg, annual, rolls)
    update_docs(agg)
    print('Report generated: docs/model_comparison.md and figures under docs/figures')


if __name__ == '__main__':
    main()
