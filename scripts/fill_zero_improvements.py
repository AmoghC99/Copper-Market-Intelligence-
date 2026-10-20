from pathlib import Path

import pandas as pd


def main():
    p_base = Path('data/processed/baseline_backtest_results.parquet')
    p_agg = Path('docs/figures/aggregate_model_metrics.csv')
    base = pd.read_parquet(p_base)
    # baseline file uses 'model_name' and 'target_horizon' naming
    if 'model' in base.columns:
        model_col = 'model'
        horizon_col = 'horizon'
    else:
        model_col = 'model_name'
        horizon_col = 'target_horizon'
    zero = base[base[model_col] == 'zero']
    zero_mae = zero.groupby(horizon_col).apply(lambda g: (g['actual_return'] - g['predicted_return']).abs().mean()).to_dict()
    print('zero_mae:', zero_mae)

    agg = pd.read_csv(p_agg)

    def mase(row):
        z = zero_mae.get(row['horizon'])
        return row['MAE'] / z if z and pd.notna(z) else float('nan')

    def improv(row):
        z = zero_mae.get(row['horizon'])
        if z is None or z == 0:
            return pd.Series({'AbsImp': float('nan'), 'PctImp': float('nan')})
        absimp = z - row['MAE']
        pct = absimp / z * 100
        return pd.Series({'AbsImp': absimp, 'PctImp': pct})

    agg['MASE'] = agg.apply(mase, axis=1)
    imp = agg.apply(improv, axis=1)
    agg[['AbsImp', 'PctImp']] = imp
    agg.to_csv(p_agg, index=False)
    print('Updated', p_agg)


if __name__ == '__main__':
    main()
