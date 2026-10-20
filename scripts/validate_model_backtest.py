from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = Path('data/processed/model_backtest_results.parquet')
    df = pd.read_parquet(p)
    print('rows, cols:', df.shape)

    required = [
        'model', 'horizon', 'forecast_timestamp', 'training_end_timestamp',
        'target_end_timestamp', 'actual_return', 'predicted_return', 'error'
    ]
    present = df.columns.tolist()
    print('columns present:', present)
    missing = [c for c in required if c not in present]
    if missing:
        print('MISSING REQUIRED COLUMNS:', missing)
    else:
        print('All required columns present')

    # map known alternate names
    if 'error' not in df.columns and 'prediction_error' in df.columns:
        print("Note: 'prediction_error' column found (will use it as 'error' for checks).")
        df = df.rename(columns={'prediction_error': 'error'})

    issues = []

    # duplicates
    dup_mask = df.duplicated(subset=['model', 'horizon', 'forecast_timestamp'])
    dup_count = dup_mask.sum()
    print('duplicate forecast records for same model/horizon/timestamp:', int(dup_count))
    if dup_count:
        issues.append(f'duplicates: {int(dup_count)}')

    # chronological per model/horizon
    for (m, h), g in df.groupby(['model', 'horizon']):
        g2 = g.sort_values('forecast_timestamp')
        if not g2['forecast_timestamp'].is_monotonic_increasing:
            issues.append(f'forecast_timestamp not chronological for {m} h{h}')

        # training_end_timestamp <= forecast_timestamp
        bad = (g2['training_end_timestamp'] > g2['forecast_timestamp']).sum()
        if bad:
            issues.append(f'{bad} rows where training_end_timestamp > forecast for {m} h{h}')

        # target_end_timestamp > forecast_timestamp
        bad2 = (g2['target_end_timestamp'] <= g2['forecast_timestamp']).sum()
        if bad2:
            issues.append(f'{bad2} rows where target_end_timestamp <= forecast for {m} h{h}')

    # check idx relation if present
    if 'training_end_idx' in df.columns and 'target_end_idx' in df.columns:
        badidx = (df['target_end_idx'] <= df['training_end_idx']).sum()
        print('rows where target_end_idx <= training_end_idx:', int(badidx))
        if badidx:
            issues.append(f'{int(badidx)} rows where target_end_idx <= training_end_idx')

    # null predictions
    null_preds = df['predicted_return'].isna().sum()
    print('null predicted_return:', int(null_preds))
    if null_preds:
        issues.append(f'null predicted_return: {int(null_preds)}')

    # numeric checks
    for c in ['actual_return', 'predicted_return', 'error']:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            issues.append(f'column {c} not numeric')

    # error consistency
    if 'error' in df.columns:
        diff = np.abs(df['error'] - (df['predicted_return'] - df['actual_return']))
        bad_err = (diff > 1e-8).sum()
        print('rows with inconsistent error values:', int(bad_err))
        if bad_err:
            issues.append(f'{int(bad_err)} rows with inconsistent error')

    print('\nValidation summary:')
    if issues:
        for it in issues:
            print('-', it)
    else:
        print('No issues found')


if __name__ == '__main__':
    main()
