import pandas as pd


def test_timestamps_and_cutoffs():
    p = 'data/processed/model_backtest_results.parquet'
    df = pd.read_parquet(p)
    # accept prediction_error alias
    if 'error' not in df.columns and 'prediction_error' in df.columns:
        df = df.rename(columns={'prediction_error': 'error'})

    # no null preds
    assert df['predicted_return'].notna().all()

    # numeric
    for c in ['actual_return', 'predicted_return', 'error']:
        assert pd.api.types.is_numeric_dtype(df[c])

    # per-row relations
    bad_training_after_forecast = (
        df["training_end_timestamp"] > df["forecast_timestamp"]
    ).sum()
    assert bad_training_after_forecast == 0

    bad_target_before_forecast = (
        df["target_end_timestamp"] <= df["forecast_timestamp"]
    ).sum()
    assert bad_target_before_forecast == 0


def test_no_duplicate_forecasts():
    p = 'data/processed/model_backtest_results.parquet'
    df = pd.read_parquet(p)
    dup = df.duplicated(subset=['model', 'horizon', 'forecast_timestamp']).sum()
    assert dup == 0


def test_reproducible_structure():
    p = 'data/processed/model_backtest_results.parquet'
    df = pd.read_parquet(p)
    cols = set(df.columns)
    required = {
        "model",
        "horizon",
        "forecast_timestamp",
        "training_end_timestamp",
        "target_end_timestamp",
        "actual_return",
        "predicted_return",
    }
    assert required.issubset(cols)
