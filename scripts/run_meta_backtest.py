from __future__ import annotations

from pathlib import Path

import pandas as pd

from copper_market_intelligence.forecasting.backtest import (
    expanding_mean_baseline,
    expanding_walk_forward_with_meta,
    momentum_baseline,
    zero_return_baseline,
)

MODEL_FUNCS = {
    "zero": zero_return_baseline,
    "mean": expanding_mean_baseline,
    "mom5": lambda df, h: momentum_baseline(df, h, lookback=5),
}


def main(features_path: str = "data/processed/features_daily.parquet", out_path: str = "data/processed/baseline_backtest_meta.parquet"):
    features = pd.read_parquet(features_path)
    # run meta backtest
    df = expanding_walk_forward_with_meta(
        features,
        horizons=(1, 5, 20),
        models=MODEL_FUNCS,
        embargo=None,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print('wrote', out_path)


if __name__ == '__main__':
    main()
