from __future__ import annotations

from pathlib import Path

import pandas as pd

from copper_market_intelligence.forecasting.backtest import (
    expanding_walk_forward_with_meta,
)
from copper_market_intelligence.forecasting.models import (
    sarimax_model_factory,
    xgboost_model_factory,
)


def main(features_path: str = "data/processed/features_daily.parquet", out_path: str = "data/processed/model_backtest_results.parquet"):
    features = pd.read_parquet(features_path)

    models = {
        "sarimax": sarimax_model_factory(order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), exog=None),
        "xgboost": xgboost_model_factory(params=None, nrounds=50, seed=42),
    }

    df = expanding_walk_forward_with_meta(
        features,
        horizons=(1,5,20),
        models=models,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print('wrote', out_path)


if __name__ == '__main__':
    main()
