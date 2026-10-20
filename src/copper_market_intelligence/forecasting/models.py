from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

MODEL_DIR = os.path.join("models")
os.makedirs(MODEL_DIR, exist_ok=True)


def _numeric_feature_columns(df: pd.DataFrame) -> list:
    exclude_prefixes = ("log_ret_", "fut_close_")
    cols = [c for c in df.select_dtypes(include=[np.number]).columns]
    cols = [c for c in cols if not any(c.startswith(p) for p in exclude_prefixes)]
    return cols


def sarimax_model_factory(
    order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), exog: Optional[list] = None
):
    def model(train_df: pd.DataFrame, h: int):
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except Exception:
            return None
        label = f"log_ret_{h}"
        y = train_df[label].dropna()
        if y.empty:
            return None
        exog_train = None
        if exog:
            exog_cols = [c for c in exog if c in train_df.columns]
            if exog_cols:
                exog_train = train_df.loc[y.index, exog_cols]
        try:
            m = SARIMAX(
                y,
                order=order,
                seasonal_order=seasonal_order,
                exog=exog_train,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = m.fit(disp=False)
            # one-step ahead forecast of the log return at forecast time
            fc = res.forecast(steps=1, exog=None)
            return float(fc.iloc[-1])
        except Exception:
            return None

    return model


def xgboost_model_factory(
    params: Optional[Dict] = None,
    nrounds: int = 100,
    seed: int = 0,
    save_model: bool = False,
    save_dir: str = MODEL_DIR,
):
    def model(train_df: pd.DataFrame, h: int):
        try:
            import xgboost as xgb
        except Exception:
            return None
        label = f"log_ret_{h}"
        df = train_df.copy()
        df = df.dropna(subset=[label])
        if df.empty:
            return None
        feats = _numeric_feature_columns(df)
        if not feats:
            return None
        X = df[feats]
        y = df[label]
        dtrain = xgb.DMatrix(X, label=y)
        p = params or {"objective": "reg:squarederror", "seed": seed}
        bst = xgb.train(p, dtrain, num_boost_round=nrounds)
        if save_model:
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"xgb_h{h}.model")
            bst.save_model(model_path)
        # predict using last row features
        last = X.iloc[[-1]]
        dp = xgb.DMatrix(last)
        pred = bst.predict(dp)
        return float(pred[0])

    return model
