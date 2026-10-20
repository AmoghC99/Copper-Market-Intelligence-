from __future__ import annotations

import dataclasses
import math
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd


@dataclasses.dataclass
class PredictionRecord:
    timestamp: pd.Timestamp
    target_horizon: int
    actual_return: float | None
    predicted_return: float | None
    prediction_error: float | None
    model_name: str


def expanding_walk_forward(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    start_index: int = 0,
    min_train_periods: int = 252,
    horizons: Iterable[int] = (1, 5, 20),
    retrain_frequency: int = 1,
    embargo: Optional[int] = None,
    models: Optional[Dict[str, Callable]] = None,
) -> pd.DataFrame:
    """
    Run an expanding-window walk-forward backtest on `df`.

    Parameters
    - df: DataFrame containing timestamp and required feature/target columns. Must
      include `log_ret_{h}` columns for each horizon `h`.
    - start_index: integer index in df to start producing forecasts (0-based).
    - min_train_periods: minimum number of historical rows required before
      starting to produce forecasts.
    - horizons: iterable of integer trading-day horizons to forecast.
    - retrain_frequency: produce forecasts every N days (1 => every day).
    - embargo: number of recent days to withhold from training to avoid overlap
      with the forecast window. If None, defaults to horizon (conservative).
        - models: dict of name->callable(train_df, horizon) returning predicted
            return for timestamp t.

    Returns a DataFrame of prediction records.
    """
    df = df.copy()
    df = df.reset_index(drop=True)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    if models is None:
        raise ValueError("At least one model must be provided in models")

    records: List[PredictionRecord] = []

    n = len(df)
    # run across rows where we can observe future target
    for idx in range(start_index, n):
        # produce forecasts only when we have enough training history
        if idx < min_train_periods:
            continue
        # retrain_frequency control
        if ((idx - start_index) % retrain_frequency) != 0:
            continue
        t = df.loc[idx, timestamp_col]
        for h in horizons:
            # determine embargo default
            emb = embargo if embargo is not None else h
            # training end index: last index available for training labels
            train_end_idx = idx - emb - 1
            if train_end_idx < 0:
                # not enough history after applying embargo
                continue
            # training rows must have labels observed: s + h <= train_end_idx
            # so max s is train_end_idx - h
            max_train_row = train_end_idx - h
            if max_train_row < 0:
                # no fully observed labels available
                continue
            train_df = df.loc[: max_train_row].copy()
            # ensure we only use rows with non-null label
            label_col = f"log_ret_{h}"
            if label_col not in train_df.columns:
                raise ValueError(f"Label column {label_col} not found in dataframe")
            train_df = train_df[train_df[label_col].notna()]

            # for prediction time idx, fetch actual if present
            actual = df.loc[idx, label_col] if label_col in df.columns else None

            for model_name, model_fn in models.items():
                try:
                    pred = model_fn(train_df, h)
                except Exception:
                    pred = None
                err = None
                if pred is None or pd.isna(pred) or actual is None or pd.isna(actual):
                    err = None
                else:
                    err = float(pred) - float(actual)
                records.append(
                    PredictionRecord(
                        timestamp=t,
                        target_horizon=h,
                        actual_return=float(actual) if not pd.isna(actual) else None,
                        predicted_return=(
                            float(pred)
                            if pred is not None and not pd.isna(pred)
                            else None
                        ),
                        prediction_error=err,
                        model_name=model_name,
                    )
                )
    # convert to DataFrame
    out = pd.DataFrame([dataclasses.asdict(r) for r in records])
    if not out.empty:
        out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out


def expanding_walk_forward_with_meta(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    start_index: int = 0,
    min_train_periods: int = 252,
    horizons: Iterable[int] = (1, 5, 20),
    retrain_frequency: int = 1,
    embargo: Optional[int] = None,
    models: Optional[Dict[str, Callable]] = None,
) -> pd.DataFrame:
    """
    Same as `expanding_walk_forward` but include audit metadata per prediction:
    - training_end_idx, training_end_timestamp
    - target_end_idx, target_end_timestamp

    Returns a DataFrame with one row per prediction and additional meta columns.
    """
    df = df.copy()
    df = df.reset_index(drop=True)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    if models is None:
        raise ValueError("At least one model must be provided in models")

    records = []
    n = len(df)
    for idx in range(start_index, n):
        if idx < min_train_periods:
            continue
        if ((idx - start_index) % retrain_frequency) != 0:
            continue
        t = df.loc[idx, timestamp_col]
        for h in horizons:
            emb = embargo if embargo is not None else h
            train_end_idx = idx - emb - 1
            if train_end_idx < 0:
                continue
            max_train_row = train_end_idx - h
            if max_train_row < 0:
                continue
            train_df = df.loc[: max_train_row].copy()
            label_col = f"log_ret_{h}"
            if label_col not in train_df.columns:
                raise ValueError(f"Label column {label_col} not found in dataframe")
            train_df = train_df[train_df[label_col].notna()]

            target_end_idx = idx + h
            if target_end_idx < n:
                target_end_ts = df.loc[target_end_idx, timestamp_col]
            else:
                target_end_ts = pd.NaT

            actual = df.loc[idx, label_col] if label_col in df.columns else None

            for model_name, model_fn in models.items():
                try:
                    pred = model_fn(train_df, h)
                except Exception:
                    pred = None
                err = None
                if pred is None or pd.isna(pred) or actual is None or pd.isna(actual):
                    err = None
                else:
                    err = float(pred) - float(actual)
                training_end_idx_val = (
                    int(train_df.index.max()) if not train_df.empty else None
                )
                target_end_idx_val = int(target_end_idx) if target_end_idx < n else None
                pred_val = (
                    float(pred) if pred is not None and not pd.isna(pred) else None
                )
                records.append(
                    {
                        "forecast_timestamp": t,
                        "training_end_idx": training_end_idx_val,
                        "training_end_timestamp": df.loc[train_end_idx, timestamp_col],
                        "target_end_idx": target_end_idx_val,
                        "target_end_timestamp": target_end_ts,
                        "horizon": h,
                        "model": model_name,
                        "actual_return": float(actual) if not pd.isna(actual) else None,
                        "predicted_return": pred_val,
                        "prediction_error": err,
                    }
                )
    out = pd.DataFrame(records)
    if not out.empty:
        out["forecast_timestamp"] = pd.to_datetime(out["forecast_timestamp"])
        out["training_end_timestamp"] = pd.to_datetime(out["training_end_timestamp"])
        out["target_end_timestamp"] = pd.to_datetime(out["target_end_timestamp"])
    return out


# Baseline models

def zero_return_baseline(train_df: pd.DataFrame, horizon: int) -> float:
    """Predict future return as zero (random-walk)."""
    return 0.0


def expanding_mean_baseline(train_df: pd.DataFrame, horizon: int) -> float:
    """Predict future return as the historical mean of `log_ret_{h}` in train_df."""
    col = f"log_ret_{horizon}"
    if col not in train_df.columns:
        # try to compute if available in parent df
        raise ValueError(f"Column {col} not in training data")
    vals = train_df[col].dropna()
    if len(vals) == 0:
        return 0.0
    return float(vals.mean())


def momentum_baseline(train_df: pd.DataFrame, horizon: int, lookback: int = 5) -> float:
    """Predict future return as the recent price return over `lookback` days.

    Use the most recent row in train_df to compute close/close_lag_{lookback}.
    """
    if train_df.empty:
        return 0.0
    # expect close and close_lag_{lookback} to exist or compute from indexed rows
    last = train_df.iloc[-1]
    if f"close_lag_{lookback}" in train_df.columns:
        cl_lag = last.get(f"close_lag_{lookback}")
        cl_now = last.get("close")
        if pd.isna(cl_lag) or pd.isna(cl_now):
            return 0.0
        return float(math.log(cl_now / cl_lag))
    # fallback: compute from index positions
    try:
        cl_now = float(last["close"])
        cl_prev = float(train_df.iloc[-1 - lookback]["close"])
        return float(math.log(cl_now / cl_prev))
    except Exception:
        return 0.0
