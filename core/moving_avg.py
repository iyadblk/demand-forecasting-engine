"""
Moving average baseline forecasters.
"""

from __future__ import annotations
import pandas as pd
import numpy as np


def simple_moving_average(series: pd.DataFrame, horizon: int = 30, window: int = 7) -> pd.DataFrame:
    """Forecast as flat last 7-day mean."""
    s = series.copy().sort_values("date")
    last_mean = s["demand"].tail(window).mean()
    last_date = pd.to_datetime(s["date"].iloc[-1])
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    std = s["demand"].tail(window * 2).std() if len(s) >= window * 2 else s["demand"].std()
    std = std if pd.notna(std) and std > 0 else last_mean * 0.15

    return pd.DataFrame({
        "date": future_dates,
        "yhat": last_mean,
        "yhat_lower_80": max(0, last_mean - 1.28 * std),
        "yhat_upper_80": last_mean + 1.28 * std,
        "yhat_lower_95": max(0, last_mean - 1.96 * std),
        "yhat_upper_95": last_mean + 1.96 * std,
        "model": "MovingAverage",
    })


def weighted_moving_average(series: pd.DataFrame, horizon: int = 30, window: int = 7) -> pd.DataFrame:
    """Linear-weighted last-7-days, more weight on recent."""
    s = series.copy().sort_values("date")
    vals = s["demand"].tail(window).values
    weights = np.arange(1, len(vals) + 1, dtype=float)
    weights /= weights.sum()
    last_mean = float(np.dot(vals, weights))
    last_date = pd.to_datetime(s["date"].iloc[-1])
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    std = s["demand"].tail(window * 2).std() if len(s) >= window * 2 else s["demand"].std()
    std = std if pd.notna(std) and std > 0 else last_mean * 0.15

    return pd.DataFrame({
        "date": future_dates,
        "yhat": last_mean,
        "yhat_lower_80": max(0, last_mean - 1.28 * std),
        "yhat_upper_80": last_mean + 1.28 * std,
        "yhat_lower_95": max(0, last_mean - 1.96 * std),
        "yhat_upper_95": last_mean + 1.96 * std,
        "model": "WeightedMA",
    })


def in_sample_ma(series: pd.DataFrame, window: int = 7) -> pd.Series:
    """In-sample 7-day MA for residual/metric calculation."""
    return series["demand"].rolling(window, min_periods=1).mean()
