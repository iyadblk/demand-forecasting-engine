"""
Data preprocessing utilities.
- Per-SKU series extraction
- Train/test split
- Feature engineering (lag, rolling, calendar)
- Outlier handling for ARIMA stability
"""

from __future__ import annotations
import pandas as pd
import numpy as np


def get_sku_series(df: pd.DataFrame, sku_id: str) -> pd.DataFrame:
    s = (
        df[df["sku_id"] == sku_id]
        .sort_values("date")
        .reset_index(drop=True)
        .copy()
    )
    s["date"] = pd.to_datetime(s["date"])
    return s[["date", "demand"]]


def train_test_split(series: pd.DataFrame, test_days: int = 30):
    cutoff = len(series) - test_days
    return series.iloc[:cutoff].copy(), series.iloc[cutoff:].copy()


def add_calendar_features(s: pd.DataFrame) -> pd.DataFrame:
    out = s.copy()
    out["dow"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["day"] = out["date"].dt.day
    out["is_weekend"] = out["dow"].isin([5, 6]).astype(int)
    out["week"] = out["date"].dt.isocalendar().week.astype(int)
    return out


def cap_outliers(y: pd.Series, lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.Series:
    lo, hi = y.quantile([lower_pct, upper_pct])
    return y.clip(lo, hi)


def fill_zeros_with_median(y: pd.Series, window: int = 14) -> pd.Series:
    """Replace stockout zeros with rolling median for model stability."""
    rolling = y.rolling(window, min_periods=3, center=True).median()
    out = y.where(y > 0, rolling)
    return out.fillna(y.median()).fillna(0)
