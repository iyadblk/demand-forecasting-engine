"""
Prophet forecasting wrapper with French holidays and CIs.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import warnings
import logging

# Silence Prophet/cmdstanpy chatter
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    PROPHET_OK = True
except Exception:  # pragma: no cover
    PROPHET_OK = False

try:
    import holidays as pyholidays
    HOLIDAYS_OK = True
except Exception:
    HOLIDAYS_OK = False


def _french_holidays(years: list[int]) -> pd.DataFrame:
    if not HOLIDAYS_OK:
        return pd.DataFrame(columns=["ds", "holiday"])
    fr = pyholidays.France(years=years)
    rows = [{"ds": pd.Timestamp(d), "holiday": name} for d, name in fr.items()]
    return pd.DataFrame(rows)


def fit_predict_prophet(
    series: pd.DataFrame,
    horizon: int = 30,
    interval_width: float = 0.80,
) -> pd.DataFrame:
    """
    series: DataFrame[date, demand]
    Returns DataFrame[date, yhat, yhat_lower_80, yhat_upper_80,
                     yhat_lower_95, yhat_upper_95, model]
    """
    if not PROPHET_OK:
        return _fallback(series, horizon)

    df = series.rename(columns={"date": "ds", "demand": "y"})[["ds", "y"]].copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = df["y"].astype(float)

    years = sorted(df["ds"].dt.year.unique().tolist())
    fy = years + [years[-1] + 1] if years else []
    hols = _french_holidays(fy) if fy else None

    m80 = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        interval_width=0.80,
        holidays=hols if hols is not None and not hols.empty else None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m80.fit(df)

    future = m80.make_future_dataframe(periods=horizon, freq="D")
    fc80 = m80.predict(future)[
        ["ds", "yhat", "yhat_lower", "yhat_upper", "trend", "weekly", "yearly"]
        if "weekly" in m80.predict(future).columns
        else ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ]

    # 95% CI — refit with wider interval
    m95 = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        interval_width=0.95,
        holidays=hols if hols is not None and not hols.empty else None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m95.fit(df)
    fc95 = m95.predict(future)[["ds", "yhat_lower", "yhat_upper"]]

    merged = fc80.merge(fc95, on="ds", suffixes=("_80", "_95"))
    merged["yhat"] = merged["yhat"].clip(lower=0)
    merged["yhat_lower_80"] = merged["yhat_lower_80"].clip(lower=0)
    merged["yhat_upper_80"] = merged["yhat_upper_80"].clip(lower=0)
    merged["yhat_lower_95"] = merged["yhat_lower_95"].clip(lower=0)
    merged["yhat_upper_95"] = merged["yhat_upper_95"].clip(lower=0)
    merged = merged.rename(columns={"ds": "date"})
    merged["model"] = "Prophet"

    out = merged.tail(horizon).reset_index(drop=True)
    return out


def prophet_components(series: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    """Return full prediction (history + future) WITH components, for seasonality charts."""
    if not PROPHET_OK:
        return pd.DataFrame()
    df = series.rename(columns={"date": "ds", "demand": "y"})[["ds", "y"]].copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = df["y"].astype(float)

    years = sorted(df["ds"].dt.year.unique().tolist())
    hols = _french_holidays(years + [years[-1] + 1]) if years else None

    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        holidays=hols if hols is not None and not hols.empty else None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(df)
    future = m.make_future_dataframe(periods=horizon, freq="D")
    fc = m.predict(future)
    return fc


def _fallback(series: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """If Prophet not installed, simple seasonal naive forecast."""
    s = series.copy().sort_values("date")
    s["dow"] = pd.to_datetime(s["date"]).dt.dayofweek
    dow_mean = s.groupby("dow")["demand"].mean()
    last_date = pd.to_datetime(s["date"].iloc[-1])
    dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    yhat = [dow_mean.get(d.weekday(), s["demand"].mean()) for d in dates]
    std = s["demand"].std()
    return pd.DataFrame({
        "date": dates,
        "yhat": yhat,
        "yhat_lower_80": np.maximum(0, np.array(yhat) - 1.28 * std),
        "yhat_upper_80": np.array(yhat) + 1.28 * std,
        "yhat_lower_95": np.maximum(0, np.array(yhat) - 1.96 * std),
        "yhat_upper_95": np.array(yhat) + 1.96 * std,
        "model": "ProphetFallback",
    })
