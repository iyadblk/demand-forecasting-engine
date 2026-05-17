"""
ARIMA / SARIMA forecasting with auto parameter selection by AIC.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from itertools import product

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SM_OK = True
except Exception:
    SM_OK = False

from core.preprocessor import fill_zeros_with_median


def _auto_select(y: pd.Series, seasonal_period: int = 7):
    best_aic = np.inf
    best_order = (1, 1, 1)
    best_seasonal = (0, 0, 0, 0)

    p_range = [0, 1, 2]
    d_range = [0, 1]
    q_range = [0, 1, 2]
    P_range = [0, 1]
    D_range = [0, 1]
    Q_range = [0, 1]

    for p, d, q in product(p_range, d_range, q_range):
        for P, D, Q in product(P_range, D_range, Q_range):
            try:
                model = SARIMAX(
                    y,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = model.fit(disp=False, maxiter=50)
                if res.aic < best_aic and np.isfinite(res.aic):
                    best_aic = res.aic
                    best_order = (p, d, q)
                    best_seasonal = (P, D, Q, seasonal_period)
            except Exception:
                continue
    return best_order, best_seasonal, best_aic


def fit_predict_arima(series: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    if not SM_OK:
        return _fallback(series, horizon)

    s = series.copy().sort_values("date").reset_index(drop=True)
    y = fill_zeros_with_median(s["demand"].astype(float))
    last_date = pd.to_datetime(s["date"].iloc[-1])

    # Smaller search on smaller series
    try:
        order, seasonal, aic = _auto_select(y, seasonal_period=7)
    except Exception:
        order, seasonal = (1, 1, 1), (0, 0, 0, 0)

    try:
        model = SARIMAX(
            y, order=order, seasonal_order=seasonal,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(disp=False, maxiter=100)
        fc = res.get_forecast(steps=horizon)
        mean = fc.predicted_mean
        ci80 = fc.conf_int(alpha=0.20)
        ci95 = fc.conf_int(alpha=0.05)
    except Exception:
        # Final fallback to ARIMA(1,1,1)
        model = SARIMAX(y, order=(1, 1, 1))
        res = model.fit(disp=False, maxiter=100)
        fc = res.get_forecast(steps=horizon)
        mean = fc.predicted_mean
        ci80 = fc.conf_int(alpha=0.20)
        ci95 = fc.conf_int(alpha=0.05)

    dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    out = pd.DataFrame({
        "date": dates,
        "yhat": np.clip(mean.values, 0, None),
        "yhat_lower_80": np.clip(ci80.iloc[:, 0].values, 0, None),
        "yhat_upper_80": np.clip(ci80.iloc[:, 1].values, 0, None),
        "yhat_lower_95": np.clip(ci95.iloc[:, 0].values, 0, None),
        "yhat_upper_95": np.clip(ci95.iloc[:, 1].values, 0, None),
        "model": "ARIMA",
    })
    out.attrs["order"] = order
    out.attrs["seasonal"] = seasonal
    return out


def _fallback(series: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """If statsmodels unavailable: weekly-mean naive."""
    s = series.copy()
    s["dow"] = pd.to_datetime(s["date"]).dt.dayofweek
    dow_mean = s.groupby("dow")["demand"].mean()
    last_date = pd.to_datetime(s["date"].iloc[-1])
    dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    yhat = np.array([dow_mean.get(d.weekday(), s["demand"].mean()) for d in dates])
    std = s["demand"].std()
    return pd.DataFrame({
        "date": dates,
        "yhat": yhat,
        "yhat_lower_80": np.maximum(0, yhat - 1.28 * std),
        "yhat_upper_80": yhat + 1.28 * std,
        "yhat_lower_95": np.maximum(0, yhat - 1.96 * std),
        "yhat_upper_95": yhat + 1.96 * std,
        "model": "ARIMAFallback",
    })
