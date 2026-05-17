"""
High-level orchestration: per-SKU full forecast pipeline + caching.
Used by the Streamlit app to keep app.py thin.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st

from core.preprocessor import get_sku_series, train_test_split
from core.moving_avg import simple_moving_average, weighted_moving_average
from core.prophet_model import fit_predict_prophet
from core.arima_model import fit_predict_arima
from core.ensemble import compute_weights, ensemble_forecast
from core.metrics import all_metrics, best_model


@st.cache_data(show_spinner=False, ttl=3600)
def forecast_all_models(
    _df_key: str,
    series_pickle: bytes,
    horizon: int,
) -> dict:
    """
    Cached per-SKU full forecast: MA, Prophet, ARIMA, Ensemble + metrics.
    We pickle the series to make the cache key stable.
    """
    import io
    series = pd.read_pickle(io.BytesIO(series_pickle))

    # Hold out last 30 days for metric calc
    test_days = 30
    if len(series) <= test_days + 30:
        test_days = max(7, len(series) // 5)
    train, test = train_test_split(series, test_days=test_days)

    fc_ma_test  = simple_moving_average(train, horizon=test_days)
    fc_wma_test = weighted_moving_average(train, horizon=test_days)
    fc_prophet_test = fit_predict_prophet(train, horizon=test_days)
    fc_arima_test = fit_predict_arima(train, horizon=test_days)

    metrics_rows = []
    for name, fc_t in [
        ("MovingAverage", fc_ma_test),
        ("WeightedMA",    fc_wma_test),
        ("Prophet",       fc_prophet_test),
        ("ARIMA",         fc_arima_test),
    ]:
        m = all_metrics(
            test["demand"].values,
            fc_t["yhat"].values,
            fc_t.get("yhat_lower_80", pd.Series([np.nan]*len(fc_t))).values,
            fc_t.get("yhat_upper_80", pd.Series([np.nan]*len(fc_t))).values,
        )
        m["Model"] = name
        metrics_rows.append(m)

    metrics_df = pd.DataFrame(metrics_rows)[["Model", "MAE", "RMSE", "MAPE", "Bias", "Coverage80"]]

    # Now ensemble weights from prophet/arima MAE
    mae_p = float(metrics_df.loc[metrics_df["Model"] == "Prophet", "MAE"].iloc[0])
    mae_a = float(metrics_df.loc[metrics_df["Model"] == "ARIMA",   "MAE"].iloc[0])
    w_p, w_a = compute_weights(mae_p, mae_a)

    fc_ens_test = ensemble_forecast(fc_prophet_test, fc_arima_test, w_p, w_a)
    m_ens = all_metrics(
        test["demand"].values,
        fc_ens_test["yhat"].values,
        fc_ens_test["yhat_lower_80"].values,
        fc_ens_test["yhat_upper_80"].values,
    )
    m_ens["Model"] = "Ensemble"
    metrics_df = pd.concat([metrics_df, pd.DataFrame([m_ens])[metrics_df.columns]], ignore_index=True)

    # Final future forecasts (full series, future horizon)
    fc_ma   = simple_moving_average(series, horizon=horizon)
    fc_wma  = weighted_moving_average(series, horizon=horizon)
    fc_prop = fit_predict_prophet(series, horizon=horizon)
    fc_arim = fit_predict_arima(series, horizon=horizon)
    fc_ens  = ensemble_forecast(fc_prop, fc_arim, w_p, w_a)

    best = best_model(metrics_df, primary="MAPE")

    return {
        "metrics":  metrics_df,
        "forecast": {
            "MovingAverage": fc_ma,
            "WeightedMA":    fc_wma,
            "Prophet":       fc_prop,
            "ARIMA":         fc_arim,
            "Ensemble":      fc_ens,
        },
        "test": {
            "MovingAverage": fc_ma_test,
            "WeightedMA":    fc_wma_test,
            "Prophet":       fc_prophet_test,
            "ARIMA":         fc_arima_test,
            "Ensemble":      fc_ens_test,
            "actual":        test,
        },
        "weights": {"Prophet": w_p, "ARIMA": w_a},
        "best":    best,
    }


def run_pipeline(df: pd.DataFrame, sku_id: str, horizon: int) -> dict:
    import io, pickle
    series = get_sku_series(df, sku_id)
    buf = io.BytesIO()
    series.to_pickle(buf)
    return forecast_all_models(
        _df_key=f"{sku_id}-{horizon}-{len(series)}",
        series_pickle=buf.getvalue(),
        horizon=horizon,
    )


@st.cache_data(show_spinner=False, ttl=3600)
def forecast_lite(_key: str, series_pickle: bytes, horizon: int) -> pd.DataFrame:
    """Prophet-only quick forecast for multi-SKU views."""
    import io
    series = pd.read_pickle(io.BytesIO(series_pickle))
    return fit_predict_prophet(series, horizon=horizon)


def run_lite(df: pd.DataFrame, sku_id: str, horizon: int) -> pd.DataFrame:
    import io
    series = get_sku_series(df, sku_id)
    buf = io.BytesIO()
    series.to_pickle(buf)
    return forecast_lite(_key=f"lite-{sku_id}-{horizon}", series_pickle=buf.getvalue(), horizon=horizon)
