"""
Weighted ensemble of Prophet + ARIMA, dynamic weights based on recent MAE.
"""

from __future__ import annotations
import pandas as pd
import numpy as np


def compute_weights(mae_prophet: float, mae_arima: float) -> tuple[float, float]:
    """Inverse-error weighting: better model => higher weight. Sum = 1."""
    eps = 1e-6
    inv_p = 1.0 / (mae_prophet + eps)
    inv_a = 1.0 / (mae_arima + eps)
    w_p = inv_p / (inv_p + inv_a)
    w_a = 1.0 - w_p
    return float(w_p), float(w_a)


def ensemble_forecast(
    fc_prophet: pd.DataFrame,
    fc_arima: pd.DataFrame,
    w_prophet: float,
    w_arima: float,
) -> pd.DataFrame:
    p = fc_prophet[["date", "yhat", "yhat_lower_80", "yhat_upper_80",
                    "yhat_lower_95", "yhat_upper_95"]].copy()
    a = fc_arima[["date", "yhat", "yhat_lower_80", "yhat_upper_80",
                  "yhat_lower_95", "yhat_upper_95"]].copy()
    m = p.merge(a, on="date", suffixes=("_p", "_a"))

    out = pd.DataFrame({"date": m["date"]})
    out["yhat"]            = w_prophet * m["yhat_p"]            + w_arima * m["yhat_a"]
    out["yhat_lower_80"]   = w_prophet * m["yhat_lower_80_p"]   + w_arima * m["yhat_lower_80_a"]
    out["yhat_upper_80"]   = w_prophet * m["yhat_upper_80_p"]   + w_arima * m["yhat_upper_80_a"]
    out["yhat_lower_95"]   = w_prophet * m["yhat_lower_95_p"]   + w_arima * m["yhat_lower_95_a"]
    out["yhat_upper_95"]   = w_prophet * m["yhat_upper_95_p"]   + w_arima * m["yhat_upper_95_a"]
    out["model"] = "Ensemble"
    out.attrs["w_prophet"] = w_prophet
    out.attrs["w_arima"] = w_arima
    return out
