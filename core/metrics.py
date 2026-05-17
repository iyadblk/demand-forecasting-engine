"""
Forecast accuracy metrics — MAE, RMSE, MAPE, coverage, bias.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def bias(y_true, y_pred) -> float:
    """Mean error (positive = over-forecast)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(y_pred - y_true))


def coverage(y_true, y_lower, y_upper) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_lower = np.asarray(y_lower, dtype=float)
    y_upper = np.asarray(y_upper, dtype=float)
    inside = (y_true >= y_lower) & (y_true <= y_upper)
    return float(np.mean(inside) * 100.0)


def all_metrics(y_true, y_pred, y_lower=None, y_upper=None) -> dict:
    out = {
        "MAE":  mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "Bias": bias(y_true, y_pred),
    }
    if y_lower is not None and y_upper is not None:
        out["Coverage80"] = coverage(y_true, y_lower, y_upper)
    return out


def best_model(metric_table: pd.DataFrame, primary: str = "MAPE") -> str:
    """Pick best model by lowest MAPE (or fallback MAE)."""
    if metric_table.empty:
        return ""
    df = metric_table.dropna(subset=[primary]).copy()
    if df.empty:
        df = metric_table.dropna(subset=["MAE"]).copy()
        primary = "MAE"
    if df.empty:
        return ""
    return df.sort_values(primary).iloc[0]["Model"]
