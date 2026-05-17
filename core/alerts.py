"""
Stock alert system — compute alert levels, days-until-stockout, recommended order.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Iterable

from data.catalog import SKU, get_sku


ALERT_COLORS = {
    "CRITICAL": "#FF4B4B",
    "WARNING":  "#FFB347",
    "WATCH":    "#FFD700",
    "OK":       "#00C896",
}


def safety_stock(avg_daily_demand: float, lead_time_days: int, factor: float = 1.5) -> float:
    return factor * avg_daily_demand * lead_time_days


def classify_alert(current_stock: float, safety: float) -> str:
    if current_stock < 0.5 * safety:
        return "CRITICAL"
    if current_stock < safety:
        return "WARNING"
    if current_stock < 1.5 * safety:
        return "WATCH"
    return "OK"


def days_until_stockout(current_stock: float, forecast: pd.DataFrame) -> int:
    """Cumulative-sum the forecast until it exceeds stock."""
    cum = forecast["yhat"].cumsum().values
    above = np.where(cum >= current_stock)[0]
    if len(above) == 0:
        return len(forecast)  # > horizon
    return int(above[0]) + 1


def recommended_order_qty(forecast: pd.DataFrame, current_stock: float, safety: float) -> float:
    forecast_total = float(forecast["yhat"].sum())
    qty = forecast_total + safety - current_stock
    return float(max(0, qty))


def estimated_stockout_cost(
    sku: SKU, forecast: pd.DataFrame, days_until: int, horizon: int
) -> float:
    """Lost margin from day of stockout to end of horizon."""
    if days_until >= horizon:
        return 0.0
    lost_units = float(forecast["yhat"].iloc[days_until:].sum())
    return lost_units * sku.margin


def build_alerts_table(
    forecasts: dict[str, pd.DataFrame],
    stocks: dict[str, float],
    lead_time: int = 3,
    horizon: int = 30,
) -> pd.DataFrame:
    """
    forecasts: dict sku_id -> DataFrame with 'date','yhat'
    stocks:    dict sku_id -> current_stock units
    Returns a per-SKU alert dataframe.
    """
    today = date.today()
    rows = []
    for sku_id, fc in forecasts.items():
        sku = get_sku(sku_id)
        if sku is None or fc is None or fc.empty:
            continue
        avg = float(fc["yhat"].head(7).mean())
        safety = safety_stock(avg, lead_time)
        stock = float(stocks.get(sku_id, 0.0))
        level = classify_alert(stock, safety)
        days = days_until_stockout(stock, fc)
        reorder_qty = recommended_order_qty(fc, stock, safety)
        order_date = today + timedelta(days=max(0, days - lead_time))
        cost = estimated_stockout_cost(sku, fc, days, horizon)

        rows.append({
            "sku_id": sku.sku_id,
            "name": sku.name,
            "zone": sku.zone,
            "supplier": sku.supplier,
            "current_stock": int(stock),
            "avg_daily_demand": round(avg, 1),
            "safety_stock": int(safety),
            "days_until_stockout": days,
            "alert_level": level,
            "reorder_qty": int(np.ceil(reorder_qty)),
            "order_by_date": order_date.isoformat(),
            "unit_cost": sku.unit_cost,
            "unit_margin": sku.margin,
            "estimated_stockout_cost": round(cost, 2),
            "order_total_cost": round(reorder_qty * sku.unit_cost, 2),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    order = {"CRITICAL": 0, "WARNING": 1, "WATCH": 2, "OK": 3}
    df["_o"] = df["alert_level"].map(order)
    df = df.sort_values(["_o", "days_until_stockout"]).drop(columns="_o")
    return df.reset_index(drop=True)
