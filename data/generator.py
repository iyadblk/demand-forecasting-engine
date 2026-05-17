"""
Synthetic historical demand generator — 2 years of daily data
for all 45 SKUs with realistic seasonality, weekly patterns,
holidays, promotions, stockouts and noise.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import List, Dict
import os

from data.catalog import ALL_SKUS, SKU


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
def _date_range(days: int = 730, end: date | None = None) -> pd.DatetimeIndex:
    end = end or date.today()
    start = end - timedelta(days=days - 1)
    return pd.date_range(start, end, freq="D")


# ---------------------------------------------------------------------------
# Seasonality multipliers
# ---------------------------------------------------------------------------
def _seasonal_multiplier(d: pd.Timestamp, sku: SKU) -> float:
    m, day = d.month, d.day
    mult = 1.0
    name = sku.name.lower()
    cat = sku.category.lower()

    # Christmas spike (Dec 15 -> Jan 5)
    if (m == 12 and day >= 15) or (m == 1 and day <= 5):
        if sku.zone == "AMBIENT":
            mult *= 2.5
        elif sku.zone == "HEAVY":
            mult *= 1.8
        elif sku.zone == "FRESH":
            mult *= 1.6
        else:
            mult *= 1.4

    # Summer (Jul-Aug)
    if m in (7, 8):
        if "glace" in cat or "ice" in name or "ben&jerry" in name:
            mult *= 1.6
        elif "soup" in cat or "soupe" in cat:
            mult *= 0.7
        elif sku.zone == "FROZEN" and ("légume" in cat or "lasagne" in name):
            mult *= 0.85
        if "eau" in cat or "evian" in name:
            mult *= 1.3
        if "sodas" in cat.lower() or "coca" in name or "orangina" in name:
            mult *= 1.25
        if "bière" in cat.lower() or "heineken" in name:
            mult *= 1.5

    # Back-to-school (September)
    if m == 9:
        if "céréales" in cat or "pâtes" in cat or "corn flake" in name:
            mult *= 1.4
        if "nutella" in name:
            mult *= 1.3

    # Valentine's day (Feb 11-17)
    if m == 2 and 11 <= day <= 17:
        if "nutella" in name or "haribo" in name or "tagada" in name:
            mult *= 1.6

    # Easter approximation (use mid April)
    if m == 4 and 1 <= day <= 18:
        if "nutella" in name or "haribo" in name or "ferrero" in sku.supplier.lower():
            mult *= 1.3

    # Apéritif / BBQ summer
    if m in (6, 7, 8) and ("apéritif" in cat or "alcool" in cat or "bière" in cat):
        mult *= 1.4

    return mult


def _weekly_multiplier(d: pd.Timestamp) -> float:
    dow = d.weekday()  # Mon=0 .. Sun=6
    return {
        0: 0.80,  # Mon
        1: 0.95,  # Tue
        2: 1.00,  # Wed
        3: 1.05,  # Thu
        4: 1.40,  # Fri
        5: 1.60,  # Sat
        6: 0.60,  # Sun
    }[dow]


def _trend_multiplier(i: int, total: int) -> float:
    # gentle upward trend +6% over 2 years
    return 1.0 + 0.06 * (i / max(total - 1, 1))


# ---------------------------------------------------------------------------
# Anomalies
# ---------------------------------------------------------------------------
def _inject_anomalies(
    series: np.ndarray,
    dates: pd.DatetimeIndex,
    sku: SKU,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict]]:
    n = len(series)
    events: list[dict] = []

    # Stockouts: 3-5 per year => 6-10 over 2 years
    n_stockouts = int(rng.integers(6, 11))
    for _ in range(n_stockouts):
        start = int(rng.integers(7, n - 6))
        length = int(rng.integers(1, 4))
        series[start : start + length] = 0
        events.append({
            "sku_id": sku.sku_id,
            "date": dates[start].date(),
            "type": "stockout",
            "length_days": length,
        })
        # Delivery surge: post-stockout
        end = start + length
        if end + 2 < n:
            series[end : end + 2] *= 1.5
            events.append({
                "sku_id": sku.sku_id,
                "date": dates[end].date(),
                "type": "delivery_surge",
                "length_days": 2,
            })

    # Promotional spikes: 3-4 per year => 6-8 over 2 years
    n_promos = int(rng.integers(6, 9))
    for _ in range(n_promos):
        start = int(rng.integers(0, n - 4))
        length = 3
        series[start : start + length] *= 2.0
        events.append({
            "sku_id": sku.sku_id,
            "date": dates[start].date(),
            "type": "promo_spike",
            "length_days": length,
        })

    return series, events


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------
def generate_sku_series(sku: SKU, days: int = 730, seed: int = 42) -> tuple[pd.DataFrame, list[dict]]:
    rng = np.random.default_rng(seed + hash(sku.sku_id) % 10000)
    dates = _date_range(days)
    base = sku.base_demand

    arr = np.zeros(len(dates), dtype=float)
    for i, d in enumerate(dates):
        m = base
        m *= _seasonal_multiplier(d, sku)
        m *= _weekly_multiplier(d)
        m *= _trend_multiplier(i, len(dates))
        # ±15% gaussian noise
        m *= max(0.0, 1.0 + rng.normal(0, 0.15))
        arr[i] = m

    arr, events = _inject_anomalies(arr, dates, sku, rng)
    arr = np.maximum(arr, 0)
    arr = np.round(arr).astype(int)

    df = pd.DataFrame({
        "date": dates,
        "sku_id": sku.sku_id,
        "sku_name": sku.name,
        "zone": sku.zone,
        "demand": arr,
    })
    return df, events


def generate_all(days: int = 730, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    all_events: list[dict] = []
    for sku in ALL_SKUS:
        df, events = generate_sku_series(sku, days=days, seed=seed)
        frames.append(df)
        all_events.extend(events)
    full = pd.concat(frames, ignore_index=True)
    events_df = pd.DataFrame(all_events)
    return full, events_df


def load_or_generate(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cache the dataset to data/sample_data.csv (and events)."""
    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(here, "sample_data.csv")
    events_path = os.path.join(here, "events.csv")

    if not force and os.path.exists(csv_path) and os.path.exists(events_path):
        df = pd.read_csv(csv_path, parse_dates=["date"])
        events = pd.read_csv(events_path, parse_dates=["date"])
        return df, events

    df, events = generate_all()
    df.to_csv(csv_path, index=False)
    events.to_csv(events_path, index=False)
    return df, events


if __name__ == "__main__":
    df, ev = generate_all()
    print(df.head())
    print(f"Rows: {len(df):,}  SKUs: {df['sku_id'].nunique()}")
    print(f"Events: {len(ev):,}")
