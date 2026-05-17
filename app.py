"""
Demand Forecasting Engine — Streamlit application.
Project 3 in the warehouse-logistics series by Iyad Belkadi.
"""

from __future__ import annotations
import os
import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data.catalog import ALL_SKUS, FROZEN, FRESH, AMBIENT, HEAVY, ZONE_COLORS, get_sku
from data.generator import load_or_generate
from core.orchestrator import run_pipeline, run_lite
from core.alerts import build_alerts_table, ALERT_COLORS
from core.recommender import build_reorder_list, group_by_supplier, export_excel, export_pdf
from core.prophet_model import prophet_components

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Demand Forecasting Engine | Iyad Belkadi",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject CSS
_css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(_css_path):
    with open(_css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

COLORS = {
    "history":  "#4B9FFF",
    "prophet":  "#00C896",
    "arima":    "#FFB347",
    "ma":       "#BF7FFF",
    "ensemble": "#FF6B6B",
    "ci":       "rgba(0,200,150,0.18)",
    "critical": "#FF4B4B",
    "warning":  "#FFB347",
    "watch":    "#FFD700",
    "ok":       "#00C896",
}

MODEL_COLOR = {
    "MovingAverage": COLORS["ma"],
    "WeightedMA":    COLORS["ma"],
    "Prophet":       COLORS["prophet"],
    "ARIMA":         COLORS["arima"],
    "Ensemble":      COLORS["ensemble"],
}

# ---------------------------------------------------------------------------
# Data load
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Generating 2 years of demand for all 45 SKUs…", ttl=24*3600)
def _bootstrap_data():
    return load_or_generate()


DEMAND_DF, EVENTS_DF = _bootstrap_data()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def kpi(label: str, value: str, sub: str = "", flavor: str = "blue") -> str:
    return f"""
    <div class="kpi-card kpi-{flavor}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>
    """


def alert_pill(level: str) -> str:
    cls = {"CRITICAL":"pill-critical","WARNING":"pill-warning","WATCH":"pill-watch","OK":"pill-ok"}.get(level,"pill-ok")
    return f'<span class="pill {cls}">{level}</span>'


def default_stock_for(sku_id: str, avg_demand: float) -> int:
    """Reasonable starting stock for each SKU (seeded but spread across alert levels)."""
    rng = np.random.default_rng(hash(sku_id) % 99991)
    factor = rng.uniform(0.4, 4.0)
    return int(max(0, avg_demand * factor * 3))  # ~3 days of stock × factor


@st.cache_data(ttl=24*3600)
def _avg_demand_per_sku() -> dict[str, float]:
    return DEMAND_DF.groupby("sku_id")["demand"].tail(30).groupby(
        DEMAND_DF["sku_id"]
    ).mean().to_dict()


def historical_slice(sku_id: str, last_days: int = 90) -> pd.DataFrame:
    s = DEMAND_DF[DEMAND_DF["sku_id"] == sku_id].sort_values("date").tail(last_days)
    return s


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 📦 Demand Forecasting")
st.sidebar.caption("Project 3 — Warehouse Logistics Series")

zone_filter = st.sidebar.selectbox(
    "Zone filter", ["ALL", "FROZEN", "FRESH", "AMBIENT", "HEAVY"], index=0
)

# SKU list filtered by zone
if zone_filter == "ALL":
    sku_pool = ALL_SKUS
else:
    sku_pool = [s for s in ALL_SKUS if s.zone == zone_filter]

# Auto-default: top 10 by recent volume
@st.cache_data(ttl=24*3600)
def _top_volume_skus(n: int = 10) -> list[str]:
    g = DEMAND_DF.tail(45 * 30).groupby("sku_id")["demand"].sum()
    return g.sort_values(ascending=False).head(n).index.tolist()


TOP10 = _top_volume_skus(10)
DEFAULT_SKU = TOP10[0] if TOP10 else sku_pool[0].sku_id

sku_options = [s.sku_id for s in sku_pool]
sku_labels = {s.sku_id: f"{s.sku_id} — {s.name}" for s in sku_pool}

primary_sku = st.sidebar.selectbox(
    "Primary SKU",
    sku_options,
    index=sku_options.index(DEFAULT_SKU) if DEFAULT_SKU in sku_options else 0,
    format_func=lambda x: sku_labels.get(x, x),
)

multi_skus = st.sidebar.multiselect(
    "Multi-SKU selection (max 10)",
    sku_options,
    default=[s for s in TOP10 if s in sku_options][:6],
    format_func=lambda x: sku_labels.get(x, x),
    max_selections=10,
)

st.sidebar.markdown("---")

horizon = st.sidebar.select_slider(
    "Forecast horizon (days)", options=[7, 14, 30, 60, 90], value=30
)

model_choice = st.sidebar.selectbox(
    "Primary model",
    ["Ensemble", "Prophet", "ARIMA", "WeightedMA", "MovingAverage"],
    index=0,
)

ci_pct = st.sidebar.radio("Confidence interval", ["80%", "95%"], index=0, horizontal=True)
show_anomalies = st.sidebar.toggle("Show anomalies", value=True)
show_promotions = st.sidebar.toggle("Show promotions", value=True)

st.sidebar.markdown("---")
lead_time = st.sidebar.number_input("Lead time (days)", min_value=1, max_value=30, value=3)

st.sidebar.markdown("**Stock levels**")
avgs = _avg_demand_per_sku()

# Build per-SKU stock map (default for ALL SKUs, editable for primary)
if "stocks" not in st.session_state:
    st.session_state.stocks = {
        sku.sku_id: default_stock_for(sku.sku_id, avgs.get(sku.sku_id, sku.base_demand))
        for sku in ALL_SKUS
    }

st.session_state.stocks[primary_sku] = st.sidebar.number_input(
    f"Stock — {primary_sku}",
    min_value=0,
    value=int(st.session_state.stocks.get(primary_sku, 100)),
    step=10,
    key=f"stock_{primary_sku}",
)

with st.sidebar.expander("Edit other stock levels"):
    for sid in multi_skus:
        if sid == primary_sku:
            continue
        st.session_state.stocks[sid] = st.number_input(
            f"Stock — {sid}",
            min_value=0,
            value=int(st.session_state.stocks.get(sid, 100)),
            step=10,
            key=f"stock_edit_{sid}",
        )

st.sidebar.markdown("---")
st.sidebar.caption("**Iyad Belkadi** · Warehouse Logistics Suite")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "<h1>📦 Demand Forecasting Engine</h1>"
    "<p style='color:#9aa0a6;margin-top:-12px'>"
    "Prophet · ARIMA · Ensemble — 45 SKUs · 2 years of history · French holidays"
    "</p>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Run pipeline for primary SKU
# ---------------------------------------------------------------------------
with st.spinner(f"Training models for {primary_sku}…"):
    PRIMARY = run_pipeline(DEMAND_DF, primary_sku, horizon)


# ---------------------------------------------------------------------------
# Run lite forecasts for ALL SKUs (used by Tab 1 alert dashboard)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Generating quick forecasts for alert dashboard…", ttl=3600)
def _all_lite_forecasts(horizon: int) -> dict[str, pd.DataFrame]:
    out = {}
    # Use Prophet-lite for all 45 SKUs
    for sku in ALL_SKUS:
        try:
            out[sku.sku_id] = run_lite(DEMAND_DF, sku.sku_id, horizon)
        except Exception:
            # Fallback: 7-day mean
            from core.moving_avg import simple_moving_average
            from core.preprocessor import get_sku_series
            out[sku.sku_id] = simple_moving_average(get_sku_series(DEMAND_DF, sku.sku_id), horizon=horizon)
    return out


ALL_FORECASTS = _all_lite_forecasts(horizon)

ALERTS_DF = build_alerts_table(
    ALL_FORECASTS,
    st.session_state.stocks,
    lead_time=lead_time,
    horizon=horizon,
)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
TABS = st.tabs([
    "🏠 Overview",
    "📈 Forecast",
    "🗂 Multi-SKU",
    "⚖️ Model Comparison",
    "🔁 Seasonality",
    "🚨 Alert Center",
    "🛒 Reorder",
    "💭 What-if",
    "⚠️ Anomalies",
    "🔥 Heatmap",
    "📊 Performance",
])


# ============================================================================
# TAB 1 — OVERVIEW
# ============================================================================
with TABS[0]:
    n_crit = int((ALERTS_DF["alert_level"] == "CRITICAL").sum())
    n_warn = int((ALERTS_DF["alert_level"] == "WARNING").sum())
    n_at_risk = n_crit + n_warn
    total_demand = float(sum(f["yhat"].sum() for f in ALL_FORECASTS.values()))
    stockout_cost = float(ALERTS_DF["estimated_stockout_cost"].sum())
    # Average MAPE for primary SKU
    avg_mape = PRIMARY["metrics"]["MAPE"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi(
            "SKUs at risk", f"{n_at_risk}",
            f"{n_crit} critical · {n_warn} warning", "red" if n_at_risk else "green",
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi(
            f"Total {horizon}-day forecast",
            f"{total_demand:,.0f} u",
            "Across 45 SKUs", "blue",
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi(
            "Avg MAPE (primary)",
            f"{avg_mape:.1f}%",
            f"on {primary_sku}", "orange",
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi(
            "Est. stockout cost",
            f"€ {stockout_cost:,.0f}",
            "Lost margin if no reorder", "red" if stockout_cost > 0 else "green",
        ), unsafe_allow_html=True)

    st.markdown("### 🔝 Top 10 most urgent SKUs")
    top10 = ALERTS_DF.head(10)
    if top10.empty:
        st.info("No alerts.")
    else:
        disp = top10[[
            "sku_id", "name", "zone", "current_stock", "avg_daily_demand",
            "days_until_stockout", "alert_level", "reorder_qty", "order_by_date",
            "estimated_stockout_cost",
        ]].rename(columns={
            "sku_id":"SKU","name":"Product","zone":"Zone","current_stock":"Stock",
            "avg_daily_demand":"Avg/day","days_until_stockout":"Days→Stockout",
            "alert_level":"Alert","reorder_qty":"Reorder","order_by_date":"Order By",
            "estimated_stockout_cost":"Cost €",
        })
        st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown("### 🗺️ Zone health heatmap")
    zone_health = ALERTS_DF.groupby(["zone", "alert_level"]).size().unstack(fill_value=0)
    for col in ["CRITICAL", "WARNING", "WATCH", "OK"]:
        if col not in zone_health.columns:
            zone_health[col] = 0
    zone_health = zone_health[["CRITICAL", "WARNING", "WATCH", "OK"]]

    fig = go.Figure(data=go.Heatmap(
        z=zone_health.values,
        x=zone_health.columns,
        y=zone_health.index,
        text=zone_health.values,
        texttemplate="%{text}",
        colorscale=[[0, "#0b0b0b"], [0.3, "#FFD700"], [0.7, "#FFB347"], [1, "#FF4B4B"]],
        showscale=False,
    ))
    fig.update_layout(
        height=260, template="plotly_dark",
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="#0b0b0b", paper_bgcolor="#0b0b0b",
    )
    st.plotly_chart(fig, use_container_width=True, key="ov_zone_heatmap")


# ============================================================================
# TAB 2 — FORECAST (single SKU, deep dive)
# ============================================================================
with TABS[1]:
    sku = get_sku(primary_sku)
    st.markdown(f"### 📈 Forecast — {sku.name} `({sku.sku_id})` · zone {sku.zone}")

    fc = PRIMARY["forecast"][model_choice]
    hist = historical_slice(primary_sku, 90)
    ci_low_col = "yhat_lower_80" if ci_pct == "80%" else "yhat_lower_95"
    ci_up_col  = "yhat_upper_80" if ci_pct == "80%" else "yhat_upper_95"

    fig = go.Figure()

    # CI band
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc[ci_up_col], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc[ci_low_col], mode="lines",
        line=dict(width=0), fill="tonexty", fillcolor=COLORS["ci"],
        name=f"CI {ci_pct}", hoverinfo="skip",
    ))

    # Historical
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["demand"], mode="lines",
        line=dict(color=COLORS["history"], width=2),
        name="Historical",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["yhat"], mode="lines",
        line=dict(color=MODEL_COLOR.get(model_choice, COLORS["prophet"]), width=3),
        name=f"{model_choice} forecast",
    ))

    # Anomalies/promotions markers
    if show_anomalies or show_promotions:
        ev = EVENTS_DF[EVENTS_DF["sku_id"] == primary_sku].copy()
        ev["date"] = pd.to_datetime(ev["date"])
        ev = ev[ev["date"] >= hist["date"].min()]
        if show_anomalies:
            so = ev[ev["type"] == "stockout"]
            if not so.empty:
                fig.add_trace(go.Scatter(
                    x=so["date"], y=[0]*len(so), mode="markers",
                    marker=dict(color="#FF4B4B", size=11, symbol="x"),
                    name="Stockout",
                ))
        if show_promotions:
            # Try event_type column first; fallback to demand>1.8×rolling mean proxy
            if not ev.empty and "type" in ev.columns:
                pr = ev[ev["type"] == "promo_spike"].copy()
            else:
                proxy = hist.copy()
                proxy["roll7"] = proxy["demand"].rolling(7, min_periods=1).mean()
                proxy = proxy[proxy["demand"] > 1.8 * proxy["roll7"]]
                pr = pd.DataFrame({
                    "date": proxy["date"].values,
                    "length_days": [1] * len(proxy),
                })

            if not pr.empty:
                # Shaded vertical regions
                for idx, row in pr.reset_index(drop=True).iterrows():
                    length = int(row.get("length_days", 3) or 3)
                    x0_str = str(pd.Timestamp(row["date"]).date())
                    x1_str = str((pd.Timestamp(row["date"]) + pd.Timedelta(days=length)).date())
                    fig.add_vrect(
                        x0=x0_str,
                        x1=x1_str,
                        fillcolor="#FFB347", opacity=0.3,
                        layer="below", line_width=0,
                        annotation_text="Promo" if idx == 0 else "",
                        annotation_position="top left",
                        annotation_font_color="#FFB347",
                    )
                # Legend stub (vrects don't show in legend)
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(color="#FFB347", size=10, symbol="square"),
                    name="Promotion event",
                ))
                # Keep original star highlights too
                ymap = hist.set_index("date")["demand"].to_dict()
                ys = [ymap.get(d, hist["demand"].mean()) for d in pr["date"]]
                fig.add_trace(go.Scatter(
                    x=pr["date"], y=ys, mode="markers",
                    marker=dict(color="#FFD700", size=12, symbol="star"),
                    name="Promo peak",
                ))

    fig.update_layout(
        template="plotly_dark", height=480,
        plot_bgcolor="#0b0b0b", paper_bgcolor="#0b0b0b",
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig, use_container_width=True, key="forecast_promo_chart")

    # 4 mini charts: model comparison
    st.markdown("### 🔍 Models side-by-side")
    cols = st.columns(4)
    for i, name in enumerate(["MovingAverage", "Prophet", "ARIMA", "Ensemble"]):
        f = PRIMARY["forecast"][name]
        with cols[i]:
            mini = go.Figure()
            mini.add_trace(go.Scatter(
                x=hist["date"].tail(45), y=hist["demand"].tail(45),
                mode="lines", line=dict(color=COLORS["history"], width=1),
                showlegend=False,
            ))
            mini.add_trace(go.Scatter(
                x=f["date"], y=f["yhat"], mode="lines",
                line=dict(color=MODEL_COLOR.get(name), width=2),
                showlegend=False,
            ))
            mini.update_layout(
                template="plotly_dark", height=170,
                margin=dict(l=10, r=10, t=30, b=10),
                title=dict(text=name, font=dict(size=12, color=MODEL_COLOR.get(name))),
                plot_bgcolor="#0b0b0b", paper_bgcolor="#0b0b0b",
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
            )
            st.plotly_chart(mini, use_container_width=True, key=f"fc_mini_{name}")

    st.markdown("### 📊 Metrics by model")
    st.dataframe(
        PRIMARY["metrics"].style.format({
            "MAE":"{:.2f}","RMSE":"{:.2f}","MAPE":"{:.1f}%","Bias":"{:+.2f}","Coverage80":"{:.1f}%"
        }),
        use_container_width=True, hide_index=True,
    )

    best = PRIMARY["best"]
    w = PRIMARY["weights"]
    st.markdown(
        f'<div class="badge-best">🏆 Best model for {primary_sku}: {best}</div>'
        f'<p style="margin-top:8px;color:#9aa0a6">Ensemble weights → '
        f'Prophet {w["Prophet"]*100:.0f}% · ARIMA {w["ARIMA"]*100:.0f}% '
        f'(inverse-MAE weighting)</p>',
        unsafe_allow_html=True,
    )


# ============================================================================
# TAB 3 — MULTI-SKU
# ============================================================================
with TABS[2]:
    st.markdown("### 🗂 Multi-SKU forecast grid (Prophet)")
    if not multi_skus:
        st.info("Select SKUs in the sidebar.")
    else:
        for i in range(0, len(multi_skus), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j >= len(multi_skus):
                    break
                sid = multi_skus[i + j]
                sku = get_sku(sid)
                with col:
                    with st.spinner(f"…{sid}"):
                        fc = run_lite(DEMAND_DF, sid, horizon)
                    hist = historical_slice(sid, 60)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=fc["date"], y=fc["yhat_upper_80"], mode="lines",
                        line=dict(width=0), showlegend=False, hoverinfo="skip",
                    ))
                    fig.add_trace(go.Scatter(
                        x=fc["date"], y=fc["yhat_lower_80"], mode="lines",
                        line=dict(width=0), fill="tonexty",
                        fillcolor="rgba(0,200,150,0.15)", showlegend=False, hoverinfo="skip",
                    ))
                    fig.add_trace(go.Scatter(
                        x=hist["date"], y=hist["demand"], mode="lines",
                        line=dict(color=ZONE_COLORS[sku.zone], width=2), name="History",
                    ))
                    fig.add_trace(go.Scatter(
                        x=fc["date"], y=fc["yhat"], mode="lines",
                        line=dict(color=COLORS["prophet"], width=2, dash="dot"),
                        name="Forecast",
                    ))
                    fig.update_layout(
                        template="plotly_dark", height=260,
                        title=dict(text=f"{sid} — {sku.name[:24]}", font=dict(size=12)),
                        plot_bgcolor="#0b0b0b", paper_bgcolor="#0b0b0b",
                        margin=dict(l=10, r=10, t=40, b=10),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"multi_sku_{sid}")


# ============================================================================
# TAB 4 — MODEL COMPARISON
# ============================================================================
with TABS[3]:
    st.markdown(f"### ⚖️ All 4 models — {primary_sku}")
    hist = historical_slice(primary_sku, 90)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["demand"], mode="lines",
        line=dict(color=COLORS["history"], width=2), name="Historical",
    ))
    for name in ["MovingAverage", "Prophet", "ARIMA", "Ensemble"]:
        f = PRIMARY["forecast"][name]
        fig.add_trace(go.Scatter(
            x=f["date"], y=f["yhat"], mode="lines",
            line=dict(color=MODEL_COLOR[name], width=2.5),
            name=name,
        ))
    fig.update_layout(
        template="plotly_dark", height=450,
        plot_bgcolor="#0b0b0b", paper_bgcolor="#0b0b0b",
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig, use_container_width=True, key="cmp_all_models")

    st.markdown("### 📐 Residuals (test set)")
    test_actual = PRIMARY["test"]["actual"]
    rcols = st.columns(2)
    for i, name in enumerate(["Prophet", "ARIMA", "Ensemble", "MovingAverage"]):
        f = PRIMARY["test"][name]
        residuals = test_actual["demand"].values - f["yhat"].values
        with rcols[i % 2]:
            hist_fig = go.Figure(data=[go.Histogram(
                x=residuals, marker_color=MODEL_COLOR[name], nbinsx=15,
            )])
            hist_fig.update_layout(
                template="plotly_dark", height=220,
                title=dict(text=f"{name} residuals", font=dict(size=12)),
                plot_bgcolor="#0b0b0b", paper_bgcolor="#0b0b0b",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(hist_fig, use_container_width=True, key=f"cmp_resid_{name}")

    st.markdown("### 📊 Comparison metrics")
    st.dataframe(
        PRIMARY["metrics"].style.format({
            "MAE":"{:.2f}","RMSE":"{:.2f}","MAPE":"{:.1f}%","Bias":"{:+.2f}","Coverage80":"{:.1f}%"
        }).background_gradient(subset=["MAPE"], cmap="RdYlGn_r"),
        use_container_width=True, hide_index=True,
    )


# ============================================================================
# TAB 5 — SEASONALITY
# ============================================================================
with TABS[4]:
    st.markdown(f"### 🔁 Seasonality decomposition — {primary_sku}")
    with st.spinner("Extracting Prophet components…"):
        try:
            comp_df = prophet_components(
                DEMAND_DF[DEMAND_DF["sku_id"] == primary_sku][["date", "demand"]],
                horizon=horizon,
            )
            ok = not comp_df.empty
        except Exception as e:
            ok = False
            comp_df = pd.DataFrame()

    if ok and "trend" in comp_df.columns:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Scatter(
                x=comp_df["ds"], y=comp_df["trend"], mode="lines",
                line=dict(color=COLORS["prophet"], width=2),
            ))
            fig.update_layout(template="plotly_dark", height=260,
                title="Trend", plot_bgcolor="#0b0b0b", paper_bgcolor="#0b0b0b",
                margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True, key="seas_trend")
        with c2:
            if "weekly" in comp_df.columns:
                wk = comp_df.copy()
                wk["dow"] = wk["ds"].dt.day_name()
                wk_mean = wk.groupby("dow")["weekly"].mean().reindex(
                    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                )
                fig = go.Figure(go.Bar(
                    x=wk_mean.index, y=wk_mean.values,
                    marker_color=COLORS["arima"],
                ))
                fig.update_layout(template="plotly_dark", height=260,
                    title="Weekly seasonality (effect on demand)",
                    plot_bgcolor="#0b0b0b", paper_bgcolor="#0b0b0b",
                    margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, key="seas_weekly")

        c3, c4 = st.columns(2)
        with c3:
            if "yearly" in comp_df.columns:
                yr = comp_df.copy()
                yr["doy"] = yr["ds"].dt.dayofyear
                yr_mean = yr.groupby("doy")["yearly"].mean()
                fig = go.Figure(go.Scatter(
                    x=yr_mean.index, y=yr_mean.values, mode="lines",
                    line=dict(color=COLORS["ma"], width=2),
                ))
                fig.update_layout(template="plotly_dark", height=260,
                    title="Yearly seasonality (effect by day-of-year)",
                    plot_bgcolor="#0b0b0b", paper_bgcolor="#0b0b0b",
                    margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, key="seas_yearly")
        with c4:
            if "holidays" in comp_df.columns:
                hd = comp_df[["ds", "holidays"]].copy()
                hd = hd[hd["holidays"] != 0]
                if not hd.empty:
                    fig = go.Figure(go.Bar(
                        x=hd["ds"], y=hd["holidays"],
                        marker_color=COLORS["ensemble"],
                    ))
                    fig.update_layout(template="plotly_dark", height=260,
                        title="French holiday effects",
                        plot_bgcolor="#0b0b0b", paper_bgcolor="#0b0b0b",
                        margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True, key="seas_holidays")
                else:
                    st.info("No holiday effects detected in the studied period.")
    else:
        st.warning("Prophet components unavailable. (Install prophet to enable.)")

    # Day-of-week × month heatmap on raw demand
    st.markdown("### 🌡️ Demand heatmap (day-of-week × month)")
    raw = DEMAND_DF[DEMAND_DF["sku_id"] == primary_sku].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw["dow"] = raw["date"].dt.day_name()
    raw["month"] = raw["date"].dt.month_name()
    pivot = raw.pivot_table(values="demand", index="dow", columns="month", aggfunc="mean")
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    month_order = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]
    pivot = pivot.reindex(index=dow_order, columns=month_order)
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale="Viridis", colorbar=dict(title="Units/day"),
    ))
    fig.update_layout(template="plotly_dark", height=340,
        plot_bgcolor="#0b0b0b", paper_bgcolor="#0b0b0b",
        margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True, key="seas_dow_month_heatmap")


# ============================================================================
# TAB 6 — ALERT CENTER
# ============================================================================
with TABS[5]:
    st.markdown("### 🚨 Stock Alert Center — 45 SKUs")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        f_zone = st.multiselect("Zone", ["FROZEN", "FRESH", "AMBIENT", "HEAVY"],
                                default=["FROZEN", "FRESH", "AMBIENT", "HEAVY"])
    with c2:
        f_level = st.multiselect("Alert level", ["CRITICAL", "WARNING", "WATCH", "OK"],
                                 default=["CRITICAL", "WARNING", "WATCH", "OK"])
    with c3:
        st.markdown(
            f"<div style='padding-top:30px;color:#9aa0a6'>"
            f"Showing alerts based on {horizon}-day Prophet forecast · lead time {lead_time}d</div>",
            unsafe_allow_html=True,
        )

    if ALERTS_DF.empty or "alert_level" not in ALERTS_DF.columns:
        st.info("No alerts at the moment.")
    else:
        filt = ALERTS_DF[
            (ALERTS_DF["zone"].isin(f_zone)) &
            (ALERTS_DF["alert_level"].isin(f_level))
        ].copy()

        if filt.empty:
            st.info("No alerts match these filters.")
        else:
            # Compute Net risk = stockout cost - order cost, then sort desc
            filt = filt.copy()
            filt["net_risk"] = filt["estimated_stockout_cost"] - filt["order_total_cost"]
            filt = filt.sort_values("net_risk", ascending=False)

            display_df = filt[[
                "sku_id","name","zone","current_stock","avg_daily_demand",
                "days_until_stockout","alert_level","reorder_qty","order_by_date",
                "estimated_stockout_cost","order_total_cost","net_risk",
            ]].rename(columns={
                "sku_id":"SKU","name":"Product","zone":"Zone","current_stock":"Stock",
                "avg_daily_demand":"Avg/day","days_until_stockout":"Days→Stockout",
                "alert_level":"Alert","reorder_qty":"Reorder qty",
                "order_by_date":"Order by","estimated_stockout_cost":"Stockout cost €",
                "order_total_cost":"Order cost €","net_risk":"Net risk €",
            })

            def colorize_row(row):
                color = ALERT_COLORS.get(row["Alert"], "#0b0b0b")
                return [f"background-color: {color}40" for _ in row]

            def colorize_net_risk(val):
                if val > 0:
                    return "color: #00C896; font-weight: 700;"
                if val < 0:
                    return "color: #888888;"
                return ""

            styled = (
                display_df.style
                .apply(colorize_row, axis=1)
                .map(colorize_net_risk, subset=["Net risk €"])
                .format({
                    "Stockout cost €":"{:.2f}",
                    "Order cost €":"{:.2f}",
                    "Net risk €":"{:+.2f}",
                    "Avg/day":"{:.1f}",
                })
            )

            st.dataframe(styled, use_container_width=True, hide_index=True, height=560)
            st.markdown(
                "<p style='color:#9aa0a6;font-size:12px;margin-top:6px'>"
                "<em>Net risk = estimated stockout loss minus reorder cost. "
                "Positive = ordering is financially justified.</em></p>",
                unsafe_allow_html=True,
            )


# ============================================================================
# TAB 7 — REORDER RECOMMENDATIONS
# ============================================================================
with TABS[6]:
    st.markdown("### 🛒 Reorder Recommendations")
    reorder_df = build_reorder_list(ALERTS_DF)

    if reorder_df.empty:
        st.success("✅ No reorders needed — inventory healthy.")
    else:
        # Add Expected delivery = Order by + lead_time days
        reorder_df = reorder_df.copy()
        reorder_df["expected_delivery"] = reorder_df["order_by_date"].apply(
            lambda d: (pd.to_datetime(d) + pd.Timedelta(days=int(lead_time))).date().isoformat()
        )
        earliest_delivery = min(reorder_df["expected_delivery"])

        total_cost = float(reorder_df["order_total_cost"].sum())
        n_skus = len(reorder_df)
        n_crit = int((reorder_df["alert_level"] == "CRITICAL").sum())

        st.markdown(
            f"<p style='color:#00C896;font-weight:600;margin:6px 0 14px 0'>"
            f"📦 Next delivery wave: <span style='color:#FAFAFA'>{earliest_delivery}</span>"
            f"</p>",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(kpi("SKUs to order", str(n_skus), "Prioritized list", "blue"), unsafe_allow_html=True)
        with c2: st.markdown(kpi("Critical", str(n_crit), "Immediate action", "red"), unsafe_allow_html=True)
        with c3: st.markdown(kpi("Total order cost", f"€ {total_cost:,.0f}", "All suppliers", "orange"), unsafe_allow_html=True)
        with c4:
            avg_priority = reorder_df["priority_score"].mean()
            st.markdown(kpi("Avg priority", f"{avg_priority:.0f}/100", "Higher = more urgent", "green"), unsafe_allow_html=True)

        st.markdown("#### 📋 Ranked reorder list")
        st.dataframe(
            reorder_df[[
                "priority_score","sku_id","name","zone","supplier",
                "current_stock","reorder_qty","order_by_date","expected_delivery",
                "alert_level","order_total_cost",
            ]].rename(columns={
                "priority_score":"Priority","sku_id":"SKU","name":"Product","zone":"Zone",
                "supplier":"Supplier","current_stock":"Stock","reorder_qty":"Qty",
                "order_by_date":"Order by","expected_delivery":"Expected delivery",
                "alert_level":"Alert","order_total_cost":"Cost €",
            }).style.format({"Priority":"{:.0f}","Cost €":"{:.2f}"}),
            use_container_width=True, hide_index=True, height=400,
        )

        st.markdown("#### 🏷️ Supplier grouping (consolidate orders)")
        suppliers = group_by_supplier(reorder_df)
        st.dataframe(suppliers.style.format({
            "total_cost":"{:.2f}","avg_priority":"{:.1f}",
        }), use_container_width=True, hide_index=True)

        st.markdown("#### 📤 Exports")
        ec1, ec2 = st.columns(2)
        with ec1:
            xlsx_bytes = export_excel(reorder_df)
            st.download_button(
                "📊 Download Excel reorder sheet",
                data=xlsx_bytes,
                file_name=f"reorder_{date.today().isoformat()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with ec2:
            pdf_bytes = export_pdf(
                reorder_df,
                company="Warehouse Operations — Iyad Belkadi",
                lead_time=int(lead_time),
            )
            st.download_button(
                "📄 Download PDF purchase order",
                data=pdf_bytes,
                file_name=f"purchase_order_{date.today().isoformat()}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


# ============================================================================
# TAB 8 — WHAT-IF SIMULATOR
# ============================================================================
with TABS[7]:
    st.markdown("### 💭 What-if simulator")
    st.markdown(
        "<p style='color:#9aa0a6;margin-top:-8px'>Simulate the impact of ordering today</p>",
        unsafe_allow_html=True,
    )

    # ---- Section 1 — Inputs --------------------------------------------------
    all_ids = [s.sku_id for s in ALL_SKUS]
    wi_c1, wi_c2, wi_c3, wi_c4 = st.columns(4)
    with wi_c1:
        wi_sku = st.selectbox(
            "SKU",
            all_ids,
            index=all_ids.index(primary_sku),
            format_func=lambda x: f"{x} — {get_sku(x).name[:24]}",
            key="whatif_sku",
        )
    with wi_c2:
        wi_qty = st.slider(
            "Units to order today", min_value=0, max_value=2000,
            value=200, step=50, key="whatif_qty",
        )
    with wi_c3:
        wi_lead = st.number_input(
            "Order arrives in N days", min_value=1, max_value=30,
            value=int(lead_time), key="whatif_lead",
        )
    with wi_c4:
        wi_stock = st.number_input(
            "Current stock", min_value=0,
            value=int(st.session_state.stocks.get(wi_sku, 100)),
            step=10, key="whatif_stock",
        )

    # ---- Compute scenarios (using already-cached forecasts) ------------------
    wi_sku_obj = get_sku(wi_sku)
    wi_fc_full = ALL_FORECASTS[wi_sku]
    wi_fc = wi_fc_full.head(60).reset_index(drop=True)
    cumulative = wi_fc["yhat"].cumsum().values

    avg_daily = float(wi_fc["yhat"].head(7).mean())
    from core.alerts import classify_alert
    safety = 1.5 * avg_daily * wi_lead

    # Current scenario — depletion
    stock_curve_now = np.maximum(wi_stock - cumulative, 0)
    above_now = np.where(cumulative >= wi_stock)[0]
    days_to_stockout_now = int(above_now[0]) + 1 if above_now.size else len(wi_fc)

    # After-order scenario — order arrives at day wi_lead
    stock_after = np.full(len(wi_fc), float(wi_stock))
    arrival_idx = min(int(wi_lead), len(wi_fc) - 1)
    stock_after[arrival_idx:] = wi_stock + wi_qty
    stock_curve_after = np.maximum(stock_after - cumulative, 0)
    out_after = np.where(stock_curve_after <= 0)[0]
    days_to_stockout_after = int(out_after[0]) + 1 if out_after.size else len(wi_fc)

    alert_now = classify_alert(wi_stock, safety)
    alert_after = classify_alert(wi_stock + wi_qty, safety)

    stockout_cost_now = (
        float(wi_fc["yhat"].iloc[days_to_stockout_now:].sum()) * wi_sku_obj.margin
        if days_to_stockout_now < len(wi_fc) else 0.0
    )
    order_cost = wi_qty * wi_sku_obj.unit_cost
    net_saving = stockout_cost_now - order_cost

    # ---- Section 2 — Live metric cards --------------------------------------
    st.markdown("#### 📊 Live impact")
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(kpi(
            "Current — days to stockout",
            f"{days_to_stockout_now} d",
            f"Alert: {alert_now}",
            "red" if alert_now in ("CRITICAL", "WARNING") else "green",
        ), unsafe_allow_html=True)
        st.markdown(kpi(
            "Stockout cost if no action",
            f"€ {stockout_cost_now:,.2f}",
            "Lost margin over 60 d",
            "red" if stockout_cost_now > 0 else "green",
        ), unsafe_allow_html=True)
    with m2:
        st.markdown(kpi(
            "After order — days to stockout",
            f"{days_to_stockout_after} d",
            f"Alert: {alert_after}",
            "green" if alert_after in ("OK", "WATCH") else "orange",
        ), unsafe_allow_html=True)
        st.markdown(kpi(
            "Order cost",
            f"€ {order_cost:,.2f}",
            f"{wi_qty} × € {wi_sku_obj.unit_cost:.2f}",
            "blue",
        ), unsafe_allow_html=True)

    # ---- Section 3 — Projection chart ---------------------------------------
    st.markdown("#### 📈 Stock projection (60 days)")

    # Every date passed to Plotly must be a string. Build the x-axis upfront.
    date_strs = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in wi_fc["date"]]
    x_first = date_strs[0]
    x_last  = date_strs[-1]

    # Y-axis upper bound for vertical-line traces
    max_stock = float(max(
        float(np.max(stock_curve_now))   if len(stock_curve_now)   else 0.0,
        float(np.max(stock_curve_after)) if len(stock_curve_after) else 0.0,
        float(safety),
        float(wi_stock + wi_qty),
    )) * 1.10
    if max_stock <= 0:
        max_stock = max(1.0, float(safety) * 1.5)

    # Order-arrival date — aligned to a forecast bucket so the vertical line
    # lands exactly on a category on the x-axis.
    arrival_idx_clamped = min(int(wi_lead), len(date_strs) - 1)
    arrival_str = date_strs[arrival_idx_clamped]

    proj_fig = go.Figure()

    # 1. Stock depletion without order (red dashed)
    proj_fig.add_trace(go.Scatter(
        x=date_strs, y=stock_curve_now, mode="lines",
        line=dict(color="#FF4B4B", width=2, dash="dash"),
        name="Without order",
    ))

    # 2. Stock after order arrives (green solid)
    proj_fig.add_trace(go.Scatter(
        x=date_strs, y=stock_curve_after, mode="lines",
        line=dict(color="#00C896", width=2.5),
        name=f"With order (+{wi_qty} u)",
    ))

    # 3. Safety-stock horizontal line (orange dotted) — drawn as a scatter
    proj_fig.add_trace(go.Scatter(
        x=[x_first, x_last],
        y=[float(safety), float(safety)],
        mode="lines",
        line=dict(color="#FFB347", width=1, dash="dot"),
        name=f"Safety stock ({safety:.0f} u)",
        hoverinfo="skip",
    ))

    # 4. Stockout floor (red) — y=0
    proj_fig.add_trace(go.Scatter(
        x=[x_first, x_last],
        y=[0, 0],
        mode="lines",
        line=dict(color="#FF4B4B", width=1),
        name="Stockout level",
        hoverinfo="skip",
    ))

    # 5. Order-arrival vertical line (green dashed) — drawn as scatter
    proj_fig.add_trace(go.Scatter(
        x=[arrival_str, arrival_str],
        y=[0, max_stock],
        mode="lines",
        line=dict(color="#00C896", width=2, dash="dash"),
        name=f"Order arrives (+{wi_lead}d)",
        hoverinfo="skip",
    ))

    # 6. Stockout vertical marker (red dotted), if a stockout occurs in window
    if days_to_stockout_now < len(date_strs):
        stockout_str = date_strs[days_to_stockout_now - 1]
        proj_fig.add_trace(go.Scatter(
            x=[stockout_str, stockout_str],
            y=[0, max_stock * 0.6],
            mode="lines",
            line=dict(color="#FF4B4B", width=1, dash="dot"),
            name=f"Stockout d+{days_to_stockout_now}",
            hoverinfo="skip",
        ))
        proj_fig.add_trace(go.Scatter(
            x=[stockout_str],
            y=[max_stock * 0.62],
            mode="markers+text",
            marker=dict(color="#FF4B4B", size=10, symbol="x"),
            text=[f"Stockout d+{days_to_stockout_now}"],
            textposition="top center",
            textfont=dict(color="#FF4B4B", size=11),
            showlegend=False,
            hoverinfo="skip",
        ))

    proj_fig.update_layout(
        template="plotly_dark", height=420,
        plot_bgcolor="#0b0b0b", paper_bgcolor="#0b0b0b",
        margin=dict(l=20, r=20, t=20, b=20),
        # type="category" prevents Plotly from interpreting the axis as a
        # date axis, eliminating any internal Timestamp arithmetic.
        xaxis=dict(title="Date", type="category", tickangle=-45, nticks=12),
        yaxis=dict(title="Units in stock", range=[0, max_stock]),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(proj_fig, use_container_width=True, key="whatif_projection")

    # ---- Section 4 — Decision recommendation --------------------------------
    st.markdown("#### 🧭 Decision recommendation")
    expected_delivery = (date.today() + timedelta(days=int(wi_lead))).isoformat()
    if net_saving > 0 and wi_qty > 0:
        st.success(
            f"✅ **Recommended**: Order **{wi_qty}** units from **{wi_sku_obj.supplier}**.  \n"
            f"Expected delivery: **{expected_delivery}**.  \n"
            f"This prevents a € **{stockout_cost_now:,.2f}** stockout and costs € **{order_cost:,.2f}**.  \n"
            f"**Net saving: € {net_saving:,.2f}**."
        )
    elif wi_qty == 0:
        cover_until = (date.today() + timedelta(days=days_to_stockout_now)).isoformat()
        reassess_in = max(1, days_to_stockout_now - int(wi_lead))
        st.info(
            f"⚠️ **Optional**: Current stock covers **{days_to_stockout_now}** days.  \n"
            f"Safety stock threshold met until **{cover_until}**.  \n"
            f"Monitor and reassess in **{reassess_in}** days."
        )
    else:
        st.warning(
            f"💡 Ordering not justified by stockout cost alone — net change € **{net_saving:+,.2f}**.  \n"
            f"Consider a smaller order or wait. Stock covers **{days_to_stockout_now}** days at current pace."
        )


# ============================================================================
# TAB 9 — ANOMALY DETECTION
# ============================================================================
with TABS[8]:
    st.markdown(f"### ⚠️ Anomalies detected — {primary_sku}")
    ev_sku = EVENTS_DF[EVENTS_DF["sku_id"] == primary_sku].copy()
    ev_sku["date"] = pd.to_datetime(ev_sku["date"])
    ev_sku = ev_sku.sort_values("date")

    raw = DEMAND_DF[DEMAND_DF["sku_id"] == primary_sku].sort_values("date")

    # Statistical anomaly detection (z-score on rolling)
    raw["rolling_mean"] = raw["demand"].rolling(14, min_periods=3).mean()
    raw["rolling_std"]  = raw["demand"].rolling(14, min_periods=3).std()
    raw["z"] = (raw["demand"] - raw["rolling_mean"]) / raw["rolling_std"].replace(0, np.nan)
    raw["is_anomaly"] = raw["z"].abs() > 2.5

    n_anom = int(raw["is_anomaly"].sum())
    n_stockout = int((ev_sku["type"] == "stockout").sum())
    n_promo = int((ev_sku["type"] == "promo_spike").sum())
    n_surge = int((ev_sku["type"] == "delivery_surge").sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi("Statistical anomalies", str(n_anom), "|z|>2.5", "orange"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("Stockouts", str(n_stockout), "Zero-demand days", "red"), unsafe_allow_html=True)
    with c3: st.markdown(kpi("Promo spikes", str(n_promo), "+100% demand", "blue"), unsafe_allow_html=True)
    with c4: st.markdown(kpi("Delivery surges", str(n_surge), "Post-stockout", "green"), unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=raw["date"], y=raw["demand"], mode="lines",
        line=dict(color=COLORS["history"], width=1), name="Demand",
    ))
    anom = raw[raw["is_anomaly"]]
    if not anom.empty:
        fig.add_trace(go.Scatter(
            x=anom["date"], y=anom["demand"], mode="markers",
            marker=dict(color="#FF4B4B", size=8, symbol="circle-open"),
            name="Anomaly (z>2.5)",
        ))
    # Event markers
    type_color = {"stockout":"#FF4B4B","promo_spike":"#FFD700","delivery_surge":"#00C896"}
    for t in ev_sku["type"].unique():
        e = ev_sku[ev_sku["type"] == t]
        ymap = raw.set_index("date")["demand"].to_dict()
        ys = [ymap.get(d, 0) for d in e["date"]]
        fig.add_trace(go.Scatter(
            x=e["date"], y=ys, mode="markers",
            marker=dict(color=type_color.get(t, "#FFB347"), size=10, symbol="diamond"),
            name=t,
        ))
    fig.update_layout(template="plotly_dark", height=400,
        plot_bgcolor="#0b0b0b", paper_bgcolor="#0b0b0b",
        margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, key="anom_chart")

    st.markdown("#### 📜 Event log")
    if ev_sku.empty:
        st.info("No injected anomalies for this SKU.")
    else:
        # Add impact column
        impact_rows = []
        for _, e in ev_sku.iterrows():
            d_from = e["date"]
            d_to = d_from + pd.Timedelta(days=int(e["length_days"]))
            window = raw[(raw["date"] >= d_from) & (raw["date"] < d_to)]
            baseline = raw[(raw["date"] >= d_from - pd.Timedelta(days=14)) &
                           (raw["date"] < d_from)]["demand"].mean()
            actual = window["demand"].sum()
            expected = baseline * int(e["length_days"]) if pd.notna(baseline) else 0
            impact_rows.append({
                "Date": e["date"].date().isoformat(),
                "Type": e["type"],
                "Length (days)": int(e["length_days"]),
                "Expected units": int(round(expected)),
                "Actual units": int(actual),
                "Impact (units)": int(actual - round(expected)),
            })
        st.dataframe(pd.DataFrame(impact_rows), use_container_width=True, hide_index=True, height=350)


# ============================================================================
# TAB 10 — HEATMAP & PATTERNS
# ============================================================================
with TABS[9]:
    st.markdown("### 🔥 Cross-SKU demand patterns")

    raw = DEMAND_DF.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw["dow"] = raw["date"].dt.day_name()
    raw["month"] = raw["date"].dt.month_name()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Demand by SKU × day-of-week")
        piv = raw.pivot_table(values="demand", index="sku_id", columns="dow", aggfunc="mean")
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        piv = piv[dow_order]
        fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index,
                                   colorscale="Viridis"))
        fig.update_layout(template="plotly_dark", height=700,
            plot_bgcolor="#0b0b0b", paper_bgcolor="#0b0b0b",
            margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True, key="heat_sku_dow")

    with c2:
        st.markdown("#### Demand by SKU × month")
        piv2 = raw.pivot_table(values="demand", index="sku_id", columns="month", aggfunc="mean")
        month_order = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        piv2 = piv2[month_order]
        fig = go.Figure(go.Heatmap(z=piv2.values, x=piv2.columns, y=piv2.index,
                                   colorscale="Plasma"))
        fig.update_layout(template="plotly_dark", height=700,
            plot_bgcolor="#0b0b0b", paper_bgcolor="#0b0b0b",
            margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True, key="heat_sku_month")

    st.markdown("### 🏆 Top 10 / Bottom 10 SKUs by volume")
    totals = raw.groupby(["sku_id", "sku_name", "zone"])["demand"].sum().reset_index()
    totals = totals.sort_values("demand", ascending=False)

    t1, t2 = st.columns(2)
    with t1:
        top = totals.head(10)
        fig = go.Figure(go.Bar(
            x=top["demand"], y=top["sku_name"], orientation="h",
            marker_color=[ZONE_COLORS[z] for z in top["zone"]],
        ))
        fig.update_layout(template="plotly_dark", height=400, title="Top 10",
            plot_bgcolor="#0b0b0b", paper_bgcolor="#0b0b0b",
            margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True, key="heat_top10")

    with t2:
        bot = totals.tail(10).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=bot["demand"], y=bot["sku_name"], orientation="h",
            marker_color=[ZONE_COLORS[z] for z in bot["zone"]],
        ))
        fig.update_layout(template="plotly_dark", height=400, title="Bottom 10",
            plot_bgcolor="#0b0b0b", paper_bgcolor="#0b0b0b",
            margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True, key="heat_bot10")

    st.markdown("### 🔗 Correlation matrix (top 15 SKUs)")
    top15_ids = totals.head(15)["sku_id"].tolist()
    pivot_demand = raw[raw["sku_id"].isin(top15_ids)].pivot(
        index="date", columns="sku_id", values="demand"
    )
    corr = pivot_demand.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
    ))
    fig.update_layout(template="plotly_dark", height=500,
        plot_bgcolor="#0b0b0b", paper_bgcolor="#0b0b0b",
        margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True, key="heat_correlation")


# ============================================================================
# TAB 11 — HISTORY & PERFORMANCE
# ============================================================================
with TABS[10]:
    st.markdown("### 📊 Model performance — primary SKU")
    test_actual = PRIMARY["test"]["actual"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_actual["date"], y=test_actual["demand"],
        mode="lines+markers", line=dict(color=COLORS["history"], width=2),
        marker=dict(size=5), name="Actual",
    ))
    for name in ["MovingAverage", "Prophet", "ARIMA", "Ensemble"]:
        f = PRIMARY["test"][name]
        fig.add_trace(go.Scatter(
            x=f["date"], y=f["yhat"], mode="lines",
            line=dict(color=MODEL_COLOR[name], width=2, dash="dot"),
            name=name,
        ))
    fig.update_layout(template="plotly_dark", height=400,
        title=f"Backtest — last {len(test_actual)} days",
        plot_bgcolor="#0b0b0b", paper_bgcolor="#0b0b0b",
        margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, key="perf_all_skus_backtest")

    st.markdown("### 📈 Cumulative forecast error")
    cumfig = go.Figure()
    for name in ["MovingAverage", "Prophet", "ARIMA", "Ensemble"]:
        f = PRIMARY["test"][name]
        cum_err = np.cumsum(np.abs(test_actual["demand"].values - f["yhat"].values))
        cumfig.add_trace(go.Scatter(
            x=f["date"], y=cum_err, mode="lines",
            line=dict(color=MODEL_COLOR[name], width=2),
            name=name,
        ))
    cumfig.update_layout(template="plotly_dark", height=320,
        plot_bgcolor="#0b0b0b", paper_bgcolor="#0b0b0b",
        margin=dict(l=10, r=10, t=20, b=10),
        title="Lower = better", hovermode="x unified")
    st.plotly_chart(cumfig, use_container_width=True, key="perf_cumerr")

    st.markdown("### 🏆 Best model per SKU — all 45 SKUs")
    with st.spinner("Computing all SKUs… (first run trains 45 × Prophet + ARIMA; subsequent runs are cached)"):
        rows = []
        for sku_obj in ALL_SKUS:
            sid = sku_obj.sku_id
            try:
                sku_res = run_pipeline(DEMAND_DF, sid, horizon)
                best_name = sku_res["best"]
                mape_val = sku_res["metrics"].set_index("Model").loc[best_name, "MAPE"]
                rows.append({
                    "SKU": sid,
                    "Product": sku_obj.name,
                    "Zone": sku_obj.zone,
                    "Best model": best_name,
                    "MAPE %": round(float(mape_val), 2) if pd.notna(mape_val) else None,
                    "MAE":  round(float(sku_res["metrics"].set_index("Model").loc[best_name, "MAE"]), 2),
                    "Prophet weight": f"{sku_res['weights']['Prophet']*100:.0f}%",
                    "ARIMA weight":   f"{sku_res['weights']['ARIMA']*100:.0f}%",
                })
            except Exception:
                continue
    all_perf_df = pd.DataFrame(rows)
    st.dataframe(all_perf_df, use_container_width=True, hide_index=True, height=600)
    if not all_perf_df.empty:
        st.caption(
            f"Best-model distribution: "
            + " · ".join(
                f"**{m}** {n}"
                for m, n in all_perf_df["Best model"].value_counts().items()
            )
        )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    "<hr style='border-color:#1a1a1a'>"
    "<p style='text-align:center;color:#6f7681;font-size:12px'>"
    "Demand Forecasting Engine · Project 3 · "
    "Iyad Belkadi · Streamlit · Prophet · ARIMA · scikit-learn"
    "</p>",
    unsafe_allow_html=True,
)
