import numpy as np
import plotly.express as px
import streamlit as st

from components.kpi_cards import render_kpis
from components.sidebar import global_filters
from models.forecasting_models import run_model
from utils.pipeline import apply_filters, load_and_prepare_data


st.title("Business Insights Dashboard")

try:
    _, _, enriched_df, _, keys, _ = load_and_prepare_data()
except Exception as exc:
    st.error(f"Data pipeline error: {exc}")
    st.stop()

time_col = keys["time_col"]
target_col = keys["target_col"]
product_col = keys["product_col"]
category_col = keys["category_col"]

filters = global_filters(enriched_df, time_col, product_col, category_col)
filtered = apply_filters(enriched_df, filters, time_col, product_col, category_col)

if filtered.empty:
    st.warning("No records found for selected filters.")
    st.stop()

total_sales = float(filtered[target_col].sum())
avg_daily = float(filtered.groupby(time_col)[target_col].sum().mean())
inv_turnover = float(total_sales / max((filtered[target_col].mean() * 30), 1.0))

aggregated = filtered.groupby(time_col, as_index=False)[target_col].sum().sort_values(time_col)
if len(aggregated) > 50:
    # Add lightweight time-derived numeric features so tree models always have valid inputs.
    aggregated["year"] = aggregated[time_col].dt.year
    aggregated["month"] = aggregated[time_col].dt.month
    aggregated["day"] = aggregated[time_col].dt.day
    aggregated["dayofweek"] = aggregated[time_col].dt.dayofweek
    aggregated["week"] = aggregated[time_col].dt.isocalendar().week.astype(int)
    aggregated["lag_1"] = aggregated[target_col].shift(1)
    aggregated["rolling_mean_7"] = aggregated[target_col].shift(1).rolling(7, min_periods=1).mean()
    aggregated = aggregated.bfill().ffill()

    try:
        metrics = run_model(aggregated, "Random Forest", time_col, target_col, test_size=0.2)["metrics"]
        forecast_acc = max(0.0, 100 - metrics["MAPE"])
    except Exception:
        # Safe fallback to keep the Insights page resilient.
        metrics = run_model(aggregated[[time_col, target_col]], "ARIMA", time_col, target_col, test_size=0.2)["metrics"]
        forecast_acc = max(0.0, 100 - metrics["MAPE"])
else:
    forecast_acc = 0.0

render_kpis(
    {
        "Total Sales": f"{total_sales:,.0f}",
        "Forecast Accuracy": f"{forecast_acc:.2f}%",
        "Inventory Turnover": f"{inv_turnover:.2f}",
    }
)

if product_col and product_col in filtered.columns:
    top_products = filtered.groupby(product_col, as_index=False)[target_col].sum().sort_values(target_col, ascending=False).head(10)
    st.plotly_chart(px.bar(top_products, x=product_col, y=target_col, title="Top Products"), use_container_width=True)

daily = filtered.groupby(time_col, as_index=False)[target_col].sum().sort_values(time_col)
daily["z_score"] = (daily[target_col] - daily[target_col].mean()) / max(daily[target_col].std(), 1e-9)
anoms = daily[np.abs(daily["z_score"]) > 2.5]
st.plotly_chart(px.line(daily, x=time_col, y=target_col, title="Demand Timeline"), use_container_width=True)
if not anoms.empty:
    st.warning(f"Demand spike alerts: {len(anoms)} anomalous dates detected.")
    st.dataframe(anoms[[time_col, target_col, "z_score"]], use_container_width=True)
else:
    st.success("No major demand anomalies detected for selected filters.")

season = filtered.groupby("month", as_index=False)[target_col].mean()
st.plotly_chart(px.line(season, x="month", y=target_col, markers=True, title="Seasonal Pattern (Monthly Average)"), use_container_width=True)

st.subheader("Scenario Simulation")
factor = st.slider("Demand Multiplier", min_value=0.5, max_value=1.5, value=1.0, step=0.05)
sim_total = total_sales * factor
sim_avg = avg_daily * factor
render_kpis({"Simulated Total Demand": f"{sim_total:,.0f}", "Simulated Daily Avg": f"{sim_avg:,.2f}"})
