import numpy as np
import pandas as pd
import streamlit as st

from components.sidebar import global_filters
from utils.pipeline import apply_filters, load_and_prepare_data


def z_value(service_level: float) -> float:
    if service_level >= 0.99:
        return 2.33
    if service_level >= 0.98:
        return 2.05
    if service_level >= 0.95:
        return 1.65
    if service_level >= 0.90:
        return 1.28
    return 1.0


st.title("Inventory Optimization")

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

lead_time_days = st.number_input("Lead Time (days)", min_value=1, max_value=90, value=14)
service_level = st.slider("Service Level", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
ordering_cost = st.number_input("Ordering Cost per Order", min_value=1.0, value=100.0, step=1.0)
holding_cost = st.number_input("Holding Cost per Unit/Year", min_value=0.1, value=5.0, step=0.1)
current_stock = st.number_input("Current Stock", min_value=0, value=500, step=10)

avg_daily_demand = float(filtered[target_col].mean())
std_daily_demand = float(filtered[target_col].std() if filtered[target_col].std() > 0 else 1.0)
annual_demand = max(avg_daily_demand * 365.0, 1.0)

z = z_value(service_level)
safety_stock = z * std_daily_demand * np.sqrt(lead_time_days)
reorder_point = avg_daily_demand * lead_time_days + safety_stock
eoq = np.sqrt((2 * annual_demand * ordering_cost) / max(holding_cost, 1e-6))

col1, col2, col3 = st.columns(3)
col1.metric("Safety Stock", f"{safety_stock:.0f}")
col2.metric("Reorder Point", f"{reorder_point:.0f}")
col3.metric("EOQ", f"{eoq:.0f}")

status = "optimal"
message = "Inventory is healthy."
if current_stock < safety_stock:
    status = "critical"
    message = "Stock below safety stock. Immediate replenishment required."
elif current_stock < reorder_point:
    status = "warning"
    message = "Stock below reorder point. Plan replenishment soon."
elif current_stock > reorder_point * 2:
    status = "warning"
    message = "Possible overstock risk."

color_map = {"critical": "#ff4b4b", "warning": "#f4c430", "optimal": "#2ca02c"}
st.markdown(
    f"<div style='padding:12px;border-radius:8px;background:{color_map[status]};color:white'><b>{status.upper()}</b> - {message}</div>",
    unsafe_allow_html=True,
)

recommendation_df = pd.DataFrame(
    {
        "metric": ["avg_daily_demand", "std_daily_demand", "safety_stock", "reorder_point", "eoq", "current_stock"],
        "value": [avg_daily_demand, std_daily_demand, safety_stock, reorder_point, eoq, current_stock],
    }
)
st.dataframe(recommendation_df, use_container_width=True)

st.download_button(
    "Download Inventory Recommendations",
    data=recommendation_df.to_csv(index=False).encode("utf-8"),
    file_name="inventory_recommendations.csv",
    mime="text/csv",
)
