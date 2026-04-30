import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import box_outliers, heatmap_day_month, line_sales_trend
from components.sidebar import global_filters
from utils.pipeline import apply_filters, load_and_prepare_data


st.title("Exploratory Data Analysis")

try:
    _, _, enriched_df, _, keys, _ = load_and_prepare_data()
except Exception as exc:
    st.error(f"Failed to prepare data: {exc}")
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

daily = filtered.groupby(time_col, as_index=False)[target_col].sum()
st.plotly_chart(line_sales_trend(daily, time_col, target_col), use_container_width=True)

if category_col and category_col in filtered.columns:
    by_cat = filtered.groupby([time_col, category_col], as_index=False)[target_col].sum()
    st.plotly_chart(line_sales_trend(by_cat, time_col, target_col, color_col=category_col), use_container_width=True)

if product_col and product_col in filtered.columns:
    top_products = filtered.groupby(product_col, as_index=False)[target_col].sum().sort_values(target_col, ascending=False).head(10)
    fig_top = px.bar(top_products, x=product_col, y=target_col, title="Top Products by Demand")
    st.plotly_chart(fig_top, use_container_width=True)

st.plotly_chart(box_outliers(filtered, "month", target_col), use_container_width=True)
st.plotly_chart(heatmap_day_month(filtered, "dayofweek", "month", target_col), use_container_width=True)

corr_cols = filtered.select_dtypes(include=["number"]).columns
if len(corr_cols) > 1:
    corr = filtered[corr_cols].corr()
    st.plotly_chart(px.imshow(corr, text_auto=True, title="Correlation Heatmap"), use_container_width=True)
