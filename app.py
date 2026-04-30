from pathlib import Path

import streamlit as st


st.set_page_config(page_title="Smart Retail Forecasting", page_icon=":shopping_trolley:", layout="wide")

st.title("Smart Retail Demand Forecasting & Inventory Optimization")
st.markdown(
    """
This production-ready app provides:
- strict data cleaning and validation
- demand forecasting (ARIMA, Prophet, Random Forest, XGBoost)
- inventory optimization (safety stock, reorder point, EOQ)
- business insights and alerts
"""
)

st.info("Use the left sidebar to navigate pages. Start with `Data Overview` to inspect cleaning report.")

data_dir = Path("data")
if not data_dir.exists():
    data_dir.mkdir(parents=True, exist_ok=True)

