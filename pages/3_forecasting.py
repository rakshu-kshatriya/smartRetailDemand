import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import forecast_plot
from components.sidebar import global_filters
from models.forecasting_models import run_model
from utils.pipeline import apply_filters, load_and_prepare_data


st.title("Demand Forecasting")

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

if len(filtered) < 60:
    st.warning("Not enough filtered data for robust forecasting. Widen filters or date range.")
    st.stop()

model_name = st.selectbox("Select Model", ["ARIMA", "Prophet", "Random Forest", "XGBoost"])
test_size = st.slider("Train/Test Split (test proportion)", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
horizon = st.slider("Forecast Horizon (days)", min_value=7, max_value=90, value=30, step=1)
run_btn = st.button("Run Forecast")

if run_btn:
    with st.spinner("Training model and generating forecast..."):
        series_df = filtered.sort_values(time_col).copy()
        if product_col and product_col in series_df.columns and filters.get("product") == "All":
            # Aggregate across products for global forecasting.
            series_df = series_df.groupby(time_col, as_index=False)[target_col].sum()

        result = run_model(series_df, model_name, time_col, target_col, test_size=test_size)
        metrics = result["metrics"]
        test_df = result["test"]
        preds = result["predictions"]
        ci = result["confidence_intervals"]

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{metrics['MAE']:.2f}")
        c2.metric("RMSE", f"{metrics['RMSE']:.2f}")
        c3.metric("MAPE", f"{metrics['MAPE']:.2f}%")

        st.plotly_chart(forecast_plot(test_df, preds, time_col, target_col, ci), use_container_width=True)

        # Simple forward projection based on the last predicted value trend.
        last_date = test_df[time_col].max()
        trend = float(np.mean(np.diff(preds[-7:]))) if len(preds) > 7 else 0.0
        projected = []
        current = float(max(preds[-1], 0))
        for i in range(1, horizon + 1):
            current = max(current + trend, 0)
            projected.append((last_date + pd.Timedelta(days=i), current))

        future_df = pd.DataFrame(projected, columns=[time_col, "forecast_demand"])
        st.plotly_chart(px.line(future_df, x=time_col, y="forecast_demand", title="Future Forecast Projection"), use_container_width=True)

        if result.get("feature_importance") is not None:
            st.subheader("Feature Importance")
            st.plotly_chart(
                px.bar(result["feature_importance"].head(15), x="feature", y="importance", title="Top Feature Importance"),
                use_container_width=True,
            )

        export_df = test_df[[time_col, target_col]].copy()
        export_df["forecast"] = preds
        st.download_button(
            "Download Forecast Results",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="forecast_results.csv",
            mime="text/csv",
        )
