import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def line_sales_trend(df: pd.DataFrame, time_col: str, target_col: str, color_col: str = ""):
    if color_col and color_col in df.columns:
        fig = px.line(df, x=time_col, y=target_col, color=color_col, title="Sales Trend")
    else:
        fig = px.line(df, x=time_col, y=target_col, title="Sales Trend")
    fig.update_layout(hovermode="x unified")
    return fig


def box_outliers(df: pd.DataFrame, x_col: str, y_col: str):
    return px.box(df, x=x_col, y=y_col, title=f"Outlier View: {y_col} by {x_col}")


def heatmap_day_month(df: pd.DataFrame, day_col: str, month_col: str, target_col: str):
    pivot = df.pivot_table(index=day_col, columns=month_col, values=target_col, aggfunc="mean")
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="Blues",
            colorbar={"title": f"Avg {target_col}"},
        )
    )
    fig.update_layout(title="Seasonality Heatmap")
    return fig


def forecast_plot(actual_df: pd.DataFrame, pred_series, time_col: str, target_col: str, ci=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_df[time_col], y=actual_df[target_col], mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=actual_df[time_col], y=pred_series, mode="lines", name="Forecast"))
    if ci is not None:
        fig.add_trace(
            go.Scatter(
                x=actual_df[time_col],
                y=ci["yhat_upper"],
                mode="lines",
                line={"width": 0},
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=actual_df[time_col],
                y=ci["yhat_lower"],
                mode="lines",
                fill="tonexty",
                line={"width": 0},
                name="Confidence Interval",
            )
        )
    fig.update_layout(title="Forecast vs Actual", hovermode="x unified")
    return fig
