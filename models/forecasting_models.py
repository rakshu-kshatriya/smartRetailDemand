from typing import Dict, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

from utils.evaluation import regression_metrics


def train_test_split_time(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def run_arima(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str):
    model = ARIMA(train_df[target_col], order=(2, 1, 2))
    fit = model.fit()
    pred = fit.forecast(steps=len(test_df))
    return np.array(pred), None


def run_prophet(train_df: pd.DataFrame, test_df: pd.DataFrame, time_col: str, target_col: str):
    model_df = train_df[[time_col, target_col]].rename(columns={time_col: "ds", target_col: "y"})
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(model_df)
    future = test_df[[time_col]].rename(columns={time_col: "ds"})
    forecast = model.predict(future)
    pred = forecast["yhat"].values
    ci = forecast[["yhat_lower", "yhat_upper"]]
    return pred, ci


def _feature_cols(df: pd.DataFrame, time_col: str, target_col: str):
    ignore = {time_col, target_col}
    return [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]


def run_random_forest(train_df: pd.DataFrame, test_df: pd.DataFrame, time_col: str, target_col: str):
    features = _feature_cols(train_df, time_col, target_col)
    if not features:
        raise ValueError("No numeric features available for Random Forest.")
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(train_df[features], train_df[target_col])
    pred = model.predict(test_df[features])
    importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    return pred, importance


def run_xgboost(train_df: pd.DataFrame, test_df: pd.DataFrame, time_col: str, target_col: str):
    features = _feature_cols(train_df, time_col, target_col)
    if not features:
        raise ValueError("No numeric features available for XGBoost.")
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(train_df[features], train_df[target_col])
    pred = model.predict(test_df[features])
    importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    return pred, importance


def run_model(df: pd.DataFrame, model_name: str, time_col: str, target_col: str, test_size: float = 0.2) -> Dict[str, object]:
    train_df, test_df = train_test_split_time(df, test_size=test_size)
    result: Dict[str, object] = {"train": train_df, "test": test_df}

    if model_name == "ARIMA":
        pred, ci = run_arima(train_df, test_df, target_col)
        result["feature_importance"] = None
    elif model_name == "Prophet":
        pred, ci = run_prophet(train_df, test_df, time_col, target_col)
        result["feature_importance"] = None
    elif model_name == "Random Forest":
        pred, importance = run_random_forest(train_df, test_df, time_col, target_col)
        ci = None
        result["feature_importance"] = importance
    elif model_name == "XGBoost":
        pred, importance = run_xgboost(train_df, test_df, time_col, target_col)
        ci = None
        result["feature_importance"] = importance
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    pred = np.maximum(pred, 0)
    result["predictions"] = pred
    result["confidence_intervals"] = ci
    result["metrics"] = regression_metrics(test_df[target_col].values, pred)
    return result
