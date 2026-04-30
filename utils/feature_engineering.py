import pandas as pd


def create_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(time_col)
    out["year"] = out[time_col].dt.year
    out["month"] = out[time_col].dt.month
    out["week"] = out[time_col].dt.isocalendar().week.astype(int)
    out["day"] = out[time_col].dt.day
    out["dayofweek"] = out[time_col].dt.dayofweek
    out["quarter"] = out[time_col].dt.quarter
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)
    return out


def create_lag_features(df: pd.DataFrame, target_col: str, group_col: str = "") -> pd.DataFrame:
    out = df.copy()
    lags = [1, 7, 14, 28]
    if group_col and group_col in out.columns:
        for lag in lags:
            out[f"lag_{lag}"] = out.groupby(group_col)[target_col].shift(lag)
            out[f"rolling_mean_{lag}"] = (
                out.groupby(group_col)[target_col].shift(1).rolling(lag, min_periods=1).mean()
            )
    else:
        for lag in lags:
            out[f"lag_{lag}"] = out[target_col].shift(lag)
            out[f"rolling_mean_{lag}"] = out[target_col].shift(1).rolling(lag, min_periods=1).mean()

    out = out.bfill().ffill()
    return out
