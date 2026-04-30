import pandas as pd
import streamlit as st

from utils.data_cleaning import clean_dataset, detect_key_columns, find_dataset_path, load_dataset
from utils.feature_engineering import create_lag_features, create_time_features


@st.cache_data(show_spinner=True)
def load_and_prepare_data():
    path = find_dataset_path("data")
    raw = load_dataset(path)
    cleaned, report = clean_dataset(raw)
    keys = detect_key_columns(cleaned)

    if not keys["time_col"] or not keys["target_col"]:
        raise ValueError("Unable to detect required time or target columns after cleaning.")

    enriched = create_time_features(cleaned, keys["time_col"])
    enriched = create_lag_features(enriched, keys["target_col"], keys.get("product_col", ""))
    return raw, cleaned, enriched, report, keys, str(path)


def apply_filters(df: pd.DataFrame, filters: dict, time_col: str, product_col: str = "", category_col: str = ""):
    out = df.copy()

    if isinstance(filters.get("date_range"), tuple) and len(filters["date_range"]) == 2:
        start, end = filters["date_range"]
        out = out[(out[time_col].dt.date >= start) & (out[time_col].dt.date <= end)]

    if product_col and product_col in out.columns and filters.get("product") not in [None, "All"]:
        out = out[out[product_col].astype(str) == str(filters["product"])]

    if category_col and category_col in out.columns and filters.get("category") not in [None, "All"]:
        out = out[out[category_col].astype(str) == str(filters["category"])]

    return out
