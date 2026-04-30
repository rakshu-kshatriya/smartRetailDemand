import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def normalize_column_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_column_name(c) for c in out.columns]
    return out


def find_dataset_path(data_dir: str = "data") -> Path:
    local_data = Path(data_dir)
    supported = [".csv", ".xlsx", ".xls", ".parquet", ".json"]
    if local_data.exists():
        files = [p for p in local_data.glob("*") if p.suffix.lower() in supported]
        if files:
            return sorted(files)[0]

    # Fallback: try nearby known source folder if workspace data/ is empty.
    fallback = Path.cwd().parent / "smartRetailDetail" / "data" / "sales_data.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("No dataset found in data/ or fallback path.")


def load_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def detect_key_columns(df: pd.DataFrame) -> Dict[str, str]:
    cols = df.columns.tolist()
    time_candidates = [c for c in cols if any(k in c for k in ["date", "time", "day", "month"])]
    target_candidates = [c for c in cols if any(k in c for k in ["sales", "demand", "quantity_sold", "qty", "units"])]
    product_candidates = [c for c in cols if any(k in c for k in ["product", "item", "sku"])]
    category_candidates = [c for c in cols if any(k in c for k in ["category", "segment", "dept"])]

    return {
        "time_col": time_candidates[0] if time_candidates else "",
        "target_col": target_candidates[0] if target_candidates else "",
        "product_col": product_candidates[0] if product_candidates else "",
        "category_col": category_candidates[0] if category_candidates else "",
    }


def _coerce_types(df: pd.DataFrame, time_col: str) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    logs: List[str] = []
    if time_col and time_col in out.columns:
        out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
        before = len(out)
        out = out.dropna(subset=[time_col])
        logs.append(f"Converted `{time_col}` to datetime and removed {before - len(out)} invalid rows.")

    for c in out.columns:
        if out[c].dtype == "object" and c != time_col:
            numeric = pd.to_numeric(out[c], errors="coerce")
            if numeric.notna().mean() > 0.9:
                out[c] = numeric
                logs.append(f"Converted `{c}` to numeric.")
            else:
                out[c] = out[c].astype(str).str.strip().str.title()
    return out, logs


def _handle_missing(df: pd.DataFrame, time_col: str) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    logs: List[str] = []
    missing_before = out.isna().sum().sum()

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    cat_cols = [c for c in out.columns if c not in numeric_cols and c != time_col]

    for c in numeric_cols:
        if out[c].isna().any():
            out[c] = out[c].fillna(out[c].median())
    for c in cat_cols:
        if out[c].isna().any():
            out[c] = out[c].fillna("Unknown")

    if time_col and out[time_col].isna().any():
        out = out.dropna(subset=[time_col])

    missing_after = out.isna().sum().sum()
    logs.append(f"Handled missing values: {int(missing_before)} -> {int(missing_after)}.")
    return out, logs


def _handle_outliers_iqr(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    logs: List[str] = []
    numeric_cols = out.select_dtypes(include=[np.number]).columns

    for c in numeric_cols:
        q1 = out[c].quantile(0.25)
        q3 = out[c].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        before = out[c].copy()
        out[c] = out[c].clip(lower, upper)
        changed = int((before != out[c]).sum())
        if changed > 0:
            logs.append(f"Clipped {changed} outliers in `{c}` using IQR bounds.")
    return out, logs


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    report: Dict[str, object] = {"steps": []}
    raw = normalize_columns(df)
    report["steps"].append("Normalized column names to lowercase snake_case.")

    keys = detect_key_columns(raw)
    report["detected_columns"] = keys

    dup_before = int(raw.duplicated().sum())
    raw = raw.drop_duplicates()
    report["steps"].append(f"Removed duplicate rows: {dup_before}.")

    raw, type_logs = _coerce_types(raw, keys["time_col"])
    report["steps"].extend(type_logs)

    raw, miss_logs = _handle_missing(raw, keys["time_col"])
    report["steps"].extend(miss_logs)

    raw, outlier_logs = _handle_outliers_iqr(raw)
    report["steps"].extend(outlier_logs)

    report["shape_after_cleaning"] = raw.shape
    report["missing_by_column"] = raw.isna().sum().to_dict()
    report["dtypes"] = {k: str(v) for k, v in raw.dtypes.items()}
    return raw, report
