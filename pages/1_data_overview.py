import streamlit as st

from utils.pipeline import load_and_prepare_data


st.title("Data Overview")

try:
    raw_df, cleaned_df, _, report, keys, dataset_path = load_and_prepare_data()
except Exception as exc:
    st.error(f"Failed to load/clean dataset: {exc}")
    st.stop()

st.success(f"Dataset loaded from: `{dataset_path}`")

col1, col2 = st.columns(2)
col1.metric("Raw Shape", f"{raw_df.shape[0]} x {raw_df.shape[1]}")
col2.metric("Clean Shape", f"{cleaned_df.shape[0]} x {cleaned_df.shape[1]}")

st.subheader("Dataset Preview (Cleaned)")
st.dataframe(cleaned_df.head(50), use_container_width=True)

st.subheader("Schema")
st.dataframe(cleaned_df.dtypes.astype(str).rename("dtype").reset_index().rename(columns={"index": "column"}))

st.subheader("Missing Values Summary")
missing_df = cleaned_df.isna().sum().rename("missing_count").reset_index().rename(columns={"index": "column"})
st.dataframe(missing_df, use_container_width=True)

st.subheader("Detected Key Columns")
st.json(keys)

st.subheader("Cleaning Report")
for step in report.get("steps", []):
    st.write(f"- {step}")

st.download_button(
    "Download Cleaned Dataset",
    data=cleaned_df.to_csv(index=False).encode("utf-8"),
    file_name="cleaned_retail_data.csv",
    mime="text/csv",
)
