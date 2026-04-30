import streamlit as st


def global_filters(df, time_col: str, product_col: str = "", category_col: str = ""):
    st.sidebar.header("Global Filters")
    min_date = df[time_col].min().date()
    max_date = df[time_col].max().date()
    date_range = st.sidebar.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    products = ["All"]
    if product_col and product_col in df.columns:
        products += sorted(df[product_col].astype(str).unique().tolist())
    selected_product = st.sidebar.selectbox("Product", products)

    categories = ["All"]
    if category_col and category_col in df.columns:
        categories += sorted(df[category_col].astype(str).unique().tolist())
    selected_category = st.sidebar.selectbox("Category", categories)

    return {"date_range": date_range, "product": selected_product, "category": selected_category}
