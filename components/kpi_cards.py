import streamlit as st


def render_kpis(kpi_map):
    cols = st.columns(len(kpi_map))
    for idx, (label, value) in enumerate(kpi_map.items()):
        cols[idx].metric(label, value)
