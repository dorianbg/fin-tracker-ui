import streamlit as st

from utils import get_tickers_w_desc, correlation_matrix


def render():
    # Streamlit sidebars are global, so keep tab-specific controls in-page.
    settings_col, content_col = st.columns([1, 4], gap="large")

    with settings_col:
        st.subheader("Correlation Settings")
        selected_assets = st.multiselect("Asset", options=get_tickers_w_desc())

    with content_col:
        assets = [x.split(" - ")[0] for x in list(selected_assets)]
        correlation_matrix(assets=assets)
