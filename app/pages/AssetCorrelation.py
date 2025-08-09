# just display asset correlations
import streamlit as st

from utils import get_tickers_w_desc, correlation_matrix

selected_assets = st.sidebar.multiselect("Asset", options=get_tickers_w_desc())
assets = [x.split(" - ")[0] for x in list(selected_assets)]
correlation_matrix(assets=assets)
