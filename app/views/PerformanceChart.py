import datetime

import streamlit as st

from utils import (
    plot_performance,
    deduct_datetime_interval,
)
from config import time_strings
from data import get_distinct_instruments, get_distinct_fund_types, get_min_date_all


def render():
    # Streamlit sidebars are global, so keep tab-specific controls in-page.
    settings_col, content_col = st.columns([1, 4], gap="large")

    with settings_col:
        st.subheader("Chart Settings")
        min_date_possible = get_min_date_all()

        selected_lookback = st.selectbox(
            label="Lookback period (overrides date range)",
            options=[None] + time_strings,
            index=4,
        )

        start_date: datetime.date = st.date_input(
            "Select start date",
            value=min_date_possible,
            min_value=min_date_possible,
            max_value=datetime.date.today(),
            format="DD/MM/YYYY",
        )
        end_date: datetime.date = st.date_input(
            "Select end date",
            value=datetime.date.today(),
            min_value=min_date_possible,
            max_value=datetime.date.today(),
            format="DD/MM/YYYY",
        )

        if selected_lookback is not None:
            end_date = datetime.date.today()
            start_date = deduct_datetime_interval(end_date, selected_lookback)

        selected_inst: list[str] = st.multiselect(
            label="Instrument", options=get_distinct_instruments(), default=None
        )
        selected_fund_types: list[str] = st.multiselect(
            label="Asset class", options=get_distinct_fund_types(), default=None
        )

    with content_col:
        plot_performance(
            start_date, end_date, selected_inst, selected_fund_types, show_df=True
        )
