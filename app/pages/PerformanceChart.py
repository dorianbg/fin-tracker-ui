import datetime
from typing import Optional

import streamlit as st

from app.utils import (
    plot_performance,
    deduct_datetime_interval,
)
from config import time_strings
from data import get_distinct_instruments, get_distinct_fund_types, get_min_date_all

col1, col2, col3 = st.columns(3)
min_date_possible: Optional[datetime.date] = None
max_date_possible: Optional[datetime.date] = None

if min_date_possible is None and max_date_possible is None:
    min_date_possible = get_min_date_all()

with col1:
    selected_lookback = st.selectbox(
        label="Lookback period (overrides date range)",
        options=[None] + time_strings,
        index=4,
    )

with col2:
    start_date: datetime.date = st.date_input(
        "Select start date",
        value=min_date_possible,
        min_value=min_date_possible,
        max_value=datetime.date.today(),
        format="DD/MM/YYYY",
    )
with col3:
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

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        selected_inst: list[str] = st.multiselect(
            label="Instrument", options=get_distinct_instruments(), default=None
        )
    with col2:
        selected_fund_types: list[str] = st.multiselect(
            label="Asset class", options=get_distinct_fund_types(), default=None
        )

    plot_performance(
        start_date, end_date, selected_inst, selected_fund_types, show_df=True
    )
