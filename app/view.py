import datetime
import platform
import time
from typing import Optional

import pandas as pd
import streamlit as st

import duckdb_importer as di
from app.config import table_height, time_strings
from app.data import (
    get_distinct_instruments,
    get_distinct_fund_types,
    get_data,
    get_min_date_all,
    get_fund_types,
    create_query,
)
from app.utils import (
    deduct_datetime_interval,
    custom_sort_df_cols,
    style_performance_table,
    plot_performance,
    filter_dataframe,
)

st.set_page_config(
    page_icon="🏠", page_title="Financial instrument tracker", layout="wide"
)


@st.cache_resource
def run_import_one_off():
    if platform.system() == "Darwin":
        with st.spinner("Processing"):
            di.run()
            time.sleep(1)


run_import_one_off()

st.title("Fin tracker")

top_col1, top_col2 = st.columns([4, 1])
with top_col1:
    st.write(
        "Disclaimer: this is a non-commercial project and data is purely source from Yahoo! finance API and exclusively intended for personal use only.  \n"
        "Data quality issues with smaller UCITS ETFs are common in which case the data in tables will be missing or obviously wrong."
    )
with top_col2:
    with st.popover("Things to note"):
        st.markdown(
            "1) Performance includes dividends (Accumulating ETFs are preferred where possible) and is standardised to GBP (some ETFs are GBP hedged).   \n"
            "2) UK cash rate is taken as risk free rate for Sharpe ratio   \n"
            "3) You can change the selection of instruments on the project that feeds data to this dashboard, source code is: https://github.com/dorianbg/fin-tracker/"
        )


st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 3rem;
                    padding-bottom: 1rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["Performance table", "Performance chart"])

with tab1:
    selected_fund_types = st.multiselect(
        label="Fund types",
        options=get_fund_types(),
        default=get_fund_types(),
    )

    col1, col2, col3 = st.columns([2, 2, 7])

    with col1:
        with st.container():
            vol_adjust = st.toggle(label="Show Sharpe ratio", value=True)
            show_returns = st.toggle(label="Show Gross return", value=True)

    with col2:
        with st.container():
            sort_sharpe = st.toggle(label="Custom sort on Sharpe", value=False)
            sort_returns = st.toggle(label="Custom sort on Returns", value=False)
            if sort_sharpe and sort_returns:
                st.warning("Cannot enable custom sorting on both")
    with col3:
        returns_cols = st.multiselect(
            label="Returns",
            options=di.selectable_returns,
            default=di.default_selected_returns,
        )

    custom_weights = []
    if sort_sharpe or sort_returns:
        weight_cols = st.columns(len(di.selectable_returns))
        for i, weight_col in enumerate(weight_cols):
            with weight_col:
                custom_weights.append(
                    st.number_input(
                        f"Weight for {di.selectable_returns[i]}",
                        value=0,
                    )
                )
        if (
            len(custom_weights)
            and sum(custom_weights) > 0
            and sum(custom_weights) != 100
        ):
            st.warning(
                f"Custom weights must add up to 100% - current is {sum(custom_weights)}%"
            )

    with st.container():
        modify = st.checkbox("Add filters")
        df: pd.DataFrame = get_data(
            query=create_query(
                table=di.perf_tbl,
                fund_types=selected_fund_types,
                vol_adjust=vol_adjust,
                show_returns=show_returns,
                returns_cols=returns_cols,
            ),
        )
        if modify:
            df = filter_dataframe(df, modify=True)
        else:
            if (
                (sort_sharpe or sort_returns)
                and sum(custom_weights) == 100
                and filter(lambda x: x >= 1, custom_weights)
            ):
                total_w = sum(custom_weights)
                custom_weights_normalised = [x / total_w for x in custom_weights]
                columns_sort = (
                    di.perf_sharpe_cols if sort_sharpe else di.perf_returns_cols
                )
                df = custom_sort_df_cols(columns_sort, custom_weights_normalised, df)
        styled_df = style_performance_table(
            df,
            vol_adjust=vol_adjust,
            show_returns=show_returns,
            returns_cols=returns_cols,
        )
        event = st.dataframe(
            data=styled_df,
            hide_index=True,
            height=table_height,
            use_container_width=True,
            on_select="rerun",
            selection_mode="multi-row",
        )
        if event:
            st.text("Price performance")
            filtered_df = df.iloc[event.selection.rows]

            plot_performance(
                start_date=datetime.date.today() - datetime.timedelta(days=3 * 365),
                end_date=datetime.date.today(),
                selected_inst=list(filtered_df["ticker"].unique()),
                selected_fund_types=[],
                show_df=True,
            )

with tab2:
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
