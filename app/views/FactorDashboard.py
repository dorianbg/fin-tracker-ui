"""
Factor Performance Dashboard — compare factor ETFs (value, momentum, quality, size, min-vol)
across multiple return periods to see which factors are currently leading/lagging.
"""

import streamlit as st
import plotly.express as px

from data import add_sparkline_column, load_latest_perf
from duckdb_importer import RETURN_LABELS, RETURN_PERIODS


def render():
    st.title("📊 Factor Performance Dashboard")

    FACTOR_GROUPS = {
        "Value": ["IWVL", "IUVL", "IEVL"],
        "Quality": ["IWQU", "IUQA"],
        "Momentum": ["IWMO", "IUMF"],
        "Min Volatility": ["MVOL"],
        "Size (Small)": ["IEFS"],
        "Multi-Factor": ["IFSW"],
        "Growth": ["R1GB"],
    }

    BENCHMARKS = {
        "FTSE All-World": "VWRP",
        "S&P 500": "CSP1",
        "MSCI Europe": "IMEA",
    }

    ALL_FACTOR_TICKERS = [t for tickers in FACTOR_GROUPS.values() for t in tickers]
    ALL_BENCHMARK_TICKERS = list(BENCHMARKS.values())
    ALL_TICKERS = ALL_FACTOR_TICKERS + ALL_BENCHMARK_TICKERS

    # Skip 1D — too noisy for factor analysis
    periods = RETURN_PERIODS[1:]  # r_1w onwards
    labels = RETURN_LABELS[1:]  # 1W onwards

    df = load_latest_perf(tickers=tuple(ALL_TICKERS))

    if df.empty:
        st.warning(
            "No factor data found. Check that factor ETFs are in instrument_info.csv."
        )
        st.stop()

    ticker_to_factor = {}
    for factor, tickers in FACTOR_GROUPS.items():
        for t in tickers:
            ticker_to_factor[t] = factor
    for label, t in BENCHMARKS.items():
        ticker_to_factor[t] = f"Benchmark ({label})"

    df["factor"] = df["ticker"].map(ticker_to_factor)

    # Streamlit sidebars are global, so keep tab-specific controls in-page.
    settings_col, content_col = st.columns([1, 4], gap="large")

    with settings_col:
        st.subheader("Factor Settings")
        period = st.selectbox(
            "Return period", options=labels, index=5, key="factor_return_period"
        )
        period_col = periods[labels.index(period)]

    with content_col:
        st.header("Returns Heatmap")
        st.markdown(
            "Each row is a factor ETF, columns are return periods. Colour: green = positive, red = negative."
        )

        heatmap_data = df[["description", "ticker", *periods]].copy()
        add_sparkline_column(heatmap_data)
        add_sparkline_column(heatmap_data, col_name="Price (1y)", days=365)
        heatmap_data = heatmap_data.set_index("description")
        heatmap_data = heatmap_data[["Price (90d)", "Price (1y)", *periods]]
        heatmap_data.columns = ["Price (90d)", "Price (1y)", *labels]
        default_sort = "1Y" if "1Y" in labels else labels[-2]
        heatmap_data = heatmap_data.sort_values(default_sort, ascending=False)

        st.dataframe(
            heatmap_data.style.format("{:+.2f}%", subset=labels).background_gradient(
                cmap="RdYlGn", axis=None, vmin=-20, vmax=20, subset=labels
            ),
            column_config={
                "Price (90d)": st.column_config.LineChartColumn(
                    "Price (90d)", width="small"
                ),
                "Price (1y)": st.column_config.LineChartColumn(
                    "Price (1y)", width="small"
                ),
            },
            height=max(300, len(heatmap_data) * 35 + 38),
            width="stretch",
        )

        st.header("Factor Comparison")

        comparison = df[["description", "ticker", "factor", period_col]].copy()
        comparison = comparison.sort_values(period_col, ascending=True).reset_index(
            drop=True
        )

        fig_bar = px.bar(
            comparison,
            x=period_col,
            y="description",
            color="factor",
            orientation="h",
            labels={period_col: f"Return ({period}) %", "description": ""},
            title=f"Factor ETF Returns — {period}",
        )
        fig_bar.update_layout(
            yaxis=dict(autorange="reversed"), height=max(350, len(comparison) * 28)
        )
        st.plotly_chart(fig_bar, width="stretch")

        st.header("Factor Spread")
        st.markdown(
            "Average return per factor group across periods — shows which factor style is leading."
        )

        factor_avg = df.groupby("factor")[periods].mean()
        factor_avg.columns = labels

        fig_spread = px.imshow(
            factor_avg.values,
            x=labels,
            y=list(factor_avg.index),
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            text_auto=".1f",
            aspect="auto",
            labels=dict(color="Avg Return %"),
            title="Average Return by Factor Group",
        )
        fig_spread.update_layout(height=max(300, len(factor_avg) * 40))
        st.plotly_chart(fig_spread, width="stretch")

        st.dataframe(
            factor_avg.style.format("{:+.2f}%"),
            height=300,
        )
