"""
Factor Performance Dashboard — compare factor ETFs (value, momentum, quality, size, min-vol)
across multiple return periods to see which factors are currently leading/lagging.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

import duckdb_importer as di
from data import get_conn

st.title("📊 Factor Performance Dashboard")

# ── Factor ETF definitions ──
# Grouped by factor, with World / USA / Europe variants where available
FACTOR_GROUPS = {
    "Value": ["IWVL", "IUVL", "IEVL"],
    "Quality": ["IWQU", "IUQA"],
    "Momentum": ["IWMO", "IUMF"],
    "Min Volatility": ["MVOL"],
    "Size (Small)": ["IEFS"],
    "Multi-Factor": ["IFSW"],
    "Growth": ["R1GB"],
}

# Broad market benchmarks for context
BENCHMARKS = {
    "S&P 500": "CSP1",
    "MSCI Europe": "IMEA",
}

ALL_FACTOR_TICKERS = [t for tickers in FACTOR_GROUPS.values() for t in tickers]
ALL_BENCHMARK_TICKERS = list(BENCHMARKS.values())
ALL_TICKERS = ALL_FACTOR_TICKERS + ALL_BENCHMARK_TICKERS

RETURN_PERIODS = [
    "r_1w",
    "r_2w",
    "r_1mo",
    "r_3mo",
    "r_6mo",
    "r_1y",
    "r_2y",
    "r_3y",
    "r_5y",
]
RETURN_LABELS = ["1W", "2W", "1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y"]


@st.cache_data(ttl=300)
def load_factor_data() -> pd.DataFrame:
    cols = (
        di.perf_desc_cols_start
        + di.perf_mavg_cols
        + di.perf_returns_cols
        + di.perf_desc_cols_end
        + di.perf_rownames_cols
    )
    tickers_str = "','".join(ALL_TICKERS)
    query = f"""
        SELECT {",".join(cols)}
        FROM {di.perf_tbl}
        WHERE rown = 1 AND ticker IN ('{tickers_str}')
        ORDER BY description ASC
    """
    return get_conn().execute(query).df()


df = load_factor_data()

if df.empty:
    st.warning(
        "No factor data found. Check that factor ETFs are in instrument_info.csv."
    )
    st.stop()

# ── assign factor labels ──
ticker_to_factor = {}
for factor, tickers in FACTOR_GROUPS.items():
    for t in tickers:
        ticker_to_factor[t] = factor
for label, t in BENCHMARKS.items():
    ticker_to_factor[t] = f"Benchmark ({label})"

df["factor"] = df["ticker"].map(ticker_to_factor)


# ═══════════════════════════════════════════
# Section 1: Return Heatmap
# ═══════════════════════════════════════════
st.header("Returns Heatmap")
st.markdown(
    "Each row is a factor ETF, columns are return periods. Colour: green = positive, red = negative."
)

heatmap_data = df.set_index("description")[RETURN_PERIODS].copy()
heatmap_data.columns = RETURN_LABELS

fig_heatmap = px.imshow(
    heatmap_data.values,
    x=RETURN_LABELS,
    y=list(heatmap_data.index),
    color_continuous_scale="RdYlGn",
    color_continuous_midpoint=0,
    text_auto=".1f",
    aspect="auto",
    labels=dict(color="Return %"),
)
fig_heatmap.update_layout(height=max(400, len(heatmap_data) * 30))
st.plotly_chart(fig_heatmap, use_container_width=True)


# ═══════════════════════════════════════════
# Section 2: Factor vs Factor Comparison
# ═══════════════════════════════════════════
st.header("Factor Comparison")

period = st.selectbox("Return period", options=RETURN_LABELS, index=5)
period_col = RETURN_PERIODS[RETURN_LABELS.index(period)]

comparison = df[["description", "ticker", "factor", period_col]].copy()
comparison = comparison.sort_values(period_col, ascending=True).reset_index(drop=True)

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
st.plotly_chart(fig_bar, use_container_width=True)


# ═══════════════════════════════════════════
# Section 3: Factor Spread (best factor - worst factor)
# ═══════════════════════════════════════════
st.header("Factor Spread")
st.markdown(
    "Average return per factor group across periods — shows which factor style is leading."
)

# average returns by factor group
factor_avg = df.groupby("factor")[RETURN_PERIODS].mean()
factor_avg.columns = RETURN_LABELS

fig_spread = px.imshow(
    factor_avg.values,
    x=RETURN_LABELS,
    y=list(factor_avg.index),
    color_continuous_scale="RdYlGn",
    color_continuous_midpoint=0,
    text_auto=".1f",
    aspect="auto",
    labels=dict(color="Avg Return %"),
    title="Average Return by Factor Group",
)
fig_spread.update_layout(height=max(300, len(factor_avg) * 40))
st.plotly_chart(fig_spread, use_container_width=True)

# table view
st.dataframe(
    factor_avg.style.format("{:+.2f}%"),
    height=300,
)
