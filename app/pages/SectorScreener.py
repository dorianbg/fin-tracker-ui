"""
Sector Screener — identify oversold sectors and momentum recovery opportunities.

Signals:
  1) Oversold: instruments trading well below their 252-day moving average.
  2) Relative underperformers now recovering: instruments that lagged a benchmark
     over a long lookback but are outperforming over a short lookback.
  3) Recovery score ranking: composite of (1) and (2).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

import config
import duckdb_importer as di
from data import get_conn

st.title("🔍 Sector Screener")

# ── mapping from selectable return period to column name ──
RETURN_COL_MAP = {
    "1w": "r_1w",
    "2w": "r_2w",
    "1mo": "r_1mo",
    "3mo": "r_3mo",
    "6mo": "r_6mo",
    "1y": "r_1y",
    "2y": "r_2y",
    "3y": "r_3y",
    "5y": "r_5y",
}


# ── cache the heavy data load once ──
@st.cache_data(ttl=300)
def load_all_latest_performance() -> pd.DataFrame:
    """Load latest-day performance row for every instrument (all fund types)."""
    cols = (
        di.perf_desc_cols_start
        + di.perf_vol_cols
        + di.perf_mavg_cols
        + di.perf_returns_cols
        + di.perf_desc_cols_end
        + di.perf_rownames_cols
    )
    query = f"""
        SELECT {",".join(cols)}
        FROM {di.perf_tbl}
        WHERE rown = 1
        ORDER BY description ASC
    """
    return get_conn().execute(query).df()


# Load once — all downstream filtering is in pandas (fast)
_all_data = load_all_latest_performance()

if _all_data.empty:
    st.warning("No performance data loaded. Make sure the DuckDB importer has run.")
    st.stop()


# ── sidebar controls ──
st.sidebar.header("Screener Settings")

benchmark_label = st.sidebar.selectbox(
    "Benchmark",
    options=list(config.BENCHMARKS.keys()),
    index=0,
)
benchmark_ticker = config.BENCHMARKS[benchmark_label]

underperf_threshold = st.sidebar.slider(
    "Underperformance threshold (%)",
    min_value=5,
    max_value=50,
    value=10,
    step=5,
    help="How much worse than benchmark over the long lookback to qualify",
)

long_lookback = st.sidebar.selectbox(
    "Long lookback (underperformance period)",
    options=["6mo", "1y", "2y", "3y", "5y"],
    index=1,
)

short_lookback = st.sidebar.selectbox(
    "Short lookback (recovery period)",
    options=["1w", "2w", "1mo", "3mo", "6mo"],
    index=3,
)

fund_type_filter = st.sidebar.multiselect(
    "Fund types",
    options=[
        "eq",
        "eq-reit",
        "commod",
        "bonds",
        "bonds-em",
        "bonds-corp",
        "bonds-il",
        "bonds-cash",
    ],
    default=["eq"],
)

# ── filter data in pandas (instant) ──
if fund_type_filter:
    pattern = "^(" + "|".join(fund_type_filter) + ")"
    df = _all_data[_all_data["fund_type"].str.match(pattern)].copy()
else:
    df = _all_data.copy()

# ── get benchmark returns ──
long_col = RETURN_COL_MAP[long_lookback]
short_col = RETURN_COL_MAP[short_lookback]

benchmark_row = _all_data[_all_data["ticker"] == benchmark_ticker]
if benchmark_row.empty:
    st.error(f"Benchmark ticker '{benchmark_ticker}' not found in data.")
    st.stop()

benchmark_r_long = float(benchmark_row[long_col].iloc[0])
benchmark_r_short = float(benchmark_row[short_col].iloc[0])
benchmark_desc = benchmark_row["description"].iloc[0]

st.caption(
    f"Benchmark: **{benchmark_desc}** ({benchmark_ticker}) — "
    f"{long_lookback} return: {benchmark_r_long:+.2f}% · "
    f"{short_lookback} return: {benchmark_r_short:+.2f}%"
)

# ═══════════════════════════════════════════
# Section 1: Oversold Sectors
# ═══════════════════════════════════════════
st.header("📉 Oversold Sectors")
st.markdown("Instruments trading furthest below their **252-day moving average**.")

oversold = (
    df[["description", "ticker", "fund_type", "ma_21", "ma_63", "ma_126", "ma_252"]]
    .sort_values("ma_252", ascending=True)
    .head(25)
    .reset_index(drop=True)
)

col_left, col_right = st.columns([3, 4])

with col_left:
    # simple format, no heavy gradient styling
    st.dataframe(
        oversold.style.format(
            subset=["ma_21", "ma_63", "ma_126", "ma_252"], formatter="{:+.2f}%"
        ),
        hide_index=True,
        height=550,
    )

with col_right:
    fig_oversold = px.bar(
        oversold.head(20),
        x="ma_252",
        y="description",
        orientation="h",
        color="ma_252",
        color_continuous_scale="RdYlGn",
        range_color=[-30, 10],
        labels={"ma_252": "% vs 252d MA", "description": ""},
        title="Top 20 — Distance from 252-day Moving Average",
    )
    fig_oversold.update_layout(
        yaxis=dict(autorange="reversed"), height=550, showlegend=False
    )
    st.plotly_chart(fig_oversold, use_container_width=True)


# ═══════════════════════════════════════════
# Section 2: Underperformers Now Recovering
# ═══════════════════════════════════════════
st.header("🔄 Underperformers Now Recovering")
st.markdown(
    f"Instruments that **underperformed** the benchmark by ≥{underperf_threshold}% "
    f"over **{long_lookback}** but are showing relative strength over **{short_lookback}**."
)

# vectorised pandas — no loops
recovery = df[df["ticker"] != benchmark_ticker][
    ["description", "ticker", "fund_type", long_col, short_col]
].copy()
recovery["excess_long"] = recovery[long_col] - benchmark_r_long
recovery["excess_short"] = recovery[short_col] - benchmark_r_short
underperformers = (
    recovery[recovery["excess_long"] <= -underperf_threshold]
    .sort_values("excess_short", ascending=False)
    .reset_index(drop=True)
)

if underperformers.empty:
    st.info(
        f"No instruments underperformed the benchmark by ≥{underperf_threshold}% over {long_lookback}."
    )
else:
    underperformers["recovering"] = underperformers["excess_short"] > 0

    col_left2, col_right2 = st.columns([3, 4])

    with col_left2:
        display_cols = [
            "description",
            "ticker",
            long_col,
            short_col,
            "excess_long",
            "excess_short",
            "recovering",
        ]
        st.dataframe(
            underperformers[display_cols].style.format(
                subset=[long_col, short_col, "excess_long", "excess_short"],
                formatter="{:+.2f}%",
            ),
            hide_index=True,
            height=450,
        )

    with col_right2:
        fig_scatter = px.scatter(
            underperformers,
            x="excess_long",
            y="excess_short",
            color="recovering",
            color_discrete_map={True: "#2ecc71", False: "#e74c3c"},
            text="ticker",
            labels={
                "excess_long": f"Excess return ({long_lookback})",
                "excess_short": f"Excess return ({short_lookback})",
            },
            title="Long-term Pain vs Short-term Recovery",
        )
        fig_scatter.update_traces(textposition="top center", marker=dict(size=10))
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
        fig_scatter.add_vline(
            x=-underperf_threshold, line_dash="dash", line_color="grey", opacity=0.5
        )
        fig_scatter.update_layout(height=450)
        st.plotly_chart(fig_scatter, use_container_width=True)


# ═══════════════════════════════════════════
# Section 3: Recovery Score Ranking
# ═══════════════════════════════════════════
st.header("🏆 Recovery Score Ranking")
st.markdown(
    "**Deeper past underperformance × stronger recent recovery** = higher score. "
    "Only instruments currently outperforming the benchmark short-term are shown."
)

if not underperformers.empty:
    ranked = underperformers[underperformers["excess_short"] > 0].copy()
    if ranked.empty:
        st.info(
            "No instruments are currently recovering (none have positive excess short-term returns)."
        )
    else:
        ranked["recovery_score"] = (-ranked["excess_long"]) * ranked["excess_short"]
        ranked = ranked.sort_values("recovery_score", ascending=False).reset_index(
            drop=True
        )
        ranked.index = ranked.index + 1
        ranked.index.name = "Rank"

        display_ranked = ranked[
            [
                "description",
                "ticker",
                long_col,
                short_col,
                "excess_long",
                "excess_short",
                "recovery_score",
            ]
        ]
        st.dataframe(
            display_ranked.style.format(
                subset=[long_col, short_col, "excess_long", "excess_short"],
                formatter="{:+.2f}%",
            ).format(subset=["recovery_score"], formatter="{:.1f}"),
            height=350,
        )

        top_n = ranked.head(15)
        fig_rank = px.bar(
            top_n,
            x="recovery_score",
            y="description",
            orientation="h",
            color="recovery_score",
            color_continuous_scale="YlGn",
            labels={"recovery_score": "Recovery Score", "description": ""},
            title="Top Recovery Candidates",
        )
        fig_rank.update_layout(
            yaxis=dict(autorange="reversed"), height=400, showlegend=False
        )
        st.plotly_chart(fig_rank, use_container_width=True)
else:
    st.info("No underperforming instruments to rank.")
