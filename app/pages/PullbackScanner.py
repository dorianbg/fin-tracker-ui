"""
Pullback Scanner — find instruments experiencing pullbacks or recoveries.

Modes:
  1) Classic Pullback: instruments in uptrends pulling back to shorter MAs
  2) Underperformers Recovering: lagged instruments now showing short-term strength
  3) Recovery Score: composite ranking of recovery candidates
"""

import streamlit as st
import pandas as pd
import plotly.express as px

import config
import duckdb_importer as di
from data import get_conn

st.title("🎯 Pullback Scanner")
st.markdown(
    "Find **pullback opportunities** in uptrending instruments and "
    "**recovery candidates** among recent underperformers."
)

# ── sidebar controls ──
st.sidebar.header("Scanner Settings")

# Pullback MA selector
pullback_ma = st.sidebar.selectbox(
    "Pullback detection MA",
    options=["21-day (short)", "63-day (medium)", "126-day (intermediate)"],
    index=0,
    help="Which moving average to use for detecting pullbacks",
)
pullback_ma_col = {
    "21-day (short)": "ma_21",
    "63-day (medium)": "ma_63",
    "126-day (intermediate)": "ma_126",
}[pullback_ma]
pullback_ma_label = pullback_ma.split(" ")[0]

pullback_depth = st.sidebar.slider(
    f"Max {pullback_ma_label} MA deviation (%)",
    min_value=-20,
    max_value=5,
    value=0,
    step=1,
    help=f"How far below the {pullback_ma_label} MA the price must be (0 = any pullback below MA)",
)

min_uptrend_strength = st.sidebar.slider(
    "Min 252-day MA position (%)",
    min_value=-5,
    max_value=30,
    value=0,
    step=1,
    help="How far above the 252-day MA — 0 means just above it",
)

require_intermediate_ok = st.sidebar.checkbox(
    "Require above 126-day MA",
    value=False,
    help="Extra filter: intermediate trend (126-day) must still be intact",
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

# Recovery settings
st.sidebar.markdown("---")
st.sidebar.header("Recovery Settings")

RETURN_COL_MAP = {
    "1w": "r_1w",
    "2w": "r_2w",
    "1mo": "r_1mo",
    "3mo": "r_3mo",
    "6mo": "r_6mo",
    "1y": "r_1y",
    "2y": "r_2y",
    "3y": "r_3y",
}

benchmark_label = st.sidebar.selectbox(
    "Benchmark",
    options=list(config.BENCHMARKS.keys()),
    index=0,
)
benchmark_ticker = config.BENCHMARKS[benchmark_label]

long_lookback = st.sidebar.selectbox(
    "Underperformance period",
    options=["3mo", "6mo", "1y", "2y"],
    index=1,
)

short_lookback = st.sidebar.selectbox(
    "Recovery period",
    options=["1w", "2w", "1mo", "3mo"],
    index=2,
)

underperf_threshold = st.sidebar.slider(
    "Underperformance threshold (%)",
    min_value=0,
    max_value=40,
    value=5,
    step=5,
    help="How much worse than benchmark to qualify as underperformer",
)


@st.cache_data(ttl=300)
def load_all_latest() -> pd.DataFrame:
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


_all_data = load_all_latest()

if _all_data.empty:
    st.warning("No data loaded.")
    st.stop()

# ── filter by fund type ──
if fund_type_filter:
    pattern = "^(" + "|".join(fund_type_filter) + ")"
    df = _all_data[_all_data["fund_type"].str.match(pattern)].copy()
else:
    df = _all_data.copy()


# ═══════════════════════════════════════════
# Section 1: Pullback Candidates
# ═══════════════════════════════════════════
st.header("📋 Pullback Candidates")
st.markdown(
    f"Instruments **above 252-day MA** by ≥{min_uptrend_strength}% and "
    f"**pulling back** below {pullback_ma_label} MA by ≤{pullback_depth}%."
)

# strong uptrend: above 252-day MA
pullbacks = df[df["ma_252"] >= min_uptrend_strength].copy()

# intermediate trend intact (optional)
if require_intermediate_ok:
    pullbacks = pullbacks[pullbacks["ma_126"] >= 0]

# pulling back: below selected MA by at least the threshold
pullbacks = pullbacks[pullbacks[pullback_ma_col] <= pullback_depth]

if pullbacks.empty:
    st.info(
        "No instruments match current filters. Try relaxing the thresholds in the sidebar."
    )
else:
    # sort by deepest pullback — biggest opportunity first
    pullbacks = pullbacks.sort_values(pullback_ma_col, ascending=True).reset_index(
        drop=True
    )

    # pullback quality score
    pullbacks["pullback_score"] = pullbacks["ma_252"] * (-pullbacks[pullback_ma_col])
    # bonus: if 1-week return is positive, the bounce may already be starting
    pullbacks["bounce_signal"] = pullbacks["r_1w"] > 0

    st.markdown(f"**{len(pullbacks)} candidates found**")

    display_cols = [
        "description",
        "ticker",
        "fund_type",
        pullback_ma_col,
        "ma_63",
        "ma_126",
        "ma_252",
        "drawdown_52w",  # Added as requested
        "r_1w",
        "r_1mo",
        "r_3mo",
        "pullback_score",
        "bounce_signal",
    ]
    # deduplicate in case pullback_ma_col is already one of the fixed cols
    display_cols = list(dict.fromkeys(display_cols))

    st.dataframe(
        pullbacks[display_cols]
        .style.format(
            subset=[
                c
                for c in [
                    pullback_ma_col,
                    "ma_63",
                    "ma_126",
                    "ma_252",
                    "drawdown_52w",
                    "r_1w",
                    "r_1mo",
                    "r_3mo",
                ]
                if c in display_cols
            ],
            formatter="{:+.2f}%",
        )
        .format(subset=["pullback_score"], formatter="{:.1f}"),
        hide_index=True,
        height=450,
    )

    # Scatter: uptrend strength vs pullback depth
    st.subheader("Uptrend Strength vs Pullback Depth")

    fig = px.scatter(
        pullbacks,
        x="ma_252",
        y=pullback_ma_col,
        color="bounce_signal",
        color_discrete_map={True: "#2ecc71", False: "#e67e22"},
        text="ticker",
        size="pullback_score",
        size_max=20,
        labels={
            "ma_252": "% above 252-day MA (uptrend strength)",
            pullback_ma_col: f"% from {pullback_ma_label} MA (pullback depth)",
            "bounce_signal": "1W bounce?",
            "drawdown_52w": "Drawdown from 52W High (%)",
        },
        hover_data=["drawdown_52w"],
        title="Sweet Spot: Strong Uptrend + Deep Pullback (top-left = best)",
    )
    fig.update_traces(textposition="top center")
    fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.3)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Top ranked by pullback score
    st.subheader("🏆 Top Pullback Candidates")
    st.markdown(
        "Ranked by **pullback score** = (uptrend strength) × (pullback depth). "
        "Bounce signal = positive 1-week return."
    )

    top = pullbacks.sort_values("pullback_score", ascending=False).head(15)
    fig_bar = px.bar(
        top,
        x="pullback_score",
        y="description",
        orientation="h",
        color="bounce_signal",
        color_discrete_map={True: "#2ecc71", False: "#e67e22"},
        labels={
            "pullback_score": "Pullback Score",
            "description": "",
            "bounce_signal": "1W bounce?",
        },
        title="Top 15 — Pullback Score",
    )
    fig_bar.update_layout(yaxis=dict(autorange="reversed"), height=450, showlegend=True)
    st.plotly_chart(fig_bar, use_container_width=True)


# ═══════════════════════════════════════════
# Section 2: Trend Reversal Candidates
# ═══════════════════════════════════════════
st.header("🔄 Trend Reversal Candidates")
st.markdown("""
Instruments currently **below** their 252-day moving average (Long-term downtrend) 
but trading **above** their 21-day and 63-day moving averages (Short-term recovery/Golden Cross setup).
""")

# Logic: Under 252 MA (< 0) but Above 21 MA (> 0) and Above 63 MA (> 0)
reversal_mask = (df["ma_252"] < 0) & (df["ma_21"] > 0) & (df["ma_63"] > 0)

df_reversal = df[reversal_mask].copy()

if not df_reversal.empty:
    # Sort by strength of short term recovery (ma_21)
    df_reversal = df_reversal.sort_values("ma_21", ascending=False)

    cols = [
        "ticker",
        "description",
        "price",
        "r_1w",
        "ma_21",
        "ma_63",
        "ma_252",
        "vol_1y",
    ]
    # ensure cols exist
    cols = [c for c in cols if c in df_reversal.columns]

    st.dataframe(
        df_reversal[cols].style.format(
            {
                "r_1w": "{:+.2f}%",
                "ma_21": "{:+.2f}%",
                "ma_63": "{:+.2f}%",
                "ma_252": "{:+.2f}%",
                "vol_1y": "{:.2f}%",
            },
            na_rep="-",
            precision=2,
        ),
        use_container_width=True,
        height=300,
    )
else:
    st.info(
        "No instruments found matching Reversal criteria (Under 252MA, Above 21MA & 63MA)."
    )

st.markdown("---")


# ═══════════════════════════════════════════
# Section 3: Underperformers Now Recovering
# ═══════════════════════════════════════════
st.header("📉 Underperformers Now Recovering")

long_col = RETURN_COL_MAP[long_lookback]
short_col = RETURN_COL_MAP[short_lookback]

benchmark_row = _all_data[_all_data["ticker"] == benchmark_ticker]
if benchmark_row.empty:
    st.error(f"Benchmark ticker '{benchmark_ticker}' not found.")
    st.stop()

benchmark_r_long = float(benchmark_row[long_col].iloc[0])
benchmark_r_short = float(benchmark_row[short_col].iloc[0])
benchmark_desc = benchmark_row["description"].iloc[0]

st.caption(
    f"Benchmark: **{benchmark_desc}** ({benchmark_ticker}) — "
    f"{long_lookback} return: {benchmark_r_long:+.2f}% · "
    f"{short_lookback} return: {benchmark_r_short:+.2f}%"
)

st.markdown(
    f"Instruments that **underperformed** the benchmark by ≥{underperf_threshold}% "
    f"over **{long_lookback}** but are showing relative strength over **{short_lookback}**."
)

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
        f"No instruments underperformed the benchmark by ≥{underperf_threshold}% over {long_lookback}. "
        f"Try lowering the threshold."
    )
else:
    underperformers["recovering"] = underperformers["excess_short"] > 0

    col_left, col_right = st.columns([3, 4])

    with col_left:
        display_cols_r = [
            "description",
            "ticker",
            long_col,
            short_col,
            "excess_long",
            "excess_short",
            "recovering",
        ]
        st.dataframe(
            underperformers[display_cols_r].style.format(
                subset=[long_col, short_col, "excess_long", "excess_short"],
                formatter="{:+.2f}%",
            ),
            hide_index=True,
            height=450,
        )

    with col_right:
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
    "Top candidates that are turning the corner."
)

if not underperformers.empty:
    # Show both currently recovering AND all underperformers with any positive short-term excess
    ranked = underperformers[underperformers["excess_short"] > 0].copy()

    if ranked.empty:
        # Fallback: show top underperformers by least-negative short-term excess (closest to recovering)
        st.info(
            "No instruments are currently outperforming the benchmark short-term. "
            "Showing the **closest to recovery** instead."
        )
        ranked = (
            underperformers.sort_values("excess_short", ascending=False).head(15).copy()
        )
        ranked["recovery_score"] = (-ranked["excess_long"]) * (
            1 + ranked["excess_short"]
        )
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
            height=400,
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
            height=400,
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
            yaxis=dict(autorange="reversed"), height=450, showlegend=False
        )
        st.plotly_chart(fig_rank, use_container_width=True)
else:
    st.info("No underperforming instruments to rank.")
