"""
Puke Detector — identify capitulation / extreme stress events across instruments.

Strategy: buy when instruments experience extreme vol spikes, outsized moves,
and deep drawdowns from highs ("puking") — especially when early bounce signs appear.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

import duckdb_importer as di
from data import get_conn

st.set_page_config(page_title="Puke Detector", layout="wide")

st.title("🌋 Puke Detector")
st.markdown(
    "Find instruments in **extreme stress** — outsized moves, vol spikes, deep drawdowns from highs. "
    "These are potential capitulation events where buying into fear can be rewarding."
)

# ── Market Sentiment (Fear & Greed) ──
try:
    import fear_and_greed

    fg = fear_and_greed.get()
    val = fg.value
    desc = fg.description

    # color mapping
    color = "grey"
    if val < 25:
        color = "red"  # Extreme Fear
    elif val < 45:
        color = "orange"  # Fear
    elif val > 75:
        color = "green"  # Extreme Greed
    elif val > 55:
        color = "lightgreen"  # Greed

    st.markdown(
        f"""
        <div style="padding: 10px; border-radius: 5px; border: 1px solid #333; margin-bottom: 20px;">
            <h3 style="margin:0; text-align: center;">
                CNN Fear & Greed: <span style="color:{color}">{val:.0f} ({desc})</span>
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
except ImportError:
    pass  # fail silently if package not installed (though it should be)
except Exception as e:
    st.error(f"Could not load Fear & Greed: {e}")

# ── sidebar controls ──
st.sidebar.header("Puke Detection Settings")

z_score_threshold = st.sidebar.slider(
    "Z-score threshold (σ)",
    min_value=1.0,
    max_value=4.0,
    value=1.5,
    step=0.5,
    help="Minimum z-score to flag as extreme move (2σ = ~5% probability)",
)

vol_ratio_threshold = st.sidebar.slider(
    "Vol spike ratio threshold",
    min_value=1.0,
    max_value=3.0,
    value=1.2,
    step=0.1,
    help="Short-term vol / long-term vol — above this = stress regime",
)

drawdown_threshold = st.sidebar.slider(
    "Min drawdown from 52W high (%)",
    min_value=-60,
    max_value=-5,
    value=-15,
    step=5,
    help="How far below the 52-week high to qualify as capitulation",
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


@st.cache_data(ttl=300)
def load_all_latest() -> pd.DataFrame:
    cols = (
        di.perf_desc_cols_start
        + di.perf_z_score_cols
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

# computed columns
# computed columns
df["vol_ratio"] = df["vol_1mo"] / df["vol_1y"].replace(0, np.nan)

# drawdown horizon selector
dd_horizon = st.sidebar.radio(
    "Drawdown Horizon",
    options=["52W High", "3Y High"],
    index=0,
    horizontal=True,
    help="Calculate drawdown from 52-week high or 3-year high",
)

# determine column based on selection
if dd_horizon == "3Y High":
    drawdown_col = "drawdown_3y"
    drawdown_label = "3Y High"
else:
    drawdown_col = "drawdown_52w"
    drawdown_label = "52W High"

# fallback if column missing
if drawdown_col not in df.columns:
    if "drawdown_52w" in df.columns and dd_horizon == "3Y High":
        st.warning(f"3Y High data not available yet. Using 52W High.")
        drawdown_col = "drawdown_52w"
        drawdown_label = "52W High"
    elif "ma_252" in df.columns:
        if (
            dd_horizon != "52W High"
        ):  # only warn if they explicitly asked for something else
            st.warning(f"{dd_horizon} data not available. Using 252d MA proxy.")
        drawdown_col = "ma_252"
        drawdown_label = "252d MA"


# ═══════════════════════════════════════════
# Section 1: Extreme Movers — Crashers vs Spikers
# ═══════════════════════════════════════════
st.header("⚡ Extreme Movers")
st.markdown(
    f"Instruments with z-scores **≥ {z_score_threshold}σ** — statistically unusual moves. "
    "Split into **crashers** (negative returns) and **spikers** (positive returns)."
)

z_cols_available = [c for c in ["z_1d", "z_1w", "z_2w", "z_1mo"] if c in df.columns]

if z_cols_available:
    # find rows where any z-score exceeds threshold
    z_mask = (
        df[z_cols_available].apply(lambda row: row.max(), axis=1) >= z_score_threshold
    )
    extreme = df[z_mask].copy()

    if extreme.empty:
        st.info(
            f"No instruments with z-scores ≥ {z_score_threshold}σ right now. Markets are calm. Try lowering the threshold."
        )
    else:
        extreme["max_z"] = extreme[z_cols_available].max(axis=1)
        extreme["max_z_period"] = extreme[z_cols_available].idxmax(axis=1)

        # map z-score period to corresponding return column
        z_to_return = {"z_1d": "r_1d", "z_1w": "r_1w", "z_2w": "r_2w", "z_1mo": "r_1mo"}

        # direction of the return for the period that triggered the z-score
        extreme["trigger_return"] = extreme.apply(
            lambda row: row.get(z_to_return.get(row["max_z_period"], "r_1d"), 0), axis=1
        )

        # count how many of r_1d, r_1w, r_1mo are negative — need majority negative to be a crasher
        ret_cols_for_direction = [
            c for c in ["r_1d", "r_1w", "r_1mo"] if c in extreme.columns
        ]
        extreme["neg_count"] = (extreme[ret_cols_for_direction] < 0).sum(axis=1)

        # crasher = trigger return negative AND majority of short-term returns negative
        is_crashing = (extreme["trigger_return"] < 0) & (extreme["neg_count"] >= 2)

        crashers = (
            extreme[is_crashing]
            .sort_values("max_z", ascending=False)
            .reset_index(drop=True)
        )
        spikers = (
            extreme[~is_crashing]
            .sort_values("max_z", ascending=False)
            .reset_index(drop=True)
        )

        cols_display = (
            ["description", "ticker", "fund_type"]
            + z_cols_available
            + ["r_1d", "r_1w", "r_1mo", "r_3mo", drawdown_col, "max_z", "max_z_period"]
        )
        cols_display = [c for c in cols_display if c in extreme.columns]
        fmt_cols = [
            c
            for c in z_cols_available
            + ["r_1d", "r_1w", "r_1mo", "r_3mo", drawdown_col, "max_z"]
            if c in cols_display
        ]

        st.subheader(f"📉 Crashers ({len(crashers)})")
        st.markdown("Extreme moves + negative momentum — pain in progress.")
        if crashers.empty:
            st.info("No crashers detected.")
        else:
            st.dataframe(
                crashers[cols_display].style.format(
                    subset=fmt_cols,
                    formatter="{:.2f}",
                ),
                hide_index=True,
                height=450,
            )

        st.markdown("---")

        st.subheader(f"📈 Spikers ({len(spikers)})")
        st.markdown(
            "Extreme moves + positive momentum — could be a bounce or breakout."
        )
        if spikers.empty:
            st.info("No spikers detected.")
        else:
            st.dataframe(
                spikers[cols_display].style.format(
                    subset=fmt_cols,
                    formatter="{:.2f}",
                ),
                hide_index=True,
                height=450,
            )

        # summary metrics
        st.markdown("---")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Total Extreme Movers", len(extreme))
        mcol2.metric("📉 Crashers", len(crashers))
        mcol3.metric("📈 Spikers", len(spikers))
        mcol4.metric("Avg Max Z-Score", f"{extreme['max_z'].mean():.2f}σ")
else:
    st.warning(
        "Z-score columns not available. Re-export parquet with updated `duckdb_importer.py` "
        "(now includes z_1d, z_1w, z_2w, z_1mo)."
    )


# ═══════════════════════════════════════════
# Section 2: Drawdown from 52-Week High
# ═══════════════════════════════════════════
st.header(f"📉 Drawdown from {drawdown_label}")
st.markdown(
    f"Instruments ranked by how far they've fallen from their **{drawdown_label}**. "
    f"Deeper drawdowns = more pain = potential opportunity."
)

if drawdown_col in df.columns:
    dd = df[
        [
            "description",
            "ticker",
            "fund_type",
            drawdown_col,
            "ma_252",
            "r_1w",
            "r_1mo",
            "r_3mo",
            "r_6mo",
            "r_1y",
        ]
    ].copy()
    dd = dd.sort_values(drawdown_col, ascending=True).head(30).reset_index(drop=True)

    col_dd1, col_dd2 = st.columns([3, 4])

    with col_dd1:
        dd_fmt_cols = [
            c
            for c in [drawdown_col, "ma_252", "r_1w", "r_1mo", "r_3mo", "r_6mo", "r_1y"]
            if c in dd.columns
        ]
        st.dataframe(
            dd.style.format(subset=dd_fmt_cols, formatter="{:+.2f}%"),
            hide_index=True,
            height=550,
        )

    with col_dd2:
        fig_dd = px.bar(
            dd.head(20),
            x=drawdown_col,
            y="description",
            orientation="h",
            color=drawdown_col,
            color_continuous_scale="RdYlGn",
            labels={drawdown_col: f"% from {drawdown_label}", "description": ""},
            title=f"Top 20 — Deepest Drawdowns from {drawdown_label}",
        )
        fig_dd.update_layout(
            yaxis=dict(autorange="reversed"), height=550, showlegend=False
        )
        st.plotly_chart(fig_dd, use_container_width=True)
else:
    st.info(
        f"`{drawdown_col}` column not available yet. Re-run data import after SQL update."
    )


# ═══════════════════════════════════════════
# Section 3: Capitulation Candidates
# ═══════════════════════════════════════════
st.header("💀 Capitulation Candidates")
st.markdown(
    f"Instruments combining: **vol spike** (ratio > {vol_ratio_threshold}x) + "
    f"**deep drawdown** ({drawdown_label} < {drawdown_threshold}%). "
    "Full-blown capitulation pattern."
)

# 1. Strict Filter
mask_strict = df["vol_ratio"] >= vol_ratio_threshold
if drawdown_col in df.columns:
    mask_strict = mask_strict & (df[drawdown_col] <= drawdown_threshold)
else:
    mask_strict = mask_strict & (df["ma_252"] <= drawdown_threshold)

cap_strict = df[mask_strict].copy()

# 2. Relaxed Filter (Watchlist)
# relax vol ratio by 20% and drawdown by 25% (or 5% absolute if drawdown is small)
vol_relaxed = max(0.8, vol_ratio_threshold * 0.8)
dd_relaxed = min(-5.0, drawdown_threshold * 0.75)

mask_relaxed = (df["vol_ratio"] >= vol_relaxed) & (
    ~mask_strict
)  # strict ones are already in strict
if drawdown_col in df.columns:
    mask_relaxed = mask_relaxed & (df[drawdown_col] <= dd_relaxed)
else:
    mask_relaxed = mask_relaxed & (df["ma_252"] <= dd_relaxed)

cap_watchlist = df[mask_relaxed].copy()

# ─── DISPLAY LOGIC ───


# helper for display columns
def get_cap_cols(dframe):
    cols = ["description", "ticker", "fund_type", "vol_1mo", "vol_1y", "vol_ratio"]
    if drawdown_col in dframe.columns:
        cols.append(drawdown_col)
    cols += ["ma_252", "r_1w", "r_1mo", "r_3mo", "severity", "bounce_starting"]
    return [c for c in cols if c in dframe.columns]


# Calculate severity/bounce for both
for c_df in [cap_strict, cap_watchlist]:
    if not c_df.empty:
        dd_val = c_df[drawdown_col] if drawdown_col in c_df.columns else c_df["ma_252"]
        c_df["severity"] = (-dd_val) * c_df["vol_ratio"]
        c_df["bounce_starting"] = c_df["r_1w"] > 0
        c_df.sort_values("severity", ascending=False, inplace=True, ignore_index=True)


active_cap = pd.DataFrame()

if cap_strict.empty:
    st.info(
        f"No instruments meet strict criteria (Vol > {vol_ratio_threshold}x, Drawdown < {drawdown_threshold}%). "
        "Showing **Watchlist** candidates (close matches)."
    )
    # Show Watchlist as main if strict is empty
    if cap_watchlist.empty:
        st.warning(
            "No watchlist candidates found either. Showing **Relative Stress Leaders** (highest Vol × Drawdown) instead."
        )
        # Fallback: Top 20 by Stress Score (Severity)
        # We need to compute severity for everyone
        fallback_df = df.copy()

        # Determine drawdown column
        if drawdown_col in fallback_df.columns:
            dd_val = fallback_df[drawdown_col]
        elif "ma_252" in fallback_df.columns:
            dd_val = fallback_df["ma_252"]
        else:
            # absolute worst case, shouldn't happen given prior checks
            dd_val = pd.Series(0, index=fallback_df.index)

        # Ensure vol_ratio exists and handle NaNs
        if "vol_ratio" not in fallback_df.columns:
            # recalculate if missing
            v1 = fallback_df["vol_1mo"] if "vol_1mo" in fallback_df.columns else 0
            v12 = fallback_df["vol_1y"] if "vol_1y" in fallback_df.columns else 1
            fallback_df["vol_ratio"] = v1 / v12.replace(0, np.nan)

        # Safe fill for vol_ratio (neutral if missing)
        fallback_df["vol_ratio"] = fallback_df["vol_ratio"].fillna(1.0)

        # Severity = Drawdown magnitude * Vol Ratio
        # We want deep negative drawdown (so -drawdown is positive magnitude)
        # Handle potential NaNs in drawdown
        dd_val = dd_val.fillna(0)

        fallback_df["severity"] = (-dd_val) * fallback_df["vol_ratio"]
        fallback_df["bounce_starting"] = fallback_df["r_1w"] > 0

        # Relaxed filter: just ensure we have *some* drawdown or stress
        # If severity is 0, it means no drawdown or 0 vol ratio.
        # We'll take top 20 absolute, effectively no filter other than valid data.

        top_stress = (
            fallback_df.sort_values("severity", ascending=False)
            .head(20)
            .reset_index(drop=True)
        )
        active_cap = top_stress

        cols_f = get_cap_cols(top_stress)
        st.dataframe(
            top_stress[cols_f].style.format(
                formatter="{:.2f}",
                subset=[
                    c for c in cols_f if pd.api.types.is_numeric_dtype(top_stress[c])
                ],
            ),
            hide_index=True,
            height=400,
        )
        active_cap = top_stress

    else:
        st.subheader(f"👀 Watchlist Candidates ({len(cap_watchlist)})")
        st.markdown(
            f"Relaxed criteria: Vol > {vol_relaxed:.1f}x, Drawdown < {dd_relaxed:.0f}%"
        )

        cols = get_cap_cols(cap_watchlist)
        st.dataframe(
            cap_watchlist[cols].style.format(
                formatter="{:.2f}",
                subset=[
                    c for c in cols if pd.api.types.is_numeric_dtype(cap_watchlist[c])
                ],
            ),
            hide_index=True,
            height=400,
        )
        # For charts/buys, we'll use watchlist since strict is empty
        active_cap = cap_watchlist
else:
    st.markdown(f"**{len(cap_strict)} strict candidates** found")

    cols = get_cap_cols(cap_strict)
    col_left, col_right = st.columns([3, 4])

    with col_left:
        st.dataframe(
            cap_strict[cols].style.format(
                formatter="{:.2f}",
                subset=[
                    c for c in cols if pd.api.types.is_numeric_dtype(cap_strict[c])
                ],
            ),
            hide_index=True,
            height=400,
        )

    with col_right:
        scatter_x = drawdown_col if drawdown_col in cap_strict.columns else "ma_252"
        fig = px.scatter(
            cap_strict,
            x=scatter_x,
            y="vol_ratio",
            color="bounce_starting",
            color_discrete_map={True: "#2ecc71", False: "#e74c3c"},
            size="severity",
            size_max=25,
            text="ticker",
            labels={
                scatter_x: f"% from {drawdown_label}",
                "vol_ratio": "Vol Ratio (1mo/1y)",
                "bounce_starting": "1W bounce?",
            },
            title="Strict Candidates: Drawdown vs Vol Spike",
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    # If we have strict candidates, we can also show watchlist below if desired,
    # OR just keep it focused. User asked for "close" candidates.
    # Let's show watchlist in a collapsed expander if strict exists
    if not cap_watchlist.empty:
        with st.expander(
            f"See {len(cap_watchlist)} more 'Close' Candidates (Watchlist)"
        ):
            cols_w = get_cap_cols(cap_watchlist)
            st.dataframe(
                cap_watchlist[cols_w].style.format(
                    formatter="{:.2f}",
                    subset=[
                        c
                        for c in cols_w
                        if pd.api.types.is_numeric_dtype(cap_watchlist[c])
                    ],
                ),
                hide_index=True,
            )

    active_cap = cap_strict


# ═══════════════════════════════════════════
# Section 4: Puke Buy Signals
# ═══════════════════════════════════════════
st.header("🛒 Puke Buy Signals")
st.markdown(
    "Candidates where **1-week return is positive** (or least negative). "
    "Prioritizes Strict candidates, falls back to Watchlist."
)

if active_cap.empty and cap_watchlist.empty:
    st.info("No candidates available for buy signals.")
else:
    # merge pool for buy signals?
    # Logic:
    # 1. Look for Strict + Bounce
    # 2. If none, look for Watchlist + Bounce
    # 3. If none, look for Strict + Least Negative
    # 4. If none, look for Watchlist + Least Negative

    # Actually, simplest is to use 'active_cap' derived above (Strict if exists, else Watchlist)
    # BUT user might want to see bouncing Watchlist items even if there are non-bouncing Strict items.

    # Let's pool them for the Buy Signal ranker, but mark them
    pool = active_cap.copy()
    pool["tier"] = "Strict" if active_cap is cap_strict else "Watchlist"

    # If we are showing Strict, but there are bouncing Watchlist items, maybe include them?
    # Let's stick to the active pool to avoid confusion, separate tiers is cleaner.

    buys = pool[pool["bounce_starting"]].copy()

    if buys.empty:
        if active_cap is cap_strict and not cap_watchlist.empty:
            # Strict has no bounces. Check Watchlist for bounces?
            watchlist_buys = cap_watchlist[cap_watchlist["bounce_starting"]].copy()
            if not watchlist_buys.empty:
                st.info(
                    "No strict candidates bouncing yet, but these **Watchlist** items are bouncing:"
                )
                buys = watchlist_buys
                pool = cap_watchlist.copy()  # switch context for "closest"
                pool["tier"] = "Watchlist"

    if buys.empty:
        st.info("No green weekly candles detected.")
        st.subheader("👀 Closest to Bouncing")
        st.markdown("Candidates with the **least-negative 1W returns**.")

        # Show top 10 from the pool (sorted by r_1w desc)
        near_bounce = (
            pool.sort_values("r_1w", ascending=False).head(10).reset_index(drop=True)
        )
        near_bounce.index += 1

        disp_cols = [
            "description",
            "ticker",
            "r_1w",
            "r_1mo",
            drawdown_col,
            "vol_ratio",
            "severity",
        ]
        disp_cols = [c for c in disp_cols if c in near_bounce.columns]

        st.dataframe(
            near_bounce[disp_cols].style.format(
                formatter="{:.2f}",
                subset=[
                    c
                    for c in disp_cols
                    if pd.api.types.is_numeric_dtype(near_bounce[c])
                ],
            ),
            height=350,
        )

    else:
        # We have buys!
        buys = buys.sort_values("severity", ascending=False).reset_index(drop=True)
        buys.index += 1

        st.subheader(f"✅ {len(buys)} Buy Signals Triggered")

        disp_cols = [
            "description",
            "ticker",
            "r_1w",
            "r_1mo",
            drawdown_col,
            "vol_ratio",
            "severity",
        ]
        # if 'tier' in buys.columns: disp_cols.append('tier')
        disp_cols = [c for c in disp_cols if c in buys.columns]

        st.dataframe(
            buys[disp_cols].style.format(
                formatter="{:.2f}",
                subset=[c for c in disp_cols if pd.api.types.is_numeric_dtype(buys[c])],
            ),
            height=350,
        )

        # Plot for buys
        fig_buy = px.bar(
            buys.head(10),
            x="severity",
            y="description",
            orientation="h",
            color="r_1w",
            color_continuous_scale="YlGn",
            labels={
                "severity": "Severity Score",
                "description": "",
                "r_1w": "1W Return %",
            },
            title="Top Buy Signals (Severity × Bounce)",
        )
        fig_buy.update_layout(
            yaxis=dict(autorange="reversed"), height=350, showlegend=False
        )
        st.plotly_chart(fig_buy, use_container_width=True)


# ═══════════════════════════════════════════
# Section 5: Vol Spike Heatmap
# ═══════════════════════════════════════════
st.header("📊 Volatility Spike Heatmap")
st.markdown(
    "All instruments ranked by vol ratio (1-month / 1-year). "
    "Values > 1.0 = short-term vol exceeds long-term — stress."
)

vol_view = (
    df[["description", "ticker", "vol_1mo", "vol_1y", "vol_ratio"]]
    .dropna(subset=["vol_ratio"])
    .copy()
)
if drawdown_col in df.columns:
    vol_view[drawdown_col] = df[drawdown_col]

vol_view = (
    vol_view.sort_values("vol_ratio", ascending=False).head(30).reset_index(drop=True)
)

fig_vol = px.bar(
    vol_view,
    x="vol_ratio",
    y="description",
    orientation="h",
    color="vol_ratio",
    color_continuous_scale="YlOrRd",
    labels={"vol_ratio": "Vol Ratio (1mo/1y)", "description": ""},
    title="Top 30 — Highest Vol Spike Ratio",
)
fig_vol.update_layout(
    yaxis=dict(autorange="reversed"),
    height=max(400, len(vol_view) * 25),
    showlegend=False,
)
fig_vol.add_vline(
    x=1.0, line_dash="dash", line_color="grey", opacity=0.5, annotation_text="Neutral"
)
st.plotly_chart(fig_vol, use_container_width=True)
