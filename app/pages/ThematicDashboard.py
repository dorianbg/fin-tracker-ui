"""
Thematic Dashboard — compare performance across thematic/megatrend ETFs grouped by theme.

Themes include AI, cybersecurity, cloud, robotics, blockchain, fintech, gaming,
defence, healthcare, clean energy, hydrogen, batteries, uranium, and more.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

import duckdb_importer as di
from data import get_conn

st.title("🚀 Thematic Dashboard")
st.markdown(
    "Compare **thematic / megatrend ETFs** across investment themes. "
    "Identify which themes are leading, lagging, or turning around."
)

# ── Theme definitions ──
THEME_GROUPS = {
    "🤖 AI & Big Data": ["AIAG", "XAIX"],
    "🔒 Cybersecurity": ["ISPY", "LOCK"],
    "☁️ Cloud Computing": ["WCLD", "FSKY"],
    "🦾 Robotics & Automation": ["ROBG", "RBTX"],
    "⛓️ Blockchain & Crypto": ["BCHN", "DAGB"],
    "💳 FinTech & Digital": ["FING", "DGIT"],
    "📡 IoT": ["SNSG"],
    "⚛️ Quantum Computing": ["QNTG", "WQTM"],
    "🎮 Gaming & eSports": ["ESGB", "PLAY"],
    "🚀 Space & Innovation": ["JEDI", "DFND", "NATO"],
    "💊 Healthcare & Biotech": ["HEAL", "BTEC"],
    "🧬 Nanotech / Health": ["DOCG"],
    "👴 Aging Population": ["AGED"],
    "🏙️ Smart Cities": ["IQCY"],
    "💾 Hardware & Semis": ["SMH", "SOXX"],
    "🌿 Clean Energy": ["INRG", "ICLN"],
    "🛢️ Traditional Energy": ["WENS"],
    "💧 Water": ["IH2O"],
    "🔋 Hydrogen": ["HTWG", "HDRO"],
    "🔌 Battery & EV": ["BATG", "CHRG", "LITG"],
    "🚗 Autonomous & EV": ["ECAR"],
    "☢️ Uranium & Nuclear": ["URNG", "NUCG"],
    "🛍️ Consumer Discretionary": ["IUCD", "XLY"],
    "🍞 Consumer Staples": ["IUCS", "XLP"],
    "📦 E-commerce & Logistics": ["ECOG", "EBIG"],
    "💎 Luxury": ["LUXG"],
    "🌾 Agribusiness": ["SPAG", "KROG"],
    "🏠 Global Real Estate": ["IWDP"],
    "🏗️ Infrastructure": ["INFR"],
    "⛏️ Gold Miners": ["GDX"],
}

ALL_THEMATIC_TICKERS = [t for tickers in THEME_GROUPS.values() for t in tickers]

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
def load_thematic_data() -> pd.DataFrame:
    cols = (
        di.perf_desc_cols_start
        + di.perf_vol_cols
        + di.perf_mavg_cols
        + di.perf_returns_cols
        + di.perf_desc_cols_end
        + di.perf_rownames_cols
    )
    tickers_str = "','".join(ALL_THEMATIC_TICKERS)
    query = f"""
        SELECT {",".join(cols)}
        FROM {di.perf_tbl}
        WHERE rown = 1 AND ticker IN ('{tickers_str}')
        ORDER BY description ASC
    """
    return get_conn().execute(query).df()


df = load_thematic_data()

if df.empty:
    st.warning(
        "No thematic ETF data found. Ensure thematic tickers are in instrument_info.csv."
    )
    st.stop()

# assign theme labels
ticker_to_theme = {}
for theme, tickers in THEME_GROUPS.items():
    for t in tickers:
        ticker_to_theme[t] = theme

df["theme"] = df["ticker"].map(ticker_to_theme)
# drop any rows where the ticker wasn't found in our theme map
df = df.dropna(subset=["theme"]).reset_index(drop=True)

st.caption(f"Tracking **{len(df)}** ETFs across **{df['theme'].nunique()}** themes")


# ═══════════════════════════════════════════
# Section 1: Theme-Level Return Heatmap
# ═══════════════════════════════════════════
st.header("📊 Theme Return Heatmap")
st.markdown(
    "Average return per theme across time horizons. Green = positive, Red = negative."
)

theme_avg = df.groupby("theme")[RETURN_PERIODS].mean()
theme_avg.columns = RETURN_LABELS

# sort by 1Y performance for a useful default ordering
if "1Y" in theme_avg.columns:
    theme_avg = theme_avg.sort_values("1Y", ascending=False)

fig_theme = px.imshow(
    theme_avg.values,
    x=RETURN_LABELS,
    y=list(theme_avg.index),
    color_continuous_scale="RdYlGn",
    color_continuous_midpoint=0,
    text_auto=".1f",
    aspect="auto",
    labels=dict(color="Avg Return %"),
)
fig_theme.update_layout(height=max(450, len(theme_avg) * 32))
st.plotly_chart(fig_theme, use_container_width=True)


# ═══════════════════════════════════════════
# Section 2: Theme Comparison Bar Chart
# ═══════════════════════════════════════════
st.header("📈 Theme Comparison")

period = st.selectbox("Return period", options=RETURN_LABELS, index=5)
period_col = RETURN_PERIODS[RETURN_LABELS.index(period)]

theme_period_avg = (
    df.groupby("theme")[period_col].mean().sort_values(ascending=True).reset_index()
)
theme_period_avg.columns = ["theme", "avg_return"]

fig_bar = px.bar(
    theme_period_avg,
    x="avg_return",
    y="theme",
    orientation="h",
    color="avg_return",
    color_continuous_scale="RdYlGn",
    color_continuous_midpoint=0,
    labels={"avg_return": f"Avg Return ({period}) %", "theme": ""},
    title=f"Theme Performance — {period}",
)
fig_bar.update_layout(
    yaxis=dict(autorange="reversed"),
    height=max(450, len(theme_period_avg) * 30),
    showlegend=False,
)
fig_bar.add_vline(x=0, line_dash="solid", line_color="grey", opacity=0.5)
st.plotly_chart(fig_bar, use_container_width=True)


# ═══════════════════════════════════════════
# Section 3: Individual ETF Breakdown
# ═══════════════════════════════════════════
st.header("🔬 Individual ETF Breakdown")

theme_filter = st.multiselect(
    "Filter by theme",
    options=sorted(df["theme"].unique()),
    default=[],
    help="Leave empty to show all themes",
)

if theme_filter:
    filtered = df[df["theme"].isin(theme_filter)].copy()
else:
    filtered = df.copy()

# sort by theme then description
filtered = filtered.sort_values(["theme", "description"]).reset_index(drop=True)

etf_heatmap = filtered.set_index("description")[RETURN_PERIODS].copy()
etf_heatmap.columns = RETURN_LABELS

fig_etf = px.imshow(
    etf_heatmap.values,
    x=RETURN_LABELS,
    y=list(etf_heatmap.index),
    color_continuous_scale="RdYlGn",
    color_continuous_midpoint=0,
    text_auto=".1f",
    aspect="auto",
    labels=dict(color="Return %"),
)
fig_etf.update_layout(height=max(400, len(etf_heatmap) * 25))
st.plotly_chart(fig_etf, use_container_width=True)

# table view with MA and vol data
st.subheader("Detailed View")
detail_cols = [
    "description",
    "ticker",
    "theme",
    "ma_21",
    "ma_63",
    "ma_126",
    "ma_252",
    "vol_1mo",
    "vol_1y",
] + RETURN_PERIODS
available_cols = [c for c in detail_cols if c in filtered.columns]
st.dataframe(
    filtered[available_cols].style.format(
        subset=[
            c
            for c in ["ma_21", "ma_63", "ma_126", "ma_252", "vol_1mo", "vol_1y"]
            + RETURN_PERIODS
            if c in available_cols
        ],
        formatter="{:+.2f}%",
    ),
    hide_index=True,
    height=500,
)


# ═══════════════════════════════════════════
# Section 4: Momentum Leaders
# ═══════════════════════════════════════════
st.header("🏆 Momentum Leaders")
st.markdown(f"Top and bottom performing thematic ETFs by **{period}** return.")

col_top, col_bottom = st.columns(2)

with col_top:
    st.subheader("🟢 Top 10")
    top10 = df.nlargest(10, period_col)[
        ["description", "ticker", "theme", period_col]
    ].reset_index(drop=True)
    top10.index = top10.index + 1
    st.dataframe(
        top10.style.format(subset=[period_col], formatter="{:+.2f}%"),
        height=400,
    )

with col_bottom:
    st.subheader("🔴 Bottom 10")
    bottom10 = df.nsmallest(10, period_col)[
        ["description", "ticker", "theme", period_col]
    ].reset_index(drop=True)
    bottom10.index = bottom10.index + 1
    st.dataframe(
        bottom10.style.format(subset=[period_col], formatter="{:+.2f}%"),
        height=400,
    )


# ═══════════════════════════════════════════
# Section 5: Theme Trend Status
# ═══════════════════════════════════════════
st.header("📍 Theme Trend Status")
st.markdown("Average moving average position per theme — above/below key MAs.")

ma_cols = ["ma_21", "ma_63", "ma_126", "ma_252"]
theme_ma = df.groupby("theme")[ma_cols].mean()

# sort by ma_252 for a trend-strength ordering
theme_ma = theme_ma.sort_values("ma_252", ascending=False)

fig_ma = px.imshow(
    theme_ma.values,
    x=["21d MA", "63d MA", "126d MA", "252d MA"],
    y=list(theme_ma.index),
    color_continuous_scale="RdYlGn",
    color_continuous_midpoint=0,
    text_auto=".1f",
    aspect="auto",
    labels=dict(color="% from MA"),
    title="Average MA Position by Theme",
)
fig_ma.update_layout(height=max(450, len(theme_ma) * 32))
st.plotly_chart(fig_ma, use_container_width=True)
