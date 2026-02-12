"""
Market Sentiment — aggregate sentiment signals from multiple sources:
  1) VIX level (from tracked ^VIX data)
  2) Internal breadth indicators (computed from our own instrument data)
  3) CNN Fear & Greed Index (optional, via fear-and-greed package)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import duckdb_importer as di
from data import get_conn

st.title("🧭 Market Sentiment")
st.markdown(
    "Aggregate view of market fear/greed using **internal breadth signals**, "
    "**VIX**, and **external sentiment indicators**."
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

# separate equity data for breadth calcs
eq_data = _all_data[_all_data["fund_type"].str.match("^eq")].copy()
eq_data["vol_ratio"] = eq_data["vol_1mo"] / eq_data["vol_1y"].replace(0, np.nan)


# ═══════════════════════════════════════════
# Section 1: VIX Dashboard
# ═══════════════════════════════════════════
st.header("📈 VIX — Fear Gauge")

vix_row = _all_data[_all_data["ticker"] == "^VIX"]

if vix_row.empty:
    st.info(
        "VIX data not available. Add `^VIX` to `instrument_info.csv` and re-run the data import. "
        "In the meantime, internal breadth signals below are available."
    )
else:
    vix = vix_row.iloc[0]
    # VIX returns are the level itself from the MA columns (% deviation)
    vix_ma252 = float(vix.get("ma_252", 0))
    vix_r1w = float(vix.get("r_1w", 0))
    vix_r1mo = float(vix.get("r_1mo", 0))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "VIX vs 252d MA", f"{vix_ma252:+.1f}%", help="Positive = above average fear"
        )
    with col2:
        st.metric("VIX 1W Change", f"{vix_r1w:+.1f}%")
    with col3:
        st.metric("VIX 1M Change", f"{vix_r1mo:+.1f}%")

    # VIX regime interpretation
    if vix_ma252 > 20:
        st.error(
            "🔴 **Extreme Fear** — VIX well above average. Historically good for contrarian buying."
        )
    elif vix_ma252 > 5:
        st.warning(
            "🟡 **Elevated Fear** — VIX above average. Watch for opportunities if it spikes further."
        )
    elif vix_ma252 < -15:
        st.info(
            "🟢 **Complacency** — VIX well below average. Low fear can precede corrections."
        )
    else:
        st.success("⚪ **Neutral** — VIX near its average.")


# ═══════════════════════════════════════════
# Section 2: Internal Breadth Signals
# ═══════════════════════════════════════════
st.header("📊 Market Breadth Dashboard")
st.markdown(
    "Computed from our **equity instrument universe**. These signals measure "
    "how broadly the market is participating in trends."
)

# Breadth metrics
total_eq = len(eq_data)

pct_above_252ma = (eq_data["ma_252"] > 0).sum() / total_eq * 100 if total_eq > 0 else 0
pct_above_126ma = (eq_data["ma_126"] > 0).sum() / total_eq * 100 if total_eq > 0 else 0
pct_above_63ma = (eq_data["ma_63"] > 0).sum() / total_eq * 100 if total_eq > 0 else 0
pct_above_21ma = (eq_data["ma_21"] > 0).sum() / total_eq * 100 if total_eq > 0 else 0

pct_positive_1w = (eq_data["r_1w"] > 0).sum() / total_eq * 100 if total_eq > 0 else 0
pct_positive_1mo = (eq_data["r_1mo"] > 0).sum() / total_eq * 100 if total_eq > 0 else 0
pct_positive_3mo = (eq_data["r_3mo"] > 0).sum() / total_eq * 100 if total_eq > 0 else 0

avg_vol_ratio = eq_data["vol_ratio"].mean()

# z-score stress
z_cols = [c for c in ["z_1d", "z_1w"] if c in eq_data.columns]
avg_z = eq_data[z_cols].mean().mean() if z_cols else 0

# Display breadth metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Above 252d MA", f"{pct_above_252ma:.0f}%", help="Long-term breadth")
with col2:
    st.metric("Above 126d MA", f"{pct_above_126ma:.0f}%", help="Intermediate breadth")
with col3:
    st.metric("Above 63d MA", f"{pct_above_63ma:.0f}%", help="Medium-term breadth")
with col4:
    st.metric("Above 21d MA", f"{pct_above_21ma:.0f}%", help="Short-term breadth")

col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric("Positive 1W", f"{pct_positive_1w:.0f}%")
with col6:
    st.metric("Positive 1M", f"{pct_positive_1mo:.0f}%")
with col7:
    st.metric("Positive 3M", f"{pct_positive_3mo:.0f}%")
with col8:
    st.metric("Avg Vol Ratio", f"{avg_vol_ratio:.2f}x", help=">1 = stress")


# Breadth gauge
st.subheader("🎯 Composite Breadth Score")

# compute a 0-100 score
breadth_components = [
    pct_above_252ma,
    pct_above_126ma,
    pct_above_63ma,
    pct_above_21ma,
    pct_positive_1w,
    pct_positive_1mo,
]
breadth_score = sum(breadth_components) / len(breadth_components)

# gauge chart
fig_gauge = go.Figure(
    go.Indicator(
        mode="gauge+number+delta",
        value=breadth_score,
        title={"text": "Market Breadth Score (0-100)"},
        delta={
            "reference": 50,
            "increasing": {"color": "green"},
            "decreasing": {"color": "red"},
        },
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 20], "color": "#ff4444"},
                {"range": [20, 40], "color": "#ff8800"},
                {"range": [40, 60], "color": "#ffcc00"},
                {"range": [60, 80], "color": "#88cc00"},
                {"range": [80, 100], "color": "#00cc44"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": breadth_score,
            },
        },
    )
)
fig_gauge.update_layout(height=300)
st.plotly_chart(fig_gauge, use_container_width=True)

if breadth_score < 25:
    st.error(
        "🔴 **Extreme Fear / Oversold** — Very few instruments in uptrends. Contrarian buy zone."
    )
elif breadth_score < 40:
    st.warning(
        "🟡 **Weak Breadth** — Market participation declining. Watch for further deterioration or bounce."
    )
elif breadth_score > 75:
    st.info(
        "🟢 **Strong Breadth** — Broad participation. Momentum strategies favoured."
    )
elif breadth_score > 90:
    st.warning(
        "🟡 **Extreme Greed / Overbought** — Near-universal participation. Be cautious of complacency."
    )
else:
    st.success("⚪ **Neutral** — Mixed breadth signals.")


# ═══════════════════════════════════════════
# Section 3: Breadth Breakdown by Fund Type
# ═══════════════════════════════════════════
st.header("📋 Breadth by Asset Category")

breadth_rows = []
for ft in _all_data["fund_type"].unique():
    ft_data = _all_data[_all_data["fund_type"] == ft]
    n = len(ft_data)
    if n < 2:
        continue
    breadth_rows.append(
        {
            "Fund Type": ft,
            "Count": n,
            "% > 252d MA": (ft_data["ma_252"] > 0).sum() / n * 100,
            "% > 63d MA": (ft_data["ma_63"] > 0).sum() / n * 100,
            "% +ve 1M": (ft_data["r_1mo"] > 0).sum() / n * 100,
            "Avg 1M Return": ft_data["r_1mo"].mean(),
        }
    )

if breadth_rows:
    breadth_df = pd.DataFrame(breadth_rows).sort_values("% > 252d MA", ascending=False)
    st.dataframe(
        breadth_df.style.format(
            {
                "% > 252d MA": "{:.0f}%",
                "% > 63d MA": "{:.0f}%",
                "% +ve 1M": "{:.0f}%",
                "Avg 1M Return": "{:+.2f}%",
            }
        ),
        hide_index=True,
        height=350,
    )


# ═══════════════════════════════════════════
# Section 4: CNN Fear & Greed (Optional)
# ═══════════════════════════════════════════
st.header("🌡️ CNN Fear & Greed Index")

try:
    import fear_and_greed

    fng = fear_and_greed.get()
    fng_value = fng.value
    fng_description = fng.description

    fig_fng = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=fng_value,
            title={"text": f"Fear & Greed: {fng_description}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 25], "color": "#ff4444"},
                    {"range": [25, 45], "color": "#ff8800"},
                    {"range": [45, 55], "color": "#ffcc00"},
                    {"range": [55, 75], "color": "#88cc00"},
                    {"range": [75, 100], "color": "#00cc44"},
                ],
            },
        )
    )
    fig_fng.update_layout(height=300)
    st.plotly_chart(fig_fng, use_container_width=True)

    st.caption(
        "Source: CNN Business Fear & Greed Index. "
        "0 = Extreme Fear, 100 = Extreme Greed."
    )

except ImportError:
    st.info(
        "CNN Fear & Greed not available. Install with: `pip install fear-and-greed`\n\n"
        "Internal breadth signals above provide similar information."
    )
except Exception as e:
    st.warning(f"Could not fetch Fear & Greed data: {e}")


# ═══════════════════════════════════════════
# Section 5: Sentiment Summary
# ═══════════════════════════════════════════
st.header("📝 Sentiment Summary")

st.markdown("### Strategy Implications")

if breadth_score < 30:
    st.markdown(
        "- 🎯 **Puke buying territory** — look at the Puke Detector for specific candidates\n"
        "- 📉 Breadth is very weak — most instruments below key MAs\n"
        "- ⚠️ Could get worse before it gets better — consider scaling in gradually"
    )
elif breadth_score < 50:
    st.markdown(
        "- 🔍 **Watch for capitulation** — check Puke Detector for vol spikes\n"
        "- 🔄 Look for underperformers recovering in the Pullback Scanner\n"
        "- 📊 Mixed signals — be selective with entries"
    )
elif breadth_score > 75:
    st.markdown(
        "- 🚀 **Momentum favoured** — ride existing trends via Thematic Dashboard\n"
        "- 📈 Broad participation supports continued upside\n"
        "- ⚠️ Watch for breadth divergence (fewer instruments making new highs)"
    )
else:
    st.markdown(
        "- ⚖️ **Balanced approach** — mix of momentum and value strategies\n"
        "- 🔍 Scan for sector rotation opportunities in Cross-Asset Regime\n"
        "- 📊 Use Factor Dashboard to identify leading factors"
    )
