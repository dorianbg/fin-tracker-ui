"""
Relative Strength (RS) Ranking — percentile-rank all instruments by return across
configurable lookback periods. Identifies the strongest and weakest performers
in the universe with an IBD-style 1-99 RS rating.
"""

import streamlit as st
import plotly.express as px

from duckdb_importer import RETURN_COL_MAP
from data import (
    load_latest_perf,
    fund_type_sidebar,
    filter_by_fund_type,
    add_sparkline_column,
)


def render():
    st.title("💪 Relative Strength Ranking")
    st.markdown(
        "Percentile-rank **every instrument** by return. "
        "RS Rating 99 = top 1%, RS Rating 1 = bottom 1%. "
        "Inspired by IBD Relative Strength."
    )

    controls_col, content_col = st.columns([1, 4], gap="large")

    with controls_col:
        st.header("Relative Strength")
        rs_period = st.selectbox(
            "Primary RS period",
            options=list(RETURN_COL_MAP.keys()),
            index=4,
            key="relative_strength_primary_period",
        )
        rs_col = RETURN_COL_MAP[rs_period]

        fund_type_filter = fund_type_sidebar(key="relative_strength_fund_types")

        st.markdown("---")
        sort_by = st.radio(
            "Sort by",
            ["Single-period RS", "Composite RS"],
            key="relative_strength_sort_by",
        )

        st.subheader("Composite Weights")
        w_3m = st.slider(
            "3M weight", 0, 100, 40, 5, key="relative_strength_3m_weight"
        )
        w_6m = st.slider(
            "6M weight", 0, 100, 30, 5, key="relative_strength_6m_weight"
        )
        w_1y = st.slider(
            "1Y weight", 0, 100, 30, 5, key="relative_strength_1y_weight"
        )

    with content_col:
        _all_data = load_latest_perf()

        if _all_data.empty:
            st.warning("No data loaded.")
            st.stop()

        df = filter_by_fund_type(_all_data, fund_type_filter)

        df["rs_rating"] = (
            df[rs_col]
            .rank(pct=True)
            .mul(99)
            .round(0)
            .fillna(50)
            .astype(int)
            .clip(1, 99)
        )

        total_w = w_3m + w_6m + w_1y
        if total_w > 0:
            df["composite_return"] = (
                df["r_3mo"] * (w_3m / total_w)
                + df["r_6mo"] * (w_6m / total_w)
                + df["r_1y"] * (w_1y / total_w)
            )
            df["composite_rs"] = (
                df["composite_return"]
                .rank(pct=True)
                .mul(99)
                .round(0)
                .fillna(50)
                .astype(int)
                .clip(1, 99)
            )
        else:
            df["composite_return"] = 0
            df["composite_rs"] = 50

        st.header(f"📊 RS Distribution — {rs_period}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Top Decile (≥90)", f"{(df['rs_rating'] >= 90).sum()}")
        with col2:
            st.metric(
                "Upper (70-89)",
                f"{((df['rs_rating'] >= 70) & (df['rs_rating'] < 90)).sum()}",
            )
        with col3:
            st.metric(
                "Mid (30-69)",
                f"{((df['rs_rating'] >= 30) & (df['rs_rating'] < 70)).sum()}",
            )
        with col4:
            st.metric("Bottom (<30)", f"{(df['rs_rating'] < 30).sum()}")

        fig_hist = px.histogram(
            df,
            x="rs_rating",
            nbins=20,
            color_discrete_sequence=["#3498db"],
            labels={"rs_rating": f"RS Rating ({rs_period})", "count": "Instruments"},
            title=f"Distribution of RS Ratings — {rs_period}",
        )
        fig_hist.update_layout(height=350)
        st.plotly_chart(fig_hist, use_container_width=True)

        st.header("🏆 RS Rankings")

        sort_col = "rs_rating" if sort_by == "Single-period RS" else "composite_rs"

        ranked = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        add_sparkline_column(ranked)
        add_sparkline_column(ranked, col_name="Price (1y)", days=365)
        ranked.index = ranked.index + 1
        ranked.index.name = "Rank"

        display_cols = [
            "description",
            "ticker",
            "fund_type",
            "rs_rating",
            "composite_rs",
            "Price (90d)",
            "Price (1y)",
            "r_1w",
            "r_1mo",
            "r_3mo",
            "r_6mo",
            "r_1y",
            "ma_252",
            "drawdown_52w",
        ]
        available = [c for c in display_cols if c in ranked.columns]

        st.dataframe(
            ranked[available].style.format(
                subset=[
                    c
                    for c in [
                        "r_1w",
                        "r_1mo",
                        "r_3mo",
                        "r_6mo",
                        "r_1y",
                        "ma_252",
                        "drawdown_52w",
                    ]
                    if c in available
                ],
                formatter="{:+.2f}%",
            ),
            column_config={
                "Price (90d)": st.column_config.LineChartColumn(
                    "Price (90d)", width="small"
                ),
                "Price (1y)": st.column_config.LineChartColumn(
                    "Price (1y)", width="small"
                ),
                "description": st.column_config.TextColumn(
                    "description", width="medium"
                ),
                "ticker": st.column_config.TextColumn("ticker", width="small"),
                "fund_type": st.column_config.TextColumn("fund_type", width="small"),
                "rs_rating": st.column_config.NumberColumn(
                    "rs_rating", width="small"
                ),
                "composite_rs": st.column_config.NumberColumn(
                    "composite_rs", width="small"
                ),
                "r_1w": st.column_config.NumberColumn(
                    "r_1w", format="%.2f%%", width="small"
                ),
                "r_1mo": st.column_config.NumberColumn(
                    "r_1mo", format="%.2f%%", width="small"
                ),
                "r_3mo": st.column_config.NumberColumn(
                    "r_3mo", format="%.2f%%", width="small"
                ),
                "r_6mo": st.column_config.NumberColumn(
                    "r_6mo", format="%.2f%%", width="small"
                ),
                "r_1y": st.column_config.NumberColumn(
                    "r_1y", format="%.2f%%", width="small"
                ),
                "ma_252": st.column_config.NumberColumn(
                    "ma_252", format="%.2f%%", width="small"
                ),
                "drawdown_52w": st.column_config.NumberColumn(
                    "DD 52W", format="%.2f%%", width="small"
                ),
            },
            height=600,
        )

        st.header("📈 Leaders vs Laggards")

        col_top, col_bot = st.columns(2)

        with col_top:
            st.subheader("🟢 Top 15")
            top15 = df.nlargest(15, sort_col)
            fig_top = px.bar(
                top15,
                x=sort_col,
                y="description",
                orientation="h",
                color=rs_col,
                color_continuous_scale="YlGn",
                labels={
                    sort_col: "RS Rating",
                    "description": "",
                    rs_col: f"{rs_period} Return",
                },
            )
            fig_top.update_layout(
                yaxis=dict(autorange="reversed"), height=450, showlegend=False
            )
            st.plotly_chart(fig_top, use_container_width=True)

        with col_bot:
            st.subheader("🔴 Bottom 15")
            bot15 = df.nsmallest(15, sort_col)
            fig_bot = px.bar(
                bot15,
                x=sort_col,
                y="description",
                orientation="h",
                color=rs_col,
                color_continuous_scale="OrRd_r",
                labels={
                    sort_col: "RS Rating",
                    "description": "",
                    rs_col: f"{rs_period} Return",
                },
            )
            fig_bot.update_layout(
                yaxis=dict(autorange="reversed"), height=450, showlegend=False
            )
            st.plotly_chart(fig_bot, use_container_width=True)

        st.header("🎯 RS vs Trend Alignment")
        st.markdown(
            "Instruments with **high RS + above 252d MA** are in strong uptrends with momentum. "
            "High RS + below MA = momentum without trend confirmation (risky)."
        )

        fig_align = px.scatter(
            df,
            x="ma_252",
            y="composite_rs",
            color="rs_rating",
            color_continuous_scale="RdYlGn",
            text="ticker",
            hover_data=["description", "r_3mo", "r_6mo"],
            labels={
                "ma_252": "% above 252d MA",
                "composite_rs": "Composite RS Rating",
                "rs_rating": f"RS ({rs_period})",
                "description": "Instrument",
            },
            title="Momentum (RS) vs Trend (MA position)",
        )
        fig_align.update_traces(textposition="top center", marker=dict(size=8))
        fig_align.add_hline(
            y=70,
            line_dash="dash",
            line_color="green",
            opacity=0.3,
            annotation_text="RS 70",
        )
        fig_align.add_vline(
            x=0,
            line_dash="dash",
            line_color="grey",
            opacity=0.3,
            annotation_text="252d MA",
        )
        fig_align.update_layout(height=550)
        st.plotly_chart(fig_align, use_container_width=True)
