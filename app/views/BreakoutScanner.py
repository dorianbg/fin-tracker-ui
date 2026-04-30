"""
Breakout Scanner — identify instruments breaking above technical resistance levels.

Sections:
  1) MA Breakout Detector: price just crossed above a selected moving average
  2) Multi-MA Breakout Strength: stacking breakouts across multiple MAs
  3) 52-Week High Breakouts: instruments near or at new 52-week highs
  4) Finviz-Style Return Heatmap: treemap of all instruments colored by returns
"""

import streamlit as st
import pandas as pd
import plotly.express as px

import duckdb_importer as di
from duckdb_importer import RETURN_LABELS, RETURN_PERIODS
from data import (
    get_conn,
    load_latest_perf,
    fund_type_sidebar,
    filter_by_fund_type,
    add_sparkline_column,
)


def render():
    st.title("🚀 Breakout Scanner")
    st.markdown(
        "Identify instruments **breaking above resistance levels** — "
        "moving averages, multi-MA stacks, and 52-week highs. "
        "Plus a **heatmap** of returns across all instruments."
    )

    # Streamlit sidebars are global, so keep tab-specific controls in-page.
    settings_col, content_col = st.columns([1, 4], gap="large")

    with settings_col:
        st.subheader("Breakout Settings")

        breakout_ma = st.selectbox(
            "Breakout MA",
            options=[
                "21-day (short)",
                "63-day (medium)",
                "126-day (intermediate)",
                "252-day (long)",
            ],
            index=1,
            help="Which moving average to detect breakouts above",
        )
        breakout_ma_col = {
            "21-day (short)": "ma_21",
            "63-day (medium)": "ma_63",
            "126-day (intermediate)": "ma_126",
            "252-day (long)": "ma_252",
        }[breakout_ma]
        breakout_ma_label = breakout_ma.split(" ")[0]

        max_above_ma = st.slider(
            f"Max distance above {breakout_ma_label} MA (%)",
            min_value=1,
            max_value=15,
            value=5,
            step=1,
            help="Only show instruments that recently broke above (not already far above)",
        )

        fund_type_filter = fund_type_sidebar(key="breakout_fund_types")

        st.markdown("---")
        st.header("52W High Settings")

        max_drawdown_52w = st.slider(
            "Max drawdown from 52W high (%)",
            min_value=-10,
            max_value=0,
            value=-2,
            step=1,
            help="How close to 52-week high to qualify (0 = at the high, -2 = within 2%)",
        )

        st.markdown("---")
        st.header("Heatmap Settings")

        # Skip 1D — heatmap starts from 1W
        heatmap_labels = RETURN_LABELS[1:]  # 1W onwards
        heatmap_periods = RETURN_PERIODS[1:]  # r_1w onwards

        heatmap_period_label = st.selectbox(
            "Heatmap color by",
            options=heatmap_labels,
            index=3,  # 3M default
        )
        heatmap_period_col = heatmap_periods[heatmap_labels.index(heatmap_period_label)]

        heatmap_size_metric = st.selectbox(
            "Heatmap box size",
            options=["Equal", "1Y Volatility"],
            index=0,
            help="Equal = same size boxes, 1Y Vol = larger boxes for more volatile instruments",
        )

    with content_col:
        _all_data = load_latest_perf()

        if _all_data.empty:
            st.warning("No data loaded.")
            st.stop()

        df = filter_by_fund_type(_all_data, fund_type_filter)

        # ═══════════════════════════════════════════
        # Section 1: MA Breakout Detector
        # ═══════════════════════════════════════════
        st.header("📈 MA Breakout Detector")
        st.markdown(
            f"Instruments that **just crossed above** the {breakout_ma_label} moving average "
            f"(within {max_above_ma}% above it). Fresh breakouts = potential new uptrends."
        )

        # Logic: above the selected MA (> 0) but not too far above (< max_above_ma)
        breakouts = df[
            (df[breakout_ma_col] > 0) & (df[breakout_ma_col] <= max_above_ma)
        ].copy()

        if breakouts.empty:
            st.info(
                "No breakout candidates match current filters. "
                "Try increasing the max distance or changing the MA."
            )
        else:
            breakouts = breakouts.sort_values(
                breakout_ma_col, ascending=True
            ).reset_index(drop=True)

            # Breakout freshness: closer to 0 = more recent breakout
            breakouts["freshness"] = max_above_ma - breakouts[breakout_ma_col]

            st.markdown(f"**{len(breakouts)} breakout candidates found**")

            display_cols = [
                "description",
                "ticker",
                "fund_type",
                "Price (90d)",
                breakout_ma_col,
                "ma_21",
                "ma_63",
                "ma_126",
                "ma_252",
                "r_1w",
                "r_1mo",
                "r_3mo",
                "drawdown_52w",
            ]
            display_cols = list(dict.fromkeys(display_cols))

            add_sparkline_column(breakouts)

            st.dataframe(
                breakouts[display_cols],
                column_config={
                    "Price (90d)": st.column_config.LineChartColumn(
                        "Price (90d)", width="small"
                    ),
                    "description": st.column_config.TextColumn(
                        "description", width="medium"
                    ),
                    "ticker": st.column_config.TextColumn("ticker", width="small"),
                    "fund_type": st.column_config.TextColumn(
                        "fund_type", width="small"
                    ),
                    breakout_ma_col: st.column_config.NumberColumn(
                        breakout_ma_col, format="%.2f%%", width="small"
                    ),
                    "ma_21": st.column_config.NumberColumn(
                        "ma_21", format="%.2f%%", width="small"
                    ),
                    "ma_63": st.column_config.NumberColumn(
                        "ma_63", format="%.2f%%", width="small"
                    ),
                    "ma_126": st.column_config.NumberColumn(
                        "ma_126", format="%.2f%%", width="small"
                    ),
                    "ma_252": st.column_config.NumberColumn(
                        "ma_252", format="%.2f%%", width="small"
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
                    "drawdown_52w": st.column_config.NumberColumn(
                        "DD 52W", format="%.2f%%", width="small"
                    ),
                },
                hide_index=True,
                height=450,
            )

            # Scatter: MA position vs short-term momentum
            st.subheader("Breakout Position vs Momentum")

            fig = px.scatter(
                breakouts,
                x=breakout_ma_col,
                y="r_1w",
                color="r_1mo",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                text="ticker",
                size="freshness",
                size_max=18,
                hover_data=["description"],
                labels={
                    breakout_ma_col: f"% above {breakout_ma_label} MA",
                    "r_1w": "1-Week Return (%)",
                    "r_1mo": "1-Month Return (%)",
                    "description": "Instrument",
                },
                title="Freshest breakouts with positive momentum (bottom-left + green = ideal)",
            )
            fig.update_traces(textposition="top center")
            fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.3)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ═══════════════════════════════════════════
        # Section 2: Multi-MA Breakout Strength
        # ═══════════════════════════════════════════
        st.header("🏗️ Multi-MA Breakout Strength")
        st.markdown(
            "Instruments trading above **multiple** moving averages simultaneously. "
            "More MAs above = stronger trend confirmation."
        )

        ma_cols = ["ma_21", "ma_63", "ma_126", "ma_252"]
        ma_labels = ["21d", "63d", "126d", "252d"]

        multi = df.copy()
        for col, label in zip(ma_cols, ma_labels):
            multi[f"above_{label}"] = multi[col] > 0

        multi["mas_above"] = sum(
            multi[f"above_{label}"].astype(int) for label in ma_labels
        )

        # Only show instruments above at least 1 MA
        multi = multi[multi["mas_above"] >= 1].copy()

        # Breakout score: number of MAs above weighted by how far above each
        multi["breakout_score"] = 0.0
        for col in ma_cols:
            multi["breakout_score"] += multi[col].clip(lower=0)

        multi = multi.sort_values("mas_above", ascending=False).reset_index(drop=True)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Above all 4 MAs", f"{(multi['mas_above'] == 4).sum()}")
        with col2:
            st.metric("Above 3 MAs", f"{(multi['mas_above'] == 3).sum()}")
        with col3:
            st.metric("Above 2 MAs", f"{(multi['mas_above'] == 2).sum()}")
        with col4:
            st.metric("Above 1 MA", f"{(multi['mas_above'] == 1).sum()}")

        # Table of instruments above all 4 MAs
        show_min_mas = st.selectbox(
            "Minimum MAs above",
            options=[4, 3, 2, 1],
            index=0,
        )

        filtered_multi = multi[multi["mas_above"] >= show_min_mas].copy()

        if filtered_multi.empty:
            st.info(f"No instruments above ≥{show_min_mas} MAs.")
        else:
            display_multi = [
                "description",
                "ticker",
                "fund_type",
                "mas_above",
                "ma_21",
                "ma_63",
                "ma_126",
                "ma_252",
                "breakout_score",
                "r_1w",
                "r_1mo",
                "drawdown_52w",
            ]

            st.dataframe(
                filtered_multi[display_multi]
                .sort_values("breakout_score", ascending=False)
                .style.format(
                    subset=[
                        "ma_21",
                        "ma_63",
                        "ma_126",
                        "ma_252",
                        "r_1w",
                        "r_1mo",
                        "drawdown_52w",
                    ],
                    formatter="{:+.2f}%",
                )
                .format(subset=["breakout_score"], formatter="{:.1f}"),
                hide_index=True,
                height=450,
            )

            # Top 15 by breakout score
            top_breakout = filtered_multi.nlargest(15, "breakout_score")
            fig_bar = px.bar(
                top_breakout,
                x="breakout_score",
                y="description",
                orientation="h",
                color="mas_above",
                color_continuous_scale="YlGn",
                labels={
                    "breakout_score": "Breakout Score (sum of % above each MA)",
                    "description": "",
                    "mas_above": "MAs Above",
                },
                title="Top 15 — Multi-MA Breakout Score",
            )
            fig_bar.update_layout(
                yaxis=dict(autorange="reversed"), height=450, showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        @st.fragment
        def section_52w_highs():
            # ═══════════════════════════════════════════
            # Section 3: 52-Week High Breakouts
            # ═══════════════════════════════════════════
            st.header("👑 52-Week High Breakouts")
            st.markdown(
                f"Instruments within **{abs(max_drawdown_52w)}%** of their 52-week high. "
                "Breaking to new highs = strongest form of resistance breakout."
            )

            near_high = df[df["drawdown_52w"] >= max_drawdown_52w].copy()

            if near_high.empty:
                st.info(
                    f"No instruments within {abs(max_drawdown_52w)}% of 52W high. "
                    "Try increasing the max drawdown threshold."
                )
            else:
                near_high = near_high.sort_values(
                    "drawdown_52w", ascending=False
                ).reset_index(drop=True)

                st.markdown(
                    f"**{len(near_high)} instruments** near or at 52-week highs"
                )

                col_left, col_right = st.columns([3, 4])

                with col_left:
                    display_52w = [
                        "description",
                        "ticker",
                        "fund_type",
                        "drawdown_52w",
                        "ma_252",
                        "r_1w",
                        "r_1mo",
                        "r_3mo",
                        "r_1y",
                    ]
                    st.dataframe(
                        near_high[display_52w].style.format(
                            subset=[
                                "drawdown_52w",
                                "ma_252",
                                "r_1w",
                                "r_1mo",
                                "r_3mo",
                                "r_1y",
                            ],
                            formatter="{:+.2f}%",
                        ),
                        hide_index=True,
                        height=500,
                    )

                with col_right:
                    fig_52w = px.bar(
                        near_high.head(25),
                        x="drawdown_52w",
                        y="description",
                        orientation="h",
                        color="r_3mo",
                        color_continuous_scale="RdYlGn",
                        color_continuous_midpoint=0,
                        labels={
                            "drawdown_52w": "Drawdown from 52W High (%)",
                            "description": "",
                            "r_3mo": "3M Return (%)",
                        },
                        title="Distance from 52-Week High (colored by 3M momentum)",
                    )
                    fig_52w.update_layout(
                        yaxis=dict(autorange="reversed"), height=500, showlegend=False
                    )
                    st.plotly_chart(fig_52w, use_container_width=True)

                # ── History Chart ──
                st.subheader("📅 New 52-Week Highs Over Time")

                @st.cache_data(ttl=3600)
                def load_52w_history(fund_types):
                    where_clause = ""
                    if fund_types:
                        ft_str = "','".join(fund_types)
                        where_clause = f"WHERE fund_type IN ('{ft_str}')"

                    query = f"""
                        WITH raw AS (
                            SELECT date, ticker, price
                            FROM {di.px_tbl}
                            {where_clause}
                        ),
                        rolling AS (
                            SELECT
                                date,
                                ticker,
                                price,
                                MAX(price) OVER (
                                    PARTITION BY ticker
                                    ORDER BY date
                                    ROWS BETWEEN 252 PRECEDING AND CURRENT ROW
                                ) as max_252
                            FROM raw
                        )
                        SELECT date, COUNT(*) as new_highs
                        FROM rolling
                        WHERE price >= max_252
                          AND date >= '2023-01-01'
                        GROUP BY date
                        ORDER BY date
                    """
                    return get_conn().execute(query).df()

                try:
                    hist_df = load_52w_history(fund_type_filter)
                    if not hist_df.empty:
                        hist_df["ma_10"] = hist_df["new_highs"].rolling(10).mean()

                        fig_hist = px.bar(
                            hist_df,
                            x="date",
                            y="new_highs",
                            title="Number of Instruments Hitting 52-Week Highs (Daily)",
                            labels={"new_highs": "Count", "date": "Date"},
                            color_discrete_sequence=["#4caf50"],
                        )
                        fig_hist.add_scatter(
                            x=hist_df["date"],
                            y=hist_df["ma_10"],
                            mode="lines",
                            name="10-Day MA",
                            line=dict(color="black", width=2),
                        )
                        fig_hist.update_layout(height=400, hovermode="x unified")
                        st.plotly_chart(fig_hist, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not load 52-week high history: {e}")

        section_52w_highs()

        st.markdown("---")

        @st.fragment
        def section_heatmap():
            # ═══════════════════════════════════════════
            # Section 4: Finviz-Style Return Heatmap
            # ═══════════════════════════════════════════
            st.header("🗺️ Return Heatmap")
            st.markdown(
                f"All instruments colored by **{heatmap_period_label} return**. "
                "Grouped by fund type. Hover for details."
            )

            # Use all data (not filtered by fund type) for the heatmap
            heatmap_df = _all_data.copy()

            # Clean up for display
            heatmap_df["return_val"] = heatmap_df[heatmap_period_col]
            heatmap_df["return_display"] = heatmap_df[heatmap_period_col].apply(
                lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
            )

            # Label for treemap
            heatmap_df["label"] = (
                heatmap_df["ticker"] + "\n" + heatmap_df["return_display"]
            )

            # Size metric
            if heatmap_size_metric == "1Y Volatility":
                heatmap_df["size"] = heatmap_df["vol_1y"].clip(lower=0.1).fillna(1)
            else:
                heatmap_df["size"] = 1

            # Drop rows with NaN returns
            heatmap_df = heatmap_df.dropna(subset=[heatmap_period_col])

            # Fund type display names
            fund_type_names = {
                "eq": "Equity",
                "eq-reit": "Real Estate",
                "commod": "Commodities",
                "bonds": "Bonds",
                "bonds-em": "EM Bonds",
                "bonds-corp": "Corp Bonds",
                "bonds-il": "IL Bonds",
                "bonds-cash": "Cash",
            }
            heatmap_df["category"] = heatmap_df["fund_type"].map(
                lambda x: fund_type_names.get(x, x)
            )

            # Clamp color range for better contrast
            color_max = max(
                abs(heatmap_df["return_val"].quantile(0.05)),
                abs(heatmap_df["return_val"].quantile(0.95)),
                5,
            )

            fig_treemap = px.treemap(
                heatmap_df,
                path=["category", "description"],
                values="size",
                color="return_val",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                range_color=[-color_max, color_max],
                custom_data=["ticker", "return_display", "fund_type"],
                title=f"Return Heatmap — {heatmap_period_label}",
            )

            fig_treemap.update_traces(
                textinfo="label+text",
                texttemplate="%{label}<br>%{customdata[1]}",
                hovertemplate=(
                    "<b>%{customdata[0]}</b> — %{label}<br>"
                    f"{heatmap_period_label} Return: %{{customdata[1]}}<br>"
                    "Fund Type: %{customdata[2]}<extra></extra>"
                ),
                textfont=dict(size=14),
            )
            fig_treemap.update_layout(
                height=700,
                margin=dict(t=50, l=10, r=10, b=10),
            )
            st.plotly_chart(fig_treemap, use_container_width=True)

            # Also show a classic heatmap grid (like the ThematicDashboard)
            st.subheader("📊 Multi-Period Return Grid")
            st.markdown(
                "All instruments across all return periods — sorted by 1Y return."
            )

            grid_df = _all_data.copy()
            if fund_type_filter:
                pattern = "^(" + "|".join(fund_type_filter) + ")"
                grid_df = grid_df[grid_df["fund_type"].str.match(pattern)]

            grid_df = grid_df.sort_values("r_1y", ascending=False).reset_index(
                drop=True
            )
            grid_heatmap = grid_df.set_index("description")[heatmap_periods].copy()
            grid_heatmap.columns = heatmap_labels

            # Force numeric to ensure robust_max works
            for col in grid_heatmap.columns:
                grid_heatmap[col] = pd.to_numeric(grid_heatmap[col], errors="coerce")

            # Robust color scaling
            all_vals = grid_heatmap.values.flatten()
            all_vals = all_vals[pd.notna(all_vals)]

            if len(all_vals) > 0:
                robust_max = float(pd.Series(all_vals).abs().quantile(0.95))
                robust_max = max(robust_max, 15.0)
            else:
                robust_max = 20.0

            fig_grid = px.imshow(
                grid_heatmap.values,
                x=heatmap_labels,
                y=list(grid_heatmap.index),
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                text_auto=".1f",
                aspect="auto",
                labels=dict(color="Return %"),
            )

            fig_grid.update_layout(
                height=max(500, len(grid_heatmap) * 22),
                coloraxis=dict(cmin=-robust_max, cmax=robust_max),
            )
            st.plotly_chart(fig_grid, use_container_width=True)

        section_heatmap()
