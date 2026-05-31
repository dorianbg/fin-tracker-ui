"""
Today's Crossings — a daily alert page showing instruments that crossed key
thresholds today: new 52W highs, MA crossovers, z-score spikes, big movers.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from data import load_latest_perf, add_sparkline_column

try:
    from strategy_scanners import (
        scan_ma_crossovers,
        scan_new_52w_highs,
        scan_z_score_spikes,
        split_latest_two_rows,
    )
except ModuleNotFoundError:
    from app.strategy_scanners import (
        scan_ma_crossovers,
        scan_new_52w_highs,
        scan_z_score_spikes,
        split_latest_two_rows,
    )


def _add_sparkline_columns(df: pd.DataFrame) -> pd.DataFrame:
    add_sparkline_column(df)
    add_sparkline_column(df, col_name="Price (1y)", days=365)
    return df


def _sparkline_config() -> dict:
    return {
        "Price (90d)": st.column_config.LineChartColumn("Price (90d)", width="small"),
        "Price (1y)": st.column_config.LineChartColumn("Price (1y)", width="small"),
    }


def render():
    st.title("🚨 Today's Crossings")
    st.markdown(
        "Instruments that hit notable thresholds **today**. "
        "Use this as a daily watchlist / alert screen."
    )

    raw = load_latest_perf(max_rown=2)

    if raw.empty:
        st.warning("No data loaded.")
        st.stop()

    today, yesterday = split_latest_two_rows(raw)

    data_date = today["date"].iloc[0] if "date" in today.columns else "Unknown"
    st.markdown(
        f"**Data as of**: {data_date} &nbsp;|&nbsp; **Instruments tracked**: {len(today)}"
    )
    st.markdown("---")

    # Streamlit sidebars are global, so keep tab-specific controls in-page.
    settings_col, content_col = st.columns([1, 4], gap="large")

    with settings_col:
        st.subheader("Crossing Settings")
        z_threshold = st.slider("Z-score threshold", 1.0, 4.0, 2.0, 0.5)

    with content_col:
        # ═══════════════════════════════════════════
        # Section 1: New 52-Week Highs & Lows
        # ═══════════════════════════════════════════
        st.header("🏔️ New 52-Week Highs")

        new_highs = scan_new_52w_highs(today, yesterday)

        if new_highs.empty:
            st.info("No new 52-week highs today.")
        else:
            st.success(f"**{len(new_highs)}** instrument(s) hit new 52-week highs!")
            display = new_highs[
                [
                    "description",
                    "fund_type",
                    "r_1d",
                    "r_1w",
                    "r_1mo",
                    "drawdown_52w",
                    "prev_drawdown",
                ]
            ].copy()
            display["ticker"] = display.index
            _add_sparkline_columns(display)

            # Reorder to include Price (90d)
            display = display[
                [
                    "description",
                    "fund_type",
                    "r_1d",
                    "r_1w",
                    "r_1mo",
                    "drawdown_52w",
                    "prev_drawdown",
                    "Price (90d)",
                    "Price (1y)",
                    "ticker",
                ]
            ]

            display.columns = [
                "Instrument",
                "Type",
                "1D Return",
                "1W Return",
                "1M Return",
                "Drawdown Today",
                "Drawdown Yesterday",
                "Price (90d)",
                "Price (1y)",
                "ticker",
            ]
            st.dataframe(
                display.style.format(
                    subset=[
                        "1D Return",
                        "1W Return",
                        "1M Return",
                        "Drawdown Today",
                        "Drawdown Yesterday",
                    ],
                    formatter="{:+.2f}%",
                ),
                column_config={
                    **_sparkline_config(),
                    "Instrument": st.column_config.TextColumn(
                        "Instrument", width="medium"
                    ),
                    "ticker": st.column_config.TextColumn("ticker", width="small"),
                    "Type": st.column_config.TextColumn("Type", width="small"),
                    "1D Return": st.column_config.NumberColumn(
                        "1D Return", format="%.2f%%", width="small"
                    ),
                    "1W Return": st.column_config.NumberColumn(
                        "1W Return", format="%.2f%%", width="small"
                    ),
                    "1M Return": st.column_config.NumberColumn(
                        "1M Return", format="%.2f%%", width="small"
                    ),
                    "Drawdown Today": st.column_config.NumberColumn(
                        "Drawdown Today", format="%.2f%%", width="small"
                    ),
                    "Drawdown Yesterday": st.column_config.NumberColumn(
                        "Drawdown Yesterday", format="%.2f%%", width="small"
                    ),
                },
                height=min(300, 50 + len(display) * 35),
            )

        st.markdown("---")

        # Near 52W highs (within -2%)
        near_highs = today[
            (today["drawdown_52w"] >= -2) & (today["drawdown_52w"] < -0.5)
        ].copy()
        if not near_highs.empty:
            st.subheader(f"🔶 Near 52W Highs ({len(near_highs)} instruments within 2%)")
            display_near = near_highs[
                ["description", "fund_type", "r_1d", "r_1w", "drawdown_52w"]
            ].copy()
            display_near["ticker"] = display_near.index
            _add_sparkline_columns(display_near)

            # Reorder
            display_near = display_near[
                [
                    "description",
                    "fund_type",
                    "r_1d",
                    "r_1w",
                    "drawdown_52w",
                    "Price (90d)",
                    "Price (1y)",
                    "ticker",
                ]
            ]

            display_near.columns = [
                "Instrument",
                "Type",
                "1D Return",
                "1W Return",
                "Drawdown",
                "Price (90d)",
                "Price (1y)",
                "ticker",
            ]
            st.dataframe(
                display_near.sort_values("Drawdown", ascending=False).style.format(
                    subset=["1D Return", "1W Return", "Drawdown"],
                    formatter="{:+.2f}%",
                ),
                column_config={
                    **_sparkline_config(),
                    "Instrument": st.column_config.TextColumn(
                        "Instrument", width="medium"
                    ),
                    "ticker": st.column_config.TextColumn("ticker", width="small"),
                    "Type": st.column_config.TextColumn("Type", width="small"),
                    "1D Return": st.column_config.NumberColumn(
                        "1D Return", format="%.2f%%", width="small"
                    ),
                    "1W Return": st.column_config.NumberColumn(
                        "1W Return", format="%.2f%%", width="small"
                    ),
                    "Drawdown": st.column_config.NumberColumn(
                        "Drawdown", format="%.2f%%", width="small"
                    ),
                },
                height=min(300, 50 + len(display_near) * 35),
            )

        # ═══════════════════════════════════════════
        # Section 2: MA Crossovers
        # ═══════════════════════════════════════════
        st.header("📐 Moving Average Crossovers")
        st.markdown(
            "Instruments that **crossed above or below** a moving average today."
        )

        ma_cols = {
            "21d MA": "ma_21",
            "63d MA": "ma_63",
            "126d MA": "ma_126",
            "252d MA": "ma_252",
        }

        cross_df = scan_ma_crossovers(today, yesterday)
        if not cross_df.empty:
            cross_df["ticker"] = cross_df["Ticker"]
            _add_sparkline_columns(cross_df)

            st.markdown(f"**{len(cross_df)}** crossover event(s) detected.")

            # show bullish and bearish separately
            bullish = cross_df[cross_df["Direction"].str.contains("ABOVE")]
            bearish = cross_df[cross_df["Direction"].str.contains("BELOW")]

            col_b, col_s = st.columns(2)
            with col_b:
                st.subheader("🟢 Bullish Crossovers")
                if not bullish.empty:
                    st.dataframe(
                        bullish[
                            [
                                "Instrument",
                                "MA",
                                "Today",
                                "Yesterday",
                                "1D Return",
                                "Price (90d)",
                                "Price (1y)",
                            ]
                        ].style.format(
                            subset=["Today", "Yesterday", "1D Return"],
                            formatter="{:+.2f}%",
                        ),
                        column_config={
                            **_sparkline_config(),
                            "Instrument": st.column_config.TextColumn(
                                "Instrument", width="medium"
                            ),
                            "MA": st.column_config.TextColumn("MA", width="small"),
                            "Today": st.column_config.NumberColumn(
                                "Today", format="%.2f%%", width="small"
                            ),
                            "Yesterday": st.column_config.NumberColumn(
                                "Yesterday", format="%.2f%%", width="small"
                            ),
                            "1D Return": st.column_config.NumberColumn(
                                "1D Return", format="%.2f%%", width="small"
                            ),
                        },
                        hide_index=True,
                    )
                else:
                    st.info("None today.")

            with col_s:
                st.subheader("🔴 Bearish Crossovers")
                if not bearish.empty:
                    st.dataframe(
                        bearish[
                            [
                                "Instrument",
                                "MA",
                                "Today",
                                "Yesterday",
                                "1D Return",
                                "Price (90d)",
                                "Price (1y)",
                            ]
                        ].style.format(
                            subset=["Today", "Yesterday", "1D Return"],
                            formatter="{:+.2f}%",
                        ),
                        column_config={
                            **_sparkline_config(),
                            "Instrument": st.column_config.TextColumn(
                                "Instrument", width="medium"
                            ),
                            "MA": st.column_config.TextColumn("MA", width="small"),
                            "Today": st.column_config.NumberColumn(
                                "Today", format="%.2f%%", width="small"
                            ),
                            "Yesterday": st.column_config.NumberColumn(
                                "Yesterday", format="%.2f%%", width="small"
                            ),
                            "1D Return": st.column_config.NumberColumn(
                                "1D Return", format="%.2f%%", width="small"
                            ),
                        },
                        hide_index=True,
                    )
                else:
                    st.info("None today.")
        else:
            st.info("No MA crossover events today.")

        # ═══════════════════════════════════════════
        # Section 3: Z-Score Spikes
        # ═══════════════════════════════════════════
        st.header("⚡ Unusual Moves (Z-Score Spikes)")
        st.markdown(
            "Instruments with a z-score > 2 on any lookback — statistically unusual moves."
        )

        z_df = scan_z_score_spikes(today, z_threshold)
        if not z_df.empty:
            z_df["ticker"] = z_df["Ticker"]
            _add_sparkline_columns(z_df)

            st.markdown(
                f"**{len(z_df)}** unusual move(s) detected (z ≥ {z_threshold})."
            )
            st.dataframe(
                z_df[
                    [
                        "Instrument",
                        "Metric",
                        "Z-Score",
                        "1D Return",
                        "1W Return",
                        "Price (90d)",
                        "Price (1y)",
                    ]
                ]
                .style.format(
                    subset=["Z-Score"],
                    formatter="{:.2f}",
                )
                .format(
                    subset=["1D Return", "1W Return"],
                    formatter="{:+.2f}%",
                ),
                column_config={
                    **_sparkline_config(),
                    "Instrument": st.column_config.TextColumn(
                        "Instrument", width="medium"
                    ),
                    "Metric": st.column_config.TextColumn("Metric", width="small"),
                    "Z-Score": st.column_config.NumberColumn(
                        "Z-Score", format="%.2f", width="small"
                    ),
                    "1D Return": st.column_config.NumberColumn(
                        "1D Return", format="%.2f%%", width="small"
                    ),
                    "1W Return": st.column_config.NumberColumn(
                        "1W Return", format="%.2f%%", width="small"
                    ),
                },
                hide_index=True,
                height=min(400, 50 + len(z_df) * 35),
            )
        else:
            st.info(f"No z-score spikes above {z_threshold} today.")

        # ═══════════════════════════════════════════
        # Section 4: Biggest Daily Movers
        # ═══════════════════════════════════════════
        st.header("🔥 Biggest Daily Movers")

        col_up, col_dn = st.columns(2)

        # top 10 gainers
        top_gainers = today.nlargest(10, "r_1d")[
            ["description", "fund_type", "r_1d", "r_1w"]
        ].copy()
        top_gainers.columns = ["Instrument", "Type", "1D Return", "1W Return"]

        # top 10 losers
        top_losers = today.nsmallest(10, "r_1d")[
            ["description", "fund_type", "r_1d", "r_1w"]
        ].copy()
        top_losers.columns = ["Instrument", "Type", "1D Return", "1W Return"]

        with col_up:
            st.subheader("🟢 Top Gainers")
            fig_gain = px.bar(
                top_gainers,
                x="1D Return",
                y="Instrument",
                orientation="h",
                color="1D Return",
                color_continuous_scale="Greens",
                labels={"1D Return": "Return (%)", "Instrument": ""},
            )
            fig_gain.update_layout(
                yaxis=dict(autorange="reversed"), height=350, showlegend=False
            )
            st.plotly_chart(fig_gain, width="stretch")

        with col_dn:
            st.subheader("🔴 Top Losers")
            fig_lose = px.bar(
                top_losers,
                x="1D Return",
                y="Instrument",
                orientation="h",
                color="1D Return",
                color_continuous_scale="Reds_r",
                labels={"1D Return": "Return (%)", "Instrument": ""},
            )
            fig_lose.update_layout(
                yaxis=dict(autorange="reversed"), height=350, showlegend=False
            )
            st.plotly_chart(fig_lose, width="stretch")

        # ═══════════════════════════════════════════
        # Section 5: Summary Metrics
        # ═══════════════════════════════════════════
        st.header("📊 Market Breadth")

        # % above each MA
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        for col_widget, (ma_label, ma_col) in zip(
            [col_m1, col_m2, col_m3, col_m4],
            ma_cols.items(),
        ):
            pct_above = (today[ma_col] > 0).mean() * 100
            with col_widget:
                st.metric(f"% above {ma_label}", f"{pct_above:.0f}%")

        # advance/decline
        advancing = (today["r_1d"] > 0).sum()
        declining = (today["r_1d"] < 0).sum()
        unchanged = (today["r_1d"] == 0).sum()
        col_ad1, col_ad2, col_ad3 = st.columns(3)
        with col_ad1:
            st.metric("Advancing", f"{advancing}")
        with col_ad2:
            st.metric("Declining", f"{declining}")
        with col_ad3:
            st.metric("A/D Ratio", f"{advancing / max(declining, 1):.2f}")
