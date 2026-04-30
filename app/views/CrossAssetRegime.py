"""
Cross-Asset Regime View — compare equities, bonds, commodities, REITs, and cash
side-by-side to identify risk-on / risk-off / inflationary / deflationary regimes.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data import load_latest_perf, load_prices


def render():
    st.title("🌍 Cross-Asset Regime View")

    # ── Representative instruments per asset class ──
    ASSET_CLASS_MAP = {
        # Equities — broad
        "US Equity (S&P 500)": {"ticker": "CSP1", "class": "Equity", "sub": "Broad"},
        "US Small Cap (R2000)": {"ticker": "IWM", "class": "Equity", "sub": "Broad"},
        "Europe Equity": {"ticker": "IMEA", "class": "Equity", "sub": "Broad"},
        "UK Equity (FTSE 100)": {"ticker": "VUKG", "class": "Equity", "sub": "Broad"},
        "EM Equity": {"ticker": "EEM", "class": "Equity", "sub": "Broad"},
        "Japan Equity": {"ticker": "EWJ", "class": "Equity", "sub": "Broad"},
        "India Equity": {"ticker": "INDA", "class": "Equity", "sub": "Broad"},
        "Brazil Equity": {"ticker": "EWZ", "class": "Equity", "sub": "Broad"},
        "Australia Equity": {"ticker": "EWA", "class": "Equity", "sub": "Broad"},
        # Equities — sectors
        "US Financials": {"ticker": "XLF", "class": "Equity", "sub": "Sector"},
        "US Technology": {"ticker": "XLK", "class": "Equity", "sub": "Sector"},
        "US Energy": {"ticker": "XLE", "class": "Equity", "sub": "Sector"},
        "US Health Care": {"ticker": "XLV", "class": "Equity", "sub": "Sector"},
        "US Utilities": {"ticker": "XLU", "class": "Equity", "sub": "Sector"},
        "US Industrials": {"ticker": "XLI", "class": "Equity", "sub": "Sector"},
        # Bonds
        "US Aggregate Bond": {"ticker": "AGG", "class": "Bonds", "sub": "Core"},
        "US 7-10y Treasury": {"ticker": "IEF", "class": "Bonds", "sub": "Core"},
        "US 20+ Treasury": {"ticker": "TLT", "class": "Bonds", "sub": "Core"},
        "EM USD Bond": {"ticker": "EMB", "class": "Bonds", "sub": "EM"},
        "US High Yield": {"ticker": "HYG", "class": "Bonds", "sub": "Credit"},
        "US TIPS": {"ticker": "TIP", "class": "Bonds", "sub": "IL"},
        "US 0-5Y TIPS": {"ticker": "STIP", "class": "Bonds", "sub": "IL"},
        "UK Gilts 15+": {"ticker": "GLTL", "class": "Bonds", "sub": "Core"},
        # Commodities
        "Gold": {"ticker": "GLD", "class": "Commodity", "sub": "Precious"},
        "Broad Commodities": {"ticker": "GSG", "class": "Commodity", "sub": "Broad"},
        # REITs
        "Global Property": {"ticker": "DPYA", "class": "REIT", "sub": "Global"},
        "US Property": {"ticker": "IUSP", "class": "REIT", "sub": "US"},
        # Cash
        "UK Cash Rate": {"ticker": "CSH2", "class": "Cash", "sub": "Cash"},
        # Infrastructure / Alternatives
        "Global Infrastructure": {
            "ticker": "INFR",
            "class": "Alternative",
            "sub": "Infra",
        },
        "Clean Energy": {"ticker": "ICLN", "class": "Alternative", "sub": "Thematic"},
        "Gold Miners": {"ticker": "GDX", "class": "Alternative", "sub": "Mining"},
    }

    ALL_TICKERS = [v["ticker"] for v in ASSET_CLASS_MAP.values()]

    # Subset of return columns relevant for cross-asset comparison
    RETURN_PERIODS = ["r_1w", "r_1mo", "r_3mo", "r_6mo", "r_1y", "r_2y", "r_3y"]
    RETURN_LABELS = ["1W", "1M", "3M", "6M", "1Y", "2Y", "3Y"]

    df = load_latest_perf(tickers=tuple(ALL_TICKERS))

    if df.empty:
        st.warning("No regime data found.")
        st.stop()

    # map ticker → display name, asset class, and sub-class
    ticker_to_name = {v["ticker"]: name for name, v in ASSET_CLASS_MAP.items()}
    ticker_to_class = {v["ticker"]: v["class"] for v in ASSET_CLASS_MAP.values()}
    ticker_to_sub = {v["ticker"]: v["sub"] for v in ASSET_CLASS_MAP.values()}
    df["asset_name"] = df["ticker"].map(ticker_to_name)
    df["asset_class"] = df["ticker"].map(ticker_to_class)
    df["sub_class"] = df["ticker"].map(ticker_to_sub)

    # order by asset class for visual grouping
    class_order = ["Equity", "Bonds", "Commodity", "REIT", "Alternative", "Cash"]
    df["class_rank"] = df["asset_class"].map({c: i for i, c in enumerate(class_order)})
    df = df.sort_values(["class_rank", "asset_name"]).reset_index(drop=True)

    # Streamlit sidebars are global, so keep tab-specific controls in-page.
    settings_col, content_col = st.columns([1, 4], gap="large")

    with settings_col:
        st.subheader("Regime Settings")
        period_sel = st.selectbox("Period", options=RETURN_LABELS, index=4)
        period_col = RETURN_PERIODS[RETURN_LABELS.index(period_sel)]

        corr_window = st.slider("Rolling correlation window (days)", 21, 126, 63, 21)

    with content_col:
        # ═══════════════════════════════════════════
        # Section 1: Cross-Asset Return Heatmap
        # ═══════════════════════════════════════════
        st.header("📊 Return Heatmap")
        st.markdown(
            "Green = positive returns, Red = negative. Compare across asset classes to spot regime shifts."
        )

        heatmap_data = df.set_index("asset_name")[RETURN_PERIODS].copy()
        heatmap_data.columns = RETURN_LABELS

        fig = px.imshow(
            heatmap_data.values,
            x=RETURN_LABELS,
            y=list(heatmap_data.index),
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            text_auto=".1f",
            aspect="auto",
            labels=dict(color="Return %"),
        )
        fig.update_layout(height=max(500, len(heatmap_data) * 28))
        st.plotly_chart(fig, use_container_width=True)

        # ═══════════════════════════════════════════
        # Section 2: Regime Indicator
        # ═══════════════════════════════════════════
        st.header("🚦 Regime Indicator")

        # average return per asset class
        class_avg = (
            df.groupby("asset_class")[period_col].mean().reindex(class_order).dropna()
        )

        col1, col2 = st.columns([2, 3])

        with col1:
            # regime rules of thumb
            eq_r = class_avg.get("Equity", 0)
            bond_r = class_avg.get("Bonds", 0)
            commod_r = class_avg.get("Commodity", 0)

            if eq_r > 0 and bond_r < 0 and commod_r > 0:
                regime = "🟢 **Risk-On / Inflationary**"
                desc = "Equities and commodities up, bonds down — classic reflationary/growth environment."
            elif eq_r > 0 and bond_r > 0:
                regime = "🟡 **Goldilocks**"
                desc = (
                    "Both equities and bonds positive — supportive monetary conditions."
                )
            elif eq_r < 0 and bond_r > 0:
                regime = "🔵 **Risk-Off / Flight to Safety**"
                desc = "Equities falling, bonds rising — investors seeking safety."
            elif eq_r < 0 and bond_r < 0:
                regime = "🔴 **Stagflation / Tightening**"
                desc = "Both equities and bonds negative — tough environment, possible rate hikes."
            elif eq_r > 0 and commod_r < 0:
                regime = "🟢 **Disinflationary Growth**"
                desc = (
                    "Equities up, commodities down — growth without inflation pressure."
                )
            else:
                regime = "⚪ **Mixed**"
                desc = "No clear regime signal."

            st.markdown(f"### {regime}")
            st.markdown(desc)
            st.markdown("---")
            for cls, ret in class_avg.items():
                emoji = "🟢" if ret > 0 else "🔴"
                st.markdown(f"{emoji} **{cls}**: {ret:+.2f}%")

        with col2:
            fig_bar = px.bar(
                df,
                x=period_col,
                y="asset_name",
                color="asset_class",
                orientation="h",
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={period_col: f"Return ({period_sel}) %", "asset_name": ""},
                title=f"Returns by Asset Class — {period_sel}",
            )
            fig_bar.update_layout(
                yaxis=dict(autorange="reversed"), height=max(500, len(df) * 24)
            )
            fig_bar.add_vline(x=0, line_dash="solid", line_color="grey", opacity=0.5)
            st.plotly_chart(fig_bar, use_container_width=True)

        # ═══════════════════════════════════════════
        # Section 3: Moving Average Status
        # ═══════════════════════════════════════════
        st.header("📈 Trend Status")
        st.markdown(
            "Position relative to key moving averages. Positive = above MA (uptrend), Negative = below (downtrend)."
        )

        ma_data = df[
            ["asset_name", "asset_class", "ma_21", "ma_63", "ma_126", "ma_252"]
        ].copy()

        fig_ma = px.imshow(
            ma_data.set_index("asset_name")[
                ["ma_21", "ma_63", "ma_126", "ma_252"]
            ].values,
            x=["21d MA", "63d MA", "126d MA", "252d MA"],
            y=list(ma_data["asset_name"]),
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            text_auto=".1f",
            aspect="auto",
            labels=dict(color="% from MA"),
        )
        fig_ma.update_layout(height=max(500, len(ma_data) * 28))
        st.plotly_chart(fig_ma, use_container_width=True)

        # ═══════════════════════════════════════════
        # Section 4: Volatility Regime
        # ═══════════════════════════════════════════
        st.header("🌊 Volatility Regime")
        st.markdown(
            "Compare **1-month** vs **1-year** volatility. When short-term vol exceeds long-term, "
            "markets are in a stress regime (highlighted in red)."
        )

        vol_data = (
            df[["asset_name", "asset_class", "vol_1mo", "vol_1y"]]
            .dropna(subset=["vol_1mo", "vol_1y"])
            .copy()
        )
        vol_data["vol_ratio"] = vol_data["vol_1mo"] / vol_data["vol_1y"].replace(
            0, float("nan")
        )
        vol_data["stress"] = vol_data["vol_ratio"] > 1.0

        col_v1, col_v2 = st.columns([3, 2])

        with col_v1:
            fig_vol = px.scatter(
                vol_data,
                x="vol_1y",
                y="vol_1mo",
                color="stress",
                color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
                text="asset_name",
                labels={
                    "vol_1y": "1-Year Volatility (%)",
                    "vol_1mo": "1-Month Volatility (%)",
                    "stress": "Short-term stress?",
                },
                title="Short-Term vs Long-Term Volatility",
            )
            # add 45-degree line — above = stress
            max_vol = max(vol_data["vol_1y"].max(), vol_data["vol_1mo"].max()) * 1.1
            fig_vol.add_trace(
                go.Scatter(
                    x=[0, max_vol],
                    y=[0, max_vol],
                    mode="lines",
                    line=dict(dash="dash", color="grey"),
                    showlegend=False,
                )
            )
            fig_vol.update_traces(
                textposition="top center", selector=dict(mode="markers+text")
            )
            fig_vol.update_layout(height=500)
            st.plotly_chart(fig_vol, use_container_width=True)

        with col_v2:
            # average vol ratio per asset class
            class_vol = (
                vol_data.groupby("asset_class")["vol_ratio"]
                .mean()
                .sort_values(ascending=False)
            )
            st.markdown("#### Avg Vol Ratio by Class")
            st.markdown("Values >1 = short-term vol exceeding long-term (stress)")
            for cls, ratio in class_vol.items():
                emoji = "🔴" if ratio > 1.0 else "🟢"
                st.markdown(f"{emoji} **{cls}**: {ratio:.2f}x")

            stressed = vol_data[vol_data["stress"]].sort_values(
                "vol_ratio", ascending=False
            )
            if not stressed.empty:
                st.markdown("---")
                st.markdown("#### ⚠️ Stressed Assets")
                for _, row in stressed.iterrows():
                    st.markdown(
                        f"• **{row['asset_name']}** — vol ratio: {row['vol_ratio']:.2f}x "
                        f"(1m: {row['vol_1mo']:.1f}%, 1y: {row['vol_1y']:.1f}%)"
                    )

        # ═══════════════════════════════════════════
        # Section 5: Equity-Bond-Gold Spread
        # ═══════════════════════════════════════════
        st.header("⚖️ Cross-Asset Relative Performance")
        st.markdown(
            "How different asset classes perform relative to each other across time horizons. "
            "Positive = first asset outperforms second."
        )

        # compute class-average returns per period
        class_period_avg = df.groupby("asset_class")[RETURN_PERIODS].mean()

        pairs = [
            ("Equity vs Bonds", "Equity", "Bonds"),
            ("Equity vs Commodity", "Equity", "Commodity"),
            ("Bonds vs Commodity", "Bonds", "Commodity"),
            ("Equity vs REIT", "Equity", "REIT"),
        ]

        spread_rows = []
        for label, cls_a, cls_b in pairs:
            if cls_a in class_period_avg.index and cls_b in class_period_avg.index:
                diff = class_period_avg.loc[cls_a] - class_period_avg.loc[cls_b]
                row = {"Spread": label}
                for period, lbl in zip(RETURN_PERIODS, RETURN_LABELS):
                    row[lbl] = diff[period]
                spread_rows.append(row)

        if spread_rows:
            spread_df = pd.DataFrame(spread_rows).set_index("Spread")
            fig_spread = px.imshow(
                spread_df.values,
                x=RETURN_LABELS,
                y=list(spread_df.index),
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                text_auto=".1f",
                aspect="auto",
                labels=dict(color="Spread %"),
                title="Asset Class Return Spreads",
            )
            fig_spread.update_layout(height=300)
            st.plotly_chart(fig_spread, use_container_width=True)

        # ═══════════════════════════════════════════
        # Section 6: Risk Appetite Dashboard
        # ═══════════════════════════════════════════
        st.header("🎲 Risk Appetite Signals")
        st.markdown(
            "Relative performance of risky vs safe pairs. Positive = risk appetite, Negative = risk aversion."
        )

        risk_pairs = [
            ("HY vs IG (Credit)", "HYG", "AGG"),
            ("EM vs DM Equity", "EEM", "CSP1"),
            ("Small vs Large Cap", "IWM", "CSP1"),
            ("EM vs DM Bonds", "EMB", "IEF"),
            ("Cyclicals vs Defensives", "XLI", "XLU"),
            ("Tech vs Utilities", "XLK", "XLU"),
        ]

        ra_period_col = RETURN_PERIODS[RETURN_LABELS.index(period_sel)]

        risk_rows = []
        for label, ticker_a, ticker_b in risk_pairs:
            row_a = df[df["ticker"] == ticker_a]
            row_b = df[df["ticker"] == ticker_b]
            if not row_a.empty and not row_b.empty:
                spread_val = float(row_a[ra_period_col].iloc[0]) - float(
                    row_b[ra_period_col].iloc[0]
                )
                risk_rows.append({"Pair": label, "Spread": spread_val})

        if risk_rows:
            ra_df = pd.DataFrame(risk_rows).sort_values("Spread", ascending=True)

            fig_ra = px.bar(
                ra_df,
                x="Spread",
                y="Pair",
                orientation="h",
                color="Spread",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                labels={"Spread": f"Return Spread ({period_sel}) %", "Pair": ""},
                title=f"Risk Appetite Pairs — {period_sel}",
            )
            fig_ra.update_layout(
                yaxis=dict(autorange="reversed"), height=350, showlegend=False
            )
            fig_ra.add_vline(x=0, line_dash="solid", line_color="grey", opacity=0.5)
            st.plotly_chart(fig_ra, use_container_width=True)

            # summary
            positive_count = sum(1 for r in risk_rows if r["Spread"] > 0)
            total = len(risk_rows)
            if positive_count >= total * 0.7:
                st.success(
                    f"**Risk-On**: {positive_count}/{total} pairs favour risk assets"
                )
            elif positive_count <= total * 0.3:
                st.error(
                    f"**Risk-Off**: {positive_count}/{total} pairs favour risk assets"
                )
            else:
                st.info(f"**Mixed**: {positive_count}/{total} pairs favour risk assets")

        # ═══════════════════════════════════════════
        # Section 7: Regime-Conditional Returns
        # ═══════════════════════════════════════════
        st.header("🏷️ Regime-Conditional Returns")
        st.markdown(
            "What has **historically** performed best in each regime type? "
            "Based on rolling 3-month windows classified by the same equity/bond/commodity rules."
        )

        perf_hist = load_latest_perf(tickers=tuple(ALL_TICKERS), max_rown=252)

        if not perf_hist.empty and "r_3mo" in perf_hist.columns:
            perf_hist["asset_class"] = perf_hist["ticker"].map(ticker_to_class)

            # classify each date's regime from 3-month returns
            daily_class_avg = (
                perf_hist.groupby(["date", "asset_class"])["r_3mo"].mean().unstack()
            )

            regimes = []
            for dt, row in daily_class_avg.iterrows():
                eq = row.get("Equity", 0)
                bd = row.get("Bonds", 0)
                cm = row.get("Commodity", 0)
                if pd.isna(eq) or pd.isna(bd):
                    continue
                if eq > 0 and bd < 0 and (pd.notna(cm) and cm > 0):
                    r = "Risk-On / Inflationary"
                elif eq > 0 and bd > 0:
                    r = "Goldilocks"
                elif eq < 0 and bd > 0:
                    r = "Risk-Off"
                elif eq < 0 and bd < 0:
                    r = "Stagflation"
                elif eq > 0 and (pd.isna(cm) or cm < 0):
                    r = "Disinflationary Growth"
                else:
                    r = "Mixed"
                regimes.append({"date": dt, "regime": r})

            if regimes:
                regime_df = pd.DataFrame(regimes)
                # merge regime labels back to performance data
                perf_with_regime = perf_hist.merge(regime_df, on="date", how="inner")

                # average 3-month return per asset class per regime
                regime_returns = (
                    perf_with_regime.groupby(["regime", "asset_class"])["r_3mo"]
                    .mean()
                    .unstack(fill_value=0)
                )

                if not regime_returns.empty:
                    fig_rc = px.imshow(
                        regime_returns.values,
                        x=list(regime_returns.columns),
                        y=list(regime_returns.index),
                        color_continuous_scale="RdYlGn",
                        color_continuous_midpoint=0,
                        text_auto=".1f",
                        aspect="auto",
                        labels=dict(color="Avg 3M Return %"),
                        title="Average 3-Month Return by Regime & Asset Class",
                    )
                    fig_rc.update_layout(height=350)
                    st.plotly_chart(fig_rc, use_container_width=True)

                    # regime frequency
                    regime_freq = regime_df["regime"].value_counts()
                    st.markdown("**Regime Frequency** (over last ~year of snapshots):")
                    for r_name, r_count in regime_freq.items():
                        pct = r_count / len(regime_df) * 100
                        st.markdown(f"• **{r_name}**: {r_count} days ({pct:.0f}%)")

        # ═══════════════════════════════════════════
        # Section 8: Rolling Correlation Breakdown
        # ═══════════════════════════════════════════
        st.header("🔗 Correlation Breakdown Detection")
        st.markdown(
            "Track rolling correlations between key asset pairs. "
            "When historically correlated pairs **diverge**, something is shifting in the macro environment."
        )

        corr_prices = load_prices(tickers=tuple(ALL_TICKERS))

        CORR_PAIRS = [
            ("Equity vs Bonds", "CSP1", "AGG"),
            ("Equity vs Gold", "CSP1", "GLD"),
            ("Bonds vs Gold", "AGG", "GLD"),
            ("US vs EM Equity", "CSP1", "EEM"),
            ("US vs Europe", "CSP1", "IMEA"),
            ("HY vs Treasuries", "HYG", "IEF"),
            ("Small vs Large Cap", "IWM", "CSP1"),
        ]

        pivot_corr = corr_prices.pivot_table(
            index="date", columns="ticker", values="price"
        )
        returns_corr = pivot_corr.pct_change().dropna(how="all")

        corr_lines = []
        breakdown_alerts = []

        for label, t_a, t_b in CORR_PAIRS:
            if t_a in returns_corr.columns and t_b in returns_corr.columns:
                rolling = returns_corr[t_a].rolling(corr_window).corr(returns_corr[t_b])
                for dt, val in rolling.dropna().items():
                    corr_lines.append({"date": dt, "pair": label, "correlation": val})

                # check for breakdown: current corr vs long-term average
                recent_corr = rolling.iloc[-1] if not rolling.dropna().empty else None
                long_term_corr = rolling.dropna().mean()
                if recent_corr is not None and abs(recent_corr - long_term_corr) > 0.3:
                    breakdown_alerts.append(
                        {
                            "Pair": label,
                            "Current Corr": f"{recent_corr:.2f}",
                            "Long-term Avg": f"{long_term_corr:.2f}",
                            "Shift": f"{recent_corr - long_term_corr:+.2f}",
                            "Status": "⚠️ Diverging"
                            if recent_corr < long_term_corr
                            else "🔄 Converging",
                        }
                    )

        if corr_lines:
            corr_df = pd.DataFrame(corr_lines)

            # alerts first
            if breakdown_alerts:
                st.subheader("⚠️ Correlation Breakdowns Detected")
                st.dataframe(pd.DataFrame(breakdown_alerts), hide_index=True)
            else:
                st.success("No significant correlation breakdowns detected.")

            # rolling correlation chart
            fig_corr = px.line(
                corr_df,
                x="date",
                y="correlation",
                color="pair",
                labels={
                    "correlation": f"Rolling {corr_window}d Correlation",
                    "date": "",
                    "pair": "Pair",
                },
                title=f"Rolling {corr_window}-Day Correlations",
            )
            fig_corr.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.3)
            fig_corr.update_layout(height=500, hovermode="x unified")
            st.plotly_chart(fig_corr, use_container_width=True)

            # current snapshot table
            st.subheader("📋 Current Correlation Snapshot")
            current_data = []
            for label, t_a, t_b in CORR_PAIRS:
                if t_a in returns_corr.columns and t_b in returns_corr.columns:
                    # short-term and long-term
                    short = returns_corr[t_a].rolling(21).corr(returns_corr[t_b])
                    medium = returns_corr[t_a].rolling(63).corr(returns_corr[t_b])
                    long = returns_corr[t_a].rolling(252).corr(returns_corr[t_b])
                    current_data.append(
                        {
                            "Pair": label,
                            "21d Corr": short.iloc[-1]
                            if not short.dropna().empty
                            else None,
                            "63d Corr": medium.iloc[-1]
                            if not medium.dropna().empty
                            else None,
                            "252d Corr": long.iloc[-1]
                            if not long.dropna().empty
                            else None,
                        }
                    )

            if current_data:
                snap_df = pd.DataFrame(current_data)
                st.dataframe(
                    snap_df.style.format(
                        subset=["21d Corr", "63d Corr", "252d Corr"],
                        formatter="{:.2f}",
                        na_rep="—",
                    ),
                    hide_index=True,
                )
