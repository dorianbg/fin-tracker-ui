"""Bull-market consolidation scanner for stocks and ETFs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def _latest(series: pd.Series) -> float:
    return float(series.iloc[-1]) if len(series) else np.nan


def scan_consolidation_setups(
    prices: pd.DataFrame,
    adr_window: int = 20,
    ma_window: int = 200,
    slope_window: int = 20,
    consolidation_window: int = 30,
    max_extension_adr: float = 4.0,
    max_breakout_gap_adr: float = 2.0,
    max_consolidation_range_adr: float = 8.0,
) -> pd.DataFrame:
    """Return assets in bull regimes that are coiling below resistance.

    The regime rule mirrors the source notes: separate slow regime detection from
    trade filtering, use the 200-day slope, and add an ADR band to avoid binary
    above/below-MA whipsaws.
    """
    required = {"ticker", "date", "price", "price_high", "price_low"}
    if prices.empty or not required.issubset(prices.columns):
        return pd.DataFrame()

    rows = []
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    min_rows = ma_window + slope_window + consolidation_window
    for ticker, group in df.groupby("ticker"):
        group = group.dropna(subset=["price", "price_high", "price_low"]).copy()
        if len(group) < min_rows:
            continue

        close = group["price"].astype(float)
        high = group["price_high"].astype(float)
        low = group["price_low"].astype(float)
        adr = (high - low).rolling(adr_window).mean()
        ma = close.rolling(ma_window).mean()
        resistance = high.shift(1).rolling(consolidation_window).max()
        support = low.shift(1).rolling(consolidation_window).min()

        latest_price = _latest(close)
        latest_adr = _latest(adr)
        latest_ma = _latest(ma)
        prior_ma = float(ma.iloc[-slope_window - 1])
        latest_resistance = _latest(resistance)
        latest_support = _latest(support)
        if not np.isfinite(
            [
                latest_price,
                latest_adr,
                latest_ma,
                prior_ma,
                latest_resistance,
                latest_support,
            ]
        ).all():
            continue
        if latest_adr <= 0:
            continue

        ma_slope_adr = (latest_ma - prior_ma) / latest_adr
        distance_from_ma_adr = (latest_price - latest_ma) / latest_adr
        breakout_gap_adr = (latest_resistance - latest_price) / latest_adr
        consolidation_range_adr = (latest_resistance - latest_support) / latest_adr

        lower_band = latest_ma - latest_adr
        upper_band = latest_ma + latest_adr
        if latest_price > upper_band and ma_slope_adr > 0:
            regime = "Bull"
        elif latest_price < lower_band or ma_slope_adr < 0:
            regime = "Bear"
        else:
            regime = "Neutral"

        if regime != "Bull":
            continue
        if distance_from_ma_adr > max_extension_adr:
            continue
        if breakout_gap_adr < 0 or breakout_gap_adr > max_breakout_gap_adr:
            continue
        if consolidation_range_adr > max_consolidation_range_adr:
            continue

        compression_score = max_consolidation_range_adr - consolidation_range_adr
        proximity_score = max_breakout_gap_adr - breakout_gap_adr
        trend_score = max(ma_slope_adr, 0)
        setup_score = (
            compression_score
            + proximity_score
            + trend_score
            - max(distance_from_ma_adr, 0) * 0.25
        )

        rows.append(
            {
                "ticker": ticker,
                "description": group.get("description", pd.Series([ticker])).iloc[-1],
                "fund_type": group.get("fund_type", pd.Series([""])).iloc[-1],
                "date": group["date"].iloc[-1],
                "price": latest_price,
                "regime": regime,
                "ma200": latest_ma,
                "adr20": latest_adr,
                "ma200_slope_adr": ma_slope_adr,
                "extension_adr": distance_from_ma_adr,
                "breakout_level": latest_resistance,
                "breakout_gap_adr": breakout_gap_adr,
                "consolidation_range_adr": consolidation_range_adr,
                "setup_score": setup_score,
            }
        )

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values("setup_score", ascending=False)
        .reset_index(drop=True)
    )


def scan_breakout_triggers(
    prices: pd.DataFrame,
    adr_window: int = 20,
    ma_window: int = 200,
    slope_window: int = 20,
    consolidation_window: int = 30,
    max_breakout_extension_adr: float = 1.5,
    max_extension_adr: float = 8.0,
) -> pd.DataFrame:
    """Return fresh breakouts above prior consolidation resistance that are not overextended."""
    required = {"ticker", "date", "price", "price_high", "price_low"}
    if prices.empty or not required.issubset(prices.columns):
        return pd.DataFrame()

    rows = []
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    min_rows = ma_window + slope_window + consolidation_window
    for ticker, group in df.groupby("ticker"):
        group = group.dropna(subset=["price", "price_high", "price_low"]).copy()
        if len(group) < min_rows:
            continue

        close = group["price"].astype(float)
        high = group["price_high"].astype(float)
        low = group["price_low"].astype(float)
        adr = (high - low).rolling(adr_window).mean()
        ma = close.rolling(ma_window).mean()
        resistance = high.shift(1).rolling(consolidation_window).max()

        latest_price = _latest(close)
        previous_price = float(close.iloc[-2])
        latest_adr = _latest(adr)
        latest_ma = _latest(ma)
        prior_ma = float(ma.iloc[-slope_window - 1])
        latest_resistance = _latest(resistance)
        if not np.isfinite(
            [
                latest_price,
                previous_price,
                latest_adr,
                latest_ma,
                prior_ma,
                latest_resistance,
            ]
        ).all():
            continue
        if latest_adr <= 0:
            continue

        ma_slope_adr = (latest_ma - prior_ma) / latest_adr
        lower_band = latest_ma - latest_adr
        upper_band = latest_ma + latest_adr
        is_bull = latest_price > upper_band and ma_slope_adr > 0
        crossed_today = previous_price <= latest_resistance < latest_price
        distance_from_ma_adr = (latest_price - latest_ma) / latest_adr
        breakout_extension_adr = (latest_price - latest_resistance) / latest_adr
        if not is_bull or not crossed_today:
            continue
        if distance_from_ma_adr > max_extension_adr:
            continue
        if breakout_extension_adr > max_breakout_extension_adr:
            continue

        breakout_score = (
            ma_slope_adr
            + max(max_breakout_extension_adr - breakout_extension_adr, 0)
            + max(max_extension_adr - distance_from_ma_adr, 0) * 0.5
        )

        rows.append(
            {
                "ticker": ticker,
                "description": group.get("description", pd.Series([ticker])).iloc[-1],
                "fund_type": group.get("fund_type", pd.Series([""])).iloc[-1],
                "date": group["date"].iloc[-1],
                "price": latest_price,
                "breakout_level": latest_resistance,
                "breakout_extension_adr": breakout_extension_adr,
                "extension_adr": distance_from_ma_adr,
                "ma200": latest_ma,
                "ma200_slope_adr": ma_slope_adr,
                "adr20": latest_adr,
                "breakout_score": breakout_score,
            }
        )

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values("breakout_score", ascending=False)
        .reset_index(drop=True)
    )


@st.cache_data(ttl=300)
def load_price_history() -> pd.DataFrame:
    import duckdb_importer as di
    from data import get_conn

    query = f"""
        SELECT
            ticker,
            ticker_full,
            date,
            price_orig AS price,
            high_orig AS price_high,
            low_orig AS price_low,
            description,
            fund_type,
            currency
        FROM {di.px_tbl}
        ORDER BY ticker, date
    """
    df = get_conn().execute(query).df()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df


def render():
    from data import add_sparkline_column, filter_by_fund_type, fund_type_sidebar

    st.title("Bull Consolidation Setups")
    st.markdown(
        "Find stocks and ETFs in a bull regime that are **not overextended**, have **compressed sideways ranges**, "
        "and are **near but not through** breakout resistance. Signals use each asset's native listing price."
    )

    settings_col, content_col = st.columns([1, 4], gap="large")
    with settings_col:
        st.subheader("Scanner Settings")
        fund_types = fund_type_sidebar(
            default=["eq", "stock"], key="consolidation_fund_types"
        )
        max_extension_adr = st.slider(
            "Max extension from 200D MA (ADR)", 1.0, 8.0, 4.0, 0.5
        )
        max_breakout_gap_adr = st.slider(
            "Max gap to breakout (ADR)", 0.5, 5.0, 2.0, 0.5
        )
        max_range_adr = st.slider("Max consolidation range (ADR)", 3.0, 15.0, 8.0, 0.5)
        top_n = st.slider("Rows to show", 10, 100, 40, 10)

    with content_col:
        prices = filter_by_fund_type(load_price_history(), fund_types)
        setups = scan_consolidation_setups(
            prices,
            max_extension_adr=max_extension_adr,
            max_breakout_gap_adr=max_breakout_gap_adr,
            max_consolidation_range_adr=max_range_adr,
        )
        if setups.empty:
            st.info("No consolidation setups match the current thresholds.")
            return

        st.caption(
            "Bull regime = price above a 1-ADR band around the 200-day average and 200-day average sloping up over 20 days. "
            "Breakout gap must be positive, so names already above resistance are excluded."
        )
        show = setups.head(top_n).copy()
        add_sparkline_column(show)

        display_cols = [
            "description",
            "ticker",
            "fund_type",
            "Price (90d)",
            "setup_score",
            "extension_adr",
            "breakout_gap_adr",
            "consolidation_range_adr",
            "ma200_slope_adr",
            "price",
            "breakout_level",
        ]
        st.dataframe(
            show[display_cols],
            hide_index=True,
            height=650,
            column_config={
                "Price (90d)": st.column_config.LineChartColumn(
                    "Price (90d)", width="small"
                ),
                "description": st.column_config.TextColumn(
                    "description", width="medium"
                ),
                "setup_score": st.column_config.NumberColumn("Score", format="%.2f"),
                "extension_adr": st.column_config.NumberColumn(
                    "Extension ADR", format="%.2f"
                ),
                "breakout_gap_adr": st.column_config.NumberColumn(
                    "Gap ADR", format="%.2f"
                ),
                "consolidation_range_adr": st.column_config.NumberColumn(
                    "Range ADR", format="%.2f"
                ),
                "ma200_slope_adr": st.column_config.NumberColumn(
                    "200D Slope ADR", format="%.2f"
                ),
            },
        )

        fig = px.scatter(
            show,
            x="breakout_gap_adr",
            y="consolidation_range_adr",
            color="extension_adr",
            size="setup_score",
            text="ticker",
            color_continuous_scale="Viridis_r",
            labels={
                "breakout_gap_adr": "ADR to breakout",
                "consolidation_range_adr": "Consolidation range in ADR",
                "extension_adr": "Extension from 200D MA in ADR",
            },
            title="Best setups are lower-left: tight range, near breakout, not extended",
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(height=550)
        st.plotly_chart(fig, width="stretch")
