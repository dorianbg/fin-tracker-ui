"""Faber-style sector rotation strategy.

Ranks sector ETFs by trailing relative strength, holds the top sectors equal
weight, and optionally moves to cash when the benchmark is below its 10-month
simple moving average.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


SECTOR_UNIVERSES = {
    "US Select Sector SPDRs": {
        "benchmark": "CSP1",
        "cash": "ERNS.L",
        "tickers": [
            "XLC",
            "XLY",
            "XLP",
            "XLE",
            "XLF",
            "XLV",
            "XLI",
            "XLB",
            "XLRE",
            "XLK",
            "XLU",
        ],
    },
    "US iShares UCITS S&P 500 Sectors": {
        "benchmark": "CSP1",
        "cash": "ERNS.L",
        "tickers": [
            "IUCD.L",
            "IUCS.L",
            "IUES.L",
            "IUFS.L",
            "IHCU.L",
            "IUIS.L",
            "IUMS.L",
            "IUIT.L",
            "IUSU.L",
            "XRES.L",
        ],
    },
    "Europe iShares MSCI Europe Sectors": {
        "benchmark": "IMEA.L",
        "cash": "ERNS.L",
        "tickers": [
            "ESIC.L",
            "ESIE.L",
            "ESIF.L",
            "ESIH.L",
            "ESIN.L",
            "ESIT.L",
        ],
    },
}

LOOKBACK_MONTHS = [1, 3, 6, 9, 12]
TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_YEAR = 252


def normalise_prices(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])
    df = df.dropna(subset=["ticker", "date", "price"])
    df = df[df["price"] > 0]
    return df


def make_price_matrix(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    df = normalise_prices(prices)
    matrix = df[df["ticker"].isin(tickers)].pivot_table(
        index="date", columns="ticker", values="price", aggfunc="last"
    )
    return matrix.sort_index().ffill()


def month_end_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.Series(index, index=index).groupby(index.to_period("M")).max().values


def relative_strength_scores(
    monthly_prices: pd.DataFrame,
    lookback_months: list[int] | None = None,
) -> pd.DataFrame:
    if lookback_months is None:
        lookback_months = LOOKBACK_MONTHS

    returns = []
    for months in lookback_months:
        returns.append(monthly_prices.pct_change(months))
    return sum(returns) / len(returns)


def current_sector_ranks(
    prices: pd.DataFrame,
    tickers: list[str],
    lookback_months: list[int] | None = None,
) -> pd.DataFrame:
    matrix = make_price_matrix(prices, tickers)
    monthly = matrix.loc[month_end_index(matrix.index)]
    scores = relative_strength_scores(monthly, lookback_months).iloc[-1].dropna()
    rows = []
    for ticker, score in scores.sort_values(ascending=False).items():
        rows.append({"ticker": ticker, "rs_score": score * 100})
    return pd.DataFrame(rows)


def build_sector_rotation_backtest(
    prices: pd.DataFrame,
    sector_tickers: list[str],
    benchmark_ticker: str,
    cash_ticker: str | None,
    top_n: int,
    use_market_filter: bool,
    lookback_months: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if lookback_months is None:
        lookback_months = LOOKBACK_MONTHS

    required_tickers = list(dict.fromkeys(sector_tickers + [benchmark_ticker]))
    if cash_ticker:
        required_tickers.append(cash_ticker)
    matrix = make_price_matrix(prices, required_tickers)
    sector_matrix = matrix[sector_tickers].dropna(how="all")
    monthly = sector_matrix.loc[month_end_index(sector_matrix.index)]
    scores = relative_strength_scores(monthly, lookback_months)

    max_lookback = max(lookback_months)
    daily_returns = matrix.pct_change().fillna(0)
    cash_returns = (
        daily_returns[cash_ticker]
        if cash_ticker and cash_ticker in daily_returns.columns
        else pd.Series(0.0, index=daily_returns.index)
    )
    benchmark_returns = daily_returns[benchmark_ticker]
    benchmark_monthly = matrix[benchmark_ticker].loc[monthly.index].ffill()
    benchmark_sma = benchmark_monthly.rolling(10).mean()

    weights = pd.DataFrame(0.0, index=daily_returns.index, columns=sector_tickers)
    cash_weight = pd.Series(0.0, index=daily_returns.index)
    holdings_rows = []

    rebalance_dates = list(monthly.index[max_lookback:])
    for i, signal_date in enumerate(rebalance_dates):
        next_signal_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else daily_returns.index[-1]
        active_days = daily_returns.index[
            (daily_returns.index > signal_date) & (daily_returns.index <= next_signal_date)
        ]
        if len(active_days) == 0:
            continue

        market_ok = True
        if use_market_filter:
            market_ok = benchmark_monthly.loc[signal_date] >= benchmark_sma.loc[signal_date]

        ranked = scores.loc[signal_date].dropna().sort_values(ascending=False)
        selected = ranked.head(top_n).index.tolist() if market_ok else []

        if selected:
            weights.loc[active_days, selected] = 1 / len(selected)
        else:
            cash_weight.loc[active_days] = 1.0

        holdings_rows.append(
            {
                "signal_date": signal_date,
                "effective_from": active_days[0],
                "market_filter": "Risk on" if market_ok else "Cash",
                "holdings": ", ".join(selected) if selected else cash_ticker or "Cash",
                "rs_score": ranked.head(top_n).mean() * 100 if not ranked.empty else np.nan,
            }
        )

    strategy_returns = (weights * daily_returns[sector_tickers]).sum(axis=1) + cash_weight * cash_returns
    equity = (1 + strategy_returns).cumprod() * 100
    benchmark_equity = (1 + benchmark_returns).cumprod() * 100
    result = pd.DataFrame(
        {
            "date": daily_returns.index,
            "strategy_return": strategy_returns.values,
            "benchmark_return": benchmark_returns.values,
            "Strategy": equity.values,
            "Benchmark": benchmark_equity.values,
            "cash_weight": cash_weight.values,
        }
    )
    first_live = result[result["Strategy"] != 100].index.min()
    if pd.notna(first_live):
        result = result.loc[first_live:].reset_index(drop=True)
        base_strategy = result["Strategy"].iloc[0]
        base_benchmark = result["Benchmark"].iloc[0]
        result["Strategy"] = result["Strategy"] / base_strategy * 100
        result["Benchmark"] = result["Benchmark"] / base_benchmark * 100

    return result, pd.DataFrame(holdings_rows)


def performance_stats(backtest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, ret_col, equity_col in [
        ("Strategy", "strategy_return", "Strategy"),
        ("Benchmark", "benchmark_return", "Benchmark"),
    ]:
        returns = backtest[ret_col].dropna()
        equity = backtest[equity_col].dropna()
        years = len(returns) / TRADING_DAYS_PER_YEAR
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
        vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe = cagr / vol if vol and not np.isnan(vol) else np.nan
        drawdown = equity / equity.cummax() - 1
        rows.append(
            {
                "Series": name,
                "CAGR": cagr * 100,
                "Volatility": vol * 100,
                "Sharpe": sharpe,
                "Max Drawdown": drawdown.min() * 100,
            }
        )
    return pd.DataFrame(rows)


def render():
    from data import add_sparkline_column, load_prices

    st.title("Sector Rotation")
    st.markdown(
        "Faber-style relative strength rotation: rank sectors monthly by average "
        "1/3/6/9/12-month trailing returns, hold the top sectors equal weight, "
        "and optionally use a 10-month SMA market filter."
    )

    controls_col, content_col = st.columns([1, 4], gap="large")
    with controls_col:
        st.header("Settings")
        universe_name = st.selectbox("Universe", list(SECTOR_UNIVERSES.keys()))
        universe = SECTOR_UNIVERSES[universe_name]
        top_n = st.slider("Hold top N sectors", 1, min(5, len(universe["tickers"])), 3)
        use_market_filter = st.checkbox("10-month SMA cash filter", value=True)
        benchmark = st.text_input("Benchmark", value=universe["benchmark"])
        cash = st.text_input("Cash proxy", value=universe["cash"])

    with content_col:
        tickers = universe["tickers"]
        all_tickers = tuple(dict.fromkeys(tickers + [benchmark, cash]))
        prices = load_prices(all_tickers)
        available = set(prices["ticker"].unique()) if not prices.empty else set()
        missing = [ticker for ticker in all_tickers if ticker and ticker not in available]
        if missing:
            st.warning("Missing price history for: " + ", ".join(missing))

        available_sectors = [ticker for ticker in tickers if ticker in available]
        if len(available_sectors) < 3 or benchmark not in available:
            st.error("Not enough sector/benchmark data to run the strategy.")
            st.stop()

        ranks = current_sector_ranks(prices, available_sectors)
        meta = prices.sort_values("date").drop_duplicates("ticker", keep="last")
        ranks = ranks.merge(meta[["ticker", "description", "fund_type"]], on="ticker", how="left")
        ranks.insert(0, "Rank", range(1, len(ranks) + 1))
        add_sparkline_column(ranks)
        add_sparkline_column(ranks, col_name="Price (1y)", days=365)

        st.header("Current Sector Ranks")
        st.dataframe(
            ranks[["Rank", "ticker", "description", "Price (90d)", "Price (1y)", "rs_score"]]
            .style.format({"rs_score": "{:+.2f}"}),
            hide_index=True,
            column_config={
                "Price (90d)": st.column_config.LineChartColumn("Price (90d)", width="small"),
                "Price (1y)": st.column_config.LineChartColumn("Price (1y)", width="small"),
                "rs_score": st.column_config.NumberColumn("RS score", format="%+.2f"),
            },
            height=430,
        )

        backtest, holdings = build_sector_rotation_backtest(
            prices=prices,
            sector_tickers=available_sectors,
            benchmark_ticker=benchmark,
            cash_ticker=cash if cash in available else None,
            top_n=top_n,
            use_market_filter=use_market_filter,
        )
        if backtest.empty:
            st.error("Backtest has no live period after lookback warm-up.")
            st.stop()

        stats = performance_stats(backtest)
        st.header("Backtest")
        st.dataframe(
            stats.style.format(
                {"CAGR": "{:+.2f}%", "Volatility": "{:.2f}%", "Sharpe": "{:.2f}", "Max Drawdown": "{:+.2f}%"}
            ),
            hide_index=True,
        )

        chart_df = backtest.melt(
            id_vars="date", value_vars=["Strategy", "Benchmark"], var_name="Series", value_name="Index"
        )
        fig = px.line(chart_df, x="date", y="Index", color="Series", title="Growth of 100")
        fig.update_layout(height=520, yaxis_type="log")
        st.plotly_chart(fig, width="stretch")

        st.header("Recent Rebalances")
        recent = holdings.tail(12).sort_values("signal_date", ascending=False).copy()
        st.dataframe(
            recent.style.format({"rs_score": "{:+.2f}"}),
            hide_index=True,
            height=450,
        )

        st.caption(
            "Implementation note: prices use the app's total-return price series where available. "
            "Signals are formed on month-end closes and applied from the next trading day. "
            "Taxes, commissions, slippage, and FX conversion effects are not modelled."
        )
