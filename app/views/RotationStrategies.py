"""
Rotation Strategies Master Page.
Consolidates all rotation strategies into a single interface.

Strategies:
1.  **Momentum**: Buy strongest trend (Relative Strength).
2.  **Mean Reversion**: Buy weakest trend (Laggards).
3.  **Low Volatility**: Buy lowest volatility (Safety).
4.  **Vol-Adjusted**: Buy best Sharpe Ratio (Risk-Adjusted).
5.  **Puke (Contrarian)**: Buy lowest Sharpe Ratio (Deep Value / Distress).
"""

import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from itertools import combinations

import duckdb_importer as di
from data import get_conn, add_sparkline_column


def render():
    st.title("🔄 Rotation Strategies")

    # Streamlit sidebars are global, so keep tab-specific controls in-page.
    settings_col, content_col = st.columns([1, 4], gap="large")

    with settings_col:
        st.subheader("Strategy Settings")

        # ── Strategy Selection ──
        STRAT_MOMENTUM = "Momentum (Trend)"
        STRAT_MEAN_REV = "Mean Reversion (Laggards)"
        STRAT_LOW_VOL = "Low Volatility (Safety)"
        STRAT_VOL_ADJ = "Vol-Adjusted (Sharpe)"
        STRAT_PUKE = "Puke (Contrarian)"

        STRAT_MAP = {
            "Momentum": STRAT_MOMENTUM,
            "MeanReversion": STRAT_MEAN_REV,
            "LowVol": STRAT_LOW_VOL,
            "VolAdj": STRAT_VOL_ADJ,
            "Puke": STRAT_PUKE,
        }

        # Check query params
        query_strat = st.query_params.get("strategy", None)
        default_idx = 0
        if query_strat and query_strat in STRAT_MAP:
            target = STRAT_MAP[query_strat]
            # Find index in list
            options = [
                STRAT_MOMENTUM,
                STRAT_MEAN_REV,
                STRAT_LOW_VOL,
                STRAT_VOL_ADJ,
                STRAT_PUKE,
            ]
            if target in options:
                default_idx = options.index(target)

        strategy_choice = st.selectbox(
            "Select Strategy",
            [STRAT_MOMENTUM, STRAT_MEAN_REV, STRAT_LOW_VOL, STRAT_VOL_ADJ, STRAT_PUKE],
            index=default_idx,
            key="rotation_strategy_choice",
        )

        # ── Dynamic Config based on Strategy ──
        if strategy_choice == STRAT_MOMENTUM:
            st.markdown(
                "**Goal**: Buy **Winners**. Invest in assets with highest trailing returns."
            )
            default_regime = False
            sort_desc = True
            sort_col = "ret"
            val_map_col = "ret"
            val_map_label = "Return"

        elif strategy_choice == STRAT_MEAN_REV:
            st.markdown(
                "**Goal**: Buy **Losers**. Invest in assets with lowest trailing returns (overreaction)."
            )
            default_regime = False
            sort_desc = False
            sort_col = "ret"
            val_map_col = "ret"
            val_map_label = "Return"

        elif strategy_choice == STRAT_LOW_VOL:
            st.markdown(
                "**Goal**: Buy **Safety**. Invest in assets with lowest annualized volatility."
            )
            default_regime = False
            sort_desc = False  # Lowest vol is best
            sort_col = "ann_vol"
            val_map_col = "ann_vol"
            val_map_label = "Volatility"

        elif strategy_choice == STRAT_VOL_ADJ:
            st.markdown(
                "**Goal**: Buy **Quality**. Invest in assets with highest Sharpe Ratio (Return/Vol)."
            )
            default_regime = False
            sort_desc = True
            sort_col = "sharpe"
            val_map_col = "sharpe"
            val_map_label = "Sharpe"

        elif strategy_choice == STRAT_PUKE:
            st.markdown(
                "**Goal**: Buy **Pain**. Invest in lowest Sharpe Ratio (High Vol + Deep Drawdowns)."
            )
            default_regime = False  # No filter for catching knives
            sort_desc = False  # Lowest sharpe is best (negative)
            sort_col = "sharpe"
            val_map_col = "sharpe"
            val_map_label = "Sharpe"

        # ── Common Settings ──
        st.markdown("---")
        st.header("Settings")

        # Defaults from Query Params
        qp = st.query_params
        def_top_n = int(qp.get("top_n", 5 if "Momentum" in strategy_choice else 10))
        def_lookback = qp.get("lookback", "3M")
        def_ft = qp.get("fund_type", "eq").split(",")
        def_bench = qp.get("benchmark", "VWRP")
        def_regime = qp.get("regime", "false").lower() == "true"

        top_n = st.slider(
            "Hold top N instruments", 1, 30, def_top_n, key="rotation_top_n"
        )

        lookback_label = st.selectbox(
            "Ranking lookback",
            options=["1M", "3M", "6M", "12M"],
            index=["1M", "3M", "6M", "12M"].index(def_lookback)
            if def_lookback in ["1M", "3M", "6M", "12M"]
            else 1,
            key="rotation_lookback",
        )

        LOOKBACK_MAP = {
            "1M": {"col": "r_1mo", "days": 21},
            "3M": {"col": "r_3mo", "days": 63},
            "6M": {"col": "r_6mo", "days": 126},
            "12M": {"col": "r_1y", "days": 252},
        }
        lb_config = LOOKBACK_MAP[lookback_label]
        lookback_col_db = lb_config["col"]  # for DB latest view
        LOOKBACK_DAYS = lb_config["days"]

        fund_type_filter = st.multiselect(
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
            default=def_ft,
            key="rotation_fund_types",
        )

        _BENCH_OPTIONS = ["VWRP", "CSP1", "IMEA"]
        benchmark_ticker = st.selectbox(
            "Benchmark",
            options=_BENCH_OPTIONS,
            index=_BENCH_OPTIONS.index(def_bench) if def_bench in _BENCH_OPTIONS else 0,
            key="rotation_benchmark",
        )

        use_regime_filter = st.checkbox(
            "Market Regime Filter (Cash if Bench < MA200)",
            value=def_regime,
            key="rotation_regime_filter",
        )

        # Check query params for corr method
        qp_method = qp.get("method", "None")
        map_method = {"None": 0, "Greedy": 1, "Optimize": 2}
        def_idx_meth = 0
        for k, v in map_method.items():
            if k in qp_method or qp_method in k:
                def_idx_meth = v
                break

        corr_method = st.radio(
            "Correlation Method",
            ["None", "Greedy Filter", "Portfolio Optimization (Min Avg Corr)"],
            index=def_idx_meth,
            help="None: Top N by rank.\nGreedy: Skip if corr > threshold.\nOptimize: Find subset of N with lowest avg correlation from Top N+5.",
            key="rotation_corr_method",
        )

        max_corr = 1.0
        if corr_method == "Greedy Filter":
            max_corr = st.slider(
                "Max Correlation Threshold",
                0.0,
                1.0,
                float(qp.get("max_corr", 1.0)),
                help="Skip asset if correlation with any better pick > this.",
                key="rotation_max_corr",
            )
        elif corr_method.startswith("Portfolio"):
            st.caption("Selecting best subset from Top N + 5 candidates.")

        # ── Sync State to URL ──
        # Update params at end of run so URL reflects current state
        if st.session_state.get("first_run", True):
            st.session_state["first_run"] = False
        else:
            # Reverse map strategy
            strat_key = "Momentum"
            for k, v in STRAT_MAP.items():
                if v == strategy_choice:
                    strat_key = k
                    break

            st.query_params["strategy"] = strat_key
            st.query_params["top_n"] = str(top_n)
            st.query_params["lookback"] = lookback_label
            st.query_params["fund_type"] = ",".join(fund_type_filter)
            st.query_params["benchmark"] = benchmark_ticker
            st.query_params["regime"] = str(use_regime_filter).lower()
            st.query_params["method"] = (
                "Greedy"
                if "Greedy" in corr_method
                else ("Optimize" if "Optimization" in corr_method else "None")
            )
            if "Greedy" in corr_method:
                st.query_params["max_corr"] = str(max_corr)
            else:
                if "max_corr" in st.query_params:
                    del st.query_params["max_corr"]

    with content_col:
        # ── Data Loading ──
        @st.cache_data(ttl=300)
        def load_price_history_pl() -> pl.DataFrame:
            conn = get_conn()

            # Check if table exists
            try:
                conn.execute(f"SELECT 1 FROM {di.px_tbl} LIMIT 1")
            except Exception:
                # Retry initialization explicitly if table missing
                st.warning(f"Table {di.px_tbl} missing. Re-initializing connection...")
                from data import init_conn

                init_conn(di.duckdb_file)
                conn = get_conn()

            query = f"""
                SELECT ticker, date, price, description, fund_type
                FROM {di.px_tbl}
                ORDER BY date ASC
            """
            try:
                return conn.execute(query).pl()
            except Exception as e:
                st.error(f"Failed to load data: {e}")
                # Fallback to pandas if polars fails
                return pl.from_pandas(conn.execute(query).df())

        _prices = load_price_history_pl()
        if _prices.is_empty():
            st.error(f"No data available in {di.px_tbl}.")
            st.stop()

        # Filter Data (Polars)
        if fund_type_filter:
            pattern = "^(" + "|".join(fund_type_filter) + ")"
            prices_df = _prices.filter(pl.col("fund_type").str.contains(pattern))
            bench_hist = _prices.filter(pl.col("ticker") == benchmark_ticker)
            prices_df = pl.concat([prices_df, bench_hist]).unique()
        else:
            prices_df = _prices

        prices_df = prices_df.with_columns(pl.col("date").cast(pl.Date))

        # ── Ranking Logic ──
        def get_rankings(
            date,
            lookback_days,
            top_n,
            strategy_type,
            exclude_ticker=None,
            correlation_threshold=1.0,
            corr_method="None",
        ):
            lb_date = date - timedelta(days=lookback_days * 1.5)

            # Filter Window
            w = prices_df.filter((pl.col("date") >= lb_date) & (pl.col("date") <= date))
            if exclude_ticker:
                w = w.filter(pl.col("ticker") != exclude_ticker)

            w = w.sort(["ticker", "date"])

            # Calc returns and vol
            # We need aggregations
            stats = (
                w.group_by("ticker")
                .agg(
                    [
                        pl.col("description").first(),
                        pl.col("price").first().alias("start"),
                        pl.col("price").last().alias("end"),
                        pl.col("date").count().alias("days"),
                        pl.col("fund_type").first(),
                    ]
                )
                .filter(pl.col("days") >= lookback_days * 0.7)
            )

            # Calculate metrics based on strategy requirements
            # Momentum/MeanRev need Return
            # LowVol needs Vol
            # VolAdj/Puke need Sharpe (Ret/Vol)

            # Standard metrics
            stats = stats.with_columns(
                ((pl.col("end") / pl.col("start") - 1) * 100).alias("ret")
            )

            if strategy_type in [STRAT_LOW_VOL, STRAT_VOL_ADJ, STRAT_PUKE]:
                # Need Volatility calculation.
                # This requires daily returns std dev inside the group_by, which is heavier.
                # Let's do a separate calc or improve the agg.
                # Polars group_by agg of expression is fast.

                # We need daily returns first.
                w_ret = w.with_columns(
                    pl.col("price").pct_change().over("ticker").alias("ret_d")
                )

                # Re-agg with std
                stats = (
                    w_ret.group_by("ticker")
                    .agg(
                        [
                            pl.col("description").first(),
                            pl.col("price").first().alias("start"),
                            pl.col("price").last().alias("end"),
                            pl.col("ret_d").std().alias("std_dev"),
                            pl.col("date").count().alias("days"),
                            pl.col("fund_type").first(),
                        ]
                    )
                    .filter(pl.col("days") >= lookback_days * 0.7)
                )

                stats = stats.with_columns(
                    [
                        ((pl.col("end") / pl.col("start") - 1) * 100).alias("ret"),
                        (pl.col("std_dev") * np.sqrt(252) * 100).alias("ann_vol"),
                    ]
                )

                if strategy_type in [STRAT_VOL_ADJ, STRAT_PUKE]:
                    stats = stats.with_columns(
                        (pl.col("ret") / pl.col("ann_vol")).fill_nan(0).alias("sharpe")
                    )

            # Sort based on strategy
            if strategy_type == STRAT_MOMENTUM:
                stats = stats.sort("ret", descending=True)
            elif strategy_type == STRAT_MEAN_REV:
                stats = stats.sort("ret", descending=False)
            elif strategy_type == STRAT_LOW_VOL:
                stats = stats.sort("ann_vol", descending=False)
            elif strategy_type == STRAT_VOL_ADJ:
                stats = stats.sort("sharpe", descending=True)
            elif strategy_type == STRAT_PUKE:
                stats = stats.sort("sharpe", descending=False)

            # ── Correlation Filter ──
            # If using filter, take a larger pool and select uncorrelated assets
            if corr_method != "None":
                is_opt = corr_method.startswith("Portfolio")
                pool_size = top_n + 5 if is_opt else top_n * 4
                candidates_df = stats.head(pool_size)
                candidate_tickers = candidates_df["ticker"].to_list()

                if len(candidate_tickers) > top_n:
                    # 1. Get Daily Returns for candidates
                    w_pool = w.filter(pl.col("ticker").is_in(candidate_tickers))

                    # Pivot to wide: date x ticker
                    w_pivot = w_pool.pivot(
                        index="date",
                        on="ticker",
                        values="price",
                        aggregate_function="first",
                    ).sort("date")

                    # Forward fill then returns
                    w_pivot = w_pivot.with_columns(pl.all().forward_fill())
                    daily_rets = (
                        w_pivot.select(pl.all().exclude("date"))
                        .to_pandas()
                        .pct_change()
                        .iloc[1:]
                    )

                    # Correlation Matrix
                    corr_matrix = daily_rets.corr().abs()

                    selected_tickers = []

                    if is_opt:
                        # Combinatorial Optimization: Find subset of size top_n with min avg correlation
                        # Only efficient for small N. If N > 15, fallback to greedy?
                        # User asked for "Top 15 take 10". 15C10=3003. Fast.
                        # If pool=N+5, then (N+5)CN is polynomial in N? No, (N+5)!/(N!5!) approx N^5.
                        # If N=30, 35C30 = 35C5 = 324,632. fast enough (0.1s).

                        valid_tickers = [
                            t for t in candidate_tickers if t in corr_matrix.index
                        ]
                        if len(valid_tickers) <= top_n:
                            return stats.head(top_n)

                        best_score = float("inf")
                        best_subset = []

                        # Iterate combinations
                        # Limit pool just in case N is HUGE
                        search_pool = valid_tickers[
                            : min(len(valid_tickers), top_n + 5)
                        ]

                        for combo in combinations(search_pool, top_n):
                            # Calc avg correlation of this subset
                            # Access submatrix
                            sub_df = corr_matrix.loc[list(combo), list(combo)]
                            # Sum of upper triangle
                            score = (
                                sub_df.values.sum() - sub_df.shape[0]
                            ) / 2  # Sum(corr) - Diagonal(N) / 2
                            if score < best_score:
                                best_score = score
                                best_subset = list(combo)

                        selected_tickers = best_subset

                    else:  # Greedy
                        # Helper to check correlation
                        def is_uncorrelated(ticker, current_list):
                            if not current_list:
                                return True
                            for existing in current_list:
                                c = (
                                    corr_matrix.loc[ticker, existing]
                                    if ticker in corr_matrix.index
                                    and existing in corr_matrix.columns
                                    else 0
                                )
                                if c > correlation_threshold:
                                    return False
                            return True

                        for row in candidates_df.iter_rows(named=True):
                            t = row["ticker"]
                            if is_uncorrelated(t, selected_tickers):
                                selected_tickers.append(t)
                            if len(selected_tickers) >= top_n:
                                break

                    # Filter stats to selected
                    if selected_tickers:
                        return stats.filter(
                            pl.col("ticker").is_in(selected_tickers)
                        ).head(top_n)

            return stats.head(top_n)

        # ── Current Picks ──
        st.header(f"📋 Current Picks ({lookback_label})")

        latest_date = prices_df.select(pl.col("date").max()).item()
        current_picks = get_rankings(
            latest_date,
            LOOKBACK_DAYS,
            top_n,
            strategy_choice,
            exclude_ticker=benchmark_ticker,
            correlation_threshold=max_corr,
            corr_method=corr_method,
        )

        top_picks = current_picks.to_pandas()
        top_picks.index = top_picks.index + 1
        top_picks.index.name = "Rank"

        # Display columns
        cols = ["ticker", "description", "fund_type", "Price (90d)", "Price (1y)", "ret"]
        if "ann_vol" in top_picks.columns:
            cols.append("ann_vol")
        if "sharpe" in top_picks.columns:
            cols.append("sharpe")

        add_sparkline_column(top_picks)
        add_sparkline_column(top_picks, col_name="Price (1y)", days=365)

        fmt = {"ret": "{:+.2f}%", "ann_vol": "{:.2f}%", "sharpe": "{:.2f}"}

        st.dataframe(
            top_picks[cols].style.format(fmt),
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
                "ret": st.column_config.NumberColumn(
                    "ret", format="%.2f%%", width="small"
                ),
                "ann_vol": st.column_config.NumberColumn(
                    "ann_vol", format="%.2f%%", width="small"
                ),
                "sharpe": st.column_config.NumberColumn(
                    "sharpe", format="%.2f", width="small"
                ),
            },
            height=min(400, 50 + top_n * 35),
        )
        st.caption("Strategy assumes **Equal Weight** allocation.")

        with st.expander("🔍 Correlation Matrix (Current Picks)"):
            if len(top_picks) > 1:
                c_tickers = top_picks["ticker"].to_list()
                # Fetch history for correlation
                lb_date_corr = latest_date - timedelta(days=LOOKBACK_DAYS)
                w_corr = prices_df.filter(
                    (pl.col("ticker").is_in(c_tickers))
                    & (pl.col("date") >= lb_date_corr)
                )

                # Pivot and Correlation
                w_p = w_corr.pivot(
                    index="date",
                    on="ticker",
                    values="price",
                    aggregate_function="first",
                ).sort("date")
                w_p = w_p.with_columns(pl.all().forward_fill())
                d_rets = (
                    w_p.select(pl.all().exclude("date"))
                    .to_pandas()
                    .pct_change()
                    .iloc[1:]
                )

                if not d_rets.empty:
                    corr_m = d_rets.corr()

                    # Map Ticker to Description for Viz
                    desc_map = dict(zip(top_picks["ticker"], top_picks["description"]))
                    labels = [f"{t} ({desc_map.get(t, '')})" for t in corr_m.columns]

                    fig_corr = px.imshow(
                        corr_m,
                        x=labels,
                        y=labels,
                        text_auto=".2f",
                        color_continuous_scale="RdBu_r",
                        zmin=-1,
                        zmax=1,
                        title=f"Correlation Matrix ({lookback_label} Lookback)",
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.warning("Insufficient data for correlation.")
            else:
                st.info("Need at least 2 assets for correlation.")

        fig_picks = px.bar(
            top_picks,
            x=sort_col,
            y="description",
            orientation="h",
            color=sort_col,
            color_continuous_scale="RdYlGn" if sort_desc else "RdYlGn_r",
            labels={sort_col: val_map_label, "description": ""},
            title=f"Top {top_n} by {val_map_label}",
        )
        fig_picks.update_layout(
            yaxis=dict(autorange="reversed"), height=max(300, top_n * 40)
        )
        st.plotly_chart(fig_picks, use_container_width=True)

        # ── Backtest Engine ──
        st.header("📅 Backtest Performance")
        st.markdown(f"Running simulation for **{strategy_choice}** over last 3 years.")

        max_date = latest_date
        min_date = max_date - timedelta(days=365 * 3)

        dates_agg = (
            prices_df.filter(pl.col("date") >= min_date)
            .with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
            .group_by("month")
            .agg(pl.col("date").max().alias("rebal_date"))
            .sort("rebal_date")
        )
        rebal_dates = dates_agg["rebal_date"].to_list()

        portfolio = {}
        closed_trades = []
        curve_dates = []
        strat_vals = []
        bench_vals = []
        curr_strat_val = 100.0
        curr_bench_val = 100.0
        current_holdings = []

        # Benchmark 200d MA Map
        bench_data = _prices.filter(pl.col("ticker") == benchmark_ticker).sort("date")
        bench_data = bench_data.with_columns(
            pl.col("price").rolling_mean(window_size=200).alias("ma200")
        )
        bench_map = dict(
            zip(bench_data["date"], zip(bench_data["price"], bench_data["ma200"]))
        )

        # Store Holdings Logic
        holdings_history = []  # List of dicts

        for i in range(len(rebal_dates) - 1):
            start_date = rebal_dates[i]
            end_date = rebal_dates[i + 1]

            # Regime Check
            is_bull = True
            if use_regime_filter:
                b_info = bench_map.get(start_date)
                if b_info and b_info[0] is not None and b_info[1] is not None:
                    if b_info[0] < b_info[1]:
                        is_bull = False

            # Select Targets
            if is_bull:
                top = get_rankings(
                    start_date,
                    LOOKBACK_DAYS,
                    top_n,
                    strategy_choice,
                    exclude_ticker=benchmark_ticker,
                    correlation_threshold=max_corr,
                    corr_method=corr_method,
                )
                target_indices = top["ticker"].to_list()
                target_info = top.select(["ticker", "description", "end"]).to_dicts()
                # Sort targets by rank (based on strategy sort)
                # The get_rankings returns sorted df.

                target_map = {x["ticker"]: x for x in target_info}

                # Record Holdings for Report
                rank_i = 1
                for t_row in target_info:
                    holdings_history.append(
                        {
                            "Rebalance Date": start_date,
                            "Rank": rank_i,
                            "Ticker": t_row["ticker"],
                            "Name": t_row["description"],
                            "Allocation": f"{100 / len(target_indices):.1f}%"
                            if target_indices
                            else "0%",
                        }
                    )
                    rank_i += 1

            else:
                target_indices = []  # Cash
                target_map = {}
                holdings_history.append(
                    {
                        "Rebalance Date": start_date,
                        "Rank": "-",
                        "Ticker": "CASH",
                        "Name": "Market Regime Risk-Off (Cash)",
                        "Allocation": "100%",
                    }
                )

            # Execute Trades
            curr_h_prices = (
                prices_df.filter(
                    (pl.col("date") == start_date)
                    & (pl.col("ticker").is_in(current_holdings))
                )
                .select(["ticker", "price"])
                .to_dicts()
            )
            h_px_map = {x["ticker"]: x["price"] for x in curr_h_prices}

            # Sells
            for ticker in list(current_holdings):
                if ticker not in target_indices:
                    exit_px = h_px_map.get(ticker)
                    if ticker in portfolio and exit_px:
                        ent = portfolio.pop(ticker)
                        ret = (exit_px / ent["entry_price"] - 1) * 100
                        closed_trades.append(
                            {
                                "Entry Date": ent["entry_date"],
                                "Exit Date": start_date,
                                "Ticker": ticker,
                                "Name": ent["name"],
                                "Entry Price": ent["entry_price"],
                                "Exit Price": exit_px,
                                "Return": ret,
                                "Status": "Closed",
                            }
                        )
                    if ticker in current_holdings:
                        current_holdings.remove(ticker)

            # Buys
            for ticker in target_indices:
                if ticker not in current_holdings:
                    info = target_map.get(ticker)
                    if info:
                        portfolio[ticker] = {
                            "entry_date": start_date,
                            "entry_price": info["end"],
                            "name": info["description"],
                        }
                        current_holdings.append(ticker)

            # Daily Simulation
            interval_data = prices_df.filter(
                (pl.col("date") > start_date)
                & (pl.col("date") <= end_date)
                & (pl.col("ticker").is_in(current_holdings + [benchmark_ticker]))
            ).sort("date")

            i_dates = interval_data["date"].unique().sort()

            prev_prices = {}

            # Init prev_prices
            start_px_rows = prices_df.filter(
                (pl.col("date") == start_date)
                & (pl.col("ticker").is_in(current_holdings + [benchmark_ticker]))
            ).to_dicts()
            for r in start_px_rows:
                prev_prices[r["ticker"]] = r["price"]

            for d in i_dates:
                d_rows = interval_data.filter(pl.col("date") == d).to_dicts()
                d_px = {r["ticker"]: r["price"] for r in d_rows}

                day_rets = []
                for t in current_holdings:
                    curr = d_px.get(t)
                    prev = prev_prices.get(t)
                    if curr and prev:
                        day_rets.append(curr / prev - 1)

                avg_ret = np.mean(day_rets) if day_rets else 0.0
                curr_strat_val *= 1 + avg_ret

                b_curr = d_px.get(benchmark_ticker)
                b_prev = prev_prices.get(benchmark_ticker)
                if b_curr and b_prev:
                    b_ret = b_curr / b_prev - 1
                    curr_bench_val *= 1 + b_ret

                prev_prices.update(d_px)

                curve_dates.append(d)
                strat_vals.append(curr_strat_val)
                bench_vals.append(curr_bench_val)

        # Handle Active
        last_sim_date = (
            curve_dates[-1]
            if curve_dates
            else (rebal_dates[-1] if rebal_dates else max_date)
        )
        final_px_rows = prices_df.filter(
            (pl.col("date") == last_sim_date)
            & (pl.col("ticker").is_in(current_holdings))
        ).to_dicts()
        final_px_map = {r["ticker"]: r["price"] for r in final_px_rows}

        for ticker, ent in portfolio.items():
            cur_px = final_px_map.get(ticker)
            if cur_px:
                ret = (cur_px / ent["entry_price"] - 1) * 100
                closed_trades.append(
                    {
                        "Entry Date": ent["entry_date"],
                        "Exit Date": None,
                        "Ticker": ticker,
                        "Name": ent["name"],
                        "Entry Price": ent["entry_price"],
                        "Exit Price": cur_px,
                        "Return": ret,
                        "Status": "Active",
                    }
                )

        # Visuals
        # 1. Equity Curve
        st.subheader("📈 Equity Curve (Daily)")
        if len(curve_dates) > 0:
            fig_curve = go.Figure()
            fig_curve.add_trace(
                go.Scatter(
                    x=curve_dates,
                    y=strat_vals,
                    name=f"{strategy_choice}",
                    line=dict(color="#2980b9", width=2),
                )
            )
            fig_curve.add_trace(
                go.Scatter(
                    x=curve_dates,
                    y=bench_vals,
                    name=f"Benchmark ({benchmark_ticker})",
                    line=dict(dash="dash", color="gray"),
                )
            )
            fig_curve.update_layout(
                hovermode="x unified", title=f"3-Year Backtest vs {benchmark_ticker}"
            )
            st.plotly_chart(fig_curve, use_container_width=True)

            tot = (strat_vals[-1] / 100 - 1) * 100
            bm = (bench_vals[-1] / 100 - 1) * 100
            st.markdown(
                f"**Total Return**: {tot:+.1f}% vs Bench: {bm:+.1f}% | **Alpha**: {tot - bm:+.1f}% | **Simulated Days**: {len(curve_dates)}"
            )
        else:
            st.info("No simulation data generated.")

        # 2. Holdings Report (Monthly Snapshot)
        with st.expander("📅 Monthly Holdings Report"):
            if holdings_history:
                df_hist = pd.DataFrame(holdings_history).sort_values(
                    ["Rebalance Date", "Rank"], ascending=[False, True]
                )
                st.dataframe(
                    df_hist.style.format({"Rebalance Date": "{:%Y-%m-%d}"}),
                    hide_index=True,
                    use_container_width=True,
                    height=400,
                )
            else:
                st.info("No holdings history to display.")

        # 3. Trade Log
        if closed_trades:
            log_df = pd.DataFrame(closed_trades)
            st.subheader("📜 Trade Log")

            active = log_df[log_df["Status"] == "Active"]
            if not active.empty:
                st.markdown("#### 🟢 Active Positions")
                st.dataframe(
                    active[
                        [
                            "Entry Date",
                            "Ticker",
                            "Name",
                            "Entry Price",
                            "Exit Price",
                            "Return",
                        ]
                    ].style.format(
                        {
                            "Entry Price": "{:.2f}",
                            "Exit Price": "{:.2f}",
                            "Return": "{:+.2f}%",
                        }
                    ),
                    use_container_width=True,
                )

            closed = log_df[log_df["Status"] == "Closed"]
            if not closed.empty:
                st.markdown("#### 🔴 Closed Positions")
                st.dataframe(
                    closed.sort_values("Exit Date", ascending=False)[
                        [
                            "Entry Date",
                            "Exit Date",
                            "Ticker",
                            "Name",
                            "Entry Price",
                            "Exit Price",
                            "Return",
                        ]
                    ].style.format(
                        {
                            "Entry Price": "{:.2f}",
                            "Exit Price": "{:.2f}",
                            "Return": "{:+.2f}%",
                        }
                    ),
                    use_container_width=True,
                    height=300,
                )
