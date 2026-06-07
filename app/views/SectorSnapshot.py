"""Daily relative-strength snapshot view and PNG exporter.

The snapshot ranks instruments by their current relative performance versus a
benchmark and renders the leaders as a small-multiple chart grid.  It is used by
both Streamlit and the scheduled email exporter.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


DEFAULT_BENCHMARK = "CSP1.L"
DEFAULT_DAYS = 252
DEFAULT_TOP_N = 25
DEFAULT_FUND_TYPE_PREFIXES = ("eq", "commod")
EXCLUDED_FUND_TYPES = ("bonds", "volatility")
INSTRUMENT_INFO_FILE = (
    Path(__file__).resolve().parents[2] / "resources" / "instrument_info.csv"
)
CORE_SECTOR_NAMES = (
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Healthcare",
    "Health Care",
    "Industrials",
    "Materials",
    "Real Estate",
    "Technology",
    "Utilities",
)

SNAPSHOT_GROUPS = {
    "thematic": {
        "label": "Thematic ETFs",
        "title": "S&P 500 vs Thematic ETFs",
        "description": "AI, space, semiconductors, clean energy, digital, water and other themes.",
    },
    "international": {
        "label": "International Markets",
        "title": "S&P 500 vs International Equity ETFs",
        "description": "Regional and country equity ETFs: China, Japan, Europe, UK, EM and others.",
    },
    "core_sectors": {
        "label": "Core Sectors",
        "title": "S&P 500 vs Core Sector ETFs",
        "description": "Basic market sectors plus key sector-industry ETFs not already covered in the thematic snapshot.",
    },
}

PREFERRED_GROUP_TICKERS = {
    "thematic": {
        "AI": ["AIQ", "WTAI", "AGIX"],
        "Robotics": ["BOTZ"],
        "Semiconductors": ["SOXX", "SMH.L"],
        "Cybersecurity": ["CIBR"],
        "Cloud": ["SKYY"],
        "FinTech": ["FINX"],
        "Aerospace/Defense": ["ITA"],
        "Infrastructure": ["PAVE"],
        "Agribusiness": ["MOO"],
        "Copper Miners": ["COPX"],
        "Rare Earth Metals": ["REMX"],
        "Timber/Forestry": ["WOOD"],
        "Homebuilders": ["XHB"],
        "Transportation": ["IYT"],
        "Insurance": ["KIE"],
        "Regional Banks": ["KRE"],
        "Software": ["IGV"],
        "Clean Energy": ["ICLN", "QCLN", "INRG.L"],
        "Solar": ["TAN"],
        "Wind": ["FAN"],
        "Lithium/Battery": ["LIT", "ECAR.L"],
        "Uranium/Nuclear": ["URA", "NLR", "URNM", "URNG.L"],
        "Gaming": ["ESPO", "PLAY.L"],
        "Biotech": ["IBB", "XBI"],
        "Medical Devices": ["IHI"],
        "Genomics": ["GNOM"],
        "Space": ["UFO", "ARKX", "JEDI.L"],
        "Water": ["PHO", "IH2O.L"],
    },
    "international": {
        "Developed ex-US": ["VEA", "EFV", "EFG"],
        "Total international": ["IXUS"],
        "Europe": ["VGK", "IEV", "IMEA.L"],
        "Eurozone large cap": ["FEZ"],
        "Germany": ["EWG"],
        "France": ["EWQ"],
        "UK": ["EWU", "VUKG.L"],
        "Canada": ["EWC", "CSCA.L"],
        "Switzerland": ["EWL"],
        "Spain": ["EWP"],
        "Italy": ["EWI"],
        "Netherlands": ["EWN"],
        "Sweden": ["EWD"],
        "Japan": ["EWJ", "IJPA.L"],
        "Japan small cap": ["SCJ"],
        "Asia Pacific ex-Japan": ["VPL", "VDPG.L"],
        "Australia": ["EWA"],
        "Singapore": ["EWS"],
        "Thailand": ["THD"],
        "Indonesia": ["EIDO"],
        "Vietnam": ["VNM"],
        "Saudi Arabia": ["KSA"],
        "South Africa": ["EZA"],
        "South Korea": ["EWY"],
        "Taiwan": ["EWT"],
        "India": ["INDA"],
        "China": ["FXI"],
        "China internet": ["KWEB"],
        "Emerging Markets": ["EEM"],
        "EM ex-China": ["EMXC", "EXCS.L"],
        "EM small cap": ["EEMS"],
        "Brazil": ["EWZ"],
        "Mexico": ["EWW"],
        "Eastern Europe": ["CEC.PA"],
        "Turkey": ["TUR", "ITKY.L"],
    },
    "core_sectors": {
        "Communication Services": ["XLC", "VOX", "WTEL.L"],
        "Consumer Discretionary": ["XLY", "IUCD.L", "WCOD.L"],
        "Consumer Staples": ["XLP", "IUCS.L", "WCOS.L"],
        "Energy": ["XLE", "IUES.L", "WNRG.L"],
        "Financials": ["XLF", "IUFS.L", "WFIN.L"],
        "Health Care": ["XLV", "IHCU.L", "WHEA.L"],
        "Industrials": ["XLI", "IUIS.L", "WNDU.L"],
        "Materials": ["XLB", "IUMS.L", "WMAT.L"],
        "Real Estate": ["VNQ", "USRT", "XRES.L"],
        "Technology": ["XLK", "VGT", "IUIT.L", "WTEC.L"],
        "Utilities": ["XLU", "IUSU.L", "WUTI.L"],
        "Metals & Mining": ["XME"],
        "Gold Miners": ["GDX"],
        "Oil & Gas E&P": ["XOP"],
    },
}


def _empty_snapshot() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker",
            "description",
            "fund_type",
            "relative_strength",
            "trend_delta",
            "history",
            "dates",
        ]
    )


def _canonicalise_tickers(
    df: pd.DataFrame, tickers: list[str], benchmark: str
) -> tuple[pd.DataFrame, list[str], str]:
    """Use full Yahoo tickers when available, while accepting base tickers."""
    if "ticker_full" not in df.columns:
        return df, tickers, benchmark
    out = df.copy()
    lookup = (
        out.dropna(subset=["ticker"])
        .drop_duplicates("ticker", keep="last")
        .set_index("ticker")["ticker_full"]
        .dropna()
        .astype(str)
        .to_dict()
    )
    out["ticker"] = out["ticker_full"].fillna(out["ticker"])
    tickers = [lookup.get(ticker, ticker) for ticker in tickers]
    benchmark = lookup.get(benchmark, benchmark)
    return out, tickers, benchmark


def enrich_snapshot_categories(prices: pd.DataFrame) -> pd.DataFrame:
    """Add detailed CSV category labels when DuckDB only has broad fund types."""
    if prices.empty or "snapshot_category" in prices.columns:
        return prices
    out = prices.copy()
    if "sector" in out.columns:
        out["snapshot_category"] = out["sector"]
        return out
    if not INSTRUMENT_INFO_FILE.exists():
        out["snapshot_category"] = (
            out["fund_type"] if "fund_type" in out.columns else ""
        )
        return out

    info = pd.read_csv(INSTRUMENT_INFO_FILE, usecols=["ticker", "sector"])
    out = out.merge(
        info.rename(columns={"sector": "snapshot_category"}), on="ticker", how="left"
    )
    if "ticker_full" in out.columns:
        missing = out["snapshot_category"].isna()
        if missing.any():
            full_info = info.rename(
                columns={"ticker": "ticker_full", "sector": "snapshot_category_full"}
            )
            out = out.merge(full_info, on="ticker_full", how="left")
            out["snapshot_category"] = out["snapshot_category"].fillna(
                out["snapshot_category_full"]
            )
            out = out.drop(columns=["snapshot_category_full"])
    fallback = out["fund_type"] if "fund_type" in out.columns else ""
    out["snapshot_category"] = out["snapshot_category"].fillna(fallback)
    return out


def infer_arrow_points(series) -> list[int]:
    """Return per-step direction markers for a series."""
    values = list(series)
    if len(values) < 2:
        return [0] * len(values)
    points = [0]
    for prev, cur in zip(values, values[1:], strict=False):
        points.append(1 if cur >= prev else -1)
    return points


def infer_arrow_annotation(
    series, lookback: int = 42
) -> tuple[float, float, float, float, str]:
    """Return arrow endpoints for the recent trend in axes-like data units."""
    values = pd.Series(list(series), dtype="float64").dropna()
    if len(values) < 2:
        return (0.0, 0.0, 0.0, 0.0, "#6b7280")
    start_idx = max(0, len(values) - lookback)
    x0 = float(start_idx)
    x1 = float(len(values) - 1)
    y0 = float(values.iloc[start_idx])
    y1 = float(values.iloc[-1])
    colour = "#22c55e" if y1 >= y0 else "#ef4444"
    return (x0, y0, x1, y1, colour)


def relative_strength_snapshot(
    prices: pd.DataFrame,
    tickers: list[str],
    benchmark: str,
    days: int = DEFAULT_DAYS,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    """Build ranked relative-strength rows.

    Relative strength is measured as percentage total return of the instrument
    less percentage total return of the benchmark, rebased to 0 at the first
    available point in the selected window.
    """
    df = prices.copy()
    if df.empty or "ticker" not in df.columns or "price" not in df.columns:
        return _empty_snapshot()

    df, tickers, benchmark = _canonicalise_tickers(df, tickers, benchmark)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["ticker"].isin([*tickers, benchmark])]
    pivot = (
        df.pivot_table(index="date", columns="ticker", values="price", aggfunc="last")
        .sort_index()
        .ffill()
    )
    if benchmark not in pivot.columns:
        return _empty_snapshot()

    window = pivot.tail(days)
    if window.empty:
        return _empty_snapshot()

    bench = window[benchmark].dropna()
    if bench.empty:
        return _empty_snapshot()
    window = window.loc[bench.index]
    bench_return = bench / bench.iloc[0] - 1
    meta = (
        df.sort_values("date")
        .drop_duplicates("ticker", keep="last")
        .set_index("ticker")
    )
    rows = []
    for ticker in tickers:
        if ticker not in window.columns:
            continue
        series = window[ticker].dropna()
        common_index = series.index.intersection(bench_return.index)
        if len(common_index) < max(20, min(days, 60)):
            continue
        series = series.loc[common_index]
        sector_return = series / series.iloc[0] - 1
        rel = (sector_return - bench_return.loc[common_index]) * 100
        if rel.empty:
            continue
        trend_window = rel.tail(min(63, len(rel)))
        description = ticker
        fund_type = ""
        if ticker in meta.index:
            description = (
                meta.at[ticker, "description"] if "description" in meta else ticker
            )
            fund_type = meta.at[ticker, "fund_type"] if "fund_type" in meta else ""
        rows.append(
            {
                "ticker": ticker,
                "description": description if pd.notna(description) else ticker,
                "fund_type": fund_type if pd.notna(fund_type) else "",
                "relative_strength": float(rel.iloc[-1]),
                "trend_delta": float(trend_window.iloc[-1] - trend_window.iloc[0]),
                "history": rel.tolist(),
                "dates": [d.to_pydatetime() for d in rel.index],
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (
        out.sort_values("relative_strength", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def select_snapshot_universe(
    prices: pd.DataFrame,
    benchmark: str = DEFAULT_BENCHMARK,
    fund_type_prefixes: tuple[str, ...] = DEFAULT_FUND_TYPE_PREFIXES,
) -> list[str]:
    """Select eligible tickers from loaded price history."""
    if prices.empty or "ticker" not in prices.columns:
        return []
    df = prices.copy()
    df, _, benchmark = _canonicalise_tickers(df, [], benchmark)
    if "fund_type" in df.columns and fund_type_prefixes:
        fund_type = df["fund_type"].fillna("").astype(str)
        include = fund_type.str.startswith(fund_type_prefixes)
        exclude = fund_type.str.startswith(EXCLUDED_FUND_TYPES)
        df = df[include & ~exclude]
    tickers = sorted(
        t for t in df["ticker"].dropna().astype(str).unique() if t != benchmark
    )
    return tickers


def _preferred_tickers_for_group(
    latest: pd.DataFrame, benchmark: str, group: str
) -> list[str]:
    """Return one preferred available ticker per exposure for a group."""

    def is_us_listed(ticker: str) -> bool:
        return "." not in ticker

    available = set(latest["ticker"].dropna().astype(str))
    if "ticker_full" in latest.columns:
        available.update(latest["ticker_full"].dropna().astype(str))
    canonical = {}
    if "ticker_full" in latest.columns:
        canonical = (
            latest.dropna(subset=["ticker"])
            .drop_duplicates("ticker", keep="last")
            .set_index("ticker")["ticker_full"]
            .dropna()
            .astype(str)
            .to_dict()
        )
    selected: list[str] = []
    seen: set[str] = set()
    for candidates in PREFERRED_GROUP_TICKERS[group].values():
        for candidate in candidates:
            resolved = canonical.get(candidate, candidate)
            if not is_us_listed(resolved):
                continue
            if (
                candidate in available or resolved in available
            ) and resolved != benchmark:
                if resolved not in seen:
                    selected.append(resolved)
                    seen.add(resolved)
                break
    return selected


def select_grouped_snapshot_universes(
    prices: pd.DataFrame,
    benchmark: str = DEFAULT_BENCHMARK,
) -> dict[str, list[str]]:
    """Select tickers for the three dashboard/email snapshot groups."""
    if prices.empty or "ticker" not in prices.columns:
        return {group: [] for group in SNAPSHOT_GROUPS}

    df = enrich_snapshot_categories(prices)
    df, _, benchmark = _canonicalise_tickers(df, [], benchmark)
    latest = df.sort_values("date").drop_duplicates("ticker", keep="last").copy()
    latest["snapshot_category"] = latest["snapshot_category"].fillna("").astype(str)
    latest["fund_type"] = (
        latest["fund_type"].fillna("").astype(str)
        if "fund_type" in latest.columns
        else ""
    )

    return {
        group: _preferred_tickers_for_group(latest, benchmark, group)
        for group in SNAPSHOT_GROUPS
    }


def build_grouped_snapshots(
    prices: pd.DataFrame,
    benchmark: str = DEFAULT_BENCHMARK,
    days: int = DEFAULT_DAYS,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, pd.DataFrame]:
    universes = select_grouped_snapshot_universes(prices, benchmark)
    return {
        group: relative_strength_snapshot(
            prices, tickers, benchmark, days=days, top_n=top_n
        )
        for group, tickers in universes.items()
    }


def render_snapshot_figure(
    snapshot: pd.DataFrame,
    benchmark: str = DEFAULT_BENCHMARK,
    title: str = "Daily Relative Strength Snapshot",
    subtitle: str | None = None,
    cols: int = 5,
):
    """Render a small-multiple Matplotlib figure for Streamlit and email."""
    if snapshot.empty:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "No snapshot data available", ha="center", va="center")
        ax.axis("off")
        return fig

    rows = ceil(len(snapshot) / cols)
    fig_height = 1.95 * rows + 0.9
    fig, axes = plt.subplots(rows, cols, figsize=(16, fig_height), squeeze=False)
    fig.patch.set_facecolor("white")

    header_ax = fig.add_axes((0.0, 0.94, 1.0, 0.06))
    header_ax.set_facecolor("#173b5c")
    header_ax.set_xticks([])
    header_ax.set_yticks([])
    for spine in header_ax.spines.values():
        spine.set_visible(False)
    header_ax.text(
        0.02,
        0.52,
        title.upper(),
        color="white",
        fontsize=18,
        fontweight="bold",
        va="center",
        ha="left",
    )
    header_ax.text(
        0.98,
        0.52,
        (subtitle or f"RELATIVE STRENGTH VS. {benchmark}").upper(),
        color="white",
        fontsize=11,
        fontweight="bold",
        va="center",
        ha="right",
    )

    for ax, (_, row) in zip(axes.ravel(), snapshot.iterrows(), strict=False):
        dates = row.get("dates") or list(range(len(row["history"])))
        history = pd.Series(row["history"], dtype="float64")
        ax.plot(dates, history, color="#173b5c", linewidth=1.15)
        ax.axhline(0, color="#d1d5db", linewidth=0.8)
        x0, y0, x1, y1, colour = infer_arrow_annotation(history, lookback=42)
        arrow_dates = dates if len(dates) == len(history) else list(range(len(history)))
        ax.annotate(
            "",
            xy=(arrow_dates[int(x1)], y1),
            xytext=(arrow_dates[int(x0)], y0),
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 2.4,
                "color": colour,
                "mutation_scale": 18,
            },
        )
        label = str(row["description"])
        if len(label) > 34:
            label = label[:31] + "..."
        ax.set_title(label, fontsize=8, fontweight="bold", pad=2)
        ax.tick_params(axis="both", labelsize=7, length=2)
        ax.grid(True, axis="y", alpha=0.18)
        if dates and not isinstance(dates[0], (int, np.integer)):
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_alpha(0.2)
        ax.spines["bottom"].set_alpha(0.2)

    for ax in axes.ravel()[len(snapshot) :]:
        ax.axis("off")

    fig.subplots_adjust(
        top=0.88, left=0.04, right=0.985, bottom=0.06, hspace=0.62, wspace=0.28
    )
    fig.text(
        0.015,
        0.012,
        "FinTracker — for personal use only. Relative strength is instrument total return minus benchmark total return over the selected window.",
        fontsize=7,
        color="#6b7280",
    )
    return fig


def save_snapshot_png(
    prices: pd.DataFrame,
    output_path: str | Path,
    benchmark: str = DEFAULT_BENCHMARK,
    days: int = DEFAULT_DAYS,
    top_n: int = DEFAULT_TOP_N,
    tickers: list[str] | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Create the snapshot PNG and return ``(path, ranked_rows)``."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if tickers is None:
        tickers = select_snapshot_universe(prices, benchmark)
    snapshot = relative_strength_snapshot(
        prices, tickers, benchmark, days=days, top_n=top_n
    )
    latest_date = ""
    if not prices.empty and "date" in prices.columns:
        latest_date = pd.to_datetime(prices["date"]).max().strftime("%Y-%m-%d")
    fig = render_snapshot_figure(
        snapshot,
        benchmark=benchmark,
        subtitle=f"Top {top_n} relative strength vs. {benchmark} · {latest_date}".strip(),
    )
    fig.savefig(output, dpi=150, bbox_inches="tight", pil_kwargs={"compress_level": 6})
    plt.close(fig)
    return output, snapshot


def save_grouped_snapshot_pngs(
    prices: pd.DataFrame,
    output_dir: str | Path,
    benchmark: str = DEFAULT_BENCHMARK,
    days: int = DEFAULT_DAYS,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, tuple[Path, pd.DataFrame]]:
    """Create one PNG per snapshot group."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshots = build_grouped_snapshots(
        prices, benchmark=benchmark, days=days, top_n=top_n
    )
    latest_date = ""
    if not prices.empty and "date" in prices.columns:
        latest_date = pd.to_datetime(prices["date"]).max().strftime("%Y-%m-%d")

    results: dict[str, tuple[Path, pd.DataFrame]] = {}
    for group, snapshot in snapshots.items():
        meta = SNAPSHOT_GROUPS[group]
        fig = render_snapshot_figure(
            snapshot,
            benchmark=benchmark,
            title=str(meta["title"]),
            subtitle=f"Top {top_n} relative strength vs. {benchmark} · {latest_date}".strip(),
        )
        path = output / f"{group}_relative_strength_snapshot.png"
        fig.savefig(
            path, dpi=150, bbox_inches="tight", pil_kwargs={"compress_level": 6}
        )
        plt.close(fig)
        results[group] = (path, snapshot)
    return results


def render() -> None:
    from data import load_prices

    st.title("Daily Relative Strength Snapshot")
    st.caption(
        "Three relative-strength snapshots versus the selected S&P 500 benchmark: "
        "thematic ETFs, international equity ETFs, and core sector ETFs."
    )

    controls = st.columns([1, 1, 1])
    with controls[0]:
        benchmark = st.text_input("Benchmark", value=DEFAULT_BENCHMARK)
    with controls[1]:
        top_n = st.slider("Top N", 5, 30, DEFAULT_TOP_N)
    with controls[2]:
        days = st.slider("Trading days", 63, 504, DEFAULT_DAYS, step=21)

    prices = load_prices()
    snapshots = build_grouped_snapshots(
        prices, benchmark=benchmark, days=days, top_n=top_n
    )
    tabs = st.tabs([str(SNAPSHOT_GROUPS[group]["label"]) for group in SNAPSHOT_GROUPS])
    for tab, group in zip(tabs, SNAPSHOT_GROUPS, strict=False):
        with tab:
            meta = SNAPSHOT_GROUPS[group]
            snapshot = snapshots[group]
            st.subheader(str(meta["title"]))
            st.caption(str(meta["description"]))
            if snapshot.empty:
                st.warning("No eligible instruments found for this group.")
                continue
            fig = render_snapshot_figure(
                snapshot, benchmark=benchmark, title=str(meta["title"])
            )
            st.pyplot(fig, clear_figure=True, use_container_width=True)
            table = snapshot[
                [
                    "ticker",
                    "description",
                    "fund_type",
                    "relative_strength",
                    "trend_delta",
                ]
            ].rename(
                columns={
                    "relative_strength": "RS % vs benchmark",
                    "trend_delta": "Recent RS change",
                }
            )
            st.dataframe(
                table.style.format(
                    {"RS % vs benchmark": "{:+.2f}", "Recent RS change": "{:+.2f}"}
                ),
                hide_index=True,
                height=420,
            )
