from __future__ import annotations

import argparse
import os
import tempfile
from email.utils import make_msgid
from pathlib import Path

import duckdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.alerts.consolidated import _build_consolidated
from app.alerts.session import VALID_SESSIONS, session_label
from app.alerts.freshness import assert_fresh_data
from app.alerts.signals import build_all_signals
from app.alerts.state import (
    DEFAULT_STATE_DIR,
    detect_changes,
    load_previous,
    save_current,
)
from send_breakout_alerts import (
    _period_return,
    add_sma_overlays,
    exchange_label,
    save_compressed_png,
    send_email,
)


ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
DB_FILE = Path(__file__).resolve().parents[1] / "duckdb.db"


def load_env_file() -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _connect() -> tuple[duckdb.DuckDBPyConnection, str]:
    return duckdb.connect(str(DB_FILE), read_only=True), ""


def load_price_history() -> pd.DataFrame:
    conn, prefix = _connect()
    query = f"""
        SELECT ticker, ticker_full, date,
               open_orig AS price_open, high_orig AS price_high, low_orig AS price_low,
               price_orig AS price, description, fund_type, currency
        FROM {prefix}total_return
        ORDER BY ticker, date
    """
    try:
        df = conn.execute(query).df()
    finally:
        conn.close()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df


def load_performance(max_rown: int) -> pd.DataFrame:
    conn, prefix = _connect()
    query = f"""
        SELECT *
        FROM {prefix}latest_performance_sharpe
        WHERE rown <= {int(max_rown)}
        ORDER BY description ASC
    """
    try:
        df = conn.execute(query).df()
    finally:
        conn.close()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols):
        df[numeric_cols] = df[numeric_cols].mask(~np.isfinite(df[numeric_cols]))
    return df


def attach_ticker_full(perf: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if (
        perf.empty
        or "ticker_full" in perf.columns
        or prices.empty
        or "ticker_full" not in prices.columns
    ):
        return perf
    latest_idx = prices.groupby("ticker")["date"].idxmax()
    latest_meta = prices.loc[latest_idx]
    ticker_full = latest_meta.set_index("ticker")["ticker_full"].to_dict()
    out = perf.copy()
    out["ticker_full"] = out["ticker"].map(ticker_full).fillna(out["ticker"])
    return out


def _selected(strategies: list, selected: set[str] | None):
    if not selected:
        return strategies
    return [strategy for strategy in strategies if strategy.strategy_id in selected]


def _history_for_signal(prices: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    ticker = str(row.get("ticker", ""))
    alert_ticker = str(row.get("alert_ticker", ticker))
    ticker_full = (
        prices["ticker_full"] if "ticker_full" in prices.columns else prices["ticker"]
    )
    return prices[
        (prices["ticker"].astype(str) == ticker)
        | (ticker_full.astype(str) == alert_ticker)
    ].sort_values("date")


def _performance_text(history: pd.DataFrame) -> str:
    perf = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}
    parts = []
    for label, days in perf.items():
        value = _period_return(history, days)
        if value is not None:
            parts.append(f"{label} {value:+.1f}%")
    return ", ".join(parts) or "not enough history"


def _volatility_text(row: pd.Series) -> str:
    parts = []
    if pd.notna(row.get("vol_1y")):
        parts.append(f"Vol 1Y: {float(row['vol_1y']):.1f}%")
    if pd.notna(row.get("vol_1mo")):
        parts.append(f"Vol 1M: {float(row['vol_1mo']):.1f}%")
    return " | ".join(parts) or "not available"


def _signal_start_text(row: pd.Series) -> str:
    start_date = row.get("start_date")
    start_price = row.get("start_price")
    current_price = row.get("price")
    if not start_date or pd.isna(start_price) or pd.isna(current_price):
        return ""
    ret = (float(current_price) / float(start_price) - 1) * 100
    return f"Signal started {start_date} @ {float(start_price):.2f} ({ret:+.1f}% since)"


def _signal_start_html(row: pd.Series) -> str:
    start_date = row.get("start_date")
    start_price = row.get("start_price")
    current_price = row.get("price")
    if not start_date or pd.isna(start_price) or pd.isna(current_price):
        return ""
    ret = (float(current_price) / float(start_price) - 1) * 100
    return f"<p style='margin:2px 0;color:#059669;'><strong>Signal:</strong> {start_date} @ {float(start_price):.2f} ({ret:+.1f}%)</p>"


# ── Individual strategy email builders ──


def build_strategy_body(
    title: str, commentary: str, signals: pd.DataFrame, prices: pd.DataFrame
) -> str:
    lines = [
        title,
        "",
        f"What this targets: {commentary}",
        "",
        "Each alert includes the trigger, recent performance, volatility context, and 1Y/3Y charts.",
        "",
    ]
    for i, row in signals.reset_index(drop=True).iterrows():
        history = _history_for_signal(prices, row)
        ticker = str(row.get("alert_ticker", row.get("ticker", "")))
        score = row.get("score")
        score_text = f"Score {float(score):.1f}" if pd.notna(score) else "Signal"
        change = f" [{row.get('change')}]" if row.get("change") else ""
        lines.append(
            f"ALERT #{i + 1} ({score_text}): {ticker} - {row.get('description', '')}{change}\n"
            f"Exchange: {exchange_label(ticker)} ({ticker}).\n\n"
            f"Reason: {commentary}\n\n"
            f"Trigger: {row.get('summary', row.get('signal', 'Current strategy signal'))}.\n\n"
            f"Performance: {_performance_text(history)}.\n\n"
            f"Volatility: {_volatility_text(row)}.\n\n"
            f"Charts: 1Y signal context and 3Y context below.\n"
        )
    return "\n".join(lines)


def build_strategy_html(
    title: str,
    commentary: str,
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    content_ids: dict[str, str],
) -> str:
    blocks = [
        f"<h2>{title}</h2>",
        f"""
        <section style="margin:0 0 22px 0;padding:14px 16px;border:1px solid #e5e7eb;border-radius:12px;background:#f8fafc;font-family:Arial,sans-serif;">
          <p style="margin:0 0 8px 0;"><strong>What this targets:</strong> {commentary}</p>
          <p style="margin:0;color:#4b5563;">Each alert includes the trigger, recent performance, volatility context, and 1Y/3Y charts so you can quickly decide whether it belongs on the watchlist.</p>
        </section>
        """,
    ]
    for i, row in signals.reset_index(drop=True).iterrows():
        ticker = str(row.get("alert_ticker", row.get("ticker", "")))
        history = _history_for_signal(prices, row)
        score = row.get("score")
        score_text = f"Score {float(score):.1f}" if pd.notna(score) else "Signal"
        cid = content_ids.get(f"{ticker}_signal")
        context_cid = content_ids.get(f"{ticker}_context")
        chart_html = (
            f'<img src="cid:{cid}" style="max-width:100%; width:900px; height:auto; border:1px solid #ddd;" /><br><img src="cid:{context_cid}" style="max-width:100%; width:900px; height:auto; border:1px solid #ddd; margin-top:8px;" />'
            if cid and context_cid
            else "<p><em>No chart available.</em></p>"
        )
        blocks.append(f"""
        <section style="margin:0 0 28px 0; font-family:Arial, sans-serif;">
          <h3 style="margin-bottom:6px;">ALERT #{i + 1} ({score_text}): {ticker} - {row.get("description", "")}</h3>
          <p><strong>Exchange:</strong> {exchange_label(ticker)} ({ticker}).</p>
          <p><strong>Reason:</strong> {commentary}</p>
          <p><strong>Trigger:</strong> {row.get("summary", row.get("signal", "Current strategy signal"))}.</p>
          <p><strong>Performance:</strong> {_performance_text(history)}.</p>
          <p><strong>Volatility:</strong> {_volatility_text(row)}.</p>
          {chart_html}
        </section>
        """)
    return "\n".join(blocks)


# ── Consolidated email builders ──


def build_consolidated_body(consolidated: pd.DataFrame, prices: pd.DataFrame) -> str:
    lines = [
        "FinTracker consolidated alerts",
        "",
        "Instruments ranked by signal count across strategies.",
        "",
    ]
    for i, row in consolidated.iterrows():
        history = _history_for_signal(prices, row)
        ticker = str(row.get("alert_ticker", row.get("ticker", "")))
        score = row.get("score")
        signals = row.get("signal", [])
        signal_text = ", ".join(signals) if isinstance(signals, list) else str(signals)
        score_text = f"Score {float(score):.1f}" if pd.notna(score) else "Signal"
        change = f" [{row.get('change')}]" if row.get("change") else ""
        lines.append(
            f"#{i + 1} ({len(signals)} signals, {score_text}): {ticker} - {row.get('description', '')}{change}"
        )
        lines.append(f"Exchange: {exchange_label(ticker)} ({ticker}).")
        lines.append(f"Signals: {signal_text}")
        summaries = row.get("summary", [])
        strategy_titles = row.get("strategy_title", [])
        if (
            isinstance(strategy_titles, list)
            and isinstance(summaries, list)
            and len(strategy_titles) == len(summaries)
        ):
            for title, summary in zip(strategy_titles, summaries):
                lines.append(f"  - {title}: {summary}")
        elif isinstance(summaries, list) and summaries:
            for summary in summaries:
                lines.append(f"  - {summary}")
        lines.append(f"Perf: {_performance_text(history)}.")
        vol_text = _volatility_text(row)
        if vol_text != "not available":
            lines.append(f"Vol: {vol_text}.")
        start_text = _signal_start_text(row)
        if start_text:
            lines.append(start_text)
        lines.append("")
    return "\n".join(lines)


def build_consolidated_html(
    consolidated: pd.DataFrame,
    prices: pd.DataFrame,
    content_ids: dict[str, str],
) -> str:
    blocks = [
        "<h2>FinTracker consolidated alerts</h2>",
        "<p>Instruments ranked by signal count across strategies.</p>",
    ]
    for i, row in consolidated.iterrows():
        ticker = str(row.get("alert_ticker", row.get("ticker", "")))
        history = _history_for_signal(prices, row)
        score = row.get("score")
        signals = row.get("signal", [])
        signal_text = ", ".join(signals) if isinstance(signals, list) else str(signals)
        score_text = f"Score {float(score):.1f}" if pd.notna(score) else "Signal"
        cid = content_ids.get(f"{ticker}_signal")
        context_cid = content_ids.get(f"{ticker}_context")
        chart_html = (
            f'<img src="cid:{cid}" style="max-width:100%;width:900px;height:auto;border:1px solid #ddd;" /><br><img src="cid:{context_cid}" style="max-width:100%;width:900px;height:auto;border:1px solid #ddd;margin-top:8px;" />'
            if cid and context_cid
            else "<p><em>No chart available.</em></p>"
        )
        strategy_titles = row.get("strategy_title", [])
        summaries = row.get("summary", [])
        summary_html = ""
        if (
            isinstance(strategy_titles, list)
            and isinstance(summaries, list)
            and len(strategy_titles) == len(summaries)
        ):
            summary_items = "".join(
                f"<li>{title}: {summary}</li>"
                for title, summary in zip(strategy_titles, summaries)
            )
            summary_html = f"<ul style='margin:4px 0;'>{summary_items}</ul>"
        elif isinstance(summaries, list) and summaries:
            summary_items = "".join(f"<li>{summary}</li>" for summary in summaries)
            summary_html = f"<ul style='margin:4px 0;'>{summary_items}</ul>"
        vol_text = _volatility_text(row)
        vol_html = (
            f"<p><strong>Vol:</strong> {vol_text}.</p>"
            if vol_text != "not available"
            else ""
        )
        blocks.append(f"""
        <section style="margin:0 0 16px 0;font-family:Arial,sans-serif;font-size:13px;">
          <h3 style="margin:0 0 4px 0;font-size:15px;">#{i + 1} ({len(signals)} signals, {score_text}): {ticker} - {row.get("description", "")}</h3>
          <p style="margin:2px 0;"><strong>Exchange:</strong> {exchange_label(ticker)} ({ticker}).</p>
          <p style="margin:2px 0;"><strong>Signals:</strong> {signal_text}</p>
          {summary_html}
          <p style="margin:2px 0;"><strong>Perf:</strong> {_performance_text(history)}.</p>
          {vol_html}
          {_signal_start_html(row)}
          {chart_html}
        </section>
        """)
    return "\n".join(blocks)


# ── Chart generation ──


def chart_strategy_signals(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    output_dir: Path,
    chart_years: float,
    chart_kind: str,
) -> list[Path]:
    chart_paths = []
    for _, row in signals.reset_index(drop=True).iterrows():
        history_all = _history_for_signal(prices, row)
        if history_all.empty:
            continue
        start_date = history_all["date"].max() - pd.DateOffset(
            days=int(chart_years * 365)
        )
        history = (
            history_all[history_all["date"] >= start_date].copy().sort_values("date")
        )
        if history.empty:
            continue
        x = mdates.date2num(history["date"].dt.to_pydatetime())
        open_px = history["price_open"].fillna(history["price"])
        high_px = history["price_high"].fillna(history["price"])
        low_px = history["price_low"].fillna(history["price"])
        close_px = history["price"].astype(float)
        fig, ax = plt.subplots(figsize=(12, 6))
        width = max((x[-1] - x[0]) / max(len(x), 1) * 0.7, 0.2)
        up = close_px >= open_px
        colors = up.map({True: "#1b9e77", False: "#d95f02"})
        ax.vlines(x, low_px, high_px, color=colors, linewidth=0.6, alpha=0.9)
        for xi, open_i, close_i, color in zip(
            x, open_px, close_px, colors, strict=False
        ):
            lower = min(open_i, close_i)
            height = abs(close_i - open_i) or close_i * 0.001
            ax.add_patch(
                plt.Rectangle(
                    (xi - width / 2, lower),
                    width,
                    height,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.4,
                    alpha=0.9,
                )
            )
        add_sma_overlays(ax, history)
        return_lines = []
        for label, days in [
            ("1W", 5),
            ("1M", 21),
            ("3M", 63),
            ("6M", 126),
            ("1Y", 252),
            ("3Y", min(756, len(history) - 1)),
        ]:
            value = _period_return(history, days)
            if value is not None:
                return_lines.append(f"{label}: {value:+.1f}%")
        if return_lines:
            box_text = "Perf\n" + " ".join(return_lines[:3])
            if len(return_lines) > 3:
                box_text += "\n" + " ".join(return_lines[3:])
            ax.text(
                0.01,
                0.02,
                box_text,
                transform=ax.transAxes,
                fontsize=10,
                ha="left",
                va="bottom",
                linespacing=1.3,
                bbox={
                    "facecolor": "#fff7ed",
                    "alpha": 0.94,
                    "edgecolor": "#fb923c",
                    "boxstyle": "round,pad=0.55",
                },
            )
        ticker = str(row.get("alert_ticker", row.get("ticker", "")))
        ax.set_title(
            f"{ticker} - {row.get('description', '')} - {chart_years:g}Y {chart_kind}",
            fontsize=12,
        )
        ax.set_ylabel("Price", fontsize=10)
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis="both", labelsize=9)
        fig.tight_layout()
        chart_path = output_dir / f"{ticker.replace('/', '_')}_{chart_kind}.png"
        save_compressed_png(fig, chart_path)
        plt.close(fig)
        chart_paths.append(chart_path)
    return chart_paths


# ── Email sending ──


def _emit(
    subject: str,
    body: str,
    html: str,
    dry_run: bool,
    inline_images: dict[str, Path] | None = None,
) -> None:
    if dry_run:
        print("=" * 80)
        print(subject)
        print(body)
        return
    send_email(subject, body, html, inline_images)


def _send_individual(
    title: str,
    commentary: str,
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    dry_run: bool,
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        signal_charts = chart_strategy_signals(
            signals, prices, Path(tmp_dir), 1, "signal"
        )
        context_charts = chart_strategy_signals(
            signals, prices, Path(tmp_dir), 3, "context"
        )
        chart_by_key = {
            path.name.removesuffix(".png"): path
            for path in signal_charts + context_charts
        }
        content_ids = {
            key: make_msgid(domain="fintracker.local")[1:-1] for key in chart_by_key
        }
        inline_images = {
            content_ids[key]: path
            for key, path in chart_by_key.items()
            if key in content_ids
        }
        _emit(
            f"{title}: {len(signals)}",
            build_strategy_body(title, commentary, signals, prices),
            build_strategy_html(title, commentary, signals, prices, content_ids),
            dry_run,
            inline_images,
        )


def _send_consolidated(
    consolidated: pd.DataFrame,
    prices: pd.DataFrame,
    session: str,
    dry_run: bool,
) -> None:
    if dry_run:
        market = session_label(session)
        total_signals = sum(
            len(s) if isinstance(s, list) else 1 for s in consolidated.get("signal", [])
        )
        subject = (
            f"FinTracker consolidated alerts ({market}): "
            f"{len(consolidated)} instruments, {total_signals} signals"
        )
        _emit(
            subject,
            build_consolidated_body(consolidated, prices),
            build_consolidated_html(consolidated, prices, {}),
            dry_run,
        )
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        signal_charts = chart_strategy_signals(
            consolidated, prices, Path(tmp_dir), 1, "signal"
        )
        context_charts = chart_strategy_signals(
            consolidated, prices, Path(tmp_dir), 3, "context"
        )
        chart_by_key = {
            path.name.removesuffix(".png"): path
            for path in signal_charts + context_charts
        }
        content_ids = {
            key: make_msgid(domain="fintracker.local")[1:-1] for key in chart_by_key
        }
        inline_images = {
            content_ids[key]: path
            for key, path in chart_by_key.items()
            if key in content_ids
        }
        market = session_label(session)
        total_signals = sum(
            len(s) if isinstance(s, list) else 1 for s in consolidated.get("signal", [])
        )
        subject = (
            f"FinTracker consolidated alerts ({market}): "
            f"{len(consolidated)} instruments, {total_signals} signals"
        )
        _emit(
            subject,
            build_consolidated_body(consolidated, prices),
            build_consolidated_html(consolidated, prices, content_ids),
            dry_run,
            inline_images,
        )


# ── Main entry point ──


def send_strategy_alerts(args: argparse.Namespace) -> None:
    prices = load_price_history()
    latest = attach_ticker_full(load_performance(1), prices)
    raw_two_rows = latest
    assert_fresh_data(
        latest,
        label=f"strategy alerts ({session_label(args.session)})",
        allow_stale=args.allow_stale_data,
        dry_run=args.dry_run,
    )
    selected = set(args.strategy.split(",")) if args.strategy else None
    state_dir = Path(args.state_dir) if args.state_dir else DEFAULT_STATE_DIR

    strategies = build_all_signals(
        latest, raw_two_rows, prices, args.session, args.max_items, selected
    )

    changes_by_strategy: dict[str, pd.DataFrame] = {}
    any_changes = False
    any_active = False
    any_previous = False
    for strategy in strategies:
        previous = load_previous(state_dir, strategy.strategy_id, args.session)
        if previous:
            any_previous = True
        changes = detect_changes(strategy.active, previous)
        changes_by_strategy[strategy.strategy_id] = changes
        if not changes.empty:
            any_changes = True
        if not strategy.active.empty:
            any_active = True
        current_date = (
            str(latest["date"].iloc[0])
            if not latest.empty and "date" in latest.columns
            else None
        )
        if not args.dry_run:
            save_current(
                state_dir,
                strategy.strategy_id,
                args.session,
                strategy.active,
                current_date,
            )

    if args.changes_only and not any_changes:
        print(f"No strategy alert changes ({session_label(args.session)}).")
        return

    if args.active_only and not any_active:
        print(f"No active strategy signals ({session_label(args.session)}).")
        return

    if not any_changes and not args.active_only:
        print(
            f"No strategy alert changes ({session_label(args.session)}); "
            "sending consolidated active alerts."
        )

    # ── Send consolidated email ──
    consolidated = _build_consolidated(strategies, args.max_items)
    if consolidated.empty:
        print(
            f"No active strategy signals to consolidate ({session_label(args.session)})."
        )
        return

    if args.changes_only:
        changed_tickers: set[str] = set()
        for strategy in strategies:
            changes = changes_by_strategy[strategy.strategy_id]
            if not changes.empty:
                changed_tickers.update(
                    changes["alert_ticker"].dropna().astype(str).tolist()
                )
                changed_tickers.update(changes["ticker"].dropna().astype(str).tolist())
        group_col = (
            "alert_ticker" if "alert_ticker" in consolidated.columns else "ticker"
        )
        consolidated = consolidated[
            consolidated[group_col].astype(str).isin(changed_tickers)
        ].copy()
        if consolidated.empty:
            print(
                f"No changed strategy signals to consolidate ({session_label(args.session)})."
            )
            return

    _send_consolidated(consolidated, prices, args.session, args.dry_run)


def parse_args() -> argparse.Namespace:
    load_env_file()
    parser = argparse.ArgumentParser(
        description="Send FinTracker strategy alert emails."
    )
    parser.add_argument(
        "--session",
        choices=VALID_SESSIONS,
        default="all",
        help="Market session filter: European Yahoo suffixes are EU/UK; .T/.KS/.KQ/.HK are Asia; other tickers are US.",
    )
    parser.add_argument("--strategy", help="Comma-separated strategy ids to send.")
    parser.add_argument(
        "--max-items",
        type=int,
        default=int(os.environ.get("STRATEGY_ALERT_MAX_ITEMS", "100")),
        help="Max alert items per consolidated email; 0 means no cap.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print emails instead of sending SMTP messages and do not update state.",
    )
    parser.add_argument(
        "--allow-stale-data",
        action="store_true",
        help="Allow sending with non-today data; intended only for testing/development.",
    )
    parser.add_argument(
        "--active-only", action="store_true", help="Only send active-signal emails."
    )
    parser.add_argument(
        "--changes-only", action="store_true", help="Only send changes emails."
    )
    parser.add_argument("--state-dir", help="Override alert state directory.")
    return parser.parse_args()


def main() -> None:
    send_strategy_alerts(parse_args())


if __name__ == "__main__":
    main()
