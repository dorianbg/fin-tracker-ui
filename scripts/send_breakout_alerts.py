from __future__ import annotations

import os
import smtplib
import tempfile
import argparse
from email.message import EmailMessage
from email.utils import make_msgid
from pathlib import Path

import duckdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from app.views.ConsolidationSetup import scan_breakout_triggers
from app.alerts.freshness import assert_fresh_data
from app.alerts.session import VALID_SESSIONS, filter_by_session, session_label
from app.alerts.state import (
    DEFAULT_STATE_DIR,
    detect_changes,
    load_previous,
    save_current,
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


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    raise RuntimeError("Missing required environment variable: " + " or ".join(names))


def _period_return(history: pd.DataFrame, days: int) -> float | None:
    if len(history) <= days:
        return None
    start = float(history["price"].iloc[-days - 1])
    end = float(history["price"].iloc[-1])
    if start == 0:
        return None
    return (end / start - 1) * 100


def add_sma_overlays(ax, history: pd.DataFrame) -> None:
    sma_specs = [
        (21, "#f59e0b", "21 SMA", 1.05),
        (50, "#7c3aed", "50 SMA", 1.15),
        (200, "#111827", "200 SMA", 1.25),
    ]
    for window, color, label, linewidth in sma_specs:
        sma_history = history.assign(
            sma=history["price"]
            .astype(float)
            .rolling(window, min_periods=window)
            .mean()
        ).dropna(subset=["sma"])
        if not sma_history.empty:
            ax.plot(
                sma_history["date"],
                sma_history["sma"],
                color=color,
                linewidth=linewidth,
                label=label,
            )


def save_compressed_png(fig, chart_path: Path, dpi: int = 130) -> None:
    try:
        fig.savefig(
            chart_path,
            dpi=dpi,
            pil_kwargs={"optimize": False, "compress_level": 6},
        )
    except TypeError:
        fig.savefig(chart_path, dpi=dpi)


def exchange_label(ticker_full: str) -> str:
    suffix_map = {
        ".L": "London Stock Exchange",
        ".PA": "Euronext Paris",
        ".DE": "Xetra",
        ".AS": "Euronext Amsterdam",
        ".SW": "SIX Swiss Exchange",
        ".CO": "Nasdaq Copenhagen",
        ".ST": "Nasdaq Stockholm",
        ".MI": "Borsa Italiana",
        ".MC": "Bolsa de Madrid",
        ".BR": "Euronext Brussels",
        ".HE": "Nasdaq Helsinki",
        ".T": "Tokyo Stock Exchange",
        ".HK": "Hong Kong Exchange",
        ".KS": "Korea Exchange",
        ".KQ": "KOSDAQ",
        ".V": "TSX Venture",
    }
    for suffix, label in suffix_map.items():
        if ticker_full.endswith(suffix):
            return label
    if ticker_full.startswith("^"):
        return "Index"
    return "US exchange"


def build_email_body(
    alerts: pd.DataFrame, prices: pd.DataFrame, vol_map: dict[str, float]
) -> str:
    lines = ["Fresh breakout alerts", ""]
    rank = 0
    for row in alerts.itertuples(index=False):
        rank += 1
        history = prices[prices["ticker"] == row.ticker].sort_values("date")
        ticker_full = (
            str(history["ticker_full"].iloc[-1])
            if "ticker_full" in history
            else row.ticker
        )
        exchange = exchange_label(ticker_full)
        adr_pct = (row.adr20 / row.price * 100) if row.price > 0 else 0
        vol_1y = vol_map.get(row.ticker)
        sizing = f"ADR: {row.adr20:.2f} ({adr_pct:.1f}% of price)"
        if vol_1y is not None:
            sizing += f"  |  Vol 1Y: {vol_1y:.1f}%"
        perf = {
            "1W": _period_return(history, 5),
            "1M": _period_return(history, 21),
            "3M": _period_return(history, 63),
            "6M": _period_return(history, 126),
            "1Y": _period_return(history, 252),
        }
        perf_text = ", ".join(
            f"{label} {value:+.1f}%"
            for label, value in perf.items()
            if value is not None
        )
        lines.append(
            f"ALERT #{rank} (Score {row.breakout_score:.1f}): {row.ticker} - {row.description}\n"
            f"Exchange: {exchange} ({ticker_full}).\n"
            f"Reason: close crossed prior 30-day resistance and remains controlled "
            f"({row.breakout_extension_adr:.2f} ADR above breakout, "
            f"{row.extension_adr:.1f} ADR from 200MA &gt;= {row.ma200:.0f}).\n"
            f"Trigger: native close {row.price:.2f} > breakout level {row.breakout_level:.2f}.\n"
            f"Performance: {perf_text or 'not enough history'}.\n"
            f"Volatility: {sizing}.\n"
            f"Charts: 1Y trigger and 3Y context below.\n"
        )
    return "\n".join(lines)


def build_email_html(
    alerts: pd.DataFrame,
    prices: pd.DataFrame,
    content_ids: dict[str, str],
    vol_map: dict[str, float],
) -> str:
    blocks = ["<h2>Fresh breakout alerts</h2>"]
    rank = 0
    for row in alerts.itertuples(index=False):
        rank += 1
        history = prices[prices["ticker"] == row.ticker].sort_values("date")
        ticker_full = (
            str(history["ticker_full"].iloc[-1])
            if "ticker_full" in history
            else row.ticker
        )
        exchange = exchange_label(ticker_full)
        adr_pct = (row.adr20 / row.price * 100) if row.price > 0 else 0
        vol_1y = vol_map.get(row.ticker)
        sizing = f"ADR: {row.adr20:.2f} ({adr_pct:.1f}% of price)"
        if vol_1y is not None:
            sizing += f"  |  Vol 1Y: {vol_1y:.1f}%"
        perf = {
            "1W": _period_return(history, 5),
            "1M": _period_return(history, 21),
            "3M": _period_return(history, 63),
            "6M": _period_return(history, 126),
            "1Y": _period_return(history, 252),
        }
        perf_text = ", ".join(
            f"{label} {value:+.1f}%"
            for label, value in perf.items()
            if value is not None
        )
        cid = content_ids.get(f"{row.ticker}_breakout")
        context_cid = content_ids.get(f"{row.ticker}_context")
        chart_html = (
            f'<img src="cid:{cid}" style="max-width:100%; width:900px; height:auto; border:1px solid #ddd;" />'
            f'<br><img src="cid:{context_cid}" style="max-width:100%; width:900px; height:auto; border:1px solid #ddd; margin-top:8px;" />'
            if cid and context_cid
            else "<p><em>No chart available.</em></p>"
        )
        blocks.append(
            f"""
            <section style="margin:0 0 28px 0; font-family:Arial, sans-serif;">
              <h3 style="margin-bottom:6px;">ALERT #{rank} (Score {row.breakout_score:.1f}): {row.ticker} - {row.description}</h3>
              <p><strong>Exchange:</strong> {exchange} ({ticker_full}).</p>
              <p><strong>Reason:</strong> close crossed prior 30-day resistance ({row.breakout_extension_adr:.2f} ADR above breakout, {row.extension_adr:.1f} ADR from 200MA &gt;= {row.ma200:.0f}).</p>
              <p><strong>Trigger:</strong> native close {row.price:.2f} &gt; breakout level {row.breakout_level:.2f}.</p>
              <p><strong>Performance:</strong> {perf_text or "not enough history"}.</p>
              <p><strong>Volatility:</strong> {sizing}.</p>
              {chart_html}
            </section>
            """
        )
    return "\n".join(blocks)


def load_price_history_cli() -> pd.DataFrame:
    conn = duckdb.connect(str(DB_FILE), read_only=True)
    query = """
        SELECT ticker, ticker_full, date,
               open_orig AS price_open, high_orig AS price_high, low_orig AS price_low,
               price_orig AS price, description, fund_type, currency
        FROM total_return ORDER BY ticker, date
    """
    try:
        df = conn.execute(query).df()
    finally:
        conn.close()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df


def load_volatility_map() -> dict[str, float]:
    conn = duckdb.connect(str(DB_FILE), read_only=True)
    query = "SELECT ticker, vol_1y FROM latest_performance WHERE rown = 1 AND vol_1y IS NOT NULL"
    try:
        rows = conn.execute(query).fetchall()
    finally:
        conn.close()
    return {ticker: float(vol) for ticker, vol in rows if vol is not None}


def chart_breakouts(
    alerts: pd.DataFrame,
    prices: pd.DataFrame,
    output_dir: Path,
    chart_years: float,
    chart_kind: str,
) -> list[Path]:
    chart_paths = []
    end_date = prices["date"].max()
    start_date = end_date - pd.DateOffset(days=int(chart_years * 365))
    alert_by_ticker = alerts.set_index("ticker")

    for ticker in alerts["ticker"]:
        history = prices[
            (prices["ticker"] == ticker) & (prices["date"] >= start_date)
        ].copy()
        if history.empty:
            continue

        history = history.sort_values("date")
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

        alert = alert_by_ticker.loc[ticker]
        returns = {
            "1W": _period_return(history, 5),
            "1M": _period_return(history, 21),
            "3M": _period_return(history, 63),
            "6M": _period_return(history, 126),
            "1Y": _period_return(history, 252),
            "3Y": _period_return(history, min(756, len(history) - 1)),
        }
        return_lines = [
            f"{label}: {value:+.1f}%"
            for label, value in returns.items()
            if value is not None
        ]
        ax.axhline(
            alert["breakout_level"],
            color="#377eb8",
            linestyle="--",
            linewidth=1.2,
            label="Breakout level",
        )
        if return_lines:
            box_title = "Historical performance"
            box_text = box_title + "\n" + "   ".join(return_lines[:3])
            if len(return_lines) > 3:
                box_text += "\n" + "   ".join(return_lines[3:])
            ax.text(
                0.01,
                0.02,
                box_text,
                transform=ax.transAxes,
                fontsize=10,
                ha="left",
                va="bottom",
                linespacing=1.45,
                bbox={
                    "facecolor": "#fff7ed",
                    "alpha": 0.94,
                    "edgecolor": "#fb923c",
                    "boxstyle": "round,pad=0.55",
                },
            )
        ax.set_title(
            f"{ticker} - {alert['description']} - {chart_years:g}Y {chart_kind}"
        )
        ax.set_ylabel("Price")
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper left")
        fig.tight_layout()

        chart_path = output_dir / f"{ticker.replace('/', '_')}_{chart_kind}.png"
        save_compressed_png(fig, chart_path)
        plt.close(fig)
        chart_paths.append(chart_path)

    return chart_paths


def send_email(
    subject: str,
    body: str,
    html_body: str | None = None,
    inline_images: dict[str, Path] | None = None,
) -> None:
    host = _required_env("SMTP_HOST")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = _first_env("SMTP_USER", "EMAIL_SENDER")
    password = _required_env("SMTP_PASSWORD")
    sender = (
        os.environ.get("BREAKOUT_ALERT_FROM") or os.environ.get("EMAIL_SENDER") or user
    )
    recipient_text = _first_env("BREAKOUT_ALERT_TO", "EMAIL_RECIPIENTS")
    recipients = [x.strip() for x in recipient_text.split(",") if x.strip()]

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message.set_content(body)

    if html_body:
        message.add_alternative(html_body, subtype="html")
        html_part = message.get_payload()[-1]
        for cid, path in (inline_images or {}).items():
            html_part.add_related(
                path.read_bytes(),
                maintype="image",
                subtype="png",
                cid=f"<{cid}>",
                filename=path.name,
            )

    smtp_cls = smtplib.SMTP_SSL if port == 465 else smtplib.SMTP
    with smtp_cls(host, port) as smtp:
        if port != 465:
            smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send FinTracker breakout alerts.")
    parser.add_argument("--session", choices=VALID_SESSIONS, default="all")
    parser.add_argument(
        "--allow-stale-data",
        action="store_true",
        help="Allow sending with non-today data; intended only for testing/development.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file()
    max_extension = float(os.environ.get("BREAKOUT_MAX_EXTENSION_ADR", "1.5"))
    max_ma_extension = float(os.environ.get("BREAKOUT_MAX_MA_EXTENSION_ADR", "8.0"))
    max_items = int(os.environ.get("BREAKOUT_ALERT_MAX_ITEMS", "100"))
    chart_years = float(os.environ.get("BREAKOUT_ALERT_CHART_YEARS", "1"))
    prices = filter_by_session(load_price_history_cli(), args.session)
    assert_fresh_data(
        prices,
        label=f"breakout alerts ({session_label(args.session)})",
        allow_stale=args.allow_stale_data,
    )
    vol_map = load_volatility_map()
    alerts = scan_breakout_triggers(
        prices,
        max_breakout_extension_adr=max_extension,
        max_extension_adr=max_ma_extension,
    )
    if alerts.empty:
        print("No fresh breakout alerts.")
        return

    # dedup: skip if identical to last run
    breakouts = alerts.copy()
    breakouts["rank"] = range(1, len(breakouts) + 1)
    breakouts["signal"] = "Fresh resistance breakout"
    previous = load_previous(DEFAULT_STATE_DIR, "breakout", args.session)
    changes = detect_changes(breakouts, previous)
    if changes.empty:
        print(f"No new breakout changes ({session_label(args.session)}).")
        return
    print(f"Sending {len(changes)} breakout changes ({session_label(args.session)}).")

    if max_items > 0:
        alerts = alerts.head(max_items).copy()
    market = session_label(args.session)
    body = build_email_body(alerts, prices, vol_map)
    print(body)
    with tempfile.TemporaryDirectory() as tmp_dir:
        trigger_charts = chart_breakouts(
            alerts, prices, Path(tmp_dir), chart_years, "breakout"
        )
        context_charts = chart_breakouts(alerts, prices, Path(tmp_dir), 3, "context")
        chart_by_key = {
            path.name.removesuffix(".png"): path
            for path in trigger_charts + context_charts
        }
        content_ids = {
            key: make_msgid(domain="fintracker.local")[1:-1] for key in chart_by_key
        }
        html = build_email_html(alerts, prices, content_ids, vol_map)
        inline_images = {
            content_ids[key]: path
            for key, path in chart_by_key.items()
            if key in content_ids
        }
        send_email(
            f"FinTracker breakout alerts ({market}): {len(alerts)}",
            body,
            html,
            inline_images,
        )

    save_current(DEFAULT_STATE_DIR, "breakout", args.session, breakouts)


if __name__ == "__main__":
    main()
