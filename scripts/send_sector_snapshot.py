from __future__ import annotations

import argparse
import os
import tempfile
from email.utils import make_msgid
from pathlib import Path

import duckdb
import pandas as pd

from app.alerts.freshness import assert_fresh_data
from app.views.SectorSnapshot import (
    DEFAULT_BENCHMARK,
    DEFAULT_DAYS,
    DEFAULT_TOP_N,
    SNAPSHOT_GROUPS,
    save_grouped_snapshot_pngs,
)
from send_breakout_alerts import send_email


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


def load_price_history() -> pd.DataFrame:
    conn = duckdb.connect(str(DB_FILE), read_only=True)
    query = """
        SELECT ticker, ticker_full, date, price_orig AS price, description, fund_type
        FROM total_return
        ORDER BY ticker, date
    """
    try:
        df = conn.execute(query).df()
    finally:
        conn.close()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df


def build_body(grouped: dict[str, tuple[Path, pd.DataFrame]], benchmark: str) -> str:
    lines = [
        "FinTracker daily relative strength snapshot",
        "",
        f"Benchmark: {benchmark}",
        "",
    ]
    for group, (_, snapshot) in grouped.items():
        meta = SNAPSHOT_GROUPS[group]
        lines.extend([str(meta["title"]), str(meta["description"])])
        if snapshot.empty:
            lines.extend(["No eligible instruments.", ""])
            continue
        for i, row in snapshot.iterrows():
            rank = int(i) + 1
            lines.append(
                f"#{rank}: {row['ticker']} - {row['description']} | "
                f"RS {float(row['relative_strength']):+.2f}% | "
                f"recent {float(row['trend_delta']):+.2f}%"
            )
        lines.append("")
    return "\n".join(lines)


def build_html(
    grouped: dict[str, tuple[Path, pd.DataFrame]],
    benchmark: str,
    content_ids: dict[str, str],
) -> str:
    sections = []
    for group, (_, snapshot) in grouped.items():
        meta = SNAPSHOT_GROUPS[group]
        cid = content_ids[group]
        rows = "".join(
            f"<tr><td>{int(i) + 1}</td><td>{row['ticker']}</td><td>{row['description']}</td>"
            f"<td style='text-align:right'>{float(row['relative_strength']):+.2f}%</td>"
            f"<td style='text-align:right'>{float(row['trend_delta']):+.2f}%</td></tr>"
            for i, row in snapshot.iterrows()
        )
        table = (
            f"""
            <table cellpadding="4" cellspacing="0" style="border-collapse:collapse;font-size:13px;margin-top:8px;">
              <thead><tr><th>#</th><th>Ticker</th><th>Description</th><th>RS</th><th>Recent</th></tr></thead>
              <tbody>{rows}</tbody>
            </table>
            """
            if rows
            else "<p><em>No eligible instruments.</em></p>"
        )
        sections.append(
            f"""
            <section style="font-family:Arial,sans-serif;margin:0 0 28px 0;">
              <h3 style="margin-bottom:4px;">{meta["title"]}</h3>
              <p style="margin-top:0;color:#4b5563;">{meta["description"]}</p>
              <img src="cid:{cid}" style="max-width:100%;width:1200px;height:auto;border:1px solid #e5e7eb;" />
              {table}
            </section>
            """
        )
    return f"""
    <section style="font-family:Arial,sans-serif;">
      <h2>FinTracker daily relative strength snapshot</h2>
      <p>Relative strength vs {benchmark}.</p>
      {"".join(sections)}
    </section>
    """


def _output_dir(output: str | None) -> Path | None:
    if not output:
        return None
    path = Path(output)
    if path.suffix.lower() == ".png":
        return path.with_suffix("")
    return path


def send_sector_snapshot(args: argparse.Namespace) -> None:
    load_env_file()
    prices = load_price_history()
    assert_fresh_data(
        prices,
        label="sector snapshot",
        allow_stale=args.allow_stale_data,
        dry_run=args.dry_run,
    )
    tmp = None
    output_dir = _output_dir(args.output)
    if output_dir is None:
        tmp = tempfile.TemporaryDirectory()
        output_dir = Path(tmp.name)
    grouped = save_grouped_snapshot_pngs(
        prices,
        output_dir,
        benchmark=args.benchmark,
        days=args.days,
        top_n=args.top_n,
    )
    for image_path, _ in grouped.values():
        print(f"Wrote {image_path}")

    total = sum(len(snapshot) for _, snapshot in grouped.values())
    subject = f"FinTracker daily relative strength snapshot: {len(grouped)} groups, {total} instruments"
    body = build_body(grouped, args.benchmark)
    if args.dry_run:
        print(subject)
        print(body)
        return

    content_ids = {
        group: make_msgid(domain="fintracker.local")[1:-1] for group in grouped
    }
    inline_images = {content_ids[group]: path for group, (path, _) in grouped.items()}
    send_email(
        subject,
        body,
        build_html(grouped, args.benchmark, content_ids),
        inline_images,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export and email the FinTracker daily relative strength snapshot."
    )
    parser.add_argument(
        "--benchmark", default=os.environ.get("SNAPSHOT_BENCHMARK", DEFAULT_BENCHMARK)
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=int(os.environ.get("SNAPSHOT_TOP_N", DEFAULT_TOP_N)),
    )
    parser.add_argument(
        "--days", type=int, default=int(os.environ.get("SNAPSHOT_DAYS", DEFAULT_DAYS))
    )
    parser.add_argument(
        "--output",
        help="Write PNG to this path instead of using only a temporary email image.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Generate/print but do not send email."
    )
    parser.add_argument(
        "--allow-stale-data",
        action="store_true",
        help="Allow non-today data for development/testing.",
    )
    return parser.parse_args()


def main() -> None:
    send_sector_snapshot(parse_args())


if __name__ == "__main__":
    main()
