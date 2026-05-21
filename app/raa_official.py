from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen


RAA_HOLDINGS_URL = "https://3fourteensmi.com/raa#holdings"
CACHE_FILE = Path(__file__).resolve().parent.parent / "resources" / "raa_current_allocation.json"

LABEL_MAP = {
    "Bitcoin": "Bitcoin",
    "Commodities": "Commodities",
    "Energy": "Energy",
    "Gold": "Gold",
    "Managed Futures": "Managed Futures",
    "Miners": "Miners",
    "Real Estate": "Real Estate",
    "Dividend Payers": "Dividend Payers",
    "Emerging Ex. China": "EM ex-China",
    "Europe": "Europe",
    "Japan": "Japan",
    "Nasdaq": "Nasdaq",
    "US Large Cap": "US Large Cap",
    "US Small Cap": "US Small Cap",
    "Corporate Bonds": "Corporate Bonds",
    "Emerging Mkt Bonds": "EM Bonds",
    "High Yield": "High Yield",
    "Long Term Treasuries": "Long-Term Treasuries",
    "T-Bills": "T-Bills",
    "TIPS": "TIPS",
}


@dataclass(frozen=True)
class OfficialRAAAllocation:
    allocation: dict[str, float]
    as_of: str | None
    source: str


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items: list[str] = []

    def handle_data(self, data):
        text = " ".join(data.split())
        if text:
            self.items.append(text)


def fetch_official_raa_allocation(
    url: str = RAA_HOLDINGS_URL,
    timeout: int = 20,
    use_cache: bool = True,
) -> OfficialRAAAllocation | None:
    try:
        allocation = _fetch_live_allocation(url=url, timeout=timeout)
        _write_cache(allocation)
        return allocation
    except Exception:
        if use_cache:
            return _read_cache()
        raise


def _fetch_live_allocation(url: str, timeout: int) -> OfficialRAAAllocation:
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
            )
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    return parse_official_raa_allocation(html, source=url)


def parse_official_raa_allocation(html: str, source: str = RAA_HOLDINGS_URL) -> OfficialRAAAllocation:
    parser = _TextExtractor()
    parser.feed(html)
    items = parser.items

    try:
        start = items.index("Asset Allocation")
        end = items.index("Holdings", start)
    except ValueError as exc:
        raise ValueError("Could not find RAA Asset Allocation section") from exc

    section = items[start:end]
    as_of = _extract_as_of(section)
    allocation: dict[str, float] = {}
    for i, item in enumerate(section[:-1]):
        asset = LABEL_MAP.get(item)
        if asset is None:
            continue
        pct = _parse_percent(section[i + 1])
        if pct is not None:
            allocation[asset] = pct

    missing = sorted(set(LABEL_MAP.values()) - set(allocation))
    if missing:
        raise ValueError(f"Missing RAA allocation assets: {missing}")

    total = sum(allocation.values())
    if total <= 0:
        raise ValueError("RAA allocation parsed to zero")
    allocation = {asset: weight / total for asset, weight in allocation.items()}
    return OfficialRAAAllocation(allocation=allocation, as_of=as_of, source=source)


def _extract_as_of(items: list[str]) -> str | None:
    for i, item in enumerate(items[:-1]):
        if item == "As of":
            return items[i + 1]
        match = re.search(r"As of\s+([0-9]{1,2}/[0-9]{1,2}/[0-9]{4})", item)
        if match:
            return match.group(1)
    return None


def _parse_percent(value: str) -> float | None:
    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)%", value.strip())
    if not match:
        return None
    return float(match.group(1)) / 100.0


def _write_cache(allocation: OfficialRAAAllocation) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps(
            {
                "as_of": allocation.as_of,
                "source": allocation.source,
                "allocation": allocation.allocation,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _read_cache() -> OfficialRAAAllocation | None:
    if not CACHE_FILE.exists():
        return None
    data = json.loads(CACHE_FILE.read_text())
    allocation = {asset: float(weight) for asset, weight in data["allocation"].items()}
    return OfficialRAAAllocation(
        allocation=allocation,
        as_of=data.get("as_of"),
        source=data.get("source", str(CACHE_FILE)),
    )


if __name__ == "__main__":
    official = fetch_official_raa_allocation(use_cache=False)
    print(json.dumps({"as_of": official.as_of, "allocation": official.allocation}, indent=2))
