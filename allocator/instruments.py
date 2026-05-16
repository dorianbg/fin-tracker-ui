"""Instrument metadata for the antifragile allocator.

Wrapper rules:
- SIPP       → UCITS ETFs, ETCs, platform mutual funds, and direct stocks
- ISA        → UCITS ETFs, ETCs, and platform mutual funds
- GIA        → HMRC reporting funds and direct stocks
"""

import csv
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Instrument:
    ticker: str
    name: str
    sleeve: str
    aum_bn_usd: float  # For stocks this is left as 0.0 unless populated later.
    listing: str  # 'LSE_GBP' | 'LSE_USD' | 'NYSE' | 'NASDAQ' | 'PLATFORM' | 'US' | 'EU'
    ccy: str
    vehicle_type: str  # 'ucits_etf' | 'mutual_fund' | 'us_etf' | 'etc' | 'stock'
    domicile: str      # 'UK' | 'IE' | 'LU' | 'US' | 'DE'
    accumulating: bool
    is_reporting_fund: bool  # assumed True for the recommended US ETFs
    yfinance_symbol: str | None = None
    ibkr_symbol: str | None = None
    ibkr_primary_exchange: str | None = None
    ibkr_currency: str | None = None
    country: str | None = None
    wrapper_eligible: dict[str, bool] = field(
        default_factory=lambda: {"SIPP": False, "ISA": False, "GIA": False}
    )


# ── Sleeve labels (single source of truth) ──
# ── Equity Sectors & Factors (Bucket 1 & Bucket 2) ────────────────────
SLEEVE_MARKET_CAP   = "equity_market_cap"
SLEEVE_MIN_VOL      = "equity_min_vol"
SLEEVE_QUALITY      = "equity_quality"
SLEEVE_VALUE        = "equity_value"
SLEEVE_EM           = "equity_em"
SLEEVE_JAPAN        = "equity_japan"
SLEEVE_EUROPE       = "equity_europe"
SLEEVE_UK_COMMODITY = "equity_uk_commodity"
SLEEVE_DEFENCE      = "equity_defence"

# ── Inflation & Bonds (Bucket 1 & Bucket 2) ───────────────────────────
SLEEVE_LINKERS_GLOBAL = "bonds_linkers_global"
SLEEVE_LINKERS_UK     = "bonds_linkers_uk"
SLEEVE_EM_BONDS_LOCAL = "bonds_em_local"
SLEEVE_EM_BONDS       = "bonds_em_hard_ccy"
SLEEVE_LONG_DUR       = "bonds_long_duration"
SLEEVE_TIPS_US        = "bonds_tips_us"
SLEEVE_CASH_GBP       = "cash_gbp"
SLEEVE_CASH_USD       = "cash_usd"

# ── Real Assets (Bucket 2) ────────────────────────────────────────────
SLEEVE_PRECIOUS_METALS = "real_precious_metals"
SLEEVE_METALS_MINERS   = "real_metals_miners"
SLEEVE_GOLD            = SLEEVE_PRECIOUS_METALS  # back-compat alias
SLEEVE_GOLD_MINERS     = SLEEVE_METALS_MINERS    # back-compat alias
SLEEVE_COMMODITIES     = "real_commodities"
SLEEVE_ENERGY         = "real_energy"
SLEEVE_INFRA          = "real_infrastructure"
SLEEVE_REITS          = "real_reits"

# ── Thematic / satellite sleeves (GIA optional overweights) ──
SLEEVE_NUCLEAR = "thematic_nuclear"       # uranium + nuclear energy
SLEEVE_CLEAN_ENERGY = "thematic_clean_energy"  # solar/wind/renewables
SLEEVE_AI = "thematic_ai"
SLEEVE_SEMICONDUCTORS = "thematic_semiconductors"
SLEEVE_CYBERSECURITY = "thematic_cybersecurity"
SLEEVE_HEALTHCARE = "equity_healthcare"   # low-PEG defensive growth
SLEEVE_BIOTECH = "equity_biotech"
SLEEVE_SOFTWARE = "equity_software"
SLEEVE_TECHNOLOGY = "equity_technology"
SLEEVE_INDUSTRIALS = "equity_industrials" # reshoring capex supercycle
SLEEVE_MATERIALS = "equity_materials"     # resource nationalism + energy transition metals
SLEEVE_FINANCIALS = "equity_financials"   # historically cheap vs own PE history
SLEEVE_UTILITIES = "equity_utilities"
SLEEVE_CONSUMER_STAPLES = "equity_consumer_staples"
SLEEVE_CONSUMER_DISCRETIONARY = "equity_consumer_discretionary"
SLEEVE_COMMUNICATION = "equity_communication"
SLEEVE_REAL_ESTATE = "equity_real_estate"
SLEEVE_US_EQUAL_WEIGHT = "equity_us_equal_weight"
SLEEVE_US_SMALL_CAP = "equity_us_small_cap"
SLEEVE_US_MID_CAP = "equity_us_mid_cap"
SLEEVE_US_LOW_VOL = "equity_us_low_vol"
SLEEVE_GLOBAL_EX_US = "equity_global_ex_us"
SLEEVE_DEV_GROWTH = "equity_dm_growth"
SLEEVE_DEV_VALUE = "equity_dm_value"
SLEEVE_EM_VALUE = "equity_em_value"
SLEEVE_EM_DIVIDEND = "equity_em_dividend"
SLEEVE_EM_MIN_VOL = "equity_em_min_vol"
SLEEVE_BRAZIL = "equity_brazil"
SLEEVE_LATAM = "equity_latam"
SLEEVE_TAIWAN = "equity_taiwan"
SLEEVE_KOREA = "equity_korea"
SLEEVE_INDIA = "equity_india"
SLEEVE_MALAYSIA = "equity_malaysia"
SLEEVE_CHINA = "equity_china"
SLEEVE_CHINA_TECH = "equity_china_tech"
SLEEVE_FRANCE = "equity_france"
SLEEVE_PACIFIC = "equity_pacific"
SLEEVE_TRANSPORT = "equity_transport"
SLEEVE_HOMEBUILDERS = "equity_homebuilders"


def _ucits(*wrappers: str) -> dict[str, bool]:
    """Helper to mark UCITS instruments eligible across SIPP/ISA/GIA."""
    base = {"SIPP": False, "ISA": False, "GIA": False}
    for w in wrappers:
        base[w] = True
    return base


def _eligible_in_wrapper(ins: Instrument, wrapper: str) -> bool:
    """Apply hard wrapper rules, not just manually-entered eligibility."""
    if not ins.wrapper_eligible.get(wrapper, False):
        return False

    if ins.vehicle_type == "stock":
        return wrapper in {"SIPP", "GIA"}

    if wrapper in {"SIPP", "ISA"}:
        return ins.vehicle_type in {"ucits_etf", "mutual_fund", "etc"}

    if wrapper == "GIA":
        return ins.is_reporting_fund

    return False


def _us_etf(
    ticker: str,
    name: str,
    sleeve: str,
    aum_bn_usd: float,
    listing: str = "NYSE",
    reporting: bool = True,
) -> Instrument:
    return Instrument(
        ticker=ticker,
        name=name,
        sleeve=sleeve,
        aum_bn_usd=aum_bn_usd,
        listing=listing,
        ccy="USD",
        vehicle_type="us_etf",
        domicile="US",
        accumulating=False,
        is_reporting_fund=reporting,
        wrapper_eligible={"SIPP": False, "ISA": False, "GIA": True},
    )


def _stock(
    ticker: str,
    name: str,
    sleeve: str,
    *,
    ccy: str,
    listing: str,
    domicile: str,
    yfinance_symbol: str | None = None,
    ibkr_symbol: str | None = None,
    ibkr_primary_exchange: str | None = None,
    ibkr_currency: str | None = None,
    country: str | None = None,
) -> Instrument:
    return Instrument(
        ticker=ticker,
        name=name,
        sleeve=sleeve,
        aum_bn_usd=0.0,
        listing=listing,
        ccy=ccy,
        vehicle_type="stock",
        domicile=domicile,
        accumulating=False,
        is_reporting_fund=True,
        yfinance_symbol=yfinance_symbol or ticker,
        ibkr_symbol=ibkr_symbol or ticker,
        ibkr_primary_exchange=ibkr_primary_exchange,
        ibkr_currency=ibkr_currency or ccy,
        country=country or domicile,
        wrapper_eligible={"SIPP": True, "ISA": False, "GIA": True},
    )


def _sleeve_from_sector(sector: str) -> str:
    text = (sector or "").strip().lower()
    mapping = [
        (("aerospace", "defence"), SLEEVE_DEFENCE),
        (("mining", "metals", "chemicals"), SLEEVE_MATERIALS),
        (("oil", "gas", "energy"), SLEEVE_ENERGY),
        (("bank", "insurance", "financial", "investment"), SLEEVE_FINANCIALS),
        (("pharmaceutical", "biotechnology", "health"), SLEEVE_HEALTHCARE),
        (("software",), SLEEVE_SOFTWARE),
        (("technology", "electronic"), SLEEVE_TECHNOLOGY),
        (("telecommunications", "mobile telecommunications", "media", "communication"), SLEEVE_COMMUNICATION),
        (("real estate",), SLEEVE_REAL_ESTATE),
        (("utilities", "power producers"), SLEEVE_UTILITIES),
        (("household goods", "home construction", "homebuilding"), SLEEVE_HOMEBUILDERS),
        (("food", "beverages", "tobacco", "personal goods", "staples"), SLEEVE_CONSUMER_STAPLES),
        (("retail", "travel", "leisure", "hospitality"), SLEEVE_CONSUMER_DISCRETIONARY),
        (("industrial", "support services", "engineering", "industrial goods"), SLEEVE_INDUSTRIALS),
        (("materials",), SLEEVE_MATERIALS),
    ]
    for needles, sleeve in mapping:
        if any(needle in text for needle in needles):
            return sleeve
    return SLEEVE_MARKET_CAP


def _load_stock_universe_from_csv(path: str) -> dict[str, Instrument]:
    if not os.path.exists(path):
        return {}

    out: dict[str, Instrument] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        first_line = fh.readline()
        fh.seek(0)

        if first_line.lower().startswith("ticker,"):
            reader = csv.DictReader(fh)
            rows = list(reader)
        else:
            reader = csv.reader(fh)
            rows = []
            for raw in reader:
                if len(raw) < 6:
                    continue
                raw_ticker = str(raw[0]).strip().upper()
                if "." in raw_ticker:
                    # Legacy headerless file is primarily the US stock universe.
                    # Skip Yahoo-formatted non-US symbols here; dedicated CSVs
                    # provide cleaner metadata for UK/EU names.
                    continue
                rows.append(
                    {
                        "ticker": raw_ticker,
                        "name": raw[1],
                        "ccy": raw[2],
                        "listing": "US",
                        "domicile": "US",
                        "country": "US",
                        "sector": raw[5],
                        "yfinance_symbol": raw_ticker,
                        "ibkr_symbol": raw_ticker,
                        "ibkr_primary_exchange": "SMART",
                        "ibkr_currency": raw[2],
                    }
                )

        for row in rows:
            ticker = str(row.get("ticker") or "").strip().upper()
            name = str(row.get("name") or "").strip()
            if not ticker or not name:
                continue
            out[ticker] = _stock(
                ticker=ticker,
                name=name,
                sleeve=_sleeve_from_sector(str(row.get("sector") or "")),
                ccy=str(row.get("ccy") or "USD").strip().upper(),
                listing=str(row.get("listing") or "US").strip().upper(),
                domicile=str(row.get("domicile") or row.get("country") or "US").strip().upper(),
                yfinance_symbol=str(row.get("yfinance_symbol") or "").strip() or None,
                ibkr_symbol=str(row.get("ibkr_symbol") or "").strip() or None,
                ibkr_primary_exchange=str(row.get("ibkr_primary_exchange") or "").strip() or None,
                ibkr_currency=str(row.get("ibkr_currency") or "").strip().upper() or None,
                country=str(row.get("country") or "").strip() or None,
            )
    return out


# ── Universe ──
# All AUMs are approximate as of plan date (2026-04). Update via the
# instruments.py refresh script if AUM drops below the wrapper rule
# (>$2B for UCITS, >$5B for US ETFs).
INSTRUMENTS: dict[str, Instrument] = {
    # Broad market-cap beta (UCITS, all wrappers)
    "CSPX": Instrument(
        ticker="CSPX",
        name="iShares Core S&P 500 UCITS ETF",
        sleeve=SLEEVE_MARKET_CAP,
        aum_bn_usd=50.0,
        listing="LSE_GBP",
        ccy="GBP",
        vehicle_type="ucits_etf",
        domicile="IE",
        accumulating=True,
        is_reporting_fund=True,
        wrapper_eligible=_ucits("SIPP", "ISA", "GIA"),
    ),
    "IWDA": Instrument(
        ticker="IWDA",
        name="iShares Core MSCI World UCITS ETF",
        sleeve=SLEEVE_MARKET_CAP,
        aum_bn_usd=80.0,
        listing="LSE_GBP",
        ccy="GBP",
        vehicle_type="ucits_etf",
        domicile="IE",
        accumulating=True,
        is_reporting_fund=True,
        wrapper_eligible=_ucits("SIPP", "ISA", "GIA"),
    ),
    # Equity factor (UCITS, all wrappers)
    "IWQU": Instrument(
        ticker="IWQU",
        name="iShares Edge MSCI World Quality Factor UCITS",
        sleeve=SLEEVE_QUALITY,
        aum_bn_usd=5.0,
        listing="LSE_GBP",
        ccy="GBP",
        vehicle_type="ucits_etf",
        domicile="IE",
        accumulating=True,
        is_reporting_fund=True,
        wrapper_eligible=_ucits("SIPP", "ISA", "GIA"),
    ),
    "MVOL": Instrument(
        ticker="MVOL",
        name="iShares Edge MSCI World Minimum Volatility UCITS",
        sleeve=SLEEVE_MIN_VOL,
        aum_bn_usd=2.5,
        listing="LSE_GBP",
        ccy="GBP",
        vehicle_type="ucits_etf",
        domicile="IE",
        accumulating=True,
        is_reporting_fund=True,
        wrapper_eligible=_ucits("SIPP", "ISA", "GIA"),
    ),
    "IWVL": Instrument(
        ticker="IWVL",
        name="iShares Edge MSCI World Value Factor UCITS",
        sleeve=SLEEVE_VALUE,
        aum_bn_usd=2.5,
        listing="LSE_GBP",
        ccy="GBP",
        vehicle_type="ucits_etf",
        domicile="IE",
        accumulating=True,
        is_reporting_fund=True,
        wrapper_eligible=_ucits("SIPP", "ISA", "GIA"),
    ),
    # Equity regional core (UCITS, all wrappers)
    "EIMI": Instrument(
        ticker="EIMI",
        name="iShares Core MSCI EM IMI UCITS",
        sleeve=SLEEVE_EM,
        aum_bn_usd=20.0,
        listing="LSE_GBP",
        ccy="GBP",
        vehicle_type="ucits_etf",
        domicile="IE",
        accumulating=True,
        is_reporting_fund=True,
        wrapper_eligible=_ucits("SIPP", "ISA", "GIA"),
    ),
    "IJPA": Instrument(
        ticker="IJPA",
        name="iShares MSCI Japan UCITS",
        sleeve=SLEEVE_JAPAN,
        aum_bn_usd=5.0,
        listing="LSE_GBP",
        ccy="GBP",
        vehicle_type="ucits_etf",
        domicile="IE",
        accumulating=True,
        is_reporting_fund=True,
        wrapper_eligible=_ucits("SIPP", "ISA", "GIA"),
    ),
    "VEUR": Instrument(
        ticker="VEUR",
        name="Vanguard FTSE Developed Europe UCITS ETF",
        sleeve=SLEEVE_EUROPE,
        aum_bn_usd=3.6,
        listing="LSE",
        ccy="GBP",
        vehicle_type="ucits_etf",
        domicile="IE",
        accumulating=False,
        is_reporting_fund=True,
        wrapper_eligible={"SIPP": True, "ISA": True, "GIA": True},
    ),
    "NATO": Instrument(
        ticker="NATO",
        name="HANetf Future of Defence UCITS ETF",
        sleeve=SLEEVE_DEFENCE,
        aum_bn_usd=0.4,
        listing="LSE",
        ccy="GBP",
        vehicle_type="ucits_etf",
        domicile="IE",
        accumulating=True,
        is_reporting_fund=True,
        wrapper_eligible={"SIPP": True, "ISA": True, "GIA": True},
    ),
    # Precious metals (UCITS ETCs, all wrappers) — PHGP is the primary gold
    # exposure because yfinance reports SGLN with an un-adjusted share split
    # artefact around 2026-03-30 that shows as a spurious -25% move.
    "PHGP": Instrument(
        ticker="PHGP",
        name="WisdomTree Physical Gold",
        sleeve=SLEEVE_PRECIOUS_METALS,
        aum_bn_usd=20.0,
        listing="LSE_GBP",
        ccy="GBP",
        vehicle_type="etc",
        domicile="IE",
        accumulating=True,
        is_reporting_fund=True,
        wrapper_eligible=_ucits("SIPP", "ISA", "GIA"),
    ),
    "PHSP": Instrument(
        ticker="PHSP",
        name="WisdomTree Physical Silver",
        sleeve=SLEEVE_PRECIOUS_METALS,
        aum_bn_usd=1.5,
        listing="LSE_GBP",
        ccy="GBP",
        vehicle_type="etc",
        domicile="IE",
        accumulating=True,
        is_reporting_fund=True,
        wrapper_eligible=_ucits("SIPP", "ISA", "GIA"),
    ),
    "SGLN": Instrument(
        ticker="SGLN",
        name="iShares Physical Gold ETC",
        sleeve=SLEEVE_PRECIOUS_METALS,
        aum_bn_usd=15.0,
        listing="LSE_GBP",
        ccy="GBP",
        vehicle_type="etc",
        domicile="IE",
        accumulating=True,
        is_reporting_fund=True,
        wrapper_eligible=_ucits("SIPP", "ISA", "GIA"),
    ),
    # ── Bonds & Linkers ───────────────────────────────────────────────────
    "IGIL": Instrument(
        ticker="IGIL",
        name="iShares Global Inflation Linked Govt Bond UCITS ETF",
        sleeve=SLEEVE_LINKERS_GLOBAL,
        aum_bn_usd=2.3,
        listing="LSE",
        ccy="GBP",
        vehicle_type="ucits_etf",
        domicile="IE",
        accumulating=False,
        is_reporting_fund=True,
        wrapper_eligible={"SIPP": True, "ISA": True, "GIA": True},
    ),
    "INXG": Instrument(
        ticker="INXG",
        name="iShares £ Inflation-Linked Gilts UCITS ETF",
        sleeve=SLEEVE_LINKERS_UK,
        aum_bn_usd=1.3,
        listing="LSE",
        ccy="GBP",
        vehicle_type="ucits_etf",
        domicile="IE",
        accumulating=False,
        is_reporting_fund=True,
        wrapper_eligible={"SIPP": True, "ISA": True, "GIA": True},
    ),
    # GIA-only US ETFs (real-asset sandbox). GLD + SLV are the primary
    # GIA precious-metals exposure on liquidity; IAU kept as a cheaper
    # gold alternate. GDX + SIL form the metals-miners sleeve.
    "GLD": Instrument(
        ticker="GLD",
        name="SPDR Gold Shares",
        sleeve=SLEEVE_PRECIOUS_METALS,
        aum_bn_usd=80.0,
        listing="NYSE",
        ccy="USD",
        vehicle_type="us_etf",
        domicile="US",
        accumulating=True,
        is_reporting_fund=True,  # assumed per user direction
        wrapper_eligible=_ucits("GIA"),
    ),
    "IAU": Instrument(
        ticker="IAU",
        name="iShares Gold Trust",
        sleeve=SLEEVE_PRECIOUS_METALS,
        aum_bn_usd=35.0,
        listing="NYSE",
        ccy="USD",
        vehicle_type="us_etf",
        domicile="US",
        accumulating=True,
        is_reporting_fund=True,  # assumed per user direction
        wrapper_eligible=_ucits("GIA"),
    ),
    "SLV": Instrument(
        ticker="SLV",
        name="iShares Silver Trust",
        sleeve=SLEEVE_PRECIOUS_METALS,
        aum_bn_usd=14.0,
        listing="NYSE",
        ccy="USD",
        vehicle_type="us_etf",
        domicile="US",
        accumulating=True,
        is_reporting_fund=True,  # assumed per user direction
        wrapper_eligible=_ucits("GIA"),
    ),
    "GDX": Instrument(
        ticker="GDX",
        name="VanEck Gold Miners ETF",
        sleeve=SLEEVE_METALS_MINERS,
        aum_bn_usd=13.0,
        listing="NYSE",
        ccy="USD",
        vehicle_type="us_etf",
        domicile="US",
        accumulating=False,
        is_reporting_fund=True,  # assumed per user direction
        wrapper_eligible=_ucits("GIA"),
    ),
    "SIL": Instrument(
        ticker="SIL",
        name="Global X Silver Miners ETF",
        sleeve=SLEEVE_METALS_MINERS,
        aum_bn_usd=1.5,
        listing="NYSE",
        ccy="USD",
        vehicle_type="us_etf",
        domicile="US",
        accumulating=False,
        is_reporting_fund=True,  # assumed per user direction
        wrapper_eligible=_ucits("GIA"),
    ),
    "PDBC": Instrument(
        ticker="PDBC",
        name="Invesco Optimum Yield Diversified Commodity Strategy",
        sleeve=SLEEVE_COMMODITIES,
        aum_bn_usd=5.0,
        listing="NASDAQ",
        ccy="USD",
        vehicle_type="us_etf",
        domicile="US",
        accumulating=True,
        is_reporting_fund=True,  # assumed per user direction
        wrapper_eligible=_ucits("GIA"),
    ),
    "IGF": Instrument(
        ticker="IGF",
        name="iShares Global Infrastructure ETF",
        sleeve=SLEEVE_INFRA,
        aum_bn_usd=5.0,
        listing="NASDAQ",
        ccy="USD",
        vehicle_type="us_etf",
        domicile="US",
        accumulating=False,
        is_reporting_fund=True,  # assumed per user direction
        wrapper_eligible=_ucits("GIA"),
    ),
    "VNQ": Instrument(
        ticker="VNQ",
        name="Vanguard Real Estate ETF",
        sleeve=SLEEVE_REITS,
        aum_bn_usd=30.0,
        listing="NYSE",
        ccy="USD",
        vehicle_type="us_etf",
        domicile="US",
        accumulating=False,
        is_reporting_fund=True,  # assumed per user direction
        wrapper_eligible=_ucits("GIA"),
    ),
    "TIP": Instrument(
        ticker="TIP",
        name="iShares TIPS Bond ETF",
        sleeve=SLEEVE_TIPS_US,
        aum_bn_usd=20.0,
        listing="NYSE",
        ccy="USD",
        vehicle_type="us_etf",
        domicile="US",
        accumulating=False,
        is_reporting_fund=True,  # assumed per user direction
        wrapper_eligible=_ucits("GIA"),
    ),
    "EMB": Instrument(
        ticker="EMB",
        name="iShares JPM USD Emerging Markets Bond ETF",
        sleeve=SLEEVE_EM_BONDS,
        aum_bn_usd=14.0,
        listing="NASDAQ",
        ccy="USD",
        vehicle_type="us_etf",
        domicile="US",
        accumulating=False,
        is_reporting_fund=True,  # assumed per user direction
        wrapper_eligible=_ucits("GIA"),
    ),
    "TLT": Instrument(
        ticker="TLT",
        name="iShares 20+ Year Treasury Bond ETF",
        sleeve=SLEEVE_LONG_DUR,
        aum_bn_usd=50.0,
        listing="NASDAQ",
        ccy="USD",
        vehicle_type="us_etf",
        domicile="US",
        accumulating=False,
        is_reporting_fund=True,  # assumed per user direction
        wrapper_eligible=_ucits("GIA"),
    ),
    # Commodity-producer equity proxies
    "ISF": Instrument(
        ticker="ISF",
        name="iShares Core FTSE 100 UCITS ETF",
        sleeve=SLEEVE_UK_COMMODITY,
        aum_bn_usd=13.0,
        listing="LSE_GBP",
        ccy="GBP",
        vehicle_type="ucits_etf",
        domicile="IE",
        accumulating=False,  # distributing; use ISF (dist) or CSUK (acc) — prefer acc in ISA/SIPP
        is_reporting_fund=True,
        wrapper_eligible=_ucits("SIPP", "ISA", "GIA"),
    ),
    "XLE": Instrument(
        ticker="XLE",
        name="Energy Select Sector SPDR Fund",
        sleeve=SLEEVE_ENERGY,
        aum_bn_usd=40.0,
        listing="NYSE",
        ccy="USD",
        vehicle_type="us_etf",
        domicile="US",
        accumulating=False,
        is_reporting_fund=True,  # assumed per user direction
        wrapper_eligible=_ucits("GIA"),
    ),
    # Local-currency EM bonds
    "SEML": Instrument(
        ticker="SEML",
        name="iShares JPM EM Local Govt Bond UCITS ETF",
        sleeve=SLEEVE_EM_BONDS_LOCAL,
        aum_bn_usd=1.5,   # borderline — best available UCITS for this exposure
        listing="LSE_GBP",
        ccy="GBP",
        vehicle_type="ucits_etf",
        domicile="IE",
        accumulating=False,
        is_reporting_fund=True,
        wrapper_eligible=_ucits("SIPP", "ISA", "GIA"),
    ),
    "EMLC": Instrument(
        ticker="EMLC",
        name="VanEck EM Local Currency Bond ETF",
        sleeve=SLEEVE_EM_BONDS_LOCAL,
        aum_bn_usd=2.8,   # largest US-listed local-currency EM bond ETF
        listing="NYSE",
        ccy="USD",
        vehicle_type="us_etf",
        domicile="US",
        accumulating=False,
        is_reporting_fund=True,  # assumed per user direction
        wrapper_eligible=_ucits("GIA"),
    ),
    # ── Nuclear energy (thematic, GIA only via US ETFs) ─────────────────
    # Structural uranium supply shortage post-Fukushima decade of underinvestment.
    # Antifragile to energy crises; low PEG relative to earnings recovery trajectory.
    # Apply MA200 filter strictly before adding.
    "NLR": Instrument(
        ticker="NLR",
        name="VanEck Uranium+Nuclear Energy ETF",
        sleeve=SLEEVE_NUCLEAR,
        aum_bn_usd=1.5,
        listing="NYSE",
        ccy="USD",
        vehicle_type="us_etf",
        domicile="US",
        accumulating=False,
        is_reporting_fund=True,  # assumed — verify against HMRC list
        wrapper_eligible=_ucits("GIA"),
    ),
    "URA": Instrument(
        ticker="URA",
        name="Global X Uranium ETF",
        sleeve=SLEEVE_NUCLEAR,
        aum_bn_usd=2.5,
        listing="NYSE",
        ccy="USD",
        vehicle_type="us_etf",
        domicile="US",
        accumulating=False,
        is_reporting_fund=True,  # assumed — verify against HMRC list
        wrapper_eligible=_ucits("GIA"),
    ),
    "URNM": Instrument(
        ticker="URNM",
        name="Sprott Uranium Miners ETF",
        sleeve=SLEEVE_NUCLEAR,
        aum_bn_usd=1.2,
        listing="NYSE",
        ccy="USD",
        vehicle_type="us_etf",
        domicile="US",
        accumulating=False,
        is_reporting_fund=True,  # assumed — verify against HMRC list
        wrapper_eligible=_ucits("GIA"),
    ),
    # ── Clean / renewable energy ─────────────────────────────────────────
    # 2022-2024 rate-driven selloff → PE vs 10y history near multi-decade lows.
    # MA200 filter critical: only add when price/MA200 > 0.85 (not still crashing).
    "ICLN": Instrument(
        ticker="ICLN",
        name="iShares Global Clean Energy ETF",
        sleeve=SLEEVE_CLEAN_ENERGY,
        aum_bn_usd=2.0,
        listing="NASDAQ",
        ccy="USD",
        vehicle_type="us_etf",
        domicile="US",
        accumulating=False,
        is_reporting_fund=True,  # assumed — verify against HMRC list
        wrapper_eligible=_ucits("GIA"),
    ),
    "INRG": Instrument(
        ticker="INRG",
        name="iShares Global Clean Energy UCITS ETF",
        sleeve=SLEEVE_CLEAN_ENERGY,
        aum_bn_usd=2.0,  # GBP UCITS equivalent of ICLN
        listing="LSE_GBP",
        ccy="GBP",
        vehicle_type="ucits_etf",
        domicile="IE",
        accumulating=False,
        is_reporting_fund=True,
        wrapper_eligible=_ucits("SIPP", "ISA", "GIA"),
    ),
    # ── Low-PEG / historically cheap sector ETFs (GIA only via US ETFs) ──
    # Use as optional tilts when sector PE z-score < -1 vs own 10y history
    # AND price/MA200 filter passes. Max combined sector allocation: 5% of GIA.
    "XLV": Instrument(
        ticker="XLV",
        name="Health Care Select Sector SPDR Fund",
        sleeve=SLEEVE_HEALTHCARE,
        aum_bn_usd=40.0,
        listing="NYSE",
        ccy="USD",
        vehicle_type="us_etf",
        domicile="US",
        accumulating=False,
        is_reporting_fund=True,  # assumed — verify against HMRC list
        wrapper_eligible=_ucits("GIA"),
    ),
    "XLI": Instrument(
        ticker="XLI",
        name="Industrial Select Sector SPDR Fund",
        sleeve=SLEEVE_INDUSTRIALS,
        aum_bn_usd=20.0,
        listing="NYSE",
        ccy="USD",
        vehicle_type="us_etf",
        domicile="US",
        accumulating=False,
        is_reporting_fund=True,  # assumed — verify against HMRC list
        wrapper_eligible=_ucits("GIA"),
    ),
    "XLB": Instrument(
        ticker="XLB",
        name="Materials Select Sector SPDR Fund",
        sleeve=SLEEVE_MATERIALS,
        aum_bn_usd=7.0,
        listing="NYSE",
        ccy="USD",
        vehicle_type="us_etf",
        domicile="US",
        accumulating=False,
        is_reporting_fund=True,  # assumed — verify against HMRC list
        wrapper_eligible=_ucits("GIA"),
    ),
    "XLF": Instrument(
        ticker="XLF",
        name="Financial Select Sector SPDR Fund",
        sleeve=SLEEVE_FINANCIALS,
        aum_bn_usd=55.0,
        listing="NYSE",
        ccy="USD",
        vehicle_type="us_etf",
        domicile="US",
        accumulating=False,
        is_reporting_fund=True,  # assumed — verify against HMRC list
        wrapper_eligible=_ucits("GIA"),
    ),
    # Platform mutual funds for ISA/SIPP alternatives.
    # These need platform/NAV data ingestion rather than the ETF market-data path.
    "PIMGLRR": Instrument(
        ticker="PIMGLRR",
        name="PIMCO GIS Global Low Duration Real Return Fund",
        sleeve=SLEEVE_LINKERS_GLOBAL,
        aum_bn_usd=2.0,
        listing="PLATFORM",
        ccy="GBP",
        vehicle_type="mutual_fund",
        domicile="IE",
        accumulating=True,
        is_reporting_fund=True,
        wrapper_eligible={"SIPP": True, "ISA": True, "GIA": False},
    ),
    "LGGINF": Instrument(
        ticker="LGGINF",
        name="L&G Global Infrastructure Index Fund",
        sleeve=SLEEVE_INFRA,
        aum_bn_usd=1.0,
        listing="PLATFORM",
        ccy="GBP",
        vehicle_type="mutual_fund",
        domicile="UK",
        accumulating=True,
        is_reporting_fund=True,
        wrapper_eligible={"SIPP": True, "ISA": True, "GIA": False},
    ),
    "FOSRRE": Instrument(
        ticker="FOSRRE",
        name="FP Foresight Sustainable Real Estate Fund",
        sleeve=SLEEVE_REITS,
        aum_bn_usd=0.5,
        listing="PLATFORM",
        ccy="GBP",
        vehicle_type="mutual_fund",
        domicile="UK",
        accumulating=True,
        is_reporting_fund=True,
        wrapper_eligible={"SIPP": True, "ISA": True, "GIA": False},
    ),
    "LGGHLTH": Instrument(
        ticker="LGGHLTH",
        name="L&G Global Health and Pharmaceuticals Index Fund",
        sleeve=SLEEVE_HEALTHCARE,
        aum_bn_usd=1.0,
        listing="PLATFORM",
        ccy="GBP",
        vehicle_type="mutual_fund",
        domicile="UK",
        accumulating=True,
        is_reporting_fund=True,
        wrapper_eligible={"SIPP": True, "ISA": True, "GIA": False},
    ),
}

INSTRUMENTS.update(
    {
        "SOXX": _us_etf("SOXX", "iShares Semiconductor ETF", SLEEVE_SEMICONDUCTORS, 12.0, "NASDAQ"),
        "EWZ": _us_etf("EWZ", "iShares MSCI Brazil ETF", SLEEVE_BRAZIL, 6.0),
        "XME": _us_etf("XME", "SPDR S&P Metals & Mining ETF", SLEEVE_MATERIALS, 2.0),
        "ILF": _us_etf("ILF", "iShares Latin America 40 ETF", SLEEVE_LATAM, 1.5),
        "EWT": _us_etf("EWT", "iShares MSCI Taiwan ETF", SLEEVE_TAIWAN, 5.0),
        "XLB": _us_etf("XLB", "Materials Select Sector SPDR Fund", SLEEVE_MATERIALS, 7.0),
        "AGIX": _us_etf("AGIX", "KraneShares Artificial Intelligence & Technology ETF", SLEEVE_AI, 0.3, "NASDAQ"),
        "INDA": _us_etf("INDA", "iShares MSCI India ETF", SLEEVE_INDIA, 8.0),
        "EEM": _us_etf("EEM", "iShares MSCI Emerging Markets ETF", SLEEVE_EM, 25.0),
        "FNDE": _us_etf("FNDE", "Schwab Fundamental Emerging Markets Large Company ETF", SLEEVE_EM_VALUE, 7.0),
        "EWM": _us_etf("EWM", "iShares MSCI Malaysia ETF", SLEEVE_MALAYSIA, 0.4),
        "DVYE": _us_etf("DVYE", "iShares Emerging Markets Dividend ETF", SLEEVE_EM_DIVIDEND, 0.8),
        "EFG": _us_etf("EFG", "iShares MSCI EAFE Growth ETF", SLEEVE_DEV_GROWTH, 12.0),
        "XOP": _us_etf("XOP", "SPDR S&P Oil & Gas Exploration & Production ETF", SLEEVE_ENERGY, 3.0),
        "VGK": _us_etf("VGK", "Vanguard FTSE Europe ETF", SLEEVE_EUROPE, 20.0),
        "WTAI": _us_etf("WTAI", "WisdomTree Artificial Intelligence and Innovation Fund", SLEEVE_AI, 0.2, "NASDAQ"),
        "EFV": _us_etf("EFV", "iShares MSCI EAFE Value ETF", SLEEVE_DEV_VALUE, 18.0),
        "VEA": _us_etf("VEA", "Vanguard FTSE Developed Markets ETF", SLEEVE_GLOBAL_EX_US, 130.0),
        "EEMV": _us_etf("EEMV", "iShares MSCI Emerging Markets Min Vol Factor ETF", SLEEVE_EM_MIN_VOL, 4.0),
        "QQQ": _us_etf("QQQ", "Invesco QQQ Trust", SLEEVE_MARKET_CAP, 300.0, "NASDAQ"),
        "IXUS": _us_etf("IXUS", "iShares Core MSCI Total International Stock ETF", SLEEVE_GLOBAL_EX_US, 45.0),
        "XLY": _us_etf("XLY", "Consumer Discretionary Select Sector SPDR Fund", SLEEVE_CONSUMER_DISCRETIONARY, 20.0),
        "KWEB": _us_etf("KWEB", "KraneShares CSI China Internet ETF", SLEEVE_CHINA_TECH, 5.0, "NYSE"),
        "FXI": _us_etf("FXI", "iShares China Large-Cap ETF", SLEEVE_CHINA, 7.0),
        "SPY": _us_etf("SPY", "SPDR S&P 500 ETF Trust", SLEEVE_MARKET_CAP, 550.0),
        "EWQ": _us_etf("EWQ", "iShares MSCI France ETF", SLEEVE_FRANCE, 0.5),
        "VPL": _us_etf("VPL", "Vanguard FTSE Pacific ETF", SLEEVE_PACIFIC, 10.0),
        "EWJ": _us_etf("EWJ", "iShares MSCI Japan ETF", SLEEVE_JAPAN, 13.0),
        "XLU": _us_etf("XLU", "Utilities Select Sector SPDR Fund", SLEEVE_UTILITIES, 18.0),
        "MDY": _us_etf("MDY", "SPDR S&P MidCap 400 ETF Trust", SLEEVE_US_MID_CAP, 20.0),
        "IWM": _us_etf("IWM", "iShares Russell 2000 ETF", SLEEVE_US_SMALL_CAP, 60.0),
        "XHB": _us_etf("XHB", "SPDR S&P Homebuilders ETF", SLEEVE_HOMEBUILDERS, 2.0),
        "IJR": _us_etf("IJR", "iShares Core S&P Small-Cap ETF", SLEEVE_US_SMALL_CAP, 80.0),
        "EWY": _us_etf("EWY", "iShares MSCI South Korea ETF", SLEEVE_KOREA, 4.0),
        "RSP": _us_etf("RSP", "Invesco S&P 500 Equal Weight ETF", SLEEVE_US_EQUAL_WEIGHT, 60.0),
        "IYT": _us_etf("IYT", "iShares Transportation Average ETF", SLEEVE_TRANSPORT, 1.0),
        "USMV": _us_etf("USMV", "iShares MSCI USA Min Vol Factor ETF", SLEEVE_US_LOW_VOL, 25.0),
        "XLP": _us_etf("XLP", "Consumer Staples Select Sector SPDR Fund", SLEEVE_CONSUMER_STAPLES, 18.0),
        "XBI": _us_etf("XBI", "SPDR S&P Biotech ETF", SLEEVE_BIOTECH, 7.0),
        "IGV": _us_etf("IGV", "iShares Expanded Tech-Software Sector ETF", SLEEVE_SOFTWARE, 8.0),
        "CIBR": _us_etf("CIBR", "First Trust NASDAQ Cybersecurity ETF", SLEEVE_CYBERSECURITY, 7.0, "NASDAQ"),
    }
)

_RESOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))
_STOCK_UNIVERSE_FILES = [
    os.path.join(_RESOURCE_DIR, "stocks_info.csv"),
    os.path.join(_RESOURCE_DIR, "ftse100_stocks.csv"),
    os.path.join(_RESOURCE_DIR, "europe_large_cap_stocks.csv"),
    os.path.join(_RESOURCE_DIR, "asia_adr_stocks.csv"),
]

for _path in _STOCK_UNIVERSE_FILES:
    INSTRUMENTS.update(_load_stock_universe_from_csv(_path))

# ── iShares UK product codes for P/E retrieval ────────────────────────
# URL template: https://www.ishares.com/uk/individual/en/products/{id}/{slug}/1478372549651.ajax
#               ?tab=keyFacts&fileType=json
# Verify at https://www.ishares.com/uk/individual/en/products/etf-product-list
# before assuming these are stable — iShares occasionally renumbers.
ISHARES_PRODUCTS: dict[str, tuple[str, str]] = {
    #  ticker     (product_id,  url-slug)
    "EIMI":  ("264659", "ISHARES-CORE-MSCI-EM-IMI-UCITS-ETF"),
    "CSPX":  ("253743", "ISHARES-CORE-SP500-UCITS-ETF"),
    "IWDA":  ("251850", "ISHARES-CORE-MSCI-WORLD-UCITS-ETF"),
    "IWQU":  ("264478", "ISHARES-EDGE-MSCI-WORLD-QUALITY-FACTOR-UCITS-ETF"),
    "IWVL":  ("264476", "ISHARES-EDGE-MSCI-WORLD-VALUE-FACTOR-UCITS-ETF"),
    "MVOL":  ("264477", "ISHARES-EDGE-MSCI-WORLD-MINIMUM-VOLATILITY-UCITS-ETF"),
    "IJPA":  ("251841", "ISHARES-MSCI-JAPAN-UCITS-ETF"),
    "ISF":   ("251795", "ISHARES-CORE-FTSE-100-UCITS-ETF"),
    "SEML":  ("264656", "ISHARES-JP-MORGAN-EM-LOCAL-GOVERNMENT-BOND-UCITS-ETF"),
    "INXG":  ("251839", "ISHARES-INDEX-LINKED-GILTS-UCITS-ETF"),
    "SGLN":  ("251902", "ISHARES-PHYSICAL-GOLD-ETC"),
}


def for_sleeve(sleeve: str, wrapper: str) -> list[Instrument]:
    """Return instruments matching a sleeve that are eligible for a wrapper."""
    return [
        ins
        for ins in INSTRUMENTS.values()
        if ins.sleeve == sleeve and _eligible_in_wrapper(ins, wrapper)
    ]


def lookup(ticker: str) -> Instrument | None:
    return INSTRUMENTS.get(ticker.upper())


def wrapper_rules_summary(wrapper: str) -> str:
    if wrapper == "SIPP":
        return "UCITS ETFs, ETCs, platform mutual funds, and direct stocks"
    if wrapper == "ISA":
        return "UCITS ETFs, ETCs, and platform mutual funds only"
    if wrapper == "GIA":
        return "HMRC reporting funds and direct stocks"
    return "Unknown wrapper"


def get_yfinance_ticker_map() -> dict[str, str]:
    """Map internal tickers to yfinance symbols for all price-trackable instruments."""
    mapping: dict[str, str] = {}
    for ticker, ins in INSTRUMENTS.items():
        if ins.listing == "PLATFORM":
            continue
        if ins.yfinance_symbol:
            mapping[ticker] = ins.yfinance_symbol
            continue
        if ins.listing in {"NYSE", "NASDAQ"}:
            mapping[ticker] = ticker
        elif ins.listing == "US":
            mapping[ticker] = ticker
        elif ins.listing in {"LSE", "LSE_GBP", "LSE_USD"}:
            mapping[ticker] = f"{ticker}.L"
    return mapping


def get_yfinance_to_internal_map() -> dict[str, str]:
    """Reverse map from yfinance symbol back to internal ticker."""
    return {yf_symbol.upper(): ticker for ticker, yf_symbol in get_yfinance_ticker_map().items()}


STOCK_LOOKTHROUGH_ALIASES: dict[str, str] = {
    "2330.TW": "TSM",
    "9988.HK": "BABA",
    "9618.HK": "JD",
    "9888.HK": "BIDU",
    "9999.HK": "NTES",
    "0700.HK": "TCEHY",
    "HDFCBANK.NS": "HDB",
    "ICICIBANK.NS": "IBN",
    "INFY.NS": "INFY",
}


# ── Thematic extras: optional GIA tilts ──────────────────────────────
# These are NOT in the baseline bucket targets (buckets.py).
# They are funded by trimming baseline GIA sleeves when activated.
# Activation condition: price/MA200 in (0.85, 1.30) AND sector PE z-score < -1
# (same "not crashing, not extended" filter as the regional equity engine).
# Max per-theme weight caps sum to 20% of GIA theoretically, but in practice
# 2-3 themes activate at once (the MA200 filter prevents buying crashing sectors),
# so typical live thematic exposure is 8-12% of GIA funded from other baseline sleeves.
THEMATIC_EXTRAS: dict[str, dict] = {
    SLEEVE_NUCLEAR: {
        "preferred_ticker": "NLR",
        "max_weight_gia": 0.05,
        "rationale": (
            "Uranium supply shock post-Fukushima; low PEG vs earnings recovery; "
            "antifragile to energy crises and geopolitical risk."
        ),
        "activation": "MA200 ratio in (0.85, 1.30); uranium spot > long-run supply cost",
        "alternatives": ["URA", "URNM"],
    },
    SLEEVE_CLEAN_ENERGY: {
        "preferred_ticker": "ICLN",
        "max_weight_gia": 0.04,
        "rationale": (
            "2022-2024 rate selloff reset PE to multi-decade lows vs own history. "
            "Structural IRA/EU policy tailwind intact. Rate-sensitive (long duration "
            "cashflows) — monitor US 10y trajectory."
        ),
        "activation": "MA200 ratio > 0.85 (not still crashing); US 10y < 4.5%",
        "alternatives": ["INRG"],
    },
    SLEEVE_HEALTHCARE: {
        "preferred_ticker": "XLV",
        "max_weight_gia": 0.03,
        "rationale": (
            "Aging demographics + GLP-1 pipeline. PE ~16x with EPS growth 8-10%/yr "
            "gives PEG < 1. Defensive factor in drawdowns."
        ),
        "activation": "Sector PE z-score < -0.5 vs own 10y history; MA200 filter passes",
        "alternatives": [],
    },
    SLEEVE_INDUSTRIALS: {
        "preferred_ticker": "XLI",
        "max_weight_gia": 0.03,
        "rationale": (
            "Reshoring/defence capex supercycle. PE ~18x with 12%+ EPS growth "
            "gives PEG < 1. Cyclical — apply MA200 filter carefully."
        ),
        "activation": "Sector PE z-score < -0.5 vs own 10y history; MA200 filter passes",
        "alternatives": [],
    },
    SLEEVE_MATERIALS: {
        "preferred_ticker": "XLB",
        "max_weight_gia": 0.03,
        "rationale": (
            "Resource nationalism + energy transition metals (copper, lithium). "
            "PE ~13-15x vs 10y avg 18x; z-score ~-1.2. Commodity cycle upside."
        ),
        "activation": "Sector PE z-score < -1 vs own 10y history; MA200 filter passes",
        "alternatives": [],
    },
    SLEEVE_FINANCIALS: {
        "preferred_ticker": "XLF",
        "max_weight_gia": 0.02,
        "rationale": (
            "Historically cheap vs own PE history (z ~-0.8). High rates boost NIM. "
            "Lower conviction — keep allocation small."
        ),
        "activation": "Sector PE z-score < -1 vs own 10y history; MA200 filter passes",
        "alternatives": [],
    },
}
