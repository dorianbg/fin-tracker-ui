"""Wrapper-aware instrument universe.

Maps logical sleeves to concrete ETF tickers with metadata on which tax
wrappers can hold each, reporting fund status, and spread/AUM notes.

Three wrappers:
  - SIPP (Freetrade/Fidelity): GBP UCITS only, AUM > $2B, spread < 5bps
  - ISA  (Freetrade/Fidelity): same as SIPP
  - GIA  (IBKR): GBP UCITS + large US ETFs (AUM > $5B, spread < 2bps)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Instrument:
    ticker: str
    description: str
    sleeve: str
    ccy: str = "GBP"
    wrappers: frozenset[str] = field(default_factory=lambda: frozenset({"GIA"}))
    is_reporting_fund: bool = True
    aum_bn: float | None = None


_ALL = frozenset({"SIPP", "ISA", "GIA"})
_GIA = frozenset({"GIA"})

# ── SIPP / ISA universe (GBP UCITS only) ─────────────────────────────
SIPP_ISA_INSTRUMENTS: dict[str, Instrument] = {
    "IWQU": Instrument("IWQU", "iShares Edge MSCI World Quality", "equity", "GBP", _ALL, aum_bn=5.0),
    "MVOL": Instrument("MVOL", "iShares Edge MSCI World Min Vol", "equity", "GBP", _ALL, aum_bn=2.5),
    "IWVL": Instrument("IWVL", "iShares Edge MSCI World Value", "equity", "GBP", _ALL, aum_bn=2.5),
    "EIMI": Instrument("EIMI", "iShares Core MSCI EM IMI", "equity", "GBP", _ALL, aum_bn=20.0),
    "IJPA": Instrument("IJPA", "iShares MSCI Japan", "equity", "GBP", _ALL, aum_bn=5.0),
    "VEUR": Instrument("VEUR", "Vanguard FTSE Developed Europe ex-UK", "equity", "GBP", _ALL, aum_bn=5.0),
    "IWMO": Instrument("IWMO", "iShares Edge MSCI World Momentum", "equity", "GBP", _ALL, aum_bn=2.0),
    "NATO": Instrument("NATO", "HANetf Future of Defence", "equity", "GBP", _ALL, aum_bn=1.5),
    "SGLN": Instrument("SGLN", "iShares Physical Gold", "real_defensive", "GBP", _ALL, aum_bn=15.0),
    "INXG": Instrument("INXG", "iShares £ Index-Linked Gilts", "bonds", "GBP", _ALL, aum_bn=2.0),
    "ERNS": Instrument("ERNS", "iShares £ Ultrashort Bond", "cash", "GBP", _ALL, aum_bn=3.0),
    "CSH2": Instrument("CSH2", "Lyxor Smart Overnight", "cash", "GBP", _ALL, aum_bn=5.0),
    "INFR": Instrument("INFR", "iShares Global Infrastructure", "real_defensive", "GBP", _ALL, aum_bn=1.5),
    "VWRP": Instrument("VWRP", "Vanguard FTSE All-World", "equity", "GBP", _ALL, aum_bn=15.0),
}

# ── GIA universe (IBKR — US ETFs + USD UCITS) ────────────────────────
GIA_INSTRUMENTS: dict[str, Instrument] = {
    "IAU":  Instrument("IAU", "iShares Gold Trust", "real_defensive", "USD", _GIA, aum_bn=35.0),
    "GDX":  Instrument("GDX", "VanEck Gold Miners", "real_cyclical", "USD", _GIA, aum_bn=13.0),
    "IGF":  Instrument("IGF", "iShares Global Infrastructure", "real_defensive", "USD", _GIA, aum_bn=5.0),
    "VNQ":  Instrument("VNQ", "Vanguard Real Estate", "real_defensive", "USD", _GIA, aum_bn=30.0),
    "TIP":  Instrument("TIP", "iShares TIPS Bond", "bonds", "USD", _GIA, aum_bn=20.0),
    "TLT":  Instrument("TLT", "iShares 20+ Year Treasury", "bonds", "USD", _GIA, aum_bn=50.0),
    "EMB":  Instrument("EMB", "iShares JPM USD EM Bond", "bonds_tactical", "USD", _GIA, aum_bn=14.0),
    "SGOV": Instrument("SGOV", "iShares 0-3 Month Treasury", "cash", "USD", _GIA, aum_bn=30.0),
    "BIL":  Instrument("BIL", "SPDR Bloomberg 1-3 Month T-Bill", "cash", "USD", _GIA, aum_bn=35.0),
    "XLE":  Instrument("XLE", "Energy Select Sector SPDR", "equity", "USD", _GIA, aum_bn=35.0),
    "EEM":  Instrument("EEM", "iShares MSCI Emerging Markets", "equity", "USD", _GIA, aum_bn=20.0),
    "VGK":  Instrument("VGK", "Vanguard FTSE Europe", "equity", "USD", _GIA, aum_bn=15.0),
    "EWJ":  Instrument("EWJ", "iShares MSCI Japan", "equity", "USD", _GIA, aum_bn=10.0),
}

ALL_INSTRUMENTS: dict[str, Instrument] = {**SIPP_ISA_INSTRUMENTS, **GIA_INSTRUMENTS}


def instruments_for_wrapper(wrapper: str) -> dict[str, Instrument]:
    return {t: i for t, i in ALL_INSTRUMENTS.items() if wrapper in i.wrappers}


def non_reporting_fund_tickers() -> list[str]:
    return [t for t, i in ALL_INSTRUMENTS.items() if not i.is_reporting_fund]


def wrapper_violations(holdings_df) -> list[dict]:
    """Flag holdings placed in a wrapper where the instrument isn't eligible."""
    violations = []
    for _, row in holdings_df.iterrows():
        ticker = row.get("asset", "")
        acct = row.get("account_type", "")
        inst = ALL_INSTRUMENTS.get(ticker)
        if inst and acct and acct not in inst.wrappers:
            violations.append({
                "ticker": ticker,
                "account_type": acct,
                "allowed_wrappers": ", ".join(sorted(inst.wrappers)),
                "reason": f"{ticker} is not eligible for {acct}",
            })
    return violations
