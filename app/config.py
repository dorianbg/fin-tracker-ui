charts_width: int = 800
table_height: int = 600
time_strings = ["1W", "1M", "3M", "6M", "9M", "1Y", "18M", "2Y", "3Y", "5Y", "10Y"]

# Risk & returns
RISK_FREE_RATE = 0.05

# Correlation matrix
CORRELATION_LOOKBACK_DEFAULT = 60
MAX_CORRELATION_ASSETS = 15

# Benchmark tickers (display_name -> ticker)
BENCHMARKS = {
    "FTSE All-World (VWRP)": "VWRP",
    "S&P 500 (CSP1)": "CSP1",
    "MSCI Europe (IMEA)": "IMEA",
    "Emerging Markets (EEM)": "EEM",
}
DEFAULT_BENCHMARK = "VWRP"

# Portfolio manager
BROKER_OPTIONS = ["Fidelity", "HL", "IBKR", "Other"]

# Fund type filter (shared across dashboard pages)
FUND_TYPE_OPTIONS = [
    "eq",
    "stock",
    "eq-reit",
    "commod",
    "bonds",
    "bonds-em",
    "bonds-corp",
    "bonds-il",
    "bonds-cash",
]

FUND_TYPE_DISPLAY_NAMES = {
    "eq": "Equity (ETFs)",
    "stock": "Stocks",
    "eq-reit": "Real Estate",
    "commod": "Commodities",
    "bonds": "Bonds",
    "bonds-em": "EM Bonds",
    "bonds-corp": "Corp Bonds",
    "bonds-il": "IL Bonds",
    "bonds-cash": "Cash",
}
