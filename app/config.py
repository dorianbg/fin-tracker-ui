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
    "S&P 500 (CSP1)": "CSP1",
    "MSCI USA (CUSS)": "CUSS",
    "MSCI Europe (IMEA)": "IMEA",
    "Emerging Markets (EEM)": "EEM",
}
DEFAULT_BENCHMARK = "CSP1"

# Portfolio manager
BROKER_OPTIONS = ["Fidelity", "HL", "IBKR", "Other"]
