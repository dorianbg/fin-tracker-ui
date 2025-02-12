from __future__ import division

import datetime

import numpy as np

import duckdb_importer as di
from app.data import get_data


def returns_from_prices(prices, log_returns=False):
    """
    Calculate the returns given prices.

    :param prices: adjusted (daily) closing prices of the asset, each row is a
                   date and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param log_returns: whether to compute using log returns
    :type log_returns: bool, defaults to False
    :return: (daily) returns
    :rtype: pd.DataFrame
    """
    if log_returns:
        returns = np.log(1 + prices.pct_change()).dropna(how="all")
    else:
        returns = prices.pct_change().dropna(how="all")
    return returns


tickers = {
    "ILF": 12,
    "ISFD": 5,
    "FXI": 5,
    "ISPE": 5,
    "ICLN": 5,
    "VEGI": 5,
    "INFR": 5,
    "VMIG": 5,
    "EWJ": 5,
    "EMB": 7,
    "LEMB": 7,
    "GLTL": 5,
    "ITPG": 3,
    "ERNS": 10,
    "IB01": 10,
}
target_volatility = 0.12

# Load historical closing prices for the ETFs
today = datetime.date.today()
prices = get_data(
    table=di.px_tbl,
    start_date=today - datetime.timedelta(days=720),
    end_date=today,
    instruments=list(tickers.keys()),
)
prices["date"] = prices["date"].dt.date
print(len(prices))
prices = prices.drop_duplicates(["date", "ticker"])
print(len(prices))
prices = prices.pivot(index="date", columns="ticker", values="price")
returns = returns_from_prices(prices)
returns = returns.dropna()

Sigma = prices.cov()
risk_budget = []
for p in prices.columns:
    if p in tickers:
        risk_budget.append(tickers[p] / 100)


# doens't work with risk budgeting but just does general optimisation
# doc: https://github.com/robertmartin8/PyPortfolioOpt/blob/master/cookbook/2-Mean-Variance-Optimisation.ipynb
def pyportfoliomode():
    from pypfopt import risk_models
    from pypfopt import expected_returns
    from pypfopt import EfficientFrontier

    semicov = risk_models.semicovariance(prices, benchmark=0)

    # Create an EfficientFrontier object
    mu = expected_returns.ema_historical_return(prices, span=750)

    S = risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf()

    es = EfficientFrontier(
        expected_returns=mu, cov_matrix=S, weight_bounds=(0, 0.25), verbose=False
    )
    # es.max_sharpe(target_volatility=10)
    for ticker, weight in tickers.items():
        tckr_index = es.tickers.index(ticker)
        es.add_constraint(lambda w: (w[tckr_index] >= weight / (100 * 5)))
        es.add_constraint(lambda w: (w[tckr_index] <= weight / (100 / 5)))

    # es.efficient_risk(target_volatility=12)
    # es.min_volatility()
    es.max_sharpe(risk_free_rate=0.04)
    weights = es.clean_weights()
    print(f"Custom weights {weights}")
    res = es.portfolio_performance(verbose=True, risk_free_rate=0.05)
    print(f"Perf res {res}")

    print(f"Original weights {tickers}")


# works well, https://github.com/convexfi/riskparity.py
def riskparityportfolio_m():
    import riskparityportfolio as rp
    import numpy as np

    print(f"Weights add up to {sum(tickers.values())}")
    my_portfolio = rp.RiskParityPortfolio(covariance=Sigma, budget=risk_budget)
    # my_portfolio.add_variance(lmd=5)
    my_portfolio.design()
    print(f"Volatility: {my_portfolio.volatility}")
    weights: np.ndarray = my_portfolio.weights
    for i, w, rc in zip(
        Sigma.columns, weights.tolist(), my_portfolio.risk_contributions
    ):
        print(f"Weight for {i}: {round(w * 100, 2)}%. RC: {round(rc * 100, 2)}%")


# doesn't work - https://nbviewer.org/github/dcajasn/Riskfolio-Lib/blob/master/examples/Tutorial%2033%20-%20Risk%20Parity%20with%20Constraints%20using%20the%20Risk%20Budgeting%20Approach.ipynb
def riskfolio():
    import riskfolio as rp

    # Building the portfolio object
    port = rp.Portfolio(returns=prices)

    # Calculating optimal portfolio

    # Select method and estimate input parameters:

    method_mu = "hist"  # Method to estimate expected returns based on historical data.
    method_cov = (
        "hist"  # Method to estimate covariance matrix based on historical data.
    )

    port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

    # Estimate optimal portfolio:

    model = "Classic"  # Could be Classic (historical) or FM (Factor Model)
    rm = "MV"  # Risk measure used, this time will be variance
    hist = True  # Use historical scenarios for risk measures that depend on scenarios
    rf = 0.04  # Risk free rate
    b = risk_budget  # Risk contribution constraints vector

    port.lowerret = 0.00056488 * 1.5

    w_rp = port.rp_optimization(model=model, rm=rm, rf=rf, b=b, hist=hist)

    print(w_rp)


# works well, https://github.com/jcrichard/pyrb
def pyrb():
    from pyrb import RiskBudgeting
    import pandas as pd
    import numpy as np

    covariance_matrix = Sigma * 260

    RB = RiskBudgeting(cov=covariance_matrix, budgets=risk_budget)
    RB.solve()

    optimal_weights = RB.x
    risk_contributions = RB.get_risk_contributions(scale=False)
    risk_contributions_scaled = RB.get_risk_contributions()
    allocation = pd.DataFrame(
        np.concatenate(
            [[optimal_weights, risk_contributions, risk_contributions_scaled]]
        ).T,
        index=covariance_matrix.index,
        columns=["optinal weigths", "risk contribution", "risk contribution(scaled)"],
    )
    print(allocation)
    print(f"Volatility: {RB.get_variance()}")
    assert np.round(
        np.dot(np.dot(RB.x, covariance_matrix), RB.x) ** 0.5, 10
    ) == np.round(allocation["risk contribution"].sum(), 10)


if __name__ == "__main__":
    # pyportfoliomode()
    # riskparityportfolio_m()
    pyrb()
