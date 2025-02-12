import datetime

import duckdb
import pandas as pd

from app.data import create_query
from duckdb_importer import duckdb_file


def get_df(tickers=None, start_date=None, end_date=None, fill_na=True):
    query = create_query(
        table="total_return",
        cols=["ticker_full", "date", "price"],
        instruments=tickers,
        start_date=start_date,
        end_date=end_date,
    )
    with duckdb.connect(database=duckdb_file, read_only=False) as conn:
        df = conn.execute(query=query).df()
        df["date"] = df["date"].dt.date
        X = (
            df.groupby(["date", "ticker_full"])["price"]
            .mean()
            .reset_index()
            .set_index(["date", "ticker_full"])["price"]
            .unstack()
        )
        if fill_na:
            X = X.ffill()
            X = X.dropna()
        return X


today = datetime.date.today()
tickers = {
    "ILF": 12,
    "ISFD": 5,
    "FXI": 5,
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
X: pd.DataFrame = get_df(tickers=tickers.keys())

############################################
############################################
############################################
############################################
############################################
"""
==================================
Risk Parity - Covariance shrinkage
==================================

This tutorial shows how to incorporate covariance shrinkage in the
:class:`~skfolio.optimization.RiskBudgeting` optimization.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the S&P 500 Index composition starting from 1990-01-02 up to 2022-12-28:

from plotly.io import show
from sklearn.model_selection import train_test_split

from skfolio import Population, RiskMeasure
from skfolio.moments import ShrunkCovariance
from skfolio.optimization import RiskBudgeting
from skfolio.prior import EmpiricalPrior

X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

# %%
# Model
# =====
# We create a risk parity model by using :class:`~skfolio.moments.ShrunkCovariance` as
# the covariance estimator then fit it on the training set:
model = RiskBudgeting(
    risk_measure=RiskMeasure.VARIANCE,
    prior_estimator=EmpiricalPrior(
        covariance_estimator=ShrunkCovariance(shrinkage=0.9)
    ),
    portfolio_params=dict(name="Risk Parity - Covariance Shrinkage"),
)
model.fit(X_train)
model.weights_

# %%
# To compare this model, we use a basic risk parity without covariance shrinkage:
bench = RiskBudgeting(
    risk_measure=RiskMeasure.VARIANCE,
    portfolio_params=dict(name="Risk Parity - Basic"),
)
bench.fit(X_train)
bench.weights_

# %%
# Prediction
# ==========
# We predict the model and the benchmark on the test set:
ptf_model_test = model.predict(X_test)
ptf_bench_test = bench.predict(X_test)

# %%
# Analysis
# ========
# For improved analysis, it's possible to load both predicted portfolios into a
# :class:`~skfolio.population.Population`:
population = Population([ptf_model_test, ptf_bench_test])

# %%
# Let's plot each portfolio cumulative returns:
fig = population.plot_cumulative_returns()
show(fig)

# %%
# |
#
# Finally, we print a full summary of both strategies evaluated on the test set:
population.summary()
