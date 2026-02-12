from __future__ import division

import datetime

import numpy as np
import riskfolio as rp

import duckdb_importer as di
from app.data import get_conn, create_query


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


# Load historical closing prices for the ETFs
today = datetime.date.today()
query = create_query(
    table=di.px_tbl,
    start_date=today - datetime.timedelta(days=720),
    end_date=today,
)
prices = get_conn().execute(query).df()
prices["date"] = prices["date"].dt.date
print(len(prices))

prices = prices.drop_duplicates(["date", "ticker"])
print(len(prices))
prices = prices.pivot(index="date", columns="ticker", values="price")
returns = returns_from_prices(prices)
returns = returns.dropna()

tickers = ["ILF", "FXI", "EWJ"]
# Estimate optimal portfolio:

model = "HERC2"  # Could be HRP or HERC or HERC2
codependence = "pearson"  # Correlation matrix used to group assets in clusters
rm = "MV"  # Risk measure used, this time will be variance
rf = 0.05  # Risk free rate
linkage = "ward"  # "single"  # Linkage method used to build clusters
max_k = 8  # Max number of clusters used in two difference gap statistic, only for HERC model
leaf_order = True  # Consider optimal order of leafs in dendrogram

# Plotting Assets Clusters

ax = rp.plot_dendrogram(
    returns=returns,
    codependence=codependence,
    linkage=linkage,
    k=None,
    max_k=max_k,
    leaf_order=True,
    ax=None,
)
fig = ax.figure  # Retrieve the figure associated with the axis
fig.show()  # Show the figure

# Building the portfolio object
port = rp.HCPortfolio(returns=returns, w_max=0.2, w_min=0)


w = port.optimization(
    model=model,
    codependence=codependence,
    rm=rm,
    rf=rf,
    linkage=linkage,
    max_k=max_k,
    leaf_order=leaf_order,
    obj="ERC",
)

ax2 = rp.plot_pie(
    w=w,
    title="HRP Naive Risk Parity",
    others=0.05,
    nrow=50,
    cmap="tab20",
    height=16,
    width=20,
    ax=None,
)

fig2 = ax2.figure  # Retrieve the figure associated with the axis
fig2.show()  # Show the figure
