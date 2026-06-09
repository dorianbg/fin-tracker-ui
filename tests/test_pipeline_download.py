import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from pipeline.executor import (
    _download_batch,
    _normalize_yfinance_df,
    filter_invalid_price_rows,
    repair_invalid_ohlc,
)


def test_download_batch_returns_dataframe_per_ticker():
    mock_df = pd.DataFrame(
        {"Close": [100.0, 101.0]},
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"]),
    )
    with patch("pipeline.executor.yf.download") as mock_dl:
        mock_dl.return_value = pd.concat(
            {("Close", "AAPL"): mock_df["Close"], ("Close", "MSFT"): mock_df["Close"]},
            axis=1,
        )
        mock_dl.return_value.columns = pd.MultiIndex.from_tuples(
            [("Close", "AAPL"), ("Close", "MSFT")]
        )
        result = _download_batch(["AAPL", "MSFT"], "2024-01-01", "2024-01-04")

    assert "AAPL" in result
    assert "MSFT" in result
    assert len(result["AAPL"]) == 2
    assert result["AAPL"]["close"].iloc[-1] == 101.0


def test_download_batch_handles_missing_ticker():
    mock_df = pd.DataFrame(
        {"Close": [100.0]},
        index=pd.DatetimeIndex(["2024-01-02"]),
    )
    with patch("pipeline.executor.yf.download") as mock_dl:
        mock_dl.return_value = pd.concat({("Close", "AAPL"): mock_df["Close"]}, axis=1)
        mock_dl.return_value.columns = pd.MultiIndex.from_tuples([("Close", "AAPL")])
        result = _download_batch(["AAPL", "BADTICKER"], "2024-01-01", "2024-01-04")

    assert "AAPL" in result
    assert "BADTICKER" in result
    assert len(result["AAPL"]) == 1
    assert result["BADTICKER"].empty


def test_normalize_yfinance_df_fills_missing_dividend_column():
    df = pd.DataFrame(
        {"Close": [100.0]},
        index=pd.DatetimeIndex(["2024-01-02"]),
    )
    result = _normalize_yfinance_df(df, "AAPL")
    assert "dividends" in result.columns
    assert (result["dividends"] == 0.0).all()
    assert "stock_splits" in result.columns


def test_filter_invalid_price_rows_drops_yfinance_placeholders():
    df = pd.DataFrame(
        {
            "close": [100.0, np.nan, 0.0],
            "volume": [100, 200, 300],
        }
    )

    result = filter_invalid_price_rows(df)

    assert len(result) == 1
    assert result["close"].iloc[0] == 100.0
