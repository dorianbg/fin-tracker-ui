# Fin Tracker UI

Streamlit application utilising DuckDB for querying in-memory data (loaded from encrypted parquet files) alongside heavy caching due to the reactive architecture of Streamlit.

### Development

For local setup you should just `uv sync` the project.

Consider setting up `pyenv` and `uv` as per this [article](https://dorianbg.github.io/posts/python-project-setup-best-practice/).

After installing dependencies above, you could run locally with:  
```python -m streamlit run app/PerformanceTable.py``` .

With PyCharm (or IntelliJ Idea) you can also utilise the debugger.
Just make sure that inside your `Run configuration` module is set to `streamlit` and script parameters are set to `run app/PerformanceTable.py`.

### Data

Dataset is easily generated from [this repo](https://github.com/dorianbg/fin-tracker/).

The committed parquet files are encrypted, so you won't be able to re-use the full dataset locally.
Instead use the `*_unencrypted.parquet` files to get a high level sense of the data.   

Or consider simply running the script from above mentioned repo.

### Pages

| Page | Description |
|------|-------------|
| **Performance Table** | Main dashboard with colour-coded returns, Sharpe ratios, moving averages, and custom sorting |
| **Performance Chart** | Interactive timeseries charts for price performance and volatility |
| **Asset Correlation** | Correlation matrix heatmap for selected instruments |
| **Portfolio Manager** | Track holdings across brokers with asset allocation pie chart |
| **Sector Screener** | Identify oversold sectors and momentum recovery opportunities vs a benchmark |
| **Factor Dashboard** | Compare value, momentum, quality, size, and min-vol factor ETFs |
| **Cross-Asset Regime** | Equities vs bonds vs commodities regime detection (risk-on/off/goldilocks/stagflation) |
| **Pullback Scanner** | Find healthy pullbacks in strong uptrends — buy-the-dip candidates |

### Screenshots

General performance tab:
![Screenshot 2024-02-29 at 23.08.08.jpg](img%2FScreenshot%202024-02-29%20at%2023.08.08.jpg)

Performance plotting tab:
![Screenshot 2024-02-29 at 23.07.50.jpg](img%2FScreenshot%202024-02-29%20at%2023.07.50.jpg)