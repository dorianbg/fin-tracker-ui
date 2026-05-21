---
description: How to add a new scanner/screener page to fin-tracker-ui
---

# Adding a New Page

All scanner/screener pages follow the same pattern. Use this as a template.

## Step 1: Create the file

Create `app/pages/YourPage.py`. Streamlit auto-discovers pages in this directory.

## Step 2: Standard imports and data loading

```python
import streamlit as st
import pandas as pd
import plotly.express as px

import duckdb_importer as di
from data import get_conn

st.title("🔍 Your Page Title")

# Load latest performance data (one row per instrument)
@st.cache_data(ttl=300)
def load_all_latest() -> pd.DataFrame:
    cols = (
        di.perf_desc_cols_start
        + di.perf_vol_cols
        + di.perf_mavg_cols
        + di.perf_returns_cols
        + di.perf_desc_cols_end
        + di.perf_rownames_cols
    )
    query = f"""
        SELECT {",".join(cols)}
        FROM {di.perf_tbl}
        WHERE rown = 1
        ORDER BY description ASC
    """
    return get_conn().execute(query).df()

_all_data = load_all_latest()
if _all_data.empty:
    st.warning("No data loaded.")
    st.stop()
```

## Step 3: Sidebar controls

```python
st.sidebar.header("Settings")

fund_type_filter = st.sidebar.multiselect(
    "Fund types",
    options=["eq", "eq-reit", "commod", "bonds", "bonds-em", "bonds-corp", "bonds-il", "bonds-cash"],
    default=["eq"],
)

# Filter data
if fund_type_filter:
    pattern = "^(" + "|".join(fund_type_filter) + ")"
    df = _all_data[_all_data["fund_type"].str.match(pattern)].copy()
else:
    df = _all_data.copy()
```

## Step 4: Available columns

After loading with `rown = 1`, each row has:

**Identifiers**: `date`, `description`, `ticker`, `fund_type`

**Moving averages** (% above/below MA):
- `ma_21`, `ma_63`, `ma_126`, `ma_252` — positive = above MA

**Drawdowns**:
- `drawdown_52w` — % from 52-week high (always ≤ 0, 0 = at high)
- `drawdown_3y` — % from 3-year high

**Returns** (% change):
- `r_1d`, `r_1w`, `r_2w`, `r_1mo`, `r_3mo`, `r_6mo`, `r_1y`, `r_2y`, `r_3y`, `r_5y`

**Volatility**: `vol_1mo`, `vol_1y`

**Z-scores**: `z_1d`, `z_1w`, `z_2w`, `z_1mo`

## Step 5: Loading price history (for time series charts)

```python
@st.cache_data(ttl=300)
def load_price_history(tickers: list[str]) -> pd.DataFrame:
    tickers_str = "','".join(tickers)
    query = f"""
        SELECT ticker, date, price, description
        FROM {di.px_tbl}
        WHERE ticker IN ('{tickers_str}')
        ORDER BY date ASC
    """
    return get_conn().execute(query).df()
```

## Step 6: Common chart patterns

**Heatmap** (like ThematicDashboard):
```python
fig = px.imshow(
    data.values, x=col_labels, y=row_labels,
    color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
    text_auto=".1f", aspect="auto",
)
```

**Treemap** (like BreakoutScanner heatmap):
```python
fig = px.treemap(
    df, path=["category", "description"], values="size",
    color="return_val", color_continuous_scale="RdYlGn",
    color_continuous_midpoint=0,
)
```

**Scatter** (like PullbackScanner):
```python
fig = px.scatter(
    df, x="ma_252", y="ma_21", color="r_1w",
    color_continuous_scale="RdYlGn", text="ticker",
)
```

**Horizontal bar** (like SectorScreener):
```python
fig = px.bar(
    df, x="score", y="description", orientation="h",
    color="score", color_continuous_scale="YlGn",
)
fig.update_layout(yaxis=dict(autorange="reversed"))
```

## Step 7: Formatting tables

```python
st.dataframe(
    df[display_cols].style.format(
        subset=["ma_21", "ma_63", "r_1w", "r_1mo"],
        formatter="{:+.2f}%",
    ),
    hide_index=True,
    height=450,
)
```

## Conventions

- Always use `@st.cache_data(ttl=300)` for data loading functions
- Use `st.header()` and `st.markdown("---")` to separate sections
- Sort/filter in pandas (fast), not SQL — only one SQL query per page
- Use `RdYlGn` color scale with midpoint 0 for return-based charts
- Display `{:+.2f}%` format for percentage columns
