# Real Asset Allocation: The World Has Changed - Deep Strategy Summary And Build Plan

Source: `RealAssetAllocation.pdf`, 3Fourteen Research, May 4, 2023.

This document summarizes the PDF and translates it into an implementation strategy we can align on before building anything.

## 1. Executive Summary

The report argues that the investment regime that supported the classic 60/40 stock/bond portfolio from 1998 through 2021 has ended. The core claim is that the negative stock/bond correlation of that period was not a permanent law of markets. It was a product of unusually powerful disinflationary forces: globalization, China joining the WTO, cheap manufactured goods, shale energy growth, low inflation expectations, quantitative easing, and declining term premia.

Since 2022, stocks and bonds have started moving together again. The report argues this is likely to persist because inflation risk is structurally higher, fiscal debt is much larger, geopolitical energy risk has increased, China’s demographic tailwind has reversed, shale supply is less elastic, and the energy transition requires large real-asset investment.

The portfolio implication is severe: if stocks and bonds become positively correlated, the traditional 60/40 portfolio loses its main defensive feature. It can still make money, but it should produce more upside and downside volatility, weaker drawdown protection, and lower Sharpe ratios.

The proposed solution is the Real Asset Allocation Model (RAAM):

- Broaden the asset universe beyond stocks and bonds.
- Include real assets, commodities, managed futures, and inflation-sensitive assets.
- Rank assets dynamically using a more nuanced trend-following framework.
- Reduce high-volatility concentration through hierarchical risk parity (HRP).
- Backtest and monitor against both a 60/40 benchmark and a static expanded-asset benchmark.

The report’s key point is not just that investors should own commodities or alternatives. The key point is that the next regime may require a dynamic multi-asset allocation system because the old stock/bond hedge may no longer work.

## 2. The Regime-Change Argument

### 2.1 Pre-1998 Market Structure

Before 1998, stocks and bonds often moved together. Bonds were not a reliable hedge on severe equity down days. The report shows that before 1998, bonds were down on 65 of the 100 worst stock-market days, meaning bonds rose on only 35 of those days.

This matters because many modern investors implicitly treat negative stock/bond correlation as normal. The report argues it was not normal over longer market history.

### 2.2 1998-2021: The Golden Era Of 60/40

From roughly 1998 through 2021, stocks and bonds had a negative correlation. During this period, bonds usually rallied when stocks sold off. The report shows bonds rose on 83 of the 100 worst stock-market days from 1998 to 2021.

This negative correlation made the 60/40 portfolio unusually efficient:

- Equities provided growth and long-run real returns.
- Bonds provided income, duration returns, and crisis protection.
- Falling yields boosted bond total returns.
- Negative stock/bond correlation reduced portfolio volatility.
- Portfolio drawdowns were cushioned during equity stress.

The report emphasizes that the magic of 60/40 was not simply owning 40% bonds. It was owning bonds in a regime where bonds were negatively correlated with equities.

### 2.3 Why Negative Stock/Bond Correlation Emerged

The report attributes the post-1998 regime to a series of disinflationary forces.

#### Asian Financial Crisis

The Asian Financial Crisis weakened regional currencies. This helped set up Asia as a major low-cost manufacturing base for the world.

#### China Entering The WTO

China entered the WTO around 2000 while its working-age population was booming. This created a major global labor supply shock and helped reduce manufactured-goods inflation.

#### Globalization And The Euro Era

Globalization increased trade integration and reinforced lower consumer-goods prices. The euro’s emergence also contributed to a more integrated global financial and trade environment.

#### Shale Energy Boom

After the Global Financial Crisis, the shale revolution added another disinflationary force. Investors funded shale production aggressively, often despite poor profitability, effectively subsidizing energy consumers through increased hydrocarbon supply.

#### Fed Policy And Term Premium Compression

Post-GFC quantitative easing reduced duration supply available to the market and helped suppress long-term term premia. The report also argues that negative stock/bond correlation itself created an equity-hedge bid for long bonds.

### 2.4 Why The Report Thinks The Regime Has Changed

The report argues that many of those disinflationary forces are reversing.

Key post-COVID shifts:

- The U.S. added about $8 trillion of federal debt.
- Structural deficits remain large.
- China’s working-age population has started shrinking.
- Russia’s invasion of Ukraine placed a large share of global energy supply at geopolitical risk.
- Many shale basins have passed peak production.
- Western governments are committed to a capital-intensive green energy transition.
- Inflation is likely to remain a bigger problem than it was from 1998 to 2021.
- The Fed has moved away from the zero-bound regime.
- Long bonds may face a rising or normalizing term premium.

The central portfolio implication: inflation risk becomes a dominant macro factor again. When inflation is the problem, both stocks and bonds can sell off together.

## 3. Evidence Presented In The Report

### 3.1 Stock/Bond Return Scatterplots

The report compares 90-day returns of the S&P 500 and long-term Treasuries.

From 1998 through 2021:

- The relationship was negative.
- Very few observations showed both stocks and bonds falling together.
- Bonds generally cushioned equity weakness.

Since 2022:

- Stocks and bonds moved together more often.
- The lower-left quadrant, where both stocks and bonds are down, became more populated.
- The relationship looked closer to the pre-1998 pattern.

### 3.2 Worst Equity Days

The report compares bond performance during the worst 100 equity days.

Before 1998:

- Bonds were down on 65 of the 100 worst equity days.

From 1998 through 2021:

- Bonds were down on only 17 of the 100 worst equity days.

During 2022:

- Bonds were up on only 4 of the 10 worst stock-market days, closer to the pre-1998 relationship.

### 3.3 Sharpe Ratio Sensitivity To Stock/Bond Correlation

The report shows that the Sharpe ratio of a 60/40 portfolio declines as stock/bond correlation rises, holding returns constant.

This is a key conceptual bridge:

- If stock and bond expected returns are unchanged but correlation rises, diversification benefits fall.
- Lower diversification means higher portfolio volatility.
- Higher volatility for the same returns means lower Sharpe ratio.

### 3.4 Term Premium Risk

The report argues that the secular decline in rates from the early 1980s may be ending. Long bonds are exposed to the possibility of a rising term premium.

The authors give three reasons term premia were suppressed:

- Structurally lower inflation.
- Fed QE crowding out the long-duration bond market.
- The equity-hedge bid created by negative stock/bond correlation.

They expect all three to reverse or weaken.

### 3.5 Commodity And Energy Evidence

The report shows that commodity secular bull markets have historically tended to occur during weaker stock-market periods. The intuition is that commodity inflation pressures margins, consumer demand, and productivity.

Energy is highlighted because in 2022 it was the only positive S&P 500 sector, and it spent much of 2021/2022 negatively correlated to many other sectors and bonds.

The report also argues energy follows long boom/bust cycles. Years of underinvestment may create a prolonged supportive backdrop for energy assets.

## 4. The RAAM Framework

RAAM stands for Real Asset Allocation Model.

The model has three major steps:

1. Expanded asset menu.
2. Trend-based dynamic ranking.
3. Volatility reduction through hierarchical risk parity.

The model starts from a high-level benchmark allocation:

| Bucket | Benchmark Weight |
|---|---:|
| Equities | 50% |
| Fixed Income | 30% |
| Alternatives | 20% |

The idea is not to abandon equities or bonds entirely. The idea is to reduce dependence on the stock/bond relationship by adding assets with different drivers.

## 5. Expanded Asset Menu

The report’s first step is to expand from a narrow stock/bond universe to a 20-asset universe.

### 5.1 Alternatives

| Asset | Benchmark | Max Weight | Role |
|---|---:|---:|---|
| Bitcoin | 2% | 3% | High-volatility diversifier; included only from 2018 in the backtest |
| Commodities | 4% | 16% | Direct inflation and real-asset exposure |
| Energy | 2% | 10% | Energy supply constraint and inflation hedge |
| Gold | 2% | 10% | Monetary/inflation/geopolitical hedge |
| Managed Futures | 6% | 16% | Trend-following diversifier across asset classes |
| Miners | 2% | 10% | Metals and energy-transition demand exposure |
| Real Estate | 2% | 16% | Real asset/inflation-sensitive income exposure |

Total alternatives benchmark: 20%.

### 5.2 Equities

| Asset | Benchmark | Max Weight | Role |
|---|---:|---:|---|
| Dividend Payers | 5% | 10% | Quality/income equity exposure |
| Emerging Markets ex-China | 1% | 10% | Diversified EM growth outside China |
| Europe | 1% | 10% | International developed-market exposure |
| Japan | 1% | 10% | International developed-market exposure |
| Nasdaq | 20% | 40% | Growth/technology equity exposure |
| U.S. Large Cap | 20% | 40% | Core equity exposure |
| U.S. Small Cap | 2% | 10% | Domestic cyclicality and small-cap exposure |

Total equity benchmark: 50%.

### 5.3 Fixed Income

| Asset | Benchmark | Max Weight | Role |
|---|---:|---:|---|
| Corporate Bonds | 10% | 30% | Credit income |
| Emerging-Market Bonds | 2% | 10% | EM credit/currency-linked risk premium |
| High Yield | 6% | 20% | Credit spread exposure |
| Long-Term Treasuries | 10% | 40% | Duration hedge, if it works |
| T-Bills | 0% | 20% | Cash-like defensive allocation |
| TIPS | 2% | 40% | Inflation-linked fixed income |

Total fixed-income benchmark: 30%.

## 6. Why These Asset Groups Matter

### 6.1 Commodities

Commodities are included because they can perform well in inflationary regimes where both stocks and nominal bonds struggle.

Strategic purpose:

- Hedge inflation shocks.
- Benefit from underinvestment and supply constraints.
- Provide exposure to real-economy scarcity rather than purely financial assets.
- Diversify away from equity-duration dependence.

### 6.2 Energy

Energy is separated from broad commodities because energy has specific supply-cycle dynamics. The report emphasizes years of underinvestment, shale maturation, geopolitical risk, and policy-driven supply constraints.

Strategic purpose:

- Capture energy supply scarcity.
- Hedge oil/gas inflation.
- Provide an asset that can be negatively correlated with broad equities during energy shocks.

### 6.3 Miners

Miners are separated from energy because they may be driven by different demand forces, especially electrification and the green energy transition.

Strategic purpose:

- Capture demand for copper, lithium, nickel, uranium, and other transition-linked materials.
- Add a different real-asset driver from energy.
- Provide asymmetric exposure to capital expenditure cycles.

### 6.4 Gold

Gold is included as a monetary and geopolitical hedge, not as a yield asset.

Strategic purpose:

- Hedge monetary instability.
- Hedge negative real-rate or currency-confidence shocks.
- Diversify from both equities and credit.

### 6.5 Managed Futures

Managed futures are one of the most important diversifiers in the report.

The report describes them as trend-following strategies that can trade:

- Commodities.
- Currencies.
- Stocks.
- Fixed income.

They can go long or short and adjust faster than traditional long-only asset allocations.

The RAAM does not try to create a hedge fund internally. It wants exposure to the beta of managed-futures strategies through an index-like proxy, specifically the SocGen Trend Index in the report.

Strategic purpose:

- Diversify when stocks and bonds become more correlated.
- Profit from sustained macro trends.
- Potentially perform in crisis regimes where traditional assets fail.

### 6.6 Bitcoin

Bitcoin is included at a small weight. The report is cautious: as a standalone asset, the authors are not enthusiastic, but they find it interesting inside a portfolio.

Important backtest adjustment:

- Bitcoin enters the asset menu only from 2018.
- The authors explicitly avoid assuming a meaningful Bitcoin allocation in 2011.
- This avoids an unrealistic backtest boost.

Strategic purpose:

- High-convexity alternative asset.
- Potential monetary debasement hedge.
- Small allocation because of extreme volatility.

## 7. Trend Ranking System

The report’s second major step is a trend system.

The authors argue that expanding the asset menu improves the opportunity set, but adding trend following improves both returns and volatility.

### 7.1 Why Not Simple Momentum

The report criticizes traditional academic momentum, such as 12-month return, because it uses only two data points:

- Starting price.
- Ending price.

It ignores the path between those points.

### 7.2 Why Not Simple Moving Averages

The report criticizes moving-average systems because they are binary:

- Above moving average = own.
- Below moving average = sell.

The authors argue this causes whipsaws and does not allow gradual position changes.

### 7.3 Regression Trendlines And Trend Breadth

The report’s preferred approach is to run regression trendlines across many timeframes.

Each regression can provide:

- Slope: direction and strength of trend.
- Statistical quality: whether the trend is meaningful.
- Residual score: distance from trendline, useful for mean reversion.

The report calls this approach `Trend Breadth`.

An asset with many green/uptrend regression lines has strong trend breadth. An asset with many red/downtrend regression lines has weak trend breadth.

### 7.4 Three Trend Components

The model combines three rankings.

#### 1. Trend Breadth Rank

This ranks assets by the percentage of relevant timeframes where the asset is in an uptrend.

Implementation interpretation:

- Choose multiple lookback windows.
- For each asset and each window, run a regression of log price on time.
- Count windows where slope is positive and statistically useful.
- Convert this into a breadth score.

Example formula:

```text
trend_breadth = count(positive_significant_slopes) / count(valid_windows)
```

#### 2. Trend Strength Rank

This ranks assets cross-sectionally by trend magnitude across statistically meaningful horizons.

Implementation interpretation:

- Use regression slope over windows such as 63, 126, 189, and 252 trading days.
- Annualize or normalize slope by volatility.
- Rank assets from strongest to weakest.

Example formula:

```text
trend_strength = weighted_average(zscore(regression_slope_window_i))
```

#### 3. Mean Reversion Rank

This inverse-ranks assets based on short-term trends. The idea is to avoid blindly chasing assets that are too extended in the short term.

Implementation interpretation:

- Measure short-term distance above trend or short-term return extension.
- Penalize the most extended assets.
- Reward assets with strong broader trends but less short-term overextension.

Example formula:

```text
mean_reversion_score = -zscore(short_term_residual_vs_trend)
```

### 7.5 Final Trend Rank

The three components combine into a final cross-sectional rank.

Example approximation:

```text
final_trend_score =
  0.40 * trend_breadth_rank +
  0.40 * trend_strength_rank +
  0.20 * mean_reversion_rank
```

The PDF does not disclose exact formulas or weights. Any implementation would need to choose transparent approximations.

### 7.6 How Trend Rank Changes Weights

The report shows that as an asset’s trend rank improves, its model weight rises above benchmark. As trend rank weakens, the model weight falls below benchmark.

This is not a binary in/out system. It is an incremental weighting system.

Practical interpretation:

- Each asset has a benchmark weight and max weight.
- Trend rank determines whether the asset gets a discount or boost.
- The boost is bounded by max weights.
- Weights are later adjusted by volatility/risk parity.

Example approximation:

```text
trend_multiplier = 0.25 + 1.50 * percentile_rank(final_trend_score)
trend_weight_raw = benchmark_weight * trend_multiplier
trend_weight_capped = min(trend_weight_raw, max_weight)
```

Then normalize within constraints.

## 8. Volatility Dampening Through HRP

The third step is volatility control.

The report notes a major flaw in pure trend ranking: high-volatility assets naturally dominate. For example, even a strong bond trend may not match the magnitude of an average Bitcoin rally.

To avoid overallocating to high-volatility assets, RAAM applies hierarchical risk parity.

### 8.1 What HRP Does In This Context

The model explicitly defines three high-level groups:

- Alternatives.
- Equities.
- Fixed income.

Then it applies risk parity in two layers:

1. Across the high-level buckets.
2. Within each bucket across the individual assets.

The practical effect:

- High-volatility buckets receive less capital for the same risk contribution.
- High-volatility assets inside a bucket are scaled down.
- Low-volatility assets can receive more capital if they help balance risk.

### 8.2 Why HRP Instead Of Standard Risk Parity

Classic risk parity can produce unintuitive allocations if all assets are treated in one flat universe. HRP groups related assets and prevents a single cluster from dominating.

For this strategy, hierarchy matters because assets naturally cluster:

- Equities behave more like each other.
- Credit and Treasuries behave more like fixed income.
- Alternatives can include very different but often high-volatility assets.

### 8.3 Final Weight Blend

The report says the final model blends the HRP weight with the trend system.

The exact blend is not disclosed.

Implementation approximation:

```text
final_weight_raw = blend(trend_weight, hrp_weight)
```

Possible blend:

```text
final_weight = 0.60 * trend_weight + 0.40 * hrp_weight
```

or:

```text
final_weight = trend_weight * volatility_scaler
```

The right choice should be validated with backtests and turnover analysis.

## 9. Reported Backtest Results

The report compares RAAM against a benchmark defined as 60% S&P 500 and 40% long-term Treasuries.

Backtest period: January 3, 1995 to May 3, 2023.

### 9.1 Key Metrics

| Metric | RAAM Strategy | 60/40 Benchmark |
|---|---:|---:|
| Cumulative Return | 2,002.31% | 863.62% |
| CAGR | 11.34% | 8.32% |
| Sharpe | 1.12 | 0.78 |
| Smart Sharpe | 1.11 | 0.77 |
| Sortino | 1.56 | 1.10 |
| Max Drawdown | -23.63% | -35.07% |
| Longest Drawdown | 1048 days | 1238 days |
| Annualized Volatility | 10.08% | 11.13% |
| Calmar | 0.48 | 0.24 |
| Worst Year | -13.19% | -21.63% |
| Best Year | 34.40% | 29.78% |
| 10Y Annualized | 9.17% | 7.91% |
| 5Y Annualized | 8.94% | 7.16% |
| 3Y Annualized | 11.25% | 5.88% |

### 9.2 Important Interpretation

The backtest shows stronger returns, lower volatility, and smaller drawdowns. However, for our implementation, we should treat these as directional goals rather than exact targets because:

- The PDF uses institutional indices, not necessarily investable ETFs.
- Some assets have limited live histories.
- Managed futures index data may not be accessible in the same way.
- Bitcoin inclusion begins in 2018, which is more realistic than 2011 but still a modeling choice.
- The exact trend and HRP formulas are not fully disclosed.
- Trading costs, taxes, spreads, UCITS constraints, and rebalance timing can change results.

## 10. Strategy Translation For Our Build

The build should aim for an investable approximation, not an exact clone, unless we have access to the same index data and formula details.

The first objective should be a transparent model we can inspect, test, and iterate.

## 11. Proposed Investable Proxy Universe

The report uses indices. For a practical implementation, we need tradable proxies.

Possible U.S.-listed ETF proxy set:

| RAAM Asset | Possible Proxy | Notes |
|---|---|---|
| Bitcoin | `IBIT`, `FBTC`, or `BTC-USD` | Decide whether crypto is allowed |
| Commodities | `DBC`, `GSG`, `PDBC` | Broad commodity exposure |
| Energy | `XLE`, `VDE` | Energy equities, not physical energy |
| Gold | `GLD`, `IAU` | Gold spot proxy |
| Managed Futures | `DBMF`, `KMLM`, `CTA` | Shorter live history than SocGen Trend Index |
| Miners | `XME`, `PICK`, `COPX` | Need choose broad miners vs specific metals |
| Real Estate | `VNQ`, `IYR` | REIT exposure |
| Dividend Payers | `NOBL`, `VIG`, `SCHD` | Dividend quality/proxy choice matters |
| Emerging ex-China | `EMXC` | Clean proxy for EM ex-China |
| Europe | `VGK`, `IEUR`, `FEZ` | Broad Europe |
| Japan | `EWJ`, `DXJ` | Currency hedging choice matters |
| Nasdaq | `QQQ` | Nasdaq-100, not full Nasdaq Composite |
| U.S. Large Cap | `SPY`, `VOO`, `IVV` | S&P 500 proxy |
| U.S. Small Cap | `IWM`, `VB` | Russell 2000 or CRSP small cap |
| Corporate Bonds | `LQD`, `VCIT` | Duration choice matters |
| EM Bonds | `EMB`, `VWOB` | Hard currency vs blended exposure |
| High Yield | `HYG`, `JNK` | Credit spread exposure |
| Long-Term Treasuries | `TLT`, `VGLT` | Duration hedge/risk |
| T-Bills | `BIL`, `SGOV` | Cash-like defensive sleeve |
| TIPS | `TIP`, `SCHP` | Inflation-linked bonds |

Possible UK/UCITS proxy mapping would need separate work if the target is UK-investable implementation.

## 12. Proposed Model Architecture

### 12.1 Data Layer

Inputs needed:

- Daily adjusted close prices for all proxy instruments.
- Asset metadata: bucket, benchmark weight, max weight, start date, expense ratio if available.
- Optional: cash rate or T-bill return data.

Minimum data checks:

- Missing prices.
- ETF inception dates.
- Survivorship and proxy substitutions.
- Currency consistency.
- Whether dividends are included via adjusted prices.

### 12.2 Return Calculation

Use adjusted close total-return-like price series where available.

Compute daily returns:

```text
return_t = price_t / price_{t-1} - 1
```

For monthly rebalancing, compute weights at month-end using only data available up to that date, then apply weights to next month’s returns.

### 12.3 Trend Features

For each asset on each rebalance date:

Run rolling regressions of log price on time across multiple windows.

Suggested windows:

- 21 trading days.
- 42 trading days.
- 63 trading days.
- 126 trading days.
- 189 trading days.
- 252 trading days.
- 378 trading days.

For each window compute:

- Slope.
- Annualized slope.
- T-stat or R-squared.
- Residual z-score.
- Direction: positive or negative.

Trend breadth:

```text
positive_slope_count / valid_window_count
```

Trend strength:

```text
weighted average of annualized slopes, optionally scaled by regression quality
```

Mean reversion:

```text
negative of short-term residual z-score or negative of short-term extension
```

Final score:

```text
final_trend_score = weighted combination of breadth, strength, and mean reversion
```

Then convert final scores to percentile ranks across assets.

### 12.4 Trend Weighting

For each asset:

```text
trend_multiplier = low_multiplier + rank_percentile * multiplier_range
raw_trend_weight = benchmark_weight * trend_multiplier
capped_trend_weight = min(raw_trend_weight, max_weight)
```

Possible initial parameters:

- `low_multiplier = 0.25`
- `multiplier_range = 1.50`

This creates a range from 0.25x benchmark for weakest assets to 1.75x benchmark for strongest assets, before caps and normalization.

Alternative approach:

- Allocate underweight/overweight budget within each bucket so the bucket-level benchmark remains stable.
- This may be easier to interpret and less likely to create excessive bucket drift.

### 12.5 Volatility And HRP Layer

Start with a simple version before full HRP.

#### Version 1: Volatility Scaler

For each asset:

```text
volatility = annualized standard deviation of daily returns over 63 or 126 days
vol_scaler = target_asset_vol / volatility
scaled_weight = trend_weight * clipped(vol_scaler)
```

Then normalize and enforce max weights.

This is easier to implement and debug.

#### Version 2: Two-Level HRP Approximation

Within each bucket:

- Estimate covariance matrix from trailing daily returns.
- Use inverse volatility or HRP to allocate bucket risk.
- Blend with trend weights.

Across buckets:

- Estimate bucket return streams.
- Allocate bucket weights by inverse volatility or HRP.
- Blend with strategic bucket weights.

#### Version 3: Full HRP

Implement hierarchical clustering using correlations and recursively allocate risk budgets.

This is closest to the report but more complex and easier to get wrong.

Recommendation: start with Version 1 or Version 2, then compare against full HRP later.

### 12.6 Final Weight Construction

Possible first-pass formula:

```text
trend_weight = benchmark_weight * trend_multiplier
vol_adjusted_weight = trend_weight / trailing_volatility
capped_weight = min(vol_adjusted_weight, max_weight)
final_weight = capped_weight / sum(capped_weights)
```

Then optionally constrain bucket weights:

```text
alternatives: 10% to 35%
equities: 30% to 65%
fixed_income: 15% to 55%
```

These ranges are not specified in the PDF and would be our implementation choice.

## 13. Rebalance Logic

The PDF does not specify an exact rebalance frequency in the extracted text.

Practical options:

| Frequency | Pros | Cons |
|---|---|---|
| Monthly | Good fit for trend system; manageable turnover | More trading than quarterly |
| Quarterly | Lower turnover; easier for real portfolios | Slower reaction to trend shifts |
| Weekly | Faster response | Higher turnover and noise |
| Signal-triggered | Adaptive | More complex and harder to backtest cleanly |

Recommendation for first implementation: monthly rebalance.

Reason:

- The report’s strategy is dynamic, but not necessarily daily trading.
- Monthly reduces whipsaw and turnover.
- Monthly is easier to validate and explain.

## 14. Backtest Design

We should test at least three portfolios.

### 14.1 Benchmarks

1. 60/40 benchmark:

```text
60% S&P 500 proxy + 40% long-term Treasury proxy
```

2. Static expanded benchmark:

```text
20-asset RAAM benchmark weights, rebalanced monthly
```

3. Dynamic RAAM approximation:

```text
Expanded asset menu + trend ranking + volatility/HRP scaling
```

### 14.2 Required Metrics

Report:

- CAGR.
- Cumulative return.
- Annualized volatility.
- Sharpe.
- Sortino.
- Max drawdown.
- Calmar ratio.
- Worst month.
- Worst year.
- Rolling 12-month return.
- Rolling drawdown.
- Rolling equity beta.
- Rolling stock/bond correlation.
- Turnover.
- Average number of assets held.
- Current weights.
- Weight changes since last rebalance.

### 14.3 Bias Controls

Avoid:

- Same-day signal and same-day execution.
- Using future data in rankings.
- Backfilling ETF data before inception without explicit proxy rules.
- Letting Bitcoin or managed-futures ETFs distort early history unrealistically.
- Ignoring cash returns if T-bill allocation is meaningful.

Use:

- Signal calculated at rebalance close.
- Trades applied from next trading day or next month.
- Asset only eligible after sufficient lookback data exists.
- Explicit start date per asset.

## 15. Dashboard / UI Plan

If implemented in a dashboard, the user-facing output should explain both current recommendations and why they exist.

### 15.1 Main RAAM Page

Sections:

- Current model allocation.
- Bucket allocation: equities, fixed income, alternatives.
- Current overweight/underweight versus benchmark.
- Latest trend ranks.
- Latest volatility scalers or HRP risk contributions.
- Suggested trades from previous allocation to current allocation.
- Backtest performance versus 60/40 and static expanded benchmark.

### 15.2 Current Weights Table

Columns:

- Asset.
- Proxy ticker.
- Bucket.
- Benchmark weight.
- Current weight.
- Max weight.
- Active weight.
- Trend breadth rank.
- Trend strength rank.
- Mean reversion rank.
- Final trend rank.
- Volatility.
- Risk contribution.
- Weight change.

### 15.3 Diagnostics

Charts:

- Equity curve versus benchmarks.
- Drawdown chart.
- Rolling Sharpe.
- Rolling volatility.
- Rolling equity beta.
- Rolling stock/bond correlation.
- Allocation history by bucket.
- Allocation history by asset.
- Heatmap of monthly returns.
- Heatmap of monthly excess returns versus benchmark.

### 15.4 Explainability

Every model weight should be explainable in plain English:

```text
Gold is overweight because its medium-term trend breadth is high, its 252-day regression slope ranks in the top quartile, and its realized volatility is moderate relative to other alternatives.
```

or:

```text
Bitcoin is underweight despite strong trend because its realized volatility causes the risk-scaling layer to cap its contribution.
```

## 16. Phased Build Plan

### Phase 0: Alignment

Decide:

- Faithful clone vs investable approximation.
- U.S.-ETF vs UK/UCITS implementation.
- Whether crypto is allowed.
- Managed-futures proxy.
- Monthly vs quarterly rebalance.
- Simple volatility scaling vs HRP first.
- Whether this is research-only or intended to generate portfolio targets.

### Phase 1: Static Expanded Asset Benchmark

Goal: reproduce the first principle of the PDF: broaden the asset menu.

Deliverables:

- Asset metadata table.
- Proxy tickers.
- Benchmark weights and max weights.
- Monthly rebalanced static portfolio.
- Comparison against 60/40.

Verification:

- Weights sum to 100% on every rebalance date.
- No asset is used before its valid start date.
- Static benchmark returns match manual spot checks.

### Phase 2: Trend Signals

Goal: implement regression-based trend breadth and trend strength.

Deliverables:

- Rolling regression slope features.
- Trend breadth score.
- Trend strength score.
- Mean-reversion score.
- Cross-sectional ranks.
- Trend-only portfolio weights.

Verification:

- Trend scores use only historical data.
- Known trending assets receive high ranks in obvious historical periods.
- No lookahead in rebalance timing.

### Phase 3: Volatility Scaling

Goal: prevent trend ranking from overallocating to high-volatility assets.

Deliverables:

- Realized-volatility estimates.
- Volatility-adjusted trend weights.
- Max-weight enforcement.
- Bucket-level weight diagnostics.

Verification:

- High-vol assets are scaled down versus trend-only weights.
- Max weights are never exceeded.
- Weights sum to 100%.

### Phase 4: HRP Or HRP-Like Risk Model

Goal: approximate the report’s hierarchical risk parity step.

Deliverables:

- Bucket-level risk model.
- Within-bucket risk model.
- Risk contribution table.
- Comparison of inverse-volatility scaling versus HRP.

Verification:

- Risk contributions are calculated correctly.
- Bucket hierarchy is respected.
- Results are stable enough for use.

### Phase 5: Full Backtest And Dashboard

Goal: make the strategy inspectable.

Deliverables:

- Backtest metrics.
- Allocation history.
- Drawdown and performance charts.
- Current recommendation table.
- Explanation text for weights.

Verification:

- Backtest is reproducible.
- Metrics match independent calculations.
- UI clearly distinguishes model output from financial advice.

## 17. Key Implementation Decisions To Make

### 17.1 Faithful Clone Or Practical Approximation

Faithful clone:

- Pros: closer to the PDF.
- Cons: may require inaccessible indices and undisclosed formulas.

Practical approximation:

- Pros: implementable, transparent, testable, investable.
- Cons: results will not match the report exactly.

Recommendation: practical approximation.

### 17.2 Full HRP Or Simpler Volatility Scaling First

Full HRP:

- Pros: closer to the report.
- Cons: more complex and harder to debug.

Simple volatility scaling:

- Pros: transparent and fast to validate.
- Cons: less sophisticated than HRP.

Recommendation: start with simple volatility scaling, then add HRP as an enhancement.

### 17.3 Crypto Inclusion

Include Bitcoin:

- Pros: closer to the report; captures convex alternative exposure.
- Cons: high volatility, regulatory/product risk, can dominate backtests if mishandled.

Exclude Bitcoin:

- Pros: cleaner institutional-style allocation for many investors.
- Cons: deviates from the report.

Recommendation: make Bitcoin optional with a default cap of 0-3%.

### 17.4 Managed Futures Proxy

The report uses the SocGen Trend Index. For an investable proxy, possible ETFs include `DBMF`, `KMLM`, and `CTA`.

Issue: live histories are short. Backtests before ETF inception require either excluding the asset, using index data, or using a synthetic/proxy series.

Recommendation: start with available ETF history only, then optionally add index-history support if data exists.

### 17.5 Rebalance Frequency

Recommendation: monthly.

Rationale:

- Good compromise between trend responsiveness and turnover.
- Easy to explain.
- More practical for actual allocation.

## 18. Risks And Limitations

### 18.1 Model Risk

The exact RAAM formulas are not fully disclosed. Any version we build is an approximation.

### 18.2 Data Risk

ETF histories are shorter than index histories. Backtests can be distorted by proxy choices.

### 18.3 Overfitting Risk

The more parameters we add, the easier it is to fit history rather than build a robust strategy.

Mitigation:

- Use simple, stable parameter choices.
- Avoid optimizing weights to maximize backtest returns.
- Compare across subperiods.
- Track turnover and drawdowns.

### 18.4 Implementation Risk

Trend systems can whipsaw. HRP can produce unstable weights if covariance estimates are noisy.

Mitigation:

- Monthly rebalance.
- Weight caps.
- Smoothing or turnover limits.
- Minimum data requirements.

### 18.5 Investment Suitability Risk

The PDF is a research report, not personalized advice. A dashboard should clearly label outputs as model/research outputs.

## 19. Minimal First Version

The smallest useful version should include:

- 20-asset proxy universe.
- Benchmark and max weights.
- Monthly rebalance.
- Regression trend breadth and trend strength.
- Simple short-term mean-reversion penalty.
- Inverse-volatility scaling.
- Weight caps.
- Backtest versus 60/40 and static expanded benchmark.
- Current weights and explanation table.

Do not build first:

- Full HRP clustering.
- Tax-aware trade optimization.
- Broker execution.
- Leveraged portfolios.
- Complex parameter optimization.

## 20. Success Criteria For A First Build

A first implementation is successful if:

- The model can produce current target weights for all eligible assets.
- The weights are explainable from benchmark, trend rank, and volatility adjustment.
- The backtest has no obvious lookahead bias.
- The model can be compared to 60/40 and static expanded benchmark.
- The implementation makes clear where it approximates the PDF rather than copying it exactly.
- The outputs are stable enough that small data changes do not create absurd allocation swings.

## 21. Bottom Line

The PDF’s strategy is a regime-change response. It assumes the old stock/bond diversification engine has weakened because inflation and real-asset scarcity have returned as central macro forces.

The strategy is best understood as:

```text
Expanded asset universe
+ cross-sectional regression-based trend ranking
+ volatility-aware risk scaling
= dynamic real asset allocation framework
```

For our purposes, the right next step is not to copy every chart or chase the exact backtest. The right next step is to build a transparent, investable approximation that captures the core mechanics: broader assets, trend ranking, volatility control, and clear comparison against 60/40.
