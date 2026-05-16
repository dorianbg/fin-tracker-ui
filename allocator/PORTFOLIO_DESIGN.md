# Antifragile Two-Bucket Portfolio — Investment Design Document

*Living reference document for the `allocator/` module. All code is designed to
operationalise the principles here. Last revised: 2026-04.*

---

## 1. Investor Profile

| Parameter | Value |
|-----------|-------|
| Age | 30 |
| Tax status | UK additional-rate taxpayer (45% income / 39.35% dividend / 24% CGT) |
| Investable wealth | ~£550k |
| Horizon | 15–25 years (target: FIRE) |
| Engagement | Low — quarterly rebalance maximum |
| Defining constraint | **Drawdown aversion**: –10% on liquid money, –20% on locked-up money |
| Philosophy | Taleb-flavoured: asymmetric, antifragile, "low valuation = less fragile" |
| Hard exclusions | Trend-following, crypto, conventional bonds as primary hedge |

---

## 2. Investment Philosophy (encoded as rules)

### 2.1 Core principles

1. **Equity is the long-term real-return engine** — but only at non-stretched valuations.
   Region tilts are a function of relative earnings yield + a "not crashing, not
   extended" momentum filter. Encoded in `valuation.compute_region_tilts()`.

2. **Real assets are structural ballast** — gold + miners + commodities + materials /
   energy + REITs + infrastructure deliver inflation pass-through and crisis convexity
   that bonds no longer reliably provide in a financial-repression regime.

3. **Bonds are tactical only** — zero baseline. Add only when:
   - UK linkers: UK 10y real yield > 1.5%
   - EM hard-currency: EM HY spread > 600bps over UST
   - Long duration: US 10y nominal > 5%

4. **Cash is a position, not residual** — at 4–5% MMF/T-bill rates it's a real-return
   asset _and_ a deployment reserve. Especially load-bearing in ISA (–10% budget).

5. **Antifragility ≠ deep value** — apply "not crashing, not extended" price filter:
   price ∈ (0.85 × MA200, 1.30 × MA200). This prevents buying collapsing markets
   _and_ chasing parabolic ones. Encoded as `momentum_dampener = 0.7` outside range.

6. **Valuation-triggered deployment** — never lump-sum. 30% day-1 (defensive only),
   then ~5%/month over 12 months, accelerated on ACWI drawdowns (up to 25% on –20%+
   draw), decelerated when global PE > 90th percentile.

### 2.2 New thematic extensions (2026-04)

The core framework is extended with four additional conviction themes. These are held
as **optional thematic overweights** in the GIA bucket, funded by trimming cash or in
lieu of part of the standard equity sleeve. The valuation filter applies: only hold
when the instrument is not in a falling-knife or parabolic state (MA200 check).

| Theme | Rationale | Instrument candidates | Notes |
|-------|-----------|----------------------|-------|
| **Nuclear energy** | Structural supply shortage post-Fukushima decade of underinvestment; zero-carbon base load; antifragile to energy crises; uranium spot currently cheap vs long-run supply cost | NLR, URA, URNM (US ETFs); NUCL.L (GBP UCITS, smaller) | Low PE, low PEG vs long-run growth; inflation-linked utility pricing |
| **Clean / renewable energy** | 2023–2024 rate-driven selloff → sector PE vs 10y history near multi-decade lows; structural policy tailwind (IRA, UK/EU green mandates); compounding installed base | ICLN, QCLN (US ETFs); INRG.L (GBP UCITS, ~$2B) | Apply MA200 filter strictly — sector was in free-fall 2022–2024; only buy once stabilised (ratio > 0.85) |
| **Low PEG sectors** | Buy growth cheaply. PEG = PE / EPS growth rate. Sectors currently with PEG < 1.0 often include healthcare, industrials, materials, energy | XLV (healthcare), XLI (industrials), XLB (materials) | Screen quarterly. Use as allocation tilt, not permanent allocation |
| **Historically cheap sectors** | Own what's in the bottom quartile of its own 10-year PE z-score AND passes MA200 filter | Energy (XLE ✓ already), Financials (XLF), Materials (XLB), European equities (VEUR ✓ already), EM (EIMI ✓ already) | PE z-score logic extends `compute_region_tilts()` to sector level |

---

## 3. Two-Bucket Structure

| Bucket | Size | Wrapper | Drawdown target | Vol target | Broker |
|--------|------|---------|-----------------|-----------|--------|
| **SIPP** | ~£300k | Locked to age 57+ | **–20%** | ~10% | Freetrade / Fidelity SIPP |
| **ISA** | ~£150k | Liquid | **–10%** | ~7% | Freetrade / Fidelity ISA |
| **GIA** | ~£100k | Liquid | **–10%** | ~7% | IBKR |

ISA + GIA share one risk budget. ISA holds GBP UCITS (defensive equity + linkers +
cash). GIA holds USD real-asset sandbox where tightest-spread names live. Premium
Bonds (£50k) substitute for GBP cash in the ISA computation — ~4% effective
tax-free yield for an additional-rate taxpayer.

---

## 4. Target Allocations

### 4.1 SIPP — drawdown –20%, vol ~10%

GBP-listed UCITS only. No US ETFs (platform restriction).

| Sleeve | Weight | Primary ticker | AUM | Rationale |
|--------|--------|---------------|-----|-----------|
| Equity: S&P 500 beta | 8% | CSPX | $50B | Mag7 / US tech exposure at low cost |
| Equity: Quality factor | 8% | IWQU | $5B | High ROIC, low leverage; defensive in downturns |
| Equity: Min-vol factor | 5% | MVOL | $2.5B | Targets –20% drawdown constraint |
| Equity: Value factor | 8% | IWVL | $2.5B | Low-PE tilt; reversion driver |
| Equity: Emerging markets | 9% | EIMI | $20B | Cheap vs history; structural growth |
| Equity: Japan | 4% | IJPA | $5B | Low PE, improving corporate governance |
| Equity: Europe ex-UK | 4% | VEUR | $5B | Deeply cheap vs US; cyclical recovery play |
| Equity: UK FTSE 100 | 4% | ISF | $13B | Energy/mining proxy; inflation equity |
| Real: Gold | 18% | SGLN | $15B | Crisis convexity; no counter-party risk |
| Bonds: UK linkers | 8% | INXG | £2B | Baseline: UK 10y real yield currently positive |
| Bonds: EM local | 4% | SEML | $1.5B | Carry + EM currency diversification |
| Cash: GBP ultrashort | 20% | ERNS | $3B | MMF proxy; real return + deployment reserve |
| **Total** | **100%** | | | |

**Equity total: 50% / Real assets: 18% / Bonds: 12% / Cash: 20%**

Projected real return: ~3.5% (eq 5% × 0.50 + real 3% × 0.18 + bonds 2% × 0.12 + cash 1% × 0.20)

### 4.2 ISA — drawdown –10%, vol ~7%

GBP UCITS only. Defensive posture — the –10% constraint is binding.

| Sleeve | Weight | Primary ticker | AUM | Rationale |
|--------|--------|---------------|-----|-----------|
| Equity: MSCI World beta | 10% | IWDA | $80B | Broad DM; lower concentration than just US |
| Equity: Min-vol | 15% | MVOL | $2.5B | Primary driver of drawdown control |
| Equity: Quality | 15% | IWQU | $5B | Defensive factor in drawdowns |
| Bonds: UK linkers | 30% | INXG | £2B | Real yield protection; main inflation hedge in ISA |
| Real: Gold | 10% | SGLN | $15B | Crisis convexity; no income drag in ISA |
| Cash: GBP ultrashort | 20% | ERNS | $3B | At cap; £50k substituted by Premium Bonds |
| **Total** | **100%** | | | |

**Equity: 40% / Real: 10% / Bonds: 30% / Cash: 20%**

Projected real return: ~3.1%

### 4.3 GIA — drawdown –10%, vol ~7%

IBKR. Full universe: large US ETFs (assumed UK Reporting Fund status) + GBP UCITS.
This is the **real-asset and thematic sandbox** of the portfolio.

| Sleeve | Weight | Primary ticker | AUM | Rationale |
|--------|--------|---------------|-----|-----------|
| Real: Gold | 25% | IAU | $35B | Largest gold ETF; tightest spread; no income |
| Real: Gold miners | 6% | GDX | $13B | Leveraged gold exposure; cheap PEG when producers underfollowed |
| Real: Broad commodities | 10% | PDBC | $5B | Roll-yield optimised; diversified basket |
| Real: Energy equities | 6% | XLE | $40B | Low PE vs history; inflation equity proxy |
| Real: Infrastructure | 10% | IGF | $5B | Inflation-linked revenues; low beta |
| Real: REITs | 6% | VNQ | $30B | Real income; long-run inflation hedge |
| Bonds: US TIPS | 10% | TIP | $20B | USD inflation hedge; complement to INXG in ISA |
| Bonds: EM local currency | 7% | EMLC | $2.8B | EM carry; local currency diversification |
| Bonds: EM hard-currency | 0–6% | EMB | $14B | **Tactical**: only when EM spread > 600bps over UST |
| Bonds: Long duration | 0–6% | TLT | $50B | **Tactical**: only when US 10y nominal > 5% |
| Cash: USD T-bills | 20% | SGOV | $30B | At cap; redeployed when tactical bonds trigger |
| **Total (ex-tactical)** | **100%** | | | |

**Real assets: 63% / Bonds: 17% / Cash: 20% (tactical out of cash)**

Projected real return: ~2.8%

**Blended portfolio real return (£300k SIPP + £150k ISA + £100k GIA):**
≈ **(300×3.5 + 150×3.1 + 100×2.8) / 550 ≈ 3.3% real**

---

## 5. Thematic / Satellite Opportunity Set (GIA)

These are **not in the standard target weights** above. They are **optional tilts**
available within the GIA cash sleeve or as substitutes for part of the real-asset
sleeve when valuation and momentum signals are favourable. Maximum combined thematic
weight: **15% of GIA** (funded from SGOV cash).

### 5.1 Nuclear Energy

**Why it fits the antifragile framework:**
- Uranium spot has been in a structural bear market for a decade (post-Fukushima) →
  severe underinvestment → supply shortage already materialising
- Operating nuclear plants are inflation-protected (regulated utility pricing)
- Benefits from energy security crises (antifragile to geopolitical risk)
- Forward PE of uranium miners ≈ 15–20× but PEG < 1 (earnings growing fast as spot
  price rises); utilities PEG ≈ 0.5–0.8
- **MA200 filter**: URNM and URA are recovering — check ratio before adding

| Ticker | Name | AUM | Exchange | Eligible wrapper | Notes |
|--------|------|-----|----------|-----------------|-------|
| NLR | VanEck Uranium+Nuclear Energy | ~$1.5B | NYSE | GIA | Blended utilities + miners |
| URA | Global X Uranium ETF | ~$2.5B | NYSE | GIA | Miners-heavy; higher vol |
| URNM | Sprott Uranium Miners ETF | ~$1.2B | NYSE | GIA | Pure-play miners |
| NUCL.L | ETC Group Nuclear Energy UCITS | ~$0.3B | LSE | SIPP/ISA/GIA | Small AUM — borderline; check spread |

**Suggested allocation**: 3–5% of GIA when price/MA200 ∈ (0.85, 1.30). Primary: NLR.

### 5.2 Clean / Renewable Energy

**Why it fits the antifragile framework:**
- 2022–2024 rate-driven selloff reset the entire sector; PE vs 10-year history near
  lows for many names. The structural tailwind (IRA credits, EU Green Deal, declining
  solar/wind costs) is intact.
- **Caveat**: sector was in an extended falling-knife phase 2022–2024. The MA200 filter
  is critical here — only enter when price > 0.85 × MA200 (stabilised, not crashing)
- Long duration of cashflows → rate-sensitive → monitor US 10y trajectory

| Ticker | Name | AUM | Exchange | Eligible wrapper | Notes |
|--------|------|-----|----------|-----------------|-------|
| ICLN | iShares Global Clean Energy ETF | ~$2B | NASDAQ | GIA | Broad; most liquid |
| INRG.L | iShares Global Clean Energy UCITS | ~$2B | LSE | SIPP/ISA/GIA | GBP, UCITS; same underlying as ICLN |
| QCLN | First Trust NASDAQ Clean Edge | ~$1.5B | NASDAQ | GIA | More US-concentrated |

**Suggested allocation**: 2–4% of GIA (or ISA via INRG.L) when MA200 filter passes.
Zero allocation when price < 0.85 × MA200 — sector is in a drawdown regime.

### 5.3 Low PEG Sectors (screen quarterly)

PEG ratio = forward PE / 5-year EPS growth rate. Sectors with PEG < 1.0 are paying
less than 1× for a year of growth — a useful "value + growth" combined filter.

Currently (2026-04) low-PEG sectors tend to cluster in:

| Sector | Why cheap PEG | ETF options | Eligible wrappers |
|--------|--------------|-------------|------------------|
| Healthcare | PE ≈ 16× but earnings growing 8–10%/yr on aging demographics + GLP-1 pipeline | XLV ($40B, NYSE), IUHC.L (iShares UCITS) | XLV → GIA; IUHC.L → SIPP/ISA |
| Energy | PE ≈ 10–12× with free cash flow yields 8–12%; capital-return discipline | XLE ($40B, NYSE) — **already in GIA** | GIA |
| Industrials | Infrastructure capex supercycle; PE ≈ 18× with 12%+ EPS growth (reshoring) | XLI ($20B, NYSE) | GIA |
| Materials | Resource nationalism + energy transition metals demand; PE ≈ 13× | XLB ($7B, NYSE) | GIA |
| European value | Entire European market PEG ≈ 0.7× vs own history | VEUR (UCITS) — **already in SIPP** | SIPP/ISA/GIA |

**How to use**: when quarterly sector PE z-score screen flags a sector in the bottom
quartile vs its own 10-year history _and_ MA200 filter passes → substitute up to 5%
from the relevant US equity sleeve.

### 5.4 Historically Cheap Sectors (PE z-score < –1)

Track each sector's current forward PE vs its own 10-year distribution. Sectors with
z-score below –1 (bottom ~16th percentile) are candidates for a valuation tilt.

Currently (approximate, 2026-04):

| Sector | Approx fwd PE | 10y avg PE | Z-score (rough) | Signal |
|--------|--------------|-----------|----------------|-------|
| Energy | 10–12× | 16× | ≈ –1.5 | ✅ Cheap — XLE already in GIA |
| Financials | 12–13× | 14× | ≈ –0.8 | ⚠️ Borderline — XLF candidate |
| Materials | 13–15× | 18× | ≈ –1.2 | ✅ Cheap — XLB candidate |
| Healthcare | 15–17× | 18× | ≈ –0.5 | ⚠️ Borderline |
| Clean energy | 20–25× | 40×+ | ≈ –2.0 | ✅ Very cheap vs history — ICLN/INRG |
| US tech (NASDAQ) | 25–30× | 22× | ≈ +0.7 | ❌ Avoid / underweight |
| Consumer staples | 20× | 19× | ≈ +0.2 | Neutral |

The `compute_region_tilts()` function already encodes this logic at the regional level.
The sector extension would run the same z-score + MA200 logic on sector ETF PE inputs —
a planned addition to the Valuation tab.

---

## 6. Valuation Rule Engine

All rules are pure functions in `allocator/valuation.py`. No discretion; re-runs quarterly.

### 6.1 Regional equity tilts

```python
def compute_region_tilts(regions: list[RegionData]) -> dict[str, float]:
    """
    For each region, compute tilt multiplier in [0.35, 1.5]:
      - Earnings-yield z-score vs cross-sectional median → ±50% tilt
      - 'Not crashing, not extended' filter: if price ∉ (0.85·MA200, 1.30·MA200)
        → multiply by 0.7 (dampener)
    Multipliers are applied to ACWI base weights, then renormalised to 1.0.
    """
```

**Spec verification** (all confirmed by unit tests):
- US PE=22, EM=12, Japan=14, Europe=14, UK=11 → UK=1.50, EM=1.39, JP=EU=1.00, US=0.50 ✅
- Dampener at price=0.80×MA200 → tilt × 0.7 exactly ✅
- Weights renormalise to 1.0 ✅

### 6.2 Bond triggers

```python
def compute_bond_triggers(macro: MacroData) -> dict[str, float]:
    return {
        "linkers_extra": 0.06 if macro.uk_real_yield_10y > 0.015 else 0.0,
        "em_usd":        0.06 if macro.em_hy_spread      > 0.06   else 0.0,
        "long_dur":      0.06 if macro.us_10y_nominal    > 0.05   else 0.0,
    }
```

**Spec verification**: at (uk_real=1.8%, em_spread=7%, us_10y=4.5%) →
linkers ✅, em_usd ✅, long_dur = 0.0 ✅

### 6.3 Deployment pace

```python
def compute_deployment_pace(state, macro) -> tuple[float, str]:
    """
    Default: 1/months_remaining of what is left.
    Drawdown >5%:  base + 5pp extra
    Drawdown >10%: base + 10pp extra
    Drawdown >20%: min(25%, cash_remaining / total)  ← hard cap
    PE > 90th pct: base × 0.5
    """
```

**Spec verification**: default £30k/tranche, DD>20% → £75k, high-val → £15k ✅

---

## 7. Data Sources

| Metric | Source | Frequency | Implementation |
|--------|--------|-----------|----------------|
| ETF prices + MA200 | yfinance (daily OHLCV) | Daily (8h TTL) | `data_sources.refresh_etf_prices()` |
| UK 10y nominal gilt | FRED `IRLTLT01GBM156N` | Daily | `data_sources.refresh_macro()` |
| US 10y TIPS (real) | FRED `DFII10` | Daily | `data_sources.refresh_macro()` |
| US 10y nominal | FRED `DGS10` | Daily | `data_sources.refresh_macro()` |
| EM HY spread | FRED `BAMLH0A0HYM2EY` | Daily | `data_sources.refresh_macro()` |
| GBP/USD FX | yfinance `GBPUSD=X` | Daily | Embedded in ETF price refresh |
| Shiller US CAPE | shillerdata.com XLS | Monthly (30d TTL) | `data_sources.refresh_shiller()` |
| Regional forward P/E | iShares UK product JSON, yfinance fallback | Daily (24h TTL) | `data_sources.refresh_region_pe()` |
| ACWI 30-day drawdown | Derived from yfinance ACWI prices | Daily | `data_sources.get_acwi_drawdown_30d()` |

All data is cached in `allocator/allocator_cache.duckdb`. The app works fully offline
using the last-fetched values.

---

## 8. Wrapper-Aware Instrument Universe

Three instrument universes. Hard rules prevent mis-wiring an instrument into the wrong wrapper.

| Wrapper | Eligible universe | Hard rules | Platform |
|---------|-----------------|-----------|---------|
| SIPP | GBP-listed UCITS only | AUM > £1B; quoted spread < 5bps; accumulating preferred | Freetrade / Fidelity |
| ISA | GBP-listed UCITS only | Same as SIPP | Freetrade / Fidelity |
| GIA | GBP UCITS + large US ETFs | US ETFs: AUM > $5B; assumed UK Reporting Fund status | IBKR |

**UK Reporting Fund status**: per user direction, all recommended US ETFs (IAU, GLD,
GDX, PDBC, IGF, VNQ, TIP, EMB, TLT, SGOV, BIL, XLE, NLR, URA, ICLN, XLV, XLI, XLB)
are **assumed** to hold Reporting Fund status. Verify against HMRC list before purchase.
The `is_reporting_fund` flag in `instruments.py` flips to False if a holding fails
verification — this fires a red warning in Tab 6 of the allocator UI.

---

## 9. Deployment Plan

User starts with ~£550k cash on day 1.

**Day 1** — deploy 30% of each bucket into **defensive sleeves only**:
- SIPP: SGLN (gold) + ERNS (cash MMF) + INXG (linkers)
- ISA: INRG.L or SGLN + INXG + ERNS
- GIA: IAU (gold) + SGOV (T-bills)

Zero equity buying on day 1.

**Months 1–12** — deploy remaining 70% via `compute_deployment_pace()`:
- Default: ~5%/month of original total
- Accelerated on ACWI drawdowns (up to 25% of total on –20%+ draw)
- Decelerated at global PE > 90th percentile
- **Hard floor**: max 1 tranche into any single asset class on a non-trigger day

All executed tranches are recorded in `allocator.db:deployment_log`.

---

## 10. Expected Returns & Tradeoffs

### 10.1 Baseline projections

Using: equity 5% real, real assets 3%, bonds 2%, cash 1% (all real, gross of tax).

| Bucket | Equity | Real assets | Bonds | Cash | Proj. real return |
|--------|--------|------------|-------|------|-------------------|
| SIPP | 50% | 18% | 12% | 20% | **3.5%** |
| ISA | 40% | 10% | 30% | 20% | **3.1%** |
| GIA | 0% | 63% | 17% | 20% | **2.8%** |
| **Blended** | | | | | **≈ 3.3%** |

At £550k for 20 years at 3.3% real, with no contributions: **≈ £1.06M real**.
With max ISA (£20k/yr) + SIPP contributions: plausible range £2–3M real → viable FIRE.

### 10.2 The cost of the –10% ISA constraint

The –10% constraint forces heavy cash + linker allocation in ISA. If relaxed to –12%:
- ISA equity sleeve can grow ~10pp
- Blended real return rises ~+0.3–0.5pp
- At 20 years, that's +£70–120k real on the current base

The drawdown tolerance slider in Tab 1 of the allocator UI lets you see this tradeoff
in real time.

### 10.3 Thematic upside

If nuclear + clean energy tilts (5–7% combined in GIA) deliver 8–10% real over the
next decade (vs the 3% baseline for real assets), the contribution to blended return is:
- +0.5–0.7pp on GIA's return → +0.1pp on blended (small GIA weight)

The main value of the thematic positions is **optionality and convexity**, not return enhancement.

---

## 11. Code Structure

```
allocator/
├── PORTFOLIO_DESIGN.md    ← this document
├── __init__.py            package marker
├── buckets.py             Bucket dataclasses + target sleeve weights
├── instruments.py         Full instrument universe + wrapper flags
├── valuation.py           Pure rule-engine functions (no I/O)
├── holdings.py            SQLite CRUD (holdings, deployment log, cash)
├── data_sources.py        External data fetching + DuckDB cache
├── main.py                Streamlit UI (6 tabs)
├── allocator.db           SQLite holdings store
└── allocator_cache.duckdb price/macro cache

app/pages/
└── 00_PortfolioAllocator.py   sidebar integration shim

tests/
└── test_valuation.py          22 unit tests (all passing)
```

### Run commands
```bash
make test       # uv run python -m pytest tests/ -v
make allocator  # standalone full-screen Streamlit app
make ui         # full fin-tracker-ui; allocator at top of sidebar
```

---

## 12. Decisions Still Open

1. **ISA / GIA £ split**: currently £150k / £100k. Confirm actual amounts to size tickers.
2. **Premium Bonds**: assumed £50k in the cash sleeve (ISA side). Confirm.
3. **Reporting Fund status**: verify each US ETF (especially NLR, URA, ICLN) against
   the live HMRC reporting funds list before purchase.
4. **Thematic activation**: when does nuclear/clean energy cross the MA200 filter?
   Monitor monthly. Add to the Valuation tab sector PE screen (planned).
5. **Sector PE z-score screen**: extend `compute_region_tilts` logic to sector ETFs
   in the GIA — implementation ready when user wants to phase in thematic positions.
6. **NUCL.L / INRG.L AUM**: both GBP UCITS options for nuclear + clean energy, but
   AUM is borderline ($0.3B and ~$2B respectively). Confirm platform availability on
   Freetrade / Fidelity before including in SIPP or ISA.
