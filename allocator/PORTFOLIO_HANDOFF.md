# Portfolio Allocator Implementation Handoff

This is the implementation handoff for the `allocator/` work. It is intended for another engineer or agent to continue the project without reconstructing context from commits or chat history.

Comment directly in this document against open questions, assumptions, and next steps. This file is the canonical handoff. A mirrored copy lives at `/Users/dbg/.claude/plans/portfolio-allocator-handoff.md`.

Last updated: 2026-04-26 (bond allocation overhaul, cash removal, NATO.L addition, allocator_v2 risk-parity engine)

---

## 0. High-Level Investment Thesis

The full design is in `PORTFOLIO_DESIGN.md`. This section is a compact summary for fast orientation.

**Investor**: 30yo UK additional-rate taxpayer, ~£550k investable, 15–25 year horizon, targeting FIRE. Low engagement (quarterly rebalance max). Strong drawdown aversion: –10% on liquid money, –20% on locked-up money.

**Philosophy**: Taleb-flavoured antifragile. Equity is the long-run return engine but only at non-stretched valuations. Real assets (gold, commodities, infrastructure, REITs) provide inflation pass-through and crisis convexity that bonds no longer reliably provide. Bonds are tactical-only — zero baseline, added only when specific macro triggers fire (real yield > 1.5%, EM spread > 600bps, US 10y > 5%). Cash was historically a reserve but is now removed as a strategic sleeve — all buckets are fully invested.

**Entry discipline**: never lump-sum. 30% day-1 into defensive sleeves only (gold, linkers). Remaining 70% deployed ~5%/month over 12 months, accelerated on drawdowns (up to 25% on –20%+ ACWI draw), decelerated when global PE > 90th percentile. The MA200 filter prevents buying collapsing markets or chasing parabolic ones: price must be within (0.85×MA200, 1.30×MA200) to qualify as a buy-now entry.

**Three-bucket structure**:
- **SIPP** (~£300k, locked to age 57): drawdown –20%, vol ~10%. Equity-heavy (65%), global linkers (10%), gold (20%), EM bonds (5%).
- **ISA** (~£150k, liquid): drawdown –10%, vol ~7%. Defensive factors (min-vol, quality, value), EM, global linkers, gold, clean energy tactical. Fully invested.
- **GIA** (~£100k, liquid, IBKR): drawdown –10%, vol ~7%. Real-asset sandbox: gold, miners, commodities, energy, infrastructure, REITs, US TIPS, EM local. Tactical bonds (EMB, TLT) activated by macro triggers.

**Valuation tilt engine** (`valuation.py`): region tilts computed from earnings-yield z-score vs cross-sectional median. Regions cheap vs peers get up to 1.5× overweight; expensive get 0.5× underweight. MA200 dampener (×0.7) applies when price is outside the (0.85, 1.30) band. Bond triggers are binary (on/off) based on real yields and spreads. Deployment pace accelerates on market drawdowns and decelerates at expensive valuations.

**Theme regime layer** (`themes.py`): six labels (`washed_out`, `repairing`, `strong`, `strong_but_stretched`, `dead_money`, `falling_knife`). Only `repairing` and `strong` produce buy-now entries from the constructor. `falling_knife` → REJECT. `dead_money` and `strong_but_stretched` → WATCHLIST. `washed_out` + strong factor signal → contrarian add allowed.

**Two allocator implementations**:
- `allocator/` — strategy/factor allocator. Sleeve targets in `buckets.py`, factor signals in `strategy.py`, construction gating in `construction.py`.
- `allocator_v2/` — risk-parity allocator. Runs ERC and HRP inside sleeve caps, meshes them with an All-Weather quadrant prior, backtests the blend.

---

## 1. Current State

### Universe and wrappers

- Direct stocks are first-class instruments for `SIPP` and `GIA`.
- `ISA` remains fund-only.
- Cash sleeves are **fully removed** as approved strategic targets across all wrappers. All three buckets are fully invested.
- Current universe counts:
  - Total instruments: `714`
  - Direct stocks: `637`
  - Funds / ETFs / ETCs / mutual funds: `77`
  - SIPP-eligible: `656`
  - ISA-eligible: `19`
  - GIA-eligible: `710`

### Data sources

- DuckDB is the main market and research store.
  - Prices and cached technical fields live in `allocator/allocator_cache.duckdb`.
  - Factor snapshots live in DuckDB.
  - IBKR fundamentals live in DuckDB.
  - ETF constituent snapshots live in DuckDB.
- SQLite is used for mutable allocator state in `allocator/allocator.db`.
  - Holdings
  - Bucket targets
  - Deployment log
  - Bucket cash
- The allocator prefers the repo price/performance pipeline when available.
- If the repo parquet export is unavailable in the current shell/runtime, the allocator falls back to its own DuckDB cache and yfinance refresh path.

### IBKR coverage

- IB Gateway is working and useful for many direct stocks and ADRs.
- ETF fundamentals from IBKR / Reuters are inconsistent and should not be treated as the primary ETF valuation source.
- Practical current role of IBKR:
  - direct stock fundamentals
  - account / position integration path
  - optional enrichment for research
- Practical non-role of IBKR:
  - broad ETF PE / PEG truth source

### Bucket targets (current as of 2026-04-26)

**SIPP** (drawdown –20%, vol ~10%, GBP UCITS only, fully invested):

| Sleeve | Weight | Ticker |
|--------|--------|--------|
| Equity: S&P 500 | 10% | CSPX |
| Equity: Quality | 10% | IWQU |
| Equity: Min-vol | 6% | MVOL |
| Equity: Value | 8% | IWVL |
| Equity: EM | 10% | EIMI |
| Equity: Japan | 5% | IJPA |
| Equity: Europe | 8% | VEUR |
| Equity: UK FTSE | 5% | ISF |
| Equity: Defence | 3% | NATO |
| Real: Gold | 20% | SGLN |
| Bonds: Global linkers | 10% | IGIL |
| Bonds: EM local | 5% | SEML |

Key changes vs April 15: INXG (UK linkers) removed entirely → replaced with IGIL (global inflation-linked); NATO.L added at 3% (European defence / Draghi capex thesis, PEG < 1); VEUR raised 4% → 8%; ERNS cash sleeve removed; equity sleeves expanded.

**ISA** (drawdown –10%, GBP UCITS only, fully invested):

| Sleeve | Weight | Ticker |
|--------|--------|--------|
| Equity: Market cap | 15% | IWDA |
| Equity: Min-vol | 20% | MVOL |
| Equity: Quality | 10% | IWQU |
| Equity: Value | 10% | IWVL |
| Equity: EM | 10% | EIMI |
| Equity: Japan | 5% | IJPA |
| Equity: Europe | 5% | VEUR |
| Bonds: Global linkers | 10% | IGIL |
| Real: Gold | 15% | SGLN |
| Thematic: Clean energy | 5% cap | INRG (tactical) |

Key changes vs April 15: INXG removed; IGIL now carries the linker role; ISA significantly broadened — Value, EM, Japan, Europe all added; ERNS cash sleeve removed; Clean energy added as 5% tactical cap.

**GIA** (drawdown –10%, IBKR, fully invested):

| Sleeve | Weight | Ticker |
|--------|--------|--------|
| Real: Gold | 24% | IAU |
| Real: Gold miners | 8% | GDX |
| Real: Commodities | 16% | PDBC |
| Real: Energy | 10% | XLE |
| Real: Infrastructure | 12% | IGF |
| Real: REITs | 8% | VNQ |
| Bonds: US TIPS | 12% | TIP |
| Bonds: EM local | 10% | EMLC |
| Bonds: EM hard-ccy | 6% cap | EMB (tactical) |
| Bonds: Long duration | 6% cap | TLT (tactical) |

Key changes vs April 15: SGOV (USD T-bill cash sleeve) removed; PDBC raised 10%→16%; IAU 25%→24%; XLV and XLI fully removed from strategic baseline (now satellite-only via THEMATIC_EXTRAS).

### allocator_v2 — new risk-parity engine

A second allocator implementation now exists at `allocator_v2/`. It is a **separate Streamlit app** (`make allocator` or `streamlit run allocator_v2/main.py`) that runs ERC and HRP solvers on a shared 90-day covariance matrix, then meshes them 50/50 (user-adjustable).

Key design:
- `allocator_v2/sleeves.py` — bucket-constrained solver: sleeve policy (equity/real/bonds/cash split) is user-defined, ERC/HRP runs _inside_ each sleeve, not across the full universe. Prevents bonds from dominating due to low vol.
- `allocator_v2/universe.py` — 18-asset investable universe with `tilt_bias` per asset.
- `allocator_v2/sizers/erc.py`, `hrp.py` — standalone solvers.
- `allocator_v2/backtest.py` — rolling backtest of the blended weights.
- `allocator_v2/ensemble.py` — AW (All-Weather quadrant prior) sizer.
- MA-relative tilt and barbell mode available in the sleeve layer.

The v2 engine is **independent** of `allocator/`. It does not share holdings, targets, or data sources. It reads prices from the repo pipeline parquet or yfinance directly.

### UI state

- `🧭 Draft`
  - first-draft portfolio by wrapper
  - buy-now / stage-later split
  - core draft by sleeve
  - optional alternatives / satellite ideas
- `🎯 Allocation`
  - current vs target
  - bucket gaps
- `🔬 Factors`
  - raw factor table
  - strategy buy list
  - constructor output
  - theme performance snapshot
  - theme correlation
  - theme-aware stock screen
  - ETF lookthrough and true exposure views

### Current constructor behavior

- Core constructor lives in `allocator/construction.py`.
- Current default mode is `primary_first`.
  - It keeps the strategic primary ticker for a sleeve instead of freely substituting another same-sleeve candidate.
- Timing-based demotion is active.
  - Neutral `HOLD` signals are no longer treated as buy-now.
  - Entries are demoted to `WATCHLIST` if they are too hot on recent performance / proximity to highs.
- Current action semantics:
  - `APPROVED`: acceptable entry now
  - `BUILD_CORE`: core sleeve that is acceptable now but not a special bargain
  - `WATCHLIST`: valid sleeve, bad timing or stretched entry
  - `NO_DATA`: strategic sleeve retained, but factor / timing coverage is incomplete
  - `REJECT`: do not buy now

---

## 2. What Changed Over Time (chronological)

- The original design started much more cash-heavy and defensive. That seed was later rejected.
- `INXG`, `ERNS`, `CSH2`, `SGOV`, and `BIL` were removed from the approved construction path.
  - `INXG` was rejected because of spread / implementation friction.
  - `ERNS`, `CSH2`, `SGOV`, `BIL` were removed because cash stopped being an approved strategic sleeve.
- The early overlap-demo approach used names like `AAPL` and `MSFT` to prove direct-vs-ETF duplication.
  - That was rejected because those names did not fit the stated entry discipline.
- The constructor was tightened after `EIMI` surfaced as buyable despite being near highs.
  - Neutral `HOLD` plus stretched timing no longer shows up as buy-now.
- Theme analytics were added because the valuation-only lens was too weak for actual portfolio construction.
  - The current system now exposes theme performance, theme correlation, and theme-grouped stock screening.

**2026-04-26 changes:**
- `INXG` removed from SIPP and ISA. UK gilt risk is too concentrated and subject to fiscal/currency flight. Replaced with `IGIL` (global government inflation-linked bonds diversified across US/EU/JP/UK).
- `NATO.L` added to SIPP at 3%. Aligned with Russell Napier value thesis and Draghi European investment plan. Defence capex supercycle = long-duration industrial spending + European fiscal expansion. PEG < 1.
- `VEUR` raised 4% → 8% in SIPP. Europe deeply cheap vs US on CAPE basis.
- All cash sleeves removed from strategic targets (`ERNS`, `CSH2`, `SGOV`). All three buckets are now fully invested. Cash that exists in live holdings is treated as execution residue.
- ISA broadened materially: Value (IWVL), EM (EIMI), Japan (IJPA), Europe (VEUR) all added; INRG added as 5% tactical clean energy cap.
- GIA restructured: SGOV removed, PDBC raised, XLV and XLI fully removed from strategic baseline.
- `allocator_v2/` created as a separate risk-parity engine (ERC + HRP + AW quadrant ensemble). Runs independently of the v1 strategy/factor allocator.

---

## 3. Current Gaps / Problems

- The factor model is still too valuation-centric.
- Technical analysis exists, but it is still shallow.
  - Current timing fields are useful, but they are not yet a proper regime model.
- Theme regime classification is missing.
  - The UI shows momentum / drawdown / correlation, but it does not classify a theme into a usable state.
- Theme-aware stock coverage exists, but theme selection logic is still weak.
  - Themes are currently mapped by sleeve membership, not by a richer thematic ontology.
- Many `NO_DATA` sleeves remain because factor coverage is incomplete.
  - This is especially true for gold, linkers, commodities, and some non-equity sleeves.
- Repo performance pipeline integration is inconsistent in bare-shell or partially configured environments.
  - The allocator falls back correctly, but the behavior is not yet unified.
- Theme correlation is currently equal-weight theme correlation.
  - It is not portfolio-weighted.
  - It is not ETF-lookthrough-adjusted.
- The constructor still needs a stricter distinction between:
  - strategic target
  - buy-now list
  - watchlist
  - removed sleeves

---

## 4. Concrete Next Steps

1. Add theme regime labels.
   - `washed out`
   - `repairing`
   - `strong`
   - `strong but stretched`
   - `dead money`
2. Upgrade the constructor so theme regime is a first-class input.
3. Add theme-overlap / concentration using ETF lookthrough.
4. Improve stock theme coverage and theme membership quality.
5. Improve factor coverage for no-data sleeves such as gold, linkers, and commodities.
6. Add an explicit first-draft portfolio editor workflow in the UI.
7. Revisit strategic targets only after theme and technical layers are stable.

Implementation intent for the next pass:

- Theme regime should drive construction decisions more than raw valuation alone.
- Buy-now should require both:
  - acceptable valuation / factor state
  - acceptable technical / regime state
- Theme correlation and overlap should influence diversification decisions, not just be shown diagnostically.

---

## 5. Review Prompts

Another agent should comment directly on these:

1. Which strategic sleeves should be removed entirely?
2. Which themes deserve dedicated regime labels first?
3. Should direct stocks remain optional satellites or become core sleeve candidates?
4. Should theme correlation stay equal-weight, or move to ETF-weighted or portfolio-weighted?
5. Which `NO_DATA` sleeves need data-source work first?
6. Should the constructor keep `primary_first`, or allow controlled same-sleeve substitution?

### 5a. Review comments (2026-04-15 pass)

**1. Strategic sleeves to remove.**
- Keep: `equity_market_cap`, `equity_quality`, `equity_min_vol`, `equity_value`, `equity_em`, `equity_europe`, `equity_japan`, `equity_uk_commodity`, `equity_defence`, `real_gold`, `real_gold_miners`, `real_commodities`, `real_energy`, `real_infrastructure`, `real_reits`, `bonds_tips_us`, `bonds_linkers_global`, `bonds_em_local`.
- Consider removing now: `equity_healthcare` (XLV) and `equity_industrials` (XLI) from the GIA *strategic* baseline. They currently sit at 4% each and overlap materially with IWDA/CSPX held in SIPP/ISA on an ETF-lookthrough basis. They belong in the thematic overlay (`THEMATIC_EXTRAS`), which already defines them. Today they are double-counted — a strategic 4% *and* a satellite candidate.
- Tactical: `bonds_em_hard_ccy` (EMB) and `bonds_long_duration` (TLT) stay, but the constructor should emit a visible "NOT ACTIVE" row when the macro trigger is not firing, rather than silently dropping them via `include_tactical=False`.

**2. Themes that deserve regime labels first.**
Implemented in this pass (see `themes.classify_theme_regimes`): the six-label vocabulary (`washed_out`, `repairing`, `strong`, `strong_but_stretched`, `dead_money`, `falling_knife`) is applied uniformly to every theme present in the theme snapshot. No theme needs bespoke treatment yet; the single model already separates clean energy (`falling_knife`/`washed_out` historically) from gold (`strong`/`strong_but_stretched` recently). If a theme becomes materially miscalled, add a per-theme threshold override — do not add a new label.

**3. Direct stocks — satellites or core.**
Leave as satellites. Making stocks strategic sleeves re-introduces the idiosyncratic-drawdown risk that the –10% ISA constraint is specifically designed to prevent, and they already clear through the lookthrough view for overlap detection. Keep direct stocks as optional GIA-only overlays outside the strategic baseline.

**4. Theme correlation weighting.**
Moved to portfolio-weighted by default (strategic GBP target per ticker). Equal-weight is still available behind a radio toggle in the Factors tab for diagnostic comparisons. Equal-weight was not a truthful view of the portfolio's diversification because it gave equal airtime to 50-ticker direct-stock sleeves and 1-ticker ETF sleeves.

**5. `NO_DATA` sleeves that need data-source work first.**
In priority order:
1. `real_gold` / `real_gold_miners` / `real_commodities` — PE is ill-defined; need to swap the PE lens for a relative-valuation vs own history lens (gold: real yield + TIPS spread; miners: gold / oil ratio; commodities: Bloomberg Commodity / CPI).
2. `bonds_linkers_global` / `bonds_tips_us` — swap forward P/E for the real-yield level. The valuation engine already has `compute_bond_triggers`; factor coverage should expose the same field so the constructor can say "linkers cheap because real yields > 1.5%".
3. `bonds_em_local` — factor proxy is the local-currency carry minus US 10y nominal; we have the ingredients in `macro_latest` but do not surface them per-instrument.
4. `real_infrastructure` / `real_reits` — forward P/E is reported by yfinance; these should not currently be NO_DATA. If they still are, the `factor_data` refresh is failing silently on those tickers; check `refresh_factor_data` logs.

**6. `primary_first` vs same-sleeve substitution.**
Keep `primary_first`. Substitution is a slippery slope when the user is hand-authoring the strategic baseline in `buckets.py`. If another instrument in the same sleeve scores better, it should be promoted into the `buckets.py` target, not substituted silently at construction time. The Draft tab already shows alternative candidates separately.

---

## 6a. Open problems addressed in the 2026-04-15 pass

- **Theme regime labels**: added. Classifier in `allocator/themes.py` (`classify_theme_regimes`, `get_theme_regimes`) produces a six-label regime per theme from the snapshot's median momentum/drawdown/MA200/range fields.
- **Theme regime as constructor input**: added. `construction.build_portfolio_plan` accepts a `theme_regimes` kwarg and gates the final action through `_gate_action`. Regime `falling_knife` → REJECT; `dead_money`/`strong_but_stretched` → WATCHLIST; `washed_out` stays WATCHLIST unless the factor signal is BUY/ACCUMULATE (contrarian add allowed). `repairing`/`strong` pass through.
- **Portfolio-weighted theme correlation**: added. `themes.build_portfolio_weighted_theme_correlation` takes `{ticker: gbp_weight}` and weights each ticker's daily return by its strategic allocation. `themes.build_sleeve_ticker_weights(bucket_sizes)` produces the weights directly from `ALL_BUCKETS`. The Factors tab offers both views behind a radio toggle; portfolio-weighted is the default.
- **Constructor rationale trail**: `_gate_action` now returns explicit reasons that get concatenated into `rationale`, so every demotion is explainable without reading code.
- **Buy-now discipline**: BUY-now in the draft tab now requires BOTH an acceptable factor signal AND a supportive theme regime. `HOLD` + stretched regime was already blocked; this pass also blocks BUY + `strong_but_stretched`, BUY + `dead_money`, and escalates BUY + `falling_knife` to REJECT.
- **Theme-level lookthrough aggregation**: added. `themes.build_theme_lookthrough_concentration` aggregates `summarize_true_exposure` rows by the mapped stock's sleeve so a user holding IWDA + CSPX + XLV sees "X% Equity Technology after lookthrough" instead of an underlying-ticker soup. Rendered in the Factors tab as a table plus a horizontal bar of the top 15 themes.
- **Healthcare / industrials double-counting**: resolved. XLV and XLI are removed from `BUCKET2_GIA_TARGETS` strategic baseline (freed 8% redistributed: +2% IAU, +2% PDBC, +2% TIP, +2% EMLC). They remain available via `THEMATIC_EXTRAS` as satellite overlays.
- **Tactical sleeves rendered as `NOT_ACTIVE`**: `build_portfolio_plan` now accepts a `bond_triggers` kwarg (from `valuation.compute_bond_triggers`). Bond-trigger-gated sleeves (EMB, TLT, bonds_linkers_*) whose trigger is dormant become `NOT_ACTIVE` rows with zero target weight and an explicit "trigger not firing (cap X%)" rationale, instead of silently disappearing via `include_tactical=False`. `include_tactical` now defaults to True. Thematic tactical sleeves (e.g. INRG clean energy) are *not* subject to this — they stay in the factor/regime gate path because they are not macro-trigger-driven.
- **Hide NO_DATA filter in Draft tab**: added. A checkbox "Hide NO_DATA / NOT_ACTIVE rows" hides dormant tactical + missing-data rows from the Core Draft and Stage Later tables so the draft is purely actionable. Dormant tactical sleeves are always visible in their own "Dormant Tactical Sleeves" sub-section for awareness, regardless of the toggle. Buy-Now list is untouched because it was already filtered to `APPROVED`/`BUILD_CORE` only.

## 6b. Problems still open

- **Stock theme coverage**: still mapped by sleeve membership only. Moving to a multi-label thematic ontology (a stock can be in both `equity_healthcare` and `thematic_ai`) is blocked on a data model decision. Suggest: add a `themes: tuple[str, ...]` field to `Instrument` alongside the existing `sleeve` and treat `sleeve` as the primary theme, `themes` as secondary tags.
- **No-data sleeves**: ranked in 5a(5). Real-asset PE is the biggest false signal today — gold / commodities showing NO_DATA means the constructor cannot gate them at all. The factor layer needs per-sleeve valuation adapters (gold → real-yield spread; commodities → Bloomberg Commodity / CPI ratio; linkers → real-yield level) rather than a single forward-PE lens.

---

## 6. Assumptions

- Repo handoff is canonical.
- The `.claude/plans` file is mirror-only.
- `PORTFOLIO_DESIGN.md` remains the thesis / reference document, not the implementation-state document.
- Comment-in-place is the intended review mode.
- No repo restructuring is required beyond these documentation additions.

---

## 7. File / Component Landmarks

Primary files another agent will need immediately (v1 strategy allocator):

- `allocator/PORTFOLIO_DESIGN.md`
- `allocator/PORTFOLIO_HANDOFF.md`
- `allocator/instruments.py`
- `allocator/buckets.py`
- `allocator/construction.py`
- `allocator/themes.py`
- `allocator/main.py`
- `allocator/data_sources.py`

Primary files for v2 risk-parity allocator:

- `allocator_v2/main.py` — Streamlit UI (ERC / HRP / AW ensemble, side-by-side)
- `allocator_v2/sleeves.py` — bucket-constrained solver, MA tilt, factor tilt, min/max weight
- `allocator_v2/universe.py` — 18-asset universe with tilt biases
- `allocator_v2/sizers/erc.py` — ERC solver
- `allocator_v2/sizers/hrp.py` — HRP solver
- `allocator_v2/ensemble.py` — All-Weather quadrant prior sizer
- `allocator_v2/backtest.py` — rolling backtest
- `allocator_v2/data.py` — price loading (repo parquet or yfinance fallback)

Primary runtime stores:

- `allocator/allocator_cache.duckdb` — v1 price/macro cache
- `allocator/allocator.db` — v1 SQLite holdings, targets, deployment log

Primary scripts:

- `scripts/export_ibkr_stock_fundamentals.py`
- `scripts/refresh_etf_constituents.py`
- `scripts/seed_allocator_portfolio.py`

Run commands:

```bash
make allocator    # v2 risk-parity app (allocator_v2/main.py)
make allocator-v1 # v1 strategy/factor app (allocator/main.py)
make ui           # full fin-tracker-ui with both allocators in sidebar
```

---

## 8. Acceptance Check

Another agent should be able to read only this handoff and understand:

- what exists
- what changed
- what is incomplete
- what the next implementation path is
- where to leave comments or challenges
