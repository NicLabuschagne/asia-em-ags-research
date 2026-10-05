# Emerging Market and Asia Ags Research

Quantitative research on emerging-market agricultural futures, in two markets:

- **South Africa (JSE/SAFEX):** white and yellow maize, with CBOT corn and USDZAR as
  the cross-market legs. The recurring object of interest is the SAFEX/CBOT
  import-parity arb and the outright maize price that forms one leg of it.
- **China (DCE):** the soybean complex: soymeal, soybean oil and the beans
  themselves. The questions here are about spreads: the crush margin, and how the
  soymeal curve moves against the outright.

Several separate studies live here. They share the data pipeline in `src/` and the
roll-safe return convention, but each answers its own question and reaches its own
verdict. Most of those verdicts are negative, which is the point: the repo is a
record of what was tested and discarded as much as what survived.

## The studies

### SAFEX maize

**Factor analysis** — a CTA-style feature screen against forward returns on both
the outright and the arb. Rank IC with Newey-White t-stats, decay profiling,
correlation pruning, Bonferroni selection, then walk-forward XGB. Nothing survives
on the outright; the arb is better but still short of significance.
`notebooks/factor_analysis.ipynb`, written up in
`notebooks/factor_analysis_findings.md`.

**CTA and time-series momentum** — whether outright momentum state can be used as
a gate to time the arb, and whether a standalone systematic outright system exists
on yellow or white maize. It cannot and it does not; one scarcity-proxy lead
survives as something worth pre-registering rather than trading.
`notebooks/ts_momentum.ipynb`, written up in `notebooks/ym_cta_findings.md`.

**Regime detection** — whether YM trades in a small number of distinct, reliably
identifiable market states that could each host different positioning. It does
not: corrected for how little independent information slow-moving descriptors
carry, the data supports one state rather than three. What does survive is a
seasonal cycle in the arb, which turns out to be the usable finding for
positioning. `notebooks/seasonality.ipynb`.

**Arb exploration** — the structure of the SAFEX/CBOT arb across the contract
panel and the liquid rolled series. `notebooks/ym_corn_arb.ipynb`.

### DCE soybean complex

**Soymeal curve divergence** — a daily strategy traded in the most-active soymeal
contract (M1), with signals from the shape of the curve against the next two
expiries (M2, M3). Two trades, netted into one position:

- *Curve baseline:* short M1 when the curve is in contango and the M1–M2 spread has
  weakened over the past week, long otherwise. In the market about 90% of the time.
- *Divergence (lead-lag):* the curve tends to move before the outright. When the
  front of the curve has firmed but M1 has not rallied, go long; the reverse, go
  short. Small divergences are held one day, large ones five.

Net of costs over 2019–2026 the combined book has a Sharpe of 1.8, the divergence
trade 1.8 and the curve baseline 1.3. The divergence trade has decayed: 3.1 before
mid-2022 and 0.9 since, so the combined book now leans on the curve baseline.
`notebooks/curve_divergence.ipynb`, engine in `src/lead_lag.py`.

**Soybean crush** — whether the Chinese crush margin is a stationary, tradeable
spread. On No.2 (imported) beans rather than No.1 it is stationary around +81 CNY/t
with no time-varying level: a plain AR(1) beats both Kalman variants on AIC. The
28-day half-life first measured did not survive: about a third of the backtest's
gross P&L came from roll gaps in the unadjusted continuous data, and once those are
cleaned the deviation predicts nothing at 1–20 days. The crush is a fair-value map
for direction, not a weekly trade. `notebooks/asia_analysis.ipynb`, first-pass
write-up in `notebooks/dce_crush_findings.md`; a cointegration view is in
`notebooks/crush_spread_vecm.ipynb`.

**Model stacking** — a learning exercise on soymeal daily returns: XGBoost on
calendar and lagged-return features, combined by a linear meta-model. Trained
properly on out-of-fold forecasts, the meta-model gives every base model zero
weight: none of the features carried a forecast it could trust.
`notebooks/ml_model_stacking.ipynb`.

## Layout

    src/features.py    feature generation
    src/regimes.py     regime clustering, reliability diagnostics, arb response
    src/model.py       XGB + walk-forward validation + performance stats
    src/utilities.py   IC screening, decay, correlation, plots
    src/ingest.py      raw SAFEX/CBOT/FX loaders
    src/roll.py        roll spine and liquid series
    src/lead_lag.py    DCE curve-divergence strategy: curve, signals, books, backtest
    src/lead_lag_plots.py  charts for lead_lag.run() output
    theme.py           shared light/dark chart theme
    build.py           builds the parquet panels in data/processed
    export_report.py   code-free HTML export of any notebook, for sharing

## The convention everything depends on

The panel is a stitched front-contract series. Differencing raw price across a
roll measures the gap between two delivery months, not a market move, and it is
not a small effect — naive daily log returns hit 26% against 6% once returns are
taken within expiry.

So returns are always computed within a contract and summed back into a continuous
series: `clog_` for prices, `c_` for spreads. Every feature is built on those,
never on the raw column. The same trap recurs in different clothing throughout the
repo — roll dates arrive as NaNs that silently void long rolling windows, and they
have to be handled explicitly rather than dropped.

Rolls also follow the calendar, so calendar features will happily predict them.
Before the fix, `season_cos` and `days_to_harvest` were the strongest features in
the factor screen. Both vanished once rolls could no longer leak.

The DCE work follows the same rule: `src/lead_lag.py` chains each contract's own
daily return across changes of main contract. The crush study is the cautionary
case, where a half-life measured on unadjusted continuous data turned out to be
mostly roll gaps.

## Running it

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python build.py

Then run any notebook top to bottom. They read the parquets in `data/processed/`
and never call the loaders directly. All use `%autoreload`, so edits to `src/`
take effect without restarting the kernel.

To share a notebook with someone who does not want to read code:

    python export_report.py notebooks/seasonality.ipynb

That writes markdown, tables and charts only, with images embedded, in both a dark
and a light theme. There is no LaTeX or headless browser installed here, so print
to PDF from the browser if a PDF is needed.
