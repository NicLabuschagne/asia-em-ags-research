# Research Registry

Ideas to come back to: features, signals and model concepts. Market-agnostic: every entry is practice
for a transferable method, whatever market it was first tried on. Append new entries; update the
status rather than deleting. Statuses: `idea` → `testing` → `kept` / `dropped` (with the reason).

---

## Research process (reference)

| Step | Features are tested against | Tools | Baseline to beat |
|---|---|---|---|
| Nowcast inputs (measurement model, e.g. predicting stocks) | Weekly **change** in the fundamental target (e.g. stocks) | IC, Newey-West t-stats, per-period stability | "Same as last week" |
| Fair-value regression inputs (e.g. stocks-to-use in the spread FV) | **Not IC.** Ablation: does adding the input improve the residual? | Residual stationarity, beta stability, residual IC, bucket monotonicity, with vs without | The FV without that input; a plain rolling z-score of the raw spread |
| Signal | Forward spread / price change | IC, IC decay over horizons, IC by regime, half-life | No signal; simple carry or momentum rule |
| Meta (filter / size) | Did the primary signal's trade pay? | Precision / F1, points per trade, trade count | The primary signal alone |

- Screen features **per period** (by year / time block): is the relationship stable? Walk-forward **folds** are for fitted models.
- Test changes, not levels: trending and seasonal levels give spurious correlations.
- Many features scanned = multiple testing: require t > ~3 and stable per-period IC.
- Screening on the full sample before a walk-forward is a mild selection leak: screen on early years, or say so.

---

## Alpha due-diligence checklist

Source: Delphic Alpha, *Alpha Due Diligence: A Research Framework* (paid post, 2026-07-23). Eight tests run
on every alpha before capital; six are not about the Sharpe ratio.

| # | Test | Question | Already in my toolkit | Gap / note |
|---|---|---|---|---|
| 1 | Predictive content | Rank IC, and enough breadth for it to matter | `information_coefficient_table` (`src/alpha_backtest.py`), `ut.ic_rank_spearman` | Note the effective number of observations, not the raw count (overlap, autocorrelation) |
| 2 | Standalone economics | Sharpe, Calmar, hit rate, return per trade, turnover | `backtest_alpha` + `performance_summary` (costs in bps) | Add Calmar and % winning months; report turnover next to return per trade |
| 3 | Statistical validation | Is it real? (block **permutation** null) How precise? (**bootstrap** CI) | — | **New.** Shuffle in blocks to keep autocorrelation; the block length should exceed the signal's holding period |
| 4 | Horizon decay | IC vs forward horizon; half-life of predictive content | `ic_decay` | Forecast horizon ≠ execution latency (that is #7) |
| 5 | Conditional efficacy | Can you predict *when* the alpha works? | `ic_by_regime`; the meta-model work | Must be out of sample: same as meta-labelling. Rolling-mean-return = momentum of the alpha's own P&L |
| 6 | Portfolio interaction | Does it improve the existing book at the margin? | `alpha_pnl_correlation` | Add **marginal Sharpe** (book with vs without), not just pairwise correlation |
| 7 | Implementation feasibility | Does the edge survive realistic entry delay? | `extra_delay_days` in `backtest_alpha` | Delay within the bar matters for intraday; the daily version is a one-day delay |
| 8 | Regime stability | Year-by-year / rolling Sharpe; does it change slowly enough to adapt? | IC by year; per-fold reading | Treat Hurst estimates as indicative only |

**Not in the eight, but needed:** multiple-testing control across all alphas tried (deflated Sharpe /
stricter t-stat), point-in-time data and a lookahead check on feature alignment, and capacity.

---

## Meta-model conditioning features

Information the primary signal does **not** already contain. Never filter with the features the signal is built from.

| Feature | Rationale | Status |
|---|---|---|
| Recent realised volatility (5d / 20d) | Signals behave differently in calm and volatile regimes | idea |
| Curve state (M1–M2 level, 5d change) | Curve change predicted M1 at 1–5d in meal/bean/oil | idea |
| Days since the last stocks / supply report | Information is fresh after releases, stale before them | idea |
| Session / calendar effects (Golden Week, CNY reopen, Brazil arrival window) | Known seasonal flows and closure gaps | idea |
| Width of the recent range (for breakout primaries) | A break from a tight range vs a wide one | idea |
| Size of the trigger bar | Small breaks vs large breaks | idea |

---

## Price-based features

| Feature | Definition | Notes | Status |
|---|---|---|---|
| Lag-1 spread change | `yw_spread_chg.shift(1)` | IC +0.075 on the yellow/white maize spread; negative slope in the earliest fold; check whether it comes from a stale yellow-maize settlement | testing |
| Breakout (5 / 10 / 20d) | Close vs prior n-day max/min, shifted 1 | Corr −0.70 with lag1 at 5d, −0.51 at longer windows; on the YW spread breakouts *continue* (use as trend, not fade) | testing |

---

## Spread Atlas families

Source: Delphic Alpha, [The Spread Alpha Atlas](https://delphicalpha.substack.com/p/the-spread-alpha-atlas).
14 feature families (z-score, momentum, ROC, distance from mean, Bollinger %B, skew, kurtosis, variance ratio,
OU half-life, RSI, MACD, cross-leg correlation, return differential, relative vol) × lookbacks 5/10/20/50,
on Kalman-hedged spreads. Finding: mean-reversion families beat momentum on spreads.
Code: the 10 Standard/Exotic Atlas signals in `src/utilities.py` (`list_signals()`).

| Note | Why it matters | Status |
|---|---|---|
| Build hedge ratio with a **filtered** Kalman beta (`pykalman` `filter()`, not `smooth()`), frozen between rebalances | A smoothed beta uses future data: lookahead | idea |
| Group the families by **correlation cluster** before testing; test each cluster as one bet | 9 of the 10 utility signals are mean-reversion variants (`mean_reversion`, `fade_ma`, `keltner_reversion` ≈ the same); fewer independent bets = less multiple testing | idea |
| Test **level vs change** horizons separately | Spread levels can mean-revert over weeks while daily changes continue (YW maize: lag1 IC +0.075, fade-breakout IC negative). "Spreads mean-revert" is a hypothesis per spread and horizon, not a prior | idea |
| Trust the **family-level** result over the best cell | ~1,700 feature × lookback × pair × frequency combinations, no stated false-discovery control | idea |
| Fixed 1:1 physical spreads (YW maize, crush) differ from estimated-hedge statistical pairs | Atlas results may not transfer to fixed-ratio physical spreads | idea |

---

## Signal concepts

| Concept | Description | Status |
|---|---|---|
| Stocks nowcast → fair value residual | ML nowcasts weekly crusher meal stocks; a convex fair value curve maps stocks to a fair spread; residual = signal | idea |
| Report surprise event study | Surprise = reported − nowcast; spread move 1–5d after each release, by size/sign of surprise | idea |
| Breakout primary + meta filter | Breakout picks the side, a classifier on conditioning features decides take/skip | testing |

---

## Model map: which model for which problem

Pick by four questions: data vs noise (few obs / weak signal → few parameters), known non-linearity,
change over time, and whether a PM needs it explained.

| Problem | Default | Worth knowing | Why |
|---|---|---|---|
| Long-run fair value between related prices | OLS on levels + cointegration test | ECM / VECM | Relationship and correction speed in one model |
| Fair value with a drifting relationship | Rolling OLS | Kalman (state-space) | Smoothly changing hedge ratio; use filtered, not smoothed |
| Known non-linear fair value (spread vs stocks) | Splines / piecewise regression | GAM | Flexible curve per input that you can still plot |
| Mean-reversion speed | AR(1) on the residual | OU fit | Half-life and expected move |
| Combining a few correlated signals | Ridge | ElasticNet | Shares weight across correlated features; stable coefficients |
| Many correlated features (curve points, 40 alphas) | PCA → regression | PLS | Compress first; PLS picks factors that predict the target |
| Outlier days dominate | Huber | Quantile regression | Huber limits big days; quantile forecasts tails/ranges |
| Volatility, not direction | GARCH | HAR (realised vol) | Vol is far more predictable than returns; for sizing and conditioning |
| Regime detection | Vol / curve-state rules | HMM / Markov switching | For conditioning and sizing more than timing |
| Interactions and thresholds, many events | Shallow GBM (XGBoost / LightGBM) | Random forest | Needs data; fits noise on weak features |
| Take or skip a trade (meta) | Logistic | Shallow GBM classifier | Binary question on events; GBM only with many events |
| Curve structure | PCA | Nelson-Siegel-style factors | Level / slope / curvature |

To try next for learning: GAM, PCA/PLS, Kalman.

---

## Fair-value framework

Source: my Claude Doc [Ag Fair-Value and Systematic Signal Framework](https://claude.ai/artifact/XoUKZfFJKo78ghGSAymhn6)
(2026-10-13). Market-agnostic parts kept here; the doc has the full detail and references.

**Fair-value model families** (all linear regressions at the core):

| Model | Coefficients | Use for |
|---|---|---|
| Parity / identity | Set by economics | Crush, import parity, oilshare: normal level and band |
| Engle-Granger | Fixed | Two-leg spreads (use Engle-Granger critical values, not plain ADF) |
| DOLS | Fixed | Cleaner hedge ratios when the legs feed back on each other |
| **ARDL / ECM** | Fixed | **The baseline template** |
| Johansen / VECM | Fixed | Three or more legs (beans, meal, oil) |
| Threshold ECM | Piecewise | Arbitrage bands: no correction inside the cost band, correction outside |
| Markov switching | Probabilistic | Policy regimes |
| Kalman / state-space | Time-varying | Drifting hedge ratios; fast vs slow fair-value layers |
| ML regressors | Non-linear | Rarely the main fair value: no extrapolation, no stationary residual guaranteed |

**Baseline: ARDL in error-correction form**, re-estimated on a rolling window. y = what you trade;
x = long-run drivers (β·x = fair value); y − β·x = residual = the signal; α = speed of adjustment
(should be negative; half-life = ln(0.5) / ln(1 + α)); z = short-run conditioning (weather shocks,
report dummies, positioning). One fit gives fair value, residual, half-life and a test of reversion.
Residual quality: relative value / parity best, calendars good, outrights weakest (use as a bias, not a trigger).

**Pipeline:** economics first (variables, signs, parity arithmetic) → point-in-time data aligned by
publication date → order of integration (ADF + KPSS; Zivot-Andrews for breaks) → transform for linearity
(logs, 1/stocks-to-use, splines) → estimate the long-run relationship → validate the residual (stationary,
half-life matches the holding period, stable coefficients: rolling, CUSUM, Gregory-Hansen) → non-linear
adjustment only if diagnostics fail → strictly out of sample → residual z-score hands off to the meta-labeller.
**The residual must be stationary; you can't force cointegration.**

**Fitting:** parameters (β, α, thresholds) re-estimated each walk-forward step; hyperparameters (window,
lag order, z entry, Kalman noise, rolling vs expanding) chosen once on development data (~60% / 20%
validation / 20% holdout run once). BIC for lag length; Newey-West errors. Biggest lever: fix coefficients
from economics where possible; reject relationships whose β swings between refits.

**Evaluating:** judge by what the residual does next, not R² (levels R² > 0.9 is near automatic).

| Test | Check |
|---|---|
| Reversion | Residual stationary; α negative and significant (HAC); half-life ≈ holding period |
| Prediction | Out-of-sample slope of forward change on residual negative; stable IC; Campbell-Thompson OOS R² > 0 |
| Monotonic payoff | Forward return by z bucket slopes cleanly; hit rate and time-to-revert per bucket |
| Stability | β, half-life, residual vol steady across windows and seasons; plausible signs |
| **Naive benchmark** | Beats a plain rolling z-score of the raw spread; if not, the fundamentals add nothing |
| After costs | Simple entry/exit rule: Sharpe, drawdown, turnover, P&L per trade |

**Signal construction:**
- Z-score lookback changes how the deviation is *measured*; Kalman speed changes what it deviates *from*.
  If the residual is mean-zero by construction, divide by rolling std only (don't subtract a rolling mean).
- **IC grid:** lookbacks (or Kalman speeds) × forward horizons; pick a region of consistently good cells,
  not the best cell (a 6 × 8 grid is 48 tries).
- Signals at different speeds are correlated: average them into one score or let the meta-labeller combine them.
- **MIDAS:** low-frequency target from many high-frequency lags with a smooth weighting (2–3 parameters for
  20 lags). For nowcasting before official releases.

**Meta-labelling:** triple-barrier labels (target, stop, time limit) matching live exits; features =
residual + multi-speed z, carry, basis-momentum, positioning, trend, vol, time to report, time of day, volume;
outputs = take/skip, timing, size; **purged k-fold CV with an embargo**. If the base model fails its
tests, a classifier can't rescue it. A direct classifier without a fair value loses interpretability,
a bounded stationary target and natural sizing.

**Sharpe is noisy** (Lo 2002): SE of annual Sharpe ≈ √((1 + SR²/2) / years). A true 1.5 needs ~4 years
to reach t ≈ 2. Judge early on per-trade evidence; deflate backtest Sharpes for variants tried
(Bailey and López de Prado).

---

## Model concepts

| Concept | What it does | Lesson so far | Status |
|---|---|---|---|
| Ridge on 3–5 screened features | Continuous forecast of the forward change (1-step or H-step) | Combining weak, low-correlated features can beat picking the best one | idea |
| Logistic with `sample_weight=|move|` | Direction model that weights days by size | Fixes the count-based intercept; matched Huber in folds 2–5 | kept (lesson) |
| Confidence threshold on `predict_proba` | Trade only outside the training fold's 20/80 percentiles | Raised points per trade, lowered the total; set the threshold from training data only | kept (lesson) |
| XGB on raw price features | Non-linear direction / size | Best parameters unstable, inner IC ≈ 0: fits noise at daily frequency on weak features | dropped |
