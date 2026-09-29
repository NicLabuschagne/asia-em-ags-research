# DCE Soybean Crush — Spread Structure & Mean Reversion

**Date:** 2026-09-29
**Notebook:** `notebooks/asia_analysis.ipynb`
**Data:** DCE No.1/No.2 soybeans, meal, oil (Sina continuous via akshare), 2018-01 → 2026-09
**Objective:** establish whether the DCE crush margin is a stationary, tradeable spread,
and derive a fair value for it.

---

## 0. Headline

| Hypothesis | Verdict |
|---|---|
| The crush should be built on No.2 soybeans, not No.1 | **Yes** — decisively, see §1 |
| Crush is stationary around a constant mean | **Yes** — ADF p=0.0015, mean +81 CNY/t |
| The mean moves through time (2022–24 "regime shift") | **No** — a plain AR(1) wins on AIC |
| Half-life is ~28 days | **Yes** — two independent estimators agree |
| Tradeable as a standalone sub-book | **No** — 1.3 bets/year is not a strategy |

The crush works as a **fair-value map**. It does not work as a book on its own.

---

## 1. The bean leg is the whole study

No.1 (A) is domestic non-GMO food-grade beans — tofu and soymilk demand, state reserve
policy. It is not a crush input. No.2 (B) is imported GMO beans, which is what Chinese
crushers actually process.

Crush = `meal × 0.785 + oil × 0.185 − bean`:

| bean used | mean (CNY/t) | sd | ADF p | half-life |
|---|---|---|---|---|
| No.1 | −815 | 493 | 0.108 | 81d |
| **No.2** | **+51** | 206 | **0.0015** | **30d** |

The No.2 version sits near zero gross, which is right — net of a ~100–150 CNY/t
processing fee it is mildly negative, matching the well-documented thinness of Chinese
crush margins. The No.1 version is economically meaningless; it measures the food-bean
premium.

**Sample starts 2018** because No.2 was a dead contract before the DCE revamp: median
volume went from 34 lots/day in 2017 to 95,412 in 2018. Prices printed on 2 lots/day
are not prices.

---

## 2. There is no time-varying mean

The 2022–24 collapse looks like a regime shift on a chart. It is not.

| model | AIC |
|---|---|
| **ARIMA(1,0,0) — constant mean** | **21521.1** |
| local level (Kalman) | 21531.2 |
| local level + AR(1) | 21535.2 |

Both state-space models fail in the same way: the level component absorbs everything
(q = 39, implied level half-life 0.19 days; adding an AR term drives φ to 0.9998 and
buys zero log-likelihood). The local level model has **no mean-reversion term** — its
deviation is white noise by assumption — so faced with a persistent AR(1) series, MLE
has nowhere to put the persistence except the level.

The constant-mean fit gives φ = 0.9753, a **27.7-day half-life**, essentially identical
to the 28 days from a direct OLS fit at the start. Two estimators, one answer.

So 2022–24 was a long excursion, not a regime change. Stationarity places no bound on
how long an excursion can last — the worst ran 139 trading days to −807 CNY/t.

---

## 3. Shape

Skew −1.63, innovation kurtosis 50, and the z-score reaches −5.49 but never exceeds
+1.68. Past 1.75σ there are **no short signals at all**.

That asymmetry is industrial, not statistical. Fat margins make crushers raise run
rates, which lifts meal and oil supply and competes the margin away — a soft ceiling.
Collapses have no brake: contracted cargoes keep arriving and plants cost money to
idle, so crushers absorb the loss until committed supply runs out.

Positive crush is arbitraged away; negative crush must be absorbed.

---

## 4. Strategy

Point-in-time expanding z-score (expanding, not rolling — the mean is constant, so a
rolling window discards information). Entry 1.5σ, exit 0.5σ, 90-bar time stop, all
pre-registered. Returns on gross notional of the three legs, point-in-time.

| | binary | continuous (cap 1.0) |
|---|---|---|
| time in market | 18.8% | 88.5% |
| annualised | 3.42% | 5.12% |
| annualised vol | 4.21% | 5.20% |
| Sharpe | 0.81 | **0.99** |
| Sharpe (in market) | 1.91 | 1.05 |
| max drawdown | −6.3% | −6.5% |
| turns/year | 1.3 | 8.9 |

The time stop never fired (max hold 70 bars), so it behaved as intended — insurance,
not a disguised exit rule.

**Why it is not a book.** Sharpe degrades from 1.91 to 0.81 by exactly √(time in
market) — 0.426 against 0.433. The signal is fine; you are simply not on. At 1.3
independent bets a year no achievable IC produces a usable information ratio.

Adding the meal–bean2 spread (cointegrated at p=0.003, β=0.735 on logs) lifts the
sleeve to 1.08 despite being worse standalone — position correlation is −0.02. Breadth,
not signal quality, is the binding constraint.

---

## 5. Caveats

**The legs are not on matched delivery months.** These are stitched continuous series,
so the crush level carries a calendar-spread component potentially the same order as
the signal. Tolerable for characterising the spread; not tolerable for trusting P&L
magnitudes. Re-run on matched contract months before believing the numbers.

**Costs are assumed, not measured** — 5 CNY/t round trip. At this turnover the result
is insensitive to it.

**Kurtosis 50** means every t-statistic here is optimistic.
