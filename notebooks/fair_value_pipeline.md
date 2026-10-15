# Fair Value Pipeline: Reference Notes

Reference for the China ags fair-value → signal → triple barrier → meta-label pipeline.
Main target: DCE soymeal calendar spreads (M 1-5, 5-9, 9-1). Soybeans, soy oil and corn
are secondary.

**Core idea:** fundamentals set the side, statistics set the timing, ML filters the trades.

---

## 1. Data sources

### China's "USDA"
**MARA (Ministry of Agriculture and Rural Affairs)** publishes the monthly **CASDE**
(China Agricultural Supply and Demand Estimates), around the 10th–12th, near the WASDE date.
It covers corn, soybeans, edible oils, cotton and sugar. It moves the market less than the
WASDE: the numbers change slowly and are read partly as policy. The **private weekly data**
is what drives prices.

### Official and state sources

| Source | What | Frequency |
|---|---|---|
| MARA / CASDE | Supply and demand balance sheets | Monthly, ~10th–12th |
| MARA | Breeding sow herd (能繁母猪存栏): leading indicator of meal demand | Monthly |
| China Customs (GACC) | Soybean / meal / oil / corn / sorghum / barley imports. First volume print ~7th–10th, country split ~20th | Monthly |
| NBS | Official crop production | Annual, ~December |
| CNGOIC (国家粮油信息中心) | State grain-and-oils info centre: balance sheets and commentary | Weekly / monthly |
| Sinograin (中储粮) | Reserve auctions (拍卖), purchases (收购), rotation (轮换) | Ad hoc; very important for corn |
| NDRC / MOFCOM | Import quotas, corn TRQ (7.2mt), substitute import rules | Ad hoc |

### Private weekly data (Mysteel, Cofeed, Sublime China Information, 玉米网; mostly via paid terminals or Wind)

**Soymeal**
- Crusher operating rate and weekly crush volume
- **Crusher meal stocks** and **crusher soybean stocks**: the storage-theory inputs
- Meal pickups from crushers (提货量): realised demand
- Port soybean stocks and the arrival schedule
- Meal cash basis at Zhangjiagang, Rizhao, Dongguan
- Crush margins on imported beans, futures-based (盘面榨利) and cash (现货榨利), driven by CBOT + Brazil/US CNF premiums

**Soybeans**
- Import shipping schedule: Brazil dominates ~March–September, US ~October–January. A gap in arrivals tightens nearby meal.
- Brazil premiums (Paranaguá), CBOT, USD/CNY, freight
- Trade policy: tariffs on US beans, state purchase announcements

**Soybean oil**
- Soy oil commercial stocks (weekly)
- **Palm oil**: MPOB report (~10th), Malaysian export data from cargo surveyors, Dalian palm. Soy oil trades very closely with palm.
- Rapeseed oil; trade policy on Canadian canola

**Demand (hogs, which consume most of the meal)**
- Hog prices, hog margins, sow herd
- Monthly feed production (China Feed Industry Association)
- MARA policy on lowering meal in feed rations (since 2023)

**Corn** (more policy than balance sheet)
- Northeast farmer selling pace (售粮进度); harvest runs ~October–December
- North ports (Jinzhou, Bayuquan) and South ports (Shekou, Guangzhou) stocks. South port stocks are the key balance number.
- Shandong deep-processor stocks and daily truck arrivals (深加工门前到车): very high frequency
- North–South freight and price spread
- Starch / ethanol processing margins (CS on DCE)
- Wheat–corn spread (wheat substitution in feed)
- Import margins for US/Brazil/Ukraine corn, sorghum and barley (substitutes)
- Northeast weather; typhoons in August and September

### Calendar notes
- **Golden Week** (Oct 1–7) and **Chinese New Year**: DCE is shut while CBOT, palm and Brazil premiums keep trading, so DCE gaps at the reopen. A known seasonal event and a possible practice primary signal.

---

## 2. Balance sheet: build it weekly

### Principle
Build the balance sheet **from weekly flows up**, then check it against the monthly
official numbers. Don't try to forecast marketing-year-end STU directly.

### Pipeline
1. **Nowcast current stocks** from weekly data.
2. **Project them through the spread's storage window.** M 5-9 prices expected stocks from
   ~May to September, not today and not year end. The horizon is weeks, and it's the
   tractable part:
   - arrivals for the next 6–8 weeks are mostly already known (beans on the water)
   - the crush rate is sticky week to week
   - pickups follow hog demand, which the sow herd flags months ahead
3. **Map the projected stocks to fair value** (section 3).

### Flow identity (consistency check)

```
opening stocks + arrivals − crush − pickups ≈ closing stocks
```

Track the **gap between reported and implied stocks**. When it widens, a data series is
wrong or has been revised: a data-quality alarm, and occasionally a signal.

### Checking against official data
- Check against **hard data**: Customs import volumes (actual cargo).
- CASDE is useful context, but partly a policy statement.
- **Surprise = official figure − your nowcast**, dated to the release. Use it as a meta
  feature, and as a short-horizon event trade.

### Point-in-time discipline
- Mysteel and Cofeed figures get **revised**. Store every figure **as first published**.
- Line data up by **release date, not reference date**.

---

## 3. Fair-value models

### Outright price vs stocks-to-use (the standard reference model)

| Form | Equation | Behaviour |
|---|---|---|
| Log-log | ln P = a + b·ln(STU) | Constant elasticity; curved; easy to interpret |
| Inverse | P = a + b / STU | Price shoots up as STU → 0; captures the scarcity spike |
| Piecewise / kinked | Different slopes above and below a threshold | "Stocks don't matter until they're tight" |

Reference: USDA ERS models (Westcott & Hoffman), season-average price vs WASDE ending STU.

**Weaknesses:**
- Few observations (~20–30 marketing years)
- The curve shifts over time (biofuels, China import growth, inflation). Deflate to real prices and let the level drift.
- The market trades **projected** STU. Fitting on final STU builds in hindsight.
- Production cost sets a floor the regression can't see

In practice **the balance sheet is the core model**. The price mapping is the easy part;
the edge is in forecasting STU better than the market.

Log-normal (multiplicative) errors are reasonable for **flat price only**.

### Calendar spreads: Working's supply-of-storage curve
- Model **spread vs inventory**, not price vs STU.
- Ceiling at **full carry** (storage + financing); anyone can profit above it. No floor
  below: the spread inverts when stocks get tight.
- **No logs**: the spread crosses zero. Normalise as **spread / full carry** and fit a curved
  function (an inverse form, or an exponential approach to the ceiling).
- x-axis: **days of cover** (库存消费比 / 可用天数) = crusher meal stocks / daily pickups.
  This is the Chinese market convention and is published weekly.

### Fitting rules
- Keep it **simple and structural**; no ML here, because observations are few.
- **Fit walk-forward.** A full-sample fit leaks the future into every later stage.
- **Corn caveat:** state activity can override storage theory. Treat Sinograin activity as a
  regime flag (auction running / state buying) rather than trying to model it.

### First step
Plot M 1-5 or 5-9 spread / full carry against crusher days of cover, one colour per year.
Look at the shape before choosing a functional form.

### Check
**Is the residual (price − fair value) mean-reverting?** Run ADF and estimate the half-life.
If it doesn't revert, stop here.

---

## 4. Primary signal

- **Side** = sign of the normalised residual (z-score of the fair-value gap).
- Enter when |z| crosses a threshold. **Keep the threshold loose**: fire often and let the meta model filter.
- **Horizon problem:** if the half-life is weeks (the crush study found ~28d), the gap
  closes on the PM's horizon. Two options:
  1. Accept the longer horizon (correlation risk with the PM), or
  2. **Use fair value for the side and as a filter, with a faster entry trigger**
     (a short-term overshoot inside a fair-value mispricing). Fits the days-long mandate better.

### Check
**Is the primary signal's hit rate or expectancy positive before any ML?** The meta model
filters a signal; it can't create one.

---

## 5. Triple barrier

- **Upper and lower barriers** scaled by volatility: rolling / EWMA std of **daily changes**
  (not % returns, since spreads cross zero).
- **Vertical barrier** = the horizon. **~3–10 days** keeps the sub-book decorrelated from the
  PM; 15–20+ days puts it on the PM's horizon.
- Labels are relative to the primary signal's side: 1 = profit-take hit first, 0 = otherwise.

---

## 6. Meta-label features

Each feature should add information **fair value doesn't already have**, i.e. why a trade
should work right now:

- **Direction of stocks** (building vs drawing), not just the level
- **Stability of fair value**: is the walk-forward estimate still moving?
- **Surprise vs official data** (CASDE / Customs − your nowcast), dated to the release
- **Reported vs implied stocks gap** from the flow identity
- Volatility regime
- Days to roll or expiry
- Seasonality: Brazil arrival window, harvest, Golden Week / CNY closures
- Demand: hog margins, meal pickup volumes, sow herd
- Arrival schedule for the next 6–8 weeks
- Corn: state-activity flag (Sinograin auctions / purchases)

---

## 7. Meta-label model

- **Small sample**: ~50 events a year per product on daily DCE data; stock-based signals give
  fewer, and overlapping barriers cut the independent count further.
- **Simple models**: logistic regression, or a shallow, heavily regularised tree model.
- **Uniqueness weights** so overlapping events don't count twice.
- **Purged k-fold with an embargo**, not plain k-fold.
- **Placebo test:** run the whole pipeline with a random primary signal (random dates and
  sides). If the meta model shows similar lift there, the real result is overfitting.

---

## 8. Practice order

1. **Plumbing run** on a toy primary signal (day-of-week or the Golden Week reopen gap):
   signal → barriers → labels → purged CV → meta model → placebo. Don't expect it to make money.
2. Build the weekly balance sheet nowcast (point-in-time).
3. Fit the storage-curve fair value walk-forward; check residual reversion.
4. Replace the toy primary signal with the fair-value signal; rerun the pipeline.

---

## 9. The general method: one framework for every trade on the desk

Every trade needs a **modelled theory of the outcome**, whether it's systematic fair value,
a scenario, a PM's discretionary call, or a rare event.

> economic hypothesis → a trigger that repeats and is defined in advance → consistent outcome
> labels → condition on context → validate out of sample → size from the distribution

Triple barrier plus meta-labeling is the implementation when events are plentiful. With
few events the steps are the same and only the tools change.

### The event count decides the tool

| Events | Examples | Tool |
|---|---|---|
| Hundreds | Fair-value crossings, technical triggers, parity-band breaks (e.g. SAFEX corn vs import/export parity), **WASDE / CASDE releases** (12 a year × decades) | Full triple barrier + meta-label ML |
| Tens | Seasonal setups, holiday closures, crop-report-specific setups | Conditional hit rates, simple logistic regression, few features |
| A handful | El Niño / La Niña switches (~a dozen since 1950), trade-war shocks, policy regime changes | **No ML.** Analogue comparison side by side, conditional base rates, scenario tables, judgement |

Note: an event that feels rare can still have a large sample. WASDE is a monthly event
with decades of history, which makes it one of the best meta-labeling datasets on the desk.
Count the events before choosing the tool.

### Meta-labeling a PM's discretionary trades
- The primary can be the **PM's call**: it gives a side at a known time.
- **Hindsight risk:** turning "the PM would have traded this" into a rule after seeing outcomes
  leaks the answer. Freeze the trigger definition in writing **before** looking at results.
- Best primary: the PM's **real trade log** (point-in-time by construction; small, and
  reflects their sizing).
- Present the output as **conditions for timing and sizing** ("hit rate goes from 55% to 70%
  when X"), not as a model second-guessing the PM.

### Monte Carlo: what it's for
It adds no information; it shows the spread of outcomes your assumptions imply.
- **Block bootstrap of trade outcomes:** P&L and drawdown distribution for sizing; how
  likely the backtest result was to be luck.
- **OU simulation of the fair-value residual** with the fitted half-life: realistic
  barrier widths and expected holding times.
- **Scenario paths** for macro trades: P&L under each scenario, weighted by probabilities you set.
- **Trap:** simulating from a model fitted on the same data you're judging gives a
  distribution narrower than reality.

### Why this serves the mandate
The framework gives a documented, testable reason why each sub-book trade differs from the
PM's: different trigger, different horizon, measured correlation.

---

## 10. Continuous series vs per-contract (reminder)

- **Continuous, rolled on open interest** (use *yesterday's* OI to decide), **additively
  back-adjusted**. Change-based indicators are valid; anything using the level is not.
- **P&L from the actual held contract**; it must match the adjusted series' changes.
- Use **per-contract groupby** only for life-cycle features (days to expiry, convergence).
- **Spreads:** stitch within one type (e.g. 5-9 across years), or keep `type` as a feature.
  A continuous "front spread" mixes different instruments.
