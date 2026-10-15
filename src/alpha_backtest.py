"""
Alpha evaluation and backtest for a panel of ag futures.

Expected input: a long DataFrame indexed by (date, market) with at least
    close           settlement of the ratio back-adjusted continuous series
    alpha_*         one column per alpha, computed only from data up to and including that date
Optional columns:
    is_event_day    True on USDA report, limit-locked or roll days (signals from these days are skipped)
    curve_state     'contango', 'neutral' or 'backwardation', known at that date

Two layers of evaluation:
    Layer 1, signal quality:  does the alpha rank future returns? (IC, IC decay, IC by regime)
    Layer 2, tradeable PnL:   does it survive sizing, costs and execution delay? (Sharpe, turnover, breakeven cost)
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------

def add_returns(panel, horizons=(1, 2, 5, 10)):
    """Add the daily return and forward returns for each market.

    forward_return_h at date t is the return from close t to close t+h.
    It is the target we score signals against, so it must never feed into a signal.
    """
    panel = panel.sort_index().copy()
    close_by_market = panel.groupby(level="market")["close"]

    panel["daily_return"] = close_by_market.pct_change()
    for horizon in horizons:
        future_close = close_by_market.shift(-horizon)
        panel[f"forward_return_{horizon}"] = future_close / panel["close"] - 1
    return panel


def delayed_signal(panel, alpha_column, extra_delay_days=0):
    """Return the alpha as it would actually be traded.

    extra_delay_days = 0: trade at the settlement of the day the signal is computed (TAS style).
    extra_delay_days = 1: trade one day later. If the edge vanishes here, it lived at the signal close.
    Signals generated on event days are set to NaN so they are never traded.
    """
    signal = panel[alpha_column].copy()
    if "is_event_day" in panel.columns:
        signal = signal.where(~panel["is_event_day"].astype(bool))
    return signal.groupby(level="market").shift(extra_delay_days)


# ---------------------------------------------------------------------------
# Layer 1: signal quality
# ---------------------------------------------------------------------------

def information_coefficient_table(panel, alpha_column, horizon=1, extra_delay_days=0):
    """Rank IC per market per year (rows = markets, columns = years).

    IC is the Spearman correlation between the signal and the forward return.
    Splitting by market-year gives many roughly independent estimates, so a t-stat
    across them is less inflated than one computed on overlapping daily returns.
    """
    data = pd.DataFrame({
        "signal": delayed_signal(panel, alpha_column, extra_delay_days),
        "target": panel[f"forward_return_{horizon}"],
    }).dropna()

    market = data.index.get_level_values("market")
    year = data.index.get_level_values("date").year

    ic_by_market_year = data.groupby([market, year]).apply(
        lambda block: block["signal"].corr(block["target"], method="spearman")
    )
    return ic_by_market_year.unstack()


def ic_decay(panel, alpha_column, horizons=(1, 2, 5, 10)):
    """Mean IC at each horizon. A fast decay says hold briefly; a slow one tolerates delay."""
    return pd.Series({
        horizon: information_coefficient_table(panel, alpha_column, horizon).stack().mean()
        for horizon in horizons
    }, name="mean_ic")


def ic_by_regime(panel, alpha_column, horizon=1, regime_column="curve_state"):
    """IC and day count per market and curve state. Low day counts mean low trust."""
    target_column = f"forward_return_{horizon}"
    data = panel[[alpha_column, target_column, regime_column]].dropna()
    market = data.index.get_level_values("market")

    def score(block):
        return pd.Series({
            "ic": block[alpha_column].corr(block[target_column], method="spearman"),
            "days": len(block),
        })

    return data.groupby([market, data[regime_column]]).apply(score).unstack(regime_column)


# ---------------------------------------------------------------------------
# Layer 2: tradeable PnL
# ---------------------------------------------------------------------------

def _backtest_one_market(signal, daily_return, cost_bps, target_annual_vol):
    """Turn one market's signal into positions and daily PnL (all in return units)."""

    # 1. Normalise the signal against its own history so its scale is stable over time.
    rolling_mean = signal.rolling(TRADING_DAYS, min_periods=TRADING_DAYS // 2).mean()
    rolling_std = signal.rolling(TRADING_DAYS, min_periods=TRADING_DAYS // 2).std()
    zscore = ((signal - rolling_mean) / rolling_std).clip(-2, 2) / 2  # now in [-1, 1]

    # 2. Scale to a constant volatility target so every market carries similar risk.
    realised_vol = daily_return.rolling(20).std() * np.sqrt(TRADING_DAYS)
    position = (zscore * target_annual_vol / realised_vol).fillna(0.0)

    # 3. A position set at close t earns the return from close t to close t+1.
    gross_pnl = position.shift(1) * daily_return

    # 4. Pay costs on every change in position (notional traded), on the day of the trade.
    turnover = position.diff().abs()
    net_pnl = gross_pnl - turnover * cost_bps / 10_000

    return pd.DataFrame({
        "position": position,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "turnover": turnover,
    })


def backtest_alpha(panel, alpha_column, cost_bps=3.5, extra_delay_days=0, target_annual_vol=0.10):
    """Backtest one alpha on every market, then combine into an equal-risk portfolio.

    Returns (portfolio, net_pnl_by_market). Each market is already vol-targeted,
    so a simple average across markets gives equal risk weights.
    """
    signal = delayed_signal(panel, alpha_column, extra_delay_days)

    per_market = {}
    for market in panel.index.get_level_values("market").unique():
        per_market[market] = _backtest_one_market(
            signal=signal.xs(market, level="market"),
            daily_return=panel["daily_return"].xs(market, level="market"),
            cost_bps=cost_bps,
            target_annual_vol=target_annual_vol,
        )

    combined = pd.concat(per_market, axis=1)  # columns: (market, field)

    def field(name):
        return combined.xs(name, axis=1, level=1)

    portfolio = pd.DataFrame({
        "gross_pnl": field("gross_pnl").mean(axis=1),
        "net_pnl": field("net_pnl").mean(axis=1),
        "turnover": field("turnover").mean(axis=1),
        "gross_exposure": field("position").abs().mean(axis=1),
    })
    return portfolio, field("net_pnl")


def performance_summary(portfolio):
    """Headline numbers for one alpha's portfolio."""
    gross = portfolio["gross_pnl"].dropna()
    net = portfolio["net_pnl"].dropna()
    annualise = np.sqrt(TRADING_DAYS)

    cumulative = net.cumsum()
    drawdown = cumulative - cumulative.cummax()
    mean_turnover = portfolio["turnover"].mean()

    return {
        "gross_sharpe": gross.mean() / gross.std() * annualise,
        "net_sharpe": net.mean() / net.std() * annualise,
        "annual_return_net": net.mean() * TRADING_DAYS,
        "annual_vol": net.std() * annualise,
        "max_drawdown": drawdown.min(),
        "daily_turnover": mean_turnover,
        # Average position size divided by how much of it trades each day ~ days held
        "holding_days": portfolio["gross_exposure"].mean() / mean_turnover,
        # Cost per side at which the alpha stops making money
        "breakeven_cost_bps": gross.mean() / mean_turnover * 10_000,
        "days": len(net),
    }


# ---------------------------------------------------------------------------
# Running everything
# ---------------------------------------------------------------------------

def evaluate_all_alphas(panel, cost_bps=3.5, horizon=1):
    """One row per alpha and delay: IC statistics next to PnL statistics."""
    alpha_columns = [column for column in panel.columns if column.startswith("alpha_")]
    rows = []

    for alpha_column in alpha_columns:
        for extra_delay_days in (0, 1):
            portfolio, _ = backtest_alpha(panel, alpha_column, cost_bps, extra_delay_days)
            summary = performance_summary(portfolio)

            ic_values = information_coefficient_table(
                panel, alpha_column, horizon, extra_delay_days
            ).stack()

            summary.update({
                "alpha": alpha_column,
                "extra_delay_days": extra_delay_days,
                "mean_ic": ic_values.mean(),
                "ic_t_stat": ic_values.mean() / ic_values.std() * np.sqrt(len(ic_values)),
                "share_positive_ic": (ic_values > 0).mean(),
            })
            rows.append(summary)

    results = pd.DataFrame(rows).set_index(["alpha", "extra_delay_days"])
    return results.sort_values("ic_t_stat", ascending=False)


def alpha_pnl_correlation(panel, alpha_columns, cost_bps=3.5):
    """Correlation of daily net PnL between alphas. High values mean the same bet twice."""
    net_pnl = {
        alpha_column: backtest_alpha(panel, alpha_column, cost_bps)[0]["net_pnl"]
        for alpha_column in alpha_columns
    }
    return pd.DataFrame(net_pnl).corr()


if __name__ == "__main__":
    # Example flow (panel is your (date, market) DataFrame with close and alpha_* columns):
    #
    # panel = add_returns(panel)
    # leaderboard = evaluate_all_alphas(panel, cost_bps=3.5)
    # print(leaderboard.round(3))
    #
    # Drill into one alpha:
    # print(ic_decay(panel, "alpha_009"))
    # print(ic_by_regime(panel, "alpha_009"))
    # portfolio, by_market = backtest_alpha(panel, "alpha_009")
    # portfolio["net_pnl"].cumsum().plot(title="alpha_009 net")
    pass
