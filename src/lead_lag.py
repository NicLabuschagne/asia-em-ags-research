"""
lead_lag.py - curve/outright lead-lag strategy (plus baseline curve state and the combined book).


Input
-----
One row per contract per day (outright closes, not spreads):
    date | contract | close | volume | hold          (hold = open interest)
Contract codes must end in YYMM (e.g. "M2605"), so they can be sorted by expiry.
A spread table (M1905-M1909 values) is not enough: the strategy needs each contract's
own daily return so that returns can be chained across rolls without jumps.

Method in brief
---------------
1. Curve each day: M1 = contract with the highest open interest (the main), M2 and M3 = the
   next two contracts by expiry. The triplet chosen at close t-1 is the one held over day t.
2. Held returns: each leg's return on day t is that contract's own close-to-close return, so a
   change of main contract never creates a fake price jump.
3. Signals at close t (5-day window):
       fly5    = 2*R1 - R2 - R3   front vs the average of the next two (curve tightening if > 0)
       own5    = R1               the outright's own move
       curve_z = fly5 / std(fly5 over the previous 120 valid days)   size of the curve move
4. Divergence: long when the curve tightened but M1 did not rally (fly5 > 0, own5 <= 0),
   short for the reverse (fly5 <= 0, own5 > 0), otherwise flat.
5. Two legs by size:
       fast: |curve_z| <  0.25, held 1 day   (small gap that closes within a day)
       slow: |curve_z| >= 0.25, held 5 days  (large dislocation that takes days to correct)
   lead_lag = fast + slow.
6. Baseline curve state: short M1 in contango with a loosening 5-day slope (M1 vs M2), long
   otherwise; flat when the 1-day divergence points the other way. Combined = both netted.
6b. Curve signal: sign of bm5 (M1's 5-day return minus M2's), averaged over the last 5 days, so
   long when the front has gained on M2 and short when it has lost. Flat when the curve is not
   scored. book() averages one book across several markets into an equal-weight portfolio.
7. Backtest: position decided at close t earns M1's return on day t+1; costs per side on every
   position change, plus a roll (close + reopen) on main-switch days while in a position.
   Every trading day after the first scored close is evaluated, including days when M3 is not
   listed (books are flat then unless a slow trade is still running), so 252 days = 1 year.

Expected output on the DCE soymeal file (2019-2026, slow_hold_days=5) - use it to check your copy:
    fast Sharpe 0.52, slow 1.37, lead-lag 1.47, baseline curve state 0.91, combined 1:1 1.40
(Earlier versions showed 0.73 / 1.62 / 1.79 / 1.28 / 1.80: they evaluated only the 1-2 days after
a scored close, about half of all trading days, which dropped held P&L and inflated annualised
figures. Earlier still, a missing slope was treated as contango; here it is not.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration: every fixed parameter lives here, set before testing
# ---------------------------------------------------------------------------
@dataclass
class Config:
    data_path: str = r"C:\Users\Amin\PycharmProjects\Asia Ags\data\raw\dce_meal_data.parquet"   # change to your file
    lookback_days: int = 5          # window for fly5 / own5 / slope change
    scale_window: int = 120         # valid days used to measure the typical size of fly5
    scale_min_periods: int = 60
    size_cut: float = 0.25          # |curve_z| below = fast leg, at or above = slow leg
    slow_hold_days: int = 5
    cost_per_side: float = 0.00035  # 3.5 bps: about 1 tick + fees per side
    split_date: str = "2022-07-01"  # for the before/after stability check
    vol_filter: bool = False        # flatten every book on high-volatility closes
    vol_window: int = 20            # days of M1 returns in the volatility measure
    vol_quantile: float = 0.90      # "high" = above this percentile of its own past
    vol_min_history: int = 252      # days of volatility history before the filter can switch on
    signal_hold_days: int = 5       # curve signal: days of daily signs averaged into the position


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
def load_contracts(path: str) -> pd.DataFrame:
    """Read the per-contract file and add each contract's own daily return."""
    if path.endswith(".parquet"):
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path, parse_dates=["date"])
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["contract", "date"])
    # Return within a contract only: never across two different contracts
    raw["ret"] = raw.groupby("contract")["close"].pct_change()
    return raw


def expiry_key(contract: str) -> str:
    """Sort key: the YYMM at the end of the code (M2605 -> '2605')."""
    return contract[-4:]


# ---------------------------------------------------------------------------
# 2. Curve construction (M1, M2, M3) and held returns
# ---------------------------------------------------------------------------
def build_curve(raw: pd.DataFrame) -> pd.DataFrame:
    """
    For each date pick M1 (highest open interest) and the next two contracts by expiry.
    Returns closes for the triplet chosen that day and the returns of the triplet held
    over that day (chosen at the previous close). Days without a listed M3 get NaN for
    the M2/M3 legs, so curve signals are only computed when the full triplet exists.
    """
    close = raw.pivot(index="date", columns="contract", values="close")
    returns = raw.pivot(index="date", columns="contract", values="ret")
    open_interest = raw.pivot(index="date", columns="contract", values="hold")
    contracts = sorted(close.columns, key=expiry_key)

    def later(name: str, steps: int):
        position = contracts.index(name) + steps
        return contracts[position] if position < len(contracts) else None

    rows = []
    for date in close.index:
        m1 = open_interest.loc[date].idxmax()
        m2, m3 = later(m1, 1), later(m1, 2)
        full_triplet = (m3 is not None
                        and not np.isnan(close.loc[date, m2])
                        and not np.isnan(close.loc[date, m3]))
        rows.append({
            "date": date, "m1": m1,
            "m2": m2 if full_triplet else None,
            "m3": m3 if full_triplet else None,
            "f1": close.loc[date, m1],
            "f2": close.loc[date, m2] if full_triplet else np.nan,
            "f3": close.loc[date, m3] if full_triplet else np.nan,
        })
    curve = pd.DataFrame(rows).set_index("date")

    # Held returns: the leg chosen at close t-1, priced on day t
    for leg in ["m1", "m2", "m3"]:
        held_contract = curve[leg].shift(1)
        curve[f"r_{leg}"] = [returns.loc[date, name] if isinstance(name, str) else np.nan
                             for date, name in zip(curve.index, held_contract)]

    curve["switch_day"] = (curve["m1"] != curve["m1"].shift(1)) & curve["m1"].shift(1).notna()
    curve["log_slope"] = np.log(curve["f1"] / curve["f2"])            # > 0 backwardation
    curve["fly_level"] = 2 * np.log(curve["f1"]) - np.log(curve["f2"]) - np.log(curve["f3"])
    return curve.dropna(subset=["r_m1"])


def compounded(returns: pd.Series, days: int) -> pd.Series:
    """Compounded return over the last `days` days (NaN if any day in the window is missing)."""
    return np.expm1(np.log1p(returns).rolling(days).sum())


# ---------------------------------------------------------------------------
# 3. Signals
# ---------------------------------------------------------------------------
def curve_signals(curve: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """fly5, own5, slope change (bm5) and the size measure curve_z, all known at the close."""
    r1, r2, r3 = curve["r_m1"], curve["r_m2"], curve["r_m3"]
    n = cfg.lookback_days
    signals = pd.DataFrame(index=curve.index)
    signals["fly5"] = 2 * compounded(r1, n) - compounded(r2, n) - compounded(r3, n)
    signals["own5"] = compounded(r1, n)
    signals["bm5"] = compounded(r1, n) - compounded(r2, n)            # 5-day slope change
    signals["valid"] = signals["fly5"].notna() & signals["own5"].notna()

    # Typical size of the curve move: std over the previous VALID days only (skips M3 gaps),
    # shifted one valid day so today's value is not in its own scale
    valid_fly = signals.loc[signals["valid"], "fly5"]
    scale = valid_fly.rolling(cfg.scale_window, min_periods=cfg.scale_min_periods).std().shift(1)
    signals["curve_z"] = signals["fly5"] / scale.reindex(curve.index)
    signals["scored"] = signals["valid"] & signals["curve_z"].notna()
    return signals


def divergence(signals: pd.DataFrame, min_size: float = 0.0) -> pd.Series:
    """
    +1: curve tightened (fly5 > 0) but M1 did not rally (own5 <= 0)
    -1: curve loosened (fly5 <= 0) but M1 rallied (own5 > 0)
     0: curve and outright agree.  NaN where the signal cannot be scored.
    """
    big_enough = signals["curve_z"].abs() >= min_size
    long_signal = (signals["fly5"] > 0) & (signals["own5"] <= 0) & big_enough
    short_signal = (signals["fly5"] <= 0) & (signals["own5"] > 0) & big_enough
    raw_signal = pd.Series(np.where(long_signal, 1.0, np.where(short_signal, -1.0, 0.0)), index=signals.index)
    return raw_signal.where(signals["scored"])


def fixed_hold(signal: pd.Series, hold_days: int) -> pd.Series:
    """
    Hold each trigger for `hold_days` closes. A new trigger in the same direction restarts
    the clock; an opposite trigger flips the position immediately.
    """
    values = signal.fillna(0).values
    position = np.zeros(len(values))
    current, remaining = 0.0, 0
    for i, value in enumerate(values):
        if value != 0:
            current, remaining = value, hold_days
        elif remaining == 0:
            current = 0.0
        position[i] = current
        remaining = max(remaining - 1, 0)
    return pd.Series(position, index=signal.index)


def lead_lag_positions(signals: pd.DataFrame, cfg: Config) -> dict[str, pd.Series]:
    """Fast leg, slow leg and the lead-lag book (positions set at each close)."""
    all_triggers = divergence(signals, min_size=0.0)
    fast = all_triggers.where(signals["curve_z"].abs() < cfg.size_cut, 0).fillna(0)
    slow = fixed_hold(divergence(signals, min_size=cfg.size_cut), cfg.slow_hold_days)
    return {"fast": fast, "slow": slow, "lead_lag": fast + slow, "divergence_1d": all_triggers.fillna(0)}


def baseline_curve_state(curve: pd.DataFrame, signals: pd.DataFrame, divergence_1d: pd.Series) -> pd.Series:
    """
    Curve-state rule: short in contango with a loosening 5-day slope, long otherwise.
    Flat when the 1-day divergence points the other way (the veto that tested best).
    """
    contango = curve["log_slope"] <= 0       # missing slope (switch days without M3) is NOT contango
    loosening = signals["bm5"] <= 0
    state = pd.Series(np.where(contango & loosening, -1.0, 1.0), index=curve.index).where(signals["valid"]).fillna(0)
    opposes = (divergence_1d != 0) & (divergence_1d != state)
    return state.where(~opposes, 0)


def curve_signal(signals: pd.DataFrame, cfg: Config) -> pd.Series:
    """
    Sign of bm5 (M1's 5-day return minus M2's): +1 when the front has gained on M2, -1 when it
    has lost. The position is the mean of the last signal_hold_days signs, so it holds about a
    week and steps between -1 and +1. A day the curve cannot be scored contributes 0.
    """
    daily = np.sign(signals["bm5"]).where(signals["valid"], 0).fillna(0)
    return daily.rolling(cfg.signal_hold_days, min_periods=1).mean()


def high_volatility(curve: pd.DataFrame, cfg: Config) -> pd.Series:
    """
    True at close t when M1's 20-day volatility is above the 90th percentile of its own history
    up to t-1. Uses only past data; off until there is a year of volatility history.
    """
    vol = curve["r_m1"].rolling(cfg.vol_window).std()
    threshold = vol.expanding(min_periods=cfg.vol_min_history).quantile(cfg.vol_quantile).shift(1)
    return (vol > threshold).fillna(False)


# ---------------------------------------------------------------------------
# 4. Backtest and metrics
# ---------------------------------------------------------------------------
def evaluation_days(signals: pd.DataFrame) -> pd.Series:
    """
    Every trading day after the first close that could be scored. Days when the signal cannot
    be scored (M3 not listed) stay in: books are flat on them unless an earlier trade is still
    held, and that trade's P&L must count. Keeping every day also makes 252 days = 1 year.
    """
    scored = signals["scored"]
    return scored.shift(1, fill_value=False).astype(bool).cummax()


def backtest(position: pd.Series, curve: pd.DataFrame, cfg: Config, mask: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Net P&L, gross P&L and held position per day (fraction of notional per unit)."""
    held = position.shift(1).fillna(0)                       # decided at close t, held over t+1
    changes = held.diff().abs().fillna(held.abs())
    roll_trades = held.abs() * curve["switch_day"].astype(float) * 2
    gross = held * curve["r_m1"]
    net = gross - (changes + roll_trades) * cfg.cost_per_side
    return net[mask], gross[mask], held[mask]


def sharpe(pnl: pd.Series) -> float:
    return pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else np.nan


def metrics(net: pd.Series, gross: pd.Series, held: pd.Series, cfg: Config) -> dict:
    equity = net.cumsum()
    drawdown = equity - equity.cummax()
    # A trade = consecutive days holding the same non-zero position
    trade_id = (held != held.shift(1)).cumsum()
    trades = pd.DataFrame({"held": held, "pnl": net, "id": trade_id})[held != 0].groupby("id")["pnl"].sum()
    wins, losses = trades[trades > 0], trades[trades <= 0]
    return {
        "sharpe_net": sharpe(net),
        "sharpe_gross": sharpe(gross),
        "ann_return_%": net.mean() * 252 * 100,
        "ann_vol_%": net.std() * np.sqrt(252) * 100,
        "max_drawdown_%": drawdown.min() * 100,
        "return_over_dd": net.mean() * 252 / abs(drawdown.min()),
        "time_in_market": (held != 0).mean(),
        "trades_per_year": len(trades) / (len(net) / 252),
        "trade_win_rate": (trades > 0).mean(),
        "avg_trade_bps": trades.mean() * 1e4,
        "payoff": abs(wins.mean() / losses.mean()) if len(losses) else np.nan,
        "sharpe_before_split": sharpe(net[:cfg.split_date]),
        "sharpe_after_split": sharpe(net[cfg.split_date:]),
    }


# ---------------------------------------------------------------------------
# 5. Run
# ---------------------------------------------------------------------------
def run(cfg: Config = Config()) -> dict:
    raw = load_contracts(cfg.data_path)
    curve = build_curve(raw)
    signals = curve_signals(curve, cfg)
    legs = lead_lag_positions(signals, cfg)
    baseline = baseline_curve_state(curve, signals, legs["divergence_1d"])
    books = {
        "fast": legs["fast"],
        "slow": legs["slow"],
        "lead_lag": legs["lead_lag"],
        "baseline_curve_state": baseline,
        "combined_1to1": baseline + legs["lead_lag"],   # netted: one position per contract
        "curve_signal": curve_signal(signals, cfg),
    }
    high_vol = high_volatility(curve, cfg)
    if cfg.vol_filter:
        books = {name: position.where(~high_vol, 0) for name, position in books.items()}
    mask = evaluation_days(signals)
    results, pnl = {}, {}
    for name, position in books.items():
        net, gross, held = backtest(position, curve, cfg, mask)
        results[name] = metrics(net, gross, held, cfg)
        pnl[name] = net
    table = pd.DataFrame(results)
    yearly = pd.DataFrame({name: series.groupby(series.index.year).sum() * 100 for name, series in pnl.items()})
    return {"config": cfg, "curve": curve, "signals": signals, "positions": books, "pnl": pnl,
            "metrics": table, "yearly": yearly, "high_vol": high_vol}


def book(runs: dict[str, dict], name: str = "curve_signal", start: str | None = None) -> dict:
    """
    Equal-weight portfolio of one book across markets. runs = {market: output of run()}.
    Starts on the first day every market is evaluated (or `start`); a market with no P&L that
    day counts as 0. Returns per-market and book P&L, metrics, correlation and yearly returns.
    """
    pnl = pd.DataFrame({market: out["pnl"][name] for market, out in runs.items()})
    first = max(series.first_valid_index() for _, series in pnl.items())
    pnl = pnl[max(first, pd.Timestamp(start)) if start else first:].fillna(0)
    pnl["book"] = pnl.mean(axis=1)
    split = next(iter(runs.values()))["config"].split_date

    def summary(x: pd.Series) -> dict:
        equity = x.cumsum()
        return {"sharpe": sharpe(x), "sharpe_before_split": sharpe(x[:split]), "sharpe_after_split": sharpe(x[split:]),
                "ann_return_%": x.mean() * 252 * 100, "ann_vol_%": x.std() * np.sqrt(252) * 100,
                "max_drawdown_%": (equity - equity.cummax()).min() * 100}

    return {"pnl": pnl, "metrics": pd.DataFrame({col: summary(pnl[col]) for col in pnl}).T,
            "correlation": pnl.drop(columns="book").corr(),
            "yearly": pnl.groupby(pnl.index.year).sum() * 100}


if __name__ == "__main__":
    output = run()
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda value: f"{value:,.2f}")
    days = output["pnl"]["lead_lag"]
    print(f"Evaluation: {len(days)} days, {days.index[0].date()} to {days.index[-1].date()}\n")
    print(output["metrics"].to_string())
    print("\nCalendar-year net return (%)")
    print(output["yearly"].round(1).to_string())