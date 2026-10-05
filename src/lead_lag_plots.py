"""
lead_lag_plots.py - notebook plotting helpers for the output of lead_lag.run().

Engine and charts are kept separate: lead_lag.py computes, this file only draws.
Every function takes the dict returned by run() (called `out` below), draws one chart,
and returns the matplotlib Axes so you can adjust titles or limits afterwards.

Usage in a notebook
-------------------
    import lead_lag as ll
    import lead_lag_plots as llp
    from theme import set_theme
    set_theme()

    out = ll.run(ll.Config(data_path="dce_meal_data.parquet"))   # run once, reuse below

    llp.plot_signal_distribution(out)
    llp.plot_signal_inputs(out, start="2024-01-01", end="2024-07-31")
    llp.plot_equity(out)
    llp.plot_drawdown(out)
    llp.plot_event_study(out)
    llp.plot_yearly(out)
    llp.plot_rolling_sharpe(out)
    llp.plot_trade_returns(out, book="slow")
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from theme import PALETTE

# Colour follows the book, not its position in a list (filtering never recolours a series)
BOOK_COLORS = {
    "fast": PALETTE[3],                   # yellow
    "slow": PALETTE[4],                   # magenta
    "lead_lag": PALETTE[1],               # orange
    "baseline_curve_state": PALETTE[2],   # aqua
    "combined_1to1": PALETTE[0],          # blue
}
DEFAULT_BOOKS = ["lead_lag", "baseline_curve_state", "combined_1to1"]


def _axes(ax, figsize=(11, 4)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
def plot_signal_distribution(out: dict, ax=None):
    """
    Histogram of curve_z on trigger days, split by fast (below the cut) and slow (at or above).
    Shows how many triggers fall in each leg and how wide the size distribution is.
    """
    ax = _axes(ax)
    signals, cut = out["signals"], out["config"].size_cut
    is_trigger = ((signals["fly5"] > 0) & (signals["own5"] <= 0)) | ((signals["fly5"] <= 0) & (signals["own5"] > 0))
    size = signals.loc[is_trigger & signals["scored"], "curve_z"].abs()
    bins = np.linspace(0, size.quantile(0.99), 40)
    ax.hist(size[size < cut], bins=bins, color=BOOK_COLORS["fast"], label=f"fast (|z| < {cut}): {int((size < cut).sum())}")
    ax.hist(size[size >= cut], bins=bins, color=BOOK_COLORS["slow"], label=f"slow (|z| >= {cut}): {int((size >= cut).sum())}")
    ax.axvline(cut, color="#898781", ls="--", lw=1)
    ax.set_xlabel("|curve_z| on trigger days (size of the 5-day curve move vs its typical size)")
    ax.set_ylabel("trigger days")
    ax.set_title("Divergence triggers by size")
    ax.legend()
    return ax


def plot_signal_inputs(out: dict, start=None, end=None, ax=None):
    """
    The two signal inputs over time: curve 5-day change (fly5) vs outright 5-day change (own5).
    Long triggers fire when the curve line is above 0 and the outright line is at or below 0.
    """
    ax = _axes(ax)
    s = out["signals"].loc[start:end]
    ax.plot(s.index, s["fly5"] * 100, color=PALETTE[2], lw=1.5, label="curve: 5-day change in 2*M1 - M2 - M3 (%)")
    ax.plot(s.index, s["own5"] * 100, color="#52514e", lw=1.1, label="outright: 5-day change in M1 (%)")
    ax.axhline(0, color="#898781", lw=1)
    for side, color, marker in [(1, PALETTE[0], "^"), (-1, PALETTE[1], "v")]:
        hit = s[((s["fly5"] > 0) & (s["own5"] <= 0)) if side > 0 else ((s["fly5"] <= 0) & (s["own5"] > 0))]
        hit = hit[hit["scored"]]
        ax.scatter(hit.index, hit["fly5"] * 100, marker=marker, color=color, s=30, zorder=3,
                   label="long trigger" if side > 0 else "short trigger")
    ax.set_ylabel("%")
    ax.set_title("Signal inputs: curve move vs outright move")
    ax.legend(ncol=2, fontsize=8.5)
    return ax


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
def plot_equity(out: dict, books=DEFAULT_BOOKS, ax=None):
    """Cumulative net return per unit of notional (summed, not compounded)."""
    ax = _axes(ax)
    for name in books:
        pnl = out["pnl"][name]
        ax.plot(pnl.index, pnl.cumsum() * 100, color=BOOK_COLORS[name], label=name)
    ax.axhline(0, color="#898781", lw=1)
    ax.set_ylabel("cumulative %, per unit")
    ax.set_title("Equity curves (net of costs)")
    ax.legend()
    return ax


def plot_drawdown(out: dict, books=DEFAULT_BOOKS, ax=None):
    """Distance below the previous equity high."""
    ax = _axes(ax)
    for name in books:
        equity = out["pnl"][name].cumsum()
        ax.plot(equity.index, (equity - equity.cummax()) * 100, color=BOOK_COLORS[name], label=name)
    ax.set_ylabel("%")
    ax.set_title("Drawdown from peak")
    ax.legend(loc="lower left")
    return ax


def plot_rolling_sharpe(out: dict, books=DEFAULT_BOOKS, window=252, ax=None):
    """Sharpe over a trailing window (252 days = 1 year). Shows whether the edge is fading."""
    ax = _axes(ax)
    for name in books:
        pnl = out["pnl"][name]
        rolling = pnl.rolling(window).mean() / pnl.rolling(window).std() * np.sqrt(252)
        ax.plot(rolling.index, rolling, color=BOOK_COLORS[name], label=name)
    ax.axhline(0, color="#898781", lw=1)
    ax.set_title(f"Rolling {window}-day Sharpe")
    ax.legend()
    return ax


def plot_yearly(out: dict, books=DEFAULT_BOOKS, ax=None):
    """Calendar-year net returns, side by side per book."""
    ax = _axes(ax)
    yearly = out["yearly"][books]
    x = np.arange(len(yearly))
    width = 0.8 / len(books)
    for i, name in enumerate(books):
        ax.bar(x + (i - (len(books) - 1) / 2) * width, yearly[name], width=width * 0.92,
               color=BOOK_COLORS[name], label=name)
    ax.set_xticks(x, yearly.index.astype(str))
    ax.axhline(0, color="#898781", lw=1)
    ax.set_ylabel("net return %")
    ax.set_title("Calendar-year net returns")
    ax.legend(ncol=len(books))
    return ax


def plot_event_study(out: dict, horizon=10, ax=None):
    """
    Average M1 return path after each trigger, in the trade's direction, with a 95% band.
    Day 0 = signal close. Shows how long the move lasts, which sets the holding period.
    """
    ax = _axes(ax)
    signals, r1 = out["signals"], out["curve"]["r_m1"]
    trigger = pd.Series(np.where((signals["fly5"] > 0) & (signals["own5"] <= 0), 1,
                        np.where((signals["fly5"] <= 0) & (signals["own5"] > 0), -1, 0)), index=signals.index)
    trigger = trigger[signals["scored"]]
    cut = out["config"].size_cut
    days = range(horizon + 1)
    # forward cumulative return from the next day: column h = sum of returns t+1 .. t+h
    forward = pd.DataFrame({h: r1.shift(-1).rolling(h).sum().shift(-(h - 1)) if h > 0 else 0.0 * r1 for h in days})
    for label, mask, color in [
        ("long, small (fast)", (trigger == 1) & (signals["curve_z"].abs() < cut), BOOK_COLORS["fast"]),
        ("long, large (slow)", (trigger == 1) & (signals["curve_z"].abs() >= cut), BOOK_COLORS["slow"]),
        ("short, all (short P&L)", trigger == -1, PALETTE[1]),
    ]:
        index = trigger[mask.reindex(trigger.index, fill_value=False)].index
        side = trigger.loc[index].values[:, None]
        paths = (forward.loc[index].values * side) * 1e4
        paths = paths[~np.isnan(paths).any(axis=1)]
        mean, se = paths.mean(axis=0), paths.std(axis=0) / np.sqrt(len(paths))
        ax.plot(list(days), mean, marker="o", ms=3, color=color, label=f"{label} (n={len(paths)})")
        ax.fill_between(list(days), mean - 1.96 * se, mean + 1.96 * se, color=color, alpha=0.12, lw=0)
    ax.axhline(0, color="#898781", lw=1)
    ax.set_xlabel("trading days after the signal close")
    ax.set_ylabel("cumulative M1 return in trade direction, bps")
    ax.set_title("What M1 does after a trigger (bands overlap, treat as rough)")
    ax.legend()
    return ax


def plot_trade_returns(out: dict, book="lead_lag", ax=None):
    """Histogram of net return per trade (bps), long and short separately."""
    ax = _axes(ax)
    trades = out["pnl"][book]
    bins = np.linspace(trades["net_bps"].quantile(0.01), trades["net_bps"].quantile(0.99), 40)
    for side, color, label in [(1, PALETTE[0], "long"), (-1, PALETTE[1], "short")]:
        subset = trades.loc[trades["side"] > 0 if side > 0 else trades["side"] < 0, "net_bps"]
        ax.hist(subset, bins=bins, alpha=0.75, color=color,
                label=f"{label}: n={len(subset)}, win {(subset > 0).mean():.0%}, avg {subset.mean():+.0f} bps")
    ax.axvline(0, color="#898781", lw=1)
    ax.set_xlabel("net return per trade, bps")
    ax.set_ylabel("trades")
    ax.set_title(f"Trade outcomes: {book}")
    ax.legend()
    return ax