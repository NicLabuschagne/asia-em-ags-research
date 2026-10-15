"""
Build continuous futures series and curve tables from individual contract files.

Pipeline (each step is a pure function, so you can rebuild with a different roll rule at any time):
    load_contract_files   one file per contract        -> contracts_raw (long table)
    build_contract_meta   contract codes               -> delivery month, roll deadline
    build_roll_schedule   OI switch + deadline         -> contract held after each close
    build_continuous      schedule + prices            -> exact returns, adjusted OHLC, totals
    build_curve           main contracts per date      -> M1/M2/M3 prices, spreads, carry
    named_spread          e.g. 1-5, 5-9                -> rebuilds your spread files
    reconcile_spreads     built vs your file           -> dates where they disagree
    to_backtest_panel     several markets              -> (date, market) panel for alpha_backtest.py

Defaults are set for DCE main contracts (codes like M2405, delivery months 1, 5, 9).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

STANDARD_COLUMNS = ["date", "contract", "open", "high", "low", "close", "settle", "volume", "open_interest"]
CONTRACT_CODE_PATTERN = re.compile(r"^([A-Z]+)(\d{3,4})$")


# ---------------------------------------------------------------------------
# Loading and metadata
# ---------------------------------------------------------------------------

def load_contract_files(folder, file_pattern="*.csv", column_map=None):
    """Read one file per contract into a single long table.

    The contract code is taken from the file name (M2405.csv -> M2405).
    column_map renames your file's columns to the standard names, for example
    {"trade_date": "date", "oi": "open_interest"}. Missing OHLC columns are fine.
    """
    frames = []
    for path in sorted(Path(folder).glob(file_pattern)):
        frame = pd.read_csv(path)
        frame.columns = [column.strip().lower() for column in frame.columns]
        if column_map:
            frame = frame.rename(columns=column_map)
        frame["contract"] = path.stem.upper()
        frame["date"] = pd.to_datetime(frame["date"])
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No files matching {file_pattern} in {folder}")

    contracts_raw = pd.concat(frames, ignore_index=True)
    kept_columns = [column for column in STANDARD_COLUMNS if column in contracts_raw.columns]
    return contracts_raw[kept_columns].sort_values(["date", "contract"]).reset_index(drop=True)


def parse_contract_code(code, first_trade_date):
    """'M2405' -> ('M', 2024-05-01).

    Three-digit codes (ZCE style, 'CF405') carry only the last digit of the year,
    so we take the first matching year on or after the contract's first trade date.
    CBOT style codes (ZCK24) need a different parser.
    """
    match = CONTRACT_CODE_PATTERN.match(code.upper())
    if match is None:
        raise ValueError(f"Cannot parse contract code {code!r}; adapt parse_contract_code for this exchange")

    root, digits = match.groups()
    month = int(digits[-2:])
    if len(digits) == 4:
        year = 2000 + int(digits[:2])
    else:
        year = (first_trade_date.year // 10) * 10 + int(digits[0])
        if year < first_trade_date.year:
            year += 10
    return root, pd.Timestamp(year=year, month=month, day=1)


def build_contract_meta(contracts_raw, roll_buffer_business_days=5):
    """One row per contract with its delivery month and roll deadline.

    roll_deadline = the last date on which we may decide to roll. It sits a few business days
    before the month preceding delivery, because DCE/ZCE individual accounts cannot hold into
    the delivery month and liquidity leaves well before then.
    BDay ignores exchange holidays, so the deadline is approximate by a day or two.
    """
    first_trade_dates = contracts_raw.groupby("contract")["date"].min()
    rows = []
    for contract, first_trade_date in first_trade_dates.items():
        root, delivery_month_start = parse_contract_code(contract, first_trade_date)
        month_before_delivery = delivery_month_start - pd.DateOffset(months=1)
        rows.append({
            "contract": contract,
            "root": root,
            "delivery_month_start": delivery_month_start,
            "delivery_month": delivery_month_start.month,
            "roll_deadline": month_before_delivery - BDay(roll_buffer_business_days),
        })
    return pd.DataFrame(rows).set_index("contract").sort_values("delivery_month_start")


# ---------------------------------------------------------------------------
# Roll schedule
# ---------------------------------------------------------------------------

def build_roll_schedule(contracts_raw, contract_meta, allowed_months=(1, 5, 9)):
    """Decide, for every date, which contract the position holds after the close.

    Rule, using only information known at the close of day t and executed at the close of t+1:
      roll to the next eligible contract when its open interest exceeds the held contract's,
      or when the held contract reaches its roll deadline.
    Eligible = delivery month in allowed_months (the main contracts) and not past its deadline.
    The one-day lag matters: final OI is published after the close, so rolling on it the same
    day would be a quiet look-ahead.
    """
    data = contracts_raw.join(contract_meta, on="contract")
    data = data[data["delivery_month"].isin(allowed_months)]

    records = []
    held_contract = None
    pending_switch = None

    for date, today in data.groupby("date"):
        today = today.set_index("contract")
        eligible = today[today["roll_deadline"] >= date]
        rolled_today = False

        if held_contract is None or held_contract not in today.index:
            # Start of data, or the held contract has no row today: take the most liquid eligible one.
            if eligible.empty:
                continue
            rolled_today = held_contract is not None
            held_contract = eligible["open_interest"].idxmax()
            pending_switch = None
        elif pending_switch is not None and pending_switch in today.index:
            # Execute yesterday's decision at today's close.
            held_contract = pending_switch
            rolled_today = True
            pending_switch = None

        # Decide tomorrow's roll with today's closing information.
        held_delivery = contract_meta.loc[held_contract, "delivery_month_start"]
        later_contracts = eligible[eligible["delivery_month_start"] > held_delivery]
        if not later_contracts.empty:
            next_contract = later_contracts["delivery_month_start"].idxmin()
            liquidity_moved = today.loc[next_contract, "open_interest"] > today.loc[held_contract, "open_interest"]
            deadline_reached = date >= contract_meta.loc[held_contract, "roll_deadline"]
            if liquidity_moved or deadline_reached:
                pending_switch = next_contract

        records.append({"date": date, "held_after_close": held_contract, "roll_day": rolled_today})

    return pd.DataFrame(records).set_index("date")


# ---------------------------------------------------------------------------
# Continuous series
# ---------------------------------------------------------------------------

def _lookup(wide, contract_per_date):
    """For each date, pick wide.loc[date, contract]; NaN where that contract has no value."""
    column_positions = wide.columns.get_indexer(contract_per_date.fillna(""))  # -1 when not found
    values = wide.to_numpy(dtype=float)[np.arange(len(wide)), np.clip(column_positions, 0, None)]
    return pd.Series(np.where(column_positions >= 0, values, np.nan), index=wide.index)


def build_continuous(contracts_raw, roll_schedule, price_field="close"):
    """Continuous series where every daily bar comes from a single contract.

    bar_contract on day t = the contract held after the close of t-1, i.e. the one whose
    P&L you actually earned during day t. Therefore:
      daily_return_t = price_t / price_{t-1} of that same contract (exact, no adjustment needed)
      open, high, low and close of day t all belong to one contract, so rolls create no fake gaps.
    Adjusted levels are rebuilt from those returns (ratio back-adjustment), anchored so the latest
    value equals the latest real price. Returns are what you trade; levels are only for features.

    price_field: on DCE, settlement is the day's VWAP, so 'close' is closer to a tradeable price.
    """
    dates = roll_schedule.index
    wide_price = contracts_raw.pivot(index="date", columns="contract", values=price_field).reindex(dates)
    bar_contract = roll_schedule["held_after_close"].shift(1)

    continuous = pd.DataFrame(index=dates)
    continuous["bar_contract"] = bar_contract
    continuous["held_after_close"] = roll_schedule["held_after_close"]
    continuous["roll_day"] = roll_schedule["roll_day"]

    price_today = _lookup(wide_price, bar_contract)
    price_yesterday = _lookup(wide_price.shift(1), bar_contract)
    continuous["price_actual"] = price_today
    continuous["prev_close_same_contract"] = price_yesterday
    continuous["daily_return"] = price_today / price_yesterday - 1

    # Ratio back-adjustment: compound the exact returns, then anchor to the latest real price.
    # A missing return is treated as zero here; check those dates rather than trusting the fill.
    growth = (1 + continuous["daily_return"].fillna(0)).cumprod()
    anchor_date = price_today.last_valid_index()
    continuous["close"] = growth * price_today[anchor_date] / growth[anchor_date]

    # Scale the same contract's open/high/low by the day's adjustment factor.
    adjustment_factor = continuous["close"] / price_today
    for field in ("open", "high", "low"):
        if field in contracts_raw.columns:
            wide_field = contracts_raw.pivot(index="date", columns="contract", values=field).reindex(dates)
            continuous[field] = _lookup(wide_field, bar_contract) * adjustment_factor

    # Volume and OI across ALL listed months, so volume alphas do not fire on roll migration.
    totals = contracts_raw.groupby("date")[["volume", "open_interest"]].sum().reindex(dates)
    continuous["volume"] = totals["volume"]
    continuous["open_interest"] = totals["open_interest"]

    # Roll days feed alpha_backtest's event filter; OR in USDA or limit days as you add them.
    continuous["is_event_day"] = continuous["roll_day"]

    return continuous.iloc[1:]  # the first date has no previous holding, so no bar


# ---------------------------------------------------------------------------
# Curve and spreads
# ---------------------------------------------------------------------------

def build_curve(contracts_raw, contract_meta, continuous, allowed_months=(1, 5, 9), depth=3, price_field="close"):
    """M1..M{depth} per date, from actual contracts.

    M1 is the contract the continuous series held during that day; M2, M3 are the next main
    contracts by delivery. carry = annualised log(far / near): positive is contango.
    M3 will be NaN wherever a contract's history starts too late (your current DCE file).
    """
    data = contracts_raw.join(contract_meta, on="contract")
    data = data[data["delivery_month"].isin(allowed_months)]
    contracts_by_date = {date: rows.set_index("contract") for date, rows in data.groupby("date")}

    rows = []
    for date, front_contract in continuous["bar_contract"].dropna().items():
        today = contracts_by_date.get(date)
        if today is None:
            continue
        front_delivery = contract_meta.loc[front_contract, "delivery_month_start"]
        curve_today = (today[today["delivery_month_start"] >= front_delivery]
                       .sort_values("delivery_month_start")
                       .head(depth))

        row = {"date": date}
        for position, (contract, values) in enumerate(curve_today.iterrows(), start=1):
            row[f"m{position}_contract"] = contract
            row[f"m{position}_price"] = values[price_field]
            row[f"m{position}_delivery"] = values["delivery_month_start"]
        rows.append(row)

    curve = pd.DataFrame(rows).set_index("date")

    for near, far in [(position, position + 1) for position in range(1, depth)]:
        near_price, far_price = f"m{near}_price", f"m{far}_price"
        if near_price not in curve.columns or far_price not in curve.columns:
            continue
        years_apart = (curve[f"m{far}_delivery"] - curve[f"m{near}_delivery"]).dt.days / 365.25
        curve[f"spread_m{near}_m{far}"] = curve[near_price] - curve[far_price]
        curve[f"carry_m{near}_m{far}"] = np.log(curve[far_price] / curve[near_price]) / years_apart

    return curve


def named_spread(contracts_raw, contract_meta, near_month, far_month, price_field="close"):
    """Rebuild a named calendar spread such as 1-5 (Jan minus May) or 5-9.

    Each near-month contract is paired with the first far-month contract delivering after it.
    On each date we use the nearest pair whose near leg has not passed its roll deadline,
    so the series switches pair at the deadline (flagged in pair_change).
    """
    near_contracts = contract_meta[contract_meta["delivery_month"] == near_month]
    far_contracts = contract_meta[contract_meta["delivery_month"] == far_month]

    pairs = []
    for near_contract, near_row in near_contracts.iterrows():
        later_far = far_contracts[far_contracts["delivery_month_start"] > near_row["delivery_month_start"]]
        if not later_far.empty:
            far_contract = later_far["delivery_month_start"].idxmin()
            pairs.append((near_contract, far_contract, near_row["roll_deadline"]))
    pairs.sort(key=lambda pair: pair[2])  # nearest deadline first

    wide = contracts_raw.pivot(index="date", columns="contract", values=price_field)
    rows = []
    for date, prices in wide.iterrows():
        for near_contract, far_contract, deadline in pairs:
            if deadline < date:
                continue
            near_price, far_price = prices.get(near_contract), prices.get(far_contract)
            if pd.notna(near_price) and pd.notna(far_price):
                rows.append({"date": date, "near_contract": near_contract, "far_contract": far_contract,
                             "spread": near_price - far_price})
            break  # only the nearest live pair is used on each date

    spread = pd.DataFrame(rows).set_index("date")
    spread["pair_change"] = spread["near_contract"] != spread["near_contract"].shift(1)
    spread.iloc[0, spread.columns.get_loc("pair_change")] = False
    return spread


def reconcile_spreads(built_spread, provided_spread, tolerance=1e-6):
    """Dates where a built spread disagrees with one of your spread files (both indexed by date).

    Disagreements usually sit at roll boundaries (different pair-switch dates) or show a
    sign convention difference (near minus far vs far minus near). Anything else is a data error.
    """
    joined = pd.DataFrame({"built": built_spread, "provided": provided_spread}).dropna()
    joined["difference"] = joined["built"] - joined["provided"]
    return joined[joined["difference"].abs() > tolerance]


def to_backtest_panel(continuous_by_market):
    """{'M': continuous_meal, 'Y': continuous_oil} -> (date, market) panel for alpha_backtest.py."""
    panel = pd.concat(continuous_by_market, names=["market", "date"])
    return panel.swaplevel().sort_index()


# ---------------------------------------------------------------------------
# Known-answer test
# ---------------------------------------------------------------------------

def _self_test():
    """Two contracts with known growth rates: May grows 1% a day, Sep grows 2% a day.

    Sep's OI overtakes May's at the close of day 4, so the roll executes at the close of day 5.
    Expected: returns of 1% through day 5 and 2% from day 6, no fake gap at the roll,
    and the adjusted close's percentage changes equal the exact returns.
    """
    dates = pd.bdate_range("2024-01-02", periods=10)
    days = np.arange(len(dates))
    may = pd.DataFrame({"date": dates, "contract": "M2405", "close": 3000 * 1.01 ** days,
                        "volume": 1000, "open_interest": np.where(days < 4, 500, 100)})
    sep = pd.DataFrame({"date": dates, "contract": "M2409", "close": 3200 * 1.02 ** days,
                        "volume": 1000, "open_interest": np.where(days < 4, 100, 500)})
    contracts_raw = pd.concat([may, sep], ignore_index=True)

    meta = build_contract_meta(contracts_raw)
    schedule = build_roll_schedule(contracts_raw, meta)
    continuous = build_continuous(contracts_raw, schedule)

    returns = continuous["daily_return"].to_numpy()
    assert np.allclose(returns[:5], 0.01), "pre-roll returns should be May's 1%"
    assert np.allclose(returns[5:], 0.02), "post-roll returns should be Sep's 2%"
    assert schedule["roll_day"].sum() == 1 and schedule["roll_day"].iloc[5], "one roll, at the close of day 5"
    assert np.allclose(continuous["close"].pct_change().dropna(), continuous["daily_return"].iloc[1:]), \
        "adjusted close must reproduce the exact returns"
    print("self test passed")


if __name__ == "__main__":
    _self_test()

    # Example flow for one market:
    # raw = load_contract_files("data/dce/meal", column_map={"oi": "open_interest"})
    # meta = build_contract_meta(raw)
    # schedule = build_roll_schedule(raw, meta, allowed_months=(1, 5, 9))
    # meal = build_continuous(raw, schedule, price_field="close")
    # curve = build_curve(raw, meta, meal)
    # spread_1_5 = named_spread(raw, meta, near_month=1, far_month=5)
    # print(reconcile_spreads(spread_1_5["spread"], your_spread_file["1-5"]))
    # panel = to_backtest_panel({"M": meal})
