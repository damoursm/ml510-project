"""Trade every model's recommendations in one account, and tune each model's order size.

A Book holds one model's recommended lists for one year, priced once. trade() runs the account through the
year: each model decides on its own schedule (back-to-back holdings of its horizon), and at a decision's next
open the account buys every stock on the model's list for `size` of its value, then sells it at the open
`horizon` sessions later. The value is the account's at the open of the decision day, the last one known when
the order is placed.

- Fractional shares (the default): an order buys exactly its value at the open.
- Whole shares: the share count is the order's value over the list price, rounded down, so a stock priced
  above its order is skipped.

Costs are cost_bps on the value traded, netted per ticker at each open: a stock one lot sells and another buys
pays only on the difference. Orders that need more cash than the account holds are scaled down together; cash
never goes negative, so there is no leverage. A lot is valued as Panel.forward_returns values a holding: splits,
reinvested dividends, large payouts held as cash, renames, and a halt or delisting at its last close.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd
from .metrics import log_growth
from .panel import Panel
from .protocol import Period

ORDER_COLUMNS = ["model", "ticker", "decision_date", "entry_date", "exit_date", "price", "size", "order_value",
                 "shares_wanted", "shares", "fill", "cost", "exit_value", "reason"]


@dataclass(frozen=True)
class Book:
    """One model's recommended lists for one year, priced: everything trade() needs except the order size."""
    model: str
    horizon: int
    start: int  # Session of the first entry of the model's schedule, and of its last exit.
    end: int
    decision: np.ndarray  # Per recommended stock: the decision's session; it is entered at the next open.
    ticker: np.ndarray
    price: np.ndarray  # The list price the model gave: its decision-day close.
    basis: np.ndarray  # The list price on the entry day's share basis, which differs across a split at the entry.
    fill: np.ndarray  # The entry open as traded; NaN when the stock does not trade then.
    path: np.ndarray  # stocks x (horizon + 1): one share's value at each open from the entry to the exit.


def book(panel: Panel, model: str, periods: list[Period], lists: list[pd.DataFrame]) -> Book:
    """Price a model's lists (ticker and price, one per holding period) for trade()."""
    horizons = {period.holding for period in periods}
    if len(horizons) != 1:
        raise ValueError(f"{model}: a book needs periods of one holding length; got {sorted(horizons)}.")
    horizon = horizons.pop()
    parts = {key: [] for key in ("decision", "ticker", "price", "basis", "fill", "path")}
    for period, chosen in zip(periods, lists, strict=True):
        if chosen.empty:
            continue
        d, e, x = period.decision, period.entry, period.exit
        codes = panel.codes(chosen.ticker)
        known = codes >= 0
        safe = np.where(known, codes, 0)
        shares = panel.growth[e, safe]  # A share bought at e, on the basis that counts splits and reinvestment.
        price = chosen.price.to_numpy(dtype=float)
        received = panel.paid_out[e:x + 1, safe] - panel.paid_out[e, safe]
        parts["decision"].append(np.full(len(chosen), d))
        parts["ticker"].append(chosen.ticker.to_numpy(dtype=object))
        parts["price"].append(price)
        parts["basis"].append(price * panel.multiplier[d, safe] / panel.multiplier[e, safe])
        parts["fill"].append(np.where(known & panel.traded[e, safe], panel.q_open[e, safe] / shares, np.nan))
        parts["path"].append(((panel.mark_open[e:x + 1, safe] + received) / shares).T)
    empty = {"decision": np.zeros(0, dtype=int), "ticker": np.zeros(0, dtype=object),
             "path": np.zeros((0, horizon + 1))}
    arrays = {key: np.concatenate(value) if value else empty.get(key, np.zeros(0)) for key, value in parts.items()}
    return Book(model, horizon, periods[0].entry, periods[-1].exit, **arrays)


def trade(sessions: pd.DatetimeIndex, books: list[Book], sizes, capital: float, cost_bps: float,
          whole_shares: bool = False, record: bool = False) -> tuple[pd.Series, pd.DataFrame | None]:
    """Run the account through its books with sizes[i] (a share of its value per order) for books[i].

    Return the value at every open from the first entry to the last exit, and with record=True one row per
    recommended stock (ORDER_COLUMNS).
    """
    rate = cost_bps / 1e4
    start = min(item.start for item in books)
    end = max(item.end for item in books)
    span = end - start + 1
    held = np.zeros(span)  # Value of the open lots at each open, before the exits there.
    inflow = np.zeros(span)  # Proceeds of the lots exiting at each open.
    sold: dict[int, dict] = {}  # Exit open -> ticker -> proceeds, for netting the costs.
    entries: dict[int, list[tuple[int, np.ndarray]]] = {}
    for index, item in enumerate(books):
        for entry in np.unique(item.decision + 1):
            entries.setdefault(int(entry), []).append((index, np.flatnonzero(item.decision + 1 == entry)))
    values = np.empty(span)
    cash = capital
    rows = []
    for s in range(start, end + 1):
        i = s - start
        cash += inflow[i]
        sales = sold.pop(s, {})
        bought: dict = {}
        if s in entries:
            worth = values[i - 1] if i > 0 else capital  # The account at the open of the decision day.
            orders = []
            for index, lots in entries[s]:
                item, size = books[index], float(sizes[index])
                fill = item.fill[lots]
                filled = np.isfinite(fill)
                target = size * worth
                if whole_shares:
                    wanted = np.where(filled, np.floor(target / item.basis[lots]), 0.0)
                else:
                    wanted = np.where(filled, target / np.where(filled, fill, 1.0), 0.0)
                orders.append((index, lots, size, target, fill, filled, np.maximum(wanted, 0.0)))
            need = sum(float(np.dot(wanted, np.where(filled, fill, 0.0)))
                       for *_, fill, filled, wanted in orders) * (1 + rate)
            available = cash - rate * sum(sales.values())
            scale = 1.0 if need <= available or need == 0 else max(available, 0.0) / need
            for index, lots, size, target, fill, filled, wanted in orders:
                item = books[index]
                shares = wanted * scale
                if whole_shares:
                    shares = np.floor(shares)
                price_fill = np.where(filled, fill, 0.0)
                cost = shares * price_fill
                cash -= float(cost.sum())
                h = item.horizon
                path = np.where(filled[:, None], item.path[lots], 0.0)  # An unfilled stock may have no marks.
                held[i:i + h] += shares @ path[:, :h]
                proceeds = shares * path[:, h]
                inflow[i + h] += float(proceeds.sum())
                exits = sold.setdefault(s + h, {})
                for ticker, spend, back in zip(item.ticker[lots], cost, proceeds):
                    if spend > 0:
                        bought[ticker] = bought.get(ticker, 0.0) + spend
                        exits[ticker] = exits.get(ticker, 0.0) + back
                if record:
                    if size == 0:
                        reasons = np.full(len(lots), "size_zero", dtype=object)
                    else:
                        reasons = np.where(~filled, "no_entry", np.where(
                            wanted == 0, "price_above_order", np.where(
                                shares == 0, "no_cash", np.where(scale < 1, "scaled", "filled")))).astype(object)
                    rows.append(pd.DataFrame({
                        "model": item.model, "ticker": item.ticker[lots],
                        "decision_date": sessions[item.decision[lots]], "entry_date": sessions[s],
                        "exit_date": sessions[s + h], "price": item.price[lots], "size": size,
                        "order_value": target, "shares_wanted": wanted, "shares": shares, "fill": fill,
                        "cost": cost, "exit_value": proceeds, "reason": reasons}))
        traded = sum(abs(bought.get(ticker, 0.0) - sales.get(ticker, 0.0)) for ticker in set(bought) | set(sales))
        cash -= rate * traded
        values[i] = cash + held[i]
    curve = pd.Series(values, index=sessions[start:end + 1], name="value")
    orders = (pd.concat(rows, ignore_index=True)[ORDER_COLUMNS] if rows else pd.DataFrame(columns=ORDER_COLUMNS)) \
        if record else None
    return curve, orders


def equal_split(picks: list[int], grid) -> tuple[float, ...]:
    """Return, per model, the largest grid size that splits the account evenly over every model's list."""
    share = 1.0 / sum(picks)
    return tuple(max([size for size in grid if size <= share + 1e-12], default=0.0) for _ in picks)


def objective(sessions: pd.DatetimeIndex, books_by_year: dict[int, list[Book]], capital: float, cost_bps: float,
              whole_shares: bool) -> Callable[[tuple[float, ...]], float]:
    """Return sizes -> the account's annualized log growth, averaged over the years of books_by_year."""
    def score(sizes) -> float:
        values = [log_growth(trade(sessions, books, sizes, capital, cost_bps, whole_shares)[0])
                  for books in books_by_year.values()]
        values = [value for value in values if not np.isnan(value)]
        return float(np.mean(values)) if values else -np.inf
    return score


def tune(score: Callable[[tuple[float, ...]], float], start: tuple[float, ...], grid,
         sweeps: int = 4) -> tuple[tuple[float, ...], pd.DataFrame]:
    """Coordinate ascent: set one model's size at a time to the grid value that scores best with the others
    fixed (the smallest of tied values), until a sweep changes nothing. Return the sizes and every trial."""
    cache: dict[tuple, float] = {}

    def value(sizes: tuple) -> float:
        if sizes not in cache:
            result = score(sizes)
            cache[sizes] = -np.inf if np.isnan(result) else result
        return cache[sizes]

    best = tuple(float(size) for size in start)
    trace = [{"sweep": 0, "model_index": None, "sizes": best, "objective": value(best)}]
    for sweep in range(1, sweeps + 1):
        changed = False
        for i in range(len(best)):
            trials = [best[:i] + (float(size),) + best[i + 1:] for size in grid]
            scores = [value(trial) for trial in trials]
            choice = trials[int(np.argmax(scores))]  # The first best: the smallest size among ties.
            trace += [{"sweep": sweep, "model_index": i, "sizes": trial, "objective": result}
                      for trial, result in zip(trials, scores)]
            if choice != best:
                best, changed = choice, True
        if not changed:
            break
    return best, pd.DataFrame(trace)
