"""The combination model: an account that starts at $0 and buys every model's recommended stocks for a fixed order
value ($1,000 by default), with no cash limit and nothing reinvested.

A Book holds one model's recommended lists for one year, priced once. trade() runs the account through the year:
each model decides on its own schedule (back-to-back holdings of its horizon), and at a decision's next open the
account buys every stock on the model's list for the order value, then sells it at the open `horizon` sessions
later. Each order is funded with new money, so what one trade earns or loses never changes another.

- Fractional shares (the default): an order buys exactly its value at the open.
- Whole shares: the share count is the order value over the list price, rounded down, so a stock priced above the
  order value is skipped.

The account's capital is the most its open orders cost at one time, and its return is its profit over that
capital. Costs are cost_bps on the value traded, netted per ticker at each open: a stock one lot sells and another
buys pays only on the difference. A lot is valued as Panel.forward_returns values a holding: splits, reinvested
dividends, large payouts held as cash, renames, and a halt or delisting at its last close.

holdings() summarizes one model's ledger per holding period: what its orders cost, their profit before and after
costs, and the turnover at its entry.

The references trade through the same ledger, with fractional shares:
- mirror(): every order the account filled, spent on one ticker (SPY) instead, on the same dates;
- basket(): the same dollars spread equally over the tradable universe of the order's decision day;
- hold(): one ticker bought with the account's capital at its first entry and held to its last exit.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd
from .panel import Panel
from .protocol import Period

ORDER_COLUMNS = ["model", "ticker", "decision_date", "entry_date", "exit_date", "price", "order_value", "shares",
                 "fill", "cost", "exit_value", "gross_profit", "reason"]
HOLDING_COLUMNS = ["decision_date", "entry_date", "exit_date", "listed", "filled", "invested", "gross_profit", "fees",
                   "net_profit", "net_return", "turnover"]


@dataclass(frozen=True)
class Book:
    """Lots to trade through one year, priced: one model's recommended stocks, or a reference's."""
    model: str
    horizon: int  # Sessions from a lot's entry open to its exit open.
    start: int  # Session of the schedule's first entry, and of its last exit.
    end: int
    decision: np.ndarray  # Per lot: the decision's session; the lot is entered at the next open.
    ticker: np.ndarray
    price: np.ndarray  # The list price the model gave, its decision-day close; NaN for a reference.
    basis: np.ndarray  # The list price on the entry day's share basis, which differs across a split at the entry.
    fill: np.ndarray  # The entry open as traded; NaN when the stock does not trade then.
    path: np.ndarray  # lots x (horizon + 1): one share's value at each open from the entry to the exit.
    amount: np.ndarray  # Per lot: the dollars to spend on it.


@dataclass(frozen=True)
class Ledger:
    """An account through one year: at every open from its first entry to its last exit, what its open lots cost,
    its profit so far after costs, the costs paid there and the dollars traded there (netted per ticker); and, when
    recorded, one row per lot (ORDER_COLUMNS)."""
    invested: pd.Series
    profit: pd.Series
    orders: pd.DataFrame | None = None
    fees: pd.Series | None = None
    traded: pd.Series | None = None

    @property
    def capital(self) -> float:
        """The most the open lots cost at one time: the money the account needed."""
        return float(self.invested.max()) if len(self.invested) else 0.0

    @property
    def path(self) -> pd.Series:
        """The cumulative return on capital at every open; flat when the account never bought anything."""
        capital = self.capital
        return (self.profit / capital if capital > 0 else self.profit * 0.0).rename("return")

    @property
    def yearly_return(self) -> float:
        """The final profit over the capital; NaN when the account never bought anything."""
        capital = self.capital
        return float(self.profit.iloc[-1] / capital) if capital > 0 and len(self.profit) else np.nan


def _priced(panel: Panel, decision: np.ndarray, tickers: np.ndarray, horizon: int):
    """Return each lot's panel column, its fill (NaN when the stock does not trade at the open after its decision),
    and one share's value at every open from that entry to the exit `horizon` opens later."""
    codes = panel.codes(tickers)
    known = codes >= 0
    safe = np.where(known, codes, 0)
    entry = decision + 1
    shares = panel.growth[entry, safe]  # A share bought at the entry, on the basis that counts splits and reinvestment.
    opens = entry[:, None] + np.arange(horizon + 1)
    received = panel.paid_out[opens, safe[:, None]] - panel.paid_out[entry, safe][:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        fill = np.where(known & panel.traded[entry, safe], panel.q_open[entry, safe] / shares, np.nan)
        path = (panel.mark_open[opens, safe[:, None]] + received) / shares[:, None]
    return safe, fill, path


def book(panel: Panel, model: str, periods: list[Period], lists: list[pd.DataFrame], order_value: float) -> Book:
    """Price a model's lists (ticker and price, one per holding period), each stock to be bought for order_value."""
    horizons = {period.holding for period in periods}
    if len(horizons) != 1 or len(lists) != len(periods):
        raise ValueError(f"{model}: a book needs one list per period, all of one holding length; got "
                         f"{len(lists)} lists, {len(periods)} periods, holdings {sorted(horizons)}.")
    horizon = horizons.pop()
    decision = np.repeat([period.decision for period in periods], [len(chosen) for chosen in lists]).astype(int)
    tickers = np.concatenate([chosen.ticker.to_numpy(dtype=object) for chosen in lists])
    price = np.concatenate([chosen.price.to_numpy(dtype=float) for chosen in lists])
    safe, fill, path = _priced(panel, decision, tickers, horizon)
    basis = price * panel.multiplier[decision, safe] / panel.multiplier[decision + 1, safe]
    return Book(model, horizon, periods[0].entry, periods[-1].exit, decision, tickers, price, basis, fill, path,
                np.full(len(decision), float(order_value)))


def _bought(item: Book, whole_shares: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return the shares each lot buys and what they cost: its amount at the fill, or with whole shares its amount
    over the list price rounded down; nothing when the stock does not trade at the entry."""
    filled = np.isfinite(item.fill)
    with np.errstate(divide="ignore", invalid="ignore"):
        wanted = np.floor(item.amount / item.basis) if whole_shares else item.amount / item.fill
    shares = np.where(filled & np.isfinite(wanted), np.maximum(wanted, 0.0), 0.0)
    cost = shares * np.where(filled, item.fill, 0.0) if whole_shares else np.where(shares > 0, item.amount, 0.0)
    return shares, cost


def spent(item: Book, whole_shares: bool = False) -> np.ndarray:
    """Return the dollars each lot of item costs at its entry: 0 when it is not bought."""
    return _bought(item, whole_shares)[1]


def trade(sessions: pd.DatetimeIndex, books: list[Book], cost_bps: float, whole_shares: bool = False,
          record: bool = False) -> Ledger:
    """Run the account through its books: each lot bought for its amount at its entry open, sold at its exit."""
    rate = cost_bps / 1e4
    start = min(item.start for item in books)
    end = max(item.end for item in books)
    span = end - start + 1
    invested = np.zeros(span)  # What the lots open at each open cost, after that open's exits and entries.
    held = np.zeros(span)  # Their value at that open.
    realized = np.zeros(span)  # Profit of the lots sold at each open.
    bought: dict[int, dict] = {}  # Open -> ticker -> dollars bought, and sold, there: costs are netted per ticker.
    sold: dict[int, dict] = {}
    rows = []
    for item in books:
        h = item.horizon
        filled = np.isfinite(item.fill)
        shares, cost = _bought(item, whole_shares)
        values = shares[:, None] * np.where(filled[:, None], item.path, 0.0)  # An unfilled stock may have no marks.
        proceeds = values[:, h]
        entry = item.decision + 1
        for k in range(h):
            np.add.at(invested, entry + k - start, cost)
            np.add.at(held, entry + k - start, values[:, k])
        np.add.at(realized, entry + h - start, proceeds - cost)
        for session, ticker, spend, back in zip(entry, item.ticker, cost, proceeds):
            if spend > 0:
                into = bought.setdefault(int(session), {})
                into[ticker] = into.get(ticker, 0.0) + spend
                out = sold.setdefault(int(session) + h, {})
                out[ticker] = out.get(ticker, 0.0) + back
        if record:
            reasons = np.where(~filled, "no_entry", np.where(shares == 0, "price_above_order", "filled"))
            rows.append(pd.DataFrame({
                "model": item.model, "ticker": item.ticker, "decision_date": sessions[item.decision],
                "entry_date": sessions[entry], "exit_date": sessions[entry + h], "price": item.price,
                "order_value": item.amount, "shares": shares, "fill": item.fill, "cost": cost,
                "exit_value": proceeds, "gross_profit": proceeds - cost, "reason": reasons.astype(object)}))
    traded = np.zeros(span)
    for session in set(bought) | set(sold):
        into, out = bought.get(session, {}), sold.get(session, {})
        traded[session - start] = sum(abs(into.get(ticker, 0.0) - out.get(ticker, 0.0))
                                      for ticker in set(into) | set(out))
    fees = rate * traded
    index = sessions[start:end + 1]
    profit = held - invested + np.cumsum(realized) - np.cumsum(fees)
    orders = None
    if record:
        orders = pd.concat(rows, ignore_index=True)[ORDER_COLUMNS] if rows else pd.DataFrame(columns=ORDER_COLUMNS)
    return Ledger(pd.Series(invested, index=index, name="invested"), pd.Series(profit, index=index, name="profit"),
                  orders, pd.Series(fees, index=index, name="fees"), pd.Series(traded, index=index, name="traded"))


def holdings(ledger: Ledger, periods: list[Period], sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Return one row per holding period of a recorded one-model ledger (HOLDING_COLUMNS): its orders, what they
    cost, their profit before and after costs, and the turnover at its entry, the dollars traded there over the
    dollars invested. The costs at an open go to the last holding entered at or before it: where back-to-back
    holdings share an open, to the one entered there; the last exit's to the last holding."""
    orders = ledger.orders
    entries = np.array([period.entry for period in periods])
    owner = np.searchsorted(entries, sessions.get_indexer(ledger.fees.index), side="right") - 1
    charged = np.bincount(owner[owner >= 0], weights=ledger.fees.to_numpy()[owner >= 0], minlength=len(periods))
    rows = []
    for k, period in enumerate(periods):
        decision, entry, exit_ = sessions[period.decision], sessions[period.entry], sessions[period.exit]
        mine = orders[orders.decision_date == decision]
        fees = float(charged[k])
        invested, gross = float(mine.cost.sum()), float(mine.gross_profit.sum())
        rows.append({"decision_date": decision, "entry_date": entry, "exit_date": exit_, "listed": len(mine),
                     "filled": int((mine.reason == "filled").sum()), "invested": invested, "gross_profit": gross,
                     "fees": fees, "net_profit": gross - fees,
                     "net_return": (gross - fees) / invested if invested > 0 else np.nan,
                     "turnover": float(ledger.traded.get(entry, 0.0)) / invested if invested > 0 else np.nan})
    return pd.DataFrame(rows, columns=HOLDING_COLUMNS)


def _reference(item: Book, name: str, decision: np.ndarray, tickers: np.ndarray, fill: np.ndarray,
               path: np.ndarray, amount: np.ndarray) -> Book:
    nan = np.full(len(decision), np.nan)
    return Book(name, item.horizon, item.start, item.end, decision, tickers, nan, nan, fill, path, amount)


def mirror(panel: Panel, item: Book, amounts: np.ndarray, ticker: str = "SPY") -> Book:
    """Return a book that spends on `ticker` what each lot of item cost (amounts), on the lot's dates."""
    keep = amounts > 0
    decision = item.decision[keep]
    tickers = np.full(len(decision), ticker, dtype=object)
    _, fill, path = _priced(panel, decision, tickers, item.horizon)
    return _reference(item, f"{item.model} as {ticker}", decision, tickers, fill, path, amounts[keep])


def basket(panel: Panel, item: Book, amounts: np.ndarray, universe: Callable[[int], pd.Index]) -> Book:
    """Return a book that spreads what item's lots cost at each decision (amounts) equally over that day's
    universe: over the stocks of it that trade at the entry."""
    totals = pd.Series(amounts).groupby(item.decision).sum()
    totals = totals[totals > 0]
    names = [np.asarray(universe(int(day)), dtype=object) for day in totals.index]
    decision = np.repeat(totals.index.to_numpy(dtype=int), [len(day) for day in names])
    tickers = np.concatenate(names) if names else np.zeros(0, dtype=object)
    _, fill, path = _priced(panel, decision, tickers, item.horizon)
    filled = pd.Series(np.isfinite(fill))
    counts = filled.groupby(decision).transform("sum").to_numpy(dtype=float)
    amount = np.where(filled, totals.reindex(decision).to_numpy() / np.maximum(counts, 1.0), 0.0)
    return _reference(item, f"{item.model} as universe", decision, tickers, fill, path, amount)


def hold(panel: Panel, ticker: str, start: int, end: int, amount: float) -> Book:
    """Return a book that buys `ticker` for `amount` at the open of session start and sells it at session end."""
    decision = np.array([start - 1])
    tickers = np.array([ticker], dtype=object)
    _, fill, path = _priced(panel, decision, tickers, end - start)
    nan = np.full(1, np.nan)
    return Book(f"hold {ticker}", end - start, start, end, decision, tickers, nan, nan, fill, path,
                np.array([float(amount)]))
