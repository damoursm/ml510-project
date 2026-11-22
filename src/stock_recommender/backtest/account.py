"""The combination model: an account that starts each test year with `capital` in cash ($10,000 by default) and buys
the stocks the models it trades recommend, in slices of the order value ($100 by default), at the 16:00 close of the
day they are recommended, while its cash lasts. Which models it trades each year is chosen in run.py
(choose_combination); a year where none qualifies gets an idle_ledger.

A Book holds one model's recommended lists for one year, priced once. trade() runs the account through the year,
one close (16:00) at a time:
- it first sells the lots that are due: at the close hold_days sessions after their entry, or earlier at the close
  where the model's exit rule fires (exits.py); their proceeds return to the cash, for later orders;
- it then buys the day's lists, the best-ranked stocks first and the models in turn (each one's first stock, then
  each one's second...), one slice each while the cash covers a slice; a stock it cannot pay for is skipped
  (no_cash);
- costs are cost_bps on the value traded, netted per ticker at each close (a stock one lot sells and another buys
  pays only on the difference), paid from the cash.
The account's equity at a close is its cash plus what its open lots are worth there; its yearly return is its last
equity over its starting capital, minus 1. Every test year starts afresh with the starting capital. Every order
records its excursions: the best and worst close return while held.

- Fractional shares (the default): an order buys exactly its slice at the close.
- Whole shares: the share count is the slice over the price, rounded down, so a stock priced above the slice is
  skipped (price_above_order).

A lot is valued as Panel.forward_returns values a holding: splits, reinvested dividends, large payouts held as cash,
renames, and a halt or delisting at its last close.

holdings() summarizes one model's ledger per decision: what its orders cost, their profit before and after costs,
and the turnover at its entry.

The references trade through the same ledger, from the same capital, with fractional shares. mirror() and basket()
spend the account's dollars on its dates, so they buy every one of its orders whatever their own cash
(trade(cash_limit=False)):
- mirror(): every order the account filled, spent on one ticker (SPY) instead, on the same dates, early exits
  included; run.choose_combination also compares each model's trades with its own mirror;
- basket(): the same dollars spread equally over the tradable universe of the order's decision day, held to the
  end of the period whatever the exit rule;
- hold(): one ticker bought with the whole capital at the account's first entry and held to its last exit.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd
from ..exits import VOLATILITY_SESSIONS, ExitRule, excursions, exit_points
from .panel import Panel
from .protocol import Period

CAPITAL = 10_000.0  # The cash a test year starts with.
# exit_rule names the rule that sold a lot early ('' at the scheduled exit); mfe and mae are its excursions.
ORDER_COLUMNS = ["model", "ticker", "decision_date", "entry_date", "exit_date", "price", "order_value", "shares",
                 "fill", "cost", "exit_value", "gross_profit", "reason", "exit_rule", "mfe", "mae"]
HOLDING_COLUMNS = ["decision_date", "entry_date", "exit_date", "listed", "filled", "invested", "gross_profit", "fees",
                   "net_profit", "net_return", "turnover"]


@dataclass(frozen=True)
class Book:
    """Lots to trade through one year, priced: one model's recommended stocks, or a reference's."""
    model: str
    horizon: int  # Sessions from a lot's entry close to its scheduled exit close.
    start: int  # Session of the schedule's first entry, and of its last exit.
    end: int
    decision: np.ndarray  # Per lot: the session it is recommended and bought at, at the close.
    ticker: np.ndarray
    price: np.ndarray  # The list price the model gave: the decision day's close; NaN for a reference.
    basis: np.ndarray  # The price whole shares are counted from.
    fill: np.ndarray  # The close it is bought at; NaN when the stock did not trade that day.
    path: np.ndarray  # lots x (horizon + 1): one share's value at each close from the entry to the scheduled exit.
    amount: np.ndarray  # Per lot: the dollars to spend on it.
    held: np.ndarray | None = None  # Per lot: closes from its entry to its sale, at most horizon; None: horizon.
    why: np.ndarray | None = None  # Per lot: the exit rule that sold it early, or ''.
    mfe: np.ndarray | None = None  # Per lot: its maximum favorable and adverse excursions (exits.excursions).
    mae: np.ndarray | None = None
    rank: np.ndarray | None = None  # Per lot: its place on its day's list, 0 for the best; None: the book's order.

    def sold_after(self) -> np.ndarray:
        """Per lot, the closes from its entry to its sale."""
        return self.held if self.held is not None else np.full(len(self.decision), self.horizon, dtype=int)

    def ranks(self) -> np.ndarray:
        return self.rank if self.rank is not None else np.arange(len(self.decision))


@dataclass(frozen=True)
class Ledger:
    """An account through one year: at every close from its first entry to its last exit, what its open lots cost,
    its profit so far after costs (its equity minus its starting capital), the costs paid there and the dollars
    traded there (netted per ticker); when recorded, one row per lot (ORDER_COLUMNS); and per book, what each of
    its lots cost (0 when it was not bought)."""
    invested: pd.Series
    profit: pd.Series
    orders: pd.DataFrame | None = None
    fees: pd.Series | None = None
    traded: pd.Series | None = None
    idle: bool = False  # It chose to trade nothing (idle_ledger): it kept its cash, a return of 0.
    capital: float = CAPITAL  # The cash it started with.
    filled: int = 0  # The orders it bought.
    spent: tuple = ()  # Per book: the dollars each lot cost at its entry, 0 when not bought.

    @property
    def path(self) -> pd.Series:
        """The cumulative return on the starting capital at every close: equity over capital, minus 1."""
        return (self.profit / self.capital).rename("return")

    @property
    def yearly_return(self) -> float:
        """The last equity over the starting capital, minus 1; 0 for an idle account, NaN for one that could not buy
        anything (no list, or no stock that traded)."""
        if self.idle:
            return 0.0
        return float(self.profit.iloc[-1] / self.capital) if self.filled and len(self.profit) else np.nan


def idle_ledger(sessions: pd.DatetimeIndex, start: int, end: int, capital: float = CAPITAL) -> Ledger:
    """Return the ledger of an account that trades nothing from session start to session end, as when no model
    qualifies for the year: no order, its cash kept, a return of 0."""
    index = sessions[start:end + 1]
    zeros = {name: pd.Series(0.0, index=index, name=name) for name in ("invested", "profit", "fees", "traded")}
    return Ledger(zeros["invested"], zeros["profit"], pd.DataFrame(columns=ORDER_COLUMNS), zeros["fees"],
                  zeros["traded"], idle=True, capital=capital)


def _priced(panel: Panel, decision: np.ndarray, tickers: np.ndarray, horizon: int):
    """Return each lot's panel column, its fill (the decision day's close; NaN when the stock did not trade that
    day), and one share's value at every close from that entry to the one `horizon` sessions later."""
    codes = panel.codes(tickers)
    known = codes >= 0
    safe = np.where(known, codes, 0)
    shares = panel.growth[decision, safe]  # A share bought at the entry, on the basis that counts splits and reinvestment.
    closes = decision[:, None] + np.arange(horizon + 1)
    received = panel.paid_out[closes, safe[:, None]] - panel.paid_out[decision, safe][:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        fill = np.where(known & panel.traded[decision, safe], panel.mark_close[decision, safe] / shares, np.nan)
        path = (panel.mark_close[closes, safe[:, None]] + received) / shares[:, None]
    return safe, fill, path


def _marks(panel: Panel, decision: np.ndarray, safe: np.ndarray, fill: np.ndarray, path: np.ndarray):
    """Return what an exit rule reads, per lot: the return from its entry price and the move from the close before,
    at each close after its entry (lots x horizon), and its daily volatility over the VOLATILITY_SESSIONS moves
    through its decision day. A lot never filled has NaN returns."""
    earlier = decision[:, None] + np.arange(-VOLATILITY_SESSIONS, 1)  # The decision day's close last.
    rows, column = np.maximum(earlier, 0), safe[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = path[:, 1:] / fill[:, None] - 1
        moves = path[:, 1:] / path[:, :-1] - 1
        before = np.where(earlier >= 0, panel.mark_close[rows, column] + panel.paid_out[rows, column], np.nan)
        volatility = (before[:, 1:] / before[:, :-1] - 1).std(axis=1, ddof=1) if len(decision) else np.zeros(0)
    return returns, moves, volatility


def book(panel: Panel, model: str, periods: list[Period], lists: list[pd.DataFrame], order_value: float,
         rule: ExitRule | None = None) -> Book:
    """Price a model's lists (ticker and price, one per decision, best first), each stock to be bought for
    order_value at its decision's close and sold at the end of its holding, or earlier when the exit rule fires."""
    horizons = {period.holding for period in periods}
    if len(horizons) != 1 or len(lists) != len(periods):
        raise ValueError(f"{model}: a book needs one list per period, all of one holding length; got "
                         f"{len(lists)} lists, {len(periods)} periods, holdings {sorted(horizons)}.")
    horizon = horizons.pop()
    if rule is not None and rule.hold_days != horizon:
        raise ValueError(f"{model}: its exit rule holds {rule.hold_days} days, but its periods last {horizon}.")
    decision = np.repeat([period.decision for period in periods], [len(chosen) for chosen in lists]).astype(int)
    tickers = np.concatenate([chosen.ticker.to_numpy(dtype=object) for chosen in lists])
    price = np.concatenate([chosen.price.to_numpy(dtype=float) for chosen in lists])
    rank = np.concatenate([np.arange(len(chosen)) for chosen in lists]).astype(int)
    safe, fill, path = _priced(panel, decision, tickers, horizon)
    returns, moves, volatility = _marks(panel, decision, safe, fill, path)
    held, why = exit_points(rule, returns, moves, volatility) if rule is not None and rule.early else (None, None)
    mfe, mae = excursions(returns, held if held is not None else np.full(len(decision), horizon))
    return Book(model, horizon, periods[0].entry, periods[-1].exit, decision, tickers, price, price.copy(), fill, path,
                np.full(len(decision), float(order_value)), held, why, mfe, mae, rank)


def _bought(item: Book, whole_shares: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return the shares each lot would buy and what they would cost: its amount at the fill, or with whole shares
    its amount over the price rounded down; nothing when the stock did not trade at the entry."""
    filled = np.isfinite(item.fill)
    with np.errstate(divide="ignore", invalid="ignore"):
        wanted = np.floor(item.amount / item.basis) if whole_shares else item.amount / item.fill
    shares = np.where(filled & np.isfinite(wanted), np.maximum(wanted, 0.0), 0.0)
    cost = shares * np.where(filled, item.fill, 0.0) if whole_shares else np.where(shares > 0, item.amount, 0.0)
    return shares, cost


def trade(sessions: pd.DatetimeIndex, books: list[Book], cost_bps: float, whole_shares: bool = False,
          record: bool = False, capital: float = CAPITAL, cash_limit: bool = True) -> Ledger:
    """Run the account through its books from `capital` in cash, one close at a time: sell the lots that are due,
    then buy the day's lots, the best-ranked first across the books, while the cash covers each (no_cash
    otherwise), and pay the costs, netted per ticker.

    Without the cash limit every lot is bought whatever the cash, which may then go below zero: as a reference that
    spends another account's dollars on its dates does. Its profit is still its equity minus its capital, which is
    what its lots earned after costs."""
    rate = cost_bps / 1e4
    start = min(item.start for item in books)
    end = max(item.end for item in books)
    span = end - start + 1
    lots, entering = [], {}  # Per book: its lots' sale, shares, cost, proceeds and values; per close, its entries.
    for b, item in enumerate(books):
        after = item.sold_after()
        shares, cost = _bought(item, whole_shares)
        values = shares[:, None] * np.where(np.isfinite(item.fill)[:, None], item.path, 0.0)  # Unfilled: no marks.
        lots.append((after, shares, cost, values[np.arange(len(after)), after], values))
        ranks = item.ranks()
        for lot in np.flatnonzero(cost > 0):
            entering.setdefault(int(item.decision[lot]), []).append((int(ranks[lot]), b, int(lot)))
    bought = [np.zeros(len(item.decision), dtype=bool) for item in books]
    short = [np.zeros(len(item.decision), dtype=bool) for item in books]
    incoming, cash_at, fees, traded = np.zeros(span), np.zeros(span), np.zeros(span), np.zeros(span)
    sold: dict[int, dict] = {}  # Close -> ticker -> dollars the lots due there bring.
    cash = float(capital)
    for session in range(start, end + 1):
        cash += incoming[session - start]  # The lots due at this close are sold first.
        into: dict = {}
        for _, b, lot in sorted(entering.get(session, ())):  # Best-ranked first, each book in turn.
            after, _, cost, proceeds, _ = lots[b]
            if cash_limit and cost[lot] > cash + 1e-9:
                short[b][lot] = True
                continue
            cash -= cost[lot]
            bought[b][lot] = True
            ticker = books[b].ticker[lot]
            into[ticker] = into.get(ticker, 0.0) + cost[lot]
            out_at = session + int(after[lot])
            incoming[out_at - start] += proceeds[lot]
            out = sold.setdefault(out_at, {})
            out[ticker] = out.get(ticker, 0.0) + proceeds[lot]
        out = sold.get(session, {})
        traded[session - start] = sum(abs(into.get(name, 0.0) - out.get(name, 0.0)) for name in set(into) | set(out))
        fees[session - start] = rate * traded[session - start]
        cash -= fees[session - start]
        cash_at[session - start] = cash
    worth, invested = np.zeros(span), np.zeros(span)  # What the open lots are worth, and cost, at each close.
    rows = []
    for b, item in enumerate(books):
        after, shares, cost, proceeds, values = lots[b]
        mine = bought[b]
        for k in range(item.horizon):
            still = mine & (after > k)
            np.add.at(worth, item.decision[still] + k - start, values[still, k])
            np.add.at(invested, item.decision[still] + k - start, cost[still])
        if record:
            reasons = np.where(~np.isfinite(item.fill), "no_entry", np.where(
                shares == 0, "price_above_order", np.where(short[b], "no_cash", "filled")))
            paid, back = np.where(mine, cost, 0.0), np.where(mine, proceeds, 0.0)
            missing = np.full(len(after), np.nan)
            rows.append(pd.DataFrame({
                "model": item.model, "ticker": item.ticker, "decision_date": sessions[item.decision],
                "entry_date": sessions[item.decision], "exit_date": sessions[item.decision + after],
                "price": item.price, "order_value": item.amount, "shares": np.where(mine, shares, 0.0),
                "fill": item.fill, "cost": paid, "exit_value": back, "gross_profit": back - paid,
                "reason": reasons.astype(object),
                "exit_rule": item.why if item.why is not None else np.full(len(after), "", dtype=object),
                "mfe": item.mfe if item.mfe is not None else missing,
                "mae": item.mae if item.mae is not None else missing}))
    index = sessions[start:end + 1]
    orders = None
    if record:
        orders = pd.concat(rows, ignore_index=True)[ORDER_COLUMNS] if rows else pd.DataFrame(columns=ORDER_COLUMNS)
    return Ledger(pd.Series(invested, index=index, name="invested"),
                  pd.Series(cash_at + worth - capital, index=index, name="profit"), orders,
                  pd.Series(fees, index=index, name="fees"), pd.Series(traded, index=index, name="traded"),
                  capital=float(capital), filled=int(sum(mine.sum() for mine in bought)),
                  spent=tuple(np.where(mine, lot[2], 0.0) for mine, lot in zip(bought, lots)))


def holdings(ledger: Ledger, periods: list[Period], sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Return one row per decision of a recorded one-model ledger (HOLDING_COLUMNS): its orders, what they cost,
    their profit before and after costs, and the turnover at its entry, the dollars traded there over the dollars
    invested. The costs at a close go to the last decision made at or before it; the last exit's to the last."""
    orders = ledger.orders
    entries = np.array([period.entry for period in periods])
    owner = np.searchsorted(entries, sessions.get_indexer(ledger.fees.index), side="right") - 1
    charged = np.bincount(owner[owner >= 0], weights=ledger.fees.to_numpy()[owner >= 0], minlength=len(periods))
    by_decision = orders.groupby("decision_date") if len(orders) else None
    rows = []
    for k, period in enumerate(periods):
        decision, entry, exit_ = sessions[period.decision], sessions[period.entry], sessions[period.exit]
        mine = by_decision.get_group(decision) if by_decision is not None and decision in by_decision.groups \
            else orders.iloc[:0]
        fees = float(charged[k])
        invested, gross = float(mine.cost.sum()), float(mine.gross_profit.sum())
        rows.append({"decision_date": decision, "entry_date": entry, "exit_date": exit_, "listed": len(mine),
                     "filled": int((mine.reason == "filled").sum()), "invested": invested, "gross_profit": gross,
                     "fees": fees, "net_profit": gross - fees,
                     "net_return": (gross - fees) / invested if invested > 0 else np.nan,
                     "turnover": float(ledger.traded.get(entry, 0.0)) / invested if invested > 0 else np.nan})
    return pd.DataFrame(rows, columns=HOLDING_COLUMNS)


def _reference(item: Book, name: str, decision: np.ndarray, tickers: np.ndarray, fill: np.ndarray,
               path: np.ndarray, amount: np.ndarray, held: np.ndarray | None = None) -> Book:
    nan = np.full(len(decision), np.nan)
    return Book(name, item.horizon, item.start, item.end, decision, tickers, nan, fill.copy(), fill, path, amount,
                held)


def mirror(panel: Panel, item: Book, amounts: np.ndarray, ticker: str = "SPY") -> Book:
    """Return a book that spends on `ticker` what each lot of item cost (amounts, 0 for the lots not bought), on
    the lot's dates: sold when the lot was, early exits included."""
    keep = amounts > 0
    decision = item.decision[keep]
    tickers = np.full(len(decision), ticker, dtype=object)
    _, fill, path = _priced(panel, decision, tickers, item.horizon)
    return _reference(item, f"{item.model} as {ticker}", decision, tickers, fill, path, amounts[keep],
                      item.held[keep] if item.held is not None else None)


def basket(panel: Panel, item: Book, amounts: np.ndarray, universe: Callable[[int], pd.Index]) -> Book:
    """Return a book that spreads what item's lots cost at each decision (amounts) equally over that day's
    universe: over the stocks of it that trade that day."""
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
    """Return a book that buys `ticker` for `amount` at the close of session start and sells it at the close of
    session end."""
    decision = np.array([start])
    tickers = np.array([ticker], dtype=object)
    _, fill, path = _priced(panel, decision, tickers, end - start)
    nan = np.full(1, np.nan)
    return Book(f"hold {ticker}", end - start, start, end, decision, tickers, nan, fill.copy(), fill, path,
                np.array([float(amount)]))
