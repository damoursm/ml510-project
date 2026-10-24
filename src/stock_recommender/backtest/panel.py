"""2021-2025 bars as traded, split-adjusted as of each decision day, and the total returns of trades between opens.

M(d) is a ticker's cumulative share multiplier: the product of split_to / split_from over its splits executed on
or before day d (2021-2025 splits only). A model deciding at day t sees prices raw(d) * M(d) / M(t), which is what
an adjusted download made on t returns: day t's close is the price as traded, and no later split shows.
D(d) reinvests dividends at the ex-day open, the product of (1 + cash / ex-day open), and the shares of curated
spin-offs. A payout of at least CHECKED_YIELD of the price (a special, a liquidation) is held as cash until the
holding's exit instead: K(d) adds it up per share. A holding's value is raw * M * D + K, so a split during a
holding changes nothing and a payout is income, not a loss. A holding across a rename continues in the new ticker.
history() reinvests every payout, the usual total-return series for features.
"""
from dataclasses import dataclass, field
from pathlib import Path
import re
import numpy as np
import pandas as pd
from ..config import Settings
from ..recommend import tradable
from . import corporate
from .window import BAR_COLUMNS, LAST_DAY, DataError, LeakageError, WindowedStore

PRICES = ["open", "high", "low", "close", "vwap"]
AUDITED_RATIO = 1.2  # Splits at least this large are checked against the prices; smaller moves are noise.
CHECKED_YIELD = 0.15  # Dividends at least this large must show in the price drop.
RELOCATE = 10  # Sessions to look ahead for the price move of a mis-dated dividend.
CONTINUE = corporate.QUIET  # Sessions a renamed holding is valued in its new ticker.
# Dollars per unit of a currency, from the CurrencyShares trust holding about 100 units per share.
FX_FUNDS = {"CAD": "FXC", "EUR": "FXE", "CHF": "FXF", "GBP": "FXB"}


def _forward_fill(values: np.ndarray) -> np.ndarray:
    """Forward-fill NaN down the rows of a 2-D array, column by column."""
    rows = np.where(np.isnan(values), 0, np.arange(len(values))[:, None])
    np.maximum.accumulate(rows, axis=0, out=rows)
    return values[rows, np.arange(values.shape[1])]


@dataclass
class Panel:
    bars: pd.DataFrame  # As traded, sorted by date then ticker; session and code columns index the matrices.
    sessions: pd.DatetimeIndex  # The trading days with a bar file, all inside 2021-2025.
    tickers: pd.Index
    day_starts: np.ndarray  # Row of each session's first bar; day_starts[-1] == len(bars).
    traded: np.ndarray  # sessions x tickers: a bar with volume and positive prices.
    multiplier: np.ndarray  # sessions x tickers: M.
    growth: np.ndarray  # sessions x tickers: M * D, the shares (with reinvested payouts) one 2021 share became.
    q_open: np.ndarray  # sessions x tickers: open * M * D where traded, else NaN.
    mark_open: np.ndarray  # What a holding is worth at each open: the open, a renamed ticker's, or the last close.
    continued: np.ndarray  # sessions x tickers: valued in the ticker it was renamed to.
    paid_out: np.ndarray  # sessions x tickers: K, large payouts received so far, on the M * D basis.
    total: np.ndarray  # sessions x tickers: M * D with every payout reinvested, for history().
    split_audit: pd.DataFrame
    dividend_audit: pd.DataFrame
    action_audit: pd.DataFrame
    renames: pd.DataFrame  # old, new, date.
    _universes: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, store: WindowedStore, strict_calendar: bool = True, dividends_path: Path | None = None,
             actions_path: Path | None = None, changes_path: Path | None = None,
             min_audited_splits: int = 20) -> "Panel":
        """Build the panel; without dividends_path, returns leave out dividends."""
        bars = store.read_bars()
        sessions = pd.DatetimeIndex(bars.date.unique())
        if sessions.max().date() > LAST_DAY:
            raise LeakageError(f"Bars dated {sessions.max().date()} reached the backtest.")
        if strict_calendar:
            expected = {2021: 252, 2022: 251, 2023: 250, 2024: 252, 2025: 250}
            counts = pd.Series(sessions.year).value_counts().to_dict()
            if counts != expected:
                raise DataError(f"Trading days per year {counts} differ from the NYSE calendar {expected}; "
                                "is the 2021-2025 download complete?")
        tickers = bars.ticker.cat.categories
        bars["session"] = sessions.get_indexer(bars.date).astype(np.int32)
        bars["code"] = bars.ticker.cat.codes.astype(np.int32)
        day_starts = np.searchsorted(bars.session.to_numpy(), np.arange(len(sessions) + 1))
        shape = (len(sessions), len(tickers))

        def matrix(values, fill=np.nan, dtype=np.float64):
            out = np.full(shape, fill, dtype=dtype)
            out[bars.session.to_numpy(), bars.code.to_numpy()] = values
            return out

        raw_open, raw_close = matrix(bars.open.to_numpy()), matrix(bars.close.to_numpy())
        present = matrix(True, False, bool)
        traded = matrix(((bars.volume > 0) & (bars.open > 0) & (bars.close > 0)).to_numpy(), False, bool)
        changes = store.read_changes(changes_path) if changes_path is not None else None
        renames = corporate.with_overrides(corporate.successions(bars, len(sessions)), changes, sessions, tickers)
        # Polygon files share classes undotted (BFB, PBRA) while the bars use BF.B, PBR.A.
        classes = {name.replace(".", ""): name for name in tickers
                   if re.fullmatch(r"[A-Z]+\.[A-Z]", name) and name.replace(".", "") not in tickers}
        splits = store.read_splits()
        splits["ticker"] = splits.ticker.map(classes).fillna(splits.ticker)
        splits = corporate.rekey(splits, "execution_date", renames, present, sessions, tickers, changes, tolerance=2)
        multiplier, split_audit = cls._multiplier(splits, sessions, tickers, raw_open, raw_close, traded,
                                                  min_audited_splits)
        dividends = store.read_dividends(dividends_path) if dividends_path is not None else None
        if dividends is not None:
            dividends = dividends.assign(ticker=dividends.ticker.map(classes).fillna(dividends.ticker))
            dividends = dividends[dividends.ticker.isin(tickers)].reset_index(drop=True)  # Others never trade here.
            dividends = corporate.rekey(dividends, "ex_dividend_date", renames, present, sessions, tickers, changes)
        reinvested, dividend_audit, cash_events = cls._reinvested(dividends, sessions, tickers, raw_open, raw_close, traded,
                                                     multiplier)
        actions = store.read_actions(actions_path) if actions_path is not None else None
        last_close = _forward_fill(np.where(traded, raw_close, np.nan))
        action_audit = corporate.distribution_factors(actions, sessions, tickers, raw_open, traded, last_close)
        for action in action_audit[action_audit.applied].itertuples():
            reinvested[action.session:, action.code] *= action.share_factor
            if action.cash > 0:  # Paid per share held before the exchange.
                cash_events.append((action.code, action.session, action.cash, action.share_factor))
        growth = multiplier * reinvested
        paid_out, total = np.zeros(shape), growth.copy()
        for code, start, cash, exchanged in cash_events:  # Per share then, times the shares one 2021 share became.
            paid_out[start:, code] += cash * growth[start, code] / exchanged
            total[start:, code] *= 1 + cash / exchanged / raw_open[start, code]
        q_open = np.where(traded, raw_open * growth, np.nan)
        q_close = np.where(traded, raw_close * growth, np.nan)
        valued, continued = traded.copy(), np.zeros(shape, dtype=bool)
        for rename in renames.itertuples():  # One old share becomes one new share at the rename.
            old, new, start = rename.old, rename.new, rename.session
            own = np.flatnonzero(present[start:start + CONTINUE, old])  # Never over the old ticker's own bars.
            span = slice(start, min(start + (own[0] if len(own) else CONTINUE), len(sessions)))
            last = np.flatnonzero(traded[:start, old])
            if len(last) == 0:
                continue
            scale = growth[last[-1], old] / growth[start, new]
            q_open[span, old] = np.where(traded[span, new], q_open[span, new] * scale, np.nan)
            q_close[span, old] = np.where(traded[span, new], q_close[span, new] * scale, np.nan)
            before = paid_out[start - 1, new] if start > 0 else 0.0
            paid_out[span, old] = paid_out[last[-1], old] + (paid_out[span, new] - before) * scale
            valued[span, old] = traded[span, new]
            continued[span, old] = traded[span, new]
        q_close_ff = _forward_fill(q_close)
        mark_open = q_open.copy()
        mark_open[1:] = np.where(valued[1:], q_open[1:], q_close_ff[:-1])
        rename_table = pd.DataFrame({"old": tickers[renames.old], "new": tickers[renames.new],
                                     "date": sessions[renames.session]})
        return cls(bars, sessions, tickers, day_starts, traded, multiplier, growth, q_open, mark_open, continued,
                   paid_out, total, split_audit, dividend_audit, action_audit, rename_table)

    @staticmethod
    def _multiplier(splits, sessions, tickers, raw_open, raw_close, traded, min_audited):
        """Build M from the splits the prices confirm; a listed split the prices contradict is not applied.

        A clear split whose jump shows a session or two after its listed date is applied where it shows.
        """
        multiplier = np.ones(raw_open.shape)
        rows = []
        codes = tickers.get_indexer(splits.ticker)
        starts = sessions.searchsorted(splits.execution_date)  # First session on or after the execution date.

        def observed_at(session, code):
            if 0 < session < len(sessions) and traded[session, code] and traded[session - 1, code]:
                return raw_open[session, code] / raw_close[session - 1, code]
            return np.nan

        for split, code, start in zip(splits.itertuples(), codes, starts):
            if code < 0 or start >= len(sessions):
                continue
            expected = split.split_from / split.split_to  # Price ratio across the split.
            observed = observed_at(start, code)
            audited = abs(np.log(expected)) >= np.log(AUDITED_RATIO) and np.isfinite(observed)
            confirmed = not audited or abs(np.log(observed / expected)) < abs(np.log(observed))
            applied_at = start if confirmed else None
            if not confirmed:
                for later in (start + 1, start + 2):
                    ratio = observed_at(later, code)
                    if np.isfinite(ratio) and abs(np.log(ratio / expected)) < abs(np.log(ratio)):
                        applied_at = later
                        break
            if applied_at is not None:
                multiplier[applied_at:, code] *= split.split_to / split.split_from
            rows.append((split.ticker, split.execution_date, split.split_from, split.split_to, observed, audited,
                         applied_at is not None, sessions[applied_at] if applied_at is not None else pd.NaT,
                         split.rekeyed_from))
        audit = pd.DataFrame(rows, columns=["ticker", "execution_date", "split_from", "split_to", "observed_ratio",
                                            "audited", "applied", "applied_on", "rekeyed_from"])
        checked = audit[audit.audited]
        if len(checked) >= min_audited and checked.applied.mean() < 0.8:
            raise DataError(f"Only {checked.applied.mean():.0%} of {len(checked)} listed splits show in the prices, "
                            "so these bars look split-adjusted already. Use the as-traded download (--unadjusted).")
        return multiplier, audit

    @staticmethod
    def _reinvested(dividends, sessions, tickers, raw_open, raw_close, traded, multiplier):
        """Build D: each dividend is reinvested at its ex-day open, as 1 + cash / open, which keeps the holder's
        value at price + payout (CRSP's total return). It is sized against the last close before it for checks.

        Amounts in other currencies are converted at the previous close of a CurrencyShares trust. A dividend
        also listed as a same-day small split (a scrip dividend) is counted once, through the split. A payout of
        at least CHECKED_YIELD must show as a price drop; if it shows within RELOCATE sessions after its listed
        ex-day it is applied there, otherwise not at all.
        """
        reinvested = np.ones(raw_close.shape)
        columns = ["ticker", "ex_dividend_date", "currency", "cash", "cash_usd", "previous_close", "yield",
                   "applied", "applied_on", "reason", "rekeyed_from"]
        if dividends is None or dividends.empty:
            return reinvested, pd.DataFrame(columns=columns), []
        # The last traded close on the M basis, so it can be put on any later session's share basis.
        basis_close = _forward_fill(np.where(traded, raw_close * multiplier, np.nan))
        funds = {currency: tickers.get_loc(fund) for currency, fund in FX_FUNDS.items() if fund in tickers}
        codes = tickers.get_indexer(dividends.ticker)
        starts = sessions.searchsorted(dividends.ex_dividend_date)
        rows, cash_events = [], []
        for dividend, code, start in zip(dividends.itertuples(), codes, starts):
            if code < 0 or start == 0 or start >= len(sessions):
                continue
            cash, reason, applied_at, paid = dividend.cash, "applied", None, np.nan
            previous = basis_close[start - 1, code] / multiplier[start, code]
            if dividend.currency != "USD":
                fund = funds.get(dividend.currency)
                fx = basis_close[start - 1, fund] / 100 if fund is not None else np.nan
                cash = cash * fx
            if not np.isfinite(cash):
                reason = "no_fx_rate"
            elif not (np.isfinite(previous) and previous > 0):
                reason = "no_previous_close"
            else:
                paid = cash / previous
                shares = multiplier[start, code] / multiplier[start - 1, code]
                if paid >= 1:
                    reason = "more_than_the_price"
                elif 1 < shares < AUDITED_RATIO and 0.75 < np.log(shares) / np.log1p(paid) < 1.33:
                    reason = "paid_in_shares"  # The same-day split already adds these shares.
                elif paid < CHECKED_YIELD:
                    opens = np.flatnonzero(traded[start:start + RELOCATE + 1, code])  # Reinvested at the next open.
                    if len(opens):
                        applied_at = start + int(opens[0])
                    else:
                        reason = "not_traded"
                else:
                    reason = "not_in_prices"
                    for session in range(start, min(start + RELOCATE + 1, len(sessions))):
                        before = basis_close[session - 1, code] / multiplier[session, code]
                        if not (traded[session, code] and np.isfinite(before) and before > 0):
                            continue
                        observed = raw_open[session, code] / before
                        payout = cash / before
                        if abs(np.log(observed * (1 + payout))) < abs(np.log(observed)):
                            applied_at, paid = session, payout
                            reason = "applied" if session == start else "relocated"
                            break
            if applied_at is not None:
                per_share = cash * multiplier[start, code] / multiplier[applied_at, code]  # On that day's basis.
                if paid < CHECKED_YIELD:
                    reinvested[applied_at:, code] *= 1 + per_share / raw_open[applied_at, code]
                else:  # A special or a liquidation stays cash: reinvesting it would lever the stub left behind.
                    cash_events.append((code, applied_at, per_share, 1.0))
            rows.append((dividend.ticker, dividend.ex_dividend_date, dividend.currency, dividend.cash, cash, previous,
                         paid, applied_at is not None, sessions[applied_at] if applied_at is not None else pd.NaT,
                         reason, dividend.rekeyed_from))
        return reinvested, pd.DataFrame(rows, columns=columns), cash_events

    def through(self, t: int) -> "Panel":
        """Return an owning copy of the panel cut after session t, holding only the tickers that traded by then.

        Nothing later is reachable from it: arrays are copies, not views of the full panel, and the audits and
        renames are left out, since whether an event was applied can depend on the sessions after it.
        """
        rows = self.day_starts[t + 1]
        seen = np.unique(self.bars.code.to_numpy()[:rows])
        remap = np.full(len(self.tickers), -1, dtype=np.int32)
        remap[seen] = np.arange(len(seen), dtype=np.int32)
        bars = self.bars.iloc[:rows].copy()
        bars["code"] = remap[bars.code.to_numpy()]
        bars["ticker"] = bars.ticker.cat.remove_unused_categories()

        def cut(values: np.ndarray) -> np.ndarray:
            return values[:t + 1][:, seen]  # Fancy indexing copies.

        def empty(table: pd.DataFrame) -> pd.DataFrame:
            return table.iloc[:0].reset_index(drop=True).copy()

        return Panel(bars, pd.DatetimeIndex(self.sessions[:t + 1].to_numpy().copy()),
                     pd.Index(self.tickers[seen].to_numpy().copy(), dtype=object), self.day_starts[:t + 2].copy(),
                     cut(self.traded), cut(self.multiplier), cut(self.growth), cut(self.q_open), cut(self.mark_open),
                     cut(self.continued), cut(self.paid_out), cut(self.total), empty(self.split_audit),
                     empty(self.dividend_audit),
                     empty(self.action_audit), empty(self.renames),
                     {key: value.copy() for key, value in self._universes.items() if key[0] <= t})

    def index(self, day) -> int:
        return int(self.sessions.get_loc(pd.Timestamp(day)))

    def history(self) -> pd.DataFrame:
        """Return every bar as traded, with each day's split_factor (M) and return_factor (M * D).

        Every row holds only what was known on its own day. To see a past day d on day t's split basis, as
        snapshot(t) does, multiply its prices by split_factor(d) / split_factor(t) and divide its volume by it;
        return_factor(d2) * close(d2) / (return_factor(d1) * close(d1)) - 1 is the total return from d1 to d2.
        """
        rows = self.bars
        session, code = rows.session.to_numpy(), rows.code.to_numpy()
        return rows[BAR_COLUMNS].assign(ticker=rows.ticker.astype(object),
                                        split_factor=self.multiplier[session, code],
                                        return_factor=self.total[session, code])

    def window(self, t: int, lookback: int, adjust: bool = True) -> pd.DataFrame:
        """Return every ticker's bars for the `lookback` sessions through session t, split-adjusted as of t.

        Like Polygon's adjusted bars, the window is adjusted for splits only, not for dividends.
        """
        rows = self.bars.iloc[self.day_starts[max(0, t - lookback + 1)]:self.day_starts[t + 1]]
        if not adjust:
            return rows[BAR_COLUMNS]
        codes = rows.code.to_numpy()
        factor = self.multiplier[rows.session.to_numpy(), codes] / self.multiplier[t, codes]
        adjusted = rows[BAR_COLUMNS].copy()
        for column in PRICES:
            adjusted[column] = adjusted[column].to_numpy() * factor
        adjusted["volume"] = adjusted.volume.to_numpy() / factor
        return adjusted

    def universe(self, t: int, lookback: int, settings: Settings) -> pd.DataFrame:
        """Return the tradable universe for a decision at session t, computed as the API does; memoized."""
        key = (t, lookback, settings.min_price, settings.min_dollar_volume)
        if key not in self._universes:
            self._universes[key] = tradable(self.window(t, lookback), self.sessions[t], settings)
        return self._universes[key]

    def codes(self, tickers) -> np.ndarray:
        return self.tickers.get_indexer(pd.Index(tickers))

    def forward_returns(self, t: int, holding: int, tickers) -> pd.DataFrame:
        """Return each ticker's total return from the open after session t to the open `holding` sessions later.

        A ticker that did not trade at the entry open is not entered (exit_reason no_entry, return NaN). One
        renamed during the holding is valued in its new ticker (renamed). One that does not trade at the exit
        open is sold at its last traded close (last_close): a halt or a delisting. entry_price is the open as
        traded; exit_value is what one share bought then is worth at the exit: after splits, with dividends
        reinvested and large payouts in cash.
        """
        entry, exit_ = t + 1, t + 1 + holding
        if exit_ > len(self.sessions) - 1:
            raise LeakageError(f"A {holding}-day holding decided on {self.sessions[t].date()} would end after "
                               f"{self.sessions[-1].date()}, the last trading day this data may use.")
        tickers = pd.Index(tickers)
        codes = self.codes(tickers)
        known = codes >= 0
        safe = np.where(known, codes, 0)
        entered = known & self.traded[entry, safe]
        shares = self.growth[entry, safe]
        received = self.paid_out[exit_, safe] - self.paid_out[entry, safe]  # Large payouts during the holding.
        entry_price = np.where(entered, self.q_open[entry, safe] / shares, np.nan)
        exit_value = np.where(entered, (self.mark_open[exit_, safe] + received) / shares, np.nan)
        reason = np.where(~entered, "no_entry", np.where(
            self.traded[exit_, safe], "open", np.where(self.continued[exit_, safe], "renamed", "last_close")))
        return pd.DataFrame({"entry_date": self.sessions[entry], "exit_date": self.sessions[exit_],
                             "entry_price": entry_price, "exit_value": exit_value,
                             "forward_return": exit_value / entry_price - 1, "exit_reason": reason}, index=tickers)
