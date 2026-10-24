"""Corporate events the price files do not tie to the right ticker: renames, and curated distributions.

Polygon keys bars by the ticker that traded each day, but most splits and dividends by the ticker a company had
when they were downloaded (2026). So before a rename a company's splits and dividends often sit under a ticker
that has not traded yet, and a holding across a rename would stop at the old ticker's last close.

A succession links an old ticker that stops trading to a new listing that starts within a few sessions at about
the same price and trading volume (FB to META, ORCC to OBDC, SQ to XYZ). Excluded on purpose: a successor that had
traded in the previous weeks (an existing listing, not a new one), a when-issued line, warrant, unit or right
standing in for a common stock, a successor that stops again within weeks, and a takeover for cash, whose target
trades pinned to the deal price before delisting (SPACs too, which sit at their trust value). A curated file
(configs/ticker_changes_2021_2025.csv) forces the links detection misses and vetoes the ones it gets wrong.
Finding a succession looks past the rename, but only to put an event on the ticker that actually traded it, so
each day's values are what was known on that day.

Some spin-offs and cash-plus-stock deals have no record at all; configs/corporate_actions_2021_2025.csv adds them.
"""
import numpy as np
import pandas as pd
from ..recommend import TEST_SYMBOLS

PRICE_TOLERANCE = np.log(1.05)  # The new ticker opens within 5% of the old one's last close.
VOLUME_RATIO = 3.0  # And trades between a third and three times its dollar volume.
MIN_VOLATILITY = 0.005  # The old ticker's daily moves over its last 20 and last 10 sessions: not pinned.
MAX_GAP = 3  # Sessions between the old ticker's last bar and the new one's first.
QUIET = 20  # Silence that makes a stop real (old ticker) or a start new (new ticker).
# A Nasdaq fifth letter W, U, R or V, or a dotted .WS/.U/.RT/.W: a warrant, unit, right or when-issued line.
SPECIAL = r"(?:\.WS|\.U|\.RT|\.W)$|^[A-Z]{4}[WURV]$|[a-z]"


def successions(bars: pd.DataFrame, n_sessions: int) -> pd.DataFrame:
    """Return renames as (old, new, session) ticker codes: new starts trading at `session`, old stopped before.

    bars needs session, code, ticker, open, close and volume columns.
    """
    frame = bars[["session", "code", "ticker", "open", "close", "volume"]].sort_values(["code", "session"])
    frame = frame.assign(dollar_volume=frame.close * frame.volume)
    by_code = frame.groupby("code", sort=False)
    gap_before = frame.session - by_code.session.shift(1)
    gap_after = by_code.session.shift(-1) - frame.session
    recent = by_code.dollar_volume.rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    moves = np.log(frame.close).groupby(frame.code, sort=False).diff()
    by_moves = moves.groupby(frame.code, sort=False)
    volatility = by_moves.rolling(20, min_periods=10).std().reset_index(level=0, drop=True)
    volatility_10 = by_moves.rolling(10, min_periods=5).std().reset_index(level=0, drop=True)
    backward = frame.iloc[::-1]
    coming = backward.groupby("code", sort=False).dollar_volume.rolling(5, min_periods=1).mean()
    coming = coming.reset_index(level=0, drop=True)
    stopping = (gap_after.isna() & (frame.session + QUIET <= n_sessions - 1)) | (gap_after > QUIET)
    stops = frame[stopping]
    stops = stops.assign(volume_before=recent[stops.index], last_move=moves[stops.index])
    # A cash target sits at its deal price: little movement, often the same close two days running.
    stops = stops[(volatility[stops.index] >= MIN_VOLATILITY) & (volatility_10[stops.index] >= MIN_VOLATILITY)
                  & (stops.last_move != 0)]
    starts = frame[gap_before.isna() | (gap_before > QUIET)]  # A new listing, not one resuming after a halt.
    starts = starts[~starts.ticker.astype(str).str.contains(TEST_SYMBOLS)]
    # A start whose ticker stops again within QUIET sessions is a when-issued or temporary line.
    ends = frame.session.where(stopping | gap_after.isna())
    next_end = ends.groupby(frame.code, sort=False).bfill()  # The ticker's next stop on or after each bar.
    starts = starts[next_end[starts.index] - starts.session >= QUIET]
    starts = starts.assign(volume_after=coming[starts.index])
    starts = starts[starts.session > 0]
    pairs = []
    for gap in range(1, MAX_GAP + 1):
        joined = stops.assign(start=stops.session + gap).merge(
            starts, left_on="start", right_on="session", suffixes=("_old", "_new"))
        pairs.append(joined[joined.code_old != joined.code_new])
    pairs = pd.concat(pairs, ignore_index=True)
    ratio = pairs.volume_after / pairs.volume_before
    new_special = pairs.ticker_new.astype(str).str.contains(SPECIAL)
    old_special = pairs.ticker_old.astype(str).str.contains(SPECIAL)
    pairs = pairs[(np.abs(np.log(pairs.open_new / pairs.close_old)) < PRICE_TOLERANCE)
                  & (ratio > 1 / VOLUME_RATIO) & (ratio < VOLUME_RATIO) & ~(new_special & ~old_special)]
    # Only unambiguous pairs: each old ticker has one successor, and each new ticker one predecessor.
    pairs = pairs[~pairs.duplicated(["code_old", "session_old"], keep=False)
                  & ~pairs.duplicated(["code_new", "session_new"], keep=False)]
    return pd.DataFrame({"old": pairs.code_old.to_numpy(), "new": pairs.code_new.to_numpy(),
                         "session": pairs.session_new.to_numpy()}).sort_values("session", ignore_index=True)


def with_overrides(renames: pd.DataFrame, changes: pd.DataFrame | None, sessions: pd.DatetimeIndex,
                   tickers: pd.Index) -> pd.DataFrame:
    """Apply the curated ticker changes to the detected renames.

    link 1 is a rename (holdings continue and earlier events move), 2 continues holdings only, 0 only moves
    earlier events, and -1 vetoes a detected link.
    """
    if changes is None or changes.empty:
        return renames
    vetoed = {(old, new) for old, new, link in zip(changes.old, changes.new, changes.link) if link < 0}
    names = list(zip(tickers[renames.old], tickers[renames.new]))
    renames = renames[[pair not in vetoed for pair in names]]
    forced = changes[changes.link.isin([1, 2]) & changes.old.isin(tickers) & changes.new.isin(tickers)]
    added = pd.DataFrame({"old": tickers.get_indexer(forced.old), "new": tickers.get_indexer(forced.new),
                          "session": sessions.searchsorted(forced.date)})
    renames = pd.concat([renames[~renames.old.isin(added.old)], added], ignore_index=True)
    return renames[renames.session < len(sessions)].sort_values("session", ignore_index=True)


def rekey(events: pd.DataFrame, column: str, renames: pd.DataFrame, present: np.ndarray,
          sessions: pd.DatetimeIndex, tickers: pd.Index, changes: pd.DataFrame | None = None,
          tolerance: int = 0) -> pd.DataFrame:
    """Move events filed under a ticker that was not trading yet onto the predecessor that traded them.

    An event at session s keyed to a ticker without a bar in the QUIET sessions through s is moved to the old
    ticker of a later rename into it, if that old ticker traded around s; chains (A to B to C) are followed. A
    curated change (link 0 or 1) moves the events filed under its new ticker before its date onto the old one,
    even if the new ticker was then trading as another company. A moved event is dropped when the destination
    already lists the same event within `tolerance` sessions (Polygon often files it under both tickers). The
    original ticker is kept in rekeyed_from.
    """
    events = events.assign(rekeyed_from=None)
    if events.empty:
        return events
    codes = tickers.get_indexer(events.ticker)
    starts = sessions.searchsorted(events[column])
    destination = codes.copy()
    if changes is not None and not changes.empty:
        for change in changes[changes.link.isin([0, 1])].itertuples():
            if change.old in tickers and change.new in tickers:
                before = events[column].to_numpy() < np.datetime64(change.date)
                destination[(codes == tickers.get_loc(change.new)) & before] = tickers.get_loc(change.old)
    into = renames.groupby("new") if not renames.empty else None
    for row, (code, start) in enumerate(zip(codes, starts)):
        if destination[row] != code or code < 0 or start >= len(sessions) or into is None:
            continue
        if present[max(start - QUIET, 0):start + 1, code].any():
            continue  # The filed ticker was trading: the event is its own.
        target = code
        for _ in range(3):  # A rename chain is at most a few links long.
            if target not in into.groups:
                target = -1
                break
            later = renames.loc[into.groups[target]]
            later = later[later.session > start]
            if later.empty:
                target = -1
                break
            target = int(later.iloc[0].old)
            if present[max(start - 1, 0):start + 1, target].any():
                break
        if target >= 0 and present[max(start - 1, 0):start + 1, target].any():
            destination[row] = target
    moved = destination != codes
    native = {(code, start) for code, start in zip(codes[~moved], starts[~moved]) if code >= 0}
    duplicate = np.array([moved[row] and any((destination[row], start + shift) in native
                                             for shift in range(-tolerance, tolerance + 1))
                          for row, start in enumerate(starts)], dtype=bool)
    events.loc[moved, "rekeyed_from"] = events.ticker[moved]
    events.loc[moved, "ticker"] = tickers[destination[moved]]
    return events[~duplicate].reset_index(drop=True)


def distribution_factors(actions: pd.DataFrame | None, sessions: pd.DatetimeIndex, tickers: pd.Index,
                         raw_open: np.ndarray, traded: np.ndarray, last_close: np.ndarray) -> pd.DataFrame:
    """Return, per curated action, the shares of its own stock one share becomes, and the cash it pays.

    A holder keeps `shares_kept` shares (1 for a spin-off) and receives `cash` plus `distributed_ratio` shares of
    `distributed`, valued at that ticker's open on the action date (both must trade that session) and reinvested
    at the parent's open that day. Only prices of that session are used.
    """
    columns = ["ticker", "date", "session", "code", "share_factor", "cash", "previous_close", "value", "applied",
               "reason"]
    if actions is None or actions.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for action in actions.itertuples():
        code = tickers.get_loc(action.ticker) if action.ticker in tickers else -1
        start = int(sessions.searchsorted(action.date))
        reason, share_factor, previous, value = "applied", np.nan, np.nan, np.nan
        if code < 0 or start >= len(sessions) or not traded[start, code]:
            reason = "parent_not_traded"
        else:
            previous = last_close[start - 1, code] if start > 0 else np.nan
            value = action.cash
            if isinstance(action.distributed, str) and action.distributed:
                spun = tickers.get_loc(action.distributed) if action.distributed in tickers else -1
                if spun < 0 or not traded[start, spun]:  # Valued that session only, never with a later price.
                    reason = "distributed_not_traded"
                else:
                    value += action.distributed_ratio * raw_open[start, spun]
            if reason == "applied":  # The shares; the cash is held as cash (see Panel).
                share_factor = action.shares_kept + (value - action.cash) / raw_open[start, code]
        rows.append((action.ticker, action.date, start, code, share_factor, action.cash, previous, value,
                     reason == "applied", reason))
    return pd.DataFrame(rows, columns=columns)
