"""The 2026 final test: a run's final choice, traded through 2026 as the backtest traded each of its test years.

Every backtest stops at 2025 (backtest/window.py); 2026 tests the final choice once, at the end. The models of a
run's selection.json, fit on 2021-2025 (model/<name>.pkl, what the API serves), rank the stocks at each 2026 close.
The account buys their lists from its capital in order_value slices at that close while the cash lasts, and sells
each lot by its exit rule, with the run's costs and filters. Each model trading alone and SPY held are reported beside
it, and so is the account with its idle cash in T-bills (BIL) or SPY. Nothing is chosen, tuned or refit on 2026.

The account skips an order its cash does not cover, as the backtest's does. The served account rule
(served_models.py, account.trade make_room) is also traded, as the make_room strategy: such an order first sells the
oldest open trades of the model that earned the least per invested dollar per day in the backtest. It was chosen
after this test, so 2026 does not test it.

Each account's profit is also split into what each model's lots and its idle cash made (account.contributions).

The year is read through FinalTestStore, the backtest's guarded store with a window to the end of 2026, so it is priced
by the same panel: as-traded bars, splits, dividends, and the curated corporate actions and ticker changes. Those
two files cover 2021-2025 only; 2026's renames are found in the bars.

Before trading 2026, it checks this data against the backtest's own (load_data) on 2021-2025, and stops unless:
- the sessions are the same;
- the news are the same, with the same LLM scores;
- the panel's values are the same for all but a few stocks.

The exceptions are stocks whose 2021-2025 history only 2026 data completes: a rename at the end of 2025 that 2026's
bars confirm, or a split or dividend Polygon files under a ticker that first trades in 2026. They are listed. Their
labels feed the models' training, so re-fitting the choice on 2021-2024 and re-trading 2025 from this data moves
2025 slightly; that re-trade is reported beside the run's.

No backtest module imports this one (tests/test_backtest.py). scripts/final_test.py runs it.
"""
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .backtest import metrics
from .backtest.account import contributions, hold, sweep_growth, trade
from .backtest.panel import Panel
from .backtest.protocol import Fold
from .backtest.registry import MODELS
from .backtest.run import BacktestConfig, Candidate, _fit, _per_day, _trade_year, load_data
from .backtest.window import WindowedStore
from .config import ROOT, Settings
from .served_models import TEST_YEAR, load_selection

SPLITS_2026 = ROOT / "data/raw/polygon/splits_2026.parquet"
DIVIDENDS_2026 = ROOT / "data/raw/polygon/dividends_2026.parquet"
CHECK_YEAR = TEST_YEAR - 1  # Re-traded from the final test's data first, beside the run's own.
# The panel's per-session values compared with the backtest's on 2021-2025, and the share of its stocks that may
# differ there (a few whose history 2026 data completes; a broken data path would change thousands).
HISTORY = ("traded", "multiplier", "growth", "q_open", "mark_open", "mark_close", "continued", "paid_out", "total")
MAX_CHANGED = 0.01
IDLE = ("BIL", "SPY")  # Where the idle cash is also held: T-bills (BIL), or SPY.
RESULT_COLUMNS = ["strategy", "idle_cash", "return", "profit", "volatility", "sharpe", "max_drawdown",
                  "return_per_day", "invested_share", "first_entry", "last_exit", "filled", "no_cash", "made_room"]
ACCOUNT_RULES = "account_rules.json"  # Beside the run: the served rule's yearly returns (scripts/account_rules.py).
PARTS_COLUMNS = ["strategy", "idle_cash", "part", "gross", "costs", "profit"]  # In dollars (account.contributions).


class FinalTestStore(WindowedStore):
    """The backtest's store with a window through the end of 2026, for the final test only. Its splits and dividends
    come from several files: the backtest's 2021-2025 ones, then 2026's."""
    last_day = date(TEST_YEAR, 12, 31)

    def __init__(self, bars_dir: Path, news_dir: Path, splits_paths, sentiment_dir: Path | None = None,
                 sentiment_model: str | None = None):
        paths = tuple(Path(path) for path in splits_paths)
        super().__init__(bars_dir, news_dir, paths[0], sentiment_dir, sentiment_model)
        self.splits_path = paths

    def _read_dated(self, path, column: str, what: str, command: str) -> pd.DataFrame:
        if not isinstance(path, (tuple, list)):
            return super()._read_dated(path, column, what, command)
        frames = []
        for one in path:
            frames.append(super()._read_dated(Path(one), column, what, command))
        return pd.concat(frames, ignore_index=True)


@dataclass
class FinalTest:
    run: str
    first_session: str
    last_session: str
    sessions: int  # Trading days of 2026 in the data.
    changed: pd.DataFrame  # The stocks whose 2021-2025 values differ from the backtest's: ticker, values, first day.
    check: dict  # CHECK_YEAR, re-traded here: strategy -> {"replayed", "saved"}.
    results: pd.DataFrame  # RESULT_COLUMNS: one row per strategy and place of its idle cash.
    orders: pd.DataFrame  # The account's orders in 2026, its idle cash kept as cash.
    audit: dict  # The files the store opened.
    served_orders: pd.DataFrame | None = None  # The same under the served rule (make_room); None without one.
    make_room: str | None = None  # The model whose oldest trades the served rule sells first.
    parts: pd.DataFrame | None = None  # PARTS_COLUMNS: each account's profit by model and idle cash.


def _config(run: Path, models: list[str]) -> BacktestConfig:
    """The run's trading settings (costs, capital, slices, filters, schedule) for the models of its final choice, on
    the final test's own data."""
    saved = json.loads((run / "manifest.json").read_text(encoding="utf-8"))["configuration"]
    return BacktestConfig(models=tuple(models), cost_bps=saved["cost_bps"], capital=saved["capital"],
                          order_value=saved["order_value"], whole_shares=saved["whole_shares"],
                          min_price=saved["min_price"], min_dollar_volume=saved["min_dollar_volume"],
                          decide_every=saved["decide_every"], offset=saved["offset"],
                          sentiment_model=saved.get("sentiment_model") or BacktestConfig.sentiment_model,
                          strict_calendar=False)


def _same_history(config: BacktestConfig, panel: Panel, news: pd.DataFrame) -> pd.DataFrame:
    """Check the final test's data against the backtest's (load_data) on 2021-2025 and return the stocks whose
    values differ there (ticker, values, first_day); raise unless the sessions and the news with their scores are the
    same and at most MAX_CHANGED of the stocks differ."""
    _, old, old_news = load_data(config)
    count = len(old.sessions)
    if not panel.sessions[:count].equals(old.sessions):
        raise RuntimeError("The final test's 2021-2025 sessions are not the backtest's.")
    if len(panel.sessions) == count or (panel.sessions[count:].year != TEST_YEAR).any():
        raise RuntimeError(f"The final test's data has no {TEST_YEAR} sessions after 2025, or others.")
    mine = news[news.published_utc < pd.Timestamp(TEST_YEAR, 1, 1, tz="UTC")]
    if len(mine) != len(old_news) or not (mine.id.to_numpy() == old_news.id.to_numpy()).all():
        raise RuntimeError("The final test's 2021-2025 news are not the backtest's.")
    if "llm_scores" in old_news and any(str(a) != str(b) for a, b in zip(mine.llm_scores, old_news.llm_scores)):
        raise RuntimeError("The final test's 2021-2025 news scores are not the backtest's.")
    columns = panel.tickers.get_indexer(old.tickers)
    if (columns < 0).any():
        raise RuntimeError("Stocks of the backtest are missing from the final test's data.")
    first: dict = {}  # Ticker -> (the values that differ, the first session they do).
    for name in HISTORY:
        theirs, ours = getattr(old, name), getattr(panel, name)[:count][:, columns]
        differ = theirs != ours if theirs.dtype == bool else ~np.isclose(theirs, ours, rtol=1e-12, atol=0.0,
                                                                         equal_nan=True)
        for code in np.flatnonzero(differ.any(axis=0)):
            values, day = first.get(old.tickers[code], ([], count))
            first[old.tickers[code]] = (values + [name], min(day, int(np.flatnonzero(differ[:, code])[0])))
    changed = pd.DataFrame([{"ticker": ticker, "values": " ".join(values), "first_day": str(old.sessions[day].date())}
                            for ticker, (values, day) in sorted(first.items())], columns=["ticker", "values", "first_day"])
    if len(changed) > MAX_CHANGED * len(old.tickers):
        raise RuntimeError(f"{len(changed)} of {len(old.tickers)} stocks have other 2021-2025 values than in the "
                           f"backtest, e.g. {changed.head(5).ticker.tolist()}.")
    return changed


def _row(strategy: str, idle: str, ledger) -> dict:
    risk = metrics.ledger_performance(ledger.path)
    orders = ledger.orders
    recorded = orders is not None and len(orders) > 0
    reasons = orders.reason.value_counts() if recorded else pd.Series(dtype=int)
    return {"strategy": strategy, "idle_cash": idle, "return": ledger.yearly_return,
            "profit": float(ledger.profit.iloc[-1]) if len(ledger.profit) else np.nan,
            **{key: risk.get(key, np.nan) for key in ("volatility", "sharpe", "max_drawdown")}, **_per_day([ledger]),
            "first_entry": str(ledger.profit.index[0].date()), "last_exit": str(ledger.profit.index[-1].date()),
            "filled": int(reasons.get("filled", 0)) if recorded else ledger.filled,
            "no_cash": int(reasons.get("no_cash", 0)) if recorded else np.nan,
            "made_room": int((orders.exit_rule == "make_room").sum()) if recorded else np.nan}


def final_test(run: str | Path, check: bool = True, splits_2026: Path = SPLITS_2026,
               dividends_2026: Path = DIVIDENDS_2026) -> FinalTest:
    """Trade the run's final choice through the 2026 data (FinalTest); with check, first compare that data with the
    backtest's on 2021-2025 (_same_history) and re-trade CHECK_YEAR from it."""
    selection = load_selection(str(run))
    if selection.errors:
        raise ValueError(f"The run cannot be tested: {'; '.join(selection.errors)}")
    folder = Path(run) if Path(run).is_absolute() else ROOT / run
    chosen = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
    names = list(chosen["combination"]["models"])
    if set(names) != set(chosen["models"]):
        raise ValueError(f"The combination {names} is not the selection's models {sorted(chosen['models'])}.")
    candidates = [Candidate(name, tuple(sorted(chosen["models"][name]["params"].items())),
                            chosen["models"][name]["candidate"]) for name in names]
    config = _config(folder, names)
    store = FinalTestStore(config.bars_dir, config.news_dir, (config.splits_path, splits_2026), config.sentiment_dir,
                           config.sentiment_model)
    panel = Panel.load(store, False, (config.dividends_path, dividends_2026), config.actions_path,
                       config.changes_path)
    news = store.read_news()
    settings = Settings(min_price=config.min_price, min_dollar_volume=config.min_dollar_volume)
    sessions = panel.sessions

    def account(books, **kwargs):
        return trade(sessions, books, config.cost_bps, config.whole_shares, capital=config.capital, **kwargs)

    room = names.index(selection.make_room) if selection.make_room in names else None  # The served rule's book.
    checked, changed = {}, pd.DataFrame(columns=["ticker", "values", "first_day"])
    if check:  # This data must be the backtest's on 2021-2025, but for the few stocks 2026 completes.
        changed = _same_history(config, panel, news)
        cutoff = sessions[sessions < pd.Timestamp(CHECK_YEAR, 1, 1)][-1]
        fitted, _ = _fit({c.label: (c.model, dict(c.params)) for c in candidates}, MODELS, panel, news, settings,
                         cutoff)
        runs = _trade_year(Fold(CHECK_YEAR, cutoff, True), candidates, fitted, panel, news, settings, config)
        for name, candidate in zip(names, candidates):
            checked[name] = {"replayed": runs[candidate.label].ledger.yearly_return,
                             "saved": chosen["models"][name]["return_by_year"][str(CHECK_YEAR)]}
        checked["account"] = {"replayed": account([runs[c.label].book for c in candidates]).yearly_return,
                              "saved": chosen["combination"]["account_return_by_year"][str(CHECK_YEAR)]}
        saved = _served_rule_return(folder / ACCOUNT_RULES, CHECK_YEAR) if room is not None else None
        if saved is not None:  # The served rule, as scripts/account_rules.py traded it from the backtest's data.
            checked["make_room"] = {"replayed": account([runs[c.label].book for c in candidates],
                                                        make_room=room).yearly_return, "saved": saved}

    last_fit = sessions[sessions < pd.Timestamp(TEST_YEAR, 1, 1)][-1]
    if selection.training_end != str(last_fit.date()):
        raise ValueError(f"The selection was fit through {selection.training_end}, not {last_fit.date()}.")
    served = {model.name: model.models for model in selection.loaded}  # The pickles, fit on 2021-2025.
    runs = _trade_year(Fold(TEST_YEAR, last_fit, False), candidates, {c.label: served[c.model] for c in candidates},
                       panel, news, settings, config)
    books = {name: runs[candidate.label].book for name, candidate in zip(names, candidates)}
    growths = {ticker: sweep_growth(panel, ticker) for ticker in IDLE}
    rows, cash, parts = [], {}, []  # cash: each account's ledger with its idle cash kept as cash.
    accounts = {"account": None, **({"make_room": room} if room is not None else {})}

    def split(strategy: str, idle: str, ledger, growth=None) -> pd.DataFrame:
        table = contributions(ledger, sessions, config.cost_bps, growth).reset_index()
        return table.assign(strategy=strategy, idle_cash=idle)[PARTS_COLUMNS]

    for strategy, index in accounts.items():
        cash[strategy] = account(list(books.values()), record=True, make_room=index)
        rows.append(_row(strategy, "cash", cash[strategy]))
        parts.append(split(strategy, "cash", cash[strategy]))
        for ticker, growth in growths.items():
            ledger = account(list(books.values()), record=True, sweep=growth, make_room=index)
            rows.append(_row(strategy, ticker, ledger))
            parts.append(split(strategy, ticker, ledger, growth))
    for name in names:
        rows.append(_row(name, "cash", account([books[name]], record=True)))
        for ticker, growth in growths.items():
            rows.append(_row(name, ticker, account([books[name]], record=True, sweep=growth)))
    start = min(item.start for item in books.values())
    end = max(item.end for item in books.values())
    rows.append(_row("spy_hold", "-", account([hold(panel, "SPY", start, end, config.capital)])))
    year = sessions[sessions.year == TEST_YEAR]
    return FinalTest(selection.run, str(year[0].date()), str(year[-1].date()), len(year), changed, checked,
                     pd.DataFrame(rows, columns=RESULT_COLUMNS), cash["account"].orders, store.audit(),
                     cash["make_room"].orders if room is not None else None,
                     selection.make_room if room is not None else None, pd.concat(parts, ignore_index=True))


def _served_rule_return(path: Path, year: int) -> float | None:
    """The served rule's return in `year` from scripts/account_rules.py's file beside the run; None without it."""
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        value = saved["strategies"]["make_room"]["return_by_year"].get(str(year))
        return float(value) if value is not None else None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
