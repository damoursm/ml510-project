"""Run a nested walk-forward backtest over 2021-2025, never 2026: fit each model candidate on the years before a
test year, trade that year, and choose a test year's hyperparameters on earlier years only.

A candidate is a model with one hyperparameter configuration (registry.GRIDS), its exit strategy (exits.py) and how
many of each day's ranked stocks it lists (picks: its best N, or its best X%) included. For each test year Y, every
candidate is fit on all the years before Y (training labels end before Y's first trade), then trades Y alone;
candidates that differ only in their exit strategy or picks share one fit and one ranking:
- at every session's close (config.decide_every) it ranks the stocks through rank, the API's own decision path,
  as the API's daily lists do, at 16:00;
- from config.capital ($10,000) in cash, it buys that day's list at that close in slices of the order value ($100),
  the best-ranked first, while its cash lasts, and sells each stock at the close hold_days later, or earlier when
  its exit rule fires; the proceeds pay for later orders (account.trade);
- the last entries of the year are those whose hold_days still fit in it.

That run is out of sample for Y and serves twice: as validation for every later test year, and as Y's test run.
Its yearly return is its last equity (cash and open lots) over its starting capital, minus 1.

For a test year T, each model's candidate is chosen on the test years before T, in two steps (select):
- its hyperparameters and exit strategy: those with the best mean yearly return among the candidates that list the
  default number of stocks (the grid's first picks, 20 a day, which keeps the $10,000 about fully invested at every
  holding);
- how many stocks it lists: among those settings' list sizes, the one with the best return per day, its profit over
  the dollars it had invested at each close, the years pooled. The cash a short list leaves idle loses nothing in a
  falling year and earns nothing in a rising one; per invested dollar it decides nothing.
The first test year has no earlier year to choose on, so it runs the defaults. The account, the combination model,
trades from the one capital every model whose trades beat SPY on those same years (choose_combination): trading
alone as chosen, its mean yearly return beats that of SPY bought with the same dollars on the same dates
(spy_table), so only the days it had trades open count. The first test year trades every model; a year where none
qualifies trades nothing. The references spend the account's money otherwise: SPY held
with its capital through the year, and SPY or the equal-weight universe bought for every order it filled.

The account is also evaluated in the years that chose it, as each candidate is: each test year's account (its chosen
candidates trading together from the one capital) in every test year before it, and the final choice's in every test
year (_account_return). Those years chose it, so this is in sample: what the choice saw, not what to expect.

Its idle cash earns nothing; each test year's account, and each strategy trading alone as chosen, is also evaluated
with that cash held in each ticker of config.idle_cash (SPY, and BIL, a Treasury bill fund, for the interest), as
account.trade's sweep (_idle_cash). An account that trades nothing then holds the ticker with all of its capital.
This is reported only: no choice reads it.

Training data never comes after the year it is scored on, and no choice for a year sees that year or a later one.
The choice made on every test year (each model's hyperparameters, then the combination), with the combination's
models fit on all of 2021-2025, is kept for the 2026 test, which this never runs, and is what the API serves.
Returns are total returns: dividends are reinvested.
"""
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from .. import sentiment
from ..config import ROOT, Settings
from ..data import DataUnavailable
from ..exits import EXIT_FIELDS, ExitRule, exit_rule
from ..recommend import buy_and_sell, decision_inputs, news_cutoff, pick_count, rank, trade_list
from . import metrics
from .account import (CAPITAL, ORDER_COLUMNS, Book, Ledger, basket, book, hold, holdings, idle_ledger, mirror,
                      sweep_growth, trade)
from .panel import Panel
from .protocol import FoldData, check_test_years, news_window, schedule, walk_forward
from .registry import GRIDS, MODELS, UNTUNED, build, configurations, configure
from .window import END_UTC, LAST_DAY, LeakageError, WindowedStore

ACCOUNT = "account"
# SPY held with the account's capital; SPY, and the equal-weight universe, bought for every order the account filled.
BENCHMARKS = ("spy_hold", "spy_mirror", "universe_ew")
EXTREME = np.log(2)  # A holding that more than doubles or halves is listed in anomalies.
SPY = "SPY"  # A strategy enters the account only when its trades beat this ticker bought on the same dates.
# spy_table: per strategy and test year, its trades alone beside the same dollars spent on SPY on their dates.
SPY_COLUMNS = ["strategy", "candidate", "year", "days_with_trades", "profit", "yearly_return", "spy_mirror_profit",
               "spy_mirror_return"]
# A candidate's settings that change only how its lists are traded, never its fit (its exit rule and how many stocks
# it lists): candidates that differ only in them share one fit and one ranking per day.
TRADING_FIELDS = (*EXIT_FIELDS, "picks")
# picks_table: per choice and model, each list size of its chosen settings, on the years that chose and the year after.
PICKS_COLUMNS = ["choice", "chosen_on", "model", "candidate", "picks", "chosen", "stocks_per_list",
                 "tuning_return_per_day", "tuning_mean_yearly_return", "test_return_per_day", "test_yearly_return",
                 "test_invested_share"]
# _idle_cash: per strategy (the account, then each one trading alone) and test year, where its idle cash was held.
IDLE_CASH = "cash"  # Kept as cash: it earns nothing, as the account does.
IDLE_COLUMNS = ["strategy", "idle_cash", "year", "yearly_return", "profit", "volatility", "max_drawdown"]


@dataclass(frozen=True)
class BacktestConfig:
    # Names from registry.MODELS (all by default); one with a grid there is tuned on it, the others run as they are.
    models: tuple[str, ...] = tuple(MODELS)
    test_years: tuple[int, ...] = (2022, 2023, 2024, 2025)  # Each trains on all earlier years; 2021 only trains.
    offset: int = 0  # Sessions to delay each year's first entry, to check how much the start day matters.
    # Sessions between a model's decisions: 1, every model decides and buys every session, and each stock bought is
    # sold by its exit strategy, so holdings overlap; None, back to back, every hold_days sessions.
    decide_every: int | None = 1
    cost_bps: float = 5.0  # Paid on the value traded, both buying and selling.
    capital: float = CAPITAL  # The cash every test year starts with; the account buys only what it can pay for.
    order_value: float = 100.0  # The slice of the cash each recommended stock is bought for.
    whole_shares: bool = False  # Round each order down to whole shares, sized from the list price.
    # Tickers the idle cash is also evaluated in (reported, never chosen on): SPY, and BIL, a 1-3 month Treasury bill
    # fund, for the interest. A ticker missing from the data is left out.
    idle_cash: tuple[str, ...] = ("SPY", "BIL")
    final_fit: bool = True  # Fit the final choice on all of 2021-2025 for the 2026 test; never scored here.
    min_price: float = 5.0
    min_dollar_volume: float = 1_000_000.0
    null_seeds: int = 20  # Random lists of the same size, to see how a model ranks among no-skill ones.
    bars_dir: Path = ROOT / "data/raw/polygon/daily_bars_unadjusted"
    news_dir: Path = ROOT / "data/raw/polygon/news"
    # The news' LLM scores (stock_recommender/sentiment.py) by sentiment_model, attached as llm_scores; None: none.
    sentiment_dir: Path | None = ROOT / "data/processed/news_sentiment"
    sentiment_model: str = "qwen3:8b"
    sentiment_max_tickers: int = 5  # As Settings.sentiment_max_tickers: articles with more are never scored.
    splits_path: Path = ROOT / "data/raw/polygon/splits_2021_2025.parquet"
    dividends_path: Path | None = ROOT / "data/raw/polygon/dividends_2021_2025.parquet"  # None: price returns.
    # Spin-offs and cash deals missing from Polygon's records (None leaves them out).
    actions_path: Path | None = ROOT / "configs/corporate_actions_2021_2025.csv"
    # Renames detection misses or gets wrong, and events Polygon filed under another ticker (None: detection only).
    changes_path: Path | None = ROOT / "configs/ticker_changes_2021_2025.csv"
    strict_calendar: bool = True  # Check the trading days per year against the NYSE calendar.

    def __post_init__(self):
        if self.offset < 0 or self.null_seeds < 0:
            raise ValueError("offset and null_seeds must be at least 0.")
        if self.decide_every is not None and (isinstance(self.decide_every, bool) or int(self.decide_every)
                                              != self.decide_every or self.decide_every < 1):
            raise ValueError(f"decide_every must be a whole number of sessions, at least 1, or None (back to back); "
                             f"got {self.decide_every!r}.")
        if self.cost_bps < 0 or not self.order_value > 0 or self.capital < self.order_value:
            raise ValueError("Costs must be non-negative, the order value positive and the capital at least one "
                             "order.")
        if not self.models:
            raise ValueError("Backtest at least one model.")
        if any(not isinstance(ticker, str) or not ticker or ticker == IDLE_CASH for ticker in self.idle_cash) \
                or len(set(self.idle_cash)) != len(self.idle_cash):
            raise ValueError(f"idle_cash names distinct tickers, never {IDLE_CASH!r}; got {self.idle_cash!r}.")
        names = list(self.models)
        reserved = set(BENCHMARKS) | {ACCOUNT}
        if len(set(names)) != len(names) or set(names) & reserved:
            raise ValueError(f"Strategy names must be unique and differ from {sorted(reserved)}; got {names}.")
        check_test_years(self.test_years)  # Before any data is read.


@dataclass(frozen=True)
class Candidate:
    model: str  # The registered model it belongs to.
    params: tuple  # Its hyperparameters, as sorted (name, value) pairs.
    label: str  # The model and its hyperparameters, unique within a run.


@dataclass
class Run:
    """One candidate trading one test year, with models fit on the years before it. Only the candidates chosen for
    the year keep what they traded (book onward); the others keep their score, which is all a later choice reads."""
    candidate: Candidate
    year: int
    score: dict  # Its row of the scores table.
    picks: int  # How many stocks it lists: its best picks, or with a pick_share that share of those it ranked.
    pick_share: float | None
    sizes: tuple  # Per holding period, the stocks on its list.
    horizon: int  # Its holding: the hold_days of its exit rule.
    rule: ExitRule | None = None
    book: Book | None = None  # Its recommended lists (one per holding period) priced, each stock for the order value.
    ledger: Ledger | None = None  # The candidate trading its lists alone, from the starting capital, a slice each.
    table: pd.DataFrame | None = None  # One row per holding period (account.holdings): its cost, profit, turnover.
    positions: pd.DataFrame | None = None  # One row per order of the ledger, with its return and how it exited.
    signals: list = field(default_factory=list)
    # Its lists, one per holding period, and those periods: what prices its book again when it did not keep it, to
    # evaluate an account holding it (_book). The lists are the day's rankings' own, shared between candidates.
    lists: list = field(default_factory=list)
    periods: list = field(default_factory=list)


@dataclass
class BacktestResult:
    config: BacktestConfig
    folds: pd.DataFrame  # One row per test year: what was chosen for it, on which years, and the account's result.
    scores: pd.DataFrame  # One row per candidate and test year: the out-of-sample runs validation reads.
    metrics: pd.DataFrame  # One row per strategy (each model as chosen, the account, references) and scope.
    profit: pd.DataFrame  # Profit at each open, starting at $0, test years added up; one column per strategy.
    periods: pd.DataFrame  # Per model as chosen and holding period: its cost, profit and turnover.
    positions: pd.DataFrame  # Per model as chosen: every order it placed trading alone.
    orders: pd.DataFrame  # The account's orders in every test year.
    signals: pd.DataFrame  # Per decision of each model as chosen: information coefficient and buy-minus-sell spread.
    null: pd.DataFrame  # Per model and random seed: mean yearly return and Sharpe of random lists of the same size.
    guards: dict
    anomalies: pd.DataFrame
    selection: dict  # The final choice, made on every test year, for the 2026 test.
    versions: dict = field(default_factory=dict)
    final_models: dict = field(default_factory=dict)  # Fit on 2021-2025 for the 2026 test; never scored here.
    # Per strategy and test year: it trading alone, as chosen, beside SPY bought with the same dollars on the same
    # dates (spy_table), which choose_combination reads.
    against_spy: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Per choice and model: the list sizes select compared by return per day, and how each did the year after.
    picks: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Per strategy (the account, then each one trading alone, as chosen) and test year: its return with its idle cash
    # kept as cash or held in each ticker of config.idle_cash (_idle_cash).
    idle_cash: pd.DataFrame = field(default_factory=pd.DataFrame)


def load_data(config: BacktestConfig) -> tuple[WindowedStore, Panel, pd.DataFrame]:
    """Read the 2021-2025 bars, splits, dividends and news through the window's guarded store."""
    store = WindowedStore(config.bars_dir, config.news_dir, config.splits_path, config.sentiment_dir,
                          config.sentiment_model)
    panel = Panel.load(store, config.strict_calendar, config.dividends_path, config.actions_path,
                       config.changes_path)
    return store, panel, store.read_news()


def _label(model: str, params: dict) -> str:
    return model + json.dumps(params, sort_keys=True, default=str) if params else model


def _candidates(config: BacktestConfig, factories: dict[str, Callable], grids: dict) -> list[Candidate]:
    """Return every model's configurations, defaults first; checked before data is read."""
    unknown = sorted({name for name in config.models if name not in factories})
    if unknown:
        raise ValueError(f"Unknown models {unknown}; choose from {sorted(factories)}.")
    for name in dict.fromkeys(config.models):
        untuned = sorted(UNTUNED & set(grids.get(name, {})))
        if untuned:
            raise ValueError(f"{name}'s grid names {untuned}: a grid gives a share of the ranked stocks as picks, "
                             "e.g. \"1%\".")
    candidates = []
    for name in config.models:
        for params in configurations(grids.get(name, {})):
            candidates.append(Candidate(name, tuple(sorted(params.items())), _label(name, params)))
    for candidate in candidates:
        try:
            build(factories[candidate.model], dict(candidate.params))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{candidate.label} does not build: {error}") from error
    return candidates


def _trained(name: str, params: dict) -> str:
    """The label of what a candidate fits: its model and hyperparameters without its trading settings (its exit rule
    and picks, TRADING_FIELDS), which change only how its lists are traded."""
    return _label(name, {key: value for key, value in params.items() if key not in TRADING_FIELDS})


def _fit(specs: dict[str, tuple[str, dict]], factories, panel, news, settings,
         cutoff: pd.Timestamp) -> tuple[dict, str | None]:
    """Fit fresh models for every candidate label (its model and hyperparameters), each on training data cut at
    cutoff with its own windows and horizon; candidates that differ only in their trading settings (exit rule and
    picks) share one fit. Return them by label and the last training label's exit."""
    fitted, training, trained = {}, {}, {}

    def fit(name: str, params: dict):
        key = _trained(name, params)
        if key not in trained:
            model = build(factories[name], {k: v for k, v in params.items() if k not in TRADING_FIELDS})  # Fresh.
            data = (model.horizon_days, model.bars_lookback, model.news_lookback_days)
            if data not in training:
                training[data] = FoldData(panel, news, settings, cutoff, *data)
            model.fit(training[data])
            trained[key] = model
        trading = {k: v for k, v in params.items() if k in TRADING_FIELDS}
        return configure(trained[key], trading) if trading else trained[key]

    for label, (name, params) in specs.items():
        fitted[label] = fit(name, params)
    exits = [data._labels.exit_date.max() for data in training.values()
             if data._labels is not None and len(data._labels)]
    return fitted, str(max(exits).date()) if exits else None


def _per_day(ledgers: list[Ledger]) -> dict:
    """Over these ledgers together: the return per day, their profit over the dollars their open lots cost at each
    close, added up (what an invested dollar earned per day held, however much cash sat idle), and the share of the
    capital those lots cost at an average close. An account that bought nothing has no return per day."""
    dollars = sum(float(ledger.invested.sum()) for ledger in ledgers)
    closes = sum(len(ledger.invested) for ledger in ledgers)
    profit = sum(float(ledger.profit.iloc[-1]) for ledger in ledgers if len(ledger.profit))
    capital = max((ledger.capital for ledger in ledgers), default=np.nan)
    return {"return_per_day": profit / dollars if dollars > 0 else np.nan,
            "invested_share": dollars / closes / capital if closes else np.nan}


def _score(candidate: Candidate, year: int, ledger: Ledger, table: pd.DataFrame, signals: list, sizes: list) -> dict:
    """A run's row of the scores table: how the candidate did trading its lists alone, which select reads.
    invested_dollar_days adds up what its open lots cost at each close, so that years pool (_pooled_per_day)."""
    summary = metrics.ledger_performance(ledger.path, table)
    ic = pd.Series([signal["ic"] for signal in signals], dtype=float)
    return {"model": candidate.model, "candidate": candidate.label,
            "params": json.dumps(dict(candidate.params), default=str), "year": year, "capital": ledger.capital,
            "profit": float(ledger.profit.iloc[-1]), "yearly_return": ledger.yearly_return,
            "invested_dollar_days": float(ledger.invested.sum()), **_per_day([ledger]),
            "daily_return": summary.get("daily_return"), "sharpe": summary.get("sharpe"),
            "max_drawdown": summary.get("max_drawdown"), "periods": summary.get("periods"),
            "avg_turnover": summary.get("avg_turnover"),
            "stocks_per_list": float(np.mean(sizes)) if sizes else np.nan,
            "ic_mean": float(ic.mean()) if ic.notna().any() else np.nan}


def _trade_year(fold, candidates: list[Candidate], fitted: dict, panel: Panel, news: pd.DataFrame,
                settings: Settings, config: BacktestConfig, keep: set | None = None) -> dict[str, Run]:
    """Trade every candidate through the fold's year, each on its schedule (a decision every config.decide_every
    sessions, by default every session), with its own exit rule and list size (recommend.pick_count); return its
    run by label. On a decision day the models share one snapshot of the inputs per window, and candidates sharing a
    fit (they differ only in their exit settings or picks) rank the stocks once. Only the candidates in `keep` (all
    when None) keep what they traded; the others keep their score."""
    year = fold.year
    models = {candidate.label: fitted[candidate.label] for candidate in candidates}
    rules = {label: exit_rule(model) for label, model in models.items()}
    rankers = {candidate.label: _trained(candidate.model, dict(candidate.params)) for candidate in candidates}
    schedules, by_holding = {}, {}
    for label, model in models.items():
        holding = rules[label].hold_days
        if holding not in by_holding:  # One schedule per holding, shared by the candidates that hold as long.
            by_holding[holding] = schedule(panel.sessions, [fold], holding, config.offset, config.decide_every)[year]
        periods = by_holding[holding]
        if not periods:
            raise ValueError(f"No {holding}-session holding fits in {year}.")
        if periods[0].decision < model.bars_lookback - 1:
            raise ValueError(f"{label}'s first decision in {year} needs {model.bars_lookback} sessions of bars; "
                             "start the test years later.")
        schedules[label] = periods
    deciding: dict[int, dict] = {}  # Session -> what ranks then (a shared fit) -> a model holding that fit.
    for label, periods in schedules.items():
        for period in periods:
            deciding.setdefault(period.decision, {}).setdefault(rankers[label], models[label])
    news_days = max(model.news_lookback_days for model in models.values())
    tables = {}
    for t in sorted(deciding):
        as_of = panel.sessions[t]
        day_news = news_window(news, news_cutoff(as_of), news_days)
        windows, snapshots = {}, {}
        for ranker, model in deciding[t].items():
            lookback = model.bars_lookback  # Each model reads exactly its own window, as in the API.
            inputs = (lookback, model.news_lookback_days)
            if inputs not in snapshots:  # One snapshot per window, so the models' features are computed once.
                windows.setdefault(lookback, panel.window(t, lookback))
                snapshots[inputs] = decision_inputs(windows[lookback], day_news, as_of, settings, inputs[1])
            try:
                tables[ranker, t] = rank(snapshots[inputs], model).table
            except DataUnavailable:
                tables[ranker, t] = pd.DataFrame(columns=["ticker", "close", "score"])
    pickers: dict[str, dict] = {}  # A ranking -> the pick rules (picks, pick_share) of the candidates sharing it.
    for candidate in candidates:
        model = models[candidate.label]
        picking = (model.picks, getattr(model, "pick_share", None))
        pickers.setdefault(rankers[candidate.label], {}).setdefault(picking, model)
    # Kept per decision, holding and list size: the lists, their signals and how each listed stock exits at the end of
    # the holding. The forward returns are not: each is read once, for every list size taken from its ranking.
    decisions, chosen, reasons, runs = {}, {}, {}, {}
    for candidate in candidates:
        label = candidate.label
        model, rule, periods, ranker = models[label], rules[label], schedules[label], rankers[label]
        lists, exits, signals, sizes = [], [], [], []
        for period in periods:
            t = period.decision
            table = tables[ranker, t]
            size = pick_count(model, len(table))
            key = (ranker, t, rule.hold_days, size)
            if key not in decisions:  # The signal reads the whole holding, whatever the exit rule does.
                forward = panel.forward_returns(t, rule.hold_days, table.ticker)
                returns = forward.forward_return
                ic = metrics.information_coefficient(table.set_index("ticker").score, returns)
                for count in {pick_count(other, len(table)) for other in pickers[ranker].values()}:
                    if (ranker, t, count) not in chosen:
                        chosen[ranker, t, count] = trade_list(table, count)
                    buys, sells = buy_and_sell(table, count)
                    reasons[ranker, t, rule.hold_days, count] = forward.exit_reason.reindex(
                        chosen[ranker, t, count].ticker).to_numpy(dtype=object)
                    decisions[ranker, t, rule.hold_days, count] = {
                        "decision_date": panel.sessions[t], "scored": len(table), "picks": count,
                        # Decisions closer than their holding share sessions (metrics.signal_summary).
                        "holding": rule.hold_days, "every": config.decide_every or rule.hold_days, "ic": ic,
                        "spread": returns.reindex(buys.ticker).mean() - returns.reindex(sells.ticker).mean()
                        if len(buys) and len(sells) else np.nan}
            lists.append(chosen[ranker, t, size])
            exits.append(reasons[key])
            sizes.append(size)
            signals.append({"strategy": candidate.model, "candidate": label, "fold": year, **decisions[key]})
        item = book(panel, candidate.model, periods, lists, config.order_value, rule)
        ledger = trade(panel.sessions, [item], config.cost_bps, config.whole_shares, record=True,
                       capital=config.capital)
        table = holdings(ledger, periods, panel.sessions).assign(fold=year)
        light = Run(candidate, year, _score(candidate, year, ledger, table, signals, sizes), model.picks,
                    getattr(model, "pick_share", None), tuple(sizes), rule.hold_days, rule, lists=lists,
                    periods=periods)
        if keep is not None and label not in keep:
            runs[label] = light  # Not traded this year: its score is all a later choice reads.
            continue
        orders = ledger.orders  # In the book's order: period by period, each list's stocks in turn.
        with np.errstate(divide="ignore", invalid="ignore"):
            returned = np.where(orders.cost > 0, orders.exit_value / orders.cost - 1, np.nan)
        exited = np.concatenate(exits) if exits else np.zeros(0, dtype=object)
        early = orders.exit_rule.to_numpy(dtype=object)
        exited = np.where(early != "", early, exited)  # Sold by its exit rule, or at the end of its holding.
        positions = orders.assign(fold=year, forward_return=returned,  # How it exited, when it was bought.
                                  exit_reason=np.where(orders.cost > 0, exited, None))
        runs[label] = replace(light, book=item, ledger=ledger, table=table, positions=positions, signals=signals)
    return runs


def _universe_scan(fold, models: list, panel: Panel, settings: Settings, config: BacktestConfig):
    """Scan the year's universe for holdings beyond 2x, on the shortest holding among the models and in the
    universe of the widest window; return those rows and that window."""
    year = fold.year
    holding = min(exit_rule(model).hold_days for model in models)
    reference = max(model.bars_lookback for model in models)
    rows = []
    for period in schedule(panel.sessions, [fold], holding, config.offset, config.decide_every)[year]:
        t = period.decision
        universe = panel.universe(t, reference, settings).index
        returns = panel.forward_returns(t, holding, universe).forward_return
        extreme = returns[np.abs(np.log1p(returns)) > EXTREME]
        rows += [{"kind": "universe_return_beyond_2x", "ticker": ticker, "decision_date": panel.sessions[t],
                  "forward_return": value} for ticker, value in extreme.items()]
    return rows, reference


def _references(panel: Panel, account: Ledger, books: list[Book], universe: Callable[[int], pd.Index],
                config: BacktestConfig) -> dict[str, Ledger]:
    """Spend the account's money otherwise through its year, from the same capital, with fractional shares: on SPY
    held with all of it from its first entry to its last exit, and on SPY or the equal-weight universe for every
    order it filled (account.spent, one array per book), with that order's dollars and dates: every one of them,
    whatever their own cash."""
    amounts = list(account.spent)
    start, end = min(item.start for item in books), max(item.end for item in books)
    spend = lambda items: trade(panel.sessions, items, config.cost_bps, capital=config.capital,  # noqa: E731
                                cash_limit=False)
    return {"spy_hold": spend([hold(panel, "SPY", start, end, config.capital)]),
            "spy_mirror": spend([mirror(panel, item, paid) for item, paid in zip(books, amounts)]),
            "universe_ew": spend([basket(panel, item, paid, universe) for item, paid in zip(books, amounts)])}


def _random_list(universe: pd.DataFrame, rng: np.random.Generator, picks: int) -> pd.DataFrame:
    # Sorted first, so the universe's row order cannot change a draw.
    tickers = rng.permutation(np.sort(np.asarray(universe.index, dtype=object)))[:picks]
    return pd.DataFrame({"ticker": tickers, "price": universe.close.reindex(tickers).to_numpy(dtype=float)})


def _null_ledgers(fold, sizes: tuple, horizon: int, reference: int, panel: Panel, settings: Settings,
                  config: BacktestConfig, rule: ExitRule | None = None) -> list[Ledger]:
    """Trade null_seeds random lists through the year, on a model's schedule and holding and with its exit rule, each
    as long as the model's list that day (sizes, one per holding period): no-skill models. The draws depend on
    neither the rule nor the sizes (a shorter list is the start of a longer one), so rules and list sizes are
    compared on the same lists."""
    year = fold.year
    periods = schedule(panel.sessions, [fold], horizon, config.offset, config.decide_every)[year]
    if len(sizes) != len(periods):
        raise ValueError(f"{len(sizes)} list sizes for {len(periods)} holding periods in {year}.")
    ledgers = []
    for seed in range(config.null_seeds):
        rng = np.random.default_rng([seed, year, horizon])
        lists = [_random_list(panel.universe(period.decision, reference, settings), rng, size)
                 for period, size in zip(periods, sizes)]
        ledgers.append(trade(panel.sessions, [book(panel, "null", periods, lists, config.order_value, rule)],
                             config.cost_bps, config.whole_shares, capital=config.capital))
    return ledgers


def _first_best(values: pd.Series, labels: list[str]) -> str:
    """The label with the highest value, the first in `labels` among ties; a missing or NaN value never wins."""
    array = values.reindex(labels).to_numpy(dtype=float)
    return labels[int(np.argmax(np.where(np.isnan(array), -np.inf, array)))]


def _means(rows: pd.DataFrame) -> pd.Series:
    """Per candidate, its mean yearly return over rows' years; NaN when one year has none."""
    return rows.groupby("candidate").yearly_return.apply(lambda values: values.mean(skipna=False))


def _pooled_per_day(rows: pd.DataFrame) -> pd.Series:
    """Per candidate, its return per day over all of rows' years together: its profit over the dollars it had
    invested at each close (invested_dollar_days), both added up over the years; NaN when it invested nothing."""
    sums = rows.groupby("candidate")[["profit", "invested_dollar_days"]].sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return (sums.profit / sums.invested_dollar_days).where(sums.invested_dollar_days > 0)


def _list_sizes(candidates: list[Candidate], label: str) -> list[str]:
    """A candidate's settings with each list size its grid tries: the candidates of its model that differ from it
    only in picks, itself included, in grid order (the default list size first)."""
    mine = next(candidate for candidate in candidates if candidate.label == label)
    rest = {key: value for key, value in mine.params if key != "picks"}
    return [candidate.label for candidate in candidates if candidate.model == mine.model
            and {key: value for key, value in candidate.params if key != "picks"} == rest]


def select(scores: pd.DataFrame, candidates: list[Candidate], years, before: int | None = None) -> dict[str, str]:
    """Return each model's candidate chosen on `years`, in two steps:
    - its other settings (hyperparameters and exit strategy): the candidate with the best mean yearly return among
      those listing the default number of stocks, its defaults' picks;
    - its list size (picks): among those settings' list sizes (_list_sizes), the one with the best return per day,
      the years pooled (_pooled_per_day): what an invested dollar earned per day held, so that the cash a short list
      leaves idle, which earns nothing in a rising year and loses nothing in a falling one, never decides. Ties, and
      no return per day, keep the default list size.
    The first in grid order among ties, and the defaults when there is no year. A year at or after `before` raises
    LeakageError."""
    years = sorted(set(years))
    if before is not None and any(year >= before for year in years):
        raise LeakageError(f"The choice for {before} may use only earlier years; got {years}.")
    chosen = {}
    for model in dict.fromkeys(candidate.model for candidate in candidates):
        mine = [candidate for candidate in candidates if candidate.model == model]
        if not years:
            chosen[model] = mine[0].label
            continue
        rows = scores[scores.candidate.isin([candidate.label for candidate in mine]) & scores.year.isin(years)]
        default = dict(mine[0].params).get("picks")
        listing = [candidate.label for candidate in mine if dict(candidate.params).get("picks") == default]
        settings = _first_best(_means(rows), listing)
        sizes = _list_sizes(mine, settings)
        if len(sizes) > 1:  # Ties keep the default list size: the settings' own.
            settings = _first_best(_pooled_per_day(rows), [settings] + [label for label in sizes if label != settings])
        chosen[model] = settings
    return chosen


def picks_table(scores: pd.DataFrame, candidates: list[Candidate], choices: dict, years) -> pd.DataFrame:
    """For each choice made on earlier years (a test year, chosen on the ones before it, and "final", chosen on all
    of them) and each model with list sizes to choose from: each list size of its chosen settings (_list_sizes), with
    what select compared on those years (the return per day, pooled) beside their mean yearly return and mean list
    (stocks_per_list), and for a test year what each did in it, out of sample (PICKS_COLUMNS)."""
    by_label = {candidate.label: candidate for candidate in candidates}
    rows = []
    for choice, chosen in choices.items():
        earlier = list(years) if choice == "final" else [year for year in years if year < choice]
        if not earlier:
            continue
        tuning = scores[scores.year.isin(earlier)]
        per_day, means = _pooled_per_day(tuning), _means(tuning)
        sizes = tuning.groupby("candidate").stocks_per_list.mean()
        test = scores[scores.year == choice].set_index("candidate") if choice != "final" else None
        for model, label in chosen.items():
            options = _list_sizes(candidates, label)
            if len(options) < 2:
                continue
            for option in options:
                row = {"choice": str(choice), "chosen_on": "-".join(map(str, earlier)), "model": model,
                       "candidate": option, "picks": str(dict(by_label[option].params).get("picks")),
                       "chosen": option == label, "stocks_per_list": sizes.get(option, np.nan),
                       "tuning_return_per_day": per_day.get(option, np.nan),
                       "tuning_mean_yearly_return": means.get(option, np.nan)}
                if test is not None and option in test.index:
                    row.update({"test_return_per_day": test.return_per_day[option],
                                "test_yearly_return": test.yearly_return[option],
                                "test_invested_share": test.invested_share[option]})
                rows.append(row)
    return pd.DataFrame(rows, columns=PICKS_COLUMNS)


def _strict_mean(values) -> float:
    """The mean; NaN when there is no value or one is NaN, so that a year without a return never helps."""
    array = np.asarray(list(values), dtype=float)
    return float(array.mean()) if len(array) and not np.isnan(array).any() else np.nan


def _finite(value) -> float | None:
    return float(value) if value is not None and np.isfinite(value) else None


def _book(run: Run, panel: Panel, config: BacktestConfig) -> Book:
    """A run's book: the one it kept as its year's choice, else its lists priced again (account.book)."""
    if run.book is not None:
        return run.book
    return book(panel, run.candidate.model, run.periods, run.lists, config.order_value, run.rule)


def _account_return(runs: dict, labels: list, year: int, panel: Panel, config: BacktestConfig) -> float:
    """The yearly return of the account trading these candidates' lists together through `year`, from the one
    capital, as the account trades its year's choice; 0 with no candidate: an account that stands aside."""
    if not labels:
        return 0.0
    books = [_book(runs[label, year], panel, config) for label in labels]
    return trade(panel.sessions, books, config.cost_bps, config.whole_shares, capital=config.capital).yearly_return


def _idle_cash(year: int, names, members, chosen: dict, runs: dict, account: Ledger, span: tuple[int, int],
               growths: dict, panel: Panel, config: BacktestConfig) -> list[dict]:
    """Rows of the idle_cash table (IDLE_COLUMNS) for one test year: the account, then each strategy trading alone as
    chosen, with its idle cash kept as cash (its own ledger), then held in each ticker of growths (sweep_growth, None
    for one missing from the data) through account.trade's sweep. The same lots, bought while the cash covers them.
    An account that trades nothing that year holds the ticker with all of its capital through `span`, the year's
    first entry to its last exit, as SPY held does."""
    rows = []

    def add(strategy: str, idle: str, ledger: Ledger):
        risk = metrics.ledger_performance(ledger.path)  # Its daily moves within the year, on the capital.
        rows.append({"strategy": strategy, "idle_cash": idle, "year": year, "yearly_return": ledger.yearly_return,
                     "profit": float(ledger.profit.iloc[-1]) if len(ledger.profit) else np.nan,
                     "volatility": risk.get("volatility", np.nan), "max_drawdown": risk.get("max_drawdown", np.nan)})

    def held(books: list[Book], growth: np.ndarray) -> Ledger:
        return trade(panel.sessions, books, config.cost_bps, config.whole_shares, capital=config.capital, sweep=growth)

    add(ACCOUNT, IDLE_CASH, account)
    for name in names:
        add(name, IDLE_CASH, runs[chosen[name], year].ledger)
    for ticker, growth in growths.items():
        if growth is None:
            continue
        books = [runs[chosen[name], year].book for name in members]
        add(ACCOUNT, ticker, held(books, growth) if books else trade(
            panel.sessions, [hold(panel, ticker, *span, config.capital)], config.cost_bps, capital=config.capital))
        for name in names:
            add(name, ticker, held([runs[chosen[name], year].book], growth))
    return rows


def spy_table(names, years, choices: dict, runs: dict, panel: Panel, config: BacktestConfig) -> pd.DataFrame:
    """Each strategy trading alone in every test year, as chosen for that year, beside SPY bought with the same
    dollars on the same dates: each order it filled spent on SPY at its entry close and sold when it was
    (account.mirror), whatever the mirror's own cash. One row per strategy and year (SPY_COLUMNS), which
    choose_combination reads; days_with_trades counts the closes where it held a lot, the only days either holds
    anything."""
    rows = []
    for year in years:
        for name in names:
            run = runs[choices[year][name], year]
            (paid,) = run.ledger.spent
            spy = trade(panel.sessions, [mirror(panel, run.book, paid, SPY)], config.cost_bps,
                        capital=config.capital, cash_limit=False)
            rows.append({"strategy": name, "candidate": run.candidate.label, "year": year,
                         "days_with_trades": int((run.ledger.invested > 0).sum()),
                         "profit": float(run.ledger.profit.iloc[-1]), "yearly_return": run.ledger.yearly_return,
                         "spy_mirror_profit": float(spy.profit.iloc[-1]), "spy_mirror_return": spy.yearly_return})
    return pd.DataFrame(rows, columns=SPY_COLUMNS)


def choose_combination(table: pd.DataFrame, names, years, before: int | None = None
                       ) -> tuple[tuple[str, ...], dict]:
    """Return the strategies the account trades, chosen on `years` of spy_table, and how each stood against SPY.

    With no year to choose on, every strategy trades: the default, as hyperparameters run their defaults. Otherwise
    a strategy qualifies when its trades beat SPY on those years: trading alone as chosen, its mean yearly return
    beats that of SPY bought with the same dollars on the same dates, so only the days it had trades open count
    (when SPY has no return there the rule cannot apply, and a strategy qualifies on a return of its own). Every
    strategy that qualifies is traded, in the strategies' order; when none does, the account trades nothing. A year
    at or after `before` raises LeakageError."""
    names, years = list(names), sorted(set(years))
    if before is not None and any(year >= before for year in years):
        raise LeakageError(f"The combination for {before} may use only earlier years; got {years}.")
    if not years:
        return tuple(names), {}
    rows = table[table.year.isin(years)]
    standing = {}
    for name in names:
        mine = rows[rows.strategy == name]
        mean, spy = _strict_mean(mine.yearly_return), _strict_mean(mine.spy_mirror_return)
        beats = None if np.isnan(spy) else bool(mean > spy)
        standing[name] = {"mean_yearly_return": _finite(mean), "spy_mirror_mean_return": _finite(spy),
                          "beats_spy": beats, "qualifies": bool(np.isfinite(mean)) and beats is not False}
    return tuple(name for name in names if standing[name]["qualifies"]), standing


def chain(ledgers: list[Ledger]) -> tuple[pd.Series, pd.Series]:
    """Join yearly ledgers into one account; return its profit and its cumulative return on capital at every close.
    Each year starts afresh from the starting capital, so its profit adds to the years before it."""
    profits, paths, profit, total = [], [], 0.0, 0.0
    for ledger in ledgers:
        if len(ledger.profit):
            profits.append(ledger.profit + profit)
            paths.append(ledger.path + total)
            profit, total = float(profits[-1].iloc[-1]), float(paths[-1].iloc[-1])
    if not profits:
        return pd.Series(dtype=float, name="profit"), pd.Series(dtype=float, name="return")
    return pd.concat(profits).rename("profit"), pd.concat(paths).rename("return")


def _scored_by_year(news: pd.DataFrame, max_tickers: int) -> dict:
    """The share of each year's eligible articles (sentiment.eligible: the ones scoring sends) that have an LLM score
    for every ticker they are scored for (sentiment.targets): 1.0 once a year is fully scored."""
    if not len(news) or "llm_scores" not in news:
        return {}
    sent = news[sentiment.eligible(news, max_tickers)]
    full = pd.Series([scores is not None and len(scores) == len(sentiment.targets(tickers))
                      for scores, tickers in zip(sent.llm_scores, sent.tickers)], index=sent.index, dtype=bool)
    scored = full.groupby(sent.published_utc.dt.year).mean()
    return {str(year): round(float(share), 4) for year, share in scored.items()}


def _mean(values) -> float:
    values = [value for value in values if pd.notna(value)]
    return float(np.mean(values)) if values else np.nan


def run_backtest(config: BacktestConfig, factories: dict[str, Callable] = MODELS,
                 data: tuple[WindowedStore, Panel, pd.DataFrame] | None = None,
                 grids: dict | None = None) -> BacktestResult:
    # Custom factories bring their own grids: the registry's describe the registry's models.
    grids = (GRIDS if factories is MODELS else {}) if grids is None else grids
    candidates = _candidates(config, factories, grids)
    by_label = {candidate.label: candidate for candidate in candidates}
    store, panel, news = data or load_data(config)
    settings = Settings(min_price=config.min_price, min_dollar_volume=config.min_dollar_volume)
    folds = walk_forward(panel.sessions, config.test_years)
    years = [fold.year for fold in folds]
    specs = {candidate.label: (candidate.model, dict(candidate.params)) for candidate in candidates}
    # A model with one candidate has nothing to choose: its defaults run in every year.
    choosing = len(candidates) > len({candidate.model for candidate in candidates})

    runs, references, extreme_rows, label_checks, versions, windows = {}, {}, [], {}, {}, {}
    score_rows, choices = [], {}
    for fold in folds:
        # Each test year's candidates are chosen on the test years before it only, so the choice is known before the
        # year trades: only the chosen candidates keep what they traded, the others their score.
        choices[fold.year] = select(pd.DataFrame(score_rows), candidates,
                                    [year for year in years if year < fold.year], before=fold.year)
        fitted, label_checks[fold.year] = _fit(specs, factories, panel, news, settings, fold.train_end)
        for candidate in candidates:
            model = fitted[candidate.label]
            versions[candidate.label] = model.version
            windows[candidate.label] = {"horizon_days": model.horizon_days, "bars_lookback": model.bars_lookback,
                                        "news_lookback_days": model.news_lookback_days, "picks": model.picks,
                                        "pick_share": getattr(model, "pick_share", None),
                                        "exit": exit_rule(model).as_dict()}
        year_runs = _trade_year(fold, candidates, fitted, panel, news, settings, config,
                                keep=set(choices[fold.year].values()))
        runs.update({(label, fold.year): run for label, run in year_runs.items()})
        score_rows += [run.score for run in year_runs.values()]
        rows, references[fold.year] = _universe_scan(fold, [fitted[candidate.label] for candidate in candidates],
                                                     panel, settings, config)
        extreme_rows += rows
    scores = pd.DataFrame(score_rows)

    # The account in each test year, on that year's choice: the strategies whose trades beat SPY on the years before.
    names = list(config.models)
    against = spy_table(names, years, choices, runs, panel, config)
    accounts, refs, order_tables, fold_rows, traded = {}, {}, [], [], {}
    growths, idle_rows = {ticker: sweep_growth(panel, ticker) for ticker in config.idle_cash}, []
    for fold in folds:
        validation = [year for year in years if year < fold.year]
        chosen = choices[fold.year]
        members, standing = choose_combination(against, names, validation, before=fold.year)
        traded[fold.year] = members
        every = [runs[chosen[name], fold.year].book for name in names]
        books = [runs[chosen[name], fold.year].book for name in members]
        if books:
            account = basis = trade(panel.sessions, books, config.cost_bps, config.whole_shares, record=True,
                                    capital=config.capital)
        else:  # No strategy's trades beat SPY: the account stands aside; the references show the year on every list.
            basis = trade(panel.sessions, every, config.cost_bps, config.whole_shares, capital=config.capital)
            account = idle_ledger(panel.sessions, min(item.start for item in every), max(item.end for item in every),
                                  config.capital)
        accounts[fold.year] = account
        span = (min(item.start for item in every), max(item.end for item in every))
        idle_rows += _idle_cash(fold.year, names, members, chosen, runs, account, span, growths, panel, config)
        reference = references[fold.year]
        refs[fold.year] = _references(panel, basis, books or every,
                                      lambda t: panel.universe(t, reference, settings).index, config)
        if len(account.orders):
            order_tables.append(account.orders.assign(fold=fold.year))
        validation_scores = scores[scores.candidate.isin(chosen.values()) & scores.year.isin(validation)]
        # The year's account, its chosen candidates trading together, evaluated on the years that chose it: in sample.
        checked = {str(year): _finite(_account_return(runs, [chosen[name] for name in members], year, panel, config))
                   for year in validation}
        fold_rows.append({
            "fold": fold.year, "train_end": fold.train_end.date(),
            "validation_years": "-".join(map(str, validation)) if validation else "",
            "tuned": bool(validation) and choosing,
            "chosen": json.dumps({model: dict(by_label[label].params) for model, label in chosen.items()},
                                 default=str),
            "validation_return": json.dumps(validation_scores.groupby("model").yearly_return.mean().to_dict()),
            "combination": json.dumps(list(members)), "against_spy": json.dumps(standing),
            "account_validation_returns": json.dumps(checked),
            "account_validation_return": _finite(_strict_mean(checked.values())),
            "spy_hold_validation_return": _finite(_strict_mean(refs[year]["spy_hold"].yearly_return
                                                               for year in validation)),
            "account_capital": account.capital, "account_profit": float(account.profit.iloc[-1]),
            "account_return": account.yearly_return,
            "spy_hold_return": refs[fold.year]["spy_hold"].yearly_return,
            "spy_mirror_return": refs[fold.year]["spy_mirror"].yearly_return,
            "first_entry": account.profit.index[0].date(), "last_exit": account.profit.index[-1].date(),
            "last_training_label_exit": label_checks.get(fold.year)})

    # The final choice, made on every test year, for the 2026 test and the API: never traded here. Each model's
    # candidate, then the combination; only the combination's models are fit, since nothing else is served.
    final = select(scores, candidates, years)
    picks = picks_table(scores, candidates, {**choices, "final": final}, years)
    final_members, final_standing = choose_combination(against, names, years)
    for name, standing in final_standing.items():
        standing["in_combination"] = name in final_members
    # The final choice trading together in every test year, as the account would: in sample, those years chose it.
    final_account = {str(year): _finite(_account_return(runs, [final[name] for name in final_members], year, panel,
                                                        config)) for year in years}
    final_models, final_check, final_cutoff = {}, None, None
    if config.final_fit:
        final_cutoff = panel.sessions[-1]
        if final_members:
            fitted, final_check = _fit({final[name]: specs[final[name]] for name in final_members}, factories,
                                       panel, news, settings, final_cutoff)
            final_models = {name: fitted[final[name]] for name in final_members}
    selection = {
        "purpose": "The choice for the 2026 final test, which the API serves: refit each model of the combination "
                   "with these hyperparameters on 2021-2025 (or load model/<name>.pkl), then, from capital in cash, "
                   "buy the stocks they recommend in slices of order_value at the 16:00 close while the cash lasts. "
                   "Not evaluated here.",
        "chosen_on_years": years, "training_end": str(final_cutoff.date()) if final_cutoff is not None else None,
        "capital": config.capital, "order_value": config.order_value, "cost_bps": config.cost_bps,
        "whole_shares": config.whole_shares,
        "min_price": config.min_price, "min_dollar_volume": config.min_dollar_volume, "models": {},
        # Every strategy whose trades beat SPY bought with the same dollars on the same dates, and the account of
        # them trading together in each of the years that chose them (in sample).
        "combination": {"models": list(final_members), "chosen_on_years": years, "strategies": final_standing,
                        "account_return_by_year": final_account,
                        "account_mean_yearly_return": _finite(_strict_mean(final_account.values()))},
        # Out of sample: each test year's account, trading the choice made on the years before it.
        "nested_estimate": {"account_return": {str(year): row["account_return"] for year, row in zip(years, fold_rows)},
                            "spy_hold_return": {str(year): row["spy_hold_return"]
                                                for year, row in zip(years, fold_rows)},
                            "combinations": {str(year): list(traded[year]) for year in years},
                            "untuned_years": [row["fold"] for row in fold_rows if not row["tuned"]]},
        "final_evaluated": False,
    }
    for model in final_members:
        label = final[model]
        mine = scores[scores.candidate == label].set_index("year").yearly_return
        selection["models"][model] = {
            "params": dict(by_label[label].params), "candidate": label, "version": versions[label],
            **windows[label], "order_value": config.order_value,
            "return_by_year": {str(year): value for year, value in mine.items()},
            "mean_yearly_return": float(mine.mean())}

    # Per strategy and scope: each test year, and the years added up.
    yearly = {name: {year: runs[choices[year][name], year].ledger for year in years} for name in names}
    yearly[ACCOUNT] = accounts
    for benchmark in BENCHMARKS:
        yearly[benchmark] = {year: refs[year][benchmark] for year in years}
    chained = {name: chain([ledgers[year] for year in years]) for name, ledgers in yearly.items()}
    period_tables, position_tables, signal_rows = [], [], []
    for name in names:
        for year in years:
            run = runs[choices[year][name], year]
            period_tables.append(run.table.assign(strategy=name, candidate=run.candidate.label))
            position_tables.append(run.positions.assign(strategy=name, candidate=run.candidate.label))
            signal_rows += run.signals
    all_periods = pd.concat(period_tables, ignore_index=True)
    signals = pd.DataFrame(signal_rows)
    orders = (pd.concat(order_tables, ignore_index=True) if order_tables
              else pd.DataFrame(columns=[*ORDER_COLUMNS, "fold"]))

    nulls, null_rows = {}, []  # Only for the lists each model trades as chosen, each day as long as its list.
    for name in names:
        keys = [(year, runs[choices[year][name], year].sizes, runs[choices[year][name], year].horizon,
                 runs[choices[year][name], year].rule) for year in years]
        for key in keys:
            if key not in nulls:
                fold = next(fold for fold in folds if fold.year == key[0])
                nulls[key] = _null_ledgers(fold, key[1], key[2], references[key[0]], panel, settings, config,
                                           key[3])
        for seed in range(config.null_seeds):
            ledgers = [nulls[key][seed] for key in keys]
            summary = metrics.ledger_performance(chain(ledgers)[1])
            null_rows.append({"strategy": name, "seed": seed,
                              "mean_yearly_return": _mean(ledger.yearly_return for ledger in ledgers),
                              "sharpe": summary.get("sharpe", np.nan)})
    null = pd.DataFrame(null_rows, columns=["strategy", "seed", "mean_yearly_return", "sharpe"])

    rows = []
    for name, ledgers in yearly.items():
        scopes = {str(year): year for year in years}
        scopes["all"] = None
        for scope, year in scopes.items():
            mine = all_periods[all_periods.strategy == name] if name in names else None
            if year is None:
                path = chained[name][1]
                row = {"strategy": name, "scope": scope, "capital": max(ledger.capital for ledger in ledgers.values()),
                       "profit": float(chained[name][0].iloc[-1]),
                       "mean_yearly_return": _mean(ledgers[year].yearly_return for year in years)}
            else:
                path = ledgers[year].path
                mine = mine[mine.fold == year] if mine is not None else None
                row = {"strategy": name, "scope": scope, "capital": ledgers[year].capital,
                       "profit": float(ledgers[year].profit.iloc[-1]), "yearly_return": ledgers[year].yearly_return}
            row.update(metrics.ledger_performance(path, mine))
            row.update(_per_day([ledgers[y] for y in years] if year is None else [ledgers[year]]))
            measure = "yearly_return" if year is not None else "mean_yearly_return"
            for benchmark in BENCHMARKS:
                if benchmark == name:
                    continue
                other = yearly[benchmark][year] if year is not None else None
                theirs = other.yearly_return if other is not None else _mean(
                    yearly[benchmark][y].yearly_return for y in years)
                row[f"excess_return_vs_{benchmark}"] = row[measure] - theirs
                reference = other.path if other is not None else chained[benchmark][1]
                row.update({f"{key}_vs_{benchmark}": value
                            for key, value in metrics.relative_daily(path, reference).items()})
            if name in names:
                scoped = signals[signals.strategy == name]
                row.update(metrics.signal_summary(scoped if year is None else scoped[scoped.fold == year]))
                if year is None and len(null):
                    mine_null = null[null.strategy == name]
                    row["null_return_percentile"] = float((mine_null.mean_yearly_return
                                                           < row["mean_yearly_return"]).mean())
                    row["null_sharpe_percentile"] = float((mine_null.sharpe < row.get("sharpe", np.nan)).mean())
            if name == ACCOUNT:
                placed = orders if year is None else orders[orders.fold == year]
                row.update({"orders_filled": int((placed.reason == "filled").sum()),
                            "orders_skipped": int(placed.reason.isin(["price_above_order", "no_entry",
                                                                      "no_cash"]).sum()),
                            "orders_no_cash": int((placed.reason == "no_cash").sum())})
            rows.append(row)
    metric_table = pd.DataFrame(rows)

    position_frame = pd.concat(position_tables, ignore_index=True) if position_tables else pd.DataFrame()
    held = position_frame[position_frame.reason == "filled"] if len(position_frame) else position_frame
    extreme = held[np.abs(np.log1p(held.forward_return)) > EXTREME] if len(held) else held
    dividends = panel.dividend_audit
    anomalies = pd.concat([
        panel.split_audit[~panel.split_audit.applied].assign(kind="split_not_in_prices"),
        dividends[dividends.reason != "applied"].assign(kind=lambda frame: "dividend_" + frame.reason),
        panel.action_audit[~panel.action_audit.applied].assign(kind="curated_action_not_applied"),
        extreme.assign(kind="position_return_beyond_2x") if len(extreme) else pd.DataFrame(),
        pd.DataFrame(extreme_rows).drop_duplicates(["ticker", "decision_date"]) if extreme_rows else pd.DataFrame(),
    ], ignore_index=True)
    exits = [table.exit_date.max() for table in period_tables if len(table)]
    exits += [ledger.profit.index[-1] for ledgers in yearly.values() for ledger in ledgers.values()]
    max_exit = max(exits).date()
    guards = {**store.audit(), "last_day_allowed": str(LAST_DAY), "max_bar_date": str(panel.sessions.max().date()),
              "max_news_published": str(news.published_utc.max()) if len(news) else None,
              "max_split_execution": str(panel.split_audit.execution_date.max().date()) if len(panel.split_audit) else None,
              "max_dividend_ex_date": str(panel.dividend_audit.ex_dividend_date.max().date())
              if len(panel.dividend_audit) else None,
              "max_exit_date": str(max_exit), "training_label_exits": label_checks,
              "selection_years": {str(year): [y for y in years if y < year] for year in years},
              "final_selection_years": years, "final_training_end": selection["training_end"],
              "final_training_label_exit": final_check, "final_evaluated": False,
              "renames": len(panel.renames),
              "splits_rekeyed": int(panel.split_audit.rekeyed_from.notna().sum()),
              "dividends_rekeyed": int(dividends.rekeyed_from.notna().sum()) if len(dividends) else 0,
              "dividends_by_reason": dividends.reason.value_counts().to_dict() if len(dividends) else {},
              "curated_actions_applied": int(panel.action_audit.applied.sum()) if len(panel.action_audit) else 0,
              "returns": "total (dividends reinvested)" if config.dividends_path else "price only",
              "news_scored_by_year": _scored_by_year(news, config.sentiment_max_tickers)}
    if max_exit > LAST_DAY or panel.sessions.max().date() > LAST_DAY or (len(news) and news.published_utc.max() >= END_UTC):
        raise LeakageError(f"The backtest reached past {LAST_DAY}: {guards}")
    profit = pd.DataFrame({name: chained[name][0] for name in yearly})
    return BacktestResult(config, pd.DataFrame(fold_rows), scores, metric_table, profit, all_periods, position_frame,
                          orders, signals, null, guards, anomalies, selection, versions, final_models,
                          against_spy=against, picks=picks, idle_cash=pd.DataFrame(idle_rows, columns=IDLE_COLUMNS))
