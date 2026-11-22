"""Run a nested walk-forward backtest over 2021-2025, never 2026: fit each model candidate on the years before a
test year, trade that year, and choose a test year's hyperparameters on earlier years only.

A candidate is a model with one hyperparameter configuration (registry.GRIDS), its exit strategy included
(exits.py). For each test year Y, every candidate is fit on all the years before Y (training labels end before Y's
first trade), then trades Y alone; candidates that differ only in their exit strategy share one fit and one
ranking:
- at every session's close (config.decide_every) it ranks the stocks through rank, the API's own decision path,
  as the API's daily lists do, at 16:00;
- from config.capital ($10,000) in cash, it buys that day's list at that close in slices of the order value ($100),
  the best-ranked first, while its cash lasts, and sells each stock at the close hold_days later, or earlier when
  its exit rule fires; the proceeds pay for later orders (account.trade);
- the last entries of the year are those whose hold_days still fit in it.

That run is out of sample for Y and serves twice: as validation for every later test year, and as Y's test run.
Its yearly return is its last equity (cash and open lots) over its starting capital, minus 1.

For a test year T, each model runs the candidate with the best mean yearly return over the test years before T;
the first test year has no earlier year to choose on, so it runs the defaults. The account, the combination model,
trades the lists of the strategies (models and ensembles) chosen for T on those same years (choose_combination):
only the strategies whose mean yearly return, each trading alone as chosen, beats SPY held over the same opens
qualify, and the set of them whose account has the best mean yearly return wins. The first test year trades every
strategy, all from the one capital; a year where none qualifies trades nothing. The references spend the account's
money otherwise: SPY held with its capital through the year, and SPY or the equal-weight universe bought for every
order it filled.

Training data never comes after the year it is scored on, and no choice for a year sees that year or a later one.
The choice made on every test year (each model's hyperparameters, then the combination), with the combination's
models fit on all of 2021-2025, is kept for the 2026 test, which this never runs, and is what the API serves.
Returns are total returns: dividends are reinvested.
"""
from dataclasses import dataclass, field, replace
from itertools import combinations
import json
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from .. import sentiment
from ..config import ROOT, Settings
from ..data import DataUnavailable
from ..ensemble import Ensemble
from ..exits import EXIT_FIELDS, ExitRule, exit_rule
from ..recommend import buy_and_sell, decision_inputs, news_cutoff, rank, trade_list
from . import metrics
from .account import CAPITAL, ORDER_COLUMNS, Book, Ledger, basket, book, hold, holdings, idle_ledger, mirror, trade
from .panel import Panel
from .protocol import FoldData, check_test_years, news_window, schedule, walk_forward
from .registry import GRIDS, MODELS, UNTUNED, build, configurations
from .window import END_UTC, LAST_DAY, LeakageError, WindowedStore

ACCOUNT = "account"
# SPY held with the account's capital; SPY, and the equal-weight universe, bought for every order the account filled.
BENCHMARKS = ("spy_hold", "spy_mirror", "universe_ew")
EXTREME = np.log(2)  # A holding that more than doubles or halves is listed in anomalies.
SPY = "SPY"  # A strategy enters the account only when it beats this ticker held over the same opens.
MAX_STRATEGIES = 10  # The account tries every set of them: 2**10 - 1 sets in each test year.


@dataclass(frozen=True)
class BacktestConfig:
    # Names from registry.MODELS (all by default); one with a grid there is tuned on it, the others run as they are.
    models: tuple[str, ...] = tuple(MODELS)
    # Weighted score blends, each backtested as one more model: (name, ((model, weight), ...)), members at
    # their default hyperparameters.
    ensembles: tuple = ()
    normalize: str = "rank"  # How an ensemble puts its members' scores on one scale: rank, zscore or none.
    test_years: tuple[int, ...] = (2022, 2023, 2024, 2025)  # Each trains on all earlier years; 2021 only trains.
    offset: int = 0  # Sessions to delay each year's first entry, to check how much the start day matters.
    # Sessions between a model's decisions: 1, every model decides and buys every session, and each stock bought is
    # sold by its exit strategy, so holdings overlap; None, back to back, every hold_days sessions.
    decide_every: int | None = 1
    cost_bps: float = 5.0  # Paid on the value traded, both buying and selling.
    capital: float = CAPITAL  # The cash every test year starts with; the account buys only what it can pay for.
    order_value: float = 100.0  # The slice of the cash each recommended stock is bought for.
    whole_shares: bool = False  # Round each order down to whole shares, sized from the list price.
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
        if not self.models and not self.ensembles:
            raise ValueError("Backtest at least one model or ensemble.")
        names = list(self.strategies)
        reserved = set(BENCHMARKS) | {ACCOUNT}
        if len(set(names)) != len(names) or set(names) & reserved:
            raise ValueError(f"Strategy names must be unique and differ from {sorted(reserved)}; got {names}.")
        if len(names) > MAX_STRATEGIES:
            raise ValueError(f"At most {MAX_STRATEGIES} models and ensembles: the account tries every set of them; "
                             f"got {len(names)}.")
        check_test_years(self.test_years)  # Before any data is read.

    @property
    def strategies(self) -> tuple[str, ...]:
        return tuple(self.models) + tuple(name for name, _ in self.ensembles)


@dataclass(frozen=True)
class Candidate:
    model: str  # The strategy it belongs to: a registered model or an ensemble.
    params: tuple  # Its hyperparameters, as sorted (name, value) pairs.
    label: str  # The model and its hyperparameters, unique within a run.


@dataclass
class Run:
    """One candidate trading one test year, with models fit on the years before it."""
    candidate: Candidate
    year: int
    book: Book  # Its recommended lists (one per holding period) priced, each stock for the order value.
    ledger: Ledger  # The candidate trading its lists alone, from the starting capital, a slice per stock.
    table: pd.DataFrame  # One row per holding period (account.holdings): its cost, profit and turnover.
    positions: pd.DataFrame  # One row per order of the ledger, with its return and how it exited.
    signals: list
    picks: int
    horizon: int  # Its holding: the hold_days of its exit rule.
    rule: ExitRule | None = None


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
    # Per set of strategies and test year: the set trading together, as chosen, and SPY held over the same opens.
    combinations: pd.DataFrame = field(default_factory=pd.DataFrame)


def load_data(config: BacktestConfig) -> tuple[WindowedStore, Panel, pd.DataFrame]:
    """Read the 2021-2025 bars, splits, dividends and news through the window's guarded store."""
    store = WindowedStore(config.bars_dir, config.news_dir, config.splits_path, config.sentiment_dir,
                          config.sentiment_model)
    panel = Panel.load(store, config.strict_calendar, config.dividends_path, config.actions_path,
                       config.changes_path)
    return store, panel, store.read_news()


def _label(model: str, params: dict) -> str:
    return model + json.dumps(params, sort_keys=True, default=str) if params else model


def _defaults(grids: dict, model: str) -> dict:
    """A model's default hyperparameters: how an ensemble uses it."""
    return configurations(grids.get(model, {}))[0]


def _candidates(config: BacktestConfig, factories: dict[str, Callable], grids: dict) -> list[Candidate]:
    """Return every model's configurations (defaults first), then each ensemble; checked before data is read."""
    members = [model for _, parts in config.ensembles for model, _ in parts]
    unknown = sorted({name for name in list(config.models) + members if name not in factories})
    if unknown:
        raise ValueError(f"Unknown models {unknown}; choose from {sorted(factories)}.")
    for name in dict.fromkeys(list(config.models) + members):
        untuned = sorted(UNTUNED & set(grids.get(name, {})))
        if untuned:
            raise ValueError(f"{name}'s grid names {untuned}: every recommended stock is bought for the order value, "
                             "so how many a model recommends is its own and never tuned.")
    candidates = []
    for name in config.models:
        for params in configurations(grids.get(name, {})):
            candidates.append(Candidate(name, tuple(sorted(params.items())), _label(name, params)))
    candidates += [Candidate(name, (), name) for name, _ in config.ensembles]
    picks: dict[str, set] = {}
    for candidate in candidates:
        if candidate.model in config.models:
            try:
                model = build(factories[candidate.model], dict(candidate.params))
            except TypeError as error:
                raise ValueError(f"{candidate.label} does not build: {error}") from error
            picks.setdefault(candidate.model, set()).add(model.picks)
    for name, values in picks.items():  # Nor through a hyperparameter of the model's own that sets picks.
        if len(values) > 1:
            raise ValueError(f"{name}'s grid changes picks ({sorted(values)}): every recommended stock is bought for "
                             "the order value, so how many a model recommends is its own and never tuned.")
    return candidates


def _trained(name: str, params: dict) -> str:
    """The label of what a candidate fits: its model and hyperparameters without its exit settings, which change
    only how its lists are traded."""
    return _label(name, {key: value for key, value in params.items() if key not in EXIT_FIELDS})


def _fit(specs: dict[str, tuple[str, dict]], config, factories, grids, panel, news, settings,
         cutoff: pd.Timestamp) -> tuple[dict, str | None]:
    """Fit fresh models for every candidate label (its model and hyperparameters), each on training data cut at
    cutoff with its own windows and horizon; candidates that differ only in their exit settings share one fit, and
    an ensemble is built from its members' fits. Return them by label and the last training label's exit."""
    fitted, training, trained = {}, {}, {}
    ensembles = dict(config.ensembles)

    def fit(name: str, params: dict):
        key = _trained(name, params)
        if key not in trained:
            model = build(factories[name], {k: v for k, v in params.items() if k not in EXIT_FIELDS})  # Fresh.
            data = (model.horizon_days, model.bars_lookback, model.news_lookback_days)
            if data not in training:
                training[data] = FoldData(panel, news, settings, cutoff, *data)
            model.fit(training[data])
            trained[key] = model
        exits = {k: v for k, v in params.items() if k in EXIT_FIELDS}
        return replace(trained[key], **exits) if exits else trained[key]

    for label, (name, params) in specs.items():
        if name not in ensembles:
            fitted[label] = fit(name, params)
    for name, members in config.ensembles:
        if name not in specs:
            continue
        parts = []
        for member, weight in members:
            params = _defaults(grids, member)
            label = _label(member, params)
            if label not in fitted:
                fitted[label] = fit(member, params)
            parts.append((fitted[label], weight))
        fitted[name] = Ensemble(tuple(parts), config.normalize)
    exits = [data._labels.exit_date.max() for data in training.values()
             if data._labels is not None and len(data._labels)]
    return fitted, str(max(exits).date()) if exits else None


def _trade_year(fold, candidates: list[Candidate], fitted: dict, panel: Panel, news: pd.DataFrame,
                settings: Settings, config: BacktestConfig) -> dict[str, Run]:
    """Trade every candidate through the fold's year, each on its schedule (a decision every config.decide_every
    sessions, by default every session) and with its own exit rule; return its run by label. On a decision day the
    models share one snapshot of the inputs per window, and candidates sharing a fit (they differ only in their exit
    settings) rank the stocks once."""
    year = fold.year
    models = {candidate.label: fitted[candidate.label] for candidate in candidates}
    rules = {label: exit_rule(model) for label, model in models.items()}
    rankers = {candidate.label: _trained(candidate.model, dict(candidate.params)) for candidate in candidates}
    schedules = {}
    for label, model in models.items():
        holding = rules[label].hold_days
        periods = schedule(panel.sessions, [fold], holding, config.offset, config.decide_every)[year]
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
    decisions, chosen, reasons, runs = {}, {}, {}, {}  # Kept per decision; the forward returns are not.
    for candidate in candidates:
        label = candidate.label
        model, rule, periods = models[label], rules[label], schedules[label]
        picks = model.picks  # The model's own: never tuned.
        lists, exits, signals = [], [], []
        for period in periods:
            ranked = (rankers[label], period.decision)
            table = tables[ranked]
            key = (*ranked, rule.hold_days)
            if ranked not in chosen:
                chosen[ranked] = trade_list(table, picks)
            if key not in decisions:  # The signal reads the whole holding, whatever the exit rule does.
                forward = panel.forward_returns(period.decision, rule.hold_days, table.ticker)
                returns = forward.forward_return
                buys, sells = buy_and_sell(table, picks)
                reasons[key] = forward.exit_reason.reindex(chosen[ranked].ticker).to_numpy(dtype=object)
                decisions[key] = {
                    "decision_date": panel.sessions[period.decision], "scored": len(table),
                    # Decisions closer than their holding share sessions (metrics.signal_summary).
                    "holding": rule.hold_days, "every": config.decide_every or rule.hold_days,
                    "ic": metrics.information_coefficient(table.set_index("ticker").score, returns),
                    "spread": returns.reindex(buys.ticker).mean() - returns.reindex(sells.ticker).mean()
                    if len(buys) and len(sells) else np.nan}
            lists.append(chosen[ranked])
            exits.append(reasons[key])
            signals.append({"strategy": candidate.model, "candidate": label, "fold": year, **decisions[key]})
        item = book(panel, candidate.model, periods, lists, config.order_value, rule)
        ledger = trade(panel.sessions, [item], config.cost_bps, config.whole_shares, record=True,
                       capital=config.capital)
        table = holdings(ledger, periods, panel.sessions).assign(fold=year)
        orders = ledger.orders  # In the book's order: period by period, each list's stocks in turn.
        with np.errstate(divide="ignore", invalid="ignore"):
            returned = np.where(orders.cost > 0, orders.exit_value / orders.cost - 1, np.nan)
        exited = np.concatenate(exits) if exits else np.zeros(0, dtype=object)
        early = orders.exit_rule.to_numpy(dtype=object)
        exited = np.where(early != "", early, exited)  # Sold by its exit rule, or at the end of its holding.
        positions = orders.assign(fold=year, forward_return=returned,  # How it exited, when it was bought.
                                  exit_reason=np.where(orders.cost > 0, exited, None))
        runs[label] = Run(candidate, year, item, ledger, table, positions, signals, picks, rule.hold_days, rule)
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
    order it filled (account.spent, one array per book), with that order's dollars and dates."""
    amounts = list(account.spent)
    start, end = min(item.start for item in books), max(item.end for item in books)
    spend = lambda items: trade(panel.sessions, items, config.cost_bps, capital=config.capital)  # noqa: E731
    return {"spy_hold": spend([hold(panel, "SPY", start, end, config.capital)]),
            "spy_mirror": spend([mirror(panel, item, paid) for item, paid in zip(books, amounts)]),
            "universe_ew": spend([basket(panel, item, paid, universe) for item, paid in zip(books, amounts)])}


def _random_list(universe: pd.DataFrame, rng: np.random.Generator, picks: int) -> pd.DataFrame:
    # Sorted first, so the universe's row order cannot change a draw.
    tickers = rng.permutation(np.sort(np.asarray(universe.index, dtype=object)))[:picks]
    return pd.DataFrame({"ticker": tickers, "price": universe.close.reindex(tickers).to_numpy(dtype=float)})


def _null_ledgers(fold, picks: int, horizon: int, reference: int, panel: Panel, settings: Settings,
                  config: BacktestConfig, rule: ExitRule | None = None) -> list[Ledger]:
    """Trade null_seeds random lists of `picks` stocks through the year, on a model's schedule and holding and with
    its exit rule: no-skill models. The draws do not depend on the rule, so rules are compared on the same lists."""
    year = fold.year
    periods = schedule(panel.sessions, [fold], horizon, config.offset, config.decide_every)[year]
    ledgers = []
    for seed in range(config.null_seeds):
        rng = np.random.default_rng([seed, year, picks, horizon])
        lists = [_random_list(panel.universe(period.decision, reference, settings), rng, picks) for period in periods]
        ledgers.append(trade(panel.sessions, [book(panel, "null", periods, lists, config.order_value, rule)],
                             config.cost_bps, config.whole_shares, capital=config.capital))
    return ledgers


def select(scores: pd.DataFrame, candidates: list[Candidate], years, before: int | None = None) -> dict[str, str]:
    """Return each model's candidate with the best mean yearly return over `years`: the first in grid order among
    ties, and the defaults when there is no year. A year at or after `before` raises LeakageError."""
    years = sorted(set(years))
    if before is not None and any(year >= before for year in years):
        raise LeakageError(f"The choice for {before} may use only earlier years; got {years}.")
    chosen = {}
    for model in dict.fromkeys(candidate.model for candidate in candidates):
        labels = [candidate.label for candidate in candidates if candidate.model == model]
        if not years:
            chosen[model] = labels[0]
            continue
        rows = scores[scores.candidate.isin(labels) & scores.year.isin(years)]
        means = rows.groupby("candidate").yearly_return.apply(lambda values: values.mean(skipna=False))
        values = means.reindex(labels).to_numpy(dtype=float)
        chosen[model] = labels[int(np.argmax(np.where(np.isnan(values), -np.inf, values)))]
    return chosen


def _sets(names) -> list[tuple[str, ...]]:
    """Every non-empty set of the strategies: the larger sets first, each size in the strategies' order."""
    names = list(names)
    return [members for size in range(len(names), 0, -1) for members in combinations(names, size)]


def _strict_mean(values) -> float:
    """The mean; NaN when there is no value or one is NaN, so that a year without a return never helps."""
    array = np.asarray(list(values), dtype=float)
    return float(array.mean()) if len(array) and not np.isnan(array).any() else np.nan


def _finite(value) -> float | None:
    return float(value) if value is not None and np.isfinite(value) else None


def combination_table(names, years, choices: dict, runs: dict, panel: Panel, config: BacktestConfig) -> pd.DataFrame:
    """Trade every non-empty set of the strategies together in every test year, each strategy as chosen for that
    year, from the one capital, and hold SPY over the same closes: one row per set and year, which
    choose_combination reads. A set of one is its strategy trading alone."""
    spy, rows = {}, []
    for year in years:
        for members in _sets(names):
            books = [runs[choices[year][name], year].book for name in members]
            ledger = trade(panel.sessions, books, config.cost_bps, config.whole_shares, capital=config.capital)
            span = (min(item.start for item in books), max(item.end for item in books))
            if span not in spy:
                spy[span] = trade(panel.sessions, [hold(panel, SPY, *span, config.capital)], config.cost_bps,
                                  capital=config.capital).yearly_return
            rows.append({"combination": "+".join(members), "members": json.dumps(list(members)),
                         "size": len(members), "year": year, "capital": ledger.capital,
                         "profit": float(ledger.profit.iloc[-1]), "yearly_return": ledger.yearly_return,
                         "spy_hold_return": spy[span]})
    return pd.DataFrame(rows)


def choose_combination(table: pd.DataFrame, names, years, before: int | None = None
                       ) -> tuple[tuple[str, ...], float, dict]:
    """Return the strategies the account trades, chosen on `years` of combination_table; that set's mean yearly
    return there; and how each strategy stood against SPY.

    With no year to choose on, every strategy trades: the default, as hyperparameters run their defaults. Otherwise
    a strategy qualifies when its mean yearly return, trading alone as chosen, beats that of SPY held over the same
    opens (when SPY has no return there the rule cannot apply, and a strategy qualifies on a return of its own). Of
    the sets of qualifying strategies, the one whose account has the best mean yearly return wins, the larger set
    first among ties. When none qualifies the set is empty: the account trades nothing. A year at or after
    `before` raises LeakageError."""
    names, years = list(names), sorted(set(years))
    if before is not None and any(year >= before for year in years):
        raise LeakageError(f"The combination for {before} may use only earlier years; got {years}.")
    if not years:
        return tuple(names), np.nan, {}
    rows = table[table.year.isin(years)]
    standing = {}
    for name in names:
        alone = rows[rows.members == json.dumps([name])]
        mean, spy = _strict_mean(alone.yearly_return), _strict_mean(alone.spy_hold_return)
        beats = None if np.isnan(spy) else bool(mean > spy)
        standing[name] = {"mean_yearly_return": _finite(mean), "spy_hold_mean_return": _finite(spy),
                          "beats_spy": beats, "qualifies": bool(np.isfinite(mean)) and beats is not False}
    sets = _sets([name for name in names if standing[name]["qualifies"]])
    if not sets:
        return (), np.nan, standing
    means = rows.groupby("members").yearly_return.apply(_strict_mean)
    values = np.array([means.get(json.dumps(list(members)), np.nan) for members in sets], dtype=float)
    pick = int(np.argmax(np.where(np.isnan(values), -np.inf, values)))  # The first among ties: the larger set.
    return sets[pick], float(values[pick]), standing


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
    for fold in folds:
        fitted, label_checks[fold.year] = _fit(specs, config, factories, grids, panel, news, settings,
                                               fold.train_end)
        for candidate in candidates:
            model = fitted[candidate.label]
            versions[candidate.label] = model.version
            windows[candidate.label] = {"horizon_days": model.horizon_days, "bars_lookback": model.bars_lookback,
                                        "news_lookback_days": model.news_lookback_days,
                                        "exit": exit_rule(model).as_dict()}
        year_runs = _trade_year(fold, candidates, fitted, panel, news, settings, config)
        runs.update({(label, fold.year): run for label, run in year_runs.items()})
        rows, references[fold.year] = _universe_scan(fold, [fitted[candidate.label] for candidate in candidates],
                                                     panel, settings, config)
        extreme_rows += rows

    score_rows = []
    for (label, year), run in runs.items():
        summary = metrics.ledger_performance(run.ledger.path, run.table)
        ic = pd.Series([signal["ic"] for signal in run.signals], dtype=float)
        score_rows.append({
            "model": run.candidate.model, "candidate": label, "params": json.dumps(dict(run.candidate.params),
                                                                                  default=str),
            "year": year, "capital": run.ledger.capital, "profit": float(run.ledger.profit.iloc[-1]),
            "yearly_return": run.ledger.yearly_return, "daily_return": summary.get("daily_return"),
            "sharpe": summary.get("sharpe"), "max_drawdown": summary.get("max_drawdown"),
            "periods": summary.get("periods"), "avg_turnover": summary.get("avg_turnover"),
            "ic_mean": float(ic.mean()) if ic.notna().any() else np.nan})
    scores = pd.DataFrame(score_rows)

    # Each test year's choice, made on the test years before it only: each model's candidate, then the strategies the
    # account trades, and the account trading them.
    names = list(config.strategies)
    choices = {fold.year: select(scores, candidates, [year for year in years if year < fold.year], before=fold.year)
               for fold in folds}
    combos = combination_table(names, years, choices, runs, panel, config)
    accounts, refs, order_tables, fold_rows, traded = {}, {}, [], [], {}
    for fold in folds:
        validation = [year for year in years if year < fold.year]
        chosen = choices[fold.year]
        members, members_return, standing = choose_combination(combos, names, validation, before=fold.year)
        traded[fold.year] = members
        every = [runs[chosen[name], fold.year].book for name in names]
        books = [runs[chosen[name], fold.year].book for name in members]
        if books:
            account = basis = trade(panel.sessions, books, config.cost_bps, config.whole_shares, record=True,
                                    capital=config.capital)
        else:  # No strategy beat SPY: the account stands aside, and the references show the year on every list.
            basis = trade(panel.sessions, every, config.cost_bps, config.whole_shares, capital=config.capital)
            account = idle_ledger(panel.sessions, min(item.start for item in every), max(item.end for item in every),
                                  config.capital)
        accounts[fold.year] = account
        reference = references[fold.year]
        refs[fold.year] = _references(panel, basis, books or every,
                                      lambda t: panel.universe(t, reference, settings).index, config)
        if len(account.orders):
            order_tables.append(account.orders.assign(fold=fold.year))
        validation_scores = scores[scores.candidate.isin(chosen.values()) & scores.year.isin(validation)]
        fold_rows.append({
            "fold": fold.year, "train_end": fold.train_end.date(),
            "validation_years": "-".join(map(str, validation)) if validation else "",
            "tuned": bool(validation) and choosing,
            "chosen": json.dumps({model: dict(by_label[label].params) for model, label in chosen.items()},
                                 default=str),
            "validation_return": json.dumps(validation_scores.groupby("model").yearly_return.mean().to_dict()),
            "combination": json.dumps(list(members)), "combination_validation_return": _finite(members_return),
            "against_spy": json.dumps(standing),
            "account_capital": account.capital, "account_profit": float(account.profit.iloc[-1]),
            "account_return": account.yearly_return,
            "spy_hold_return": refs[fold.year]["spy_hold"].yearly_return,
            "spy_mirror_return": refs[fold.year]["spy_mirror"].yearly_return,
            "first_entry": account.profit.index[0].date(), "last_exit": account.profit.index[-1].date(),
            "last_training_label_exit": label_checks.get(fold.year)})

    # The final choice, made on every test year, for the 2026 test and the API: never traded here. Each model's
    # candidate, then the combination; only the combination's models are fit, since nothing else is served.
    final = select(scores, candidates, years)
    final_members, final_return, final_standing = choose_combination(combos, names, years)
    for name, standing in final_standing.items():
        standing["in_combination"] = name in final_members
    final_models, final_check, final_cutoff = {}, None, None
    if config.final_fit:
        final_cutoff = panel.sessions[-1]
        if final_members:
            fitted, final_check = _fit({final[name]: specs[final[name]] for name in final_members}, config,
                                       factories, grids, panel, news, settings, final_cutoff)
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
        # The strategies whose mean yearly return beats SPY held, then the set of them with the best account.
        "combination": {"models": list(final_members), "chosen_on_years": years,
                        "mean_yearly_return": _finite(final_return), "strategies": final_standing},
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
            **windows[label], "picks": runs[label, years[-1]].picks, "order_value": config.order_value,
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

    nulls, null_rows = {}, []  # Only for the lists each model trades as chosen.
    for name in names:
        keys = [(year, runs[choices[year][name], year].picks, runs[choices[year][name], year].horizon,
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
                          combinations=combos)
