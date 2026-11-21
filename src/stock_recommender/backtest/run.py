"""Run a nested walk-forward backtest over 2021-2025, never 2026: fit each model candidate on the years before a
test year, trade that year, and choose a test year's hyperparameters on earlier years only.

A candidate is a model with one hyperparameter configuration (registry.GRIDS). For each test year Y, every
candidate is fit on all the years before Y (training labels end before Y's first trade), then trades Y alone:
- at a session's close it ranks the stocks through score_day, the API's own decision path;
- it buys every stock on its list for the order value ($1,000) at the next open and sells it at the open
  horizon_days later, starting from $0 and funding each order anew;
- the holdings run back to back through the year.

That run is out of sample for Y and serves twice: as validation for every later test year, and as Y's test run.
Its yearly return is its profit over its capital: the most its open orders cost at one time.

For a test year T, each model runs the candidate with the best mean yearly return over the test years before T;
the first test year has no earlier year to choose on, so it runs the defaults. The account, the combination model,
trades every model's list the same way. The references spend the account's money otherwise: SPY held with its
capital through the year, and SPY or the equal-weight universe bought for every order it filled.

Training data never comes after the year it is scored on, and no choice for a year sees that year or a later one.
The choice made on every test year, and models fit on all of 2021-2025, are kept for the 2026 test, which this
never runs. Returns are total returns: dividends are reinvested.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from .. import sentiment
from ..config import ROOT, Settings
from ..data import DataUnavailable
from ..ensemble import Ensemble
from ..recommend import buy_and_sell, news_cutoff, score_day, trade_list
from . import metrics
from .account import Book, Ledger, basket, book, hold, holdings, mirror, spent, trade
from .panel import Panel
from .protocol import FoldData, check_test_years, news_window, schedule, walk_forward
from .registry import GRIDS, MODELS, UNTUNED, build, configurations
from .window import END_UTC, LAST_DAY, LeakageError, WindowedStore

ACCOUNT = "account"
# SPY held with the account's capital; SPY, and the equal-weight universe, bought for every order the account filled.
BENCHMARKS = ("spy_hold", "spy_mirror", "universe_ew")
EXTREME = np.log(2)  # A holding that more than doubles or halves is listed in anomalies.


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
    cost_bps: float = 5.0  # Paid on the value traded, both buying and selling.
    order_value: float = 1_000.0  # Dollars spent on every recommended stock; each order is funded anew.
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
        if self.cost_bps < 0 or not self.order_value > 0:
            raise ValueError("Costs must be non-negative and the order value positive.")
        if not self.models and not self.ensembles:
            raise ValueError("Backtest at least one model or ensemble.")
        names = list(self.strategies)
        reserved = set(BENCHMARKS) | {ACCOUNT}
        if len(set(names)) != len(names) or set(names) & reserved:
            raise ValueError(f"Strategy names must be unique and differ from {sorted(reserved)}; got {names}.")
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
    ledger: Ledger  # The candidate trading its lists alone, from $0 at the order value per stock.
    table: pd.DataFrame  # One row per holding period (account.holdings): its cost, profit and turnover.
    positions: pd.DataFrame  # One row per order of the ledger, with its return and how it exited.
    signals: list
    picks: int
    horizon: int


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


def _fit(specs: dict[str, tuple[str, dict]], config, factories, grids, panel, news, settings,
         cutoff: pd.Timestamp) -> tuple[dict, str | None]:
    """Fit fresh models for every candidate label (its model and hyperparameters), each on training data cut at
    cutoff with its own windows and horizon; an ensemble is built from its members' fits. Return them by label and
    the last training label's exit."""
    fitted, training = {}, {}
    ensembles = dict(config.ensembles)

    def fit(name: str, params: dict):
        model = build(factories[name], params)  # Fresh models: nothing carries across years.
        key = (model.horizon_days, model.bars_lookback, model.news_lookback_days)
        if key not in training:
            training[key] = FoldData(panel, news, settings, cutoff, *key)
        model.fit(training[key])
        return model

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
    """Trade every candidate through the fold's year, each on its own schedule; return its run by label."""
    year = fold.year
    models = {candidate.label: fitted[candidate.label] for candidate in candidates}
    schedules = {}
    for label, model in models.items():
        periods = schedule(panel.sessions, [fold], model.horizon_days, config.offset)[year]
        if not periods:
            raise ValueError(f"No {model.horizon_days}-session holding fits in {year}.")
        if periods[0].decision < model.bars_lookback - 1:
            raise ValueError(f"{label}'s first decision in {year} needs {model.bars_lookback} sessions of bars; "
                             "start the test years later.")
        schedules[label] = periods
    deciding: dict[int, list[str]] = {}
    for label, periods in schedules.items():
        for period in periods:
            deciding.setdefault(period.decision, []).append(label)
    news_days = max(model.news_lookback_days for model in models.values())
    ranked = {label: [] for label in models}
    for t in sorted(deciding):
        as_of = panel.sessions[t]
        day_news = news_window(news, news_cutoff(as_of), news_days)
        windows = {}
        for label in deciding[t]:
            model = models[label]
            lookback = model.bars_lookback  # Each model reads exactly its own window, as in the API.
            windows.setdefault(lookback, panel.window(t, lookback))
            try:
                table = score_day(windows[lookback], day_news, as_of, settings, model).table
            except DataUnavailable:
                table = pd.DataFrame(columns=["ticker", "close", "score"])
            ranked[label].append((table, panel.forward_returns(t, model.horizon_days, table.ticker)))
    runs = {}
    for candidate in candidates:
        model = models[candidate.label]
        picks, periods = model.picks, schedules[candidate.label]  # The model's own: never tuned.
        lists, exits, signals = [], [], []
        for period, (table, forward) in zip(periods, ranked[candidate.label], strict=True):
            chosen = trade_list(table, picks)
            lists.append(chosen)
            exits.append(forward.exit_reason.reindex(chosen.ticker).to_numpy(dtype=object))
            returns = forward.forward_return
            buys, sells = buy_and_sell(table, picks)
            signals.append({
                "strategy": candidate.model, "candidate": candidate.label, "fold": year,
                "decision_date": panel.sessions[period.decision], "scored": len(table),
                "ic": metrics.information_coefficient(table.set_index("ticker").score, returns),
                "spread": returns.reindex(buys.ticker).mean() - returns.reindex(sells.ticker).mean()
                if len(buys) and len(sells) else np.nan})
        item = book(panel, candidate.model, periods, lists, config.order_value)
        ledger = trade(panel.sessions, [item], config.cost_bps, config.whole_shares, record=True)
        table = holdings(ledger, periods, panel.sessions).assign(fold=year)
        orders = ledger.orders  # In the book's order: period by period, each list's stocks in turn.
        with np.errstate(divide="ignore", invalid="ignore"):
            returned = np.where(orders.cost > 0, orders.exit_value / orders.cost - 1, np.nan)
        exited = np.concatenate(exits) if exits else np.zeros(0, dtype=object)
        positions = orders.assign(fold=year, forward_return=returned,  # How it exited, when it was bought.
                                  exit_reason=np.where(orders.cost > 0, exited, None))
        runs[candidate.label] = Run(candidate, year, item, ledger, table, positions, signals, picks, model.horizon_days)
    return runs


def _universe_scan(fold, models: list, panel: Panel, settings: Settings, config: BacktestConfig):
    """Scan the year's universe for holdings beyond 2x, on the shortest horizon among the models and in the
    universe of the widest window; return those rows and that window."""
    year = fold.year
    holding = min(model.horizon_days for model in models)
    reference = max(model.bars_lookback for model in models)
    rows = []
    for period in schedule(panel.sessions, [fold], holding, config.offset)[year]:
        t = period.decision
        universe = panel.universe(t, reference, settings).index
        returns = panel.forward_returns(t, holding, universe).forward_return
        extreme = returns[np.abs(np.log1p(returns)) > EXTREME]
        rows += [{"kind": "universe_return_beyond_2x", "ticker": ticker, "decision_date": panel.sessions[t],
                  "forward_return": value} for ticker, value in extreme.items()]
    return rows, reference


def _references(panel: Panel, account: Ledger, books: list[Book], universe: Callable[[int], pd.Index],
                config: BacktestConfig) -> dict[str, Ledger]:
    """Spend the account's money otherwise through its year, with fractional shares: on SPY held with its capital
    from its first entry to its last exit, and on SPY or the equal-weight universe for every order it filled, with
    that order's dollars and dates."""
    amounts = [spent(item, config.whole_shares) for item in books]
    start, end = min(item.start for item in books), max(item.end for item in books)
    return {
        "spy_hold": trade(panel.sessions, [hold(panel, "SPY", start, end, account.capital)], config.cost_bps),
        "spy_mirror": trade(panel.sessions, [mirror(panel, item, paid) for item, paid in zip(books, amounts)],
                            config.cost_bps),
        "universe_ew": trade(panel.sessions, [basket(panel, item, paid, universe)
                                              for item, paid in zip(books, amounts)], config.cost_bps)}


def _random_list(universe: pd.DataFrame, rng: np.random.Generator, picks: int) -> pd.DataFrame:
    # Sorted first, so the universe's row order cannot change a draw.
    tickers = rng.permutation(np.sort(np.asarray(universe.index, dtype=object)))[:picks]
    return pd.DataFrame({"ticker": tickers, "price": universe.close.reindex(tickers).to_numpy(dtype=float)})


def _null_ledgers(fold, picks: int, horizon: int, reference: int, panel: Panel, settings: Settings,
                  config: BacktestConfig) -> list[Ledger]:
    """Trade null_seeds random lists of `picks` stocks through the year, on a model's horizon: no-skill models."""
    year = fold.year
    periods = schedule(panel.sessions, [fold], horizon, config.offset)[year]
    ledgers = []
    for seed in range(config.null_seeds):
        rng = np.random.default_rng([seed, year, picks, horizon])
        lists = [_random_list(panel.universe(period.decision, reference, settings), rng, picks) for period in periods]
        ledgers.append(trade(panel.sessions, [book(panel, "null", periods, lists, config.order_value)],
                             config.cost_bps, config.whole_shares))
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


def chain(ledgers: list[Ledger]) -> tuple[pd.Series, pd.Series]:
    """Join yearly ledgers into one account; return its profit and its cumulative return on capital at every open.
    Each year adds to the years before it, since nothing is reinvested."""
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
                                        "news_lookback_days": model.news_lookback_days}
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

    # Each test year's choice, made on the test years before it only, and the account trading it.
    choices, accounts, refs, order_tables, fold_rows = {}, {}, {}, [], []
    for fold in folds:
        validation = [year for year in years if year < fold.year]
        choices[fold.year] = chosen = select(scores, candidates, validation, before=fold.year)
        books = [runs[label, fold.year].book for label in chosen.values()]
        accounts[fold.year] = account = trade(panel.sessions, books, config.cost_bps, config.whole_shares,
                                              record=True)
        reference = references[fold.year]
        refs[fold.year] = _references(panel, account, books, lambda t: panel.universe(t, reference, settings).index,
                                      config)
        order_tables.append(account.orders.assign(fold=fold.year))
        validation_scores = scores[scores.candidate.isin(chosen.values()) & scores.year.isin(validation)]
        fold_rows.append({
            "fold": fold.year, "train_end": fold.train_end.date(),
            "validation_years": "-".join(map(str, validation)) if validation else "",
            "tuned": bool(validation) and choosing,
            "chosen": json.dumps({model: dict(by_label[label].params) for model, label in chosen.items()},
                                 default=str),
            "validation_return": json.dumps(validation_scores.groupby("model").yearly_return.mean().to_dict()),
            "account_capital": account.capital, "account_profit": float(account.profit.iloc[-1]),
            "account_return": account.yearly_return,
            "spy_hold_return": refs[fold.year]["spy_hold"].yearly_return,
            "spy_mirror_return": refs[fold.year]["spy_mirror"].yearly_return,
            "first_entry": account.profit.index[0].date(), "last_exit": account.profit.index[-1].date(),
            "last_training_label_exit": label_checks.get(fold.year)})

    # The final choice, made on every test year, for the 2026 test: never traded here.
    final = select(scores, candidates, years)
    final_models, final_check, final_cutoff = {}, None, None
    if config.final_fit:
        final_cutoff = panel.sessions[-1]
        fitted, final_check = _fit({label: specs[label] for label in final.values()}, config, factories, grids,
                                   panel, news, settings, final_cutoff)
        final_models = {model: fitted[label] for model, label in final.items()}
    selection = {
        "purpose": "The choice for the 2026 final test: refit each model with these hyperparameters on 2021-2025 "
                   "(or load model/<name>.pkl), then buy every stock each model recommends for order_value, "
                   "starting from $0. Not evaluated here.",
        "chosen_on_years": years, "training_end": str(final_cutoff.date()) if final_cutoff is not None else None,
        "order_value": config.order_value, "cost_bps": config.cost_bps, "whole_shares": config.whole_shares,
        "min_price": config.min_price, "min_dollar_volume": config.min_dollar_volume, "models": {},
        # Out of sample: each test year's account, trading the choice made on the years before it.
        "nested_estimate": {"account_return": {str(year): row["account_return"] for year, row in zip(years, fold_rows)},
                            "spy_hold_return": {str(year): row["spy_hold_return"]
                                                for year, row in zip(years, fold_rows)},
                            "untuned_years": [row["fold"] for row in fold_rows if not row["tuned"]]},
        "final_evaluated": False,
    }
    for model, label in final.items():
        mine = scores[scores.candidate == label].set_index("year").yearly_return
        selection["models"][model] = {
            "params": dict(by_label[label].params), "candidate": label, "version": versions[label],
            **windows[label], "picks": runs[label, years[-1]].picks, "order_value": config.order_value,
            "return_by_year": {str(year): value for year, value in mine.items()},
            "mean_yearly_return": float(mine.mean())}

    # Per strategy and scope: each test year, and the years added up.
    names = list(config.strategies)
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
    orders = pd.concat(order_tables, ignore_index=True)

    nulls, null_rows = {}, []  # Only for the lists each model trades as chosen.
    for name in names:
        keys = [(year, runs[choices[year][name], year].picks, runs[choices[year][name], year].horizon)
                for year in years]
        for key in keys:
            if key not in nulls:
                fold = next(fold for fold in folds if fold.year == key[0])
                nulls[key] = _null_ledgers(fold, key[1], key[2], references[key[0]], panel, settings, config)
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
                            "orders_skipped": int(placed.reason.isin(["price_above_order", "no_entry"]).sum())})
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
                          orders, signals, null, guards, anomalies, selection, versions, final_models)
