"""Run a nested walk-forward backtest over 2021-2025, never 2026: fit each model candidate on the years before a
test year, trade that year, and choose a test year's hyperparameters and account order sizes on earlier years only.

A candidate is a model with one hyperparameter configuration (registry.GRIDS). For each test year Y, every
candidate is fit on all the years before Y (training labels end before Y's first trade), then trades Y alone:
- at a session's close it ranks the stocks through score_day, the API's own decision path;
- it buys its list equal-weight at the next open and sells at the open horizon_days later;
- the holdings run back to back through the year.

That run is out of sample for Y and serves twice: as validation for every later test year, and as Y's test run.

For a test year T, each model runs the candidate with the best mean annualized log growth over the test years
before T. The $10,000 account trades every model's list with a per-model order size (a share of its value per
order), tuned on those same years. The first test year has no earlier year to tune on, so it runs the defaults.

Training data never comes after the year it is scored on, and no choice for a year sees that year or a later one.
The choice made on every test year, and models fit on all of 2021-2025, are kept for the 2026 test, which this
never runs. Returns are total returns: dividends are reinvested.
"""
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from ..config import ROOT, Settings
from ..data import DataUnavailable
from ..ensemble import Ensemble
from ..recommend import buy_and_sell, news_cutoff, score_day, trade_list
from . import metrics
from .account import book, equal_split, objective, trade, tune
from .panel import Panel
from .portfolio import simulate
from .protocol import FoldData, check_test_years, news_window, schedule, walk_forward
from .registry import GRIDS, MODELS, build, configurations
from .window import END_UTC, LAST_DAY, LeakageError, WindowedStore

BENCHMARKS = ("universe_ew", "spy")
ACCOUNT = "account"
EXTREME = np.log(2)  # A holding that more than doubles or halves is listed in anomalies.
SIZES = (0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10)


@dataclass(frozen=True)
class BacktestConfig:
    models: tuple[str, ...] = ("placeholder", "random")  # Names from registry.MODELS, each tuned on its grid.
    # Weighted score blends, each backtested as one more model: (name, ((model, weight), ...)), members at
    # their default hyperparameters.
    ensembles: tuple = ()
    normalize: str = "rank"  # How an ensemble puts its members' scores on one scale: rank, zscore or none.
    test_years: tuple[int, ...] = (2022, 2023, 2024, 2025)  # Each trains on all earlier years; 2021 only trains.
    offset: int = 0  # Sessions to delay each year's first entry, to check how much the start day matters.
    cost_bps: float = 5.0  # Paid on the value traded, both buying and selling.
    capital: float = 10_000.0
    sizes: tuple[float, ...] = SIZES  # Order sizes to try, as a share of the account's value per stock.
    sweeps: int = 4  # Passes of coordinate ascent over the models when tuning the order sizes.
    whole_shares: bool = False  # Round each order down to whole shares, sized from the list price.
    final_fit: bool = True  # Fit the final choice on all of 2021-2025 for the 2026 test; never scored here.
    min_price: float = 5.0
    min_dollar_volume: float = 1_000_000.0
    null_seeds: int = 20  # Random portfolios of the same size, to see how a model ranks among no-skill ones.
    bars_dir: Path = ROOT / "data/raw/polygon/daily_bars_unadjusted"
    news_dir: Path = ROOT / "data/raw/polygon/news"
    splits_path: Path = ROOT / "data/raw/polygon/splits_2021_2025.parquet"
    dividends_path: Path | None = ROOT / "data/raw/polygon/dividends_2021_2025.parquet"  # None: price returns.
    # Spin-offs and cash deals missing from Polygon's records (None leaves them out).
    actions_path: Path | None = ROOT / "configs/corporate_actions_2021_2025.csv"
    # Renames detection misses or gets wrong, and events Polygon filed under another ticker (None: detection only).
    changes_path: Path | None = ROOT / "configs/ticker_changes_2021_2025.csv"
    strict_calendar: bool = True  # Check the trading days per year against the NYSE calendar.

    def __post_init__(self):
        if self.offset < 0 or self.null_seeds < 0 or self.sweeps < 1:
            raise ValueError("offset and null_seeds must be at least 0, and sweeps at least 1.")
        if self.cost_bps < 0 or self.capital <= 0:
            raise ValueError("Costs must be non-negative and capital positive.")
        sizes = list(self.sizes)
        if not sizes or sizes != sorted(set(sizes)) or sizes[0] < 0 or sizes[-1] > 1:
            raise ValueError(f"sizes must be distinct and ascending, between 0 and 1; got {sizes}.")
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
    scorer: str  # The label without picks: candidates that differ only in picks share one fit and one ranking.


@dataclass
class Run:
    """One candidate trading one test year, with models fit on the years before it."""
    candidate: Candidate
    year: int
    periods: list
    lists: list  # Per holding period: the stocks recommended at its decision and their price.
    curve: pd.Series
    table: pd.DataFrame  # One row per holding period, from simulate().
    positions: pd.DataFrame
    signals: list
    picks: int
    horizon: int


@dataclass
class BacktestResult:
    config: BacktestConfig
    folds: pd.DataFrame  # One row per test year: what was chosen for it, on which years, and the account's result.
    scores: pd.DataFrame  # One row per candidate and test year: the out-of-sample runs validation reads.
    metrics: pd.DataFrame  # One row per strategy (each model as chosen, the account, benchmarks) and scope.
    equity: pd.DataFrame  # Value at each open, test years chained, one column per strategy.
    periods: pd.DataFrame
    positions: pd.DataFrame
    orders: pd.DataFrame  # The account's orders in every test year.
    sizes: pd.DataFrame  # Every order-size trial of the tuning, per test year and for the final choice.
    signals: pd.DataFrame  # Per decision of each model as chosen: information coefficient and buy-minus-sell spread.
    null: pd.DataFrame  # Per model and random seed: growth, CAGR and Sharpe of random lists of the same size.
    guards: dict
    anomalies: pd.DataFrame
    selection: dict  # The final choice, made on every test year, for the 2026 test.
    versions: dict = field(default_factory=dict)
    final_models: dict = field(default_factory=dict)  # Fit on 2021-2025 for the 2026 test; never scored here.


def load_data(config: BacktestConfig) -> tuple[WindowedStore, Panel, pd.DataFrame]:
    """Read the 2021-2025 bars, splits, dividends and news through the window's guarded store."""
    store = WindowedStore(config.bars_dir, config.news_dir, config.splits_path)
    panel = Panel.load(store, config.strict_calendar, config.dividends_path, config.actions_path,
                       config.changes_path)
    return store, panel, store.read_news()


def _label(model: str, params: dict) -> str:
    return model + json.dumps(params, sort_keys=True, default=str) if params else model


def _defaults(grids: dict, model: str) -> dict:
    """A model's default hyperparameters, without picks: how an ensemble uses it."""
    return {key: value for key, value in configurations(grids.get(model, {}))[0].items() if key != "picks"}


def _candidates(config: BacktestConfig, factories: dict[str, Callable], grids: dict) -> list[Candidate]:
    """Return every model's configurations (defaults first), then each ensemble; checked before data is read."""
    members = [model for _, parts in config.ensembles for model, _ in parts]
    unknown = sorted({name for name in list(config.models) + members if name not in factories})
    if unknown:
        raise ValueError(f"Unknown models {unknown}; choose from {sorted(factories)}.")
    candidates = []
    for name in config.models:
        for params in configurations(grids.get(name, {})):
            scorer = {key: value for key, value in params.items() if key != "picks"}
            candidates.append(Candidate(name, tuple(sorted(params.items())), _label(name, params),
                                        _label(name, scorer)))
    candidates += [Candidate(name, (), name, name) for name, _ in config.ensembles]
    for candidate in candidates:
        if candidate.model in config.models:
            try:
                build(factories[candidate.model], dict(candidate.params))
            except TypeError as error:
                raise ValueError(f"{candidate.label} does not build: {error}") from error
    return candidates


def _fit(scorers: dict[str, tuple[str, dict]], config, factories, grids, panel, news, settings,
         cutoff: pd.Timestamp) -> tuple[dict, str | None]:
    """Fit fresh models for every scorer, each on training data cut at cutoff with its own windows and horizon;
    an ensemble is built from its members' fits. Return them and the last training label's exit."""
    fitted, training = {}, {}
    ensembles = dict(config.ensembles)

    def fit(name: str, params: dict):
        model = build(factories[name], params)  # Fresh models: nothing carries across years.
        key = (model.horizon_days, model.bars_lookback, model.news_lookback_days)
        if key not in training:
            training[key] = FoldData(panel, news, settings, cutoff, *key)
        model.fit(training[key])
        return model

    for scorer, (name, params) in scorers.items():
        if name not in ensembles:
            fitted[scorer] = fit(name, params)
    for name, members in config.ensembles:
        if name not in scorers:
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


def _picks(candidate: Candidate, model) -> int:
    return int(dict(candidate.params).get("picks", model.picks))


def _trade_year(fold, candidates: list[Candidate], fitted: dict, panel: Panel, news: pd.DataFrame,
                settings: Settings, config: BacktestConfig) -> dict[str, Run]:
    """Trade every candidate through the fold's year, each on its own schedule; return its run by label."""
    year = fold.year
    scorers = {candidate.scorer: fitted[candidate.scorer] for candidate in candidates}
    schedules = {}
    for scorer, model in scorers.items():
        periods = schedule(panel.sessions, [fold], model.horizon_days, config.offset)[year]
        if not periods:
            raise ValueError(f"No {model.horizon_days}-session holding fits in {year}.")
        if periods[0].decision < model.bars_lookback - 1:
            raise ValueError(f"{scorer}'s first decision in {year} needs {model.bars_lookback} sessions of bars; "
                             "start the test years later.")
        schedules[scorer] = periods
    deciding: dict[int, list[str]] = {}
    for scorer, periods in schedules.items():
        for period in periods:
            deciding.setdefault(period.decision, []).append(scorer)
    news_days = max(model.news_lookback_days for model in scorers.values())
    ranked = {scorer: [] for scorer in scorers}
    for t in sorted(deciding):
        as_of = panel.sessions[t]
        day_news = news_window(news, news_cutoff(as_of), news_days)
        windows = {}
        for scorer in deciding[t]:
            model = scorers[scorer]
            lookback = model.bars_lookback  # Each model reads exactly its own window, as in the API.
            windows.setdefault(lookback, panel.window(t, lookback))
            try:
                table = score_day(windows[lookback], day_news, as_of, settings, model).table
            except DataUnavailable:
                table = pd.DataFrame(columns=["ticker", "close", "score"])
            returns = panel.forward_returns(t, model.horizon_days, table.ticker).forward_return
            ranked[scorer].append((table, returns))
    runs = {}
    for candidate in candidates:
        model = scorers[candidate.scorer]
        picks, periods = _picks(candidate, model), schedules[candidate.scorer]
        lists, weights, signals = [], [], []
        for period, (table, returns) in zip(periods, ranked[candidate.scorer], strict=True):
            chosen = trade_list(table, picks)
            lists.append(chosen)
            weights.append(pd.Series(1.0 / picks, index=pd.Index(chosen.ticker, dtype=object)))
            buys, sells = buy_and_sell(table, picks)
            signals.append({
                "strategy": candidate.model, "candidate": candidate.label, "fold": year,
                "decision_date": panel.sessions[period.decision], "scored": len(table),
                "ic": metrics.information_coefficient(table.set_index("ticker").score, returns),
                "spread": returns.reindex(buys.ticker).mean() - returns.reindex(sells.ticker).mean()
                if len(buys) and len(sells) else np.nan})
        curve, table, positions = simulate(panel, periods, weights, config.cost_bps, 0.0, config.capital,
                                           [year] * len(periods))
        runs[candidate.label] = Run(candidate, year, periods, lists, curve, table, positions, signals, picks,
                                    model.horizon_days)
    return runs


def _random_targets(universe: pd.Index, rng: np.random.Generator, picks: int) -> pd.Series:
    chosen = rng.permutation(np.sort(np.asarray(universe, dtype=object)))[:picks]  # Row order cannot change a draw.
    return pd.Series(1.0 / picks, index=pd.Index(chosen, dtype=object))


def _benchmarks(fold, models: list, panel: Panel, settings: Settings, config: BacktestConfig):
    """Trade the equal-weight universe and SPY through the year on the shortest horizon among the models, in the
    universe of the widest window; return their curves and periods, and the extreme returns in that universe."""
    year = fold.year
    holding = min(model.horizon_days for model in models)
    reference = max(model.bars_lookback for model in models)
    periods = schedule(panel.sessions, [fold], holding, config.offset)[year]
    targets = {"universe_ew": [], "spy": []}
    extreme_rows = []
    for period in periods:
        t = period.decision
        universe = panel.universe(t, reference, settings).index
        returns = panel.forward_returns(t, holding, universe).forward_return
        extreme = returns[np.abs(np.log1p(returns)) > EXTREME]
        extreme_rows += [{"kind": "universe_return_beyond_2x", "ticker": ticker, "decision_date": panel.sessions[t],
                          "forward_return": value} for ticker, value in extreme.items()]
        targets["universe_ew"].append(pd.Series(1.0 / max(len(universe), 1), index=universe))
        targets["spy"].append(pd.Series(1.0, index=pd.Index(["SPY"], dtype=object)))
    curves, tables = {}, {}
    for name, weights in targets.items():
        curves[name], tables[name], _ = simulate(panel, periods, weights, config.cost_bps, 0.0, config.capital,
                                                 [year] * len(periods))
    return curves, tables, extreme_rows, reference


def _null_curves(fold, picks: int, horizon: int, reference: int, panel: Panel, settings: Settings,
                 config: BacktestConfig) -> list[pd.Series]:
    """Trade null_seeds random lists of `picks` stocks through the year, on a model's horizon: no-skill models."""
    year = fold.year
    periods = schedule(panel.sessions, [fold], horizon, config.offset)[year]
    curves = []
    for seed in range(config.null_seeds):
        rng = np.random.default_rng([seed, year, picks, horizon])
        weights = [_random_targets(panel.universe(period.decision, reference, settings).index, rng, picks)
                   for period in periods]
        curves.append(simulate(panel, periods, weights, config.cost_bps, 0.0, config.capital, [year] * len(periods))[0])
    return curves


def select(scores: pd.DataFrame, candidates: list[Candidate], years, before: int | None = None) -> dict[str, str]:
    """Return each model's candidate with the best mean annualized log growth over `years`: the first in grid
    order among ties, and the defaults when there is no year. A year at or after `before` raises LeakageError."""
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
        means = rows.groupby("candidate").log_growth.apply(lambda values: values.mean(skipna=False))
        values = means.reindex(labels).to_numpy(dtype=float)
        chosen[model] = labels[int(np.argmax(np.where(np.isnan(values), -np.inf, values)))]
    return chosen


def chain(curves: list[pd.Series], capital: float) -> pd.Series:
    """Join yearly curves into one: each year starts with the value the previous one ended with."""
    value, parts = capital, []
    for curve in curves:
        if len(curve):
            scaled = curve * (value / curve.iloc[0])
            parts.append(scaled)
            value = float(scaled.iloc[-1])
    return pd.concat(parts).rename("value") if parts else pd.Series(dtype=float, name="value")


def _books(panel: Panel, runs: dict, chosen: dict[str, str], years) -> dict[int, list]:
    return {year: [book(panel, model, runs[label, year].periods, runs[label, year].lists)
                   for model, label in chosen.items()] for year in years}


def run_backtest(config: BacktestConfig, factories: dict[str, Callable] = MODELS,
                 data: tuple[WindowedStore, Panel, pd.DataFrame] | None = None,
                 grids: dict | None = None) -> BacktestResult:
    # Custom factories bring their own grids: the registry's describe the registry's models.
    grids = (GRIDS if factories is MODELS else {}) if grids is None else grids
    candidates = _candidates(config, factories, grids)
    by_label = {candidate.label: candidate for candidate in candidates}
    store, panel, news = data or load_data(config)
    settings = Settings(min_price=config.min_price, min_dollar_volume=config.min_dollar_volume, refresh_time="off")
    folds = walk_forward(panel.sessions, config.test_years)
    years = [fold.year for fold in folds]
    scorers = {candidate.scorer: (candidate.model, {key: value for key, value in candidate.params if key != "picks"})
               for candidate in candidates}

    runs, bench, bench_tables, references, extreme_rows, label_checks = {}, {}, {}, {}, [], {}
    versions, windows = {}, {}
    for fold in folds:
        fitted, label_checks[fold.year] = _fit(scorers, config, factories, grids, panel, news, settings,
                                               fold.train_end)
        for candidate in candidates:
            model = fitted[candidate.scorer]
            versions[candidate.label] = model.version
            windows[candidate.label] = {"horizon_days": model.horizon_days, "bars_lookback": model.bars_lookback,
                                        "news_lookback_days": model.news_lookback_days}
        year_runs = _trade_year(fold, candidates, fitted, panel, news, settings, config)
        runs.update({(label, fold.year): run for label, run in year_runs.items()})
        bench[fold.year], bench_tables[fold.year], rows, references[fold.year] = _benchmarks(
            fold, [fitted[candidate.scorer] for candidate in candidates], panel, settings, config)
        extreme_rows += rows

    score_rows = []
    for (label, year), run in runs.items():
        summary = metrics.performance(run.curve, run.table)
        ic = pd.Series([signal["ic"] for signal in run.signals], dtype=float)
        score_rows.append({
            "model": run.candidate.model, "candidate": label, "params": json.dumps(dict(run.candidate.params),
                                                                                  default=str),
            "year": year, "growth": summary.get("growth"), "log_growth": summary.get("log_growth", np.nan),
            "daily_return": summary.get("daily_return"), "cagr": summary.get("cagr"),
            "sharpe": summary.get("sharpe"), "max_drawdown": summary.get("max_drawdown"),
            "periods": summary.get("periods"), "avg_turnover": summary.get("avg_turnover"),
            "ic_mean": float(ic.mean()) if ic.notna().any() else np.nan})
    scores = pd.DataFrame(score_rows)

    # Each test year's choice, made on the test years before it only.
    choices, sizes, account_curves, order_tables, size_traces, fold_rows = {}, {}, {}, [], [], []
    for fold in folds:
        validation = [year for year in years if year < fold.year]
        choices[fold.year] = chosen = select(scores, candidates, validation, before=fold.year)
        picks = [runs[label, fold.year].picks for label in chosen.values()]
        start = equal_split(picks, config.sizes)
        if validation:
            score = objective(panel.sessions, _books(panel, runs, chosen, validation), config.capital,
                              config.cost_bps, config.whole_shares)
            sizes[fold.year], trace = tune(score, start, config.sizes, config.sweeps)
            size_traces.append(trace.assign(choice=str(fold.year)))
            account_validation = score(sizes[fold.year])
        else:
            sizes[fold.year], account_validation = start, np.nan
        curve, orders = trade(panel.sessions, _books(panel, runs, chosen, [fold.year])[fold.year],
                              sizes[fold.year], config.capital, config.cost_bps, config.whole_shares, record=True)
        account_curves[fold.year] = curve
        order_tables.append(orders.assign(fold=fold.year))
        validation_scores = scores[scores.candidate.isin(chosen.values()) & scores.year.isin(validation)]
        fold_rows.append({
            "fold": fold.year, "train_end": fold.train_end.date(),
            "validation_years": "-".join(map(str, validation)) if validation else "",
            "tuned": bool(validation),
            "chosen": json.dumps({model: dict(by_label[label].params) for model, label in chosen.items()},
                                 default=str),
            "validation_log_growth": json.dumps(validation_scores.groupby("model").log_growth.mean().to_dict()),
            "sizes": json.dumps(dict(zip(chosen, sizes[fold.year]))),
            "account_validation_log_growth": account_validation,
            "account_growth": float(curve.iloc[-1] / curve.iloc[0]) if len(curve) else np.nan,
            "first_entry": curve.index[0].date() if len(curve) else None,
            "last_exit": curve.index[-1].date() if len(curve) else None,
            "last_training_label_exit": label_checks.get(fold.year)})

    # The final choice, made on every test year, for the 2026 test: never traded here.
    final = select(scores, candidates, years)
    final_picks = [runs[label, years[-1]].picks for label in final.values()]
    final_score = objective(panel.sessions, _books(panel, runs, final, years), config.capital, config.cost_bps,
                            config.whole_shares)
    final_sizes, trace = tune(final_score, equal_split(final_picks, config.sizes), config.sizes, config.sweeps)
    size_traces.append(trace.assign(choice="final"))
    final_models, final_check, final_cutoff = {}, None, None
    if config.final_fit:
        final_cutoff = panel.sessions[-1]
        chosen_scorers = {by_label[label].scorer: scorers[by_label[label].scorer] for label in final.values()}
        fitted, final_check = _fit(chosen_scorers, config, factories, grids, panel, news, settings, final_cutoff)
        for model, label in final.items():
            fit_model = fitted[by_label[label].scorer]
            final_models[model] = replace(fit_model, picks=_picks(by_label[label], fit_model))
    selection = {
        "purpose": "The choice for the 2026 final test: refit each model with these hyperparameters on 2021-2025 "
                   "(or load model/<name>.pkl), then trade the account with these order sizes. Not evaluated here.",
        "chosen_on_years": years, "training_end": str(final_cutoff.date()) if final_cutoff is not None else None,
        "capital": config.capital, "cost_bps": config.cost_bps, "whole_shares": config.whole_shares,
        "min_price": config.min_price, "min_dollar_volume": config.min_dollar_volume,
        # In sample: the final sizes scored on the years they were tuned on.
        "models": {}, "account": {"tuning_mean_log_growth": final_score(final_sizes)},
        # Out of sample: each test year's account, traded with the choice made on the years before it.
        "nested_estimate": {"account_growth": {str(year): row["account_growth"]
                                               for year, row in zip(years, fold_rows)},
                            "first_year_untuned": years[0]},
        "final_evaluated": False,
    }
    for (model, label), size in zip(final.items(), final_sizes):
        mine = scores[scores.candidate == label].set_index("year").log_growth
        selection["models"][model] = {
            "params": dict(by_label[label].params), "candidate": label, "version": versions[label],
            **windows[label], "picks": runs[label, years[-1]].picks, "order_size": size,
            "log_growth_by_year": {str(year): value for year, value in mine.items()},
            "mean_log_growth": float(mine.mean())}

    # Per strategy and scope: each test year, and the years chained.
    names = list(config.strategies)
    yearly = {name: {year: runs[choices[year][name], year].curve for year in years} for name in names}
    yearly[ACCOUNT] = account_curves
    for benchmark in BENCHMARKS:
        yearly[benchmark] = {year: bench[year][benchmark] for year in years}
    chained = {name: chain([curves[year] for year in years], config.capital) for name, curves in yearly.items()}
    period_tables, position_tables, signal_rows = [], [], []
    for name in names:
        for year in years:
            run = runs[choices[year][name], year]
            period_tables.append(run.table.assign(strategy=name, candidate=run.candidate.label))
            position_tables.append(run.positions.assign(strategy=name, candidate=run.candidate.label))
            signal_rows += run.signals
    for benchmark in BENCHMARKS:
        period_tables += [bench_tables[year][benchmark].assign(strategy=benchmark, candidate=benchmark)
                          for year in years]
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
                nulls[key] = _null_curves(fold, key[1], key[2], references[key[0]], panel, settings, config)
        for seed in range(config.null_seeds):
            curve = chain([nulls[key][seed] for key in keys], config.capital)
            summary = metrics.performance(curve)
            null_rows.append({"strategy": name, "seed": seed, "growth": summary["growth"],
                              "cagr": summary.get("cagr"), "sharpe": summary.get("sharpe")})
    null = pd.DataFrame(null_rows, columns=["strategy", "seed", "growth", "cagr", "sharpe"])

    rows = []
    for name, curves in yearly.items():
        scopes = {str(year): (curves[year], year) for year in years}
        scopes["all"] = (chained[name], None)
        for scope, (curve, year) in scopes.items():
            mine = all_periods[all_periods.strategy == name] if name != ACCOUNT else None
            if mine is not None and year is not None:
                mine = mine[mine.fold == year]
            row = {"strategy": name, "scope": scope, **metrics.performance(curve, mine)}
            for benchmark in BENCHMARKS:
                if benchmark != name and len(curve):
                    reference = yearly[benchmark][year] if year is not None else chained[benchmark]
                    row.update({f"{key}_vs_{benchmark}": value for key, value in metrics.relative(
                        curve, reference.loc[curve.index.min():curve.index.max()]).items()})
            if name in names:
                scoped = signals[signals.strategy == name]
                row.update(metrics.signal_summary(scoped if year is None else scoped[scoped.fold == year]))
                if year is None and len(null):
                    mine_null = null[null.strategy == name]
                    row["null_sharpe_percentile"] = float((mine_null.sharpe < row.get("sharpe", np.nan)).mean())
                    row["null_growth_percentile"] = float((mine_null.growth < row["growth"]).mean())
            if name == ACCOUNT:
                placed = orders if year is None else orders[orders.fold == year]
                row.update({"orders_filled": int(placed.reason.isin(["filled", "scaled"]).sum()),
                            "orders_scaled": int((placed.reason == "scaled").sum()),
                            "orders_skipped": int(placed.reason.isin(["price_above_order", "no_entry",
                                                                      "no_cash"]).sum())})
            rows.append(row)
    metric_table = pd.DataFrame(rows)

    position_frame = pd.concat(position_tables, ignore_index=True) if position_tables else pd.DataFrame()
    held = position_frame[position_frame.exit_reason != "no_entry"] if len(position_frame) else position_frame
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
              "returns": "total (dividends reinvested)" if config.dividends_path else "price only"}
    if max_exit > LAST_DAY or panel.sessions.max().date() > LAST_DAY or (len(news) and news.published_utc.max() >= END_UTC):
        raise LeakageError(f"The backtest reached past {LAST_DAY}: {guards}")
    sizes_table = pd.concat(size_traces, ignore_index=True) if size_traces else pd.DataFrame()
    if len(sizes_table):
        sizes_table["model"] = sizes_table.model_index.map(lambda i: names[int(i)] if pd.notna(i) else None)
        sizes_table["sizes"] = sizes_table.sizes.map(lambda values: json.dumps(dict(zip(names, values))))
        sizes_table = sizes_table.drop(columns="model_index")
    return BacktestResult(config, pd.DataFrame(fold_rows), scores, metric_table, pd.DataFrame(chained), all_periods,
                          position_frame, orders, sizes_table, signals, null, guards, anomalies, selection, versions,
                          final_models)
