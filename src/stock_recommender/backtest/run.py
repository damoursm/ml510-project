"""Run a walk-forward backtest over 2021-2025: fit each fold's models, trade the test years, measure the results.

Every strategy (one model, or a weighted combination of models) and every benchmark trades the same holdings:
decide at a session's close through score_day, the API's own decision path, on the strategy's own bar window,
buy at the next open, sell at the open `holding` sessions later, and rebalance into the next decision at that
same open. Returns are total returns: dividends are reinvested.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from ..config import ROOT, Settings
from ..data import DataUnavailable
from ..ensemble import Ensemble
from ..recommend import buy_and_sell, news_cutoff, score_day
from . import metrics
from .panel import Panel
from .portfolio import simulate
from .protocol import FoldData, check_test_years, news_window, schedule, walk_forward
from .registry import MODELS
from .window import END_UTC, LAST_DAY, LeakageError, WindowedStore

BENCHMARKS = ("universe_ew", "spy")
EXTREME = np.log(2)  # A holding that more than doubles or halves is listed in anomalies.


@dataclass(frozen=True)
class BacktestConfig:
    models: tuple[str, ...] = ("placeholder", "random")  # Names from registry.MODELS, each backtested alone.
    # Weighted combinations: (name, ((model, weight), ...)), each backtested as one more strategy.
    ensembles: tuple = (("ensemble", (("placeholder", 1.0), ("random", 1.0))),)
    normalize: str = "rank"  # How an ensemble puts its members' scores on one scale: rank, zscore or none.
    test_years: tuple[int, ...] = (2022, 2023, 2024, 2025)
    refit: bool = True  # False fits once, before the first test year.
    top_n: int = 20  # Positions in each of the long and short books.
    side: str = "long"  # long, or long_short to also short the sell list.
    rebalance_every: int | None = None  # Sessions per holding; defaults to the models' horizon_days.
    offset: int = 0  # Sessions to delay the first entry, to check how much the start day matters.
    cost_bps: float = 5.0  # Paid on the value traded, both buying and selling.
    borrow_bps: float = 0.0  # A year, on the value shorted.
    capital: float = 10_000.0
    min_price: float = 5.0
    min_dollar_volume: float = 1_000_000.0
    null_seeds: int = 20  # Random portfolios of the same size, to see how a strategy ranks among no-skill ones.
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
        if self.side not in ("long", "long_short"):
            raise ValueError(f"side must be long or long_short; got {self.side!r}.")
        if self.top_n < 1 or self.offset < 0 or self.null_seeds < 0 or (self.rebalance_every or 1) < 1:
            raise ValueError("top_n and rebalance_every must be at least 1; offset and null_seeds at least 0.")
        if self.cost_bps < 0 or self.borrow_bps < 0 or self.capital <= 0:
            raise ValueError("Costs must be non-negative and capital positive.")
        if not self.models and not self.ensembles:
            raise ValueError("Backtest at least one model or ensemble.")
        names = list(self.strategies)
        if len(set(names)) != len(names) or set(names) & set(BENCHMARKS):
            raise ValueError(f"Strategy names must be unique and differ from the benchmarks {BENCHMARKS}; got {names}.")
        check_test_years(self.test_years)  # Before any data is read.

    @property
    def strategies(self) -> tuple[str, ...]:
        return tuple(self.models) + tuple(name for name, _ in self.ensembles)


@dataclass
class BacktestResult:
    config: BacktestConfig
    folds: pd.DataFrame
    metrics: pd.DataFrame  # One row per strategy or benchmark and scope (all, or a fold's year).
    equity: pd.DataFrame  # Value at each open, one column per strategy and benchmark.
    periods: pd.DataFrame
    positions: pd.DataFrame
    signals: pd.DataFrame  # Per decision: information coefficient and buy-minus-sell spread.
    null: pd.DataFrame  # Per random seed: growth, CAGR and Sharpe.
    guards: dict
    anomalies: pd.DataFrame
    versions: dict = field(default_factory=dict)


def load_data(config: BacktestConfig) -> tuple[WindowedStore, Panel, pd.DataFrame]:
    """Read the 2021-2025 bars, splits, dividends and news through the window's guarded store."""
    store = WindowedStore(config.bars_dir, config.news_dir, config.splits_path)
    panel = Panel.load(store, config.strict_calendar, config.dividends_path, config.actions_path,
                       config.changes_path)
    return store, panel, store.read_news()


def _targets(table: pd.DataFrame, top_n: int, side: str) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    buys, sells = buy_and_sell(table, top_n)
    weights = pd.Series(1.0 / top_n, index=pd.Index(buys.ticker, dtype=object))
    if side == "long_short":
        weights = pd.concat([weights, pd.Series(-1.0 / top_n, index=pd.Index(sells.ticker, dtype=object))])
    return weights, buys, sells


def _random_targets(universe: pd.Index, rng: np.random.Generator, top_n: int, side: str) -> pd.Series:
    picks = rng.permutation(np.sort(np.asarray(universe, dtype=object)))  # File row order cannot change a draw.
    weights = pd.Series(1.0 / top_n, index=pd.Index(picks[:top_n], dtype=object))
    if side == "long_short":
        shorts = picks[top_n:2 * top_n]
        weights = pd.concat([weights, pd.Series(-1.0 / top_n, index=pd.Index(shorts, dtype=object))])
    return weights


def _probe(config: BacktestConfig, factories: dict[str, Callable]) -> tuple[list[str], int]:
    """Check the models before any data is read; return their names and their common horizon."""
    names = list(dict.fromkeys(list(config.models) + [model for _, members in config.ensembles
                                                      for model, _ in members]))
    unknown = [name for name in names if name not in factories]
    if unknown:
        raise ValueError(f"Unknown models {unknown}; choose from {sorted(factories)}.")
    horizons = {factories[name]().horizon_days for name in names}
    if len(horizons) > 1:
        raise ValueError(f"All models in one backtest must share horizon_days; got {sorted(horizons)}. "
                         "Backtest them in separate runs.")
    return names, horizons.pop()


def _fit(names, factories, config, panel, news, settings, fold, horizon) -> tuple[dict, dict, str | None]:
    """Fit fresh models for a fold, each on training data cut at the fold's cutoff with its own windows."""
    fitted, training = {}, {}
    for name in names:
        model = factories[name]()  # Fresh models: nothing carries across folds.
        key = (model.bars_lookback, model.news_lookback_days)
        if key not in training:
            training[key] = FoldData(panel, news, settings, fold.train_end, horizon, *key)
        model.fit(training[key])
        fitted[name] = model
    exits = [data._labels.exit_date.max() for data in training.values()
             if data._labels is not None and len(data._labels)]
    strategies = {name: fitted[name] for name in config.models}
    for name, members in config.ensembles:
        strategies[name] = Ensemble(tuple((fitted[model], weight) for model, weight in members), config.normalize)
    return strategies, fitted, str(max(exits).date()) if exits else None


def run_backtest(config: BacktestConfig, factories: dict[str, Callable] = MODELS,
                 data: tuple[WindowedStore, Panel, pd.DataFrame] | None = None) -> BacktestResult:
    names, horizon = _probe(config, factories)
    store, panel, news = data or load_data(config)
    settings = Settings(min_price=config.min_price, min_dollar_volume=config.min_dollar_volume, refresh_time="off")
    holding = config.rebalance_every or horizon
    folds = walk_forward(panel.sessions, config.test_years, config.refit)
    periods = schedule(panel.sessions, folds, holding, config.offset)
    flat = [(fold.year, period) for fold in folds for period in periods[fold.year]]
    if not flat:
        raise ValueError("No holding fits in the test years.")

    strategies, fold_rows, label_checks = {}, [], {}
    targets = {name: [] for name in config.strategies + BENCHMARKS}
    nulls = [[] for _ in range(config.null_seeds)]
    rngs = [np.random.default_rng(seed) for seed in range(config.null_seeds)]
    signal_rows, extreme_rows = [], []
    for fold in folds:
        if fold.refit:
            strategies, fitted, label_checks[fold.year] = _fit(names, factories, config, panel, news, settings,
                                                               fold, horizon)
            reference = max(model.bars_lookback for model in fitted.values())  # Window of the benchmarks.
            news_days = max(model.news_lookback_days for model in fitted.values())
            if flat[0][1].decision < reference - 1:
                raise ValueError(f"The first decision needs {reference} sessions of bars; start the test years later.")
        for period in periods[fold.year]:
            t = period.decision
            as_of = panel.sessions[t]
            day_news = news_window(news, news_cutoff(as_of), news_days)
            windows = {}
            for name, model in strategies.items():
                lookback = model.bars_lookback  # Each strategy reads exactly its own window, as in the API.
                windows.setdefault(lookback, panel.window(t, lookback))
                try:
                    table = score_day(windows[lookback], day_news, as_of, settings, model).table
                except DataUnavailable:
                    table = pd.DataFrame(columns=["ticker", "score"])
                weights, buys, sells = _targets(table, config.top_n, config.side)
                targets[name].append(weights)
                returns = panel.forward_returns(t, holding, table.ticker).forward_return
                signal_rows.append({
                    "strategy": name, "fold": fold.year, "decision_date": as_of, "scored": len(table),
                    "ic": metrics.information_coefficient(table.set_index("ticker").score, returns),
                    "spread": returns.reindex(buys.ticker).mean() - returns.reindex(sells.ticker).mean()
                    if len(buys) and len(sells) else np.nan})
            universe = panel.universe(t, reference, settings).index
            universe_returns = panel.forward_returns(t, holding, universe).forward_return
            extreme = universe_returns[np.abs(np.log1p(universe_returns)) > EXTREME]
            extreme_rows += [{"kind": "universe_return_beyond_2x", "ticker": ticker, "decision_date": as_of,
                              "forward_return": value} for ticker, value in extreme.items()]
            targets["universe_ew"].append(pd.Series(1.0 / max(len(universe), 1), index=universe))
            targets["spy"].append(pd.Series(1.0, index=pd.Index(["SPY"], dtype=object)))
            for rng, null in zip(rngs, nulls):
                null.append(_random_targets(universe, rng, config.top_n, config.side))
        items = periods[fold.year]
        fold_rows.append({"fold": fold.year, "train_end": fold.train_end.date(), "refit": fold.refit,
                          "holdings": len(items),
                          "first_entry": panel.sessions[items[0].entry].date() if items else None,
                          "last_exit": panel.sessions[items[-1].exit].date() if items else None,
                          "last_training_label_exit": label_checks.get(fold.year) if fold.refit else None})

    period_list = [period for _, period in flat]
    fold_list = [year for year, _ in flat]
    curves, period_tables, position_tables = {}, [], []
    for name, weights in targets.items():
        curve, table, positions = simulate(panel, period_list, weights, config.cost_bps, config.borrow_bps,
                                           config.capital, fold_list)
        curves[name] = curve
        period_tables.append(table.assign(strategy=name))
        if name not in BENCHMARKS:
            position_tables.append(positions.assign(strategy=name))
    null_rows = []
    for seed, weights in enumerate(nulls):
        curve, _, _ = simulate(panel, period_list, weights, config.cost_bps, config.borrow_bps, config.capital,
                               fold_list)
        summary = metrics.performance(curve)
        null_rows.append({"seed": seed, "growth": summary["growth"], "cagr": summary["cagr"],
                          "sharpe": summary["sharpe"]})
    null = pd.DataFrame(null_rows, columns=["seed", "growth", "cagr", "sharpe"])
    equity = pd.DataFrame(curves)
    all_periods = pd.concat(period_tables, ignore_index=True)
    signals = pd.DataFrame(signal_rows)

    rows = []
    for name, curve in curves.items():
        mine = all_periods[all_periods.strategy == name]
        scopes = {"all": (curve, mine)}
        for row in fold_rows:  # A fold's scope is its own holdings, from its first entry to its last exit.
            if row["first_entry"] is not None:
                scopes[str(row["fold"])] = (curve.loc[pd.Timestamp(row["first_entry"]):pd.Timestamp(row["last_exit"])],
                                            mine[mine.fold == row["fold"]])
        for scope, (scoped_curve, scoped_periods) in scopes.items():
            row = {"strategy": name, "scope": scope, **metrics.performance(scoped_curve, scoped_periods)}
            for benchmark in BENCHMARKS:
                if benchmark != name:
                    bench = curves[benchmark].loc[scoped_curve.index.min():scoped_curve.index.max()]
                    row.update({f"{key}_vs_{benchmark}": value
                                for key, value in metrics.relative(scoped_curve, bench).items()})
            if name not in BENCHMARKS:
                scoped_signals = signals[signals.strategy == name]
                if scope != "all":
                    scoped_signals = scoped_signals[scoped_signals.fold == int(scope)]
                row.update(metrics.signal_summary(scoped_signals))
                if scope == "all" and len(null):
                    row["null_sharpe_percentile"] = float((null.sharpe < row["sharpe"]).mean())
                    row["null_growth_percentile"] = float((null.growth < row["growth"]).mean())
            rows.append(row)
    metric_table = pd.DataFrame(rows)

    positions = pd.concat(position_tables, ignore_index=True) if position_tables else pd.DataFrame()
    held = positions[positions.exit_reason != "no_entry"] if len(positions) else positions
    extreme = held[np.abs(np.log1p(held.forward_return)) > EXTREME] if len(held) else held
    dividends = panel.dividend_audit
    anomalies = pd.concat([
        panel.split_audit[~panel.split_audit.applied].assign(kind="split_not_in_prices"),
        dividends[dividends.reason != "applied"].assign(kind=lambda frame: "dividend_" + frame.reason),
        panel.action_audit[~panel.action_audit.applied].assign(kind="curated_action_not_applied"),
        extreme.assign(kind="position_return_beyond_2x") if len(extreme) else pd.DataFrame(),
        pd.DataFrame(extreme_rows).drop_duplicates(["ticker", "decision_date"]) if extreme_rows else pd.DataFrame(),
    ], ignore_index=True)
    max_exit = all_periods.exit_date.max().date()
    guards = {**store.audit(), "last_day_allowed": str(LAST_DAY), "max_bar_date": str(panel.sessions.max().date()),
              "max_news_published": str(news.published_utc.max()) if len(news) else None,
              "max_split_execution": str(panel.split_audit.execution_date.max().date()) if len(panel.split_audit) else None,
              "max_dividend_ex_date": str(panel.dividend_audit.ex_dividend_date.max().date())
              if len(panel.dividend_audit) else None,
              "max_exit_date": str(max_exit), "training_label_exits": label_checks,
              "renames": len(panel.renames),
              "splits_rekeyed": int(panel.split_audit.rekeyed_from.notna().sum()),
              "dividends_rekeyed": int(dividends.rekeyed_from.notna().sum()) if len(dividends) else 0,
              "dividends_by_reason": dividends.reason.value_counts().to_dict() if len(dividends) else {},
              "curated_actions_applied": int(panel.action_audit.applied.sum()) if len(panel.action_audit) else 0,
              "returns": "total (dividends reinvested)" if config.dividends_path else "price only"}
    if max_exit > LAST_DAY or panel.sessions.max().date() > LAST_DAY or (len(news) and news.published_utc.max() >= END_UTC):
        raise LeakageError(f"The backtest reached past {LAST_DAY}: {guards}")
    versions = {name: model.version for name, model in strategies.items()}
    return BacktestResult(config, pd.DataFrame(fold_rows), metric_table, equity, all_periods, positions, signals,
                          null, guards, anomalies, versions)
