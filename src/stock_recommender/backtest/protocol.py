"""Walk-forward folds, the trading schedule, and the training data a fold's models may learn from.

Each test year is one fold: models are fit at the close of the last trading day before it, on data through that
day only, then trade every holding entered during the year. 2021 is for training only, so test years run
2022..2025. Training labels are purged: a decision's label counts only if its holding ended by the cutoff.
"""
from dataclasses import dataclass
import pandas as pd
from ..config import Settings
from ..models import Snapshot
from ..recommend import decision_inputs, news_cutoff
from .panel import Panel
from .window import FIRST_DAY, LAST_DAY, LeakageError


@dataclass(frozen=True)
class Fold:
    year: int  # Holdings entered this year are tested with this fold's models.
    train_end: pd.Timestamp  # Models are fit at this day's close, on data through it.
    refit: bool  # False when the models fit for an earlier fold are reused.


@dataclass(frozen=True)
class Period:
    decision: int  # Session index of the decision's close; entry is the next open.
    holding: int  # Sessions from the entry open to the exit open.

    @property
    def entry(self) -> int:
        return self.decision + 1

    @property
    def exit(self) -> int:
        return self.decision + 1 + self.holding


def check_test_years(test_years) -> list[int]:
    """Return the test years sorted, or raise unless they are consecutive years within 2022..2025."""
    years = sorted(set(test_years))
    allowed = range(FIRST_DAY.year + 1, LAST_DAY.year + 1)
    if not years or any(year not in allowed for year in years):
        raise LeakageError(f"Test years must be within {allowed.start}..{allowed.stop - 1} (2021 trains the first "
                           f"fold, and 2026 is never used); got {list(test_years)}.")
    if years != list(range(years[0], years[-1] + 1)):
        raise ValueError(f"Test years must be consecutive; got {years}.")
    return years


def walk_forward(sessions: pd.DatetimeIndex, test_years, refit: bool = True) -> list[Fold]:
    """Return one fold per test year, each trained on every earlier year (an expanding window)."""
    years = check_test_years(test_years)
    folds = []
    for position, year in enumerate(years):
        before = sessions[sessions < pd.Timestamp(year, 1, 1)]
        if before.empty:
            raise ValueError(f"No trading day before {year} to train on.")
        fits = refit or position == 0
        folds.append(Fold(year, before[-1] if fits else folds[0].train_end, fits))
    return folds


def schedule(sessions: pd.DatetimeIndex, folds: list[Fold], holding: int, offset: int = 0) -> dict[int, list[Period]]:
    """Return each fold's back-to-back holdings: each exit is the next entry, from the first trading day of the
    first test year (plus offset) until no full holding fits before the end of the last test year."""
    first = int(sessions.searchsorted(pd.Timestamp(folds[0].year, 1, 1))) + offset
    last = int(sessions.searchsorted(pd.Timestamp(folds[-1].year + 1, 1, 1))) - 1  # Last session of the last year.
    periods = {fold.year: [] for fold in folds}
    for entry in range(first, last - holding + 1, holding):
        year = sessions[entry].year
        periods[year].append(Period(entry - 1, holding))
    return periods


class FoldData:
    """The data a fold's models may learn from (models.TrainingData), cut at `cutoff`.

    It holds a copy of the panel and the news that ends at the cutoff, so nothing later exists in it, whichever
    attribute a model reaches for. Only labels() is computed once and kept; the other methods build on each call.
    """

    def __init__(self, panel: Panel, news: pd.DataFrame, settings: Settings, cutoff: pd.Timestamp,
                 horizon_days: int, bars_lookback: int, news_lookback_days: int):
        self._cutoff_index = panel.index(cutoff)
        self._panel = panel.through(self._cutoff_index)
        # An owning copy: a row slice would be a view of all the 2021-2025 news.
        self._news = news.iloc[:news.published_utc.searchsorted(news_cutoff(pd.Timestamp(cutoff)))].copy(deep=True)
        self._settings = settings
        self._labels = None
        self.cutoff = pd.Timestamp(cutoff)
        self.horizon_days = horizon_days
        self.bars_lookback = bars_lookback
        self.news_lookback_days = news_lookback_days

    def _check(self, as_of: pd.Timestamp) -> int:
        if as_of > self.cutoff:
            raise LeakageError(f"{as_of.date()} is after this fold's training cutoff {self.cutoff.date()}.")
        return self._panel.index(as_of)

    def labels(self) -> pd.DataFrame:
        if self._labels is None:
            last = self._cutoff_index - 1 - self.horizon_days  # Purge: the holding must end by the cutoff.
            frames = []
            for t in range(self.bars_lookback - 1, last + 1):
                universe = self._panel.universe(t, self.bars_lookback, self._settings)
                returns = self._panel.forward_returns(t, self.horizon_days, universe.index)
                frames.append(returns.rename_axis("ticker").reset_index().assign(
                    decision_date=self._panel.sessions[t]))
            columns = ["decision_date", "ticker", "entry_date", "exit_date", "forward_return", "exit_reason"]
            labels = pd.concat(frames, ignore_index=True)[columns] if frames else pd.DataFrame(columns=columns)
            if len(labels) and labels.exit_date.max() > self.cutoff:
                raise LeakageError("A training label ends after the cutoff.")
            self._labels = labels
        return self._labels.copy()  # A model editing its labels in place changes no other model's.

    def snapshot(self, as_of: pd.Timestamp) -> Snapshot:
        t = self._check(pd.Timestamp(as_of))
        day = self._panel.sessions[t]
        news = news_window(self._news, news_cutoff(day), self.news_lookback_days)
        return decision_inputs(self._panel.window(t, self.bars_lookback), news, day, self._settings,
                               self.news_lookback_days)

    def bars(self) -> pd.DataFrame:
        return self._panel.history()

    def news(self) -> pd.DataFrame:
        return self._news.copy()


def news_window(news: pd.DataFrame, cutoff: pd.Timestamp, days: int) -> pd.DataFrame:
    """Return the articles published in the `days` before cutoff, from a table sorted by published_utc."""
    times = news.published_utc
    return news.iloc[times.searchsorted(cutoff - pd.Timedelta(days=days)):times.searchsorted(cutoff)]

