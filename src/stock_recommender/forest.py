"""A random forest price model on the ten features of the daily bars (features.py); it reads no news.

fit(data) learns from a backtest fold's training data (models.TrainingData) how each stock's return over the horizon
compares with the day's other stocks, from its features on the decision day:
- Rows: one decision day in every horizon_days sessions, counted back from the last labelled day, so that the
  holdings never overlap and nearly identical consecutive days do not count horizon_days times.
- Target: the forward return the backtest trades (from the decision's 16:00 close, over horizon_days), minus that day's mean over
  the tradable stocks, clipped at its `clip` and 1 - `clip` quantiles over the rows. The forest learns the order of
  the stocks rather than the market's move, and a few stocks that double or collapse do not steer every tree.
- Missing features (a recent listing, a halt) stay NaN: scikit-learn's forests split on them.
predict(bars, tickers) gives each ticker's expected return over the horizon above the average tradable stock's, and
feature_importances() the forest's impurity importances, which the dashboard shows. The training rows' features are
computed once per fold's data and shared by the candidates of a grid (_rows).
"""
from dataclasses import dataclass, field
import weakref
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from .features import FEATURES, LOOKBACK, decision_features, training_features

_TRAINING = weakref.WeakKeyDictionary()  # A fold's training data -> its training rows, for its other candidates.


class _Labelled:
    """A fold's training data with only some of its labels, so that training_features computes theirs alone."""

    def __init__(self, data, labels: pd.DataFrame):
        self._data, self._labels = data, labels
        self.bars_lookback = getattr(data, "bars_lookback", LOOKBACK)

    def labels(self) -> pd.DataFrame:
        return self._labels

    def bars(self) -> pd.DataFrame:
        return self._data.bars()


def training_days(days: pd.Series, every: int) -> pd.DatetimeIndex:
    """Return one of the labelled decision days in `every`, counted back from the last."""
    labelled = pd.DatetimeIndex(np.sort(pd.unique(days)))
    return labelled[::-1][::max(1, every)][::-1]


def _rows(data) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """The training rows of a fold's data: their features, their forward returns minus their day's mean, and their
    decision days; computed once per data object."""
    try:
        cached = _TRAINING.get(data)
    except TypeError:  # Data that cannot be weakly referenced is not cached.
        cached = None
    if cached is not None:
        return cached
    labels = data.labels()
    days = training_days(labels.decision_date, data.horizon_days)
    chosen = labels[labels.decision_date.isin(days) & labels.forward_return.notna()].reset_index(drop=True)
    frame = training_features(_Labelled(data, chosen))
    relative = frame.forward_return - frame.groupby("decision_date").forward_return.transform("mean")
    rows = (frame[FEATURES].to_numpy(dtype=np.float64), relative.to_numpy(dtype=np.float64),
            pd.DatetimeIndex(frame.decision_date))
    try:
        _TRAINING[data] = rows
    except TypeError:
        pass
    return rows


@dataclass
class ForestPriceModel:
    min_samples_leaf: int = 1_000  # Stocks a leaf averages at least: most of a day's return is noise.
    max_features: float = 0.5  # Share of the features each split tries.
    n_estimators: int = 200
    max_samples: float = 0.3  # Share of the rows each tree is grown on, drawn with replacement.
    clip: float = 0.01  # Quantile at which the target is clipped at both ends.
    random_state: int = 0
    n_jobs: int = -1  # Every core, to fit and to predict.
    forest: RandomForestRegressor | None = field(default=None, repr=False)
    trained_on: dict = field(default_factory=dict)  # Rows, days and the first and last decision days fit on.

    def fit(self, data) -> None:
        """Fit a new forest on a fold's training data (models.TrainingData)."""
        features, relative, days = _rows(data)
        if len(relative) < 2 * self.min_samples_leaf:
            raise ValueError(f"{len(relative)} training rows cannot fill leaves of {self.min_samples_leaf}.")
        low, high = np.nanquantile(relative, [self.clip, 1 - self.clip])
        forest = RandomForestRegressor(n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf,
                                       max_features=self.max_features, max_samples=self.max_samples,
                                       random_state=self.random_state, n_jobs=self.n_jobs)
        self.forest = forest.fit(features, np.clip(relative, low, high))
        self.trained_on = {"rows": int(len(relative)), "days": int(days.nunique()),
                           "first_day": str(days.min().date()), "last_day": str(days.max().date()),
                           "target_clip": [float(low), float(high)]}

    def predict(self, bars: pd.DataFrame, tickers: pd.Index) -> pd.Series:
        """Return each ticker's expected return over the horizon above the average tradable stock's."""
        if self.forest is None:
            raise RuntimeError("The forest is not fitted: fit it on training data first.")
        features = decision_features(bars).reindex(tickers)
        if not len(features):
            return pd.Series(dtype=np.float64, index=tickers)
        return pd.Series(self.forest.predict(features[FEATURES].to_numpy(dtype=np.float64)), index=tickers)

    def feature_importances(self) -> dict[str, float]:
        """Return the forest's impurity importances, by feature; they add up to 1."""
        if self.forest is None:
            raise RuntimeError("The forest is not fitted: fit it on training data first.")
        return dict(zip(FEATURES, self.forest.feature_importances_.tolist()))
