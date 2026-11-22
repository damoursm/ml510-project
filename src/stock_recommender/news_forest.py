"""A random forest news model on the eight news features (news_features.py); it reads no bars.

fit(data) learns from a backtest fold's training data (models.TrainingData) how stocks with a given recent news went
on to rank among the day's tradable stocks:
- Rows: the stocks with at least one article tagged in the 7 days before a decision's news cutoff, on one decision
  day in every horizon_days sessions (forest.training_days), so that the holdings never overlap. A stock without
  news has nothing for this model to read: predict leaves it out, and Models.score counts it as neutral.
- Target: the forward return the backtest trades (from the decision's 16:00 close, over horizon_days) as a percentile rank among
  all of that day's tradable stocks, news or not, scaled to -1 (the worst) to 1 (the best); 0 is the average stock.
  A rank rather than the return: on the return, the forest of daily-bar features (forest.py) chased the most
  volatile stocks, whose few large moves set its leaves' means.
- Missing features (news that is only a mention, or not scored) stay NaN: scikit-learn's forests split on them.
predict(news, tickers, cutoff) gives each stock with news its expected scaled rank: like a sentiment, from -1 to 1,
positive when its news is expected to make it beat the average tradable stock. news_blend ranks on that alone, so
the models of backtest/registry.py's news_forest use the news features only: their price model expects no move for
any stock. feature_importances() gives the forest's impurity importances, which the dashboard shows. The training
rows are computed once per fold's data and shared by the candidates of a grid (_rows).
"""
from dataclasses import dataclass, field
import weakref
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from .forest import training_days
from .news_features import NEWS_FEATURES, NEWS_LOOKBACK_DAYS, decision_news_features, training_news_features

_TRAINING = weakref.WeakKeyDictionary()  # A fold's training data -> its training rows, for its other candidates.


class _Labelled:
    """A fold's training data with only some of its labels, so that training_news_features computes theirs alone."""

    def __init__(self, data, labels: pd.DataFrame):
        self._data, self._labels = data, labels
        self.news_lookback_days = getattr(data, "news_lookback_days", NEWS_LOOKBACK_DAYS)

    def labels(self) -> pd.DataFrame:
        return self._labels

    def news(self) -> pd.DataFrame:
        return self._data.news()


def scaled_ranks(returns: pd.Series, days: pd.Series) -> pd.Series:
    """Return each return's percentile rank among its day's returns, scaled to -1..1: 2 (rank - 0.5) / n - 1, so a
    day's ranks average 0 and run from just above -1 to just below 1. Ties share their mean rank."""
    groups = returns.groupby(days)
    return 2 * (groups.rank(method="average") - 0.5) / groups.transform("count") - 1


def with_news(features: pd.DataFrame) -> np.ndarray:
    """Return which rows have at least one article tagged with their stock in the 7 days."""
    return (features["news_count_7d"] > 0).to_numpy()


def _rows(data) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, int]:
    """The training rows of a fold's data: the features of the stocks with news, their scaled ranks among all the
    tradable stocks of their day, their decision days, and how many tradable stocks those days had; computed once
    per data object."""
    try:
        cached = _TRAINING.get(data)
    except TypeError:  # Data that cannot be weakly referenced is not cached.
        cached = None
    if cached is not None:
        return cached
    labels = data.labels()
    days = training_days(labels.decision_date, data.horizon_days)
    chosen = labels[labels.decision_date.isin(days) & labels.forward_return.notna()].reset_index(drop=True)
    chosen = chosen.assign(target=scaled_ranks(chosen.forward_return, chosen.decision_date))
    frame = training_news_features(_Labelled(data, chosen))
    frame = frame[with_news(frame)]
    rows = (frame[NEWS_FEATURES].to_numpy(dtype=np.float64), frame.target.to_numpy(dtype=np.float64),
            pd.DatetimeIndex(frame.decision_date), int(len(chosen)))
    try:
        _TRAINING[data] = rows
    except TypeError:
        pass
    return rows


def news_blend(expected_return: pd.Series, news_sentiment: pd.Series) -> pd.Series:
    """Rank on the news model alone (a module function, so that a run's models pickle)."""
    return news_sentiment


@dataclass
class NewsForestModel:
    min_samples_leaf: int = 1_000  # Stocks a leaf averages at least: most of a stock's rank over 5 days is noise.
    max_features: float = 0.5  # Share of the features each split tries.
    n_estimators: int = 200
    max_samples: float = 0.3  # Share of the rows each tree is grown on, drawn with replacement.
    random_state: int = 0
    n_jobs: int = -1  # Every core, to fit and to predict.
    forest: RandomForestRegressor | None = field(default=None, repr=False)
    trained_on: dict = field(default_factory=dict)  # Rows, days, and the first and last decision days fit on.

    def fit(self, data) -> None:
        """Fit a new forest on a fold's training data (models.TrainingData)."""
        features, target, days, tradable = _rows(data)
        if len(target) < 2 * self.min_samples_leaf:
            raise ValueError(f"{len(target)} training rows with news cannot fill leaves of {self.min_samples_leaf}.")
        forest = RandomForestRegressor(n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf,
                                       max_features=self.max_features, max_samples=self.max_samples,
                                       random_state=self.random_state, n_jobs=self.n_jobs)
        self.forest = forest.fit(features, target)
        self.trained_on = {"rows": int(len(target)), "tradable_rows": tradable, "days": int(days.nunique()),
                           "first_day": str(days.min().date()), "last_day": str(days.max().date()),
                           "mean_target": float(target.mean())}  # Above 0: stocks with news beat the others.

    def predict(self, news: pd.DataFrame, tickers: pd.Index, cutoff: pd.Timestamp | None) -> pd.Series:
        """Return the expected scaled rank of each ticker with news in the 7 days before cutoff, and NaN for the
        others, which Models.score counts as neutral."""
        if self.forest is None:
            raise RuntimeError("The forest is not fitted: fit it on training data first.")
        predicted = pd.Series(np.nan, index=pd.Index(tickers), dtype=np.float64)
        if cutoff is None or not len(predicted):
            return predicted
        features = decision_news_features(news, predicted.index, cutoff)
        rated = with_news(features)
        if rated.any():
            values = self.forest.predict(features[NEWS_FEATURES].to_numpy(dtype=np.float64)[rated])
            predicted.iloc[np.flatnonzero(rated)] = values
        return predicted

    def feature_importances(self) -> dict[str, float]:
        """Return the forest's impurity importances, by feature; they add up to 1."""
        if self.forest is None:
            raise RuntimeError("The forest is not fitted: fit it on training data first.")
        return dict(zip(NEWS_FEATURES, self.forest.feature_importances_.tolist()))
