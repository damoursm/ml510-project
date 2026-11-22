"""A random forest on the ten daily-bar features (features.py) and the eight news features (news_features.py)
together: a joint model (models.JointModel), which reads each stock's bars and news at once, so that its trees can
weigh the news by what the price did, and the other way around.

fit(data) learns from a backtest fold's training data (models.TrainingData) how each stock went on to rank among the
day's tradable stocks, from its bars and its news on the decision day:
- Rows: every tradable stock, with news or not, on one decision day in every horizon_days sessions
  (forest.training_days), so that the holdings never overlap.
- Target: the forward return the backtest trades (from the decision's 16:00 close, over horizon_days) as a percentile rank among
  that day's tradable stocks, scaled to -1 (the worst) to 1 (the best), as for news_forest.py: on the return, the
  forest of daily-bar features (forest.py) chased the most volatile stocks, whose few large moves set its leaves.
- Features: the price features, then the news features. A stock without news in the 7 days has news counts and
  earnings_7d of 0 and its other news features NaN, as is any feature that lacks its input (a recent listing,
  news that is only a mention or not scored): scikit-learn's forests split on NaN, so the trees can treat a stock
  without news apart from one with neutral news.
predict(bars, news, tickers, cutoff) gives each ticker its expected scaled rank: 0 is the average tradable stock.
`features` can restrict the forest to some of the features: to the price features alone, it reads no news, which
measures what the news adds. feature_importances() gives the forest's impurity importances, which the dashboard
shows. The training rows are computed once per fold's data and shared by the candidates of a grid (_rows).
"""
from dataclasses import dataclass, field
import weakref
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from .features import FEATURES, LOOKBACK, decision_features, training_features
from .forest import training_days
from .news_features import NEWS_FEATURES, NEWS_LOOKBACK_DAYS, decision_news_features, training_news_features
from .news_forest import scaled_ranks

ALL_FEATURES = tuple(FEATURES + NEWS_FEATURES)
_TRAINING = weakref.WeakKeyDictionary()  # A fold's training data -> {features: its training rows}.


class _Labelled:
    """A fold's training data with only some of its labels, so that the feature functions compute theirs alone."""

    def __init__(self, data, labels: pd.DataFrame):
        self._data, self._labels = data, labels
        self.bars_lookback = getattr(data, "bars_lookback", LOOKBACK)
        self.news_lookback_days = getattr(data, "news_lookback_days", NEWS_LOOKBACK_DAYS)

    def labels(self) -> pd.DataFrame:
        return self._labels

    def bars(self) -> pd.DataFrame:
        return self._data.bars()

    def news(self) -> pd.DataFrame:
        return self._data.news()


def _rows(data, features: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """The training rows of a fold's data: the features asked for, in their order, the scaled ranks and the
    decision days; computed once per data object and set of features."""
    try:
        computed = _TRAINING.setdefault(data, {})
    except TypeError:  # Data that cannot be weakly referenced is not cached.
        computed = {}
    if features in computed:
        return computed[features]
    labels = data.labels()
    days = training_days(labels.decision_date, data.horizon_days)
    chosen = labels[labels.decision_date.isin(days) & labels.forward_return.notna()].reset_index(drop=True)
    labelled = _Labelled(data, chosen[["decision_date", "ticker"]])
    columns = {}  # Both functions return their labels' rows in order, so their columns line up.
    if set(features) & set(FEATURES):
        frame = training_features(labelled)
        columns.update({name: frame[name].to_numpy(dtype=np.float64) for name in FEATURES})
    if set(features) & set(NEWS_FEATURES):
        frame = training_news_features(labelled)
        columns.update({name: frame[name].to_numpy(dtype=np.float64) for name in NEWS_FEATURES})
    target = scaled_ranks(chosen.forward_return, chosen.decision_date).to_numpy(dtype=np.float64)
    rows = (np.column_stack([columns[name] for name in features]), target, pd.DatetimeIndex(chosen.decision_date))
    computed[features] = rows
    return rows


@dataclass
class PriceNewsForestModel:
    min_samples_leaf: int = 1_000  # Stocks a leaf averages at least: most of a stock's rank over 5 days is noise.
    max_features: float = 0.5  # Share of the features each split tries.
    n_estimators: int = 200
    max_samples: float = 0.3  # Share of the rows each tree is grown on, drawn with replacement.
    random_state: int = 0
    n_jobs: int = -1  # Every core, to fit and to predict.
    features: tuple[str, ...] = ALL_FEATURES  # What the forest reads, in this order.
    forest: RandomForestRegressor | None = field(default=None, repr=False)
    trained_on: dict = field(default_factory=dict)  # Rows, days, and the first and last decision days fit on.

    def __post_init__(self):
        self.features = tuple(self.features)
        unknown = sorted(set(self.features) - set(ALL_FEATURES))
        if not self.features or unknown or len(set(self.features)) < len(self.features):
            raise ValueError(f"features must name each of some of {list(ALL_FEATURES)} once; got "
                             f"{list(self.features)}.")

    def fit(self, data) -> None:
        """Fit a new forest on a fold's training data (models.TrainingData)."""
        features, target, days = _rows(data, self.features)
        if len(target) < 2 * self.min_samples_leaf:
            raise ValueError(f"{len(target)} training rows cannot fill leaves of {self.min_samples_leaf}.")
        forest = RandomForestRegressor(n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf,
                                       max_features=self.max_features, max_samples=self.max_samples,
                                       random_state=self.random_state, n_jobs=self.n_jobs)
        self.forest = forest.fit(features, target)
        self.trained_on = {"rows": int(len(target)), "days": int(days.nunique()),
                           "first_day": str(days.min().date()), "last_day": str(days.max().date())}
        if "news_count_7d" in self.features:  # The rows of stocks with at least one article in the 7 days.
            self.trained_on["with_news"] = int((features[:, self.features.index("news_count_7d")] > 0).sum())

    def predict(self, bars: pd.DataFrame, news: pd.DataFrame, tickers: pd.Index,
                cutoff: pd.Timestamp | None) -> pd.Series:
        """Return each ticker's expected scaled rank among the tradable stocks over the horizon, from its bars and its
        news in the 7 days before cutoff; NaN for every ticker without a cutoff (no bars)."""
        if self.forest is None:
            raise RuntimeError("The forest is not fitted: fit it on training data first.")
        tickers = pd.Index(tickers)
        if cutoff is None or not len(tickers):
            return pd.Series(np.nan, index=tickers, dtype=np.float64)
        columns = {}  # Each part is indexed by tickers, in their order.
        if set(self.features) & set(FEATURES):
            price = decision_features(bars).reindex(tickers)
            columns.update({name: price[name].to_numpy(dtype=np.float64) for name in FEATURES})
        if set(self.features) & set(NEWS_FEATURES):
            recent = decision_news_features(news, tickers, cutoff)
            columns.update({name: recent[name].to_numpy(dtype=np.float64) for name in NEWS_FEATURES})
        inputs = np.column_stack([columns[name] for name in self.features])
        return pd.Series(self.forest.predict(inputs), index=tickers)

    def feature_importances(self) -> dict[str, float]:
        """Return the forest's impurity importances, by feature; they add up to 1."""
        if self.forest is None:
            raise RuntimeError("The forest is not fitted: fit it on training data first.")
        return dict(zip(self.features, self.forest.feature_importances_.tolist()))
