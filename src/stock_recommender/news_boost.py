"""A gradient-boosted news model: LightGBM on the 55 extended news features (news_features.py), predicting a stock's
rank among the day's tradable stocks over the next horizon_days (20 sessions in backtest/registry.py's news_boost).

It came out of the news-model research of 2026-11-21 (a nested walk-forward over 2022-2025 on every 5th session):
on 20-session returns it had a daily IC of about 0.02, positive in each test year, against 0.005 for news_forest's
forest on 8 features over 5 sessions, and news drifts into prices over weeks rather than days.

- Rows: the stocks with an article tagged in the 7 days before a decision's news cutoff, preferred shares excluded
  (Polygon tags a bank's preferred series with the bank's news), on one decision day in every EVERY sessions; their
  labels overlap when the horizon is longer, which trees tolerate.
- Target: the forward return over horizon_days as a percentile rank among all the day's tradable stocks, scaled to
  -1..1 (news_forest.scaled_ranks), so a few volatile stocks cannot steer it.
- It is a joint model (Models.joint): it rates only those stocks and leaves the others NaN, so Models.score drops
  them and they are never picked, unlike news_forest, whose unrated stocks counted as neutral and filled its list.
"""
from dataclasses import dataclass, field
import re
import warnings
import weakref
import numpy as np
import pandas as pd
from .forest import training_days
from .news_features import EXTENDED_NEWS_FEATURES, decision_extended_features, training_extended_features
from .news_forest import _Labelled, scaled_ranks

EVERY = 5  # Sessions between the training decision days.
PREFERRED = re.compile(r"[A-Z]+p[A-Z]*")  # Polygon's preferred series, e.g. JPMpC.
_TRAINING = weakref.WeakKeyDictionary()  # A fold's training data -> its training rows, for its other candidates.


def rated(features: pd.DataFrame) -> np.ndarray:
    """Which rows the model rates: an article in the 7 days, and not a preferred share."""
    tickers = pd.Index(features.index, dtype=object).astype(str)
    return (features["news_count_7d"] > 0).to_numpy() & ~np.asarray(tickers.str.fullmatch(PREFERRED.pattern),
                                                                     dtype=bool)


def _rows(data) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """A fold's training rows: the rated stocks' features, their scaled ranks among all the day's tradable stocks,
    and their decision days; computed once per data object."""
    try:
        cached = _TRAINING.get(data)
    except TypeError:
        cached = None
    if cached is not None:
        return cached
    labels = data.labels()
    days = training_days(labels.decision_date, EVERY)
    chosen = labels[labels.decision_date.isin(days) & labels.forward_return.notna()].reset_index(drop=True)
    chosen = chosen.assign(target=scaled_ranks(chosen.forward_return, chosen.decision_date))
    frame = training_extended_features(_Labelled(data, chosen))
    frame = frame[rated(frame.set_index("ticker"))]
    rows = (frame[EXTENDED_NEWS_FEATURES].to_numpy(dtype=np.float64), frame.target.to_numpy(dtype=np.float64),
            pd.DatetimeIndex(frame.decision_date))
    try:
        _TRAINING[data] = rows
    except TypeError:
        pass
    return rows


@dataclass
class NewsBoostModel:
    num_leaves: int = 7
    min_child_samples: int = 2_000  # Stocks a leaf holds at least: a stock's 20-session rank is mostly noise.
    n_estimators: int = 200
    learning_rate: float = 0.03
    subsample: float = 0.7  # Share of the rows each tree sees.
    colsample_bytree: float = 0.7  # Share of the features each tree sees.
    reg_lambda: float = 1.0
    max_rows: int | None = 400_000  # Training rows drawn at most, at random.
    random_state: int = 0
    n_jobs: int = -1
    booster: object = field(default=None, repr=False)
    trained_on: dict = field(default_factory=dict)

    def fit(self, data) -> None:
        """Fit a new model on a fold's training data (models.TrainingData)."""
        import lightgbm

        features, target, days = _rows(data)
        if len(target) < 2 * self.min_child_samples:
            raise ValueError(f"{len(target)} training rows cannot fill leaves of {self.min_child_samples}.")
        if self.max_rows and len(target) > self.max_rows:
            keep = np.sort(np.random.default_rng(self.random_state).choice(len(target), self.max_rows, replace=False))
            features, target, days = features[keep], target[keep], days[keep]
        self.booster = lightgbm.LGBMRegressor(
            num_leaves=self.num_leaves, min_child_samples=self.min_child_samples, n_estimators=self.n_estimators,
            learning_rate=self.learning_rate, subsample=self.subsample, subsample_freq=1,
            colsample_bytree=self.colsample_bytree, reg_lambda=self.reg_lambda, random_state=self.random_state,
            n_jobs=self.n_jobs, verbose=-1).fit(features, target)
        self.trained_on = {"rows": int(len(target)), "days": int(days.nunique()),
                           "first_day": str(days.min().date()), "last_day": str(days.max().date())}

    def predict(self, bars: pd.DataFrame, news: pd.DataFrame, tickers: pd.Index,
                cutoff: pd.Timestamp | None) -> pd.Series:
        """Return each rated stock's expected scaled rank over the horizon, NaN for the others."""
        if self.booster is None:
            raise RuntimeError("The model is not fitted: fit it on training data first.")
        tickers = pd.Index(tickers)
        predicted = pd.Series(np.nan, index=tickers, dtype=np.float64)
        if cutoff is None or not len(tickers):
            return predicted
        features = decision_extended_features(news, tickers, cutoff)
        mask = rated(features)
        if mask.any():
            with warnings.catch_warnings():  # LightGBM names even array columns, then warns on arrays: harmless.
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                predicted.iloc[np.flatnonzero(mask)] = self.booster.predict(
                    features[EXTENDED_NEWS_FEATURES].to_numpy(dtype=np.float64)[mask])
        return predicted

    def feature_importances(self) -> dict[str, float]:
        """Return the share of the trees' total gain each feature brings; they add up to 1."""
        if self.booster is None:
            raise RuntimeError("The model is not fitted: fit it on training data first.")
        gains = self.booster.booster_.feature_importance(importance_type="gain").astype(float)
        total = gains.sum() or 1.0
        return dict(zip(EXTENDED_NEWS_FEATURES, (gains / total).tolist()))
