"""The interfaces the recommender's models implement, and placeholders until trained models exist.

To plug in trained models, implement PriceModel and NewsModel (predict, and fit to train in a backtest), bundle
them in Models, then return them from load_models() to serve them, and add a factory and its hyperparameter
grid to backtest/registry.py to backtest them. A decision's output (recommend.score_day) holds the full
ranking and `trades`, the model's recommended buys: its best `picks` stocks with their price.
"""
from dataclasses import dataclass
from typing import Callable, Protocol
import numpy as np
import pandas as pd


class TrainingData(Protocol):
    """What a model may learn from before a backtest's test period; the backtest builds it (backtest/protocol.py).

    Everything stops at `cutoff`, the last trading day before the test period, so nothing from the test period
    or later can be seen, and the inputs of a training day d (snapshot, bars) hold nothing from after d; only
    labels() looks ahead, by design: they are the targets. labels() is computed on first use and kept.
    """
    cutoff: pd.Timestamp
    horizon_days: int

    def labels(self) -> pd.DataFrame:
        """Return one row per training decision day and tradable ticker: decision_date, ticker, entry_date,
        exit_date, forward_return (from the next open, over horizon_days, as the backtest trades it), and
        exit_reason. Every exit_date is on or before cutoff."""

    def snapshot(self, as_of: pd.Timestamp) -> "Snapshot":
        """Return exactly what predict() would receive for a decision at as_of's close (as_of <= cutoff)."""

    def bars(self) -> pd.DataFrame:
        """Return every ticker's bars through cutoff, each row as known on its own day: prices and volume as
        traded, plus split_factor and return_factor to compare days (see backtest.panel.Panel.history).

        Bars adjusted as of the cutoff would show a training day prices from splits that came later."""

    def news(self) -> pd.DataFrame:
        """Return the articles published before cutoff's 16:00 New York close."""


@dataclass(frozen=True)
class Snapshot:
    """The inputs of one decision: the price model gets bars, the news model gets news."""
    as_of: pd.Timestamp  # The decision is made at this day's close, to trade at the next open.
    news_cutoff: pd.Timestamp  # 16:00 New York on as_of; news is published before it.
    universe: pd.DataFrame  # Index ticker; close and dollar_volume of the tickers that pass the filters.
    bars: pd.DataFrame  # The universe's bars for the last bars_lookback trading days, split-adjusted as of as_of.
    news: pd.DataFrame  # The articles published in the news_lookback_days before news_cutoff.


class PriceModel(Protocol):
    def predict(self, bars: pd.DataFrame, tickers: pd.Index) -> pd.Series:
        """Return each ticker's expected return over the horizon, indexed by ticker.

        bars holds the daily bars of `tickers` (columns date, ticker, open, high, low, close, volume, vwap,
        transactions) for the last bars_lookback trading days. A ticker left out or NaN is not recommended.
        """

    def fit(self, data: TrainingData) -> None:
        """Train on the data before a backtest's test period. Optional: a model without fit is used as is."""


class NewsModel(Protocol):
    def predict(self, news: pd.DataFrame, tickers: pd.Index) -> pd.Series:
        """Return each ticker's news sentiment, from -1 (bearish) to 1 (bullish), indexed by ticker.

        news holds the articles (scripts/download_news.py columns) published in the news_lookback_days before
        the news cutoff, the 16:00 New York close of the bars' last day. A ticker left out counts as neutral.
        """

    def fit(self, data: TrainingData) -> None:
        """Train on the data before a backtest's test period. Optional: a model without fit is used as is."""


class PlaceholderPriceModel:
    """Expects no move for any ticker. Replace with the model trained on daily bars."""

    def fit(self, data: TrainingData) -> None:
        pass

    def predict(self, bars: pd.DataFrame, tickers: pd.Index) -> pd.Series:
        return pd.Series(0.0, index=tickers)


class PlaceholderNewsModel:
    """Rates every ticker's news as neutral. Replace with the model trained on news."""

    def fit(self, data: TrainingData) -> None:
        pass

    def predict(self, news: pd.DataFrame, tickers: pd.Index) -> pd.Series:
        return pd.Series(0.0, index=tickers)


class RandomPriceModel:
    """Scores tickers at random, the same way for a given seed and day: a no-skill baseline, not a model.

    Its backtest should track the equal-weight universe, and it gives combinations varied scores to work on.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, data: TrainingData) -> None:
        pass

    def predict(self, bars: pd.DataFrame, tickers: pd.Index) -> pd.Series:
        rng = np.random.default_rng([self.seed, pd.Timestamp(bars.date.max()).toordinal()])
        ordered = pd.Index(sorted(tickers))  # The draw does not depend on the order tickers arrive in.
        return pd.Series(rng.standard_normal(len(ordered)) * 0.01, index=ordered).reindex(tickers)


def placeholder_blend(expected_return: pd.Series, news_sentiment: pd.Series) -> pd.Series:
    """Rank on the price model alone. Replace with a fitted combination of both models' outputs."""
    return expected_return


@dataclass(frozen=True)
class Models:
    price: PriceModel
    news: NewsModel
    # (expected_return, news_sentiment) -> score, higher ranks first; both inputs share the scored tickers' index.
    blend: Callable[[pd.Series, pd.Series], pd.Series]
    version: str
    horizon_days: int  # Trading days a recommended stock is meant to be held.
    bars_lookback: int  # Trading days of bars the price model reads.
    news_lookback_days: int  # Calendar days of news the news model reads.
    picks: int = 20  # Most stocks the model recommends buying at each decision: its best scores.

    def fit(self, data: TrainingData) -> None:
        """Train each part that has a fit method, once, on the data before a test period."""
        parts = {id(part): part for part in (self.price, self.news, self.blend)}
        for part in parts.values():
            if callable(getattr(part, "fit", None)):
                part.fit(data)

    def score(self, bars: pd.DataFrame, news: pd.DataFrame, tickers: pd.Index) -> pd.DataFrame:
        """Return expected_return, news_sentiment and score for the tickers the price model scores."""
        # Tickers the price model leaves out are dropped before blending, so a fitted blend never sees NaN.
        expected_return = self.price.predict(bars, tickers).reindex(tickers).dropna()
        sentiment = self.news.predict(news, tickers).reindex(tickers).fillna(0.0)
        score = pd.Series(self.blend(expected_return, sentiment.reindex(expected_return.index)),
                          index=expected_return.index)
        return pd.DataFrame({"expected_return": expected_return, "news_sentiment": sentiment,
                             "score": score}).dropna(subset=["expected_return", "score"])


def placeholder_models() -> Models:
    """Return new, untrained placeholder models."""
    return Models(price=PlaceholderPriceModel(), news=PlaceholderNewsModel(), blend=placeholder_blend,
                  version="placeholder", horizon_days=5, bars_lookback=60, news_lookback_days=7)


def load_models() -> Models:
    """Return the models the API serves; load the trained ones from artifacts/ here once they exist.

    Backtests never call this: models trained on the test years would leak them. They build fresh models from
    backtest/registry.py instead.
    """
    return placeholder_models()
