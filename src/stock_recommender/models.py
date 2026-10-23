"""The interfaces the recommender's models implement, and placeholders until trained models exist.

To plug in trained models, implement PriceModel and NewsModel, then return them from load_models().
"""
from dataclasses import dataclass
from typing import Callable, Protocol
import pandas as pd


class PriceModel(Protocol):
    def predict(self, bars: pd.DataFrame, tickers: pd.Index) -> pd.Series:
        """Return each ticker's expected return over the horizon, indexed by ticker.

        bars holds the daily bars of `tickers` (columns date, ticker, open, high, low, close, volume, vwap,
        transactions) for the last bars_lookback trading days. A ticker left out or NaN is not recommended.
        """


class NewsModel(Protocol):
    def predict(self, news: pd.DataFrame, tickers: pd.Index) -> pd.Series:
        """Return each ticker's news sentiment, from -1 (bearish) to 1 (bullish), indexed by ticker.

        news holds the articles (scripts/download_news.py columns) published in the news_lookback_days before
        the news cutoff, the 16:00 New York close of the bars' last day. A ticker left out counts as neutral.
        """


class PlaceholderPriceModel:
    """Expects no move for any ticker. Replace with the model trained on daily bars."""

    def predict(self, bars: pd.DataFrame, tickers: pd.Index) -> pd.Series:
        return pd.Series(0.0, index=tickers)


class PlaceholderNewsModel:
    """Rates every ticker's news as neutral. Replace with the model trained on news."""

    def predict(self, news: pd.DataFrame, tickers: pd.Index) -> pd.Series:
        return pd.Series(0.0, index=tickers)


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


def load_models() -> Models:
    """Return the models that score tickers; load the trained ones from artifacts/ here once they exist."""
    return Models(price=PlaceholderPriceModel(), news=PlaceholderNewsModel(), blend=placeholder_blend,
                  version="placeholder", horizon_days=5, bars_lookback=60, news_lookback_days=7)
