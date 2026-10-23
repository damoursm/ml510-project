"""Rank the stocks to buy from the latest bars and news on disk.

A recommendation is made at the close of the bars' last day (as_of) for buying at the next open. The models
see bars through as_of and news published before as_of's 16:00 New York close: what the daily refresh
(refresh.py) has downloaded when it runs at the close. A backtest over the stored files sees the final bar
and the news as a later fetch stored it, where the API scored the 16:00 snapshot.
"""
from dataclasses import dataclass
import pandas as pd
from .config import Settings
from .data import DataUnavailable, read_bars, read_news
from .models import Models

# Exchange test symbols, which trade in the daily bars but are not securities.
TEST_SYMBOLS = r"TEST|^ZZZT|^Z[A-Z]ZZT$|^(?:ZBZX|ZEXIT|ZIEXT|ZTST)$"


@dataclass(frozen=True)
class Recommendations:
    as_of: pd.Timestamp  # Last trading day in the bars.
    news_cutoff: pd.Timestamp  # The models see only news published before it.
    news_through: pd.Timestamp | None  # Newest article the models saw, if any.
    universe_size: int  # Tickers that passed the price and liquidity filters.
    table: pd.DataFrame  # One row per scored ticker, best first.


def news_cutoff(as_of: pd.Timestamp) -> pd.Timestamp:
    """Return as_of's 16:00 New York close, in UTC."""
    return pd.Timestamp(f"{as_of.date()} 16:00", tz="America/New_York").tz_convert("UTC")


def tradable(bars: pd.DataFrame, as_of: pd.Timestamp, settings: Settings) -> pd.DataFrame:
    """Return the close and median dollar volume of the tickers that traded on as_of and pass the filters."""
    dollar_volume = (bars.close * bars.volume).groupby(bars.ticker).median()
    last = bars[(bars.date == as_of) & ~bars.ticker.str.contains(TEST_SYMBOLS)].set_index("ticker")
    universe = pd.DataFrame({"close": last.close, "dollar_volume": dollar_volume.reindex(last.index)})
    return universe[(universe.close >= settings.min_price) & (universe.dollar_volume >= settings.min_dollar_volume)]


def recommend(settings: Settings, models: Models) -> Recommendations:
    bars = read_bars(settings.bars_dir, models.bars_lookback)
    as_of = bars.date.max()
    universe = tradable(bars, as_of, settings)
    if universe.empty:
        raise DataUnavailable(f"No ticker on {as_of.date()} passes the price and dollar volume filters.")
    tickers = universe.index
    cutoff = news_cutoff(as_of)
    news = read_news(settings.news_dir, cutoff - pd.Timedelta(days=models.news_lookback_days), cutoff)
    # Tickers the price model leaves out are dropped before blending, so a fitted blend never sees NaN.
    expected_return = models.price.predict(bars[bars.ticker.isin(tickers)], tickers).reindex(tickers).dropna()
    sentiment = models.news.predict(news, tickers).reindex(tickers).fillna(0.0)
    score = pd.Series(models.blend(expected_return, sentiment.reindex(expected_return.index)),
                      index=expected_return.index)
    table = universe.assign(
        expected_return=expected_return,
        news_sentiment=sentiment,
        news_count=news.tickers.explode().value_counts().reindex(tickers, fill_value=0),
        score=score,
    ).dropna(subset=["expected_return", "score"])
    # Equal scores, as from the placeholders, rank the more liquid stock first.
    table = table.rename_axis("ticker").reset_index().sort_values(
        ["score", "dollar_volume", "ticker"], ascending=[False, False, True], ignore_index=True)
    news_through = news.published_utc.max() if len(news) else None
    return Recommendations(as_of, cutoff, news_through, len(universe), table)


def buy_and_sell(table: pd.DataFrame, limit: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a best-first table into the `limit` best scores to buy and the `limit` worst to sell, worst first.

    No ticker is in both lists, so a universe smaller than 2 * limit gives a shorter sell list.
    """
    buy = table.head(limit)
    sell = table.iloc[len(buy):].tail(limit).iloc[::-1]
    return (buy.assign(rank=range(1, len(buy) + 1)), sell.assign(rank=range(1, len(sell) + 1)))
