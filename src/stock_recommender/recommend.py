"""Rank the stocks to buy and sell: from the latest files for the API, from history for a backtest.

A recommendation is made at the close of the bars' last day (as_of) for buying at the next open. The models
see bars through as_of and news published before as_of's 16:00 New York close: what the daily refresh
(refresh.py) has downloaded when it runs at the close. A backtest over the stored files sees the final bar
and the news as a later fetch stored it, where the API scored the 16:00 snapshot. Both rank through score_day.
"""
from dataclasses import dataclass
import pandas as pd
from .config import Settings
from .data import DataUnavailable, read_bars, read_news
from .models import Models, Snapshot

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
    dollar_volume = (bars.close * bars.volume).groupby(bars.ticker, observed=True).median()
    last = bars[bars.date == as_of]
    tickers = pd.Index(last.ticker.astype(str), dtype=object, name="ticker")
    real = ~tickers.str.contains(TEST_SYMBOLS)
    last, tickers = last[real], tickers[real]
    universe = pd.DataFrame({"close": last.close.to_numpy(),
                             "dollar_volume": dollar_volume.reindex(last.ticker).to_numpy()}, index=tickers)
    return universe[(universe.close >= settings.min_price) & (universe.dollar_volume >= settings.min_dollar_volume)]


def decision_inputs(bars: pd.DataFrame, news: pd.DataFrame, as_of: pd.Timestamp, settings: Settings,
                    news_lookback_days: int) -> Snapshot:
    """Return what the models see for a decision at as_of's close: the tradable universe, its bars and the news.

    bars should hold the last bars_lookback trading days through as_of; rows after as_of are dropped, and so is
    any article outside the news_lookback_days before as_of's 16:00 New York close.
    """
    bars = bars[bars.date <= as_of]
    cutoff = news_cutoff(as_of)
    news = news[(news.published_utc >= cutoff - pd.Timedelta(days=news_lookback_days))
                & (news.published_utc < cutoff)]
    universe = tradable(bars, as_of, settings)
    model_bars = bars[bars.ticker.isin(universe.index)]
    if isinstance(model_bars.ticker.dtype, pd.CategoricalDtype):
        model_bars = model_bars.astype({"ticker": object})  # Models get plain strings, as from the files.
    return Snapshot(as_of, cutoff, universe, model_bars, news)


def score_day(bars: pd.DataFrame, news: pd.DataFrame, as_of: pd.Timestamp, settings: Settings,
              models: Models) -> Recommendations:
    """Rank the stocks for a decision at as_of's close: the one decision path of the API and the backtest."""
    snapshot = decision_inputs(bars, news, as_of, settings, models.news_lookback_days)
    if snapshot.universe.empty:
        raise DataUnavailable(f"No ticker on {as_of.date()} passes the price and dollar volume filters.")
    scores = models.score(snapshot.bars, snapshot.news, snapshot.universe.index)
    table = snapshot.universe.join(scores, how="inner").assign(
        news_count=snapshot.news.tickers.explode().value_counts().reindex(scores.index, fill_value=0))
    # Equal scores, as from the placeholders, rank the more liquid stock first.
    table = table.rename_axis("ticker").reset_index().sort_values(
        ["score", "dollar_volume", "ticker"], ascending=[False, False, True], ignore_index=True)
    news_through = snapshot.news.published_utc.max() if len(snapshot.news) else None
    return Recommendations(as_of, snapshot.news_cutoff, news_through, len(snapshot.universe), table)


def recommend(settings: Settings, models: Models) -> Recommendations:
    """Rank the stocks for the next open from the newest files on disk."""
    bars = read_bars(settings.bars_dir, models.bars_lookback)
    as_of = bars.date.max()
    cutoff = news_cutoff(as_of)
    news = read_news(settings.news_dir, cutoff - pd.Timedelta(days=models.news_lookback_days), cutoff)
    return score_day(bars, news, as_of, settings, models)


def buy_and_sell(table: pd.DataFrame, limit: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a best-first table into the `limit` best scores to buy and the `limit` worst to sell, worst first.

    No ticker is in both lists, so a universe smaller than 2 * limit gives a shorter sell list.
    """
    buy = table.head(limit)
    sell = table.iloc[len(buy):].tail(limit).iloc[::-1]
    return (buy.assign(rank=range(1, len(buy) + 1)), sell.assign(rank=range(1, len(sell) + 1)))
