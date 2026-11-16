"""Features of each stock's recent news, from the LLM scores attached to the articles (sentiment.attach).

A decision at day t's close sees the articles published before t's 16:00 New York close (the news cutoff). The
features read those of the last NEWS_LOOKBACK_DAYS days; "1d" ones only those published since the previous
weekday's close. A pair is an article and one ticker it is tagged with. A pair is relevant when the LLM rated the
article as at least one of the company's main subjects (relevance >= 1), and it weighs relevance / 2. Sentiment
(-2..2) is scaled to -1..1, impact (0..2) to 0..1. Counts include every article tagged with the stock, scored or
not; a feature that needs a scored relevant article is NaN without one (scikit-learn's random forests accept NaN).

The two entry points compute the same features:
- decision_news_features(news, tickers, cutoff): from a decision's articles (Snapshot.news) and news cutoff;
- training_news_features(data): from a TrainingData, its labels with the features of each label's decision day.
"""
import numpy as np
import pandas as pd
from .recommend import news_cutoff

NEWS_FEATURES = [
    "news_count_7d",  # Articles tagged with the stock in the last 7 days.
    "news_count_1d",  # Those published since the previous weekday's close.
    "sentiment_7d",  # Relevance-weighted mean sentiment over the 7 days, -1 to 1.
    "sentiment_1d",  # The same since the previous weekday's close.
    "major_sentiment_7d",  # Mean sentiment of the relevant articles rated major news (impact 2).
    "impact_7d",  # Highest impact of the relevant articles: 0 nothing new, 0.5 routine news, 1 major news.
    "earnings_7d",  # 1 if a relevant article is about the company's earnings or guidance, else 0.
    "analyst_sentiment_7d",  # Mean sentiment of the relevant analyst articles (rating or price-target changes).
]
NEWS_LOOKBACK_DAYS = 7  # Days of articles before the news cutoff that the features read.
_PAIRS = ["published_utc", "ticker", "event", "relevance", "impact", "sentiment"]


def pairs(news: pd.DataFrame) -> pd.DataFrame:
    """Return one row per article and tagged ticker (a ticker tagged twice counts once), sorted by publication
    time, with the LLM's scores (NaN and None when the pair is not scored)."""
    scores = news["llm_scores"] if "llm_scores" in news else pd.Series(None, index=news.index, dtype=object)
    rows = []
    for published, tickers, scored in zip(news.published_utc, news.tickers, scores):
        by_ticker = {} if scored is None else {score["ticker"]: score for score in scored}
        for ticker in dict.fromkeys([] if tickers is None else tickers):
            score = by_ticker.get(ticker) or {}
            rows.append((published, str(ticker), score.get("event"), score.get("relevance"), score.get("impact"),
                         score.get("sentiment")))
    frame = pd.DataFrame(rows, columns=_PAIRS).astype(
        {"relevance": np.float64, "impact": np.float64, "sentiment": np.float64, "ticker": object})
    if not len(frame):
        frame["published_utc"] = pd.Series(dtype="datetime64[ms, UTC]")
    return frame.sort_values("published_utc", kind="stable", ignore_index=True)


def recent_start(cutoff: pd.Timestamp) -> pd.Timestamp:
    """Return the previous weekday's news cutoff: where a decision's last day of news starts."""
    day = pd.Timestamp(cutoff).tz_convert("America/New_York").normalize().tz_localize(None)
    return news_cutoff(day - pd.offsets.BDay(1))


def summarize(window: pd.DataFrame, recent_from: pd.Timestamp) -> pd.DataFrame:
    """Return NEWS_FEATURES, indexed by ticker, for the tickers with a pair in the window."""
    if not len(window):
        return pd.DataFrame(columns=NEWS_FEATURES, index=pd.Index([], dtype=object, name="ticker"), dtype=np.float64)
    relevance, impact = window.relevance.to_numpy(), window.impact.to_numpy()
    weight = np.nan_to_num(relevance / 2)  # 0 when not scored.
    sentiment = window.sentiment.to_numpy() / 2
    scored = np.isfinite(sentiment) & (weight > 0)
    weighted = np.where(scored, weight * sentiment, 0.0)
    weight = np.where(scored, weight, 0.0)
    relevant = scored & (relevance >= 1)
    recent = (window.published_utc >= recent_from).to_numpy()
    major = relevant & (impact == 2)
    analyst = relevant & (window.event == "analyst").to_numpy()
    parts = pd.DataFrame({
        "ticker": window.ticker.to_numpy(), "count": 1.0, "count_1d": recent.astype(np.float64),
        "weight": weight, "weighted": weighted,
        "weight_1d": np.where(recent, weight, 0.0), "weighted_1d": np.where(recent, weighted, 0.0),
        "major": major.astype(np.float64), "major_sum": np.where(major, sentiment, 0.0),
        "impact": np.where(relevant, impact / 2, np.nan),
        "earnings": (relevant & window.event.isin(["earnings", "guidance"]).to_numpy()).astype(np.float64),
        "analyst": analyst.astype(np.float64), "analyst_sum": np.where(analyst, sentiment, 0.0)})
    groups = parts.groupby("ticker", sort=False)
    sums = groups.sum()
    ratio = lambda top, bottom: (sums[top] / sums[bottom]).where(sums[bottom] > 0)
    return pd.DataFrame({
        "news_count_7d": sums["count"], "news_count_1d": sums["count_1d"],
        "sentiment_7d": ratio("weighted", "weight"), "sentiment_1d": ratio("weighted_1d", "weight_1d"),
        "major_sentiment_7d": ratio("major_sum", "major"), "impact_7d": groups["impact"].max(),
        "earnings_7d": (sums["earnings"] > 0).astype(np.float64),
        "analyst_sentiment_7d": ratio("analyst_sum", "analyst")}).rename_axis("ticker")


def _for(summary: pd.DataFrame, tickers) -> pd.DataFrame:
    """Return the summary's rows for tickers; a ticker without news has counts 0 and earnings_7d 0."""
    features = summary.reindex(pd.Index(tickers, dtype=object, name="ticker"))
    return features.fillna({"news_count_7d": 0.0, "news_count_1d": 0.0, "earnings_7d": 0.0})


def _window(all_pairs: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    times = all_pairs.published_utc
    start = times.searchsorted(cutoff - pd.Timedelta(days=NEWS_LOOKBACK_DAYS))
    return all_pairs.iloc[start:times.searchsorted(cutoff)]


def decision_news_features(news: pd.DataFrame, tickers, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Return NEWS_FEATURES indexed by tickers, from the articles published before cutoff (a decision's news
    cutoff, Snapshot.news_cutoff); later articles are ignored."""
    cutoff = pd.Timestamp(cutoff)
    return _for(summarize(_window(pairs(news), cutoff), recent_start(cutoff)), tickers)


def training_news_features(data) -> pd.DataFrame:
    """Return data.labels() with the news features of each row's ticker on its decision day, from data.news()."""
    lookback = getattr(data, "news_lookback_days", NEWS_LOOKBACK_DAYS)
    if lookback < NEWS_LOOKBACK_DAYS:
        raise ValueError(f"The news features read {NEWS_LOOKBACK_DAYS} days of articles, but this model's predict "
                         f"gets {lookback}: set its news_lookback_days to at least {NEWS_LOOKBACK_DAYS}.")
    labels = data.labels().reset_index(drop=True)
    features = np.full((len(labels), len(NEWS_FEATURES)), np.nan)
    if len(labels):
        all_pairs = pairs(data.news())
        for day, rows in labels.groupby("decision_date", sort=True).indices.items():
            cutoff = news_cutoff(pd.Timestamp(day))
            summary = summarize(_window(all_pairs, cutoff), recent_start(cutoff))
            features[rows] = _for(summary, labels.ticker.iloc[rows].astype(str))[NEWS_FEATURES].to_numpy()
    return labels.join(pd.DataFrame(features, columns=NEWS_FEATURES, index=labels.index))
