"""Features of each stock's recent news, from the LLM scores attached to the articles (sentiment.attach).

A decision at day t's close sees the articles published before t's 16:00 New York close (the news cutoff). The
features read those of the last NEWS_LOOKBACK_DAYS days; "1d" ones only those published since the previous
weekday's close. A pair is an article and one ticker it is tagged with. A pair is relevant when the LLM rated the
article as at least one of the company's main subjects (relevance >= 1), and it weighs relevance / 2. Sentiment
(-2..2) is scaled to -1..1, impact (0..2) to 0..1. Counts include every article tagged with the stock, scored or
not; a feature that needs a scored relevant article is NaN without one (tree models such as LightGBM accept NaN).

The two entry points compute the same features:
- decision_news_features(news, tickers, cutoff): from a decision's articles (Snapshot.news) and news cutoff;
- training_news_features(data): from a TrainingData, its labels with the features of each label's decision day.

EXTENDED_NEWS_FEATURES (55, for news_boost.py) add to these 8 the same measures over 3, 14 and 30 days, the 90-day
count and the 7-day count's surprise against it, the spread and shares of the 7 days' sentiment, focused articles
(tagging at most 2 tickers), press-release wires against other publishers, Polygon's own sentiment labels (which
exist from mid-2024 only), and the sentiment and count of each event kind. They read EXTENDED_LOOKBACK_DAYS of
articles, through decision_extended_features and training_extended_features, which compute the same values.
"""
import re
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


_summarized = None  # The newest (news, cutoff, summary): a day's models read one snapshot's articles (recommend.rank).


def decision_news_features(news: pd.DataFrame, tickers, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Return NEWS_FEATURES indexed by tickers, from the articles published before cutoff (a decision's news
    cutoff, Snapshot.news_cutoff); later articles are ignored. The summary of the newest articles table is kept,
    so the models deciding on the same snapshot summarize it once."""
    global _summarized
    cutoff = pd.Timestamp(cutoff)
    held = _summarized
    if held is not None and held[0] is news and held[1] == cutoff:  # Held, so its id cannot be another table's.
        summary = held[2]
    else:
        summary = summarize(_window(pairs(news), cutoff), recent_start(cutoff))
        _summarized = (news, cutoff, summary)
    return _for(summary, tickers)


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


# The extended features (news_boost.py): the research dataset's 55 news features, defined exactly as it built them.
EXTENDED_LOOKBACK_DAYS = 90  # Days of articles before the news cutoff that the extended features read.
PRESS = re.compile(r"newswire|business ?wire|accesswire|newsfile|pr ?web|einpresswire|globe ?newswire", re.I)
POLYGON = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}  # Polygon's own per-ticker label (insights).
EVENT_GROUPS = {"earnings": ("earnings", "guidance"), "analyst": ("analyst",), "deal": ("deal",),
                "product": ("product",), "legal": ("legal",), "management": ("management",), "capital": ("capital",),
                "price_move": ("price_move",), "opinion": ("opinion",), "macro": ("macro",)}
EXTENDED_NEWS_FEATURES = NEWS_FEATURES + [
    "cnt_3d", "rel_cnt_3d", "sent_3d", "cnt_7d", "rel_cnt_7d", "sent_7d", "impact_mean_7d", "sent_max_7d",
    "sent_min_7d", "pos_frac_7d", "neg_frac_7d", "focused_cnt_7d", "focused_sent_7d", "pr_cnt_7d", "pr_sent_7d",
    "nonpr_sent_7d", "poly_cnt_7d", "poly_sent_7d",
    *[f"evt_{name}_{measure}_7d" for name in EVENT_GROUPS for measure in ("sent", "cnt")],
    "cnt_14d", "rel_cnt_14d", "sent_14d", "cnt_30d", "rel_cnt_30d", "sent_30d", "cnt_90d", "surprise_7d",
    "sent_chg_7d_30d"]


def extended_pairs(news: pd.DataFrame) -> pd.DataFrame:
    """Return pairs (one row per article and tagged ticker, by publication time) with what the extended features
    read besides the LLM's scores: the article's ticker count, whether a press-release wire published it, and
    Polygon's sentiment label for the ticker (-1, 0, 1; NaN without one); and each pair's relevance weight
    (relevance / 2 when scored, else 0), scaled sentiment s (-1..1), and whether it is scored and relevant."""
    scores = news["llm_scores"] if "llm_scores" in news else pd.Series(None, index=news.index, dtype=object)
    publishers = news["publisher"] if "publisher" in news else pd.Series("", index=news.index)
    insights = news["insights"] if "insights" in news else pd.Series(None, index=news.index, dtype=object)
    rows = []
    for published, tickers, scored, publisher, labels in zip(news.published_utc, news.tickers, scores, publishers,
                                                              insights):
        names = list(dict.fromkeys([] if tickers is None else [str(name) for name in tickers]))
        by_ticker = {} if scored is None or isinstance(scored, float) else {s["ticker"]: s for s in scored}
        polygon = ({item["ticker"]: item.get("sentiment") for item in labels}
                   if labels is not None and not isinstance(labels, float) and len(labels) else {})
        press = bool(PRESS.search(publisher if isinstance(publisher, str) else ""))
        for ticker in names:
            score = by_ticker.get(ticker) or {}
            rows.append((published, ticker, score.get("event"), score.get("relevance"), score.get("impact"),
                         score.get("sentiment"), len(names), press, POLYGON.get(polygon.get(ticker), np.nan)))
    frame = pd.DataFrame(rows, columns=_PAIRS + ["n_tickers", "press", "polygon"]).astype(
        {"relevance": np.float64, "impact": np.float64, "sentiment": np.float64, "ticker": object,
         "n_tickers": np.int64, "press": bool, "polygon": np.float64})
    if not len(frame):
        frame["published_utc"] = pd.Series(dtype="datetime64[ms, UTC]")
    relevance = np.nan_to_num(frame.relevance.to_numpy())
    scored = np.isfinite(frame.sentiment.to_numpy()) & (relevance > 0)
    frame["weight"] = np.where(scored, relevance / 2, 0.0)
    frame["s"] = frame.sentiment.to_numpy() / 2
    frame["scored"] = scored
    frame["relevant"] = scored & (relevance >= 1)
    return frame.sort_values("published_utc", kind="stable", ignore_index=True)


def _weighted(window: pd.DataFrame, mask: np.ndarray, groups: np.ndarray) -> pd.Series:
    """Relevance-weighted mean scaled sentiment of the masked pairs, per ticker; NaN without weight."""
    weight = np.where(mask, window.weight.to_numpy(), 0.0)
    parts = pd.DataFrame({"w": weight, "ws": weight * np.nan_to_num(window.s.to_numpy())}).groupby(groups).sum()
    return (parts.ws / parts.w).where(parts.w > 0)


def _per_ticker(values, groups: np.ndarray, how: str) -> pd.Series:
    return getattr(pd.Series(np.asarray(values, dtype=np.float64)).groupby(groups), how)()


def _window_features(window: pd.DataFrame, days: int, details: bool) -> pd.DataFrame:
    """The count, relevant count and weighted sentiment of a window's pairs, per ticker; with details (the 7 days),
    the rest of the 7-day extended features."""
    if not len(window):
        return pd.DataFrame(index=pd.Index([], dtype=object, name="ticker"))
    groups = window.ticker.to_numpy()
    relevant, scored = window.relevant.to_numpy(), window.scored.to_numpy()
    out = {f"cnt_{days}d": _per_ticker(np.ones(len(window)), groups, "sum"),
           f"rel_cnt_{days}d": _per_ticker(relevant, groups, "sum"),
           f"sent_{days}d": _weighted(window, scored, groups)}
    if details:
        s = window.s.to_numpy()
        relevant_count = _per_ticker(relevant, groups, "sum")
        focused, press = (window.n_tickers <= 2).to_numpy(), window.press.to_numpy()
        out.update({
            "impact_mean_7d": _per_ticker(np.where(relevant, window.impact.to_numpy() / 2, np.nan), groups, "mean"),
            "sent_max_7d": _per_ticker(np.where(relevant, s, np.nan), groups, "max"),
            "sent_min_7d": _per_ticker(np.where(relevant, s, np.nan), groups, "min"),
            "pos_frac_7d": (_per_ticker(relevant & (s > 0), groups, "sum") / relevant_count).where(relevant_count > 0),
            "neg_frac_7d": (_per_ticker(relevant & (s < 0), groups, "sum") / relevant_count).where(relevant_count > 0),
            "focused_cnt_7d": _per_ticker(focused, groups, "sum"),
            "focused_sent_7d": _weighted(window, scored & focused, groups),
            "pr_cnt_7d": _per_ticker(press, groups, "sum"),
            "pr_sent_7d": _weighted(window, scored & press, groups),
            "nonpr_sent_7d": _weighted(window, scored & ~press, groups),
            "poly_cnt_7d": _per_ticker(np.isfinite(window.polygon.to_numpy()), groups, "sum"),
            "poly_sent_7d": _per_ticker(window.polygon.to_numpy(), groups, "mean")})
        for name, events in EVENT_GROUPS.items():
            mask = scored & window.event.isin(events).to_numpy()
            out[f"evt_{name}_sent_7d"] = _weighted(window, mask, groups)
            out[f"evt_{name}_cnt_7d"] = _per_ticker(mask & relevant, groups, "sum")
    return pd.DataFrame(out)


def extended_summary(all_pairs: pd.DataFrame, cutoff: pd.Timestamp, tickers) -> pd.DataFrame:
    """Return EXTENDED_NEWS_FEATURES indexed by tickers, from the extended_pairs published before cutoff. A ticker
    without an article has counts 0 and its other features NaN."""
    cutoff = pd.Timestamp(cutoff)
    tickers = pd.Index(tickers, dtype=object, name="ticker")
    times = all_pairs.published_utc

    def since(days: int) -> pd.DataFrame:
        return all_pairs.iloc[times.searchsorted(cutoff - pd.Timedelta(days=days)):times.searchsorted(cutoff)]

    parts = [_for(summarize(since(NEWS_LOOKBACK_DAYS), recent_start(cutoff)), tickers)[NEWS_FEATURES]]
    for days in (3, 7, 14, 30):
        parts.append(_window_features(since(days), days, details=days == 7).reindex(tickers))
    parts.append(pd.DataFrame({"cnt_90d": since(90).ticker.value_counts()}).reindex(tickers))
    frame = pd.concat(parts, axis=1)
    counts = [column for column in frame.columns if "cnt" in column]  # No article: a count of 0.
    frame[counts] = frame[counts].fillna(0.0)
    frame["surprise_7d"] = frame["cnt_7d"] / (frame["cnt_90d"] * 7 / 90 + 0.5)
    frame["sent_chg_7d_30d"] = frame["sent_7d"] - frame["sent_30d"]
    return frame.reindex(columns=EXTENDED_NEWS_FEATURES).astype(np.float64)


_extended = None  # The newest (news, its extended_pairs): a day's models read one snapshot's articles.


def decision_extended_features(news: pd.DataFrame, tickers, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Return EXTENDED_NEWS_FEATURES indexed by tickers, from a decision's articles (Snapshot.news, which must hold
    the EXTENDED_LOOKBACK_DAYS before cutoff) published before its cutoff."""
    global _extended
    held = _extended
    if held is not None and held[0] is news:  # Held, so its id cannot be another table's.
        all_pairs = held[1]
    else:
        all_pairs = extended_pairs(news)
        _extended = (news, all_pairs)
    return extended_summary(all_pairs, cutoff, tickers)


def training_extended_features(data) -> pd.DataFrame:
    """Return data.labels() with the extended features of each row's ticker on its decision day, from data.news()."""
    lookback = getattr(data, "news_lookback_days", EXTENDED_LOOKBACK_DAYS)
    if lookback < EXTENDED_LOOKBACK_DAYS:
        raise ValueError(f"The extended news features read {EXTENDED_LOOKBACK_DAYS} days of articles, but this "
                         f"model's predict gets {lookback}: set its news_lookback_days to at least "
                         f"{EXTENDED_LOOKBACK_DAYS}.")
    labels = data.labels().reset_index(drop=True)
    features = np.full((len(labels), len(EXTENDED_NEWS_FEATURES)), np.nan)
    if len(labels):
        all_pairs = extended_pairs(data.news())
        for day, rows in labels.groupby("decision_date", sort=True).indices.items():
            summary = extended_summary(all_pairs, news_cutoff(pd.Timestamp(day)), labels.ticker.iloc[rows].astype(str))
            features[rows] = summary.to_numpy()
    return labels.join(pd.DataFrame(features, columns=EXTENDED_NEWS_FEATURES, index=labels.index))
