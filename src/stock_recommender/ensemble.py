"""Combine several models into one by weighting their cross-sectional scores.

An Ensemble behaves like a Models bundle (fit, score, horizon_days, lookbacks, version), so the API can serve it
and a backtest treats it like any other model.
"""
from dataclasses import dataclass
from typing import Literal
import pandas as pd
from .models import Models, TrainingData

Normalize = Literal["rank", "zscore", "none"]


def normalize_scores(score: pd.Series, method: Normalize) -> pd.Series:
    """Put a model's scores on a common scale across tickers: centered percentile ranks, z-scores, or as is.

    Members may score in different units (a return, a probability, a sentiment), so ranks are the default.
    """
    if method == "rank":
        return score.rank(pct=True) - 0.5 if len(score) else score
    if method == "zscore":
        spread = score.std(ddof=0)
        return (score - score.mean()) / spread if spread > 0 else score * 0.0
    if method == "none":
        return score
    raise ValueError(f"Unknown normalization {method!r}; choose rank, zscore or none.")


@dataclass(frozen=True)
class Ensemble:
    members: tuple  # (Models or Ensemble, weight) pairs.
    normalize: Normalize = "rank"

    def __post_init__(self):
        if not self.members:
            raise ValueError("An ensemble needs at least one member.")
        weights = [weight for _, weight in self.members]
        if min(weights) < 0 or sum(weights) <= 0:
            raise ValueError("Ensemble weights must be non-negative and sum to more than 0.")
        horizons = {model.horizon_days for model, _ in self.members}
        if len(horizons) > 1:
            raise ValueError(f"Ensemble members must share horizon_days; got {sorted(horizons)}.")
        normalize_scores(pd.Series(dtype=float), self.normalize)  # Reject an unknown method now.

    @property
    def horizon_days(self) -> int:
        return self.members[0][0].horizon_days

    @property
    def bars_lookback(self) -> int:
        return max(model.bars_lookback for model, _ in self.members)

    @property
    def news_lookback_days(self) -> int:
        return max(model.news_lookback_days for model, _ in self.members)

    @property
    def version(self) -> str:
        total = sum(weight for _, weight in self.members)
        return "ensemble(" + ",".join(f"{model.version}:{weight / total:.2f}" for model, weight in self.members) + ")"

    def fit(self, data: TrainingData) -> None:
        for model, _ in self.members:
            model.fit(data)

    def score(self, bars: pd.DataFrame, news: pd.DataFrame, tickers: pd.Index) -> pd.DataFrame:
        """Return the weighted mean of the members' normalized scores (and of their expected returns and
        sentiments), over the members that scored each ticker; a ticker no member scores is dropped.

        Each member sees only its own lookbacks, counted back from the newest bar and from the news cutoff.
        """
        from .recommend import news_cutoff  # recommend imports models; importing here avoids a cycle.

        days = pd.Index(bars.date.unique()).sort_values()
        cutoff = news_cutoff(pd.Timestamp(days[-1])) if len(days) else None
        totals = {column: pd.Series(0.0, index=tickers) for column in ("score", "expected_return", "news_sentiment")}
        weights = pd.Series(0.0, index=tickers)
        for model, weight in self.members:
            if weight == 0:
                continue
            member_bars = bars[bars.date.isin(days[-model.bars_lookback:])]
            member_news = news if cutoff is None else news[
                news.published_utc >= cutoff - pd.Timedelta(days=model.news_lookback_days)]
            scores = model.score(member_bars, member_news, tickers)
            scored = scores.index
            totals["score"] = totals["score"].add(weight * normalize_scores(scores.score, self.normalize), fill_value=0)
            for column in ("expected_return", "news_sentiment"):
                totals[column] = totals[column].add(weight * scores[column], fill_value=0)
            weights = weights.add(pd.Series(weight, index=scored), fill_value=0)
        scored = weights[weights > 0].index
        return pd.DataFrame({column: totals[column].reindex(scored) / weights.reindex(scored)
                             for column in ("expected_return", "news_sentiment", "score")})
