from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from stock_recommender import sentiment
from stock_recommender.backtest.panel import Panel
from stock_recommender.backtest.protocol import FoldData
from stock_recommender.backtest.window import WindowedStore
from stock_recommender.config import Settings
from stock_recommender.news_features import (NEWS_FEATURES, NEWS_LOOKBACK_DAYS, decision_news_features,
                                             training_news_features)
from stock_recommender.recommend import news_cutoff
from .backtest_fixtures import SESSIONS, random_walk, write_bars, write_news, write_splits


def article(published, scores: dict | None, tickers=None) -> dict:
    """scores maps ticker -> (event, relevance, impact, sentiment); None leaves the article unscored."""
    tickers = tickers or list(scores)
    llm = None if scores is None else [{"ticker": t, "event": e, "relevance": r, "impact": i, "sentiment": s}
                                       for t, (e, r, i, s) in scores.items()]
    return {"published_utc": pd.Timestamp(published), "tickers": tickers, "llm_scores": llm}


class DecisionNewsFeaturesTest(unittest.TestCase):
    cutoff = news_cutoff(pd.Timestamp("2025-03-12"))  # Wednesday 16:00 New York, 20:00 UTC.

    def test_each_feature(self):
        hour = pd.Timedelta(hours=1)
        news = pd.DataFrame([
            article(self.cutoff - hour, {"AAA": ("earnings", 2, 2, 2)}),  # Since Tuesday's close.
            article(self.cutoff - 30 * hour, {"AAA": ("analyst", 1, 1, -1), "BBB": ("other", 0, 0, 2)}),  # Before.
            article(self.cutoff - 8 * 24 * hour, {"AAA": ("deal", 2, 2, -2)}),  # Outside the 7 days.
            article(self.cutoff, {"AAA": ("deal", 2, 2, -2)}),  # At the cutoff: after the decision's close.
            article(self.cutoff - 2 * hour, None, ["BBB"]),  # Not scored.
        ])
        features = decision_news_features(news, ["AAA", "BBB", "CCC"], self.cutoff)
        self.assertEqual(list(features.columns), NEWS_FEATURES)
        a, b, c = (features.loc[ticker] for ticker in ("AAA", "BBB", "CCC"))
        self.assertEqual((a.news_count_7d, a.news_count_1d), (2, 1))
        self.assertAlmostEqual(a.sentiment_7d, (1 * 1 + 0.5 * -0.5) / (1 + 0.5))  # Weights relevance / 2.
        self.assertEqual((a.sentiment_1d, a.major_sentiment_7d, a.impact_7d), (1, 1, 1))
        self.assertEqual((a.earnings_7d, a.analyst_sentiment_7d), (1, -0.5))
        self.assertEqual((b.news_count_7d, b.news_count_1d, b.earnings_7d), (2, 1, 0))
        for name in ("sentiment_7d", "sentiment_1d", "major_sentiment_7d", "impact_7d", "analyst_sentiment_7d"):
            self.assertTrue(np.isnan(b[name]), name)  # Only a mention, and an unscored article.
        self.assertEqual((c.news_count_7d, c.news_count_1d, c.earnings_7d), (0, 0, 0))
        self.assertTrue(c.drop(["news_count_7d", "news_count_1d", "earnings_7d"]).isna().all())

    def test_monday_s_last_day_of_news_starts_at_friday_s_close(self):
        monday = news_cutoff(pd.Timestamp("2025-03-17"))
        friday = news_cutoff(pd.Timestamp("2025-03-14"))
        news = pd.DataFrame([article(friday - pd.Timedelta(minutes=1), {"AAA": ("other", 2, 1, -2)}),
                             article(friday + pd.Timedelta(hours=20), {"AAA": ("other", 2, 1, 2)})])
        features = decision_news_features(news, ["AAA"], monday).loc["AAA"]
        self.assertEqual((features.news_count_1d, features.sentiment_1d, features.sentiment_7d), (1, 1, 0))

    def test_a_ticker_tagged_twice_counts_once(self):
        hour = pd.Timedelta(hours=1)
        news = pd.DataFrame([article(self.cutoff - hour, {"NIO": ("other", 2, 1, 2)}, ["NIO", "NIO"]),
                             article(self.cutoff - 2 * hour, {"NIO": ("other", 2, 1, -2)})])
        nio = decision_news_features(news, ["NIO"], self.cutoff).loc["NIO"]
        self.assertEqual((nio.news_count_7d, nio.news_count_1d, nio.sentiment_7d), (2, 2, 0))

    def test_no_news(self):
        empty = pd.DataFrame(columns=["published_utc", "tickers"]).astype({"published_utc": "datetime64[ms, UTC]"})
        features = decision_news_features(empty, ["AAA"], self.cutoff)
        self.assertEqual(features.loc["AAA", "news_count_7d"], 0)


class TrainingNewsFeaturesTest(unittest.TestCase):
    """Training rows equal what predict() computes from the decision's own news, and nothing after the cutoff."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        prices = random_walk(6, SESSIONS, seed=4)
        write_bars(root / "bars", prices["open"], prices["close"])
        write_splits(root / "splits.parquet", [])
        rng = np.random.default_rng(5)
        times = pd.date_range("2024-11-01", "2025-03-31", freq="7h", tz="UTC")
        tickers = list(prices["open"].columns)
        articles = [(str(time), list(rng.choice(tickers, rng.integers(1, 3), replace=False))) for time in times]
        write_news(root / "news", articles)
        rows = []
        for path in sorted((root / "news").rglob("*.parquet")):
            for item in pd.read_parquet(path).itertuples():
                for ticker in item.tickers:
                    if rng.random() < 0.85:  # Some pairs, e.g. of a malformed answer, have no score.
                        rows.append({"id": item.id, "published_utc": item.published_utc, "ticker": ticker,
                                     "event": str(rng.choice(sentiment.EVENTS)), "relevance": int(rng.integers(0, 3)),
                                     "impact": int(rng.integers(0, 3)), "sentiment": int(rng.integers(-2, 3)),
                                     "model": "qwen3:8b", "prompt_version": sentiment.PROMPT_VERSION})
        frame = sentiment.to_frame(rows)
        for day, group in frame.groupby(frame.published_utc.dt.date):
            sentiment.write_day(group, root / "sentiment" / str(day.year) / f"{day}.parquet")
        store = WindowedStore(root / "bars", root / "news", root / "splits.parquet", root / "sentiment", "qwen3:8b")
        self.panel, self.news = Panel.load(store, strict_calendar=False), store.read_news()
        self.settings = Settings(min_price=0, min_dollar_volume=0)

    def tearDown(self):
        self.directory.cleanup()

    def data(self, cutoff: int, news_days: int = NEWS_LOOKBACK_DAYS) -> FoldData:
        return FoldData(self.panel, self.news, self.settings, self.panel.sessions[cutoff], 2, 3, news_days)

    def test_training_rows_match_the_decision_news(self):
        self.assertGreater(self.news.llm_scores.notna().mean(), 0.9)
        data = self.data(100)
        table = training_news_features(data)
        pd.testing.assert_frame_equal(table.drop(columns=NEWS_FEATURES), data.labels())
        self.assertTrue(table[NEWS_FEATURES].notna().any().all())
        for t in (2, 20, 54, 55, 80, 97):  # 2025-01-20 (54) is a Monday.
            day = self.panel.sessions[t]
            rows = table[table.decision_date == day].set_index("ticker")[NEWS_FEATURES]
            snapshot = data.snapshot(day)
            expected = decision_news_features(snapshot.news, rows.index, snapshot.news_cutoff)
            pd.testing.assert_frame_equal(rows, expected, check_names=False)

    def test_later_news_changes_no_earlier_row_and_a_short_lookback_is_refused(self):
        early, late = training_news_features(self.data(60)), training_news_features(self.data(100))
        merged = early.merge(late, on=["decision_date", "ticker"], suffixes=("", "_late"))
        self.assertEqual(len(merged), len(early))
        np.testing.assert_array_equal(merged[NEWS_FEATURES].to_numpy(),
                                      merged[[f"{name}_late" for name in NEWS_FEATURES]].to_numpy())
        with self.assertRaisesRegex(ValueError, "news_lookback_days"):
            training_news_features(self.data(100, NEWS_LOOKBACK_DAYS - 1))


if __name__ == "__main__":
    unittest.main()
