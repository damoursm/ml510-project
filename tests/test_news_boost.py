"""The gradient-boosted news model (news_boost.py) and the extended news features it reads (news_features.py)."""
from dataclasses import replace
from pathlib import Path
import pickle
import tempfile
import unittest
import numpy as np
import pandas as pd
from stock_recommender.backtest.protocol import FoldData
from stock_recommender.backtest.registry import GRIDS, MODELS, build, configurations
from stock_recommender.backtest.run import load_data, run_backtest
from stock_recommender.config import Settings
from stock_recommender.exits import ExitRule, exit_rule
from stock_recommender.models import placeholder_models
from stock_recommender.news_boost import NewsBoostModel, rated
from stock_recommender.news_features import (EXTENDED_LOOKBACK_DAYS, EXTENDED_NEWS_FEATURES, NEWS_FEATURES,
                                             decision_extended_features, decision_news_features, extended_pairs,
                                             extended_summary, training_extended_features)
from .backtest_fixtures import BAD, GOOD, NEWS_NAMES, QUIET, config, write_news_market


def small_news_boost(**changes):
    """news_boost's models with a small booster and a 5-session horizon, for a few dozen stocks and months."""
    joint = NewsBoostModel(min_child_samples=20, n_estimators=60)
    return replace(placeholder_models(), joint=joint, horizon_days=5, news_lookback_days=EXTENDED_LOOKBACK_DAYS,
                   version="news_boost(small)", **({"picks": 5} | changes))


class Days:
    """Training data holding some labels and the news before them."""
    news_lookback_days = EXTENDED_LOOKBACK_DAYS

    def __init__(self, labels: pd.DataFrame, news: pd.DataFrame):
        self._labels, self._news = labels, news

    def labels(self) -> pd.DataFrame:
        return self._labels

    def news(self) -> pd.DataFrame:
        return self._news


class NewsBoostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):  # Every test reads the same market: write and load it once.
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        write_news_market(cls.root)
        cls.config = config(cls.root, sentiment_dir=cls.root / "sentiment")
        _, cls.panel, cls.news = load_data(cls.config)
        cls.cutoff = cls.panel.sessions[cls.panel.sessions < pd.Timestamp("2025-01-01")][-1]

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.data = FoldData(self.panel, self.news, Settings(min_price=0.0, min_dollar_volume=0.0), self.cutoff,
                             horizon_days=5, bars_lookback=60, news_lookback_days=EXTENDED_LOOKBACK_DAYS)
        self.snapshot = self.data.snapshot(self.cutoff)

    def test_the_extended_features_are_the_same_at_decision_and_in_training(self):
        labels = self.data.labels()
        day = labels.decision_date.max()
        snapshot = self.data.snapshot(day)
        tickers = snapshot.universe.index
        decided = decision_extended_features(snapshot.news, tickers, snapshot.news_cutoff)
        self.assertEqual(list(decided.columns), EXTENDED_NEWS_FEATURES)
        eight = decision_news_features(snapshot.news, tickers, snapshot.news_cutoff)
        np.testing.assert_allclose(decided[NEWS_FEATURES].to_numpy(dtype=float),
                                   eight[NEWS_FEATURES].to_numpy(dtype=float), equal_nan=True)  # The current 8.
        today = labels[labels.decision_date == day].reset_index(drop=True)
        trained = training_extended_features(Days(today, self.data.news())).set_index("ticker")
        np.testing.assert_allclose(trained.loc[tickers, EXTENDED_NEWS_FEATURES].to_numpy(dtype=float),
                                   decided.to_numpy(dtype=float), equal_nan=True)
        self.assertTrue((decided.loc[QUIET, "cnt_90d"] == 0).all() and decided.loc[QUIET, "sent_30d"].isna().all())
        self.assertTrue((decided.loc[GOOD, "sent_7d"] > 0).all() and (decided.loc[BAD, "sent_7d"] < 0).all())

    def test_articles_from_the_cutoff_on_are_ignored(self):
        later = self.snapshot.news.iloc[-1:].assign(published_utc=self.snapshot.news_cutoff)  # At the cutoff itself.
        tickers = self.snapshot.universe.index
        before = decision_extended_features(self.snapshot.news, tickers, self.snapshot.news_cutoff)
        after = decision_extended_features(pd.concat([self.snapshot.news, later], ignore_index=True), tickers,
                                           self.snapshot.news_cutoff)
        pd.testing.assert_frame_equal(before, after)
        with self.assertRaisesRegex(ValueError, "at least 90"):
            training_extended_features(type("Short", (Days,), {"news_lookback_days": 7})(self.data.labels(),
                                                                                         self.data.news()))

    def test_press_releases_focused_articles_and_polygons_labels(self):
        cutoff = pd.Timestamp("2025-01-10T21:00:00Z")
        score = lambda ticker, sentiment: {"ticker": ticker, "event": "deal", "relevance": 2, "impact": 1,  # noqa: E731
                                           "sentiment": sentiment}
        news = pd.DataFrame({
            "published_utc": pd.to_datetime(["2025-01-09T14:00:00Z", "2025-01-09T15:00:00Z", "2025-01-11T15:00:00Z"]),
            "tickers": [["AAA"], ["AAA", "BBB", "CCC"], ["AAA"]],
            "publisher": ["GlobeNewswire Inc.", "The Motley Fool", "PR Newswire"],
            "insights": [None, [{"ticker": "BBB", "sentiment": "positive"}, {"ticker": "CCC", "sentiment": "negative"}],
                         None],
            "llm_scores": [[score("AAA", 2)], [score("AAA", -2), score("BBB", 1), score("CCC", 0)], [score("AAA", 2)]]})
        pairs = extended_pairs(news)
        self.assertEqual(pairs.press.tolist(), [True, False, False, False, True])
        summary = extended_summary(pairs, cutoff, ["AAA", "BBB", "CCC", "DDD"])
        self.assertEqual(summary.loc["AAA", ["cnt_7d", "pr_cnt_7d", "focused_cnt_7d"]].tolist(), [2.0, 1.0, 1.0])
        self.assertEqual((summary.loc["AAA", "pr_sent_7d"], summary.loc["AAA", "nonpr_sent_7d"]), (1.0, -1.0))
        self.assertEqual(summary.loc["AAA", "sent_7d"], 0.0)  # +1 and -1, equally relevant; the later one is ignored.
        self.assertEqual((summary.loc["BBB", "poly_sent_7d"], summary.loc["CCC", "poly_sent_7d"]), (1.0, -1.0))
        self.assertEqual(summary.loc["AAA", "evt_deal_cnt_7d"], 2.0)
        self.assertEqual(summary.loc["DDD", "cnt_90d"], 0.0)
        self.assertTrue(np.isnan(summary.loc["DDD", "sent_7d"]))

    def test_it_learns_from_the_news_and_rates_only_stocks_with_news(self):
        model = NewsBoostModel(min_child_samples=20, n_estimators=60)
        model.fit(self.data)
        self.assertGreater(model.trained_on["rows"], 0)
        predicted = model.predict(self.snapshot.bars, self.snapshot.news, self.snapshot.universe.index,
                                  self.snapshot.news_cutoff)
        self.assertGreater(predicted[GOOD].mean(), predicted[BAD].mean())
        self.assertTrue(predicted[GOOD + BAD].notna().all() and predicted[QUIET].isna().all())
        importances = model.feature_importances()
        self.assertEqual(list(importances), EXTENDED_NEWS_FEATURES)
        self.assertAlmostEqual(sum(importances.values()), 1.0)
        again = pickle.loads(pickle.dumps(model))  # The API loads it from the run's pickle.
        pd.testing.assert_series_equal(again.predict(self.snapshot.bars, self.snapshot.news,
                                                     self.snapshot.universe.index, self.snapshot.news_cutoff),
                                       predicted)
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            NewsBoostModel().predict(self.snapshot.bars, self.snapshot.news, self.snapshot.universe.index,
                                     self.snapshot.news_cutoff)
        with self.assertRaisesRegex(ValueError, "cannot fill leaves"):
            NewsBoostModel(min_child_samples=1_000_000).fit(self.data)

    def test_preferred_shares_are_never_rated(self):
        features = pd.DataFrame({"news_count_7d": [3.0, 3.0, 0.0]}, index=["JPM", "JPMpC", "XYZ"])
        self.assertEqual(rated(features).tolist(), [True, False, False])

    def test_the_backtest_buys_only_what_it_rates(self):
        result = run_backtest(config(self.root, sentiment_dir=self.root / "sentiment", models=("boost",)),
                              {"boost": small_news_boost})
        bought = result.positions[result.positions.cost > 0]
        self.assertTrue(set(bought.ticker) <= set(GOOD + BAD))  # Never a stock without news.
        self.assertGreater(bought.ticker.isin(GOOD).mean(), 0.8)

    def test_the_registry_holds_it_20_sessions_by_default(self):
        configs = configurations(GRIDS["news_boost"])
        self.assertEqual(configs[0], {"hold_days": 20, "min_child_samples": 2_000, "num_leaves": 7, "stop_loss": None,
                                      "take_profit": None, "volatility_exit": None})
        self.assertEqual(len(configs), 4 * 24)
        models = build(MODELS["news_boost"], configs[0])
        self.assertIsInstance(models.joint, NewsBoostModel)
        self.assertEqual((models.horizon_days, models.news_lookback_days, models.picks), (20, 90, 20))
        self.assertEqual(exit_rule(models), ExitRule(20))
        self.assertEqual(len(NEWS_NAMES), 30)


if __name__ == "__main__":
    unittest.main()
