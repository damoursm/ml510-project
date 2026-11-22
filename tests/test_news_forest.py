from dataclasses import replace
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock
import numpy as np
import pandas as pd
from stock_recommender import news_forest
from stock_recommender.backtest.protocol import FoldData
from stock_recommender.backtest.registry import GRIDS, MODELS, build, configurations
from stock_recommender.backtest.report import write_run
from stock_recommender.backtest.run import load_data, run_backtest
from stock_recommender.config import Settings
from stock_recommender.forest import training_days
from stock_recommender.models import PlaceholderPriceModel, placeholder_models
from stock_recommender.news_features import NEWS_FEATURES, NEWS_LOOKBACK_DAYS
from stock_recommender.news_forest import NewsForestModel, news_blend, scaled_ranks
from stock_recommender.served_models import load_selection
from .backtest_fixtures import BAD, GOOD, NEWS_NAMES, QUIET, config, write_news_market

TICKERS = len(NEWS_NAMES)


def small_news_forest(min_samples_leaf: int = 20, **changes):
    """The news forest models with a small forest, for a few dozen stocks."""
    news = NewsForestModel(min_samples_leaf=min_samples_leaf, n_estimators=40)
    return replace(placeholder_models(), news=news, blend=news_blend, version=f"news_forest({min_samples_leaf})",
                   **({"picks": 5} | changes))


class NewsForestTest(unittest.TestCase):
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

    def setUp(self):  # New fold data for each test: the training rows are kept per data object.
        self.data = FoldData(self.panel, self.news, Settings(min_price=0.0, min_dollar_volume=0.0), self.cutoff,
                             horizon_days=5, bars_lookback=60, news_lookback_days=NEWS_LOOKBACK_DAYS)
        self.snapshot = self.data.snapshot(self.cutoff)

    def predict(self, model) -> pd.Series:
        return model.predict(self.snapshot.news, self.snapshot.universe.index, self.snapshot.news_cutoff)

    def test_it_learns_which_news_beats_the_other_stocks(self):
        self.assertGreater(self.news.llm_scores.notna().mean(), 0.85)
        model = NewsForestModel(min_samples_leaf=20, n_estimators=40)
        model.fit(self.data)
        days = training_days(self.data.labels().decision_date, 5)
        self.assertEqual(model.trained_on["days"], len(days))
        self.assertEqual(model.trained_on["rows"], len(days) * len(GOOD + BAD))  # Only the stocks with news...
        self.assertEqual(model.trained_on["tradable_rows"], len(days) * TICKERS)  # ...ranked among them all.
        predicted = self.predict(model)
        self.assertGreater(predicted[GOOD].min(), 0.0)  # Above the average stock...
        self.assertLess(predicted[BAD].max(), 0.0)  # ...and below it.
        self.assertTrue(predicted[QUIET].isna().all())  # No news: left out, so neutral.
        importances = model.feature_importances()
        self.assertEqual(list(importances), NEWS_FEATURES)
        self.assertAlmostEqual(sum(importances.values()), 1.0)
        again = pickle.loads(pickle.dumps(model))  # The API loads it from the run's pickle.
        pd.testing.assert_series_equal(self.predict(again), predicted)

    def test_a_grids_candidates_share_their_folds_training_rows(self):
        with mock.patch.object(news_forest, "training_news_features",
                               wraps=news_forest.training_news_features) as computed:
            for leaf in (20, 40):
                NewsForestModel(min_samples_leaf=leaf, n_estimators=10).fit(self.data)
        self.assertEqual(computed.call_count, 1)

    def test_an_unfitted_forest_or_too_few_rows_say_so(self):
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            self.predict(NewsForestModel())
        with self.assertRaisesRegex(ValueError, "cannot fill leaves"):
            NewsForestModel(min_samples_leaf=1_000_000).fit(self.data)

    def test_without_a_cutoff_every_stock_is_neutral(self):
        model = NewsForestModel(min_samples_leaf=20, n_estimators=10)
        model.fit(self.data)
        self.assertTrue(model.predict(self.snapshot.news, self.snapshot.universe.index, None).isna().all())

    def test_ranks_are_scaled_to_minus_one_to_one_within_each_day(self):
        days = pd.Series(pd.to_datetime(["2024-01-02"] * 4 + ["2024-01-03"] * 2))
        ranks = scaled_ranks(pd.Series([0.05, -0.1, 0.0, 0.0, 0.2, 0.1]), days)
        # Day one ranks 4, 1, 2.5 and 2.5 of 4; day two 2 and 1 of 2: 2 (rank - 0.5) / n - 1.
        np.testing.assert_allclose(ranks, [0.75, -0.75, 0.0, 0.0, 0.5, -0.5])

    def test_models_rank_on_the_news_alone_and_give_it_the_decision_cutoff(self):
        class Recording:
            def predict(self, news, tickers, cutoff):
                self.cutoff = cutoff
                return pd.Series({"T001": 0.4, "T012": -0.3})

        recording = Recording()
        models = replace(placeholder_models(), news=recording, blend=news_blend)
        scores = models.score(self.snapshot.bars, self.snapshot.news, self.snapshot.universe.index)
        self.assertEqual(recording.cutoff, self.snapshot.news_cutoff)  # The 16:00 close of the bars' last day.
        self.assertEqual(len(scores), TICKERS)
        self.assertEqual((scores.score.idxmax(), scores.score.idxmin()), ("T001", "T012"))
        self.assertEqual(scores.score.drop(["T001", "T012"]).abs().max(), 0.0)  # Left out: neutral.

    def test_the_backtest_trades_it_and_the_api_can_serve_it(self):
        result = run_backtest(replace(self.config, models=("news",), test_years=(2025,)),
                              {"news": small_news_forest})
        self.assertGreater(result.signals[result.signals.strategy == "news"].ic.mean(), 0.0)
        bought = set(result.positions[result.positions.strategy == "news"].ticker)
        self.assertTrue(bought and bought <= set(GOOD), bought)
        exits = result.guards["training_label_exits"]
        self.assertLess(pd.Timestamp(exits[2025]), pd.Timestamp("2025-01-01"))  # Trained before the year it trades.
        folder = write_run(result, "news", root=self.root)
        selection = load_selection(str(folder))
        self.assertEqual((selection.error, [model.name for model in selection.loaded]), (None, ["news"]))
        (model,) = selection.loaded
        self.assertEqual(list(model.models.news.feature_importances()), NEWS_FEATURES)

    def test_the_registry_tunes_its_leaves_and_features_on_earlier_years(self):
        configs = configurations(GRIDS["news_forest"])
        # The defaults run first: the forest's, then the exit strategy's (hold 5 days, no early rule).
        self.assertEqual(configs[0], {"hold_days": 5, "max_features": 0.5, "min_samples_leaf": 1_000,
                                      "stop_loss": None, "take_profit": None, "volatility_exit": None})
        self.assertEqual(len(configs), 4 * 24)  # Two leaf sizes x two feature shares x 24 exit strategies.
        models = [build(MODELS["news_forest"], params) for params in configs]
        self.assertEqual(len({model.version for model in models}), 4)  # Exit settings leave the version.
        self.assertEqual(len({(model.hold_days, model.stop_loss, model.take_profit, model.volatility_exit)
                              for model in models}), 24)
        for model in models:
            self.assertIsInstance(model.news, NewsForestModel)
            self.assertIsInstance(model.price, PlaceholderPriceModel)  # No bars feature: no move expected.
            self.assertIs(model.blend, news_blend)
            self.assertEqual((model.horizon_days, model.picks), (5, 20))
            self.assertGreaterEqual(model.news_lookback_days, NEWS_LOOKBACK_DAYS)  # The features read 7 days.


if __name__ == "__main__":
    unittest.main()
