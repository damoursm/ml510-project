from dataclasses import replace
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock
import numpy as np
import pandas as pd
from stock_recommender import price_news_forest
from stock_recommender.backtest.protocol import FoldData
from stock_recommender.backtest.registry import GRIDS, MODELS, build, configurations
from stock_recommender.backtest.report import write_run
from stock_recommender.backtest.run import load_data, run_backtest
from stock_recommender.config import Settings
from stock_recommender.explain import importances
from stock_recommender.features import FEATURES, LOOKBACK
from stock_recommender.forest import training_days
from stock_recommender.models import PlaceholderNewsModel, PlaceholderPriceModel, placeholder_models
from stock_recommender.news_features import NEWS_FEATURES, NEWS_LOOKBACK_DAYS
from stock_recommender.price_news_forest import ALL_FEATURES, PriceNewsForestModel
from stock_recommender.served_models import load_selection
from .backtest_fixtures import BAD, GOOD, NEWS_NAMES, QUIET, config, write_news_market

TICKERS = len(NEWS_NAMES)


def small_price_news_forest(min_samples_leaf: int = 20, **changes):
    """The price and news forest's models with a small forest, for a few dozen stocks."""
    joint = PriceNewsForestModel(min_samples_leaf=min_samples_leaf, n_estimators=40)
    return replace(placeholder_models(), joint=joint, version=f"price_news_forest({min_samples_leaf})",
                   **({"picks": 5} | changes))


def refuse_blend(expected_return: pd.Series, news_sentiment: pd.Series) -> pd.Series:
    raise AssertionError("A bundle with a joint model never blends.")


class PriceNewsForestTest(unittest.TestCase):
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
                             horizon_days=5, bars_lookback=LOOKBACK, news_lookback_days=NEWS_LOOKBACK_DAYS)
        self.snapshot = self.data.snapshot(self.cutoff)

    def predict(self, model, news: pd.DataFrame | None = None) -> pd.Series:
        snapshot = self.snapshot
        return model.predict(snapshot.bars, snapshot.news if news is None else news, snapshot.universe.index,
                             snapshot.news_cutoff)

    def fitted(self, **changes) -> PriceNewsForestModel:
        model = PriceNewsForestModel(**({"min_samples_leaf": 20, "n_estimators": 40} | changes))
        model.fit(self.data)
        return model

    def test_it_learns_from_the_bars_and_the_news_of_every_tradable_stock(self):
        model = self.fitted()
        days = training_days(self.data.labels().decision_date, 5)
        self.assertEqual(model.trained_on["days"], len(days))
        self.assertEqual(model.trained_on["rows"], len(days) * TICKERS)  # Every tradable stock, news or not...
        self.assertEqual(model.trained_on["with_news"], len(days) * len(GOOD + BAD))  # ...of which these had news.
        predicted = self.predict(model)
        self.assertFalse(predicted.isna().any())  # The stocks without news are rated too.
        self.assertGreater(predicted[GOOD].min(), predicted[BAD].max())
        self.assertGreater(predicted[GOOD].mean(), predicted[QUIET].mean())
        self.assertGreater(predicted[QUIET].mean(), predicted[BAD].mean())
        measured = model.feature_importances()
        self.assertEqual(list(measured), FEATURES + NEWS_FEATURES)
        self.assertAlmostEqual(sum(measured.values()), 1.0)
        again = pickle.loads(pickle.dumps(model))  # The API loads it from the run's pickle.
        pd.testing.assert_series_equal(self.predict(again), predicted)

    def test_its_ratings_follow_the_news_it_is_given(self):
        model = self.fitted()
        rated, without_news = self.predict(model), self.predict(model, self.snapshot.news.iloc[:0])
        self.assertGreater(rated[GOOD].mean(), without_news[GOOD].mean())  # Their good news lifted them...
        self.assertLess(rated[BAD].mean(), without_news[BAD].mean())  # ...and the bad news sank these.
        pd.testing.assert_series_equal(rated[QUIET], without_news[QUIET])  # These had none.

    def test_on_the_price_features_alone_it_reads_no_news(self):
        with mock.patch.object(price_news_forest, "training_news_features") as training, \
                mock.patch.object(price_news_forest, "decision_news_features") as decision:
            model = self.fitted(features=FEATURES)
            rated = self.predict(model)
        training.assert_not_called()
        decision.assert_not_called()
        self.assertEqual(list(model.feature_importances()), FEATURES)
        self.assertNotIn("with_news", model.trained_on)
        self.assertFalse(rated.isna().any())
        pd.testing.assert_series_equal(self.predict(model, self.snapshot.news.iloc[:0]), rated)

    def test_a_grids_candidates_share_their_folds_training_rows(self):
        with mock.patch.object(price_news_forest, "training_features",
                               wraps=price_news_forest.training_features) as bars, \
                mock.patch.object(price_news_forest, "training_news_features",
                                  wraps=price_news_forest.training_news_features) as news:
            for leaf in (20, 40):
                PriceNewsForestModel(min_samples_leaf=leaf, n_estimators=10).fit(self.data)
        self.assertEqual((bars.call_count, news.call_count), (1, 1))

    def test_an_unfitted_forest_too_few_rows_or_unknown_features_say_so(self):
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            self.predict(PriceNewsForestModel())
        with self.assertRaisesRegex(ValueError, "cannot fill leaves"):
            PriceNewsForestModel(min_samples_leaf=1_000_000).fit(self.data)
        for features in ((), ("ret_1d", "ret_1d"), ("ret_1d", "close")):
            with self.assertRaisesRegex(ValueError, "features must name"):
                PriceNewsForestModel(features=features)

    def test_without_a_cutoff_no_stock_is_rated(self):
        model = self.fitted(n_estimators=10)
        rated = model.predict(self.snapshot.bars.iloc[:0], self.snapshot.news, self.snapshot.universe.index, None)
        self.assertTrue(rated.isna().all())

    def test_a_bundle_with_a_joint_model_scores_on_it_alone(self):
        class Recording:
            def __init__(self):
                self.fits = 0

            def fit(self, data):
                self.fits += 1

            def predict(self, bars, news, tickers, cutoff):
                self.seen = (len(bars), len(news), list(tickers), cutoff)
                return pd.Series({"T001": 0.4, "T012": -0.3, "T025": np.nan})  # The others left out.

        joint = Recording()
        models = replace(placeholder_models(), joint=joint, blend=refuse_blend)
        models.fit(self.data)
        self.assertEqual(joint.fits, 1)
        snapshot = self.snapshot
        scores = models.score(snapshot.bars, snapshot.news, snapshot.universe.index)
        self.assertEqual(joint.seen, (len(snapshot.bars), len(snapshot.news), list(snapshot.universe.index),
                                      snapshot.news_cutoff))  # The 16:00 close of the bars' last day.
        self.assertEqual(scores.score.to_dict(), {"T001": 0.4, "T012": -0.3})  # Left out or NaN: not recommended.
        self.assertEqual(scores[["expected_return", "news_sentiment"]].abs().to_numpy().max(), 0.0)  # Placeholders.

    def test_bundles_pickled_before_joint_models_load_without_one(self):
        old = placeholder_models()
        object.__delattr__(old, "joint")  # As a bundle pickled before the field existed: not in its state.
        loaded = pickle.loads(pickle.dumps(old))
        self.assertNotIn("joint", vars(loaded))
        self.assertIsNone(loaded.joint)
        snapshot = self.snapshot
        pd.testing.assert_frame_equal(loaded.score(snapshot.bars, snapshot.news, snapshot.universe.index),
                                      placeholder_models().score(snapshot.bars, snapshot.news, snapshot.universe.index))

    def test_the_backtest_trades_it_and_the_api_can_serve_it(self):
        result = run_backtest(replace(self.config, models=("joint",), test_years=(2025,)),
                              {"joint": small_price_news_forest})
        self.assertGreater(result.signals[result.signals.strategy == "joint"].ic.mean(), 0.0)
        bought = result.positions[result.positions.strategy == "joint"].ticker
        self.assertGreater(bought.isin(GOOD).mean(), 0.8)  # Mostly the stocks with good news.
        exits = result.guards["training_label_exits"]
        self.assertLess(pd.Timestamp(exits[2025]), pd.Timestamp("2025-01-01"))  # Trained before the year it trades.
        folder = write_run(result, "joint", root=self.root)
        selection = load_selection(str(folder))
        self.assertEqual((selection.error, [model.name for model in selection.loaded]), (None, ["joint"]))
        (model,) = selection.loaded
        self.assertEqual(list(importances(model.models)), FEATURES + NEWS_FEATURES)  # As the dashboard lists them.

    def test_the_registry_tunes_its_leaves_and_features_on_earlier_years(self):
        configs = configurations(GRIDS["price_news_forest"])
        # The defaults run first: the forest's, then the exit strategy's (hold 5 days, no early rule).
        self.assertEqual(configs[0], {"hold_days": 5, "max_features": 0.5, "min_samples_leaf": 1_000,
                                      "stop_loss": None, "take_profit": None, "volatility_exit": None})
        self.assertEqual(len(configs), 4 * 24)  # Two leaf sizes x two feature shares x 24 exit strategies.
        models = [build(MODELS["price_news_forest"], params) for params in configs]
        self.assertEqual(len({model.version for model in models}), 4)  # Exit settings leave the version.
        self.assertEqual(len({(model.hold_days, model.stop_loss, model.take_profit, model.volatility_exit)
                              for model in models}), 24)
        for model in models:
            self.assertIsInstance(model.joint, PriceNewsForestModel)
            self.assertEqual(model.joint.features, ALL_FEATURES)
            self.assertIsInstance(model.price, PlaceholderPriceModel)  # The joint model gives the score.
            self.assertIsInstance(model.news, PlaceholderNewsModel)
            self.assertEqual((model.horizon_days, model.picks), (5, 20))
            self.assertGreaterEqual(model.bars_lookback, LOOKBACK)  # The price features read 60 sessions...
            self.assertGreaterEqual(model.news_lookback_days, NEWS_LOOKBACK_DAYS)  # ...and the news ones 7 days.


if __name__ == "__main__":
    unittest.main()
