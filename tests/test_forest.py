from dataclasses import replace
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock
import numpy as np
import pandas as pd
from stock_recommender import forest
from stock_recommender.backtest.protocol import FoldData
from stock_recommender.backtest.registry import GRIDS, MODELS, build, configurations
from stock_recommender.backtest.report import write_run
from stock_recommender.backtest.run import load_data, run_backtest
from stock_recommender.config import Settings
from stock_recommender.features import FEATURES, LOOKBACK
from stock_recommender.forest import ForestPriceModel, training_days
from stock_recommender.models import placeholder_models
from stock_recommender.served_models import load_selection
from .backtest_fixtures import config, random_walk, write_bars, write_news, write_splits

SESSIONS = pd.bdate_range("2023-11-01", "2025-03-31")  # 2024 trains the 2025 fold, with 60 sessions of history.
TICKERS = 60


def write_drifting(root: Path) -> None:
    """Write random walks whose drift, and volume, rise with the ticker's number: the more a stock trades, the more
    it gains, which a forest can learn from dollar_volume_20d and the past returns."""
    prices = random_walk(TICKERS, SESSIONS, seed=5)
    drift = 0.006 * (np.arange(TICKERS) / (TICKERS - 1) - 0.5)  # Per session: -0.3% to +0.3%.
    growth = np.exp(np.outer(np.arange(len(SESSIONS)), drift))
    names = prices["open"].columns
    volumes = pd.DataFrame(np.tile(1e5 * (1 + np.arange(TICKERS)), (len(SESSIONS), 1)), index=SESSIONS, columns=names)
    write_bars(root / "bars", prices["open"] * growth, prices["close"] * growth, volumes=volumes)
    write_splits(root / "splits.parquet", [])
    write_news(root / "news", [("2024-06-03T15:00:00Z", ["T000"])])


def small_forest(min_samples_leaf: int = 20, **changes):
    """The random forest models with a small forest, for a few dozen stocks."""
    price = ForestPriceModel(min_samples_leaf=min_samples_leaf, n_estimators=40)
    return replace(placeholder_models(), price=price, version=f"forest({min_samples_leaf})", **({"picks": 5} | changes))


class ForestTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        write_drifting(self.root)
        store, self.panel, self.news = load_data(config(self.root))
        self.cutoff = self.panel.sessions[self.panel.sessions < pd.Timestamp("2025-01-01")][-1]
        self.data = FoldData(self.panel, self.news, Settings(min_price=0.0, min_dollar_volume=0.0), self.cutoff,
                             horizon_days=5, bars_lookback=LOOKBACK, news_lookback_days=7)

    def tearDown(self):
        self.directory.cleanup()

    def test_it_learns_which_stocks_beat_the_others(self):
        model = ForestPriceModel(min_samples_leaf=20, n_estimators=40)
        model.fit(self.data)
        days = pd.DatetimeIndex(np.sort(self.data.labels().decision_date.unique()))
        # One decision day in five, back from the last labelled day, whose holding ends by the cutoff.
        self.assertEqual(model.trained_on["days"], len(days[::-1][::5]))
        self.assertEqual(model.trained_on["last_day"], str(days[-1].date()))
        self.assertEqual(model.trained_on["rows"], model.trained_on["days"] * TICKERS)
        snapshot = self.data.snapshot(self.cutoff)
        predicted = model.predict(snapshot.bars, snapshot.universe.index)
        drift = pd.Series(np.arange(TICKERS, dtype=float), index=[f"T{i:03d}" for i in range(TICKERS)])
        self.assertGreater(predicted.rank().corr(drift.reindex(predicted.index).rank()), 0.6)
        self.assertLess(abs(predicted.mean()), 0.01)  # Returns above the average stock's: centred near 0.
        importances = model.feature_importances()
        self.assertEqual(list(importances), FEATURES)
        self.assertAlmostEqual(sum(importances.values()), 1.0)
        again = pickle.loads(pickle.dumps(model))  # The API loads it from the run's pickle.
        pd.testing.assert_series_equal(again.predict(snapshot.bars, snapshot.universe.index), predicted)

    def test_a_grids_candidates_share_their_folds_training_rows(self):
        with mock.patch.object(forest, "training_features", wraps=forest.training_features) as computed:
            for leaf in (20, 40):
                ForestPriceModel(min_samples_leaf=leaf, n_estimators=10).fit(self.data)
        self.assertEqual(computed.call_count, 1)

    def test_an_unfitted_forest_or_too_few_rows_say_so(self):
        snapshot = self.data.snapshot(self.cutoff)
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            ForestPriceModel().predict(snapshot.bars, snapshot.universe.index)
        with self.assertRaisesRegex(ValueError, "cannot fill leaves"):
            ForestPriceModel(min_samples_leaf=1_000_000).fit(self.data)

    def test_training_days_are_one_in_every_horizon_back_from_the_last(self):
        days = pd.Series(pd.bdate_range("2024-01-01", periods=12).repeat(3))
        self.assertEqual(list(training_days(days, 5)), list(pd.bdate_range("2024-01-01", periods=12)[[1, 6, 11]]))

    def test_the_backtest_trades_it_and_the_api_can_serve_it(self):
        result = run_backtest(config(self.root, models=("forest",), test_years=(2025,)), {"forest": small_forest})
        self.assertGreater(result.signals[result.signals.strategy == "forest"].ic.mean(), 0.0)
        exits = result.guards["training_label_exits"]
        self.assertLess(pd.Timestamp(exits[2025]), pd.Timestamp("2025-01-01"))  # Trained before the year it trades.
        folder = write_run(result, "forest", root=self.root)
        selection = load_selection(str(folder))
        self.assertEqual((selection.error, [model.name for model in selection.loaded]), (None, ["forest"]))
        (model,) = selection.loaded
        self.assertGreater(len(model.models.price.feature_importances()), 0)

    def test_the_registry_tunes_its_leaves_and_features_on_earlier_years(self):
        configs = configurations(GRIDS["random_forest"])
        # The defaults run first: the forest's, then the exit strategy's (hold 5 days, no early rule).
        self.assertEqual(configs[0], {"hold_days": 5, "max_features": 0.5, "min_samples_leaf": 1_000,
                                      "stop_loss": None, "take_profit": None, "volatility_exit": None})
        self.assertEqual(len(configs), 4 * 24)  # Two leaf sizes x two feature shares x 24 exit strategies.
        models = [build(MODELS["random_forest"], params) for params in configs]
        self.assertEqual(len({model.version for model in models}), 4)  # Exit settings leave the version.
        self.assertEqual(len({(model.hold_days, model.stop_loss, model.take_profit, model.volatility_exit)
                              for model in models}), 24)
        for model in models:
            self.assertIsInstance(model.price, ForestPriceModel)
            self.assertEqual((model.horizon_days, model.picks), (5, 20))
            self.assertGreaterEqual(model.bars_lookback, LOOKBACK)  # The features read 60 sessions.


if __name__ == "__main__":
    unittest.main()
