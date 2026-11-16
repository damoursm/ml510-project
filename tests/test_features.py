from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
import pandas as pd
from stock_recommender.backtest.panel import Panel
from stock_recommender.backtest.protocol import FoldData
from stock_recommender.backtest.window import WindowedStore
from stock_recommender.config import Settings
from stock_recommender import features as module
from stock_recommender.features import FEATURES, LOOKBACK, decision_features, training_features
from .backtest_fixtures import SESSIONS, random_walk, write_bars, write_splits


def bars_of(prices: dict[str, pd.DataFrame], volumes: pd.DataFrame) -> pd.DataFrame:
    """Long bars from sessions x tickers frames of open, high, low and close; a NaN close means no bar."""
    frame = pd.concat({name: values.stack(future_stack=True) for name, values in prices.items()}, axis=1)
    frame["volume"] = volumes.stack(future_stack=True)
    frame = frame.rename_axis(["date", "ticker"]).reset_index()
    return frame[frame.close.notna()].reset_index(drop=True)


def ohlc(sessions: pd.DatetimeIndex, tickers: int, seed: int = 0):
    walk = random_walk(tickers, sessions, seed)
    rng = np.random.default_rng(seed + 1)
    top = np.maximum(walk["open"], walk["close"]) * (1 + rng.uniform(0, 0.02, walk["open"].shape))
    bottom = np.minimum(walk["open"], walk["close"]) * (1 - rng.uniform(0, 0.02, walk["open"].shape))
    volumes = pd.DataFrame(rng.uniform(1e5, 1e6, walk["open"].shape), index=sessions, columns=walk["open"].columns)
    return {"open": walk["open"], "high": top, "low": bottom, "close": walk["close"]}, volumes


class DecisionFeaturesTest(unittest.TestCase):
    sessions = SESSIONS[:80]

    def test_each_feature_on_the_last_day(self):
        prices, volumes = ohlc(self.sessions, 3)
        features = decision_features(bars_of(prices, volumes))
        self.assertEqual(list(features.columns), FEATURES)
        for ticker in prices["close"].columns:
            o, h, l, c = (prices[name][ticker].to_numpy() for name in ("open", "high", "low", "close"))
            v = volumes[ticker].to_numpy()
            expected = {
                "ret_1d": c[-1] / c[-2] - 1, "ret_5d": c[-1] / c[-6] - 1, "ret_20d": c[-1] / c[-21] - 1,
                "momentum_60d": c[-6] / c[-60] - 1, "gap_1d": o[-1] / c[-2] - 1,
                "close_location": (c[-1] - l[-1]) / (h[-1] - l[-1]),
                "volatility_20d": np.std(np.diff(np.log(c[-21:])), ddof=1),
                "high_60d": c[-1] / h[-60:].max() - 1, "volume_ratio": v[-5:].mean() / v[-60:].mean(),
                "dollar_volume_20d": np.log(np.median(c[-20:] * v[-20:]))}
            for name, value in expected.items():
                self.assertAlmostEqual(features.loc[ticker, name], value, places=12, msg=name)

    def test_a_missing_bar_blanks_only_the_features_that_read_it(self):
        prices, volumes = ohlc(self.sessions, 2)
        prices["close"].iloc[-31, 0] = np.nan  # T000 has no bar 30 sessions before the decision.
        features = decision_features(bars_of(prices, volumes))
        blank = {"high_60d", "volume_ratio"}  # The only features whose windows reach 30 sessions back.
        self.assertEqual(set(features.columns[features.loc["T000"].isna()]), blank)
        self.assertFalse(features.loc["T001"].isna().any())

    def test_a_short_window_or_a_flat_day(self):
        prices, volumes = ohlc(self.sessions[:30], 1)
        for name in ("open", "high", "low"):
            prices[name].iloc[-1] = prices["close"].iloc[-1]  # One price all day.
        features = decision_features(bars_of(prices, volumes)).loc["T000"]
        self.assertEqual(features.close_location, 0.5)
        self.assertEqual(set(features.index[features.isna()]), {"momentum_60d", "high_60d", "volume_ratio"})

    def test_only_tickers_with_a_bar_on_the_last_day_and_no_more_than_lookback_days(self):
        prices, volumes = ohlc(self.sessions, 2)
        prices["close"].iloc[-1, 1] = np.nan
        bars = bars_of(prices, volumes)
        features = decision_features(bars)
        self.assertEqual(list(features.index), ["T000"])
        recent = bars[bars.date >= self.sessions[-LOOKBACK]]
        pd.testing.assert_frame_equal(decision_features(recent), features)
        self.assertEqual(len(decision_features(bars.iloc[:0])), 0)


class TrainingFeaturesTest(unittest.TestCase):
    """Training rows equal what predict() computes from the decision's own window, a split changes nothing, and
    later bars change no earlier row."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        prices, volumes = ohlc(SESSIONS, 6, seed=3)
        split = 85  # S, T000's twin, splits 2-for-1 at session 85's open: prices halve and volume doubles.
        for name in prices:
            prices[name]["S"] = prices[name]["T000"] * np.where(np.arange(len(SESSIONS)) >= split, 0.5, 1.0)
        volumes["S"] = volumes["T000"] * np.where(np.arange(len(SESSIONS)) >= split, 2.0, 1.0)
        prices["close"].iloc[70, 2] = np.nan  # T002 misses a session.
        bars = bars_of(prices, volumes)
        for day, group in bars.groupby("date"):
            path = root / "bars" / str(day.year) / f"{day.date()}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            group.assign(vwap=group.close, transactions=pd.array([100] * len(group), dtype="Int64")).to_parquet(path)
        write_splits(root / "splits.parquet", [("S", str(SESSIONS[split].date()), 1, 2)])
        (root / "news").mkdir()
        store = WindowedStore(root / "bars", root / "news", root / "splits.parquet")
        self.panel, self.news = Panel.load(store, strict_calendar=False), store.read_news()
        self.settings = Settings(min_price=0, min_dollar_volume=0)
        chunk = mock.patch.object(module, "_CHUNK", 2)  # Chunks of 2, 2, 2 and 1 of the 7 tickers, as at scale.
        chunk.start()
        self.addCleanup(chunk.stop)

    def tearDown(self):
        self.directory.cleanup()

    def data(self, cutoff: int, lookback: int = LOOKBACK) -> FoldData:
        return FoldData(self.panel, self.news, self.settings, self.panel.sessions[cutoff], 2, lookback, 7)

    def test_training_rows_match_the_decision_window(self):
        data = self.data(104)
        table = training_features(data)
        self.assertEqual(list(table.columns), list(data.labels().columns) + FEATURES)
        pd.testing.assert_frame_equal(table.drop(columns=FEATURES), data.labels())
        self.assertEqual(table.decision_date.min(), self.panel.sessions[LOOKBACK - 1])
        for t in (LOOKBACK - 1, 75, 84, 85, 90, 101):
            day = self.panel.sessions[t]
            rows = table[table.decision_date == day].set_index("ticker")[FEATURES]
            expected = decision_features(data.snapshot(day).bars).reindex(rows.index)
            np.testing.assert_allclose(rows.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=1e-15)
            # A split is no move: S, which split at session 85, has the features of its twin.
            np.testing.assert_allclose(rows.loc["S"].to_numpy(), rows.loc["T000"].to_numpy(), rtol=1e-12)
        missing = table[(table.ticker == "T002") & (table.decision_date == self.panel.sessions[80])]
        self.assertTrue(missing.high_60d.isna().all() and missing.ret_5d.notna().all())

    def test_chunks_change_nothing_and_a_short_lookback_is_refused(self):
        chunked = training_features(self.data(104))
        with mock.patch.object(module, "_CHUNK", 500):
            pd.testing.assert_frame_equal(training_features(self.data(104)), chunked)
        with self.assertRaisesRegex(ValueError, "bars_lookback"):
            training_features(self.data(104, LOOKBACK - 1))  # predict would lack the oldest bar the features read.

    def test_later_bars_change_no_earlier_row(self):
        early, late = training_features(self.data(90)), training_features(self.data(104))
        keys = ["decision_date", "ticker"]
        merged = early.merge(late, on=keys, suffixes=("", "_late"))
        self.assertEqual(len(merged), len(early))
        np.testing.assert_array_equal(merged[FEATURES].to_numpy(),
                                      merged[[f"{name}_late" for name in FEATURES]].to_numpy())


if __name__ == "__main__":
    unittest.main()
