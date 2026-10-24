from dataclasses import replace
import unittest
import pandas as pd
from stock_recommender.config import Settings
from stock_recommender.ensemble import Ensemble, normalize_scores
from stock_recommender.models import placeholder_models
from stock_recommender.recommend import score_day

TICKERS = pd.Index(["A", "B", "C"])


class Fixed:
    def __init__(self, scores: dict[str, float]):
        self.scores = pd.Series(scores)
        self.fitted = 0

    def fit(self, data):
        self.fitted += 1

    def predict(self, bars, tickers):
        return self.scores


def model(scores, **changes):
    return replace(placeholder_models(), price=Fixed(scores), **changes)


def bars(days: int = 3) -> pd.DataFrame:
    dates = pd.bdate_range("2025-03-03", periods=days)
    return pd.DataFrame([{"date": day, "ticker": ticker, "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
                          "volume": volume, "vwap": 10.0, "transactions": 1}
                         for day in dates for ticker, volume in zip(TICKERS, (3e6, 2e6, 1e6))])


NO_NEWS = pd.DataFrame({"published_utc": pd.Series(dtype="datetime64[ms, UTC]"), "tickers": pd.Series(dtype=object)})


class EnsembleTest(unittest.TestCase):
    def test_weights_decide_between_disagreeing_models(self):
        up = model({"A": 0.03, "B": 0.02, "C": 0.01})
        down = model({"A": -0.5, "B": 0.0, "C": 0.5})  # Opposite order, in another unit.
        heavy_up = Ensemble(((up, 0.75), (down, 0.25))).score(bars(), NO_NEWS, TICKERS)
        self.assertEqual(list(heavy_up.score.sort_values(ascending=False).index), ["A", "B", "C"])
        heavy_down = Ensemble(((up, 0.25), (down, 0.75))).score(bars(), NO_NEWS, TICKERS)
        self.assertEqual(list(heavy_down.score.sort_values(ascending=False).index), ["C", "B", "A"])
        self.assertAlmostEqual(heavy_up.expected_return["A"], 0.75 * 0.03 + 0.25 * -0.5)

    def test_a_member_that_skips_a_ticker_leaves_it_to_the_others(self):
        full = model({"A": 1.0, "B": 2.0, "C": 3.0})
        partial = model({"A": 9.0, "B": 8.0})
        scores = Ensemble(((full, 1.0), (partial, 1.0))).score(bars(), NO_NEWS, TICKERS)
        self.assertAlmostEqual(scores.score["C"], normalize_scores(pd.Series({"A": 1.0, "B": 2.0, "C": 3.0}), "rank")["C"])
        nobody = Ensemble(((model({"A": 1.0}), 1.0),)).score(bars(), NO_NEWS, TICKERS)
        self.assertEqual(list(nobody.index), ["A"])

    def test_placeholders_combine_to_the_liquidity_order(self):
        ensemble = Ensemble(((placeholder_models(), 1.0), (placeholder_models(), 2.0)))
        table = score_day(bars(), NO_NEWS, bars().date.max(), Settings(min_price=0, min_dollar_volume=0), ensemble).table
        self.assertEqual(list(table.ticker), ["A", "B", "C"])  # All tied at 0: the most traded first.

    def test_members_see_their_own_lookback(self):
        seen = []

        class Recorder(Fixed):
            def predict(self, bars, tickers):
                seen.append(bars.date.nunique())
                return self.scores

        short = replace(placeholder_models(), price=Recorder({"A": 1.0}), bars_lookback=2)
        long = replace(placeholder_models(), price=Recorder({"A": 1.0}), bars_lookback=5)
        ensemble = Ensemble(((short, 1.0), (long, 1.0)))
        self.assertEqual(ensemble.bars_lookback, 5)
        ensemble.score(bars(5), NO_NEWS, TICKERS)
        self.assertEqual(seen, [2, 5])

    def test_fit_reaches_every_member_once(self):
        members = (model({"A": 1.0}), model({"B": 1.0}))
        Ensemble(tuple((member, 1.0) for member in members)).fit(object())
        self.assertEqual([member.price.fitted for member in members], [1, 1])

    def test_invalid_ensembles(self):
        with self.assertRaises(ValueError):
            Ensemble(((model({"A": 1.0}), 1.0), (model({"A": 1.0}, horizon_days=10), 1.0)))
        with self.assertRaises(ValueError):
            Ensemble(((model({"A": 1.0}), -1.0), (model({"A": 1.0}), 2.0)))
        with self.assertRaises(ValueError):
            Ensemble(((model({"A": 1.0}), 1.0),), normalize="softmax")
        with self.assertRaises(ValueError):
            Ensemble(())

    def test_version_names_the_members_and_weights(self):
        ensemble = Ensemble(((model({}, version="a"), 3.0), (model({}, version="b"), 1.0)))
        self.assertEqual(ensemble.version, "ensemble(a:0.75,b:0.25)")


if __name__ == "__main__":
    unittest.main()
