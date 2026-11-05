"""The nested walk-forward choice, the $10,000 account, its order-size tuning, and the final selection."""
import json
from pathlib import Path
import pickle
import tempfile
import unittest
import numpy as np
import pandas as pd
from stock_recommender.backtest import metrics
from stock_recommender.backtest.account import book, equal_split, trade, tune
from stock_recommender.backtest.panel import Panel
from stock_recommender.backtest.portfolio import simulate
from stock_recommender.backtest.protocol import Period, news_window
from stock_recommender.backtest.report import write_run
from stock_recommender.backtest.run import SIZES, Candidate, load_data, run_backtest, select
from stock_recommender.backtest.window import LeakageError, WindowedStore
from stock_recommender.config import Settings
from stock_recommender.models import RandomPriceModel
from stock_recommender.recommend import news_cutoff, score_day
from .backtest_fixtures import (SESSIONS, config, random_walk, small_models, write_bars, write_dividends,
                                write_news, write_splits)

THREE_YEARS = pd.bdate_range("2023-11-01", "2025-03-31")  # 2023 trains, then the test years 2024 and 2025.


class Oracle:
    """Test-only: ranks by the realized return of the holding ahead; the wrong way round outside `good` years."""

    def __init__(self, opens: pd.DataFrame, holding: int, good=None):
        self.opens, self.holding, self.good = opens, holding, None if good is None else int(good)

    def predict(self, bars, tickers):
        day = bars.date.max()
        t = self.opens.index.get_loc(day)
        realized = self.opens.iloc[t + 1 + self.holding] / self.opens.iloc[t + 1] - 1
        sign = 1.0 if self.good is None or day.year == self.good else -1.0
        return (sign * realized).reindex(tickers)


class NestedTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.prices = random_walk(30, THREE_YEARS, seed=2)
        write_bars(self.root / "bars", self.prices["open"], self.prices["close"])
        write_splits(self.root / "splits.parquet", [])
        write_news(self.root / "news", [("2024-06-03T15:00:00Z", ["T000"])])
        self.opens = self.prices["open"]

    def tearDown(self):
        self.directory.cleanup()

    def test_a_test_year_is_chosen_on_earlier_years_only(self):
        factories = {"oracle": lambda good="2025": small_models(price=Oracle(self.opens, 2, good))}
        grids = {"oracle": {"good": ("2025", "2024")}}  # The default, "2025", is right only in 2025.
        result = run_backtest(config(self.root, models=("oracle",), test_years=(2024, 2025)), factories,
                              grids=grids)
        folds = result.folds.set_index("fold")
        self.assertFalse(folds.loc[2024, "tuned"])  # No earlier test year: the default runs.
        self.assertEqual(json.loads(folds.loc[2024, "chosen"]), {"oracle": {"good": "2025"}})
        # 2025 is chosen on 2024 alone: the candidate that was right in 2024, though it is wrong in 2025,
        # over the one right in 2025, which was wrong in 2024.
        self.assertTrue(folds.loc[2025, "tuned"])
        self.assertEqual(folds.loc[2025, "validation_years"], "2024")
        self.assertEqual(json.loads(folds.loc[2025, "chosen"]), {"oracle": {"good": "2024"}})
        scores = result.scores.set_index(["candidate", "year"]).growth
        self.assertGreater(scores['oracle{"good": "2025"}', 2025], 1.0)
        self.assertLess(scores['oracle{"good": "2024"}', 2025], 1.0)
        growth = result.metrics.set_index(["strategy", "scope"]).growth
        self.assertLess(growth["oracle", "2025"], 1.0)  # The choice fails, as it would have live.
        self.assertEqual(result.guards["selection_years"], {"2024": [], "2025": [2024]})
        self.assertEqual(result.selection["chosen_on_years"], [2024, 2025])

    def test_the_account_sizes_up_the_model_with_skill(self):
        factories = {"oracle": lambda: small_models(price=Oracle(self.opens, 2)),
                     "random": lambda: small_models(price=RandomPriceModel(0))}
        result = run_backtest(config(self.root, models=("oracle", "random"), test_years=(2024, 2025), sweeps=2),
                              factories)
        sizes = json.loads(result.folds.set_index("fold").loc[2025, "sizes"])
        self.assertGreater(sizes["oracle"], sizes["random"])  # Tuned on 2024.
        growth = result.metrics.set_index(["strategy", "scope"]).growth
        self.assertGreater(growth["account", "2025"], growth["random", "2025"])
        final = result.selection["models"]
        self.assertGreater(final["oracle"]["order_size"], final["random"]["order_size"])
        self.assertFalse(result.selection["final_evaluated"])

    def test_select_never_uses_the_year_it_chooses_for(self):
        scores = pd.DataFrame({"model": "m", "candidate": ["a", "b", "a", "b"], "year": [2024, 2024, 2025, 2025],
                               "log_growth": [0.1, 0.2, 0.9, -0.9]})
        candidates = [Candidate("m", (), "a", "a"), Candidate("m", (("x", 1),), "b", "b")]
        self.assertEqual(select(scores, candidates, [2024], before=2025), {"m": "b"})
        self.assertEqual(select(scores, candidates, [], before=2024), {"m": "a"})  # No earlier year: the default.
        with self.assertRaises(LeakageError):
            select(scores, candidates, [2024, 2025], before=2025)
        tied = scores.assign(log_growth=0.0)
        self.assertEqual(select(tied, candidates, [2024, 2025]), {"m": "a"})  # Ties keep grid order.


class AccountTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.sessions = SESSIONS[:20]

    def tearDown(self):
        self.directory.cleanup()

    def panel(self, opens: dict, splits=(), dividends=None) -> Panel:
        write_bars(self.root / "bars", pd.DataFrame(opens, index=self.sessions, dtype=float))
        write_splits(self.root / "splits.parquet", list(splits))
        path = None
        if dividends is not None:
            path = self.root / "dividends.parquet"
            write_dividends(path, dividends)
        store = WindowedStore(self.root / "bars", self.root / "news", self.root / "splits.parquet")
        return Panel.load(store, strict_calendar=False, dividends_path=path)

    @staticmethod
    def listed(tickers, prices) -> pd.DataFrame:
        return pd.DataFrame({"ticker": tickers, "price": [float(price) for price in prices]})

    def test_fractional_orders_fill_at_exactly_their_size(self):
        opens = [100.0, 100.0, 110.0, 121.0, 133.1] + [133.1] * 15
        panel = self.panel({"A": opens})
        item = book(panel, "m", [Period(1, 2)], [self.listed(["A"], [100])])
        curve, orders = trade(panel.sessions, [item], [0.5], 1000.0, 0.0, record=True)
        self.assertAlmostEqual(orders.cost.iloc[0], 500.0)  # Half the account at the 110 open.
        self.assertAlmostEqual(orders.shares.iloc[0], 500 / 110)
        self.assertAlmostEqual(curve.iloc[-1], 500 + 500 * 133.1 / 110)
        self.assertEqual((curve.index[0], curve.index[-1]), (self.sessions[2], self.sessions[4]))
        self.assertEqual(orders.reason.iloc[0], "filled")

    def test_whole_shares_round_down_and_skip_a_stock_pricier_than_its_order(self):
        panel = self.panel({"A": [100.0] * 20, "B": [300.0] * 20})
        item = book(panel, "m", [Period(1, 2)], [self.listed(["A", "B"], [100, 300])])
        curve, orders = trade(panel.sessions, [item], [0.25], 1000.0, 0.0, whole_shares=True, record=True)
        rows = orders.set_index("ticker")
        self.assertEqual((rows.shares["A"], rows.shares["B"]), (2.0, 0.0))  # floor(250 / 100), and 300 > 250.
        self.assertEqual(rows.reason.to_dict(), {"A": "filled", "B": "price_above_order"})
        self.assertAlmostEqual(curve.iloc[-1], 1000.0)

    def test_whole_shares_are_sized_on_the_entry_days_split_basis(self):
        split = 2  # 2-for-1 at the entry open: the list price is the pre-split close.
        panel = self.panel({"A": [100.0] * split + [50.0] * (20 - split)},
                           splits=[("A", str(self.sessions[split].date()), 1, 2)])
        item = book(panel, "m", [Period(1, 2)], [self.listed(["A"], [100])])
        _, orders = trade(panel.sessions, [item], [0.25], 1000.0, 0.0, whole_shares=True, record=True)
        self.assertEqual(orders.shares.iloc[0], 5.0)  # 250 / 50, not 250 / 100.

    def test_orders_beyond_the_cash_are_scaled_down_together(self):
        panel = self.panel({"A": [100.0] * 20, "B": [50.0] * 20})
        item = book(panel, "m", [Period(1, 2)], [self.listed(["A", "B"], [100, 50])])
        curve, orders = trade(panel.sessions, [item], [0.75], 1000.0, 10.0, record=True)  # 150% of the account.
        self.assertEqual(set(orders.reason), {"scaled"})
        self.assertAlmostEqual(orders.cost.sum() * 1.001, 1000.0)  # All the cash, costs included, and no more.
        self.assertAlmostEqual(orders.cost.iloc[0], orders.cost.iloc[1])
        self.assertGreaterEqual(curve.min(), 0.0)

    def test_each_model_trades_on_its_own_horizon(self):
        panel = self.panel({"A": [100.0] * 20, "B": [100.0] * 20})
        short = book(panel, "short", [Period(1, 2), Period(3, 2)], [self.listed(["A"], [100])] * 2)
        long = book(panel, "long", [Period(1, 3)], [self.listed(["B"], [100])])
        curve, orders = trade(panel.sessions, [short, long], [0.25, 0.25], 1000.0, 0.0, record=True)
        exits = orders.groupby("model").exit_date.apply(list)
        self.assertEqual(exits["short"], [self.sessions[4], self.sessions[6]])
        self.assertEqual(exits["long"], [self.sessions[5]])
        self.assertEqual((curve.index[0], curve.index[-1]), (self.sessions[2], self.sessions[6]))

    def test_a_split_and_a_dividend_during_a_lot_are_valued_like_forward_returns(self):
        opens = [100.0] * 5 + [50.0] * 3 + [49.5] * 4 + [55.0] * 8  # 2-for-1 at 5; $0.50 ex-dividend at 8.
        panel = self.panel({"A": opens}, splits=[("A", str(self.sessions[5].date()), 1, 2)],
                           dividends=[("A", str(self.sessions[8].date()), 0.5)])
        item = book(panel, "m", [Period(2, 9)], [self.listed(["A"], [100])])
        curve, _ = trade(panel.sessions, [item], [1.0], 1000.0, 0.0)
        expected = panel.forward_returns(2, 9, ["A"]).forward_return["A"]
        self.assertAlmostEqual(curve.iloc[-1], 1000 * (1 + expected))
        self.assertGreater(expected, 0.09)  # The split is no loss, and the dividend is income.

    def test_costs_are_netted_when_a_stock_is_sold_and_bought_at_the_same_open(self):
        panel = self.panel({"A": [100.0] * 20})
        item = book(panel, "m", [Period(1, 2), Period(3, 2)], [self.listed(["A"], [100])] * 2)
        curve, _ = trade(panel.sessions, [item], [0.5], 1000.0, 100.0)  # 1% a side.
        # Buy 500 (cost 5); at the next entry sell 500 and buy 497.5, paying on the 2.5 difference; sell 497.5.
        self.assertAlmostEqual(curve.iloc[-1], 1000 - 5 - 0.025 - 4.975)

    def test_a_halted_stock_is_not_entered_and_a_delisted_one_exits_at_its_last_close(self):
        halted = [10.0, 10.0, np.nan] + [10.0] * 17
        delisted = [10.0, 10.0, 10.0, 12.0] + [np.nan] * 16
        panel = self.panel({"H": halted, "D": delisted})
        item = book(panel, "m", [Period(1, 3)], [self.listed(["H", "D"], [10, 10])])
        curve, orders = trade(panel.sessions, [item], [0.5], 1000.0, 0.0, record=True)
        self.assertEqual(orders.set_index("ticker").reason.to_dict(), {"H": "no_entry", "D": "filled"})
        self.assertAlmostEqual(curve.iloc[-1], 500 + 500 * 1.2)

    def test_zero_sizes_leave_the_account_in_cash(self):
        panel = self.panel({"A": [100.0, 100.0, 150.0, 90.0, 120.0] + [100.0] * 15})
        item = book(panel, "m", [Period(1, 2), Period(3, 2)], [self.listed(["A"], [100])] * 2)
        curve, orders = trade(panel.sessions, [item], [0.0], 1000.0, 5.0, record=True)
        self.assertTrue((curve == 1000.0).all())
        self.assertEqual(set(orders.reason), {"size_zero"})

    def test_one_model_at_an_equal_size_matches_its_own_simulation(self):
        prices = random_walk(6, self.sessions, seed=4)
        write_bars(self.root / "bars", prices["open"], prices["close"])
        write_splits(self.root / "splits.parquet", [])
        panel = Panel.load(WindowedStore(self.root / "bars", self.root / "news", self.root / "splits.parquet"),
                           strict_calendar=False)
        tickers = ["T000", "T001", "T002", "T003"]
        period = Period(5, 4)
        item = book(panel, "m", [period], [self.listed(tickers, prices["close"].iloc[5][tickers])])
        curve, _ = trade(panel.sessions, [item], [0.25], 1000.0, 0.0)
        simulated, _, _ = simulate(panel, [period], [pd.Series(0.25, index=tickers)], 0.0, 0.0, 1000.0, [2024])
        pd.testing.assert_series_equal(curve, simulated, check_names=False)


class TuneTest(unittest.TestCase):
    def test_tune_moves_one_size_at_a_time_to_the_best(self):
        calls = []

        def score(sizes):
            calls.append(sizes)
            return sizes[0] - 2 * sizes[1]

        best, trace = tune(score, (0.25, 0.25), (0.0, 0.25, 0.5), sweeps=3)
        self.assertEqual(best, (0.5, 0.0))
        self.assertEqual(len(calls), len(set(calls)))  # Each set of sizes is scored once.
        self.assertEqual(trace.objective.max(), 0.5)

    def test_ties_keep_the_smaller_size(self):
        best, _ = tune(lambda sizes: 0.0, (0.25, 0.25), (0.0, 0.25, 0.5))
        self.assertEqual(best, (0.0, 0.0))  # Trading that adds nothing is not worth its risk.

    def test_equal_split_fully_invests_the_account(self):
        self.assertEqual(equal_split([20, 20], SIZES), (0.025, 0.025))
        self.assertEqual(equal_split([10, 20], (0.0, 0.01, 0.03, 0.05)), (0.03, 0.03))  # At most 1/30 each.


class OutputTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.prices = random_walk(20)
        write_bars(self.root / "bars", self.prices["open"], self.prices["close"])
        write_splits(self.root / "splits.parquet", [])
        write_news(self.root / "news", [("2025-01-15T15:00:00Z", ["T000"])])

    def tearDown(self):
        self.directory.cleanup()

    def test_a_models_trades_are_its_best_picks_at_their_close(self):
        store, panel, news = load_data(config(self.root))
        settings = Settings(min_price=0, min_dollar_volume=0, refresh_time="off")
        model = small_models(picks=3, price=RandomPriceModel(1))
        t = 20
        result = score_day(panel.window(t, model.bars_lookback), news_window(news, news_cutoff(panel.sessions[t]), 7),
                           panel.sessions[t], settings, model)
        self.assertEqual(list(result.trades.columns), ["ticker", "price"])
        self.assertEqual(list(result.trades.ticker), list(result.table.ticker[:3]))
        closes = self.prices["close"].iloc[t]
        self.assertTrue(np.allclose(result.trades.price, closes[result.trades.ticker].to_numpy()))

    def test_daily_return_and_log_growth(self):
        curve = pd.Series([100.0, 110.0, 99.0, 120.0], index=pd.bdate_range("2025-01-01", periods=4))
        summary = metrics.performance(curve)
        self.assertAlmostEqual(summary["daily_return"], np.mean([0.1, -0.1, 120 / 99 - 1]))
        self.assertAlmostEqual(summary["log_growth"], 252 * np.log(1.2) / 3)
        self.assertEqual(metrics.log_growth(pd.Series([100.0, 0.0])), -np.inf)

    def test_a_run_saves_the_final_choice_for_2026_without_evaluating_it(self):
        result = run_backtest(config(self.root), {"flat": small_models})
        self.assertEqual(set(result.final_models), {"flat"})
        self.assertEqual(result.final_models["flat"].picks, 2)
        choice = result.selection["models"]["flat"]
        self.assertEqual((choice["picks"], choice["horizon_days"], choice["bars_lookback"]), (2, 2, 3))
        self.assertIn(choice["order_size"], config(self.root).sizes)
        self.assertEqual(result.selection["training_end"], "2025-03-31")
        folder = write_run(result, "test", root=self.root)
        selection = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
        self.assertFalse(selection["final_evaluated"])
        self.assertEqual(selection["saved_models"], {"flat": "model/flat.pkl"})
        self.assertEqual(pickle.loads((folder / "model/flat.pkl").read_bytes()).picks, 2)
        for table in ("scores", "folds", "orders", "sizes", "yearly", "metrics"):
            self.assertTrue((folder / "tables" / f"{table}.csv").is_file(), table)


if __name__ == "__main__":
    unittest.main()
