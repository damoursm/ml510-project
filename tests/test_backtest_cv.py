"""The nested walk-forward choice, the account that buys every recommendation for $1,000, its references, and the
final selection."""
import json
from pathlib import Path
import pickle
import tempfile
import unittest
import numpy as np
import pandas as pd
from stock_recommender.backtest.account import basket, book, hold, mirror, spent, trade
from stock_recommender.backtest.panel import Panel
from stock_recommender.backtest.protocol import Period, news_window
from stock_recommender.backtest.registry import GRIDS, UNTUNED
from stock_recommender.backtest.report import write_run, yearly_table
from stock_recommender.backtest.run import Candidate, load_data, run_backtest, select
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
        scores = result.scores.set_index(["candidate", "year"]).yearly_return
        self.assertGreater(scores['oracle{"good": "2025"}', 2025], 0.0)
        self.assertLess(scores['oracle{"good": "2024"}', 2025], 0.0)
        returns = result.metrics.set_index(["strategy", "scope"]).yearly_return
        self.assertLess(returns["oracle", "2025"], 0.0)  # The choice fails, as it would have live.
        self.assertEqual(result.guards["selection_years"], {"2024": [], "2025": [2024]})
        self.assertEqual(result.selection["chosen_on_years"], [2024, 2025])

    def test_the_account_buys_every_models_list_for_the_order_value(self):
        factories = {"oracle": lambda: small_models(price=Oracle(self.opens, 2)),
                     "random": lambda: small_models(price=RandomPriceModel(0))}
        result = run_backtest(config(self.root, models=("oracle", "random"), test_years=(2024, 2025)), factories)
        scoped = result.metrics.set_index(["strategy", "scope"])
        for year in ("2024", "2025"):
            # Without costs the account is its models added up: two lists of two stocks, $1,000 each, at all times.
            self.assertAlmostEqual(scoped.profit["account", year],
                                   scoped.profit["oracle", year] + scoped.profit["random", year])
            self.assertEqual(scoped.capital["account", year], 4000.0)
            self.assertEqual(scoped.capital["oracle", year], 2000.0)
        self.assertGreater(scoped.yearly_return["oracle", "2025"], scoped.yearly_return["random", "2025"])
        self.assertTrue(np.allclose(result.orders.cost, 1000.0))  # Never scaled, never compounded.
        self.assertEqual(set(result.orders.model), {"oracle", "random"})
        self.assertEqual(result.profit.account.iloc[-1], scoped.profit["account", "all"])
        folds = result.folds.set_index("fold")
        self.assertAlmostEqual(folds.loc[2025, "account_return"], scoped.yearly_return["account", "2025"])
        self.assertFalse(folds.tuned.any())  # No grid: nothing is chosen, the models run as they are.
        self.assertEqual(result.selection["nested_estimate"]["untuned_years"], [2024, 2025])
        self.assertEqual(result.selection["models"]["oracle"]["order_value"], 1000.0)
        self.assertFalse(result.selection["final_evaluated"])

    def test_no_grid_tunes_how_many_stocks_a_model_buys(self):
        # Every recommended stock is bought for the order value, so picks, which sets the capital, is the model's.
        self.assertFalse(any(UNTUNED & set(grid) for grid in GRIDS.values()))
        for grids in ({"flat": {"picks": (2, 1)}}, {"flat": {"bars_lookback": (3,), "picks": (2,)}}):
            with self.assertRaisesRegex(ValueError, "picks"):
                run_backtest(config(self.root), {"flat": small_models}, grids=grids)
        ensemble = config(self.root, models=(), ensembles=(("mix", (("flat", 1.0),)),))
        with self.assertRaisesRegex(ValueError, "picks"):  # Nor through an ensemble's members.
            run_backtest(ensemble, {"flat": small_models}, grids={"flat": {"picks": (2, 1)}})
        # Nor through a hyperparameter of the model's own that sets picks; refused before any data is read.
        with self.assertRaisesRegex(ValueError, r"changes picks \(\[2, 5\]\)"):
            run_backtest(config(self.root / "missing"), {"flat": lambda top=2: small_models(picks=top)},
                         grids={"flat": {"top": (2, 5)}})
        run_backtest(config(self.root), {"flat": lambda top=2: small_models(picks=top)},
                     grids={"flat": {"top": (2,)}})  # One value: picks never changes, so it runs.

    def test_select_never_uses_the_year_it_chooses_for(self):
        scores = pd.DataFrame({"model": "m", "candidate": ["a", "b", "a", "b"], "year": [2024, 2024, 2025, 2025],
                               "yearly_return": [0.1, 0.2, 0.9, -0.9]})
        candidates = [Candidate("m", (), "a"), Candidate("m", (("x", 1),), "b")]
        self.assertEqual(select(scores, candidates, [2024], before=2025), {"m": "b"})
        self.assertEqual(select(scores, candidates, [], before=2024), {"m": "a"})  # No earlier year: the default.
        with self.assertRaises(LeakageError):
            select(scores, candidates, [2024, 2025], before=2025)
        tied = scores.assign(yearly_return=0.0)
        self.assertEqual(select(tied, candidates, [2024, 2025]), {"m": "a"})  # Ties keep grid order.


class LedgerCase(unittest.TestCase):
    """A store of 20 sessions written by each test, and lists to price on it."""

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


class AccountTest(LedgerCase):
    def test_fractional_orders_buy_exactly_the_order_value_from_zero(self):
        opens = [100.0, 100.0, 110.0, 121.0, 133.1] + [133.1] * 15
        panel = self.panel({"A": opens})
        item = book(panel, "m", [Period(1, 2)], [self.listed(["A"], [100])], 1000.0)
        ledger = trade(panel.sessions, [item], 0.0, record=True)
        self.assertAlmostEqual(ledger.orders.cost.iloc[0], 1000.0)  # At the 110 open.
        self.assertAlmostEqual(ledger.orders.shares.iloc[0], 1000 / 110)
        self.assertEqual(ledger.orders.reason.iloc[0], "filled")
        self.assertEqual(list(ledger.profit.round(6)), [0.0, 100.0, 210.0])  # From $0, at the opens it is held.
        self.assertEqual((ledger.profit.index[0], ledger.profit.index[-1]), (self.sessions[2], self.sessions[4]))
        self.assertEqual(ledger.capital, 1000.0)
        self.assertAlmostEqual(ledger.yearly_return, 133.1 / 110 - 1)

    def test_whole_shares_round_down_and_skip_a_stock_pricier_than_the_order(self):
        panel = self.panel({"A": [100.0] * 20, "B": [300.0] * 20})
        item = book(panel, "m", [Period(1, 2)], [self.listed(["A", "B"], [100, 300])], 250.0)
        ledger = trade(panel.sessions, [item], 0.0, whole_shares=True, record=True)
        rows = ledger.orders.set_index("ticker")
        self.assertEqual((rows.shares["A"], rows.shares["B"]), (2.0, 0.0))  # floor(250 / 100), and 300 > 250.
        self.assertEqual(rows.reason.to_dict(), {"A": "filled", "B": "price_above_order"})
        self.assertEqual(ledger.capital, 200.0)  # What the orders cost, not what they were worth.
        self.assertAlmostEqual(ledger.profit.iloc[-1], 0.0)

    def test_whole_shares_are_sized_on_the_entry_days_split_basis(self):
        split = 2  # 2-for-1 at the entry open: the list price is the pre-split close.
        panel = self.panel({"A": [100.0] * split + [50.0] * (20 - split)},
                           splits=[("A", str(self.sessions[split].date()), 1, 2)])
        item = book(panel, "m", [Period(1, 2)], [self.listed(["A"], [100])], 250.0)
        ledger = trade(panel.sessions, [item], 0.0, whole_shares=True, record=True)
        self.assertEqual(ledger.orders.shares.iloc[0], 5.0)  # 250 / 50, not 250 / 100.

    def test_orders_are_never_scaled_and_the_capital_is_the_most_they_cost_at_once(self):
        panel = self.panel({"A": [100.0] * 20, "B": [50.0] * 20})
        short = book(panel, "short", [Period(1, 2), Period(3, 2)], [self.listed(["A", "B"], [100, 50])] * 2, 1000.0)
        long = book(panel, "long", [Period(1, 3)], [self.listed(["A"], [100])], 1000.0)
        ledger = trade(panel.sessions, [short, long], 10.0, record=True)
        self.assertTrue(np.allclose(ledger.orders.cost, 1000.0))  # However many orders enter at once.
        self.assertEqual(list(ledger.invested), [3000.0, 3000.0, 3000.0, 2000.0, 0.0])
        self.assertEqual(ledger.capital, 3000.0)
        # 0.1% of 3,000 bought at the first entry, 1,000 sold at 5 and 2,000 at 6; the re-entry at 4 nets out.
        self.assertAlmostEqual(ledger.profit.iloc[-1], -0.001 * (3000 + 1000 + 2000))

    def test_each_model_trades_on_its_own_horizon(self):
        panel = self.panel({"A": [100.0] * 20, "B": [100.0] * 20})
        short = book(panel, "short", [Period(1, 2), Period(3, 2)], [self.listed(["A"], [100])] * 2, 250.0)
        long = book(panel, "long", [Period(1, 3)], [self.listed(["B"], [100])], 250.0)
        ledger = trade(panel.sessions, [short, long], 0.0, record=True)
        exits = ledger.orders.groupby("model").exit_date.apply(list)
        self.assertEqual(exits["short"], [self.sessions[4], self.sessions[6]])
        self.assertEqual(exits["long"], [self.sessions[5]])
        self.assertEqual((ledger.profit.index[0], ledger.profit.index[-1]), (self.sessions[2], self.sessions[6]))

    def test_a_split_and_a_dividend_during_a_lot_are_valued_like_forward_returns(self):
        opens = [100.0] * 5 + [50.0] * 3 + [49.5] * 4 + [55.0] * 8  # 2-for-1 at 5; $0.50 ex-dividend at 8.
        panel = self.panel({"A": opens}, splits=[("A", str(self.sessions[5].date()), 1, 2)],
                           dividends=[("A", str(self.sessions[8].date()), 0.5)])
        item = book(panel, "m", [Period(2, 9)], [self.listed(["A"], [100])], 1000.0)
        ledger = trade(panel.sessions, [item], 0.0)
        expected = panel.forward_returns(2, 9, ["A"]).forward_return["A"]
        self.assertAlmostEqual(ledger.profit.iloc[-1], 1000 * expected)
        self.assertGreater(expected, 0.09)  # The split is no loss, and the dividend is income.

    def test_costs_are_netted_when_a_stock_is_sold_and_bought_at_the_same_open(self):
        panel = self.panel({"A": [100.0] * 20})
        item = book(panel, "m", [Period(1, 2), Period(3, 2)], [self.listed(["A"], [100])] * 2, 500.0)
        ledger = trade(panel.sessions, [item], 100.0)  # 1% a side.
        # Buy 500 (cost 5); at the next entry sell 500 and buy 500, which nets to nothing; sell 500 (cost 5).
        self.assertAlmostEqual(ledger.profit.iloc[-1], -10.0)

    def test_a_halted_stock_is_not_entered_and_a_delisted_one_exits_at_its_last_close(self):
        halted = [10.0, 10.0, np.nan] + [10.0] * 17
        delisted = [10.0, 10.0, 10.0, 12.0] + [np.nan] * 16
        panel = self.panel({"H": halted, "D": delisted})
        item = book(panel, "m", [Period(1, 3)], [self.listed(["H", "D"], [10, 10])], 500.0)
        ledger = trade(panel.sessions, [item], 0.0, record=True)
        self.assertEqual(ledger.orders.set_index("ticker").reason.to_dict(), {"H": "no_entry", "D": "filled"})
        self.assertAlmostEqual(ledger.profit.iloc[-1], 500 * 0.2)
        self.assertEqual(ledger.capital, 500.0)

    def test_a_ledger_without_costs_earns_the_forward_returns_of_its_list(self):
        prices = random_walk(6, self.sessions, seed=4)
        write_bars(self.root / "bars", prices["open"], prices["close"])
        write_splits(self.root / "splits.parquet", [])
        panel = Panel.load(WindowedStore(self.root / "bars", self.root / "news", self.root / "splits.parquet"),
                           strict_calendar=False)
        tickers = ["T000", "T001", "T002", "T003"]
        period = Period(5, 4)
        item = book(panel, "m", [period], [self.listed(tickers, prices["close"].iloc[5][tickers])], 250.0)
        ledger = trade(panel.sessions, [item], 0.0)
        returns = panel.forward_returns(period.decision, period.holding, tickers).forward_return
        self.assertAlmostEqual(ledger.profit.iloc[-1], 250.0 * returns.sum())  # Panel's own valuation agrees.
        self.assertEqual(ledger.capital, 1000.0)
        self.assertEqual(ledger.traded.iloc[0], 1000.0)
        self.assertEqual(float(ledger.fees.sum()), 0.0)


class ReferenceTest(LedgerCase):
    def test_spy_is_bought_for_every_filled_order_and_held_with_the_capital(self):
        spy = [400.0, 400.0, 400.0, 404.0, 408.0, 412.0, 416.0] + [416.0] * 13
        panel = self.panel({"A": [100.0] * 20, "H": [10.0, 10.0, np.nan] + [10.0] * 17, "SPY": spy})
        item = book(panel, "m", [Period(1, 2), Period(3, 2)], [self.listed(["A", "H"], [100, 10])] * 2, 1000.0)
        account = trade(panel.sessions, [item], 0.0)
        paid = spent(item)
        self.assertEqual(list(paid), [1000.0, 0.0, 1000.0, 1000.0])  # H does not trade at the first entry.
        self.assertEqual(account.capital, 2000.0)
        mirrored = trade(panel.sessions, [mirror(panel, item, paid)], 0.0, record=True)
        self.assertEqual(len(mirrored.orders), 3)  # One SPY lot per filled order, on its dates.
        self.assertAlmostEqual(mirrored.profit.iloc[-1], 1000 * (408 / 400 - 1) + 2000 * (416 / 408 - 1))
        held = trade(panel.sessions, [hold(panel, "SPY", item.start, item.end, account.capital)], 0.0)
        self.assertAlmostEqual(held.profit.iloc[-1], 2000 * (416 / 400 - 1))
        self.assertAlmostEqual(held.yearly_return, 416 / 400 - 1)
        self.assertEqual(list(held.profit.index), list(account.profit.index))

    def test_the_universe_reference_spreads_each_orders_dollars_over_the_universe(self):
        panel = self.panel({"A": [100.0] * 20, "B": [50.0] * 3 + [55.0] * 17, "C": [20.0, 20.0, np.nan] + [20.0] * 17})
        item = book(panel, "m", [Period(1, 2)], [self.listed(["A"], [100])], 1000.0)
        spread = basket(panel, item, spent(item), lambda t: pd.Index(["A", "B", "C"]))
        ledger = trade(panel.sessions, [spread], 0.0, record=True)
        rows = ledger.orders.set_index("ticker")
        self.assertEqual(rows.order_value.to_dict(), {"A": 500.0, "B": 500.0, "C": 0.0})  # C does not trade then.
        self.assertAlmostEqual(ledger.profit.iloc[-1], 500 * (55 / 50 - 1))

    def test_a_reference_that_never_trades_has_no_return(self):
        panel = self.panel({"A": [100.0] * 20})
        item = book(panel, "m", [Period(1, 2)], [self.listed(["A"], [100])], 1000.0)
        ledger = trade(panel.sessions, [mirror(panel, item, spent(item))], 0.0)  # No SPY in this data.
        self.assertEqual(ledger.capital, 0.0)
        self.assertTrue(np.isnan(ledger.yearly_return))
        self.assertTrue((ledger.path == 0.0).all())


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

    def test_a_run_saves_the_final_choice_for_2026_without_evaluating_it(self):
        result = run_backtest(config(self.root), {"flat": small_models})
        self.assertEqual(set(result.final_models), {"flat"})
        self.assertEqual(result.final_models["flat"].picks, 2)
        choice = result.selection["models"]["flat"]
        self.assertEqual((choice["picks"], choice["horizon_days"], choice["bars_lookback"]), (2, 2, 3))
        self.assertEqual(choice["order_value"], 1000.0)
        self.assertEqual(result.selection["training_end"], "2025-03-31")
        folder = write_run(result, "test", root=self.root)
        selection = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
        self.assertFalse(selection["final_evaluated"])
        self.assertEqual(selection["saved_models"], {"flat": "model/flat.pkl"})
        self.assertEqual(pickle.loads((folder / "model/flat.pkl").read_bytes()).picks, 2)
        for table in ("scores", "folds", "orders", "profit", "yearly", "metrics"):
            self.assertTrue((folder / "tables" / f"{table}.csv").is_file(), table)
        yearly = yearly_table(result).set_index("strategy")
        self.assertEqual(list(yearly.index), ["flat", "account", "spy_hold", "spy_mirror", "universe_ew"])
        self.assertEqual(list(yearly.columns), ["2025", "mean"])
        self.assertAlmostEqual(yearly.loc["account", "2025"], yearly.loc["flat", "2025"])  # One model: the same.


if __name__ == "__main__":
    unittest.main()
