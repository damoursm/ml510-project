"""The nested walk-forward choice, the account that buys every recommendation for $1,000, its references, and the
final selection."""
import json
from pathlib import Path
import pickle
import tempfile
import unittest
import numpy as np
import pandas as pd
from stock_recommender.backtest.account import basket, book, hold, mirror, trade
from stock_recommender.backtest.panel import Panel
from stock_recommender.backtest.protocol import Period, news_window
from stock_recommender.backtest.registry import GRIDS, UNTUNED
from stock_recommender.backtest.report import write_run, yearly_table
from stock_recommender.backtest.run import Candidate, choose_combination, load_data, run_backtest, select
from stock_recommender.backtest.window import LeakageError, WindowedStore
from stock_recommender.config import Settings
from stock_recommender.models import RandomPriceModel
from stock_recommender.recommend import news_cutoff, score_day
from .backtest_fixtures import (SESSIONS, config, random_walk, small_models, write_bars, write_dividends,
                                write_news, write_splits)

THREE_YEARS = pd.bdate_range("2023-11-01", "2025-03-31")  # 2023 trains, then the test years 2024 and 2025.


class Oracle:
    """Test-only: ranks by the realized return of the holding ahead, from the decision's close (where the
    backtest buys) to the close `holding` sessions later; the wrong way round outside `good` years."""

    def __init__(self, closes: pd.DataFrame, holding: int, good=None):
        self.closes, self.holding, self.good = closes, holding, None if good is None else int(good)

    def predict(self, bars, tickers):
        day = bars.date.max()
        t = self.closes.index.get_loc(day)
        realized = self.closes.iloc[t + self.holding] / self.closes.iloc[t] - 1
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
        self.closes = self.prices["close"]

    def tearDown(self):
        self.directory.cleanup()

    def test_a_test_year_is_chosen_on_earlier_years_only(self):
        factories = {"oracle": lambda good="2025": small_models(price=Oracle(self.closes, 2, good))}
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

    def test_the_account_buys_the_chosen_models_lists_from_its_cash(self):
        factories = {"oracle": lambda: small_models(price=Oracle(self.closes, 2)),
                     "wrong": lambda: small_models(price=Oracle(self.closes, 2, good="1999"))}
        result = run_backtest(config(self.root, models=("oracle", "wrong"), test_years=(2024, 2025)), factories)
        scoped = result.metrics.set_index(["strategy", "scope"])
        folds = result.folds.set_index("fold")
        # 2024 has no earlier year, so the account trades every model, from one $10,000. Each buys two $100 slices
        # a session and holds them two sessions, $800 at most, so the cash never runs short: without costs the
        # account is its models added up, and on one capital so are their returns.
        self.assertEqual(json.loads(folds.loc[2024, "combination"]), ["oracle", "wrong"])
        self.assertAlmostEqual(scoped.profit["account", "2024"],
                               scoped.profit["oracle", "2024"] + scoped.profit["wrong", "2024"])
        self.assertAlmostEqual(scoped.yearly_return["account", "2024"],
                               scoped.yearly_return["oracle", "2024"] + scoped.yearly_return["wrong", "2024"])
        self.assertEqual(scoped.capital["account", "2024"], 10_000.0)
        # 2025 trades the set whose account did best in 2024: the oracle alone, since the wrong model lost. This
        # data has no SPY, so the rule that a model must beat SPY held cannot apply, and says so.
        self.assertEqual(json.loads(folds.loc[2025, "combination"]), ["oracle"])
        self.assertEqual({name: standing["beats_spy"] for name, standing
                          in json.loads(folds.loc[2025, "against_spy"]).items()}, {"oracle": None, "wrong": None})
        self.assertAlmostEqual(scoped.profit["account", "2025"], scoped.profit["oracle", "2025"])
        self.assertGreater(scoped.yearly_return["oracle", "2025"], 0.0)
        self.assertLess(scoped.yearly_return["wrong", "2025"], 0.0)
        filled = result.orders[result.orders.reason == "filled"]
        self.assertTrue(len(filled) and np.allclose(filled.cost, 100.0))  # $100 slices, never scaled.
        self.assertEqual(set(result.orders[result.orders.fold == 2025].model), {"oracle"})
        self.assertEqual(result.profit.account.iloc[-1], scoped.profit["account", "all"])
        self.assertAlmostEqual(folds.loc[2025, "account_return"], scoped.yearly_return["account", "2025"])
        self.assertFalse(folds.tuned.any())  # No grid: nothing is chosen, the models run as they are.
        self.assertEqual(result.selection["nested_estimate"]["untuned_years"], [2024, 2025])
        self.assertEqual(result.selection["nested_estimate"]["combinations"],
                         {"2024": ["oracle", "wrong"], "2025": ["oracle"]})
        # The final choice, on 2024-2025: the oracle alone, the only model fit and kept for the API.
        self.assertEqual(result.selection["combination"]["models"], ["oracle"])
        self.assertEqual(set(result.selection["models"]), {"oracle"})
        self.assertEqual(set(result.final_models), {"oracle"})
        self.assertEqual(result.selection["models"]["oracle"]["order_value"], 100.0)
        self.assertEqual(result.selection["capital"], 10_000.0)
        self.assertFalse(result.selection["final_evaluated"])
        sets = result.combinations
        self.assertEqual(len(sets), 3 * 2)  # Three sets of two models, in each of two years.
        alone = sets[(sets.combination == "oracle") & (sets.year == 2025)].iloc[0]
        self.assertAlmostEqual(alone.yearly_return, scoped.yearly_return["oracle", "2025"])  # A set of one.

    def test_only_models_that_beat_holding_spy_enter_the_account(self):
        def run(spy_end: float):  # SPY's open and close glide from 400 to spy_end over the data.
            spy = 400.0 * np.exp(np.linspace(0.0, np.log(spy_end / 400.0), len(THREE_YEARS)))
            opens = self.prices["open"].assign(SPY=spy)
            write_bars(self.root / "bars", opens, self.prices["close"].assign(SPY=spy))
            factories = {"random": lambda: small_models(price=RandomPriceModel(0))}
            return run_backtest(config(self.root, models=("random",), test_years=(2024, 2025)), factories)

        falling = run(80.0)  # SPY loses about two thirds of its value in 2024: the random model beats it.
        standing = json.loads(falling.folds.set_index("fold").loc[2025, "against_spy"])["random"]
        self.assertTrue(standing["beats_spy"])
        self.assertLess(standing["spy_hold_mean_return"], -0.5)
        self.assertEqual(falling.selection["nested_estimate"]["combinations"], {"2024": ["random"], "2025": ["random"]})
        self.assertEqual(set(falling.final_models), {"random"})

        rising = run(2000.0)  # SPY triples in 2024: the same model falls short, so the account stands aside.
        folds = rising.folds.set_index("fold")
        self.assertFalse(json.loads(folds.loc[2025, "against_spy"])["random"]["beats_spy"])
        self.assertEqual(json.loads(folds.loc[2025, "combination"]), [])
        scoped = rising.metrics.set_index(["strategy", "scope"])
        self.assertEqual((scoped.capital["account", "2025"], scoped.profit["account", "2025"]), (10_000.0, 0.0))
        self.assertEqual(scoped.yearly_return["account", "2025"], 0.0)  # Money left in cash makes nothing.
        self.assertFalse((rising.orders.fold == 2025).any())
        self.assertGreater(scoped.yearly_return["spy_hold", "2025"], 0.1)  # The references still show the year.
        alone = rising.positions[rising.positions.fold == 2025]
        self.assertTrue((alone.reason == "filled").any())  # The model still trades alone, as reported.
        self.assertAlmostEqual(scoped.mean_yearly_return["account", "all"],
                               scoped.yearly_return["account", "2024"] / 2)
        # Nothing beat SPY on 2024-2025 either: nothing is fit, and the API has nothing to serve.
        self.assertEqual(rising.selection["combination"]["models"], [])
        self.assertEqual((rising.selection["models"], rising.final_models), ({}, {}))
        self.assertFalse(rising.selection["combination"]["strategies"]["random"]["in_combination"])
        folder = write_run(rising, "test", root=self.root)
        self.assertEqual(json.loads((folder / "selection.json").read_text(encoding="utf-8"))["saved_models"], {})
        self.assertTrue((folder / "tables" / "combinations.csv").is_file())

    def test_the_combination_is_the_best_set_of_models_that_beat_spy_on_earlier_years(self):
        years, names = [2024, 2025], ["a", "b", "c"]
        own = {("a",): (0.30, 0.10), ("b",): (0.20, 0.10), ("c",): (0.05, 0.05), ("a", "b"): (0.25, 0.10),
               ("a", "c"): (0.20, 0.10), ("b", "c"): (0.10, 0.10), ("a", "b", "c"): (0.20, 0.10)}

        def table(returns, spy=(0.10, 0.10)) -> pd.DataFrame:
            return pd.DataFrame([{"members": json.dumps(list(members)), "year": year, "yearly_return": value[i],
                                  "spy_hold_return": spy[i]}
                                 for members, value in returns.items() for i, year in enumerate(years)])

        self.assertEqual(choose_combination(table(own), names, [], before=2024)[0], ("a", "b", "c"))  # The default.
        members, mean, standing = choose_combination(table(own), names, years)
        self.assertEqual(members, ("a",))  # a alone (0.20) beats a+b (0.175) and b (0.15).
        self.assertAlmostEqual(mean, 0.20)
        self.assertEqual({name: entry["qualifies"] for name, entry in standing.items()},
                         {"a": True, "b": True, "c": False})  # c's 0.05 is below SPY's 0.10: no set holds it.
        self.assertAlmostEqual(standing["c"]["spy_hold_mean_return"], 0.10)
        with self.assertRaises(LeakageError):
            choose_combination(table(own), names, years, before=2025)
        tied = {**own, ("a",): (0.3, 0.3), ("b",): (0.3, 0.3), ("a", "b"): (0.3, 0.3)}
        self.assertEqual(choose_combination(table(tied), names, years)[0], ("a", "b"))  # Ties keep the larger set.
        self.assertEqual(choose_combination(table(own, spy=(0.9, 0.9)), names, years)[0], ())  # None beats SPY.
        unknown = choose_combination(table(own, spy=(np.nan, 0.10)), names, years)  # No SPY: the rule cannot apply.
        self.assertEqual((unknown[0], unknown[2]["c"]["beats_spy"], unknown[2]["c"]["qualifies"]), (("a",), None, True))
        gap = choose_combination(table({**own, ("a",): (np.nan, 0.9)}), names, years)  # A year without a return.
        self.assertEqual((gap[0], gap[2]["a"]["qualifies"]), (("b",), False))

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
    def test_a_slice_is_bought_at_the_recommendation_close_from_the_starting_cash(self):
        closes = [100.0, 100.0, 110.0, 121.0, 133.1] + [133.1] * 15
        panel = self.panel({"A": closes})
        item = book(panel, "m", [Period(1, 2)], [self.listed(["A"], [100])], 100.0)
        ledger = trade(panel.sessions, [item], 0.0, record=True)
        order = ledger.orders.iloc[0]
        self.assertEqual((order.entry_date, order.exit_date), (self.sessions[1], self.sessions[3]))  # At 16:00.
        self.assertAlmostEqual(order.cost, 100.0)  # At session 1's close, 100.
        self.assertAlmostEqual(order.shares, 1.0)
        self.assertEqual(order.reason, "filled")
        self.assertEqual(list(ledger.profit.round(6)), [0.0, 10.0, 21.0])  # Sold at session 3's close, 121.
        self.assertEqual(list(ledger.invested), [100.0, 100.0, 0.0])
        self.assertEqual(ledger.capital, 10_000.0)
        self.assertAlmostEqual(ledger.yearly_return, 10_021.0 / 10_000 - 1)  # Last equity over the cash, minus 1.

    def test_whole_shares_round_down_and_skip_a_stock_pricier_than_the_order(self):
        panel = self.panel({"A": [100.0] * 20, "B": [300.0] * 20})
        item = book(panel, "m", [Period(1, 2)], [self.listed(["A", "B"], [100, 300])], 250.0)
        ledger = trade(panel.sessions, [item], 0.0, whole_shares=True, record=True)
        rows = ledger.orders.set_index("ticker")
        self.assertEqual((rows.shares["A"], rows.shares["B"]), (2.0, 0.0))  # floor(250 / 100), and 300 > 250.
        self.assertEqual(rows.reason.to_dict(), {"A": "filled", "B": "price_above_order"})
        self.assertEqual(ledger.invested.max(), 200.0)  # What the orders cost, not what they were worth.
        self.assertAlmostEqual(ledger.profit.iloc[-1], 0.0)

    def test_the_cash_limits_the_orders_and_sales_pay_for_later_ones(self):
        panel = self.panel({"A": [100.0] * 20, "B": [100.0] * 20, "C": [100.0] * 20, "D": [100.0] * 20})
        first = book(panel, "first", [Period(1, 2), Period(3, 2)], [self.listed(["A", "B"], [100, 100])] * 2, 100.0)
        second = book(panel, "second", [Period(1, 2), Period(3, 2)], [self.listed(["C", "D"], [100, 100])] * 2,
                      100.0)
        ledger = trade(panel.sessions, [first, second], 0.0, record=True, capital=250.0)
        orders = ledger.orders.set_index(["model", "ticker", "decision_date"]).reason
        day, later = self.sessions[1], self.sessions[3]
        # Each list's best stock first, the models in turn: A, then C; then B and D find $50 left.
        self.assertEqual([orders["first", "A", day], orders["second", "C", day], orders["first", "B", day],
                          orders["second", "D", day]], ["filled", "filled", "no_cash", "no_cash"])
        # At session 3's close A and C are sold first, so their $200 buys the next lists' best again.
        self.assertEqual([orders["first", "A", later], orders["second", "C", later]], ["filled", "filled"])
        self.assertEqual(ledger.filled, 4)
        self.assertEqual(list(ledger.invested), [200.0, 200.0, 200.0, 200.0, 0.0])
        self.assertEqual([list(spent) for spent in ledger.spent], [[100.0, 0.0, 100.0, 0.0], [100.0, 0.0, 100.0, 0.0]])

    def test_each_model_trades_on_its_own_horizon(self):
        panel = self.panel({"A": [100.0] * 20, "B": [100.0] * 20})
        short = book(panel, "short", [Period(1, 2), Period(3, 2)], [self.listed(["A"], [100])] * 2, 250.0)
        long = book(panel, "long", [Period(1, 3)], [self.listed(["B"], [100])], 250.0)
        ledger = trade(panel.sessions, [short, long], 0.0, record=True)
        exits = ledger.orders.groupby("model").exit_date.apply(list)
        self.assertEqual(exits["short"], [self.sessions[3], self.sessions[5]])
        self.assertEqual(exits["long"], [self.sessions[4]])
        self.assertEqual((ledger.profit.index[0], ledger.profit.index[-1]), (self.sessions[1], self.sessions[5]))

    def test_a_split_and_a_dividend_during_a_lot_are_valued_like_forward_returns(self):
        closes = [100.0] * 5 + [50.0] * 3 + [49.5] * 4 + [55.0] * 8  # 2-for-1 at 5; $0.50 ex-dividend at 8.
        panel = self.panel({"A": closes}, splits=[("A", str(self.sessions[5].date()), 1, 2)],
                           dividends=[("A", str(self.sessions[8].date()), 0.5)])
        item = book(panel, "m", [Period(2, 10)], [self.listed(["A"], [100])], 1000.0)
        ledger = trade(panel.sessions, [item], 0.0)
        expected = panel.forward_returns(2, 10, ["A"]).forward_return["A"]
        self.assertAlmostEqual(ledger.profit.iloc[-1], 1000 * expected)
        self.assertGreater(expected, 0.09)  # The split is no loss, and the dividend is income.

    def test_costs_are_netted_when_a_stock_is_sold_and_bought_at_the_same_close(self):
        panel = self.panel({"A": [100.0] * 20})
        item = book(panel, "m", [Period(1, 2), Period(3, 2)], [self.listed(["A"], [100])] * 2, 500.0)
        ledger = trade(panel.sessions, [item], 100.0)  # 1% a side.
        # Buy 500 (cost 5); at the next entry sell 500 and buy 500, which nets to nothing; sell 500 (cost 5).
        self.assertAlmostEqual(ledger.profit.iloc[-1], -10.0)

    def test_a_halted_stock_is_not_entered_and_a_delisted_one_exits_at_its_last_close(self):
        halted = [10.0, 10.0, np.nan] + [10.0] * 17
        delisted = [10.0, 10.0, 10.0, 12.0] + [np.nan] * 16
        panel = self.panel({"H": halted, "D": delisted})
        item = book(panel, "m", [Period(2, 3)], [self.listed(["H", "D"], [10, 10])], 500.0)  # Session 2's close.
        ledger = trade(panel.sessions, [item], 0.0, record=True)
        self.assertEqual(ledger.orders.set_index("ticker").reason.to_dict(), {"H": "no_entry", "D": "filled"})
        self.assertAlmostEqual(ledger.profit.iloc[-1], 500 * 0.2)
        self.assertEqual(ledger.invested.max(), 500.0)

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
        self.assertEqual(ledger.invested.max(), 1000.0)
        self.assertEqual(ledger.traded.iloc[0], 1000.0)
        self.assertEqual(float(ledger.fees.sum()), 0.0)


class ReferenceTest(LedgerCase):
    def test_spy_is_bought_for_every_filled_order_and_held_with_the_capital(self):
        spy = [400.0, 400.0, 404.0, 408.0, 412.0, 416.0] + [416.0] * 14
        panel = self.panel({"A": [100.0] * 20, "H": [10.0, np.nan] + [10.0] * 18, "SPY": spy})
        item = book(panel, "m", [Period(1, 2), Period(3, 2)], [self.listed(["A", "H"], [100, 10])] * 2, 1000.0)
        account = trade(panel.sessions, [item], 0.0)
        (paid,) = account.spent
        self.assertEqual(list(paid), [1000.0, 0.0, 1000.0, 1000.0])  # H does not trade on session 1.
        mirrored = trade(panel.sessions, [mirror(panel, item, paid)], 0.0, record=True)
        self.assertEqual(len(mirrored.orders), 3)  # One SPY lot per filled order, on its dates.
        self.assertAlmostEqual(mirrored.profit.iloc[-1], 1000 * (408 / 400 - 1) + 2000 * (416 / 408 - 1))
        held = trade(panel.sessions, [hold(panel, "SPY", item.start, item.end, account.capital)], 0.0)
        self.assertAlmostEqual(held.profit.iloc[-1], 10_000 * (416 / 400 - 1))  # All the cash, at session 1's close.
        self.assertAlmostEqual(held.yearly_return, 416 / 400 - 1)
        self.assertEqual(list(held.profit.index), list(account.profit.index))

    def test_the_universe_reference_spreads_each_orders_dollars_over_the_universe(self):
        panel = self.panel({"A": [100.0] * 20, "B": [50.0] * 2 + [55.0] * 18, "C": [20.0, np.nan] + [20.0] * 18})
        item = book(panel, "m", [Period(1, 2)], [self.listed(["A"], [100])], 1000.0)
        (paid,) = trade(panel.sessions, [item], 0.0).spent
        spread = basket(panel, item, paid, lambda t: pd.Index(["A", "B", "C"]))
        ledger = trade(panel.sessions, [spread], 0.0, record=True)
        rows = ledger.orders.set_index("ticker")
        self.assertEqual(rows.order_value.to_dict(), {"A": 500.0, "B": 500.0, "C": 0.0})  # C does not trade then.
        self.assertAlmostEqual(ledger.profit.iloc[-1], 500 * (55 / 50 - 1))

    def test_a_reference_that_never_trades_has_no_return(self):
        panel = self.panel({"A": [100.0] * 20})
        item = book(panel, "m", [Period(1, 2)], [self.listed(["A"], [100])], 1000.0)
        (paid,) = trade(panel.sessions, [item], 0.0).spent
        ledger = trade(panel.sessions, [mirror(panel, item, paid)], 0.0)  # No SPY in this data.
        self.assertEqual(ledger.filled, 0)
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
        settings = Settings(min_price=0, min_dollar_volume=0)
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
        self.assertEqual(choice["order_value"], 100.0)
        self.assertEqual(result.selection["training_end"], "2025-03-31")
        folder = write_run(result, "test", root=self.root)
        selection = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
        self.assertFalse(selection["final_evaluated"])
        self.assertEqual(selection["saved_models"], {"flat": "model/flat.pkl"})
        self.assertEqual(pickle.loads((folder / "model/flat.pkl").read_bytes()).picks, 2)
        for table in ("scores", "folds", "orders", "profit", "yearly", "metrics", "combinations"):
            self.assertTrue((folder / "tables" / f"{table}.csv").is_file(), table)
        yearly = yearly_table(result).set_index("strategy")
        self.assertEqual(list(yearly.index), ["flat", "account", "spy_hold", "spy_mirror", "universe_ew"])
        self.assertEqual(list(yearly.columns), ["2025", "mean"])
        self.assertAlmostEqual(yearly.loc["account", "2025"], yearly.loc["flat", "2025"])  # One model: the same.


if __name__ == "__main__":
    unittest.main()
