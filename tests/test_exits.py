"""Exit strategies: the rules (exits.py), the account selling early, the backtest tuning them with each model's
settings, and what the selection and the API say about them."""
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock
import numpy as np
import pandas as pd
from stock_recommender import features, news_features
from stock_recommender.backtest.account import book, mirror, trade
from stock_recommender.backtest.metrics import signal_summary
from stock_recommender.backtest.protocol import Period, schedule, walk_forward
from stock_recommender.backtest.run import BacktestConfig, run_backtest
from stock_recommender.ensemble import Ensemble
from stock_recommender.exits import EXIT_FIELDS, ExitRule, daily_volatility, excursions, exit_points, exit_rule
from stock_recommender.models import RandomPriceModel, placeholder_models
from stock_recommender.notify import exit_text
from stock_recommender.schemas import ExitStrategy, Recommendation, RecommendationList
from stock_recommender.served_models import load_selection
from .backtest_fixtures import SESSIONS, config, random_walk, small_models, write_bars, write_news, write_splits
from .test_backtest_cv import THREE_YEARS, LedgerCase, Oracle
from .test_served_models import edit_selection, make_run


class RuleTest(unittest.TestCase):
    def test_a_rule_checks_its_settings_and_says_itself_in_words(self):
        self.assertEqual(ExitRule(5).describe(), "hold 5 trading days")
        self.assertFalse(ExitRule(5).early)
        self.assertEqual(ExitRule(10, 0.08, 0.15, 3.0).describe(),
                         "hold up to 10 trading days; sell at a close 8% below the entry price (stop loss), 15% above "
                         "it (take profit) or with a daily move back within its daily volatility after one of at least 3 "
                         "times it (volatility settled)")
        self.assertEqual(ExitRule(20, stop_loss=0.08).describe(),
                         "hold up to 20 trading days; sell at a close 8% below the entry price (stop loss)")
        for wrong in ({"hold_days": 0}, {"hold_days": 2.5}, {"hold_days": True}, {"hold_days": 5, "stop_loss": 1.2},
                      {"hold_days": 5, "take_profit": 0.0}, {"hold_days": 5, "volatility_exit": 1.0}):
            with self.assertRaises(ValueError, msg=wrong):
                ExitRule(**wrong)

    def test_every_bundle_has_a_rule_old_pickles_and_ensembles_too(self):
        self.assertEqual(exit_rule(placeholder_models()), ExitRule(5))  # No setting: hold for the horizon.
        self.assertEqual(exit_rule(replace(placeholder_models(), hold_days=10, stop_loss=0.08)), ExitRule(10, 0.08))
        old = replace(placeholder_models(), horizon_days=3)
        for name in EXIT_FIELDS:  # As pickled before the exit settings existed.
            object.__delattr__(old, name)
        self.assertEqual(exit_rule(pickle.loads(pickle.dumps(old))), ExitRule(3))
        self.assertEqual(exit_rule(Ensemble(((placeholder_models(), 1.0),))), ExitRule(5))

    def test_each_rule_sells_at_the_open_after_the_close_it_fires(self):
        nan = np.nan
        returns = np.array([[0.00, -0.05, -0.09, -0.20, -0.30],  # The stop fires at close 2: sold at open 3.
                            [0.05, 0.16, 0.00, 0.00, 0.00],  # The target fires at close 1.
                            [0.00, 0.03, 0.02, 0.02, 0.02],  # The volatility settles at close 2.
                            [0.00, 0.00, 0.00, 0.00, -0.50],  # Only the last close: the scheduled exit stands.
                            [nan, nan, nan, nan, nan],  # Never filled.
                            [0.00, -0.09, 0.00, 0.00, 0.00],  # Stop and volatility at close 1: the stop wins.
                            [0.00, 0.03, 0.02, 0.02, 0.02]])  # No volatility measured: that rule is off.
        moves = np.array([[0.00, -0.05, -0.04, -0.12, -0.12],
                          [0.05, 0.10, -0.14, 0.00, 0.00],
                          [0.001, 0.05, 0.004, 0.00, 0.00],
                          [0.00, 0.00, 0.00, 0.00, -0.50],
                          [nan, nan, nan, nan, nan],
                          [0.05, 0.005, 0.00, 0.00, 0.00],
                          [0.001, 0.05, 0.004, 0.00, 0.00]])
        volatility = np.array([0.01, 0.01, 0.01, 0.01, nan, 0.01, 0.0])
        held, why = exit_points(ExitRule(5, 0.08, 0.15, 3.0), returns, moves, volatility)
        self.assertEqual(list(held), [3, 2, 3, 5, 5, 2, 5])
        self.assertEqual(list(why), ["stop_loss", "take_profit", "volatility", "", "", "stop_loss", ""])
        held, why = exit_points(ExitRule(5, take_profit=0.15), returns, moves, volatility)
        self.assertEqual((list(held), set(why[held == 5])), ([5, 2, 5, 5, 5, 5, 5], {""}))

    def test_excursions_are_the_best_and_worst_closes_while_held(self):
        returns = np.array([[0.02, -0.05, 0.10, -0.20, 0.30], [np.nan] * 5, [-0.01, -0.02, -0.03, -0.04, -0.05]])
        mfe, mae = excursions(returns, np.array([3, 5, 5]))
        np.testing.assert_allclose(mfe, [0.10, np.nan, 0.0])  # Closes 0-2 only for the first lot.
        np.testing.assert_allclose(mae, [-0.05, np.nan, -0.05])

    def test_the_daily_volatility_is_the_spread_of_the_last_20_moves(self):
        days = pd.bdate_range("2025-01-01", periods=25)
        rng = np.random.default_rng(1)
        closes = pd.DataFrame({"A": 50 * np.exp(np.cumsum(rng.normal(0, 0.02, 25))),
                               "B": [np.nan] * 15 + [20.0] * 10}, index=days)
        bars = closes.rename_axis("date").reset_index().melt(id_vars="date", var_name="ticker",
                                                             value_name="close").dropna()
        measured = daily_volatility(bars)
        self.assertAlmostEqual(measured["A"], closes.A.iloc[-21:].pct_change().iloc[1:].std(ddof=1))
        self.assertTrue(np.isnan(measured["B"]))  # Nine moves are not enough.


class AccountExitTest(LedgerCase):
    def setUp(self):
        super().setUp()
        self.sessions = SESSIONS[:40]

    def test_a_stop_loss_sells_at_the_close_where_it_fires_and_frees_the_money(self):
        panel = self.panel({"A": [100.0, 100.0, 100.0, 95.0, 90.0, 85.0, 80.0, 75.0] + [75.0] * 32})
        lists = [self.listed(["A"], [100])]
        held = trade(panel.sessions, [book(panel, "m", [Period(1, 5)], lists, 1000.0)], 0.0)
        self.assertAlmostEqual(held.profit.iloc[-1], 1000 * (80 / 100 - 1))  # Bought at 1's close, sold at 6's.
        item = book(panel, "m", [Period(1, 5)], lists, 1000.0, ExitRule(5, stop_loss=0.08))
        ledger = trade(panel.sessions, [item], 0.0, record=True)
        order = ledger.orders.iloc[0]
        # The close of session 4 is 10% below the entry: sold right there, at 90.
        self.assertEqual((order.exit_date, order.exit_rule), (self.sessions[4], "stop_loss"))
        self.assertAlmostEqual(ledger.profit.iloc[-1], 1000 * (90 / 100 - 1))
        self.assertEqual(list(ledger.invested), [1000.0, 1000.0, 1000.0, 0.0, 0.0, 0.0])
        self.assertEqual((order.mfe, round(order.mae, 6)), (0.0, -0.1))
        with self.assertRaisesRegex(ValueError, "holds 10 days"):
            book(panel, "m", [Period(1, 5)], lists, 1000.0, ExitRule(10, stop_loss=0.08))

    def test_a_take_profit_sells_early_and_spy_mirrors_the_same_dates(self):
        panel = self.panel({"A": [100.0, 100.0, 100.0, 105.0, 110.0, 120.0, 130.0, 140.0] + [140.0] * 32,
                            "SPY": [400.0] * 3 + [404.0, 408.0, 412.0, 416.0, 420.0] + [420.0] * 32})
        item = book(panel, "m", [Period(1, 5)], [self.listed(["A"], [100])], 1000.0, ExitRule(5, take_profit=0.15))
        ledger = trade(panel.sessions, [item], 0.0, record=True)
        order = ledger.orders.iloc[0]
        self.assertEqual((order.exit_date, order.exit_rule), (self.sessions[5], "take_profit"))  # Close 5: +20%.
        self.assertAlmostEqual(ledger.profit.iloc[-1], 1000 * (120 / 100 - 1))
        self.assertAlmostEqual(order.mfe, 0.2)
        (paid,) = ledger.spent
        mirrored = trade(panel.sessions, [mirror(panel, item, paid)], 0.0, record=True)
        self.assertEqual(mirrored.orders.exit_date.iloc[0], self.sessions[5])
        self.assertAlmostEqual(mirrored.profit.iloc[-1], 1000 * (412 / 400 - 1))

    def test_the_volatility_rule_waits_for_the_move_to_settle(self):
        calm = [100.0 if day % 2 == 0 else 101.0 for day in range(25)]  # Daily moves of about 1%.
        panel = self.panel({"A": calm + [100.0, 106.0, 106.5, 107.0] + [107.0] * 11})
        item = book(panel, "m", [Period(24, 6)], [self.listed(["A"], [100])], 1000.0,
                    ExitRule(6, volatility_exit=3.0))
        ledger = trade(panel.sessions, [item], 0.0, record=True)
        order = ledger.orders.iloc[0]
        # +6% at the close of session 26, six daily volatilities; +0.5% at 27 is back within one: sold there.
        self.assertEqual((order.exit_date, order.exit_rule), (self.sessions[27], "volatility"))
        self.assertAlmostEqual(ledger.profit.iloc[-1], 1000 * (106.5 / 100 - 1))


class BacktestExitTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.prices = random_walk(30, THREE_YEARS, seed=2)
        write_bars(self.root / "bars", self.prices["open"], self.prices["close"])
        write_splits(self.root / "splits.parquet", [])
        write_news(self.root / "news", [("2024-06-03T15:00:00Z", ["T000"])])

    def tearDown(self):
        self.directory.cleanup()

    def test_exit_settings_share_one_fit_and_each_trades_its_own_holding(self):
        fits = []

        class Counting(RandomPriceModel):
            def fit(self, data):
                fits.append(data.cutoff)

        factories = {"counted": lambda: small_models(price=Counting(0))}
        grids = {"counted": {"hold_days": (2, 4), "stop_loss": (None, 0.5)}}
        result = run_backtest(config(self.root, models=("counted",), test_years=(2024, 2025)), factories,
                              grids=grids)
        self.assertEqual(len(fits), 3)  # One fit per test year for the four settings, and one final fit.
        scores = result.scores.set_index(["candidate", "year"])
        self.assertEqual(len(scores), 4 * 2)
        two, four = 'counted{"hold_days": 2, "stop_loss": null}', 'counted{"hold_days": 4, "stop_loss": null}'
        # A decision every session of 2024's 262, the last ones those whose holding still fits in the year.
        self.assertEqual((scores.periods[two, 2024], scores.periods[four, 2024]), (261 - 2 + 1, 261 - 4 + 1))
        self.assertEqual((scores.capital[two, 2024], scores.capital[four, 2024]), (10_000.0, 10_000.0))
        chosen = result.selection["models"]["counted"]
        self.assertEqual(chosen["exit"], {"hold_days": chosen["params"]["hold_days"],
                                          "stop_loss": chosen["params"]["stop_loss"], "take_profit": None,
                                          "volatility_exit": None})
        self.assertEqual(exit_rule(result.final_models["counted"]).as_dict(), chosen["exit"])
        self.assertEqual(chosen["horizon_days"], 2)  # The model still learns 2-session returns.

    def test_every_model_buys_every_session_and_each_lot_leaves_on_its_own_day(self):
        factories = {"wrong": lambda: small_models(price=Oracle(self.prices["close"], 2, good="1999"))}
        grids = {"wrong": {"hold_days": (4,), "stop_loss": (0.01,)}}
        daily = run_backtest(config(self.root, models=("wrong",), test_years=(2024,)), factories, grids=grids)
        positions = daily.positions
        sessions = pd.Index(THREE_YEARS)
        decided = sessions.get_indexer(pd.DatetimeIndex(positions.decision_date.unique()))
        self.assertEqual(len(decided), 261 - 4 + 1)  # Every session of 2024 whose 4-session holding fits in it.
        self.assertTrue((np.diff(np.sort(decided)) == 1).all())
        self.assertEqual(set(positions.groupby("decision_date").size()), {2})  # Its two picks, each day.
        held = sessions.get_indexer(positions.exit_date) - sessions.get_indexer(positions.entry_date)
        # Each lot leaves on its own day: stopped after 1 to 3 sessions, or at the end of its 4.
        self.assertTrue(set(held) <= {1, 2, 3, 4} and len(set(held)) > 1)
        orders = daily.orders[daily.orders.fold == 2024]
        self.assertGreater(orders.groupby("exit_date").size().size, 200)  # Lots leave on most days.
        back = run_backtest(config(self.root, models=("wrong",), test_years=(2024,), decide_every=None), factories,
                            grids=grids)
        self.assertEqual(back.positions.decision_date.nunique(), (261 - 4) // 4 + 1)  # Back to back: every 4.

    def test_a_stop_loss_cuts_losing_picks_short_in_the_backtest(self):
        factories = {"wrong": lambda: small_models(price=Oracle(self.prices["close"], 2, good="1999"))}
        grids = {"wrong": {"hold_days": (4,), "stop_loss": (0.01,)}}  # It always picks the worst stocks.
        result = run_backtest(config(self.root, models=("wrong",), test_years=(2024,)), factories, grids=grids)
        positions = result.positions[result.positions.reason == "filled"]
        sessions = pd.Index(THREE_YEARS)
        held = sessions.get_indexer(positions.exit_date) - sessions.get_indexer(positions.entry_date)
        stopped = (positions.exit_reason == "stop_loss").to_numpy()
        self.assertGreater(stopped.mean(), 0.3)
        self.assertTrue((held[stopped] < 4).all() and (held[~stopped] == 4).all())
        self.assertTrue((positions.mae[stopped] <= -0.01).all())
        self.assertTrue((positions.mfe >= 0).all() and (positions.mae <= 0).all())


class DailyDecisionTest(unittest.TestCase):
    def test_the_schedule_decides_every_session_unless_told_otherwise(self):
        sessions = pd.bdate_range("2023-12-01", "2025-01-31")
        (fold,) = walk_forward(sessions, (2024,))
        daily, back = schedule(sessions, [fold], 3, every=1)[2024], schedule(sessions, [fold], 3)[2024]
        first, last = sessions.get_loc(pd.Timestamp("2024-01-01")), sessions.get_loc(pd.Timestamp("2024-12-31"))
        self.assertEqual([period.entry for period in daily], list(range(first, last - 3 + 1)))
        self.assertEqual(daily[-1].exit, last)  # Holdings stay inside the year.
        self.assertEqual([period.entry for period in back[:3]], [first, first + 3, first + 6])
        self.assertEqual(BacktestConfig().decide_every, 1)
        for wrong in (0, 1.5, True):
            with self.assertRaises(ValueError):
                BacktestConfig(decide_every=wrong)
        with self.assertRaises(ValueError):
            schedule(sessions, [fold], 3, every=0)

    def test_overlapping_decisions_count_once_per_holding_in_the_t_statistics(self):
        signals = pd.DataFrame({"ic": [0.1, 0.2, 0.0, 0.3] * 5, "spread": [0.01, -0.02, 0.03, 0.0] * 5})
        alone = signal_summary(signals)
        daily = signal_summary(signals.assign(holding=5, every=1))  # Each shares 4 sessions with its neighbours.
        back = signal_summary(signals.assign(holding=5, every=5))
        self.assertAlmostEqual(daily["ic_t"], alone["ic_t"] / np.sqrt(5))
        self.assertAlmostEqual(daily["spread_t"], alone["spread_t"] / np.sqrt(5))
        self.assertAlmostEqual(back["ic_t"], alone["ic_t"])
        self.assertEqual((daily["ic_days"], daily["ic_mean"]), (20, alone["ic_mean"]))

    def test_a_snapshots_features_are_computed_once(self):
        prices = random_walk(5, SESSIONS[:70], seed=3)
        bars = (prices["open"].rename_axis("date").reset_index().melt(id_vars="date", var_name="ticker",
                                                                       value_name="open")
                .merge(prices["close"].rename_axis("date").reset_index().melt(id_vars="date", var_name="ticker",
                                                                               value_name="close")))
        bars = bars.assign(high=bars[["open", "close"]].max(axis=1), low=bars[["open", "close"]].min(axis=1),
                           volume=1e6)
        with mock.patch.object(features, "compute", wraps=features.compute) as computed:
            first = features.decision_features(bars)
            first.iloc[0, 0] = 99.0  # A caller's copy: what is kept does not change.
            again = features.decision_features(bars)
            other = features.decision_features(bars.copy())
        self.assertEqual(computed.call_count, 2)  # The same table once, and another table again.
        pd.testing.assert_frame_equal(again, other)
        self.assertNotEqual(again.iloc[0, 0], 99.0)
        news = pd.DataFrame({"published_utc": pd.to_datetime(["2025-01-02T15:00:00Z", "2025-01-03T15:00:00Z"]),
                             "tickers": [["T000"], ["T001", "T000"]], "llm_scores": [None, None]})
        cutoff = pd.Timestamp("2025-01-03T21:00:00Z")
        with mock.patch.object(news_features, "summarize", wraps=news_features.summarize) as summarized:
            one = news_features.decision_news_features(news, ["T000", "T001"], cutoff)
            two = news_features.decision_news_features(news, ["T001", "T002"], cutoff)
            news_features.decision_news_features(news, ["T000"], cutoff - pd.Timedelta(days=1))
        self.assertEqual(summarized.call_count, 2)  # Another cutoff summarizes again.
        self.assertEqual((one.news_count_7d.tolist(), two.news_count_7d.tolist()), ([2.0, 1.0], [1.0, 0.0]))


class ServingExitTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_the_selection_and_the_saved_model_must_agree_on_the_exit(self):
        model = replace(small_models(), hold_days=4, stop_loss=0.08)
        folder = make_run(self.root, {"flat": model})  # Like a run saved before exit settings: none recorded.
        self.assertEqual(load_selection(str(folder)).errors, [])
        edit_selection(folder, lambda selection: selection["models"]["flat"].update(exit=ExitRule(4, 0.08).as_dict()))
        self.assertEqual(exit_rule(load_selection(str(folder)).loaded[0].models), ExitRule(4, 0.08))
        edit_selection(folder, lambda selection: selection["models"]["flat"].update(exit=ExitRule(5).as_dict()))
        (error,) = load_selection(str(folder)).errors
        self.assertIn("exit strategy", error)

    def test_the_lists_say_when_to_sell(self):
        def strategy(rule: ExitRule) -> ExitStrategy:
            return ExitStrategy(**rule.as_dict(), summary=rule.describe())

        now = datetime(2026, 10, 16, 20, tzinfo=timezone.utc)
        body = RecommendationList(as_of=date(2026, 10, 16), news_cutoff=now, news_through=None, generated_at=now,
                                  horizon_days=10, exits={"a": strategy(ExitRule(10, 0.08)), "b": strategy(ExitRule(5))},
                                  model_version="a + b", universe_size=2, buy=[], sell=[])
        self.assertEqual(exit_text(body), "a: hold up to 10 trading days; sell at a close 8% below the entry price "
                                          "(stop loss); b: hold 5 trading days")
        item = Recommendation(rank=1, ticker="A", score=1.0, expected_return=0.0, news_sentiment=0.0, news_count=0,
                              close=10.0, dollar_volume=1e6, daily_volatility=float("nan"))
        self.assertIsNone(item.daily_volatility)  # JSON has no NaN.


if __name__ == "__main__":
    unittest.main()
