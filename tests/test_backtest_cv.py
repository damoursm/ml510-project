"""The nested walk-forward choice, the account that buys the recommendations in $100 slices from $10,000, the
models it trades (those whose trades beat SPY on the same dates), its references, and the final selection."""
from dataclasses import replace
import json
from pathlib import Path
import pickle
import tempfile
import unittest
import numpy as np
import pandas as pd
from stock_recommender.backtest.account import basket, book, contributions, hold, mirror, sweep_growth, trade
from stock_recommender.backtest.panel import Panel
from stock_recommender.backtest.protocol import Period, news_window
from stock_recommender.backtest.registry import GRIDS, UNTUNED, picking
from stock_recommender.backtest.report import format_summary, write_run, yearly_table
from stock_recommender.backtest.run import Candidate, choose_combination, load_data, run_backtest, select
from stock_recommender.backtest.window import LeakageError, WindowedStore
from stock_recommender.config import Settings
from stock_recommender.models import RandomPriceModel
from stock_recommender.recommend import news_cutoff, pick_count, score_day
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
        # 2025's account, the candidate right in 2024, evaluated in 2024: 2024 traded the default, so its lists are
        # priced again; alone in the account, it earns what it earned alone.
        self.assertEqual(json.loads(folds.loc[2024, "account_validation_returns"]), {})  # No earlier year.
        self.assertAlmostEqual(json.loads(folds.loc[2025, "account_validation_returns"])["2024"],
                               scores['oracle{"good": "2024"}', 2024])
        final = result.selection["models"]["oracle"]["candidate"]
        for year, value in result.selection["combination"]["account_return_by_year"].items():
            self.assertAlmostEqual(value, scores[final, int(year)])
        returns = result.metrics.set_index(["strategy", "scope"]).yearly_return
        self.assertLess(returns["oracle", "2025"], 0.0)  # The choice fails, as it would have live.
        self.assertEqual(result.guards["selection_years"], {"2024": [], "2025": [2024]})
        self.assertEqual(result.selection["chosen_on_years"], [2024, 2025])

    def test_the_account_buys_the_chosen_models_lists_from_its_cash(self):
        closes = self.closes.assign(SPY=400.0)  # SPY stays flat: bought on any dates, it earns nothing.
        write_bars(self.root / "bars", self.prices["open"].assign(SPY=400.0), closes)
        factories = {"oracle": lambda: small_models(price=Oracle(closes, 2)),
                     "wrong": lambda: small_models(price=Oracle(closes, 2, good="1999"))}
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
        # 2025 trades the models whose trades beat SPY in 2024: the oracle's did; the wrong model lost money, which
        # SPY bought with the same dollars on the same dates did not.
        self.assertEqual(json.loads(folds.loc[2025, "combination"]), ["oracle"])
        self.assertEqual({name: standing["beats_spy"] for name, standing
                          in json.loads(folds.loc[2025, "against_spy"]).items()}, {"oracle": True, "wrong": False})
        self.assertAlmostEqual(scoped.profit["account", "2025"], scoped.profit["oracle", "2025"])
        self.assertAlmostEqual(scoped.return_per_day["account", "2025"], scoped.return_per_day["oracle", "2025"])
        self.assertGreater(scoped.yearly_return["oracle", "2025"], 0.0)
        self.assertLess(scoped.yearly_return["wrong", "2025"], 0.0)
        filled = result.orders[result.orders.reason == "filled"]
        self.assertTrue(len(filled) and np.allclose(filled.cost, 100.0))  # $100 slices, never scaled.
        # 2025's account, the oracle alone, evaluated in 2024, which chose it, beside SPY held that year.
        self.assertAlmostEqual(json.loads(folds.loc[2025, "account_validation_returns"])["2024"],
                               scoped.yearly_return["oracle", "2024"])
        self.assertAlmostEqual(folds.loc[2025, "spy_hold_validation_return"], scoped.yearly_return["spy_hold", "2024"])
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
        against = result.against_spy.set_index(["strategy", "year"])
        self.assertEqual(len(against), 2 * 2)  # Each model in each of two years.
        self.assertAlmostEqual(against.yearly_return["oracle", 2025], scoped.yearly_return["oracle", "2025"])
        self.assertTrue(np.allclose(against.spy_mirror_return, 0.0))  # Flat SPY earns nothing on any dates.
        self.assertTrue((against.days_with_trades > 0).all())
        # Idle cash held in SPY instead: flat and without costs here, it earns what cash does. No BIL in this data.
        idle = result.idle_cash.set_index(["strategy", "idle_cash", "year"]).yearly_return
        self.assertEqual(list(dict.fromkeys(result.idle_cash.idle_cash)), ["cash", "SPY"])
        self.assertAlmostEqual(idle["account", "cash", 2025], scoped.yearly_return["account", "2025"])
        self.assertAlmostEqual(idle["account", "SPY", 2025], idle["account", "cash", 2025])
        self.assertAlmostEqual(idle["wrong", "SPY", 2024], scoped.yearly_return["wrong", "2024"])

    def test_only_models_whose_trades_beat_spy_enter_the_account(self):
        def run(spy_end: float):  # SPY's open and close glide from 400 to spy_end over the data.
            spy = 400.0 * np.exp(np.linspace(0.0, np.log(spy_end / 400.0), len(THREE_YEARS)))
            opens = self.prices["open"].assign(SPY=spy)
            write_bars(self.root / "bars", opens, self.prices["close"].assign(SPY=spy))
            factories = {"random": lambda: small_models(price=RandomPriceModel(0))}
            return run_backtest(config(self.root, models=("random",), test_years=(2024, 2025)), factories)

        falling = run(80.0)  # SPY loses about two thirds of its value in 2024: the random model beats it.
        standing = json.loads(falling.folds.set_index("fold").loc[2025, "against_spy"])["random"]
        self.assertTrue(standing["beats_spy"])
        # SPY bought with the model's $100 slices on its dates loses money; the model's own picks barely move.
        self.assertLess(standing["spy_mirror_mean_return"], -0.02)
        self.assertGreater(standing["mean_yearly_return"], standing["spy_mirror_mean_return"])
        self.assertEqual(falling.selection["nested_estimate"]["combinations"], {"2024": ["random"], "2025": ["random"]})
        self.assertEqual(set(falling.final_models), {"random"})
        swept = falling.idle_cash.set_index(["strategy", "idle_cash", "year"]).yearly_return
        self.assertLess(swept["account", "SPY", 2024], swept["account", "cash", 2024] - 0.3)  # Idle cash in SPY sank.

        rising = run(2000.0)  # SPY triples in 2024: the same model falls short, so the account stands aside.
        folds = rising.folds.set_index("fold")
        self.assertFalse(json.loads(folds.loc[2025, "against_spy"])["random"]["beats_spy"])
        self.assertEqual(json.loads(folds.loc[2025, "combination"]), [])
        self.assertEqual(json.loads(folds.loc[2025, "account_validation_returns"]), {"2024": 0.0})  # Standing aside.
        scoped = rising.metrics.set_index(["strategy", "scope"])
        self.assertEqual((scoped.capital["account", "2025"], scoped.profit["account", "2025"]), (10_000.0, 0.0))
        self.assertEqual(scoped.yearly_return["account", "2025"], 0.0)  # Money left in cash makes nothing.
        self.assertFalse((rising.orders.fold == 2025).any())
        self.assertGreater(scoped.yearly_return["spy_hold", "2025"], 0.1)  # The references still show the year.
        alone = rising.positions[rising.positions.fold == 2025]
        self.assertTrue((alone.reason == "filled").any())  # The model still trades alone, as reported.
        self.assertAlmostEqual(scoped.mean_yearly_return["account", "all"],
                               scoped.yearly_return["account", "2024"] / 2)
        # Standing aside with its idle cash in SPY, the account is SPY held: all of its capital, all year.
        swept = rising.idle_cash.set_index(["strategy", "idle_cash", "year"]).yearly_return
        self.assertEqual(swept["account", "cash", 2025], 0.0)
        self.assertAlmostEqual(swept["account", "SPY", 2025], scoped.yearly_return["spy_hold", "2025"])
        self.assertGreater(swept["random", "SPY", 2025], swept["random", "cash", 2025])  # Its idle cash rose too.
        # Nothing beat SPY on 2024-2025 either: nothing is fit, and the API has nothing to serve.
        self.assertEqual(rising.selection["combination"]["models"], [])
        self.assertEqual((rising.selection["models"], rising.final_models), ({}, {}))
        self.assertFalse(rising.selection["combination"]["strategies"]["random"]["in_combination"])
        folder = write_run(rising, "test", root=self.root)
        self.assertEqual(json.loads((folder / "selection.json").read_text(encoding="utf-8"))["saved_models"], {})
        self.assertTrue((folder / "tables" / "against_spy.csv").is_file())

    def test_the_account_is_also_evaluated_in_the_years_that_chose_it(self):
        factories = {"oracle": lambda: small_models(price=Oracle(self.closes, 2)),
                     "random": lambda: small_models(price=RandomPriceModel(0))}
        result = run_backtest(config(self.root, models=("oracle", "random"), test_years=(2024, 2025)), factories)
        folds = result.folds.set_index("fold")
        scoped = result.metrics.set_index(["strategy", "scope"])
        # No SPY in this data, so both qualify: 2025's account is both models, as 2024's was, so evaluated in 2024 it
        # is 2024's account again, the two sharing one $10,000.
        self.assertEqual(json.loads(folds.loc[2025, "combination"]), ["oracle", "random"])
        checked = json.loads(folds.loc[2025, "account_validation_returns"])
        self.assertAlmostEqual(checked["2024"], scoped.yearly_return["account", "2024"])
        self.assertAlmostEqual(folds.loc[2025, "account_validation_return"], checked["2024"])
        final = result.selection["combination"]
        self.assertEqual(final["models"], ["oracle", "random"])
        self.assertAlmostEqual(final["account_return_by_year"]["2025"], scoped.yearly_return["account", "2025"])
        self.assertAlmostEqual(final["account_mean_yearly_return"], scoped.mean_yearly_return["account", "all"])

    def test_every_model_whose_trades_beat_spy_on_earlier_years_enters_the_account(self):
        years, names = [2024, 2025], ["a", "b", "c", "d"]
        # Per model: its yearly returns trading alone, then SPY's bought with the same dollars on the same dates.
        own = {"a": ((0.30, 0.10), (0.10, 0.10)), "b": ((0.20, 0.10), (0.12, 0.12)),
               "c": ((0.05, 0.05), (0.08, 0.04)), "d": ((-0.10, -0.10), (-0.20, -0.20))}

        def table(returns) -> pd.DataFrame:
            return pd.DataFrame([{"strategy": name, "year": year, "yearly_return": mine[i], "spy_mirror_return": spy[i]}
                                 for name, (mine, spy) in returns.items() for i, year in enumerate(years)])

        self.assertEqual(choose_combination(table(own), names, [], before=2024)[0], ("a", "b", "c", "d"))  # Default.
        members, standing = choose_combination(table(own), names, years)
        # a (0.20 against its SPY's 0.10) and b (0.15 against 0.12) both enter, though a alone did better; d lost
        # money, but less than SPY did on its dates; c's 0.05 is below its SPY's 0.06.
        self.assertEqual(members, ("a", "b", "d"))
        self.assertEqual({name: entry["beats_spy"] for name, entry in standing.items()},
                         {"a": True, "b": True, "c": False, "d": True})
        self.assertAlmostEqual(standing["c"]["mean_yearly_return"], 0.05)
        self.assertAlmostEqual(standing["c"]["spy_mirror_mean_return"], 0.06)
        with self.assertRaises(LeakageError):
            choose_combination(table(own), names, years, before=2025)
        rich = {name: (mine, (0.9, 0.9)) for name, (mine, _) in own.items()}
        self.assertEqual(choose_combination(table(rich), names, years)[0], ())  # None beats SPY.
        blind = {name: (mine, (np.nan, np.nan)) for name, (mine, _) in own.items()}  # No SPY: no rule to apply.
        unknown = choose_combination(table(blind), names, years)
        self.assertEqual((unknown[0], unknown[1]["c"]["beats_spy"], unknown[1]["c"]["qualifies"]),
                         (("a", "b", "c", "d"), None, True))
        gap = choose_combination(table({**own, "a": ((np.nan, 0.9), (0.10, 0.10))}), names, years)  # No return.
        self.assertEqual((gap[0], gap[1]["a"]["qualifies"]), (("b", "d"), False))

    def test_a_grid_tunes_how_many_stocks_a_model_trades_by_return_per_day(self):
        fits = []

        class Counted(Oracle):
            def fit(self, data):
                fits.append(data)

        factories = {"oracle": lambda: small_models(price=Counted(self.closes, 2))}
        grids = {"oracle": {"picks": (2, 1, "10%")}}  # Its best 2 or 1 stocks, or its best 10%: 3 of the 30.
        result = run_backtest(config(self.root, models=("oracle",), test_years=(2024, 2025)), factories,
                              grids=grids)
        two, one, share = 'oracle{"picks": 2}', 'oracle{"picks": 1}', 'oracle{"picks": "10%"}'
        scores = result.scores[result.scores.year == 2024].set_index("candidate")
        self.assertEqual(scores.stocks_per_list.to_dict(), {two: 2.0, one: 1.0, share: 3.0})
        self.assertEqual(len(fits), 2 + 1)  # One fit a year for all three, and the final one.
        # The oracle earns on every stock it lists, so in 2024 its longest list made the most on the $10,000, the cash
        # it left idle aside; but its best stock made the most per invested dollar per day, so 2025 lists 1 a day.
        self.assertGreater(scores.yearly_return[share], scores.yearly_return[two])
        self.assertGreater(scores.return_per_day[one], scores.return_per_day[two])
        self.assertTrue(np.allclose(scores.return_per_day, scores.profit / scores.invested_dollar_days))
        # 2 stocks a session held 2 sessions: $400 of the $10,000 at most closes.
        self.assertAlmostEqual(scores.invested_share[two], 0.04, places=3)
        self.assertEqual(json.loads(result.folds.set_index("fold").loc[2025, "chosen"]), {"oracle": {"picks": 1}})
        listed = result.orders.groupby(["fold", "decision_date"]).size()
        self.assertEqual((set(listed[2024]), set(listed[2025])), ({2}, {1}))  # 2024 runs the default, 2.
        self.assertEqual(set(result.signals[result.signals.fold == 2025].picks), {1})
        final = result.selection["models"]["oracle"]
        self.assertEqual((final["picks"], final["pick_share"], result.final_models["oracle"].picks), (1, None, 1))
        # Every list size beside the one chosen: what the choice compared, and how each did in 2025.
        picks = result.picks.set_index(["choice", "picks"])
        self.assertEqual(picks.loc["2025"].chosen.to_dict(), {"2": False, "1": True, "10%": False})
        self.assertEqual(set(picks.loc["final"].chosen_on), {"2024-2025"})
        self.assertAlmostEqual(picks.tuning_return_per_day["2025", "1"], scores.return_per_day[one])
        later = result.scores[result.scores.year == 2025].set_index("candidate")
        self.assertAlmostEqual(picks.test_return_per_day["2025", "10%"], later.return_per_day[share])
        self.assertTrue(picks.loc["final"].test_return_per_day.isna().all())  # The 2026 test is never run.
        scoped = result.metrics.set_index(["strategy", "scope"]).return_per_day
        self.assertAlmostEqual(scoped["oracle", "2025"], later.return_per_day[one])
        self.assertIn("2025 oracle: 1 a day", format_summary(result))

    def test_picks_is_a_whole_number_or_a_share_and_never_pick_share(self):
        self.assertTrue(all("picks" in grid for grid in GRIDS.values()))  # Every registered model tunes it.
        self.assertFalse(any(UNTUNED & set(grid) for grid in GRIDS.values()))
        self.assertEqual(picking(10), {"picks": 10, "pick_share": None})
        self.assertEqual(picking("0.5%"), {"pick_share": 0.005})
        self.assertEqual(pick_count(small_models(picks=7), 4676), 7)
        shared = replace(small_models(picks=7), pick_share=0.01)  # Rounded down, at least one.
        self.assertEqual([pick_count(shared, ranked) for ranked in (4676, 4700, 648, 50)], [46, 47, 6, 1])
        for wrong in (0, -1, 2.5, True, "0%", "100%", "x%", "5"):  # Refused before any data is read.
            with self.assertRaisesRegex(ValueError, "picks"):
                run_backtest(config(self.root / "missing"), {"flat": small_models},
                             grids={"flat": {"picks": (wrong,)}})
        with self.assertRaisesRegex(ValueError, "pick_share"):
            run_backtest(config(self.root / "missing"), {"flat": small_models}, grids={"flat": {"pick_share": (0.1,)}})

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

    def test_the_list_size_is_chosen_by_return_per_day_once_the_other_settings_are(self):
        # Per candidate, in grid order (the defaults first): its holding and list size, then in 2024 its yearly
        # return, its profit and the dollars its open lots cost at each close, added up.
        grid = [("a", 5, 20, 0.10, 1000.0, 2.5e6), ("b", 5, 5, 0.30, 3000.0, 1.0e6),
                ("c", 10, 20, 0.20, 2000.0, 2.5e6), ("d", 10, 5, 0.05, 500.0, 2.5e5)]
        candidates = [Candidate("m", (("hold_days", hold), ("picks", picks)), label) for label, hold, picks, *_ in grid]
        scores = pd.DataFrame([{"model": "m", "candidate": label, "year": 2024, "yearly_return": mean, "profit": profit,
                                "invested_dollar_days": dollars} for label, _, _, mean, profit, dollars in grid])
        # At the default 20 a day, holding 10 (c, 0.20) beats holding 5 (a, 0.10); b's 0.30, at 5 a day, is never
        # compared there. Then c's list sizes: d made 0.2% per invested dollar per day, c 0.08%, so 5 a day.
        self.assertEqual(select(scores, candidates, [2024]), {"m": "d"})
        # Ties, and a return per day nowhere, keep the default list size.
        self.assertEqual(select(scores.assign(profit=0.0), candidates, [2024]), {"m": "c"})
        self.assertEqual(select(scores.assign(invested_dollar_days=0.0), candidates, [2024]), {"m": "c"})
        # Years pool: profit over dollars, both added up. d loses 600 on 2.5 million dollar-days in 2025, so it earned
        # less per day than c over 2024-2025, though its two yearly ratios average more than c's.
        later = pd.DataFrame([{"model": "m", "candidate": label, "year": 2025, "yearly_return": 0.1, "profit": profit,
                               "invested_dollar_days": 2.5e6} for label, profit in
                              (("a", 100.0), ("b", 100.0), ("c", 100.0), ("d", -600.0))])
        both = pd.concat([scores, later], ignore_index=True)
        self.assertEqual(select(both, candidates, [2024, 2025]), {"m": "c"})
        ratios = both.assign(ratio=both.profit / both.invested_dollar_days).groupby("candidate").ratio.mean()
        self.assertGreater(ratios["d"], ratios["c"])


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
    def test_idle_cash_held_in_a_ticker_grows_with_it_and_pays_costs_on_what_moves(self):
        panel = self.panel({"A": [50.0] * 20, "S": [100.0 * 1.01 ** k for k in range(20)], "F": [80.0] * 20,
                            "D": [100.0] + [50.0] * 19})
        item = book(panel, "m", [Period(0, 2)], [self.listed(["A"], [50])], 100.0)  # One $100 lot, two sessions.
        plain = trade(panel.sessions, [item], 0.0, capital=1000.0)
        rising = trade(panel.sessions, [item], 0.0, capital=1000.0, sweep=sweep_growth(panel, "S"))
        # The lot is flat; the $900 it left idle was held two sessions in a ticker up 1% a session.
        self.assertAlmostEqual(plain.profit.iloc[-1], 0.0)
        self.assertAlmostEqual(rising.profit.iloc[-1], 900.0 * 1.01 ** 2 - 900.0)
        self.assertEqual((rising.filled, list(rising.spent[0])), (plain.filled, list(plain.spent[0])))
        # The sweep pays the costs on the dollars it moves: bought at the first close with the cash the order and its
        # fee left, sold at the last.
        costly = trade(panel.sessions, [item], 10.0, capital=1000.0)
        flat = trade(panel.sessions, [item], 10.0, capital=1000.0, sweep=sweep_growth(panel, "F"))
        left = 1000.0 - 100.0 - 0.1
        self.assertAlmostEqual(flat.fees.sum() - costly.fees.sum(), 0.001 * left + 0.001 * left * 0.999)
        self.assertAlmostEqual(costly.profit.iloc[-1] - flat.profit.iloc[-1], flat.fees.sum() - costly.fees.sum())
        # Only the cash moves, but it pays for the orders: halved in the sweep, it no longer covers the second.
        two = book(panel, "m", [Period(0, 2), Period(1, 2)], [self.listed(["A"], [50])] * 2, 100.0)
        self.assertEqual(trade(panel.sessions, [two], 0.0, capital=200.0).filled, 2)
        self.assertEqual(trade(panel.sessions, [two], 0.0, capital=200.0, sweep=sweep_growth(panel, "D")).filled, 1)
        self.assertIsNone(sweep_growth(panel, "NONE"))  # Not in the data.

    def test_an_order_short_of_cash_sells_the_make_room_books_oldest_lots_first(self):
        # A and B are flat; C rises 10% a session, so a lot of it sold early is worth what that session says.
        panel = self.panel({"A": [100.0] * 20, "B": [100.0] * 20, "C": [100.0 * 1.1 ** k for k in range(20)]})
        good = book(panel, "good", [Period(1, 4), Period(2, 4)], [self.listed(["A"], [100])] * 2, 100.0)
        weak = book(panel, "weak", [Period(0, 4), Period(1, 4)], [self.listed(["C"], [100]),
                                                                  self.listed(["B"], [100])], 100.0)
        skip = trade(panel.sessions, [good, weak], 0.0, record=True, capital=200.0)
        room = trade(panel.sessions, [good, weak], 0.0, record=True, capital=200.0, make_room=1)
        skipped = skip.orders.set_index(["model", "decision_date"]).reason
        # Without it, $200 buys weak's C at session 0 and good's A at session 1; then weak's B finds no cash, nor
        # does good's second A at session 2 (the rows: good's A, A, then weak's C, B).
        self.assertEqual(list(skipped), ["filled", "no_cash", "filled", "no_cash"])
        rows = room.orders.set_index(["model", "decision_date"])
        # With it, weak's own B at session 1 sells weak's older C, bought at 0, for its $110 there, and is bought.
        c = rows.loc[("weak", self.sessions[0])]
        self.assertEqual((c.exit_date, c.exit_rule), (self.sessions[1], "make_room"))
        self.assertAlmostEqual(c.exit_value, 110.0)
        self.assertAlmostEqual(c.mfe, 0.10)  # Its excursions end at its sale.
        self.assertEqual(c.mae, 0.0)
        # Session 2: good's second A finds $10, so it sells weak's oldest open lot, B (bought at 1, flat): $100.
        b = rows.loc[("weak", self.sessions[1])]
        self.assertEqual((b.reason, b.exit_date, b.exit_rule), ("filled", self.sessions[2], "make_room"))
        self.assertEqual(rows.loc[("good", self.sessions[2])].reason, "filled")
        # Every lot is bought; the account ends with its cash plus C's $10 gain, and A's lots run their course.
        self.assertEqual(room.filled, 4)
        self.assertEqual(rows.loc[("good", self.sessions[1])].exit_date, self.sessions[5])
        self.assertAlmostEqual(room.profit.iloc[-1], 10.0)
        self.assertEqual(list(room.invested), [100.0, 200.0, 200.0, 200.0, 200.0, 100.0, 0.0])
        # Never a lot bought at this close: the weak book's own new lot is not sold for the good book's order.
        same = book(panel, "weak", [Period(1, 4)], [self.listed(["B"], [100])], 100.0)
        first = book(panel, "good", [Period(1, 4)], [self.listed(["A", "C"], [100, 100])], 100.0)
        kept = trade(panel.sessions, [first, same], 0.0, record=True, capital=200.0, make_room=1)
        self.assertEqual(list(kept.orders.reason), ["filled", "no_cash", "filled"])  # A, C, then B: C finds none.
        with self.assertRaises(ValueError):
            trade(panel.sessions, [good, weak], 0.0, make_room=2)

    def test_with_its_idle_cash_in_a_sweep_an_order_spends_that_cash_before_it_sells_trades(self):
        # A and B are flat; D, where the idle cash is held, rises 20% a session.
        panel = self.panel({"A": [100.0] * 20, "B": [100.0] * 20, "D": [100.0 * 1.2 ** k for k in range(20)]})
        good = book(panel, "good", [Period(1, 4), Period(2, 4)], [self.listed(["A"], [100])] * 2, 100.0)
        weak = book(panel, "weak", [Period(0, 4)], [self.listed(["B"], [100])], 100.0)
        plain = trade(panel.sessions, [good, weak], 0.0, record=True, capital=190.0, make_room=1)
        swept = trade(panel.sessions, [good, weak], 0.0, record=True, capital=190.0, make_room=1,
                      sweep=sweep_growth(panel, "D"))
        # Kept as cash, the $90 weak's B leaves does not cover good's first A: B is sold for it at session 1, and the
        # second A finds no cash and nothing left to sell.
        rows = plain.orders.set_index(["model", "decision_date"])
        b = rows.loc[("weak", self.sessions[0])]
        self.assertEqual((b.exit_date, b.exit_rule), (self.sessions[1], "make_room"))
        self.assertEqual(rows.loc[("good", self.sessions[2])].reason, "no_cash")
        # Held in D, the $90 is worth $108 at session 1 and pays for the first A. B is sold only at session 2, when
        # the $9.60 left there does not cover the second, and every order is bought.
        rows = swept.orders.set_index(["model", "decision_date"])
        b = rows.loc[("weak", self.sessions[0])]
        self.assertEqual((b.exit_date, b.exit_rule), (self.sessions[2], "make_room"))
        self.assertEqual(list(rows.loc["good"].reason), ["filled", "filled"])
        self.assertEqual(swept.filled, 3)

    def test_an_accounts_profit_splits_into_each_models_lots_and_its_idle_cash(self):
        # A gains 10% at session 2; B is flat; S, where the idle cash may be held, rises 1% a session.
        panel = self.panel({"A": [100.0, 100.0] + [110.0] * 18, "B": [100.0] * 20,
                            "S": [100.0 * 1.01 ** k for k in range(20)]})
        one = book(panel, "one", [Period(0, 2)], [self.listed(["A"], [100])], 100.0)
        two = book(panel, "two", [Period(0, 2), Period(2, 2)], [self.listed(["B"], [100]),
                                                                self.listed(["A"], [110])], 100.0)
        plain = trade(panel.sessions, [one, two], 10.0, record=True, capital=1000.0)
        parts = contributions(plain, panel.sessions, 10.0)
        self.assertEqual(list(parts.index), ["one", "two", "idle_cash"])
        # one's A is bought for $100 and sold for $110 at session 2, where two buys A for $100: the net $10 sale
        # pays 0.1% of $10 there, split 110 : 100 between them. two's B and its A are flat.
        self.assertAlmostEqual(parts.loc["one", "gross"], 10.0)
        self.assertAlmostEqual(parts.loc["one", "costs"], 0.1 + 0.01 * 110 / 210)
        self.assertAlmostEqual(parts.loc["two", "gross"], 0.0)
        self.assertAlmostEqual(parts.loc["two", "costs"], 0.1 + 0.01 * 100 / 210 + 0.1 + 0.1)
        self.assertEqual(parts.loc["idle_cash", "gross"], 0.0)  # Kept as cash, it earns nothing.
        self.assertAlmostEqual(parts.profit.sum(), plain.profit.iloc[-1])
        # Held in S, the same lots are bought, so the models' parts do not change; the idle cash's is what the
        # account gained by it: the cash's growth, less the costs of moving it.
        sweep = sweep_growth(panel, "S")
        swept = trade(panel.sessions, [one, two], 10.0, record=True, capital=1000.0, sweep=sweep)
        held = contributions(swept, panel.sessions, 10.0, sweep)
        pd.testing.assert_frame_equal(held.loc[["one", "two"]], parts.loc[["one", "two"]])
        self.assertGreater(held.loc["idle_cash", "gross"], 0.0)
        self.assertAlmostEqual(held.loc["idle_cash", "costs"], swept.fees.sum() - plain.fees.sum())
        self.assertAlmostEqual(held.loc["idle_cash", "profit"], swept.profit.iloc[-1] - plain.profit.iloc[-1])
        self.assertAlmostEqual(held.profit.sum(), swept.profit.iloc[-1])
        with self.assertRaises(ValueError):  # Not the sweep it was traded with: the parts do not add up.
            contributions(swept, panel.sessions, 10.0)
        with self.assertRaises(ValueError):  # Not recorded.
            contributions(trade(panel.sessions, [one, two], 10.0, capital=1000.0), panel.sessions, 10.0)

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

    def test_a_mirror_buys_every_order_whatever_its_own_cash(self):
        spy = [400.0, 400.0, 300.0, 200.0, 220.0, 240.0] + [240.0] * 14
        panel = self.panel({"A": [100.0, 100.0, 150.0] + [200.0] * 17, "B": [100.0] * 20, "SPY": spy})
        lists = [self.listed(["A"], [100]), self.listed(["A", "B"], [200, 100])]
        item = book(panel, "m", [Period(1, 2), Period(3, 2)], lists, 100.0)
        (paid,) = trade(panel.sessions, [item], 0.0, capital=150.0).spent
        self.assertEqual(list(paid), [100.0, 100.0, 100.0])  # A doubled, so its sale pays for both later orders.
        same = mirror(panel, item, paid)
        # SPY halved meanwhile: held to its own cash, the mirror could pay for only one of them.
        short = trade(panel.sessions, [same], 0.0, record=True, capital=150.0)
        self.assertEqual(list(short.orders.reason), ["filled", "filled", "no_cash"])
        full = trade(panel.sessions, [same], 0.0, record=True, capital=150.0, cash_limit=False)
        self.assertEqual(list(full.orders.reason), ["filled"] * 3)
        self.assertAlmostEqual(full.profit.iloc[-1], 100 * (200 / 400 - 1) + 2 * 100 * (240 / 200 - 1))
        self.assertAlmostEqual(full.yearly_return, full.profit.iloc[-1] / 150.0)

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

    def test_idle_cash_names_distinct_tickers(self):
        for wrong in (("cash",), ("SPY", "SPY"), ("",), (5,)):  # Refused before any data is read.
            with self.assertRaisesRegex(ValueError, "idle_cash"):
                config(self.root, idle_cash=wrong)
        self.assertEqual(config(self.root).idle_cash, ("SPY", "BIL"))

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
        for table in ("scores", "folds", "orders", "profit", "yearly", "metrics", "against_spy", "picks", "idle_cash"):
            self.assertTrue((folder / "tables" / f"{table}.csv").is_file(), table)
        yearly = yearly_table(result).set_index("strategy")
        self.assertEqual(list(yearly.index), ["flat", "account", "spy_hold", "spy_mirror", "universe_ew"])
        self.assertEqual(list(yearly.columns), ["2025", "mean"])
        self.assertAlmostEqual(yearly.loc["account", "2025"], yearly.loc["flat", "2025"])  # One model: the same.


if __name__ == "__main__":
    unittest.main()
