import ast
from contextlib import redirect_stderr
from dataclasses import replace
from datetime import date
import importlib.util
import io
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from stock_recommender.backtest import metrics
from stock_recommender.backtest.panel import Panel
from stock_recommender.backtest.portfolio import simulate
from stock_recommender.backtest.protocol import FoldData, Period, news_window, schedule, walk_forward
from stock_recommender.backtest.run import BacktestConfig, load_data, run_backtest
from stock_recommender.backtest.window import (DataError, LeakageError, WindowedStore, check_window, day_files)
from stock_recommender.config import ROOT, Settings
from stock_recommender.models import RandomPriceModel
from stock_recommender.recommend import decision_inputs, news_cutoff, recommend, score_day
from .backtest_fixtures import (SESSIONS, config, random_walk, small_models, write_actions, write_bars,
                                write_changes, write_dividends, write_news, write_splits)

BACKTEST = ROOT / "src/stock_recommender/backtest"


class FixedPriceModel:
    def __init__(self, returns: dict[str, float]):
        self.returns = pd.Series(returns)

    def predict(self, bars, tickers):
        return self.returns


class Store(unittest.TestCase):
    """A temporary store: 40 random-walk tickers over SESSIONS, no splits, a few articles."""
    tickers = 40

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.prices = random_walk(self.tickers)
        write_bars(self.root / "bars", self.prices["open"], self.prices["close"])
        write_splits(self.root / "splits.parquet", [])
        write_news(self.root / "news", [("2025-01-15T15:00:00Z", ["T000"]), ("2025-02-03T12:00:00Z", ["T001"])])

    def tearDown(self):
        self.directory.cleanup()

    def run_backtest(self, factories=None, **changes):
        factories = factories or {"flat": small_models}
        return run_backtest(config(self.root, **changes), factories)


class WindowGuardTest(Store):
    def poison(self, root: Path):
        """Files a backtest must never open: unreadable ones, and a valid 2026 file with absurd prices."""
        for path in ("bars/2026/2026-01-02.parquet", "bars/2020/2020-12-31.parquet", "bars/2025/2026-01-05.parquet",
                     "news/2026/2026-01-01.parquet", "news/2020/2020-12-31.parquet"):
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_bytes(b"not a parquet file")
        write_bars(root / "bars", self.prices["open"].iloc[-1:].set_axis([pd.Timestamp("2026-01-05")]) * 1e9)

    def test_day_files_lists_only_2021_to_2025(self):
        self.poison(self.root)
        days = day_files(self.root / "bars")
        self.assertEqual((min(days), max(days)), (date(2024, 11, 1), date(2025, 3, 31)))
        days = list(day_files(self.root / "bars"))
        self.assertTrue(all(date(2021, 1, 1) <= day <= date(2025, 12, 31) for day in days))
        self.assertNotIn(date(2026, 1, 5), days)  # Misfiled in the 2025 folder, and in a 2026 folder.

    def test_a_full_run_never_opens_a_file_outside_the_window(self):
        clean = self.run_backtest()
        self.poison(self.root)
        opened = []
        original = pq.read_table

        def record(path, *args, **kwargs):
            opened.append(Path(path))
            return original(path, *args, **kwargs)

        with mock.patch("stock_recommender.backtest.window.pq.read_table", record):
            data = load_data(config(self.root))
            poisoned = run_backtest(config(self.root), {"flat": small_models}, data)
        dated = [path for path in opened if path.suffix == ".parquet" and path.stem[:4].isdigit()]
        self.assertTrue(dated)
        self.assertTrue(all(date(2021, 1, 1) <= date.fromisoformat(path.stem) <= date(2025, 12, 31) for path in dated))
        self.assertTrue(all(path.parent.name in ("2024", "2025") for path in dated))
        self.assertEqual(sorted(map(str, opened)), sorted(map(str, data[0].opened)))
        pd.testing.assert_frame_equal(poisoned.equity, clean.equity)
        self.assertEqual(poisoned.guards["files_outside_window"], 0)

    def test_rows_outside_their_files_day_raise(self):
        bad = self.prices["open"].iloc[:1]
        write_bars(self.root / "late", bad.set_axis([pd.Timestamp("2026-01-02")]))
        (self.root / "late/2026/2026-01-02.parquet").replace(self.root / "bars/2025/2025-03-31.parquet")
        with self.assertRaisesRegex(LeakageError, "2026-01-02"):
            WindowedStore(self.root / "bars", self.root / "news", self.root / "splits.parquet").read_bars()
        write_news(self.root / "late_news", [("2026-01-01T00:00:00Z", ["T000"])])
        (self.root / "news/2025").mkdir(parents=True, exist_ok=True)
        (self.root / "late_news/2026/2026-01-01.parquet").replace(self.root / "news/2025/2025-12-31.parquet")
        with self.assertRaises(LeakageError):
            WindowedStore(self.root / "bars", self.root / "news", self.root / "splits.parquet").read_news()
        write_splits(self.root / "splits.parquet", [("T000", "2026-03-02", 1, 2)])
        with self.assertRaisesRegex(LeakageError, "outside"):
            WindowedStore(self.root / "bars", self.root / "news", self.root / "splits.parquet").read_splits()

    def test_no_setting_reaches_2026_or_trains_on_nothing(self):
        for years in ((2026,), (2025, 2026), (2021,), (2020,)):
            with self.assertRaises(LeakageError):
                config(self.root, test_years=years)
        with self.assertRaises(ValueError):
            config(self.root, test_years=(2022, 2024))
        with self.assertRaises(LeakageError):
            check_window(date(2025, 6, 1), date(2026, 1, 1))

    def test_cli_refuses_2026_before_reading_anything(self):
        spec = importlib.util.spec_from_file_location("backtest_cli", ROOT / "scripts/backtest.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with (mock.patch.object(sys, "argv", ["backtest.py", "--test-years", "2025", "2026"]),
              mock.patch.object(module, "run_backtest") as run, redirect_stderr(io.StringIO())):
            with self.assertRaisesRegex(SystemExit, "2026 is never used"):
                module.main()
        run.assert_not_called()

    def test_holdings_never_end_after_the_last_session(self):
        data = load_data(config(self.root))
        panel = data[1]
        last = len(panel.sessions) - 1
        panel.forward_returns(last - 3, 2, ["T000"])  # Exit on the last session: allowed.
        with self.assertRaises(LeakageError):
            panel.forward_returns(last - 2, 2, ["T000"])
        result = run_backtest(config(self.root), {"flat": small_models}, data)
        self.assertLessEqual(result.periods.exit_date.max(), panel.sessions[-1])
        self.assertLessEqual(result.periods.exit_date.max(), pd.Timestamp("2025-12-31"))

    def test_backtest_code_reads_data_only_through_the_window_module(self):
        # Outside window.py nothing opens or lists files; nowhere does a calendar or a clock come from elsewhere.
        file_access = {"read_parquet", "read_table", "ParquetFile", "dataset", "read_csv", "glob", "rglob",
                       "iterdir", "listdir", "scandir", "walk"}
        anywhere = {"bdate_range", "BDay", "today", "now", "utcnow"}
        # The API's file readers and load_models() skip the window's guards, and trained models would leak.
        forbidden_imports = {"read_bars", "read_news", "recent_day_files", "load_models"}
        forbidden_modules = {"requests", "httpx", "urllib", "duckdb", "socket"}
        for path in BACKTEST.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = {alias.name for alias in node.names}
                    modules = names | ({node.module} if isinstance(node, ast.ImportFrom) and node.module else set())
                    self.assertFalse(forbidden_modules & {name.split(".")[0] for name in modules}, path.name)
                    self.assertFalse(forbidden_imports & names, f"{path.name}:{node.lineno}")
                if isinstance(node, ast.Call):
                    name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                    self.assertNotIn(name, anywhere, f"{path.name}:{node.lineno}")
                    if path.name != "window.py":
                        self.assertNotIn(name, file_access, f"{path.name}:{node.lineno}")


class PointInTimeTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.sessions = SESSIONS[:30]

    def tearDown(self):
        self.directory.cleanup()

    def panel(self, opens: dict, splits=(), volumes: dict | None = None, **load) -> Panel:
        frame = pd.DataFrame(opens, index=self.sessions, dtype=float)
        write_bars(self.root / "bars", frame, volumes=None if volumes is None else pd.DataFrame(volumes, index=self.sessions, dtype=float))
        write_splits(self.root / "splits.parquet", list(splits))
        store = WindowedStore(self.root / "bars", self.root / "news", self.root / "splits.parquet")
        return Panel.load(store, strict_calendar=False, **load)

    def test_window_is_split_adjusted_as_of_the_decision(self):
        split = 10  # A 2-for-1 split executes at session 10's open.
        prices = [100.0] * split + [50.0] * (30 - split)
        volumes = [1000.0] * split + [2000.0] * (30 - split)
        panel = self.panel({"A": prices}, [("A", str(self.sessions[split].date()), 1, 2)], {"A": volumes})
        before = panel.window(split - 1, 5)
        self.assertEqual(set(before.close), {100.0})  # As traded: the split has not happened yet.
        self.assertEqual(set(before.volume), {1000.0})
        after = panel.window(split + 2, 5)
        self.assertEqual(set(after.close), {50.0})  # Earlier days on the post-split basis.
        self.assertEqual(set(after.volume), {2000.0})
        self.assertEqual(set(after.close * after.volume), {100_000.0})

    def test_a_later_reverse_split_does_not_lift_a_penny_stock_into_the_universe(self):
        split = 12  # 1-for-10 at session 12: $3 becomes $30.
        panel = self.panel({"B": [3.0] * split + [30.0] * (30 - split)}, [("B", str(self.sessions[split].date()), 10, 1)])
        settings = Settings(min_price=5.0, min_dollar_volume=0.0)
        self.assertNotIn("B", panel.universe(split - 1, 5, settings).index)
        self.assertIn("B", panel.universe(split, 5, settings).index)

    def test_bars_that_look_split_adjusted_are_refused(self):
        opens = {f"S{i:02d}": [10.0 + i] * 30 for i in range(25)}  # No price jumps at the listed splits.
        splits = [(ticker, str(self.sessions[15].date()), 1, 2) for ticker in opens]
        with self.assertRaisesRegex(DataError, "--unadjusted"):
            self.panel(opens, splits)

    def test_a_split_the_prices_contradict_is_reported_not_applied(self):
        opens = {f"S{i:02d}": [20.0] * 15 + [10.0] * 15 for i in range(24)} | {"X": [7.0] * 30}
        splits = [(ticker, str(self.sessions[15].date()), 1, 2) for ticker in opens]
        panel = self.panel(opens, splits)
        audit = panel.split_audit.set_index("ticker")
        self.assertFalse(audit.loc["X"].applied)
        self.assertEqual(audit.applied.sum(), 24)
        self.assertEqual(panel.forward_returns(12, 4, ["X"]).forward_return.iloc[0], 0.0)

    def test_returns_across_splits(self):
        n = 30
        panel = self.panel({
            "A": [100.0] * 6 + [51.0] * (n - 6),  # 2-for-1 at session 6: entry 100, exit 51 -> +2%.
            "B": [2.0] * 6 + [21.0] * (n - 6),  # 1-for-10 at session 6: entry 2, exit 21 -> +5%.
            "C": [100.0] * 5 + [50.0, 55.0] + [55.0] * (n - 7),  # 2-for-1 on the entry day: 50 -> 55, +10%.
        }, [("A", str(self.sessions[6].date()), 1, 2), ("B", str(self.sessions[6].date()), 10, 1),
            ("C", str(self.sessions[5].date()), 1, 2)])
        returns = panel.forward_returns(3, 3, ["A", "B"]).forward_return  # Entry session 4, exit session 7.
        self.assertAlmostEqual(returns["A"], 0.02)
        self.assertAlmostEqual(returns["B"], 0.05)
        self.assertAlmostEqual(panel.forward_returns(4, 1, ["C"]).forward_return["C"], 0.10)

    def test_a_missing_exit_sells_at_the_last_close_and_a_missing_entry_stays_cash(self):
        n = 30
        panel = self.panel({"D": [10.0] * 6 + [np.nan] * (n - 6), "E": [10.0] * n},
                           volumes={"D": [1000.0] * n, "E": [1000.0] * 4 + [0.0] + [1000.0] * (n - 5)})
        returns = panel.forward_returns(3, 4, ["D", "E", "NOPE"])  # Entry session 4, exit session 8.
        self.assertEqual(returns.exit_reason.to_dict(), {"D": "last_close", "E": "no_entry", "NOPE": "no_entry"})
        self.assertEqual(returns.forward_return["D"], 0.0)
        self.assertTrue(np.isnan(returns.forward_return["E"]))

    def test_models_see_news_only_before_the_close(self):
        bars = pd.DataFrame({"date": pd.Timestamp("2025-07-01"), "ticker": ["A"], "open": 10.0, "high": 10.0,
                             "low": 10.0, "close": 10.0, "volume": 1e6, "vwap": 10.0, "transactions": 1})
        settings = Settings(min_price=0, min_dollar_volume=0)
        for day, inside, outside in (("2025-07-01", "19:59:59", "20:00:00"), ("2025-12-01", "20:59:59", "21:00:00")):
            news = pd.DataFrame({"published_utc": pd.to_datetime([f"{day}T{inside}Z", f"{day}T{outside}Z"], utc=True),
                                 "id": ["in", "out"], "tickers": [["A"], ["A"]]})
            snapshot = decision_inputs(bars.assign(date=pd.Timestamp(day)), news, pd.Timestamp(day), settings, 7)
            self.assertEqual(list(snapshot.news.id), ["in"])


class ProtocolTest(Store):
    def test_walk_forward_trains_on_every_earlier_year(self):
        sessions = pd.bdate_range("2021-01-04", "2025-12-31")
        folds = walk_forward(sessions, [2022, 2023, 2024, 2025])
        self.assertEqual([str(fold.train_end.date()) for fold in folds],
                         ["2021-12-31", "2022-12-30", "2023-12-29", "2024-12-31"])
        once = walk_forward(sessions, [2023, 2024], refit=False)
        self.assertEqual([fold.train_end for fold in once], [once[0].train_end] * 2)
        self.assertEqual([fold.refit for fold in once], [True, False])
        periods = schedule(sessions, folds, 5)
        flat = [period for year in periods for period in periods[year]]
        self.assertEqual(sessions[flat[0].entry], pd.Timestamp("2022-01-03"))
        self.assertTrue(all(a.exit == b.entry for a, b in zip(flat, flat[1:])))  # Back to back.
        self.assertLessEqual(sessions[flat[-1].exit], pd.Timestamp("2025-12-31"))

    def test_training_data_stops_at_the_cutoff(self):
        store, panel, news = load_data(config(self.root))
        cutoff = panel.sessions[40]
        data = FoldData(panel, news, Settings(min_price=0, min_dollar_volume=0), cutoff, 2, 3, 7)
        labels = data.labels()
        self.assertLessEqual(labels.exit_date.max(), cutoff)
        self.assertEqual(labels.decision_date.min(), panel.sessions[2])  # The first day with 3 sessions of bars.
        self.assertEqual(labels.decision_date.max(), panel.sessions[37])  # Purged: its holding ends at the cutoff.
        expected = self.prices["open"].iloc[39] / self.prices["open"].iloc[37] - 1
        row = labels[(labels.decision_date == panel.sessions[36]) & (labels.ticker == "T005")]
        self.assertAlmostEqual(row.forward_return.iloc[0], expected["T005"])
        self.assertEqual(data.bars().date.max(), cutoff)
        self.assertTrue((data.news().published_utc < news_cutoff(cutoff)).all())
        self.assertEqual(data.snapshot(cutoff).as_of, cutoff)
        with self.assertRaises(LeakageError):
            data.snapshot(panel.sessions[41])

    def test_each_fold_fits_fresh_models_on_its_own_past_only(self):
        sessions = pd.bdate_range("2023-11-01", "2025-02-28")
        prices = random_walk(25, sessions, seed=1)
        root = self.root / "two_years"
        write_bars(root / "bars", prices["open"], prices["close"])
        write_splits(root / "splits.parquet", [])
        (root / "news").mkdir()
        log = []

        class Spy:
            def __init__(self):
                self.id = len([entry for entry in log if entry[0] == "new"])
                log.append(("new", self.id, None))

            def fit(self, data):
                labels = data.labels()
                log.append(("fit", self.id, data.cutoff, labels.exit_date.max()))

            def predict(self, bars, tickers):
                log.append(("predict", self.id, bars.date.max()))
                return pd.Series(0.0, index=tickers)

        result = run_backtest(config(root, test_years=(2024, 2025)), {"flat": lambda: small_models(price=Spy())})
        fits = [entry for entry in log if entry[0] == "fit"]
        self.assertEqual([str(cutoff.date()) for _, _, cutoff, _ in fits], ["2023-12-29", "2024-12-31"])
        self.assertEqual(len({model for _, model, _, _ in fits}), 2)  # A new model per fold.
        for _, model, cutoff, last_label in fits:
            self.assertLessEqual(last_label, cutoff)
            predictions = [entry[2] for entry in log if entry[0] == "predict" and entry[1] == model]
            self.assertTrue(predictions)
            self.assertGreaterEqual(min(predictions), cutoff)  # Fit first, on the past only.
        self.assertEqual(result.guards["training_label_exits"], {2024: "2023-12-29", 2025: "2024-12-31"})
        # A fold's metrics cover its own holdings, even the last one that exits in the next year.
        scoped = result.metrics.set_index(["strategy", "scope"])
        for fold in result.folds.itertuples():
            row = scoped.loc[("flat", str(fold.fold))]
            self.assertEqual((row.start, row.end, row.periods), (fold.first_entry, fold.last_exit, fold.holdings))
        self.assertEqual(result.folds.last_exit.iloc[0].year, 2025)


class EngineTest(Store):
    tickers = 60

    def test_leakage_canary(self):
        opens = self.prices["open"]
        holding = 2

        class Momentum:
            def predict(self, bars, tickers):
                closes = bars.pivot(index="date", columns="ticker", values="close")
                return (closes.iloc[-1] / closes.iloc[-2] - 1).reindex(tickers)

        class Oracle:  # Test-only: knows the future, to show the engine would reward it.
            def predict(self, bars, tickers):
                t = opens.index.get_loc(bars.date.max())
                realized = opens.iloc[t + 1 + holding] / opens.iloc[t + 1] - 1
                return realized.reindex(tickers)

        result = self.run_backtest({"momentum": lambda: small_models(price=Momentum()),
                                    "oracle": lambda: small_models(price=Oracle())},
                                   models=("momentum", "oracle"), top_n=5)
        ic = result.signals.groupby("strategy").ic.mean()
        self.assertLess(abs(ic["momentum"]), 0.1)  # Only past bars: no edge on a random walk.
        self.assertGreater(ic["oracle"], 0.95)
        growth = result.metrics.set_index(["strategy", "scope"]).growth
        self.assertGreater(growth["oracle", "all"], growth["universe_ew", "all"] * 1.5)

    def test_the_backtest_ranks_exactly_as_the_api(self):
        store, panel, news = load_data(config(self.root))
        settings = Settings(min_price=0, min_dollar_volume=0, refresh_time="off")
        models = small_models(price=FixedPriceModel({f"T{i:03d}": (i % 7) / 100 for i in range(self.tickers)}))
        t = 25
        backtest = score_day(panel.window(t, models.bars_lookback), news_window(news, news_cutoff(panel.sessions[t]), 7),
                             panel.sessions[t], settings, models)
        api_dir = self.root / "api_bars"
        for day in panel.sessions[:t + 1]:
            source = self.root / "bars" / str(day.year) / f"{day.date()}.parquet"
            (api_dir / str(day.year)).mkdir(parents=True, exist_ok=True)
            shutil.copy(source, api_dir / str(day.year) / source.name)
        api = recommend(replace(settings, bars_dir=api_dir, news_dir=self.root / "news"), models)
        pd.testing.assert_frame_equal(backtest.table, api.table)
        self.assertEqual((backtest.as_of, backtest.universe_size), (api.as_of, api.universe_size))

    def test_growth_formulas(self):
        store, panel, _ = load_data(config(self.root))
        opens = self.prices["open"]
        periods = [Period(10, 2), Period(12, 2)]
        weights = [pd.Series({"T000": 1.0}), pd.Series({"T001": 1.0})]
        curve, table, positions = simulate(panel, periods, weights, cost_bps=10, borrow_bps=0, capital=1000,
                                           folds=[2025, 2025])
        a = opens["T000"].iloc[13] / opens["T000"].iloc[11]
        b = opens["T001"].iloc[15] / opens["T001"].iloc[13]
        expected = 1000 * (1 - 0.001)  # Buy T000 (turnover 1).
        expected *= a
        expected *= 1 - 0.001 * 2  # Sell T000, buy T001 (turnover 2).
        expected *= b
        expected *= 1 - 0.001  # Sell T001 at the end.
        self.assertAlmostEqual(curve.iloc[-1], expected)
        self.assertEqual(curve.index[0], panel.sessions[11])
        self.assertEqual(curve.index[-1], panel.sessions[15])
        self.assertEqual(list(table.turnover), [1.0, 2.0])
        mid = 999 * opens["T000"].iloc[12] / opens["T000"].iloc[11]
        self.assertAlmostEqual(curve[panel.sessions[12]], mid)

    def test_long_short_nets_out_and_pays_borrow(self):
        store, panel, _ = load_data(config(self.root))
        weights = [pd.Series({"T000": 0.5, "T001": -0.5})]
        curve, table, _ = simulate(panel, [Period(10, 2)], weights, cost_bps=0, borrow_bps=252, capital=100,
                                   folds=[2025])
        opens = self.prices["open"]
        a = opens["T000"].iloc[13] / opens["T000"].iloc[11] - 1
        b = opens["T001"].iloc[13] / opens["T001"].iloc[11] - 1
        self.assertAlmostEqual(curve.iloc[-1], 100 * (1 + 0.5 * a - 0.5 * b - 0.5 * 1e-4 * 2))

    def test_a_run_reports_every_strategy_and_benchmark(self):
        result = self.run_backtest(models=("flat", "random"), factories={
                                       "flat": small_models, "random": lambda: small_models(price=RandomPriceModel(3))},
                                   ensembles=(("mix", (("flat", 1.0), ("random", 2.0))),), null_seeds=3)
        self.assertEqual(list(result.equity.columns), ["flat", "random", "mix", "universe_ew", "spy"])
        self.assertEqual(set(result.metrics.scope), {"all", "2025"})
        self.assertEqual(len(result.null), 3)
        self.assertIn("ensemble(", result.versions["mix"])
        flat = result.positions[result.positions.strategy == "flat"]
        first = flat[flat.decision_date == flat.decision_date.min()]
        self.assertEqual(len(first), 2)  # top_n=2 with constant scores: the two most liquid tickers.


class ReviewFindingsTest(Store):
    """Behaviors the review of the simulator asked to pin down."""

    def test_training_data_holds_nothing_after_its_cutoff(self):
        store, panel, news = load_data(config(self.root))
        cutoff = panel.sessions[40]
        panel.universe(60, 3, Settings(min_price=0, min_dollar_volume=0))  # A later universe already memoized.
        data = FoldData(panel, news, Settings(min_price=0, min_dollar_volume=0), cutoff, 2, 3, 7)
        cut = data._panel
        self.assertEqual(cut.sessions.max(), cutoff)
        self.assertEqual(cut.bars.date.max(), cutoff)
        for matrix in (cut.traded, cut.multiplier, cut.growth, cut.q_open, cut.mark_open):
            self.assertEqual(len(matrix), 41)
        self.assertTrue(all(key[0] <= 40 for key in cut._universes))
        self.assertTrue((data._news.published_utc < news_cutoff(cutoff)).all())
        with self.assertRaises(LeakageError):
            cut.forward_returns(39, 2, ["T000"])  # A holding ending after the cutoff cannot be priced.
        self.assertFalse([name for name in vars(data) if not name.startswith("_")
                          and isinstance(getattr(data, name), (pd.DataFrame, Panel))])

    def test_a_models_result_does_not_depend_on_the_other_models_in_the_run(self):
        seen = []

        class Momentum:
            def predict(self, bars, tickers):
                seen.append(bars.date.nunique())
                closes = bars.pivot(index="date", columns="ticker", values="close")
                return (closes.iloc[-1] / closes.iloc[0] - 1).reindex(tickers)

        def short():
            return small_models(price=Momentum())

        def long():
            return small_models(bars_lookback=10)

        alone = self.run_backtest({"short": short}, models=("short",))
        self.assertEqual(set(seen), {3})
        beside = self.run_backtest({"short": short, "long": long}, models=("short", "long"))
        self.assertEqual(set(seen), {3})
        pd.testing.assert_series_equal(alone.equity["short"], beside.equity["short"])

    def test_decisions_and_training_never_see_a_later_split(self):
        split = SESSIONS.get_loc(pd.Timestamp("2025-02-03"))
        root = self.root / "split"
        opens = pd.DataFrame({f"T{i:02d}": 20.0 + i for i in range(10)}, index=SESSIONS)
        opens["PENNY"] = [3.0] * split + [30.0] * (len(SESSIONS) - split)  # 1-for-10 on 2025-02-03.
        write_bars(root / "bars", opens)
        write_splits(root / "splits.parquet", [("PENNY", "2025-02-03", 10, 1)])
        (root / "news").mkdir()
        calls = []

        class Recorder:
            def fit(self, data):
                calls.append(("fit", data.cutoff, data.bars(), data.snapshot(data.cutoff - pd.offsets.BDay(5))))

            def predict(self, bars, tickers):
                calls.append(("predict", bars.date.max(), list(tickers), bars[bars.ticker == "PENNY"].close.tolist()))
                return pd.Series(0.0, index=tickers)

        run_backtest(config(root, min_price=5.0), {"flat": lambda: small_models(price=Recorder())})
        predictions = [call for call in calls if call[0] == "predict"]
        self.assertTrue(predictions)
        for _, as_of, tickers, closes in predictions:
            if as_of < SESSIONS[split]:
                self.assertNotIn("PENNY", tickers)  # $3 as traded: under the $5 floor.
            else:
                self.assertIn("PENNY", tickers)
                self.assertTrue(all(close == 30.0 for close in closes))  # Earlier days on the new basis.
        _, cutoff, bars, snapshot = next(call for call in calls if call[0] == "fit")
        self.assertEqual(set(bars[bars.ticker == "PENNY"].close), {3.0})  # The split comes after the cutoff.
        self.assertEqual(snapshot.as_of, cutoff - pd.offsets.BDay(5))
        self.assertEqual(snapshot.bars.date.max(), snapshot.as_of)

    def test_ensemble_weights_reach_the_trades(self):
        tickers = [f"T{i:03d}" for i in range(self.tickers)]
        up = {ticker: i for i, ticker in enumerate(tickers)}
        down = {ticker: -i for i, ticker in enumerate(tickers)}
        factories = {"up": lambda: small_models(price=FixedPriceModel(up)),
                     "down": lambda: small_models(price=FixedPriceModel(down))}

        def picks(result, name):
            held = result.positions[result.positions.strategy == name]
            return held.groupby("decision_date").ticker.apply(frozenset)

        for heavy, light in (("up", "down"), ("down", "up")):
            result = self.run_backtest(factories, models=("up", "down"),
                                       ensembles=(("mix", ((heavy, 3.0), (light, 1.0))),))
            pd.testing.assert_series_equal(picks(result, "mix"), picks(result, heavy), check_names=False)

    def test_trading_options_reach_the_simulation(self):
        free = self.run_backtest(models=("flat",))
        costly = self.run_backtest(models=("flat",), cost_bps=10)
        self.assertGreater(costly.periods.cost.sum(), 0)
        self.assertLess(costly.equity.flat.iloc[-1], free.equity.flat.iloc[-1])
        both = self.run_backtest(models=("flat",), side="long_short", top_n=3)
        weights = both.positions.weight
        self.assertEqual((weights > 0).sum(), (weights < 0).sum())
        self.assertGreater((weights < 0).sum(), 0)

    def test_turnover_counts_drift(self):
        store, panel, _ = load_data(config(self.root))
        opens = self.prices["open"]
        weights = [pd.Series({"T000": 0.5, "T001": 0.5})] * 2
        _, table, _ = simulate(panel, [Period(10, 2), Period(12, 2)], weights, 0, 0, 1000, [2025, 2025])
        a = opens["T000"].iloc[13] / opens["T000"].iloc[11]
        b = opens["T001"].iloc[13] / opens["T001"].iloc[11]
        drift = 0.5 * a / (0.5 * a + 0.5 * b)
        self.assertAlmostEqual(table.turnover.iloc[1], 2 * abs(drift - 0.5))

    def test_dividends_are_income_not_losses(self):
        root = self.root / "dividends"
        opens = pd.DataFrame({"PAY": [100.0] * 10 + [98.0] * 20, "LIQ": [20.39] * 10 + [1.41] * 20,
                              "BIG": [50.0] * 30}, index=SESSIONS[:30])
        write_bars(root / "bars", opens)
        write_splits(root / "splits.parquet", [])
        write_dividends(root / "dividends.parquet", [
            ("PAY", str(SESSIONS[10].date()), 2.0), ("PAY", str(SESSIONS[10].date()), 2.0),  # Listed twice.
            ("LIQ", str(SESSIONS[10].date()), 18.98),
            ("BIG", str(SESSIONS[10].date()), 80.0)])  # More than the price: a data error, not applied.
        write_dividends(root / "foreign.parquet", [("PAY", str(SESSIONS[10].date()), 1.0)], currency="EUR")
        store = WindowedStore(root / "bars", root / "news", root / "splits.parquet")
        panel = Panel.load(store, strict_calendar=False, dividends_path=root / "dividends.parquet")
        returns = panel.forward_returns(7, 4, ["PAY", "LIQ", "BIG"])
        self.assertAlmostEqual(returns.forward_return["PAY"], 0.0)  # 98 + 2: the price drop is the payout.
        self.assertAlmostEqual(returns.forward_return["LIQ"], (1.41 + 18.98) / 20.39 - 1)
        stub = pd.DataFrame({"LIQ": [20.39] * 10 + [1.41, 0.705] + [0.705] * 18}, index=SESSIONS[:30])
        write_bars(root / "stub", stub)
        held = Panel.load(WindowedStore(root / "stub", root / "news", root / "splits.parquet"), strict_calendar=False,
                          dividends_path=root / "dividends.parquet")
        # The $18.98 stays cash while the $1.41 stub halves: it is not reinvested into the stub.
        self.assertAlmostEqual(held.forward_returns(7, 4, ["LIQ"]).forward_return["LIQ"], (0.705 + 18.98) / 20.39 - 1)
        history = held.history().query("ticker == 'LIQ'")
        self.assertAlmostEqual(history.return_factor.iloc[10] / history.return_factor.iloc[9], 1 + 18.98 / 1.41)
        self.assertEqual(returns.forward_return["BIG"], 0.0)
        self.assertEqual(returns.entry_price["PAY"], 100.0)  # As traded.
        self.assertFalse(panel.dividend_audit.set_index("ticker").loc["BIG"].applied)
        foreign = Panel.load(store, strict_calendar=False, dividends_path=root / "foreign.parquet")
        self.assertEqual(foreign.dividend_audit.reason.tolist(), ["no_fx_rate"])  # No EUR trust in these bars.
        priced = Panel.load(store, strict_calendar=False)
        self.assertAlmostEqual(priced.forward_returns(7, 4, ["PAY"]).forward_return["PAY"], -0.02)

    def test_same_day_split_legs_net_out_and_small_splits_are_checked(self):
        root = self.root / "legs"
        opens = pd.DataFrame({f"S{i:02d}": [20.0] * 15 + [10.0] * 15 for i in range(22)}, index=SESSIONS[:30])
        opens["PAIR"] = [5.0] * 30
        opens["BOTH"] = [10.0] * 15 + [1000.0] * 15
        opens["WARR"] = [10.0] * 30
        write_bars(root / "bars", opens)
        day = str(SESSIONS[15].date())
        write_splits(root / "splits.parquet", [(f"S{i:02d}", day, 1, 2) for i in range(22)] + [
            ("PAIR", day, 1, 10000), ("PAIR", day, 10000, 1),  # Cancels out: no split.
            ("BOTH", day, 1, 500), ("BOTH", day, 50000, 1),  # Net 1-for-100.
            ("WARR", day, 4, 5)])  # A 5-for-4 the prices do not show.
        store = WindowedStore(root / "bars", root / "news", root / "splits.parquet")
        splits = store.read_splits().set_index("ticker")
        self.assertNotIn("PAIR", splits.index)
        self.assertAlmostEqual(splits.loc["BOTH"].split_to, 0.01)
        panel = Panel.load(store, strict_calendar=False)
        self.assertAlmostEqual(panel.forward_returns(12, 4, ["BOTH"]).forward_return["BOTH"], 0.0)
        self.assertFalse(panel.split_audit.set_index("ticker").loc["WARR"].applied)
        self.assertEqual(panel.forward_returns(12, 4, ["WARR"]).forward_return["WARR"], 0.0)

    def test_losses_beyond_2x_are_reported_and_years_are_scoped_to_their_fold(self):
        root = self.root / "crash"
        opens = pd.DataFrame({f"T{i:02d}": 30.0 + i for i in range(5)}, index=SESSIONS)
        opens["CRASH"] = [100.0] * 60 + [30.0] * (len(SESSIONS) - 60)
        write_bars(root / "bars", opens)
        write_splits(root / "splits.parquet", [])
        (root / "news").mkdir()
        result = run_backtest(config(root, top_n=6), {"flat": small_models})
        kinds = result.anomalies.groupby("kind").ticker.apply(set)
        self.assertEqual(kinds["position_return_beyond_2x"], {"CRASH"})
        self.assertEqual(kinds["universe_return_beyond_2x"], {"CRASH"})
        fold = result.folds.iloc[0]
        scoped = result.metrics.set_index(["strategy", "scope"]).loc[("flat", "2025")]
        self.assertEqual((scoped.start, scoped.end), (fold.first_entry, fold.last_exit))
        self.assertEqual(scoped.periods, fold.holdings)

    def test_cli_backtests_a_model_alone(self):
        spec = importlib.util.spec_from_file_location("backtest_cli", ROOT / "scripts/backtest.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for argv, ensembles in ((["--models", "placeholder"], ()), ([], BacktestConfig().ensembles)):
            with (mock.patch.object(sys, "argv", ["backtest.py", *argv]),
                  mock.patch.object(module, "run_backtest", side_effect=ValueError("stop")) as run,
                  redirect_stderr(io.StringIO())):
                with self.assertRaises(SystemExit):
                    module.main()
            self.assertEqual(run.call_args.args[0].ensembles, ensembles)


class PointInTimeInputsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        sessions = SESSIONS[:30]
        opens = pd.DataFrame({"A": [100.0] * 10 + [50.0] * 20, "OLD": [20.0] * 30,
                              "NEW": [np.nan] * 20 + [30.0] * 10}, index=sessions)  # NEW first trades at session 20.
        write_bars(self.root / "bars", opens)
        write_splits(self.root / "splits.parquet", [("A", str(sessions[10].date()), 1, 2)])
        write_dividends(self.root / "dividends.parquet", [("A", str(sessions[10].date()), 0.5)])  # On the split day.
        store = WindowedStore(self.root / "bars", self.root / "news", self.root / "splits.parquet")
        self.panel = Panel.load(store, strict_calendar=False, dividends_path=self.root / "dividends.parquet")
        self.sessions = sessions

    def tearDown(self):
        self.directory.cleanup()

    def test_training_bars_show_each_day_as_it_was_known(self):
        history = self.panel.through(15).history()
        a = history[history.ticker == "A"].set_index("date")
        self.assertEqual(a.close.iloc[:10].unique().tolist(), [100.0])  # As traded before the split.
        self.assertEqual(a.split_factor.iloc[:10].unique().tolist(), [1.0])  # No hint of the coming split.
        self.assertEqual(a.split_factor.iloc[10:].unique().tolist(), [2.0])
        self.assertEqual(history.date.max(), self.sessions[15])
        before, after = a.iloc[9], a.iloc[10]
        total = after.return_factor * after.close / (before.return_factor * before.close) - 1
        self.assertAlmostEqual(total, 0.5 / 50)  # Split-neutral; the dividend is per new share.

    def test_a_cut_panel_does_not_know_later_tickers(self):
        cut = self.panel.through(15)
        self.assertNotIn("NEW", cut.tickers)
        self.assertNotIn("NEW", cut.bars.ticker.cat.categories)
        self.assertIn("NEW", self.panel.through(25).tickers)
        self.assertEqual(cut.forward_returns(10, 2, ["NEW"]).exit_reason["NEW"], "no_entry")


class CorporateEventsTest(unittest.TestCase):
    """Renames, scrip and mis-dated dividends, curated distributions and currencies (the logic audit)."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.sessions = SESSIONS[:60]
        filler = {f"F{i:02d}": [40.0 + i] * 60 for i in range(3)}
        self.opens = pd.DataFrame(filler, index=self.sessions)

    def tearDown(self):
        self.directory.cleanup()

    def load(self, opens, splits=(), dividends=(), actions=None, currency="USD", changes=None):
        write_bars(self.root / "bars", opens)
        if changes is not None:
            write_changes(self.root / "changes.csv", changes)
        write_splits(self.root / "splits.parquet", list(splits))
        write_dividends(self.root / "dividends.parquet", list(dividends), currency=currency)
        if actions is not None:
            write_actions(self.root / "actions.csv", actions)
        store = WindowedStore(self.root / "bars", self.root / "news", self.root / "splits.parquet")
        return Panel.load(store, strict_calendar=False, dividends_path=self.root / "dividends.parquet",
                          actions_path=self.root / "actions.csv" if actions is not None else None,
                          changes_path=self.root / "changes.csv" if changes is not None else None)

    def day(self, i: int) -> str:
        return str(self.sessions[i].date())

    def test_a_rename_continues_the_holding_and_moves_its_events_to_the_old_ticker(self):
        opens = self.opens.copy()
        opens["OLD"] = [20.0] * 30 + [np.nan] * 30  # Renamed at session 30.
        opens["NEW"] = [np.nan] * 30 + [10.2, 10.6, 11.0] + [11.0] * 27  # Moving: the exit open matters.
        # Filed under the new name, though OLD paid it (session 10) and OLD split 2-for-1 (session 20).
        opens.loc[self.sessions[20]:self.sessions[29], "OLD"] = [10.0, 10.5] * 5  # Still moving: not a takeover.
        panel = self.load(opens, splits=[("NEW", self.day(20), 1, 2)], dividends=[("NEW", self.day(10), 0.4)])
        self.assertEqual(panel.renames[["old", "new"]].values.tolist(), [["OLD", "NEW"]])
        self.assertEqual(panel.dividend_audit.set_index("ticker").loc["OLD"].rekeyed_from, "NEW")
        self.assertEqual(panel.split_audit.set_index("ticker").loc["OLD"].rekeyed_from, "NEW")
        self.assertTrue(panel.dividend_audit.set_index("ticker").loc["OLD"].applied)
        returns = panel.forward_returns(27, 4, ["OLD"])  # Entry session 28, exit session 32 (NEW's 22.0).
        self.assertEqual(returns.exit_reason["OLD"], "renamed")
        self.assertAlmostEqual(returns.forward_return["OLD"], 11.0 / 10.0 - 1)
        across_split = panel.forward_returns(17, 4, ["OLD"])  # Entry 18 at 20, exit 22 at 10 after the 2:1.
        self.assertAlmostEqual(across_split.forward_return["OLD"], 0.0)

    def test_an_event_listed_under_both_tickers_counts_once(self):
        opens = self.opens.copy()
        opens["OLD"] = [20.0, 21.0] * 5 + [10.0, 10.5] * 10 + [np.nan] * 30  # 2-for-1 at session 10.
        opens["NEW"] = [np.nan] * 30 + [10.2, 10.6, 11.0] + [11.0] * 27
        listed = [(name, self.day(10), 1, 2) for name in ("OLD", "NEW")]
        panel = self.load(opens, splits=listed, dividends=[("OLD", self.day(5), 0.4), ("NEW", self.day(5), 0.4)])
        self.assertEqual(panel.split_audit.applied.sum(), 1)
        self.assertEqual(panel.multiplier[-1, panel.tickers.get_loc("OLD")], 2.0)
        self.assertEqual(len(panel.dividend_audit), 1)
        self.assertEqual(panel.window(12, 5).query("ticker == 'OLD'").close.tolist(), [10.0, 10.5, 10.0, 10.5, 10.0])

    def test_halts_warrants_and_when_issued_lines_are_not_successors(self):
        opens = self.opens.copy()
        opens["OLD1"] = [20.0, 21.0] * 15 + [np.nan] * 30
        opens["HALTED"] = [20.0] * 25 + [np.nan] * 5 + [21.2] * 30  # Resumes after a short halt.
        opens["OLD2"] = [30.0, 31.0] * 15 + [np.nan] * 30
        opens["ABCDW"] = [np.nan] * 30 + [31.5] * 30  # A warrant line.
        opens["OLD3"] = [40.0, 41.0] * 15 + [np.nan] * 30
        opens["WI"] = [np.nan] * 30 + [41.5] * 10 + [np.nan] * 20  # Trades ten sessions, then stops.
        self.assertEqual(len(self.load(opens).renames), 0)

    def test_curated_changes_force_veto_and_move_events(self):
        opens = self.opens.copy()
        opens["OLD"] = [20.0, 21.0] * 15 + [np.nan] * 30
        opens["NEW"] = [np.nan] * 30 + [21.5] * 30  # A detected rename that is vetoed.
        opens["ORIG"] = [30.0, 31.0] * 15 + [np.nan] * 30
        opens["SUCC"] = [np.nan] * 30 + [45.0] * 30  # Too far in price to detect; forced.
        opens["USED"] = [50.0] * 60  # Trading as another company; its earlier events belong to ORIG.
        panel = self.load(opens, dividends=[("USED", self.day(10), 0.5)],
                          changes=[("OLD", "NEW", self.day(30), -1), ("ORIG", "SUCC", self.day(30), 2),
                                   ("ORIG", "USED", self.day(40), 0)])
        self.assertEqual(panel.renames[["old", "new"]].values.tolist(), [["ORIG", "SUCC"]])
        self.assertEqual(panel.forward_returns(27, 4, ["ORIG"]).exit_reason["ORIG"], "renamed")
        self.assertEqual(panel.dividend_audit[["ticker", "rekeyed_from"]].values.tolist(), [["ORIG", "USED"]])

    def test_share_class_events_reach_the_dotted_ticker(self):
        opens = self.opens.copy()
        opens["BF.B"] = [50.0] * 20 + [49.5] * 40
        panel = self.load(opens, dividends=[("BFB", self.day(20), 0.5)])
        self.assertTrue(panel.dividend_audit.set_index("ticker").loc["BF.B"].applied)
        self.assertAlmostEqual(panel.forward_returns(17, 4, ["BF.B"]).forward_return["BF.B"], 0.0)

    def test_a_renamed_holding_stops_where_the_old_ticker_trades_again(self):
        opens = self.opens.copy()
        # OLD stops after session 29 and is back at 51, as something else, inside the 20 sessions from 32.
        opens["OLD"] = [20.0, 21.0] * 15 + [np.nan] * 21 + [5.0] * 9
        opens["NEW"] = [np.nan] * 32 + [21.2] * 28
        panel = self.load(opens)
        self.assertEqual(panel.renames[["old", "new"]].values.tolist(), [["OLD", "NEW"]])
        code = panel.tickers.get_loc("OLD")
        self.assertTrue(panel.continued[32:51, code].all() and not panel.continued[51:, code].any())
        self.assertTrue(np.isfinite(panel.q_open[51:, code]).all())  # Its own prices, not NEW's.

    def test_a_spin_off_is_valued_only_on_its_own_day(self):
        opens = self.opens.copy()
        opens["PARENT"] = [100.0] * 30 + [80.0] * 30
        opens["SPIN"] = [np.nan] * 32 + [40.0] * 28  # Trades two sessions late.
        panel = self.load(opens, actions=[("PARENT", self.day(30), 0.0, 1.0, "SPIN", 0.5)])
        self.assertEqual(panel.action_audit.reason.tolist(), ["distributed_not_traded"])

    def test_a_cash_deal_pays_cash_per_old_share(self):
        opens = self.opens.copy()
        opens["DEAL"] = [150.0] * 30 + [4.0, 2.0] + [2.0] * 28  # $148 and 0.5 new share; the stub then halves.
        panel = self.load(opens, actions=[("DEAL", self.day(30), 148.0, 0.5, "", 0.0)])
        self.assertAlmostEqual(panel.forward_returns(27, 4, ["DEAL"]).forward_return["DEAL"], (148 + 0.5 * 2) / 150 - 1)
        history = panel.history().query("ticker == 'DEAL'")
        self.assertAlmostEqual(history.return_factor.iloc[30] / history.return_factor.iloc[29], 0.5 + 148 / 4)

    def test_a_different_company_is_not_mistaken_for_a_rename(self):
        opens = self.opens.copy()
        opens["GONE"] = [20.0, 21.0] * 15 + [np.nan] * 30
        opens["IPO"] = [np.nan] * 30 + [35.0] * 30  # Starts the next session at another price.
        opens["TARGET"] = [30.0] * 30 + [np.nan] * 30  # Pinned to a cash deal, then delisted.
        opens["LISTED"] = [np.nan] * 30 + [30.3] * 30  # An unrelated listing at a similar price.
        panel = self.load(opens)
        self.assertEqual(len(panel.renames), 0)
        self.assertEqual(panel.forward_returns(27, 4, ["GONE", "TARGET"]).exit_reason.tolist(),
                         ["last_close", "last_close"])

    def test_scrip_dividends_count_once(self):
        opens = self.opens.copy()
        opens["SCRIP"] = [100.0] * 20 + [97.0] * 40  # 3% paid in shares or cash, listed both ways.
        panel = self.load(opens, splits=[("SCRIP", self.day(20), 100, 103)], dividends=[("SCRIP", self.day(20), 3.0)])
        self.assertEqual(panel.dividend_audit.reason.tolist(), ["paid_in_shares"])
        self.assertAlmostEqual(panel.forward_returns(17, 4, ["SCRIP"]).forward_return["SCRIP"], 97 * 1.03 / 100 - 1)

    def test_large_dividends_must_show_in_the_prices(self):
        opens = self.opens.copy()
        opens["LATE"] = [40.0] * 25 + [30.0] * 35  # A $10 special listed for session 20, paid at 25.
        opens["NONE"] = [40.0] * 60  # A $10 special that never shows: another company's.
        panel = self.load(opens, dividends=[("LATE", self.day(20), 10.0), ("NONE", self.day(20), 10.0)])
        audit = panel.dividend_audit.set_index("ticker")
        self.assertEqual((audit.loc["LATE"].reason, audit.loc["LATE"].applied_on), ("relocated", self.sessions[25]))
        self.assertEqual(audit.loc["NONE"].reason, "not_in_prices")
        self.assertAlmostEqual(panel.forward_returns(22, 4, ["LATE"]).forward_return["LATE"], 0.0)
        self.assertEqual(panel.forward_returns(17, 4, ["NONE"]).forward_return["NONE"], 0.0)

    def test_a_split_that_shows_a_session_late_is_applied_where_it_shows(self):
        opens = self.opens.copy()
        opens["EARLY"] = [30.0] * 21 + [20.0] * 39  # A 3-for-2 listed for session 20, seen at 21.
        panel = self.load(opens, splits=[("EARLY", self.day(20), 2, 3)])
        self.assertEqual(panel.split_audit.set_index("ticker").loc["EARLY"].applied_on, self.sessions[21])
        self.assertAlmostEqual(panel.forward_returns(18, 4, ["EARLY"]).forward_return["EARLY"], 0.0)
        self.assertEqual(panel.window(20, 5).query("ticker == 'EARLY'").close.tolist(), [30.0] * 5)

    def test_curated_spin_offs_and_cash_deals_are_income(self):
        opens = self.opens.copy()
        opens["PARENT"] = [100.0] * 30 + [80.0] * 30  # Spins off 0.5 SPIN per share, worth $20.
        opens["SPIN"] = [np.nan] * 30 + [40.0] * 30
        opens["DEAL"] = [150.0] * 30 + [4.0] * 30  # $148 cash and 0.5 new share; the new company keeps the ticker.
        panel = self.load(opens, actions=[("PARENT", self.day(30), 0.0, 1.0, "SPIN", 0.5),
                                          ("DEAL", self.day(30), 148.0, 0.5, "", 0.0)])
        returns = panel.forward_returns(27, 4, ["PARENT", "DEAL"]).forward_return
        self.assertAlmostEqual(returns["PARENT"], 0.0)
        self.assertAlmostEqual(returns["DEAL"], (148 + 0.5 * 4) / 150 - 1)
        self.assertTrue(panel.action_audit.applied.all())

    def test_foreign_currency_dividends_are_converted(self):
        opens = self.opens.copy()
        opens["FXC"] = [75.0] * 60  # The CAD trust: 0.75 USD per CAD.
        opens["BANK"] = [100.0] * 20 + [99.25] * 40  # Pays 1 CAD = 0.75 USD.
        panel = self.load(opens, dividends=[("BANK", self.day(20), 1.0)], currency="CAD")
        audit = panel.dividend_audit.set_index("ticker").loc["BANK"]
        self.assertAlmostEqual(audit.cash_usd, 0.75)
        self.assertAlmostEqual(panel.forward_returns(17, 4, ["BANK"]).forward_return["BANK"], 0.0)

    def test_dividend_basis_survives_a_split_during_a_halt(self):
        opens = self.opens.copy()
        opens["HALT"] = [10.0] * 20 + [np.nan] * 5 + [1.0] * 35  # Halted; 10-for-1 at session 22; pays at 24.
        panel = self.load(opens, splits=[("HALT", self.day(22), 1, 10)], dividends=[("HALT", self.day(24), 0.01)])
        self.assertAlmostEqual(panel.dividend_audit.set_index("ticker").loc["HALT"]["yield"], 0.01)

    def test_a_cut_panel_owns_its_arrays_and_hides_audits(self):
        opens = self.opens.copy()
        opens["LATER"] = [np.nan] * 40 + [10.0] * 20
        panel = self.load(opens, dividends=[("F00", self.day(30), 0.4)])
        cut = panel.through(25)
        for array in (cut.traded, cut.multiplier, cut.growth, cut.q_open, cut.mark_open, cut.continued, cut.day_starts):
            self.assertIsNone(array.base if array.base is None or array.base.size != array.size else None)
            self.assertLessEqual(len(array), 27)
        self.assertNotIn("LATER", cut.tickers)
        self.assertEqual(cut.multiplier.shape[1], len(cut.tickers))
        self.assertTrue(cut.dividend_audit.empty and cut.split_audit.empty and cut.renames.empty)
        self.assertEqual(cut.sessions[-1], self.sessions[25])


class IsolationTest(Store):
    def test_editing_training_data_in_place_changes_nothing_else(self):
        store, panel, news = load_data(config(self.root))
        data = FoldData(panel, news, Settings(min_price=0, min_dollar_volume=0), panel.sessions[45], 2, 3, 7)
        labels = data.labels()
        labels.loc[:, "forward_return"] = 99.0
        self.assertFalse((data.labels().forward_return == 99.0).any())
        copy = data.news()
        copy.loc[:, "title"] = "EDITED"
        self.assertFalse((news.title == "EDITED").any())
        self.assertFalse((data.news().title == "EDITED").any())
        self.assertIsNone(data._news._is_copy)
        for cells in (news.tickers, data.news().tickers):  # Each list cell owns its storage: no shared buffer.
            self.assertTrue(all(cell.base is None for cell in cells if isinstance(cell, np.ndarray)))

    def test_strategy_names_must_be_unique_and_not_benchmarks(self):
        for changes in ({"ensembles": (("spy", (("flat", 1.0),)),)}, {"ensembles": (("flat", (("flat", 1.0),)),)}):
            with self.assertRaises(ValueError):
                config(self.root, **changes)


class MetricsTest(unittest.TestCase):
    def test_known_curve(self):
        days = pd.bdate_range("2025-01-01", periods=4)
        curve = pd.Series([100.0, 110.0, 99.0, 120.0], index=days)
        drawdown, peak, trough = metrics.max_drawdown(curve)
        self.assertAlmostEqual(drawdown, -0.1)
        self.assertEqual((peak, trough), (days[1], days[2]))
        summary = metrics.performance(curve)
        self.assertAlmostEqual(summary["growth"], 1.2)
        self.assertAlmostEqual(summary["cagr"], 1.2 ** (252 / 3) - 1)

    def test_information_coefficient(self):
        tickers = [f"T{i}" for i in range(30)]
        returns = pd.Series(np.arange(30.0), index=tickers)
        self.assertTrue(np.isnan(metrics.information_coefficient(pd.Series(0.0, index=tickers), returns)))
        self.assertAlmostEqual(metrics.information_coefficient(returns * 2, returns), 1.0)
        self.assertTrue(np.isnan(metrics.information_coefficient(returns[:10], returns)))  # Too few tickers.


if __name__ == "__main__":
    unittest.main()
