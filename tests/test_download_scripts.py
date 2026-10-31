"""Which days the download scripts fetch, with the network stubbed out."""
from contextlib import redirect_stdout
from datetime import date, timedelta
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import pandas as pd
from stock_recommender.config import ROOT


sys.path.insert(0, str(ROOT / "scripts"))  # download_dividends reuses download_splits.get.


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_day(out: Path, day: date, written: pd.Timestamp | None = None):
    """Store an empty file for day, written now or at the given time."""
    path = out / str(day.year) / f"{day}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    if written is not None:
        os.utime(path, (written.timestamp(), written.timestamp()))


class DownloadScriptsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.out = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def fetched(self, script: str, *args: str) -> set[date]:
        """Run the script's main() and return the days it would download."""
        module = load_script(script)
        days = []
        with (mock.patch.object(module, "load_dotenv"), mock.patch.object(module, "start_session"),
              mock.patch.object(module, "download_day", lambda day, out: days.append(day) or 1),
              mock.patch.dict(os.environ, {"POLYGON_API_KEY": "test"}),
              mock.patch.object(sys, "argv", [script, "--out", str(self.out), *args]),
              redirect_stdout(io.StringIO())):
            module.main()
        return set(days)

    def test_news_fetches_missing_and_unfinished_days(self):
        today = pd.Timestamp.now(tz="UTC").date()
        complete, partial, missing = (today - timedelta(days=n) for n in (3, 2, 1))
        write_day(self.out, complete)
        write_day(self.out, partial, written=pd.Timestamp(f"{partial} 20:00", tz="UTC"))  # A 16:00 run's file.
        self.assertEqual(self.fetched("download_news", "--start", str(complete)), {partial, missing})
        self.assertEqual(self.fetched("download_news", "--start", str(complete), "--include-today"),
                         {partial, missing, today})

    def test_bars_fetch_missing_unfinished_and_redownloaded_days(self):
        today = pd.Timestamp.now(tz="America/New_York").date()
        first, partial, last = (day.date() for day in pd.bdate_range(end=today - timedelta(days=1), periods=3))
        write_day(self.out, first)
        write_day(self.out, partial, written=pd.Timestamp(f"{partial} 16:00", tz="America/New_York"))
        write_day(self.out, last)
        self.assertEqual(self.fetched("download_daily_bars", "--start", str(first)), {partial})
        self.assertEqual(self.fetched("download_daily_bars", "--start", str(first), "--redownload", "2"),
                         {partial, last})
        with_today = self.fetched("download_daily_bars", "--start", str(first), "--include-today")
        self.assertEqual(with_today, {partial, today} if today.weekday() < 5 else {partial})

    def test_unadjusted_bars_ask_for_prices_as_traded(self):
        for flags, adjusted in (([], "true"), (["--unadjusted"], "false")):
            module = load_script("download_daily_bars")
            requested = []
            with (mock.patch.object(module, "load_dotenv"), mock.patch.object(module, "start_session"),
                  mock.patch.object(module, "download_day", lambda day, out: requested.append(dict(module.PARAMS)) or 1),
                  mock.patch.dict(os.environ, {"POLYGON_API_KEY": "test"}),
                  mock.patch.object(sys, "argv", ["x", "--out", str(self.out), "--start", "2025-01-06",
                                                  "--end", "2025-01-06", *flags]),
                  redirect_stdout(io.StringIO())):
                module.main()
            self.assertEqual(requested, [{"adjusted": adjusted, "include_otc": "false"}])

    def test_splits_are_paged_and_kept_inside_the_requested_dates(self):
        module = load_script("download_splits")
        pages = [{"results": [{"ticker": "A", "execution_date": "2021-03-01", "split_from": 1, "split_to": 2,
                               "id": "x1"}], "next_url": "https://api.polygon.io/v3/reference/splits?cursor=2"},
                 {"results": [{"ticker": "B", "execution_date": "2025-12-31", "split_from": 10, "split_to": 1,
                               "id": "x2"}]}]
        calls = []

        def get(session, url, params=None):
            calls.append((url, params))
            return pages[len(calls) - 1]

        out = self.out / "splits.parquet"
        argv = ["x", "--start", "2021-01-01", "--end", "2025-12-31", "--out", str(out)]
        with (mock.patch.object(module, "get", get), mock.patch.object(module, "load_dotenv"),
              mock.patch.dict(os.environ, {"POLYGON_API_KEY": "test"}), mock.patch.object(sys, "argv", argv),
              redirect_stdout(io.StringIO())):
            module.main()
        self.assertEqual(calls[0][1]["execution_date.lte"], "2025-12-31")
        self.assertEqual(calls[1], ("https://api.polygon.io/v3/reference/splits?cursor=2", None))
        self.assertEqual(list(pd.read_parquet(out).ticker), ["A", "B"])
        pages[1]["results"][0]["execution_date"] = "2026-01-05"  # Polygon misbehaving: nothing is written.
        calls.clear()
        out.unlink()
        with (mock.patch.object(module, "get", get), mock.patch.object(module, "load_dotenv"),
              mock.patch.dict(os.environ, {"POLYGON_API_KEY": "test"}), mock.patch.object(sys, "argv", argv)):
            with self.assertRaisesRegex(SystemExit, "outside"):
                module.main()
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()


class DividendsScriptTest(unittest.TestCase):
    def test_dividends_are_kept_inside_the_requested_dates(self):
        module = load_script("download_dividends")
        page = {"results": [{"ticker": "A", "ex_dividend_date": "2025-12-31", "cash_amount": 0.5, "currency": "USD",
                             "dividend_type": "CD", "id": "d1"}]}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "dividends.parquet"
            argv = ["x", "--start", "2021-01-01", "--end", "2025-12-31", "--out", str(out)]
            patches = lambda: (mock.patch.object(module, "get", lambda session, url, params=None: page),
                               mock.patch.object(module, "load_dotenv"),
                               mock.patch.dict(os.environ, {"POLYGON_API_KEY": "test"}),
                               mock.patch.object(sys, "argv", argv))
            first = patches()
            with first[0], first[1], first[2], first[3], redirect_stdout(io.StringIO()):
                module.main()
            self.assertEqual(pd.read_parquet(out).cash_amount.tolist(), [0.5])
            page["results"][0]["ex_dividend_date"] = "2026-01-02"
            out.unlink()
            second = patches()
            with second[0], second[1], second[2], second[3]:
                with self.assertRaisesRegex(SystemExit, "outside"):
                    module.main()
            self.assertFalse(out.exists())
