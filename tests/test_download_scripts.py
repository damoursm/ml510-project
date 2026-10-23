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


if __name__ == "__main__":
    unittest.main()
