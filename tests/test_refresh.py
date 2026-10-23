from dataclasses import replace
from datetime import datetime, timedelta
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from zoneinfo import ZoneInfo
from stock_recommender.api import LatestRecommendations, create_app
from stock_recommender.config import ROOT, Settings
from stock_recommender.models import load_models
from stock_recommender.refresh import DailyRefresh, next_run, previous_run
from .test_stock_recommender import TestClient, write_bars

NEW_YORK = ZoneInfo("America/New_York")


def new_york(*args) -> datetime:
    return datetime(*args, tzinfo=NEW_YORK)


class ScheduleTest(unittest.TestCase):
    def test_next_run_later_today(self):
        self.assertEqual(next_run(new_york(2026, 10, 23, 12, 0), "16:00"), new_york(2026, 10, 23, 16, 0))

    def test_next_run_on_friday_at_or_after_the_time_is_monday(self):
        self.assertEqual(next_run(new_york(2026, 10, 23, 16, 0), "16:00"), new_york(2026, 10, 26, 16, 0))
        self.assertEqual(next_run(new_york(2026, 10, 24, 9, 0), "16:00"), new_york(2026, 10, 26, 16, 0))

    def test_next_run_time_of_day_holds_across_daylight_saving_end(self):
        friday = new_york(2026, 10, 30, 17, 0)  # EDT; the clocks fall back on Sunday 2026-11-01.
        run = next_run(friday, "16:00")
        self.assertEqual((run.hour, run.utcoffset()), (16, timedelta(hours=-5)))
        self.assertEqual(run.timestamp() - friday.timestamp(), 72 * 3600)

    def test_previous_run(self):
        self.assertEqual(previous_run(new_york(2026, 10, 23, 16, 0), "16:00"), new_york(2026, 10, 23, 16, 0))
        self.assertEqual(previous_run(new_york(2026, 10, 26, 10, 30), "16:00"), new_york(2026, 10, 23, 16, 0))
        self.assertEqual(previous_run(new_york(2026, 10, 25, 20, 0), "16:00"), new_york(2026, 10, 23, 16, 0))


class DailyRefreshTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.settings = Settings(bars_dir=root / "bars", news_dir=root / "news", min_dollar_volume=0.0)
        write_bars(self.settings.bars_dir, "2026-10-16", {"BIG": (100, 1_000)})
        self.latest = LatestRecommendations(self.settings, load_models())
        self.refresh = DailyRefresh(self.settings, self.latest)

    def tearDown(self):
        self.directory.cleanup()

    def script(self, code: str) -> list[str]:
        return [sys.executable, "-c", code]

    def as_of(self) -> str:
        return str(self.latest.get()[0].as_of.date())

    def test_commands_fetch_today_only_after_the_close(self):
        bars, news = self.refresh.commands(new_york(2026, 10, 23, 16, 0))
        self.assertEqual(bars[1:], [str(ROOT / "scripts/download_daily_bars.py"), "--out", str(self.settings.bars_dir),
                                    "--start", "2026-04-26", "--redownload", "90", "--include-today"])
        self.assertEqual(news[1:], [str(ROOT / "scripts/download_news.py"), "--out", str(self.settings.news_dir),
                                    "--start", "2026-04-26", "--include-today"])
        # A run that fires mid-session, e.g. on waking from sleep, must not store the session so far.
        for command in self.refresh.commands(new_york(2026, 10, 26, 10, 30)):
            self.assertNotIn("--include-today", command)

    def test_missed_run_is_detected_from_the_bars_files(self):
        monday_morning = new_york(2026, 10, 26, 10, 30)  # The last run was due Friday at 16:00.
        newest = self.settings.bars_dir / "2026/2026-10-16.parquet"
        for written, missed in ((new_york(2026, 10, 23, 15, 0), True), (new_york(2026, 10, 23, 16, 1), False)):
            os.utime(newest, (written.timestamp(), written.timestamp()))
            self.assertEqual(self.refresh.missed_run(monday_morning), missed)
        newest.unlink()
        self.assertTrue(self.refresh.missed_run(monday_morning))

    def test_run_recomputes_the_recommendations_from_the_new_files(self):
        self.assertEqual(self.as_of(), "2026-10-16")
        bars_dir = str(self.settings.bars_dir).replace("\\", "/")
        self.refresh.commands = lambda now: [self.script(
            "import pandas as pd; pd.DataFrame({'date': [pd.Timestamp('2026-10-19')], 'ticker': ['BIG'], "
            "'close': [100.0], 'volume': [1000.0]}).to_parquet(" f"'{bars_dir}/2026/2026-10-19.parquet')")]
        self.refresh.run_once()
        self.assertEqual(self.refresh.last_run["errors"], [])
        self.assertEqual(str(self.latest._result[0].as_of.date()), "2026-10-19")  # Recomputed before any request.

    def test_failed_download_is_reported_and_the_other_still_runs(self):
        self.refresh.commands = lambda now: [self.script("import sys; print('no key'); sys.exit(3)"),
                                             self.script("print('news done')")]
        with self.assertLogs("uvicorn.error", "ERROR"):
            self.refresh.run_once()
        self.assertEqual(self.refresh.last_run["errors"], ["-c exited with 3: no key"])
        self.assertEqual(self.as_of(), "2026-10-16")

    def test_downloads_run_while_requests_get_the_previous_recommendations(self):
        self.assertEqual(self.as_of(), "2026-10-16")
        seen = []

        def run(command, **kwargs):
            write_bars(self.settings.bars_dir, "2026-10-19", {"BIG": (100, 1_000)})
            seen.append((self.latest.refreshing.is_set(), self.as_of()))
            return subprocess.CompletedProcess(command, 0, "done", "")

        with mock.patch("stock_recommender.refresh.subprocess.run", run):
            self.refresh.run_once()
        self.assertEqual(seen, [(True, "2026-10-16")] * 2)
        self.assertFalse(self.latest.refreshing.is_set())
        self.assertEqual(self.as_of(), "2026-10-19")

    def test_app_runs_the_schedule_until_shutdown_even_after_a_failed_run(self):
        calls = []

        def run_once(refresh):
            calls.append(refresh)
            if len(calls) == 1:
                raise RuntimeError("Polygon is down")

        with (mock.patch("stock_recommender.refresh.next_run", lambda now, at: now + timedelta(seconds=0.05)),
              mock.patch.object(DailyRefresh, "run_once", run_once),
              self.assertLogs("uvicorn.error", "ERROR")):
            with TestClient(create_app(self.settings, load_models())):  # Runs the lifespan: start, then stop.
                deadline = time.monotonic() + 5
                while len(calls) < 2 and time.monotonic() < deadline:
                    time.sleep(0.05)
        self.assertGreaterEqual(len(calls), 2)
        self.assertNotIn("daily-refresh", [thread.name for thread in threading.enumerate()])

    def test_app_catches_up_a_missed_run_at_startup(self):
        calls = []
        old = new_york(2026, 10, 1, 12, 0).timestamp()  # Written long before the last scheduled run.
        os.utime(self.settings.bars_dir / "2026/2026-10-16.parquet", (old, old))
        with mock.patch.object(DailyRefresh, "run_once", lambda refresh: calls.append(refresh)):
            with TestClient(create_app(self.settings, load_models())):
                deadline = time.monotonic() + 5
                while not calls and time.monotonic() < deadline:
                    time.sleep(0.05)
        self.assertEqual(len(calls), 1)

    def test_health_reports_the_schedule(self):
        with TestClient(create_app(self.settings, load_models())) as client:
            refresh = client.get("/health").json()["refresh"]
        self.assertEqual(refresh["schedule"], "weekdays at 16:00 America/New_York")
        self.assertEqual(datetime.fromisoformat(refresh["next_run"]).astimezone(NEW_YORK).hour, 16)
        self.assertIsNone(refresh["last_run"])
        for off in ("off", ""):
            with TestClient(create_app(replace(self.settings, refresh_time=off), load_models())) as client:
                self.assertIsNone(client.get("/health").json()["refresh"]["schedule"])
            self.assertNotIn("daily-refresh", [thread.name for thread in threading.enumerate()])

    def test_post_with_fetch_downloads_before_running_the_models(self):
        downloads = []

        def run(command, **kwargs):
            downloads.append(Path(command[1]).name)
            write_bars(self.settings.bars_dir, "2026-10-19", {"BIG": (100, 1_000)})
            return subprocess.CompletedProcess(command, 0, "done", "")

        client = TestClient(create_app(replace(self.settings, refresh_time="off"), load_models()))
        with mock.patch("stock_recommender.refresh.subprocess.run", run):
            self.assertEqual(client.get("/recommendations/latest").json()["as_of"], "2026-10-16")
            self.assertEqual(client.post("/recommendations/latest").json()["as_of"], "2026-10-16")
            self.assertEqual(downloads, [])
            body = client.post("/recommendations/latest", params={"fetch": True}).json()
        self.assertEqual(downloads, ["download_daily_bars.py", "download_news.py"])
        self.assertEqual((body["as_of"], body["fetch"]), ("2026-10-19", {"errors": []}))

    def test_a_fetch_during_another_waits_for_it_instead_of_downloading_again(self):
        downloads = []

        def run(command, **kwargs):
            downloads.append(command)
            time.sleep(0.2)
            return subprocess.CompletedProcess(command, 1, "", "Polygon is down")

        with mock.patch("stock_recommender.refresh.subprocess.run", run), self.assertLogs("uvicorn.error", "ERROR"):
            results = []
            threads = [threading.Thread(target=lambda: results.append(self.refresh.run_once())) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(len(downloads), 2)  # One run of the two scripts.
        self.assertEqual([len(errors) for errors in results], [2, 2, 2])

    def test_a_call_after_the_close_does_not_reuse_a_run_that_started_before(self):
        closed = threading.Event()  # Stands in for the clock passing 16:00 New York.
        downloads = []

        def run(command, **kwargs):
            downloads.append("--include-today" in command)
            time.sleep(0.2)
            return subprocess.CompletedProcess(command, 0, "done", "")

        with (mock.patch("stock_recommender.refresh.after_close", lambda now: closed.is_set()),
              mock.patch("stock_recommender.refresh.subprocess.run", run)):
            before = threading.Thread(target=self.refresh.run_once)
            before.start()
            time.sleep(0.1)
            closed.set()  # The scheduled 16:00 run arrives during the 15:59 one.
            self.refresh.run_once()
            before.join()
        self.assertEqual(downloads, [False, False, True, True])
        self.assertTrue(self.refresh.last_run["included_today"])

    def test_schedule_counts_from_the_start_of_the_last_run(self):
        # A catch-up run that crosses the refresh time must not skip that day's run.
        run_starts, scheduled_from = [], []
        old = new_york(2026, 10, 1, 12, 0).timestamp()
        os.utime(self.settings.bars_dir / "2026/2026-10-16.parquet", (old, old))

        def run_once(refresh):
            run_starts.append(datetime.now(NEW_YORK))
            time.sleep(0.2)

        def schedule(now, at):
            scheduled_from.append(now)
            return now + timedelta(days=1)

        with (mock.patch("stock_recommender.refresh.next_run", schedule),
              mock.patch.object(DailyRefresh, "run_once", run_once)):
            with TestClient(create_app(self.settings, load_models())):
                deadline = time.monotonic() + 5
                while len(scheduled_from) < 2 and time.monotonic() < deadline:  # Once at startup, once in the loop.
                    time.sleep(0.05)
        self.assertEqual(len(run_starts), 1)
        self.assertLessEqual(scheduled_from[-1], run_starts[0])

    def test_post_reports_a_failed_fetch(self):
        def run(command, **kwargs):
            return subprocess.CompletedProcess(command, 1, "", "POLYGON_API_KEY is missing")

        client = TestClient(create_app(replace(self.settings, refresh_time="off"), load_models()))
        with mock.patch("stock_recommender.refresh.subprocess.run", run), self.assertLogs("uvicorn.error", "ERROR"):
            body = client.post("/recommendations/latest", params={"fetch": True}).json()
        self.assertEqual(body["as_of"], "2026-10-16")  # Still served from the data on disk.
        self.assertEqual(len(body["fetch"]["errors"]), 2)
        self.assertIn("POLYGON_API_KEY is missing", body["fetch"]["errors"][0])
        # With no data to rank, the 503 says why the fetch did not help.
        empty = replace(self.settings, refresh_time="off", bars_dir=self.settings.bars_dir / "empty")
        client = TestClient(create_app(empty, load_models()))
        with mock.patch("stock_recommender.refresh.subprocess.run", run), self.assertLogs("uvicorn.error", "ERROR"):
            response = client.post("/recommendations/latest", params={"fetch": True})
        self.assertEqual(response.status_code, 503)
        self.assertIn("POLYGON_API_KEY is missing", response.json()["detail"])

    def test_malformed_time_fails_at_startup(self):
        with self.assertRaises(ValueError):
            DailyRefresh(replace(self.settings, refresh_time="4pm"), self.latest)


if __name__ == "__main__":
    unittest.main()
