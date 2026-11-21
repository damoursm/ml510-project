from dataclasses import replace
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from zoneinfo import ZoneInfo
from stock_recommender.api import LatestRecommendations, RunRecommendations, create_app
from stock_recommender.config import ROOT, Settings
from stock_recommender.models import load_models
from stock_recommender.refresh import Refresh
from .test_served_models import make_run
from .test_stock_recommender import TestClient, write_bars

NEW_YORK = ZoneInfo("America/New_York")


def new_york(*args) -> datetime:
    return datetime(*args, tzinfo=NEW_YORK)


class RefreshTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.settings = Settings(bars_dir=root / "bars", news_dir=root / "news", sentiment_dir=root / "sentiment",
                                 min_dollar_volume=0.0, score_news=False)
        write_bars(self.settings.bars_dir, "2026-10-16", {"BIG": (100, 1_000)})
        self.latest = LatestRecommendations(self.settings, load_models())
        self.refresh = Refresh(self.settings, self.latest)

    def tearDown(self):
        self.directory.cleanup()

    def script(self, code: str) -> list[str]:
        return [sys.executable, "-c", code]

    def as_of(self) -> str:
        return str(self.latest.get()[0].as_of.date())

    def test_commands_fetch_today_only_after_the_close(self):
        scoring = Refresh(replace(self.settings, score_news=True), self.latest)
        bars, news, score = scoring.commands(new_york(2026, 10, 23, 16, 0))
        self.assertEqual(bars[1:], [str(ROOT / "scripts/download_daily_bars.py"), "--out", str(self.settings.bars_dir),
                                    "--start", "2026-04-26", "--redownload", "90", "--include-today"])
        self.assertEqual(news[1:], [str(ROOT / "scripts/download_news.py"), "--out", str(self.settings.news_dir),
                                    "--start", "2026-04-26", "--include-today"])
        # Then the LLM scores the articles of the news window, by UTC day (16:00 New York is 20:00 UTC).
        self.assertEqual(score[1:], [str(ROOT / "scripts/score_news.py"), "--news-dir", str(self.settings.news_dir),
                                     "--out", str(self.settings.sentiment_dir), "--start", "2026-10-13",
                                     "--end", "2026-10-23", "--url", "http://localhost:11434", "--model", "qwen3:8b",
                                     "--max-tickers", "5", "--workers", "4"])
        self.assertEqual(len(self.refresh.commands(new_york(2026, 10, 23, 16, 0))), 2)  # score_news off.
        # A run that fires mid-session, e.g. on waking from sleep, must not store the session so far.
        for command in self.refresh.commands(new_york(2026, 10, 26, 10, 30)):
            self.assertNotIn("--include-today", command)

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

    def test_the_api_runs_nothing_on_its_own(self):
        with mock.patch("stock_recommender.refresh.subprocess.run") as run:
            with TestClient(create_app(self.settings, load_models())) as client:  # Runs the lifespan.
                self.assertEqual(client.get("/health").json()["refresh"], {"last_run": None})
        run.assert_not_called()  # Only a POST with fetch=true downloads: the 16:00 email task's.

    def test_post_with_fetch_downloads_before_running_the_models(self):
        downloads = []

        def run(command, **kwargs):
            downloads.append(Path(command[1]).name)
            write_bars(self.settings.bars_dir, "2026-10-19", {"BIG": (100, 1_000)})
            return subprocess.CompletedProcess(command, 0, "done", "")

        client = TestClient(create_app(self.settings, load_models()))
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
            closed.set()  # The 16:00 email task's fetch arrives during a 15:59 one.
            self.refresh.run_once()
            before.join()
        self.assertEqual(downloads, [False, False, True, True])
        self.assertTrue(self.refresh.last_run["included_today"])

    def test_post_reports_a_failed_fetch(self):
        def run(command, **kwargs):
            return subprocess.CompletedProcess(command, 1, "", "POLYGON_API_KEY is missing")

        client = TestClient(create_app(self.settings, load_models()))
        with mock.patch("stock_recommender.refresh.subprocess.run", run), self.assertLogs("uvicorn.error", "ERROR"):
            body = client.post("/recommendations/latest", params={"fetch": True}).json()
        self.assertEqual(body["as_of"], "2026-10-16")  # Still served from the data on disk.
        self.assertEqual(len(body["fetch"]["errors"]), 2)
        self.assertIn("POLYGON_API_KEY is missing", body["fetch"]["errors"][0])
        # With no data to rank, the 503 says why the fetch did not help.
        empty = replace(self.settings, bars_dir=self.settings.bars_dir / "empty")
        client = TestClient(create_app(empty, load_models()))
        with mock.patch("stock_recommender.refresh.subprocess.run", run), self.assertLogs("uvicorn.error", "ERROR"):
            response = client.post("/recommendations/latest", params={"fetch": True})
        self.assertEqual(response.status_code, 503)
        self.assertIn("POLYGON_API_KEY is missing", response.json()["detail"])

class RunDownloadsTest(unittest.TestCase):
    def test_the_downloads_cover_the_widest_window_of_the_runs_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wide = replace(load_models(), version="wide", bars_lookback=120, news_lookback_days=9)
            folder = make_run(root, {"plain": load_models(), "wide": wide})
            settings = Settings(bars_dir=root / "bars", news_dir=root / "news", sentiment_dir=root / "sentiment",
                                score_news=True, model_run=str(folder))
            refresh = Refresh(settings, RunRecommendations(settings, load_models()))
            bars, _, score = refresh.commands(new_york(2026, 10, 23, 16, 0))
            self.assertEqual(bars[bars.index("--redownload") + 1], "180")
            self.assertEqual(score[score.index("--start") + 1], "2026-10-11")  # 9 days of news, plus 3.


if __name__ == "__main__":
    unittest.main()
