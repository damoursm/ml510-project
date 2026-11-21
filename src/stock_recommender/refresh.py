"""Download new bars and news when a request asks to fetch them (POST /recommendations/latest?fetch=true).

The API runs nothing on a schedule: the 16:00 email task (scripts/email_recommendations.py) calls it with fetch=true.
"""
from datetime import datetime, timedelta, timezone
import logging
import math
from pathlib import Path
import subprocess
import sys
import threading
from zoneinfo import ZoneInfo
from .config import ROOT, Settings

log = logging.getLogger("uvicorn.error")  # Uvicorn's general logger, which its console handler prints.
NEW_YORK = ZoneInfo("America/New_York")


def after_close(now: datetime) -> bool:
    """Whether the day's 16:00 New York close has passed, so its bars and news can be stored."""
    return now.astimezone(NEW_YORK).hour >= 16


class Refresh:
    """Runs the download scripts, then recomputes the recommendations. The API keeps serving the previous
    recommendations while the scripts rewrite the files; status() reports the last run for /health."""

    def __init__(self, settings: Settings, latest):
        self.settings = settings
        self.latest = latest
        self.last_run: dict | None = None
        self._running = threading.Lock()

    def commands(self, now: datetime) -> list[list[str]]:
        # Weekdays covering the price model's window plus holidays, rewritten onto one split adjustment.
        bars_days = math.ceil(self.latest.models.bars_lookback * 1.5)
        start = str(now.date() - timedelta(days=2 * bars_days))
        # Today's data is stored once the 16:00 New York close has passed. A run that starts earlier, e.g. on
        # request or on waking from sleep, stops at the last complete day. The scripts fetch again a day stored
        # before it ended.
        today = ["--include-today"] if after_close(now) else []
        commands = [
            [sys.executable, str(ROOT / "scripts/download_daily_bars.py"), "--out", str(self.settings.bars_dir),
             "--start", start, "--redownload", str(bars_days), *today],
            [sys.executable, str(ROOT / "scripts/download_news.py"), "--out", str(self.settings.news_dir),
             "--start", start, *today],
        ]
        if self.settings.score_news:
            # The days of news the models read and a few more, by UTC day like the news files: the articles
            # fetched since the last run are scored, the others already are.
            utc_today = now.astimezone(timezone.utc).date()
            first = utc_today - timedelta(days=self.latest.models.news_lookback_days + 3)
            commands.append([
                sys.executable, str(ROOT / "scripts/score_news.py"), "--news-dir", str(self.settings.news_dir),
                "--out", str(self.settings.sentiment_dir), "--start", str(first), "--end", str(utc_today),
                "--url", self.settings.ollama_url, "--model", self.settings.ollama_model,
                "--max-tickers", str(self.settings.sentiment_max_tickers),
                "--workers", str(self.settings.sentiment_workers)])
        return commands

    def run_once(self) -> list[str]:
        """Download, recompute, and return the errors.

        The scripts must not write the same files at once, so a call during another run waits for it, and
        reuses it unless that run started before the close and this call comes after.
        """
        waited = not self._running.acquire(blocking=False)
        if waited:
            self._running.acquire()
        try:
            if waited and self.last_run and (self.last_run["included_today"] or not after_close(datetime.now(NEW_YORK))):
                return self.last_run["errors"]
            return self._run()
        finally:
            self._running.release()

    def _run(self) -> list[str]:
        now = datetime.now(NEW_YORK)
        included_today = after_close(now)
        errors = []
        self.latest.refreshing.set()
        try:
            for command in self.commands(now):
                script = Path(command[1]).name
                try:
                    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, errors="replace",
                                          timeout=3600)
                except (OSError, subprocess.TimeoutExpired) as error:
                    errors.append(f"{script}: {error}")
                    continue
                output = (done.stdout + done.stderr).strip()
                if done.returncode:
                    errors.append(f"{script} exited with {done.returncode}: {output[-1500:]}")
                else:
                    log.info("Refresh, %s: %s", script, output.splitlines()[-1] if output else "done")
        finally:
            self.latest.refreshing.clear()
        try:
            self.latest.get()  # Recompute now, so the first request after the refresh is fast.
        except Exception as error:  # DataUnavailable, or a model failing on the new data.
            errors.append(f"recommendations: {error}")
        for error in errors:
            log.error("Refresh failed: %s", error)
        self.last_run = {"started": now.astimezone(timezone.utc), "finished": datetime.now(timezone.utc),
                         "included_today": included_today, "errors": errors}
        return errors

    def status(self) -> dict:
        return {"last_run": self.last_run}
