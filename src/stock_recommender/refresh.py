"""Download new bars and news, on request and every weekday at a set time, inside the API process."""
from datetime import datetime, timedelta, timezone
import logging
import math
from pathlib import Path
import subprocess
import sys
import threading
import time
from zoneinfo import ZoneInfo
from .config import ROOT, Settings
from .data import recent_day_files

log = logging.getLogger("uvicorn.error")  # Uvicorn's general logger, which its console handler prints.
NEW_YORK = ZoneInfo("America/New_York")


def clock(at: str) -> tuple[int, int]:
    hour, minute = map(int, at.split(":"))
    return hour, minute


def next_run(now: datetime, at: str) -> datetime:
    """Return the first weekday at `at` (HH:MM) strictly after now, in now's time zone."""
    hour, minute = clock(at)
    run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if run <= now:
        run += timedelta(days=1)  # Wall-clock arithmetic, so the time of day holds across DST changes.
    while run.weekday() >= 5:
        run += timedelta(days=1)
    return run


def after_close(now: datetime) -> bool:
    """Whether the day's 16:00 New York close has passed, so its bars and news can be stored."""
    return now.astimezone(NEW_YORK).hour >= 16


def previous_run(now: datetime, at: str) -> datetime:
    """Return the last weekday at `at` (HH:MM) at or before now, in now's time zone."""
    hour, minute = clock(at)
    run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if run > now:
        run -= timedelta(days=1)
    while run.weekday() >= 5:
        run -= timedelta(days=1)
    return run


class DailyRefresh:
    """Runs the download scripts, then recomputes the recommendations: on request (run_once), and each weekday
    at settings.refresh_time unless it is "off".

    A scheduled run missed while the API was off happens at startup. The API keeps serving the previous
    recommendations while the scripts rewrite the files.
    """

    def __init__(self, settings: Settings, latest):
        self.settings = settings
        self.latest = latest
        self.zone = ZoneInfo(settings.refresh_timezone)
        self.scheduled = settings.refresh_time.lower() not in ("", "off")
        if self.scheduled:
            next_run(datetime.now(self.zone), settings.refresh_time)  # Reject a malformed time at startup.
        self.last_run: dict | None = None
        self._running = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="daily-refresh", daemon=True)

    def commands(self, now: datetime) -> list[list[str]]:
        # Weekdays covering the price model's window plus holidays, rewritten onto one split adjustment.
        bars_days = math.ceil(self.latest.models.bars_lookback * 1.5)
        start = str(now.date() - timedelta(days=2 * bars_days))
        # Today's data is stored once the 16:00 New York close has passed. A run that starts earlier, e.g. on
        # request or on waking from sleep, stops at the last complete day. The scripts fetch again a day stored
        # before it ended.
        today = ["--include-today"] if after_close(now) else []
        return [
            [sys.executable, str(ROOT / "scripts/download_daily_bars.py"), "--out", str(self.settings.bars_dir),
             "--start", start, "--redownload", str(bars_days), *today],
            [sys.executable, str(ROOT / "scripts/download_news.py"), "--out", str(self.settings.news_dir),
             "--start", start, *today],
        ]

    def missed_run(self, now: datetime) -> bool:
        """Whether the bars were not rewritten since the last scheduled run, e.g. because the API was off."""
        newest = recent_day_files(self.settings.bars_dir, 1)
        due = previous_run(now, self.settings.refresh_time)
        return not newest or newest[0].stat().st_mtime < due.timestamp()

    def run_once(self) -> list[str]:
        """Download, recompute, and return the errors.

        The scripts must not write the same files at once, so a call during another run waits for it, and
        reuses it unless that run started before the close and this call comes after.
        """
        waited = not self._running.acquire(blocking=False)
        if waited:
            self._running.acquire()
        try:
            if waited and self.last_run and (self.last_run["included_today"] or not after_close(datetime.now(self.zone))):
                return self.last_run["errors"]
            return self._run()
        finally:
            self._running.release()

    def _run(self) -> list[str]:
        now = datetime.now(self.zone)
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

    def _run_logged(self) -> None:
        try:
            self.run_once()
        except Exception:  # Keep the schedule alive for the next day.
            log.exception("Refresh failed")

    def _loop(self) -> None:
        started = datetime.now(self.zone)
        if self.missed_run(started):
            self._run_logged()
        while True:
            # Scheduled from when the last run started, so a run that crosses the refresh time is followed by one.
            run = next_run(started, self.settings.refresh_time)
            # Short waits against the wall clock stay on time across sleep or hibernation.
            while (remaining := run.timestamp() - time.time()) > 0:
                if self._stop.wait(min(remaining, 60)):
                    return
            started = datetime.now(self.zone)
            self._run_logged()

    def start(self) -> None:
        if self.scheduled:
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def status(self) -> dict:
        if not self.scheduled:
            return {"schedule": None, "next_run": None, "last_run": self.last_run}
        return {"schedule": f"weekdays at {self.settings.refresh_time} {self.settings.refresh_timezone}",
                "next_run": next_run(datetime.now(self.zone), self.settings.refresh_time),
                "last_run": self.last_run}
