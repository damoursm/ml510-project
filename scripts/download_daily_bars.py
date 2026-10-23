"""Download Polygon daily bars for all US stocks over all history into data/raw.

Uses the grouped daily endpoint: one request per weekday returns every ticker
that traded that day, so delisted names are included. Bars are split-adjusted
(not dividend-adjusted) and exclude OTC tickers. Each trading day becomes one
parquet file; existing files are skipped, so re-running resumes an interrupted
download or adds the days since the last run. Polygon adjusts for the splits it
knows at request time, so use --redownload to put recent days back on one basis.
--include-today stores today's bars so far; any later run fetches them again,
since a file written before its day ended is not final. Reads POLYGON_API_KEY
from .env.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
import os
from pathlib import Path
import threading
import time
from dotenv import load_dotenv
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{day}"
PARAMS = {"adjusted": "true", "include_otc": "false"}
FIRST_DAY = date(2003, 9, 10)  # Polygon has no stock bars before this day.
COLUMNS = {"T": "ticker", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap", "n": "transactions"}
# Fixed dtypes give every file one schema, even on days with integer-only volumes or no vwap.
DTYPES = {"date": "datetime64[ns]", "open": "float64", "high": "float64", "low": "float64", "close": "float64",
          "volume": "float64", "vwap": "float64", "transactions": "Int64"}
_thread = threading.local()


def start_session(key: str) -> None:
    _thread.session = requests.Session()
    _thread.session.headers["Authorization"] = f"Bearer {key}"


def fetch(day: date, attempts: int = 5) -> list[dict]:
    """Return the day's bars; an empty list means the market was closed."""
    for attempt in range(attempts):
        if attempt:
            time.sleep(2 ** attempt)
        try:
            response = _thread.session.get(URL.format(day=day), params=PARAMS, timeout=60)
            if response.status_code == 200:
                body = response.json()
                if body.get("status") in ("OK", "DELAYED"):
                    return body.get("results") or []
        except requests.RequestException as exc:
            error = repr(exc)
            continue
        error = f"HTTP {response.status_code}: {response.text[:300]}"
        if response.status_code != 429 and response.status_code < 500:
            break  # Bad key or plan limit: retrying will not help.
    raise RuntimeError(f"Could not download {day}: {error}")


def day_path(out: Path, day: date) -> Path:
    return out / str(day.year) / f"{day}.parquet"


def complete(path: Path, day: date) -> bool:
    """Whether the day's file exists and was written after the day ended (New York), so its bars are final."""
    return path.exists() and path.stat().st_mtime >= pd.Timestamp(day + timedelta(days=1), tz="America/New_York").timestamp()


def replace(source: Path, target: Path, attempts: int = 10) -> None:
    """os.replace, retried while a reader such as the recommendations API holds target open on Windows."""
    for attempt in range(attempts):
        try:
            return os.replace(source, target)
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.5)


def download_day(day: date, out: Path) -> int:
    results = fetch(day)
    if not results:
        return 0
    frame = pd.DataFrame(results).reindex(columns=list(COLUMNS)).rename(columns=COLUMNS)
    frame.insert(0, "date", pd.Timestamp(day))
    frame = frame.astype(DTYPES)
    path = day_path(out, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")  # Parquet readers skip dot-files.
    frame.sort_values("ticker").to_parquet(temporary, index=False, compression="zstd")
    replace(temporary, path)
    return len(frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=FIRST_DAY, help="First day, YYYY-MM-DD.")
    parser.add_argument("--end", type=date.fromisoformat,
                        help="Last day, YYYY-MM-DD; at most yesterday (US/Eastern), the default, or today with --include-today.")
    parser.add_argument("--out", type=Path, default=ROOT / "data/raw/polygon/daily_bars")
    parser.add_argument("--workers", type=int, default=8, help="Parallel requests.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download days that already have a file.")
    parser.add_argument("--redownload", type=int, default=0, metavar="DAYS",
                        help="Also re-download the last DAYS weekdays, so they share one split adjustment.")
    parser.add_argument("--include-today", action="store_true", help="Also store today's bars so far.")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    key = os.environ.get("POLYGON_API_KEY", "").strip()
    if not key:
        raise SystemExit("POLYGON_API_KEY is missing from .env and the environment.")
    # Today's bars are still forming until the session ends.
    last = (pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(days=0 if args.include_today else 1)).date()
    end = min(args.end or last, last)
    days = [d.date() for d in pd.bdate_range(args.start, end)]
    recent = set(days[max(len(days) - args.redownload, 0):])
    todo = [d for d in days if args.overwrite or d in recent or not complete(day_path(args.out, d), d)]
    print(f"{len(days)} weekdays from {args.start} to {end}; downloading {len(todo)} into {args.out}", flush=True)
    stored = bars = 0
    started = time.monotonic()
    with ThreadPoolExecutor(args.workers, initializer=start_session, initargs=(key,)) as pool:
        futures = [pool.submit(download_day, day, args.out) for day in todo]
        try:
            for done, future in enumerate(as_completed(futures), 1):
                count = future.result()
                stored += count > 0
                bars += count
                if done % 250 == 0 or done == len(todo):
                    print(f"{done}/{len(todo)} days, {bars:,} bars, {time.monotonic() - started:.0f}s", flush=True)
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise
    print(f"Stored {stored} trading days ({bars:,} bars); {len(todo) - stored} weekdays had no bars (market holidays).")


if __name__ == "__main__":
    main()
