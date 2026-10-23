"""Download Polygon news articles for US stocks over all history into data/raw.

Uses the ticker news endpoint without a ticker filter, so every article is
included along with the tickers it mentions. Articles are grouped by their UTC
publish day; each day becomes one parquet file (days without news get none).
Existing files are skipped, so re-running resumes an interrupted download or
adds the days since the last run. Reads POLYGON_API_KEY from .env.
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
import pyarrow as pa
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://api.polygon.io/v2/reference/news"
FIRST_DAY = date(2016, 6, 22)  # Polygon has no news before this day.
# A fixed schema gives every file the same columns, even on days without keywords or insights.
SCHEMA = pa.schema([
    ("published_utc", pa.timestamp("s", tz="UTC")),
    ("id", pa.string()),
    ("publisher", pa.string()),
    ("title", pa.string()),
    ("author", pa.string()),
    ("description", pa.string()),
    ("tickers", pa.list_(pa.string())),
    ("keywords", pa.list_(pa.string())),
    ("insights", pa.list_(pa.struct([("ticker", pa.string()), ("sentiment", pa.string()),
                                     ("sentiment_reasoning", pa.string())]))),
    ("article_url", pa.string()),
    ("amp_url", pa.string()),
    ("image_url", pa.string()),
])
_thread = threading.local()


def start_session(key: str) -> None:
    _thread.session = requests.Session()
    _thread.session.headers["Authorization"] = f"Bearer {key}"


def get(url: str, params: dict | None = None, attempts: int = 5) -> dict:
    for attempt in range(attempts):
        if attempt:
            time.sleep(2 ** attempt)
        try:
            response = _thread.session.get(url, params=params, timeout=60)
            if response.status_code == 200:
                body = response.json()
                if body.get("status") in ("OK", "DELAYED"):
                    return body
        except requests.RequestException as exc:
            error = repr(exc)
            continue
        error = f"HTTP {response.status_code}: {response.text[:300]}"
        if response.status_code != 429 and response.status_code < 500:
            break  # Bad key or plan limit: retrying will not help.
    raise RuntimeError(f"Could not download {url}: {error}")


def fetch(day: date) -> list[dict]:
    """Return every article published on the day (UTC), following pagination."""
    params = {"published_utc.gte": f"{day}T00:00:00Z", "published_utc.lt": f"{day + timedelta(days=1)}T00:00:00Z",
              "sort": "published_utc", "order": "asc", "limit": 1000}
    body = get(URL, params)
    articles = body.get("results") or []
    while body.get("next_url"):
        body = get(body["next_url"])  # The cursor in next_url carries the filters.
        articles += body.get("results") or []
    return articles


def day_path(out: Path, day: date) -> Path:
    return out / str(day.year) / f"{day}.parquet"


def download_day(day: date, out: Path) -> int:
    articles = list({a["id"]: a for a in fetch(day)}.values())  # Pages can overlap at the boundary.
    if not articles:
        return 0
    rows = [{**{name: a.get(name) for name in SCHEMA.names},
             "published_utc": pd.Timestamp(a["published_utc"]).to_pydatetime(),
             "publisher": (a.get("publisher") or {}).get("name")} for a in articles]
    table = pa.Table.from_pylist(sorted(rows, key=lambda r: (r["published_utc"], r["id"])), schema=SCHEMA)
    path = day_path(out, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")  # Parquet readers skip dot-files.
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=FIRST_DAY, help="First day, YYYY-MM-DD.")
    parser.add_argument("--end", type=date.fromisoformat, help="Last day, YYYY-MM-DD; at most yesterday (UTC), the default.")
    parser.add_argument("--out", type=Path, default=ROOT / "data/raw/polygon/news")
    parser.add_argument("--workers", type=int, default=8, help="Parallel requests.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download days that already have a file.")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    key = os.environ.get("POLYGON_API_KEY", "").strip()
    if not key:
        raise SystemExit("POLYGON_API_KEY is missing from .env and the environment.")
    # Today's news is still arriving, and a stored day is never fetched again.
    yesterday = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).date()
    end = min(args.end or yesterday, yesterday)
    days = [d.date() for d in pd.date_range(args.start, end)]
    todo = [d for d in days if args.overwrite or not day_path(args.out, d).exists()]
    print(f"{len(days)} days from {args.start} to {end}; downloading {len(todo)} into {args.out}", flush=True)
    stored = articles = 0
    started = time.monotonic()
    with ThreadPoolExecutor(args.workers, initializer=start_session, initargs=(key,)) as pool:
        futures = [pool.submit(download_day, day, args.out) for day in todo]
        try:
            for done, future in enumerate(as_completed(futures), 1):
                count = future.result()
                stored += count > 0
                articles += count
                if done % 250 == 0 or done == len(todo):
                    print(f"{done}/{len(todo)} days, {articles:,} articles, {time.monotonic() - started:.0f}s", flush=True)
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise
    print(f"Stored {stored} days ({articles:,} articles); {len(todo) - stored} days had no news.")


if __name__ == "__main__":
    main()
