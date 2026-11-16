"""Download the names Polygon gives each stock ticker, on the first weekday of every quarter between two dates.

Scoring the news hides the companies from the LLM (stock_recommender/anonymize.py), which needs every name a ticker
had: FB was Facebook, Inc. before Meta Platforms, Inc. The news are tagged with over-the-counter tickers too
(DANOY for Danone, TCEHY for Tencent), so both markets are read: stocks (the exchanges) and otc, which Polygon has
from 2022 only. Each row is a ticker with one name, its security type (CS common stock, ADRC, ETF, PFD, WARRANT...),
its market and the first and last of the dates it had that name on. The file is replaced on each run; run it again now and then for the new listings. Reads
POLYGON_API_KEY from .env.

  python scripts/download_ticker_names.py           # 2021-01-01 to today
"""
import argparse
from datetime import date, timedelta
import os
from pathlib import Path
import sys
import time
from dotenv import load_dotenv
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stock_recommender.config import Settings  # noqa: E402

HOST = "https://api.polygon.io/"
URL = HOST + "v3/reference/tickers"


def get(session: requests.Session, url: str, params: dict | None = None, attempts: int = 5) -> dict:
    if not url.startswith(HOST):
        raise RuntimeError(f"Refusing to send the Polygon key to {url[:60]}")
    for attempt in range(attempts):
        if attempt:
            time.sleep(2 ** attempt)
        try:
            response = session.get(url, params=params, timeout=60)
            if response.status_code == 200:
                return response.json()
        except requests.RequestException as exc:
            error = repr(exc)
            continue
        error = f"HTTP {response.status_code}: {response.text[:300]}"
        if response.status_code != 429 and response.status_code < 500:
            break  # Bad key or plan limit: retrying will not help.
    raise RuntimeError(f"Could not download {url}: {error}")


def quarter_days(start: date, end: date) -> list[date]:
    """The first weekday of each quarter from start's to end's, then end: every day the names are read on."""
    days, year, month = [], start.year, 3 * ((start.month - 1) // 3) + 1
    while date(year, month, 1) <= end:
        day = date(year, month, 1)
        day += timedelta(days=(7 - day.weekday()) % 7 if day.weekday() >= 5 else 0)
        if start <= day <= end:
            days.append(day)
        year, month = (year + 1, 1) if month == 10 else (year, month + 3)
    return sorted({*days, start, end})


def main():
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2021, 1, 1), help="First date, YYYY-MM-DD.")
    parser.add_argument("--end", type=date.fromisoformat, default=date.today(), help="Last date, YYYY-MM-DD.")
    parser.add_argument("--markets", nargs="+", default=["stocks", "otc"], help="Polygon markets to read.")
    parser.add_argument("--out", type=Path, default=settings.ticker_names_path, help="Parquet file to write.")
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("Need --start <= --end.")
    load_dotenv(ROOT / ".env")
    key = os.environ.get("POLYGON_API_KEY", "").strip()
    if not key:
        raise SystemExit("POLYGON_API_KEY is missing from .env and the environment.")
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {key}"
    rows = []
    for day in quarter_days(args.start, args.end):
        count = 0
        for market in args.markets:
            body = get(session, URL, {"market": market, "date": str(day), "limit": 1000, "order": "asc",
                                      "sort": "ticker"})
            found = body.get("results") or []
            while body.get("next_url"):
                body = get(session, body["next_url"])  # The cursor in next_url carries the filters.
                found += body.get("results") or []
            rows += [(item.get("ticker"), item.get("name"), item.get("type"), market, day) for item in found]
            count += len(found)
        print(f"{day}: {count} tickers", flush=True)
    names = pd.DataFrame(rows, columns=["ticker", "name", "type", "market", "date"])
    names = names.dropna(subset=["ticker", "name"])
    names["name"] = names.name.str.strip()
    names = names[names.name != ""]
    if not len(names):
        raise SystemExit("Polygon returned no ticker names; nothing written.")
    names["type"] = names.type.fillna("")
    names = (names.groupby(["ticker", "name", "type", "market"], as_index=False)
             .agg(first_date=("date", "min"), last_date=("date", "max")).sort_values(["ticker", "first_date"]))
    names["first_date"], names["last_date"] = pd.to_datetime(names.first_date), pd.to_datetime(names.last_date)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp")
    names.to_parquet(temporary, index=False)
    os.replace(temporary, args.out)
    print(f"Stored {len(names)} names of {names.ticker.nunique()} tickers into {args.out}")


if __name__ == "__main__":
    main()
