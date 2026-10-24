"""Download Polygon's stock splits executed between two dates into one parquet file.

Each row is one split: the ticker, the execution date (the first day the stock
trades at the new price) and the ratio, split_to new shares for every split_from
old ones. With --end before a year, nothing from that year onward is requested.
The file is replaced on each run. Reads POLYGON_API_KEY from .env.
"""
import argparse
from datetime import date
import os
from pathlib import Path
import time
from dotenv import load_dotenv
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://api.polygon.io/v3/reference/splits"


def get(session: requests.Session, url: str, params: dict | None = None, attempts: int = 5) -> dict:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True, help="First execution date, YYYY-MM-DD.")
    parser.add_argument("--end", type=date.fromisoformat, required=True, help="Last execution date, YYYY-MM-DD.")
    parser.add_argument("--out", type=Path, required=True, help="Parquet file to write.")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    key = os.environ.get("POLYGON_API_KEY", "").strip()
    if not key:
        raise SystemExit("POLYGON_API_KEY is missing from .env and the environment.")
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {key}"
    params = {"execution_date.gte": str(args.start), "execution_date.lte": str(args.end), "limit": 1000,
              "order": "asc", "sort": "execution_date"}
    body = get(session, URL, params)
    rows = body.get("results") or []
    while body.get("next_url"):
        body = get(session, body["next_url"])  # The cursor in next_url carries the filters.
        rows += body.get("results") or []
    splits = (pd.DataFrame(rows, columns=["ticker", "execution_date", "split_from", "split_to", "id"])
              .drop_duplicates("id").astype({"split_from": "float64", "split_to": "float64"}))
    splits["execution_date"] = pd.to_datetime(splits.execution_date)
    outside = splits[(splits.execution_date < pd.Timestamp(args.start)) | (splits.execution_date > pd.Timestamp(args.end))]
    if len(outside):
        raise SystemExit(f"Polygon returned {len(outside)} splits outside {args.start}..{args.end}; nothing written.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp")
    splits.sort_values(["execution_date", "ticker"]).to_parquet(temporary, index=False)
    os.replace(temporary, args.out)
    print(f"Stored {len(splits)} splits executed {args.start}..{args.end} into {args.out}")


if __name__ == "__main__":
    main()
