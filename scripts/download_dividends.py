"""Download Polygon's cash dividends with ex-dividend dates between two dates into one parquet file.

Each row is one distribution as declared: the ticker, the ex-dividend date (the first day the stock trades
without it), the cash amount per share and its currency, and the type (CD regular, SC special, LT/ST capital
gains). Amounts are not restated for later splits. The file is replaced on each run. Reads POLYGON_API_KEY
from .env.
"""
import argparse
from datetime import date
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import requests
from download_splits import get

ROOT = Path(__file__).resolve().parents[1]
URL = "https://api.polygon.io/v3/reference/dividends"
COLUMNS = ["ticker", "ex_dividend_date", "cash_amount", "currency", "dividend_type", "id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True, help="First ex-dividend date, YYYY-MM-DD.")
    parser.add_argument("--end", type=date.fromisoformat, required=True, help="Last ex-dividend date, YYYY-MM-DD.")
    parser.add_argument("--out", type=Path, required=True, help="Parquet file to write.")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    key = os.environ.get("POLYGON_API_KEY", "").strip()
    if not key:
        raise SystemExit("POLYGON_API_KEY is missing from .env and the environment.")
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {key}"
    params = {"ex_dividend_date.gte": str(args.start), "ex_dividend_date.lte": str(args.end), "limit": 1000,
              "order": "asc", "sort": "ex_dividend_date"}
    body = get(session, URL, params)
    rows = body.get("results") or []
    pages = 1
    while body.get("next_url"):
        body = get(session, body["next_url"])  # The cursor in next_url carries the filters.
        rows += body.get("results") or []
        pages += 1
        if pages % 50 == 0:
            print(f"{pages} pages, {len(rows):,} dividends", flush=True)
    dividends = pd.DataFrame(rows).reindex(columns=COLUMNS).drop_duplicates("id")
    dividends["ex_dividend_date"] = pd.to_datetime(dividends.ex_dividend_date)
    dividends["cash_amount"] = dividends.cash_amount.astype("float64")
    outside = dividends[(dividends.ex_dividend_date < pd.Timestamp(args.start))
                        | (dividends.ex_dividend_date > pd.Timestamp(args.end))]
    if len(outside):
        raise SystemExit(f"Polygon returned {len(outside)} dividends outside {args.start}..{args.end}; nothing written.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp")
    dividends.sort_values(["ex_dividend_date", "ticker"]).to_parquet(temporary, index=False)
    os.replace(temporary, args.out)
    print(f"Stored {len(dividends):,} dividends with ex-dates {args.start}..{args.end} into {args.out}")


if __name__ == "__main__":
    main()
