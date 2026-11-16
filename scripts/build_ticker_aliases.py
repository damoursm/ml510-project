"""Build the table of aliases that scoring hides from the LLM: the names each ticker's company goes by in the news.

Reads Polygon's ticker names (scripts/download_ticker_names.py) and the news between two days, and keeps the aliases
the news confirm (stock_recommender/anonymize.py): where each is hidden in its ticker's articles, and whether in the
others too. The table is replaced on each run. Scores already stored keep the aliases they were made with: build it
once before scoring, and again only to add new listings.

  python scripts/build_ticker_aliases.py            # from the news of 2021-2025
"""
import argparse
from datetime import date
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd  # noqa: E402
from stock_recommender.anonymize import NEVER, build_aliases  # noqa: E402
from stock_recommender.config import Settings  # noqa: E402


def main() -> int:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2021, 1, 1), help="First news day (UTC).")
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31), help="Last news day (UTC).")
    parser.add_argument("--news-dir", type=Path, default=settings.news_dir)
    parser.add_argument("--names", type=Path, default=settings.ticker_names_path)
    parser.add_argument("--out", type=Path, default=settings.ticker_aliases_path)
    parser.add_argument("--every", type=int, default=1, help="Read one news day in this many (a quick look).")
    args = parser.parse_args()
    if not args.names.exists():
        parser.error(f"{args.names} is missing: run scripts/download_ticker_names.py first.")
    names = pd.read_parquet(args.names)
    paths = [path for year in range(args.start.year, args.end.year + 1)
             for path in sorted((args.news_dir / str(year)).glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].parquet"))
             if args.start <= date.fromisoformat(path.stem) <= args.end][::max(1, args.every)]
    articles = []
    for path in paths:
        news = pd.read_parquet(path, columns=["title", "description", "tickers", "published_utc"])
        articles += list(zip(news.title, news.description, news.tickers, news.published_utc))
    print(f"{len(articles):,} articles on {len(paths)} news days {args.start}..{args.end}; "
          f"{names.ticker.nunique():,} tickers with a name", flush=True)
    table = build_aliases(names, articles)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp")
    table.to_parquet(temporary, index=False)
    os.replace(temporary, args.out)
    used = table[table.use != NEVER]
    print(f"Stored {len(table):,} aliases, {len(used):,} used for {used.ticker.nunique():,} tickers and "
          f"{(table.wide != '').sum():,} hidden in any article, into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
