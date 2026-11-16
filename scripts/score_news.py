"""Score news articles with the local LLM served by Ollama, one news day at a time.

For each article tagged with 1 to --max-tickers companies, the LLM rates the event, the relevance, the impact and
the sentiment (stock_recommender/sentiment.py) of each tagged stock the recommender may trade in the week it reads the
article: price and dollar volume above its filters, from the daily bars (--all-tickers rates every tagged company).
The other tagged companies are still hidden from it. It gets the article without any company's name or
ticker and without its dates (stock_recommender/anonymize.py), so it cannot recall what happened next: that needs
the table of names to hide, --aliases, built once by scripts/download_ticker_names.py then
scripts/build_ticker_aliases.py.
Scores go to <out>/<year>/<day>.parquet, beside the news days. Only articles with a ticker that has no current score
are sent, so re-running resumes an interrupted run, scores the articles a later news download added, and asks again
for the ones the LLM failed to score. A day's new scores are written once all its articles have been asked; stop
with Ctrl+C at any time. Changing the prompt (PROMPT_VERSION) or the model scores everything again.

The scores directory records the weights (digest) its scores come from, in model_digests.json. The run refuses to
start when Ollama has other weights under the model's name (another `ollama pull`), and stops if they change during
the run. --allow-new-digest makes the new weights the directory's from now on, for this run and later ones, even when
its range has nothing to score: give it a recent range (--start/--end today) so no older article is scored with them.
The run also stops when the server leaves many articles in a row without an answer; with --retry-minutes it waits
and goes on instead, for a run left alone for days.

  python scripts/score_news.py --dry-run            # what 2021-2025 holds to score, and how long it would take
  python scripts/score_news.py --retry-minutes 10   # score 2021-2025 (days of GPU time)
  python scripts/score_news.py --start 2026-01-01 --end 2026-03-31
The API's daily refresh runs this on the last days of news. Settings default from RECOMMENDER_<NAME> or .env.
"""
import argparse
from datetime import date, timedelta
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd  # noqa: E402
from stock_recommender.anonymize import Aliases  # noqa: E402
from stock_recommender.config import Settings  # noqa: E402
from stock_recommender.sentiment import (OllamaScorer, ScorerUnavailable, check_digest, pending,  # noqa: E402
                                         score_day, tradable_days)


def news_days(directory: Path, start: date, end: date) -> list[tuple[date, Path]]:
    days = []
    for year in range(start.year, end.year + 1):
        for path in sorted((directory / str(year)).glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].parquet")):
            day = date.fromisoformat(path.stem)
            if start <= day <= end:
                days.append((day, path))
    return days


UNAVAILABLE = 2  # Exit code: the server, its model or the table of names is not as scoring needs.
UNADJUSTED = ROOT / "data/raw/polygon/daily_bars_unadjusted"  # Closes as traded: the backtest's.


def read_bars(directories: list[Path], start: date, end: date) -> pd.DataFrame:
    """Return the bars of the sessions from start to end, each day from the first directory that has it."""
    files = {}
    for directory in reversed(directories):
        for path in Path(directory).glob("[0-9][0-9][0-9][0-9]/*.parquet"):
            try:
                day = date.fromisoformat(path.stem)
            except ValueError:
                continue
            if start <= day <= end:
                files[day] = path
    frames = [pd.read_parquet(path, columns=["date", "ticker", "close", "volume"]) for _, path in sorted(files.items())]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "ticker", "close", "volume"])


def tradable_by_day(args: argparse.Namespace, settings: Settings, days: list[date]) -> dict | None:
    """The tickers scored on each news day: those the recommender may trade while it reads the day's articles
    (sentiment.tradable_days); None with --all-tickers."""
    if args.all_tickers or not days:
        return None
    directories = args.bars_dir or [UNADJUSTED, settings.bars_dir]
    bars = read_bars(directories, min(days) - timedelta(days=100), max(days) + timedelta(days=10))
    if not len(bars):
        raise SystemExit(f"No daily bars in {[str(path) for path in directories]} to tell which stocks are "
                         f"tradable; download them, or score every target with --all-tickers.")
    return tradable_days(bars, days, settings.min_price, settings.min_dollar_volume)


def main() -> int:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2021, 1, 1), help="First news day (UTC).")
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31), help="Last news day (UTC).")
    parser.add_argument("--news-dir", type=Path, default=settings.news_dir)
    parser.add_argument("--out", type=Path, default=settings.sentiment_dir)
    parser.add_argument("--aliases", type=Path, default=settings.ticker_aliases_path,
                        help="The names of each ticker's company to hide from the LLM.")
    parser.add_argument("--url", default=settings.ollama_url, help="Ollama server.")
    parser.add_argument("--model", default=settings.ollama_model)
    parser.add_argument("--max-tickers", type=int, default=settings.sentiment_max_tickers,
                        help="Articles tagged with more companies are not scored.")
    parser.add_argument("--workers", type=int, default=settings.sentiment_workers, help="Articles sent at once.")
    parser.add_argument("--limit", type=int, help="Stop after about this many articles (a pilot run).")
    parser.add_argument("--retry-minutes", type=float, default=0,
                        help="When the server is unavailable, wait this long and go on, instead of stopping.")
    parser.add_argument("--allow-new-digest", action="store_true",
                        help="Score with the weights Ollama has now although the stored scores come from others.")
    parser.add_argument("--bars-dir", type=Path, action="append",
                        help="Daily bars that tell which stocks are tradable, by preference (repeatable). Default: "
                             f"{UNADJUSTED.relative_to(ROOT)} where it has the day, else the API's bars.")
    parser.add_argument("--all-tickers", action="store_true",
                        help="Score every target, not only the stocks the recommender may trade that week.")
    parser.add_argument("--dry-run", action="store_true", help="Count the articles to score; send nothing.")
    args = parser.parse_args()
    if args.start > args.end or args.max_tickers < 1:
        parser.error("Need --start <= --end and --max-tickers >= 1.")
    if args.retry_minutes < 0:
        parser.error("Need --retry-minutes >= 0.")
    if args.dry_run and args.allow_new_digest:
        parser.error("--allow-new-digest records Ollama's weights: it needs a real run, not --dry-run.")

    if not args.dry_run and not args.aliases.exists():
        print(f"{args.aliases} is missing: the LLM would get the companies' names. Run "
              f"scripts/download_ticker_names.py then scripts/build_ticker_aliases.py.", file=sys.stderr)
        return UNAVAILABLE
    try:
        aliases = None if args.dry_run else Aliases.read(args.aliases)
    except ValueError as error:  # A table built before the masking it serves: build it again.
        print(error, file=sys.stderr)
        return UNAVAILABLE
    days = [day for day, _ in news_days(args.news_dir, args.start, args.end)]
    args.keep = tradable_by_day(args, settings, days)
    while True:
        code = score(args, aliases)
        if code != UNAVAILABLE or not args.retry_minutes:
            return code
        print(f"Trying again in {args.retry_minutes:g} minutes.", file=sys.stderr, flush=True)
        try:
            time.sleep(args.retry_minutes * 60)
        except KeyboardInterrupt:
            return 130


def keep(args: argparse.Namespace, day: date):
    """The tickers to score on a news day, or None for every target."""
    return None if args.keep is None else args.keep.get(day, set())


def score(args: argparse.Namespace, aliases: Aliases | None) -> int:
    """Score what the range holds to score, with a new scorer; return the exit code."""
    scorer = OllamaScorer(args.url, args.model, aliases=aliases)
    if args.allow_new_digest:  # Recorded even when the range has nothing to score: it unblocks later runs.
        try:
            scorer.check()
            check_digest(args.out, scorer, allow_new=True)
        except ScorerUnavailable as error:
            print(error, file=sys.stderr)
            return UNAVAILABLE
        print(f"{args.out} now takes {args.model} scores from digest {str(scorer.digest)[:12]}.", flush=True)

    days = news_days(args.news_dir, args.start, args.end)
    todo = {}
    for day, path in days:
        news = pd.read_parquet(path)
        _, articles = pending(news, args.out / str(day.year) / f"{day}.parquet", args.model, args.max_tickers,
                              keep(args, day))
        if len(articles):
            todo[day] = (path, len(articles))
    total = sum(count for _, count in todo.values())
    print(f"{total:,} articles to score on {len(todo)} of {len(days)} news days {args.start}..{args.end} "
          f"with {args.model}", flush=True)
    if args.dry_run or not total:
        if args.dry_run and total:
            for rate in (0.8, 2.0):
                print(f"  at {rate} articles/s: {timedelta(seconds=round(total / rate))}")
        return 0

    try:
        scorer.check()
        check_digest(args.out, scorer)
    except ScorerUnavailable as error:
        print(error, file=sys.stderr, flush=True)
        return UNAVAILABLE
    start, done, failed = time.time(), 0, 0
    try:
        for day, (path, count) in todo.items():
            scorer.check()  # Stops the run if the server went away or its model changed.
            sent, missed = score_day(pd.read_parquet(path), args.out / str(day.year) / f"{day}.parquet", scorer,
                                     args.max_tickers, args.workers, keep(args, day))
            done, failed = done + sent, failed + missed
            rate = done / max(time.time() - start, 1e-9)
            note = f", {missed} not scored" if missed else ""
            print(f"{day}: {sent} articles{note}; {done:,}/{total:,} at {rate:.2f}/s, about "
                  f"{timedelta(seconds=round(max(total - done, 0) / max(rate, 1e-9)))} left", flush=True)
            if args.limit and done >= args.limit:
                break
    except (ScorerUnavailable, OSError) as error:  # OSError: a day's file could not be read or written.
        print(f"Stopped after {done:,} articles: {error!r}", file=sys.stderr, flush=True)
        return UNAVAILABLE
    except KeyboardInterrupt:
        print(f"Interrupted after {done:,} articles; the finished days are saved.", file=sys.stderr)
        return 130
    print(f"Scored {done - failed:,} articles in {timedelta(seconds=round(time.time() - start))}.")
    if failed:
        print(f"{failed:,} articles have a ticker the LLM did not score; the next run asks again.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
