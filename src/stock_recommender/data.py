"""Read recent daily bars and news from the parquet files that scripts/download_*.py write."""
from datetime import date
from pathlib import Path
import pandas as pd

NEWS_COLUMNS = ["published_utc", "id", "publisher", "title", "author", "description", "tickers", "keywords",
                "insights", "article_url", "amp_url", "image_url"]


class DataUnavailable(RuntimeError):
    """The downloaded data cannot support a recommendation."""


def recent_day_files(directory: Path, count: int) -> list[Path]:
    """Return the newest `count` <year>/<YYYY-MM-DD>.parquet files under directory, oldest first.

    Only the newest year folders are listed, so this stays cheap with decades of history on disk.
    """
    files = []
    for year in sorted(Path(directory).glob("[0-9][0-9][0-9][0-9]"), reverse=True):
        files = sorted(year.glob("*.parquet"), key=lambda path: path.stem) + files
        if len(files) >= count:
            break
    return files[-count:] if count else []


def read_bars(directory: Path, days: int) -> pd.DataFrame:
    """Return every ticker's bars for the last `days` trading days on disk."""
    files = recent_day_files(directory, days)
    if not files:
        raise DataUnavailable(f"No daily bars in {directory}; run scripts/download_daily_bars.py.")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def read_news(directory: Path, start: pd.Timestamp, end: pd.Timestamp, sentiment_dir: Path | None = None,
              model: str | None = None) -> pd.DataFrame:
    """Return the articles published in [start, end); both bounds are UTC timestamps.

    With sentiment_dir, an llm_scores column holds each article's scores by `model` (sentiment.attach).
    """
    news = _read_news(directory, start, end)
    if sentiment_dir is None:
        return news
    from .sentiment import attach, read_day, to_frame  # sentiment imports requests, which only this needs.

    paths = [path for year in range(start.year, end.year + 1) for path in (Path(sentiment_dir) / str(year)).glob("*.parquet")
             if start.date() <= date.fromisoformat(path.stem) <= end.date()]
    scores = [read_day(path, model) for path in sorted(paths)]
    return attach(news, pd.concat(scores, ignore_index=True) if scores else to_frame([]))


def _read_news(directory: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    files = [path for year in range(start.year, end.year + 1) for path in (Path(directory) / str(year)).glob("*.parquet")
             if start.date() <= date.fromisoformat(path.stem) <= end.date()]
    if not files:
        return pd.DataFrame(columns=NEWS_COLUMNS).astype({"published_utc": "datetime64[ms, UTC]"})
    news = pd.concat([pd.read_parquet(path) for path in sorted(files)], ignore_index=True)
    return news[(news.published_utc >= start) & (news.published_utc < end)].reset_index(drop=True)
