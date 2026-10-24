"""The only door to data in a backtest: 2021-2025, and never anything from 2026 or later.

Every file a backtest reads is listed and opened here. Only the year folders 2021..2025 are listed, a file is
opened only when the date in its name is inside the window, and every row read is checked against its file's
day; a row outside it raises LeakageError instead of being dropped. The bounds are constants, not settings.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from ..data import NEWS_COLUMNS

FIRST_DAY = date(2021, 1, 1)
LAST_DAY = date(2025, 12, 31)
YEARS = range(FIRST_DAY.year, LAST_DAY.year + 1)
END_UTC = pd.Timestamp(LAST_DAY + timedelta(days=1), tz="UTC")  # No article published from here on is read.
BAR_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume", "vwap", "transactions"]
DAY_FILE = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].parquet"


class LeakageError(RuntimeError):
    """Data from outside 2021-2025 reached the backtest, or a request would need it."""


class DataError(RuntimeError):
    """The backtest's input data is missing or inconsistent."""


def in_window(day: date) -> bool:
    return FIRST_DAY <= day <= LAST_DAY


def check_window(start: date, end: date) -> None:
    """Raise LeakageError unless FIRST_DAY <= start <= end <= LAST_DAY."""
    if not (FIRST_DAY <= start <= end <= LAST_DAY):
        raise LeakageError(f"Backtests use {FIRST_DAY}..{LAST_DAY} only, never 2026; got {start}..{end}.")


def day_files(directory: Path) -> dict[date, Path]:
    """Return the <year>/<YYYY-MM-DD>.parquet files dated inside the window, listing only the 2021..2025 folders.

    A file whose date is outside the window or outside its year folder (e.g. 2025/2026-01-05.parquet) is skipped
    without being opened.
    """
    files = {}
    for year in YEARS:
        for path in sorted((Path(directory) / str(year)).glob(DAY_FILE)):
            day = date.fromisoformat(path.stem)
            if day.year == year and in_window(day):
                files[day] = path
    return files


class WindowedStore:
    """Reads a backtest's bars, splits and news from inside the window, and records every file it opens."""

    def __init__(self, bars_dir: Path, news_dir: Path, splits_path: Path):
        self.bars_dir, self.news_dir, self.splits_path = Path(bars_dir), Path(news_dir), Path(splits_path)
        self.opened: list[Path] = []

    def _read(self, path: Path, day: date | None, columns: list[str] | None = None) -> pa.Table:
        if day is not None and not in_window(day):
            raise LeakageError(f"Refusing to open {path}: {day} is outside {FIRST_DAY}..{LAST_DAY}.")
        self.opened.append(path)
        return pq.read_table(path, columns=columns)

    def _read_bars_day(self, day: date, path: Path) -> pa.Table:
        table = self._read(path, day, BAR_COLUMNS)
        if table.num_rows:
            low, high = pc.min_max(table.column("date")).values()
            if low.as_py().date() != day or high.as_py().date() != day:
                raise LeakageError(f"{path} holds bars dated {low}..{high}, not only {day}.")
        return table

    def read_bars(self) -> pd.DataFrame:
        """Return every 2021-2025 bar, sorted by date then ticker, with ticker as a category."""
        files = day_files(self.bars_dir)
        if not files:
            raise DataError(f"No 2021-2025 bars in {self.bars_dir}. Download them as traded: python "
                            "scripts/download_daily_bars.py --unadjusted --start 2021-01-01 --end 2025-12-31 "
                            "--out data/raw/polygon/daily_bars_unadjusted")
        with ThreadPoolExecutor(8) as pool:
            tables = list(pool.map(lambda item: self._read_bars_day(*item), files.items()))
        bars = pa.concat_tables(tables).to_pandas(categories=["ticker"])
        bars["date"] = bars.date.astype("datetime64[ns]")
        bars = bars.sort_values(["date", "ticker"], ignore_index=True)
        if bars.duplicated(["date", "ticker"]).any():
            raise DataError("The bars hold duplicate (date, ticker) rows.")
        return bars

    def _read_dated(self, path: Path, column: str, what: str, command: str) -> pd.DataFrame:
        if not path.is_file():
            raise DataError(f"Missing {path}. Download the {what}: python {command} --start 2021-01-01 "
                            f"--end 2025-12-31 --out {path}")
        frame = self._read(path, None).to_pandas()
        frame[column] = pd.to_datetime(frame[column])
        outside = frame[(frame[column].dt.date < FIRST_DAY) | (frame[column].dt.date > LAST_DAY)]
        if len(outside):
            raise LeakageError(f"{path} holds {len(outside)} {what} outside {FIRST_DAY}..{LAST_DAY}, "
                               f"e.g. {outside.iloc[0].ticker} on {outside.iloc[0][column].date()}.")
        return frame

    def read_splits(self) -> pd.DataFrame:
        """Return one net split per ticker and execution day (ticker, execution_date, split_from, split_to).

        Polygon lists some corporate actions as two same-day legs (e.g. 1:10000 then 10000:1); their product is
        the real ratio, and a pair that cancels out is dropped.
        """
        splits = self._read_dated(self.splits_path, "execution_date", "splits", "scripts/download_splits.py")
        splits = splits[(splits.split_from > 0) & (splits.split_to > 0)].drop_duplicates("id")
        net = (splits.split_to / splits.split_from).groupby([splits.ticker, splits.execution_date]).prod()
        net = net[(net - 1).abs() > 1e-9]
        return pd.DataFrame({"ticker": net.index.get_level_values(0), "execution_date": net.index.get_level_values(1),
                             "split_from": 1.0, "split_to": net.to_numpy()})

    def read_dividends(self, path: Path) -> pd.DataFrame:
        """Return the cash paid per share per ticker and ex-dividend day (ticker, ex_dividend_date, cash, currency),
        all inside the window.

        A day with a dollar record keeps only its dollar records; otherwise the amount stays in its own currency
        (e.g. Canadian banks listed in New York pay in CAD) for the panel to convert.
        """
        dividends = self._read_dated(path, "ex_dividend_date", "dividends", "scripts/download_dividends.py")
        dividends = dividends[dividends.cash_amount > 0].drop_duplicates("id")
        # Same-day records with the same amount and type are one payment listed twice; others add up
        # (e.g. a regular and a special dividend).
        dividends = dividends.drop_duplicates(["ticker", "ex_dividend_date", "currency", "cash_amount", "dividend_type"])
        days = ["ticker", "ex_dividend_date"]
        in_dollars = dividends.loc[dividends.currency == "USD", days].drop_duplicates().assign(in_dollars=True)
        dividends = dividends.merge(in_dollars, on=days, how="left")
        dividends = dividends[(dividends.currency == "USD") | dividends.in_dollars.isna()]
        paid = dividends.groupby(["ticker", "ex_dividend_date", "currency"]).cash_amount.sum()
        return paid.rename("cash").reset_index()

    def read_changes(self, path: Path) -> pd.DataFrame:
        """Return the curated ticker changes (configs/ticker_changes_2021_2025.csv), all inside the window."""
        self.opened.append(Path(path))
        changes = pd.read_csv(path, parse_dates=["date"])
        if len(changes[(changes.date.dt.date < FIRST_DAY) | (changes.date.dt.date > LAST_DAY)]):
            raise LeakageError(f"{path} holds ticker changes outside {FIRST_DAY}..{LAST_DAY}.")
        return changes

    def read_actions(self, path: Path) -> pd.DataFrame:
        """Return the curated corporate actions (configs/corporate_actions_2021_2025.csv), all inside the window."""
        self.opened.append(Path(path))
        actions = pd.read_csv(path, parse_dates=["date"])
        outside = actions[(actions.date.dt.date < FIRST_DAY) | (actions.date.dt.date > LAST_DAY)]
        if len(outside):
            raise LeakageError(f"{path} holds {len(outside)} actions outside {FIRST_DAY}..{LAST_DAY}.")
        return actions.fillna({"cash": 0.0, "shares_kept": 1.0, "distributed": "", "distributed_ratio": 0.0})

    def _read_news_day(self, day: date, path: Path) -> pa.Table:
        table = self._read(path, day, NEWS_COLUMNS)
        if table.num_rows:
            low, high = (value.as_py() for value in pc.min_max(table.column("published_utc")).values())
            start = pd.Timestamp(day, tz="UTC")
            if pd.Timestamp(low) < start or pd.Timestamp(high) >= min(start + pd.Timedelta(days=1), END_UTC):
                raise LeakageError(f"{path} holds articles published {low}..{high}, not only on {day} (UTC).")
        return table

    def read_news(self) -> pd.DataFrame:
        """Return every article published in 2021-2025 (UTC), sorted by publication time."""
        files = day_files(self.news_dir)
        if not files:
            return pd.DataFrame(columns=NEWS_COLUMNS).astype({"published_utc": "datetime64[ms, UTC]"})
        with ThreadPoolExecutor(8) as pool:
            tables = list(pool.map(lambda item: self._read_news_day(*item), files.items()))
        news = pa.concat_tables(tables).to_pandas()
        # Arrow hands each list cell out as a view into one buffer of every article; give each cell its own
        # storage, so no slice of the news can reach later articles.
        for column in ("tickers", "keywords", "insights"):
            news[column] = [cell.copy() if isinstance(cell, np.ndarray) else cell for cell in news[column]]
        return news.sort_values("published_utc", kind="stable", ignore_index=True)

    def audit(self) -> dict:
        """Count the files opened and check that none is outside the window."""
        dated = [path for path in self.opened if path.match(DAY_FILE)]
        outside = [str(path) for path in dated if not in_window(date.fromisoformat(path.stem))]
        if outside:
            raise LeakageError(f"Opened {len(outside)} files outside the window, e.g. {outside[0]}.")
        return {"files_opened": len(self.opened), "first_file_day": str(min(date.fromisoformat(p.stem) for p in dated)),
                "last_file_day": str(max(date.fromisoformat(p.stem) for p in dated)), "files_outside_window": 0}
