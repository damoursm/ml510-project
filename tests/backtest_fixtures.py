"""Synthetic stores for backtest tests: as-traded bars, splits and news written like the download scripts do."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from stock_recommender.backtest.run import BacktestConfig
from stock_recommender.data import NEWS_COLUMNS
from stock_recommender.models import placeholder_models

SESSIONS = pd.bdate_range("2024-11-01", "2025-03-31")  # Weekdays: tests do not need exchange holidays.


def small_models(**changes):
    """Placeholder models with short windows, so a few months of synthetic data are enough."""
    return replace(placeholder_models(), **({"bars_lookback": 3, "horizon_days": 2} | changes))


def random_walk(tickers: int, sessions: pd.DatetimeIndex = SESSIONS, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Return open and close prices (sessions x tickers) with independent overnight and intraday noise."""
    rng = np.random.default_rng(seed)
    overnight = rng.normal(0, 0.01, (len(sessions), tickers))
    intraday = rng.normal(0, 0.02, (len(sessions), tickers))
    names = [f"T{i:03d}" for i in range(tickers)]
    log_open = np.log(50) + np.cumsum(overnight + np.vstack([np.zeros(tickers), intraday[:-1]]), axis=0)
    opens = pd.DataFrame(np.exp(log_open), index=sessions, columns=names)
    closes = pd.DataFrame(np.exp(log_open + intraday), index=sessions, columns=names)
    return {"open": opens, "close": closes}


def write_bars(directory: Path, opens: pd.DataFrame, closes: pd.DataFrame | None = None,
               volume: float = 1_000_000.0, volumes: pd.DataFrame | None = None):
    """Write one file per session; a NaN open means the ticker has no bar that day."""
    closes = opens if closes is None else closes
    for day in opens.index:
        row = opens.loc[day]
        present = row.notna()
        frame = pd.DataFrame({
            "date": pd.Timestamp(day), "ticker": row.index[present], "open": row[present].to_numpy(),
            "close": closes.loc[day][present].to_numpy(),
            "volume": (volumes.loc[day][present].to_numpy() if volumes is not None else volume)})
        frame["high"] = frame[["open", "close"]].max(axis=1)
        frame["low"] = frame[["open", "close"]].min(axis=1)
        frame["vwap"] = (frame.open + frame.close) / 2
        frame["transactions"] = pd.array([100] * len(frame), dtype="Int64")
        frame = frame.astype({"date": "datetime64[ns]", "volume": "float64"})
        path = directory / str(day.year) / f"{day.date()}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame[["date", "ticker", "open", "high", "low", "close", "volume", "vwap", "transactions"]].to_parquet(path)


def write_news(directory: Path, articles: list[tuple[str, list[str]]]):
    """articles lists (published_utc, tickers); each lands in its UTC day's file."""
    frame = pd.DataFrame([{column: None for column in NEWS_COLUMNS} | {
        "published_utc": pd.Timestamp(published), "id": f"{published}-{i}", "title": "", "tickers": tickers,
        "keywords": [], "insights": []} for i, (published, tickers) in enumerate(articles)])
    frame["published_utc"] = pd.to_datetime(frame.published_utc, utc=True)
    for day, group in frame.groupby(frame.published_utc.dt.date):
        path = directory / str(day.year) / f"{day}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        group[NEWS_COLUMNS].to_parquet(path)


def write_splits(path: Path, splits: list[tuple[str, str, float, float]]):
    """splits lists (ticker, execution_date, split_from, split_to)."""
    frame = pd.DataFrame(splits, columns=["ticker", "execution_date", "split_from", "split_to"])
    frame["execution_date"] = pd.to_datetime(frame.execution_date)
    frame["id"] = [f"S{i}" for i in range(len(frame))]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def write_changes(path: Path, changes: list[tuple]):
    """changes lists (old, new, date, link)."""
    frame = pd.DataFrame(changes, columns=["old", "new", "date", "link"])
    frame.assign(description="test").to_csv(path, index=False)


def write_actions(path: Path, actions: list[tuple]):
    """actions lists (ticker, date, cash, shares_kept, distributed, distributed_ratio)."""
    frame = pd.DataFrame(actions, columns=["ticker", "date", "cash", "shares_kept", "distributed", "distributed_ratio"])
    frame.assign(description="test").to_csv(path, index=False)


def write_dividends(path: Path, dividends: list[tuple[str, str, float]], currency: str = "USD"):
    """dividends lists (ticker, ex_dividend_date, cash per share)."""
    frame = pd.DataFrame(dividends, columns=["ticker", "ex_dividend_date", "cash_amount"])
    frame["ex_dividend_date"] = pd.to_datetime(frame.ex_dividend_date)
    frame["currency"], frame["dividend_type"] = currency, "CD"
    frame["id"] = [f"D{i}" for i in range(len(frame))]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def config(root: Path, **changes) -> BacktestConfig:
    defaults = dict(models=("flat",), ensembles=(), test_years=(2025,), null_seeds=2, strict_calendar=False,
                    min_price=0.0, min_dollar_volume=0.0, cost_bps=0.0, top_n=2, bars_dir=root / "bars",
                    news_dir=root / "news", splits_path=root / "splits.parquet", dividends_path=None,
                    actions_path=None, changes_path=None)
    return BacktestConfig(**(defaults | changes))
