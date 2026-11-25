"""Synthetic stores for backtest tests: as-traded bars, splits and news written like the download scripts do."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from stock_recommender import sentiment
from stock_recommender.backtest.run import BacktestConfig
from stock_recommender.data import NEWS_COLUMNS
from stock_recommender.models import placeholder_models

SESSIONS = pd.bdate_range("2024-11-01", "2025-03-31")  # Weekdays: tests do not need exchange holidays.
# The news market (write_news_market): 2024 trains the 2025 fold, with 60 sessions of history before it.
NEWS_SESSIONS = pd.bdate_range("2023-11-01", "2025-03-31")
NEWS_NAMES = [f"T{i:03d}" for i in range(30)]
GOOD, BAD, QUIET = NEWS_NAMES[:10], NEWS_NAMES[10:20], NEWS_NAMES[20:]  # Good news and rising, bad and falling, none.


def small_models(**changes):
    """Placeholder models with short windows and two picks, so a few months of synthetic data are enough."""
    return replace(placeholder_models(), **({"bars_lookback": 3, "horizon_days": 2, "picks": 2} | changes))


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
    defaults = dict(models=("flat",), test_years=(2025,), null_seeds=2, strict_calendar=False,
                    min_price=0.0, min_dollar_volume=0.0, cost_bps=0.0, bars_dir=root / "bars",
                    news_dir=root / "news", sentiment_dir=None, splits_path=root / "splits.parquet",
                    dividends_path=None,
                    actions_path=None, changes_path=None)
    return BacktestConfig(**(defaults | changes))


def write_news_market(root: Path) -> None:
    """Write random walks where the stocks with good news drift up and those with bad news drift down, every news
    stock gets an article each session, scored by the LLM 9 times in 10, and the quiet stocks get none; the
    sentiment files go to root / "sentiment"."""
    prices = random_walk(len(NEWS_NAMES), NEWS_SESSIONS, seed=7)
    drift = np.array([0.003 if name in GOOD else -0.003 if name in BAD else 0.0 for name in NEWS_NAMES])
    growth = np.exp(np.outer(np.arange(len(NEWS_SESSIONS)), drift))
    write_bars(root / "bars", prices["open"] * growth, prices["close"] * growth)
    write_splits(root / "splits.parquet", [])
    write_news(root / "news", [(f"{day.date()}T14:00:00Z", [name]) for day in NEWS_SESSIONS for name in GOOD + BAD])
    rng = np.random.default_rng(8)
    rows = []
    for path in sorted((root / "news").rglob("*.parquet")):
        for item in pd.read_parquet(path).itertuples():
            ticker = str(item.tickers[0])
            if rng.random() < 0.1:
                continue  # Not scored, as when the LLM's answer was malformed.
            sign = 1 if ticker in GOOD else -1
            rows.append({"id": item.id, "published_utc": item.published_utc, "ticker": ticker, "event": "other",
                         "relevance": 2, "impact": 1, "sentiment": sign * int(rng.integers(1, 3)),
                         "model": "qwen3:8b", "prompt_version": sentiment.PROMPT_VERSION})
    frame = sentiment.to_frame(rows)
    for day, group in frame.groupby(frame.published_utc.dt.date):
        sentiment.write_day(group, root / "sentiment" / str(day.year) / f"{day}.parquet")
