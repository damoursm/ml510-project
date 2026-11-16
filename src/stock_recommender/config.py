"""Service settings, read from RECOMMENDER_<NAME> environment variables or the repository's .env file."""
from dataclasses import dataclass, field, fields
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    bars_dir: Path = ROOT / "data/raw/polygon/daily_bars"
    news_dir: Path = ROOT / "data/raw/polygon/news"
    # News scored by a local LLM (stock_recommender/sentiment.py): one file per news day, scored by ollama_model.
    sentiment_dir: Path = ROOT / "data/processed/news_sentiment"
    # Every name each ticker had (scripts/download_ticker_names.py): scoring hides the companies from the LLM.
    ticker_names_path: Path = ROOT / "data/raw/polygon/ticker_names.parquet"
    # The names of each ticker's company that scoring hides (scripts/build_ticker_aliases.py).
    ticker_aliases_path: Path = ROOT / "data/processed/ticker_aliases.parquet"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    sentiment_max_tickers: int = 5  # Articles tagged with more tickers (roundups, lists) are not scored.
    sentiment_workers: int = 4  # Articles sent to Ollama at once.
    score_news: bool = True  # Score the new articles after each refresh's downloads; needs Ollama running.
    top_n: int = 20  # Stocks in each of the buy and sell lists when the request gives no limit.
    min_price: float = 5.0  # Last close, in dollars; excludes penny stocks.
    min_dollar_volume: float = 1_000_000.0  # Median daily close * volume over the bars window.
    refresh_time: str = "16:00"  # HH:MM each weekday to download the day's bars and news; "off" to disable.
    refresh_timezone: str = "America/New_York"
    # Email server, e.g. smtp.gmail.com with an app password; email is off until smtp_host and email_to are set.
    smtp_host: str = ""
    smtp_port: int = 587  # 465 for implicit TLS; otherwise STARTTLS when the server offers it.
    smtp_user: str = ""
    smtp_password: str = field(default="", repr=False)
    email_from: str = ""  # Defaults to smtp_user.
    email_to: str = ""  # Comma-separated recipients.
    email_daily: bool = True  # Email the top_n lists after each scheduled refresh, once email is configured.

    @classmethod
    def from_env(cls) -> "Settings":
        """Read each setting from RECOMMENDER_<NAME>, e.g. RECOMMENDER_TOP_N=50, falling back to the default.

        Variables already in the environment take precedence over the .env file.
        """
        load_dotenv(ROOT / ".env")
        values = {}
        for setting in fields(cls):
            raw = os.environ.get(f"RECOMMENDER_{setting.name.upper()}")
            if raw is not None:
                # A relative path is relative to the repository, like .env and the download scripts' outputs.
                if setting.type is Path:
                    values[setting.name] = ROOT / raw
                elif setting.type is bool:  # bool("false") would be True.
                    values[setting.name] = raw.strip().lower() in ("1", "true", "yes", "on")
                else:
                    values[setting.name] = setting.type(raw)
        return cls(**values)
