"""Service settings, read from RECOMMENDER_<NAME> environment variables or the repository's .env file."""
from dataclasses import dataclass, field, fields
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = ROOT / "data/processed/dashboard"  # Where the API writes the dashboard (dashboard.py) by default.


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
    # Stocks in each of the buy and sell lists when the request gives no limit; None: each served model's own number,
    # the one its backtest chose (its picks, or its pick_share of the stocks it ranked).
    top_n: int | None = None
    min_price: float = 5.0  # Last close, in dollars; excludes penny stocks.
    min_dollar_volume: float = 1_000_000.0  # Median daily close * volume over the bars window.
    # Email server, e.g. smtp.gmail.com with an app password, through which the API sends the lists it is asked to.
    smtp_host: str = ""
    smtp_port: int = 587  # 465 for implicit TLS; otherwise STARTTLS when the server offers it.
    smtp_user: str = ""
    smtp_password: str = field(default="", repr=False)
    email_from: str = ""  # Defaults to smtp_user.
    # Comma-separated recipients that the 16:00 email task (scripts/email_recommendations.py) asks the API to email.
    email_to: str = ""
    # The backtest run whose final selection, the best combination of models, the API serves in place of
    # load_models() (served_models.py): artifacts/<label>/<stamp>, or its selection.json, relative to the repository.
    model_run: str = ""
    # The dashboard's folder (dashboard.py), written at every call of the recommendations; None turns it off.
    # from_env() gives DASHBOARD_DIR unless RECOMMENDER_DASHBOARD_DIR is set (empty turns it off).
    dashboard_dir: Path | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        """Read each setting from RECOMMENDER_<NAME>, e.g. RECOMMENDER_TOP_N=50, falling back to the default.

        Variables already in the environment take precedence over the .env file.
        """
        load_dotenv(ROOT / ".env")
        values = {}
        for setting in fields(cls):
            raw = os.environ.get(f"RECOMMENDER_{setting.name.upper()}")
            if setting.name == "dashboard_dir":  # On by default for the API; an empty value turns it off.
                values[setting.name] = DASHBOARD_DIR if raw is None else ROOT / raw if raw.strip() else None
            elif raw is not None:
                # A relative path is relative to the repository, like .env and the download scripts' outputs.
                if setting.type is Path:
                    values[setting.name] = ROOT / raw
                elif setting.type is bool:  # bool("false") would be True.
                    values[setting.name] = raw.strip().lower() in ("1", "true", "yes", "on")
                elif setting.type == int | None:  # An empty value is None, e.g. RECOMMENDER_TOP_N= for no fixed size.
                    values[setting.name] = int(raw) if raw.strip() else None
                else:
                    values[setting.name] = setting.type(raw)
        return cls(**values)
