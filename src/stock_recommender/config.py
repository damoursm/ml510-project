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
                values[setting.name] = ROOT / raw if setting.type is Path else setting.type(raw)
        return cls(**values)
