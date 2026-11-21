"""The API's response bodies."""
from datetime import date, datetime
from pydantic import BaseModel


class Recommendation(BaseModel):
    rank: int  # Within its list: 1 is the strongest buy, or the strongest sell.
    ticker: str
    score: float
    expected_return: float  # Over horizon_days, as a fraction: 0.02 is +2%.
    news_sentiment: float
    news_count: int  # Articles mentioning the ticker that the news model saw.
    close: float  # On as_of.
    dollar_volume: float
    model: str | None = None  # The selected model that ranked it, when serving a backtest run (RecommendationList.run).


class FetchReport(BaseModel):
    errors: list[str]  # Empty when both downloads succeeded.


class EmailReport(BaseModel):
    sent_to: list[str]
    error: str | None = None


class RecommendationList(BaseModel):
    as_of: date
    news_cutoff: datetime
    news_through: datetime | None
    generated_at: datetime
    horizon_days: int
    model_version: str
    universe_size: int
    buy: list[Recommendation]  # Best scores first; with several selected models, each one's in turn.
    sell: list[Recommendation]  # Worst scores first; never a ticker that is also in its model's buy list.
    fetch: FetchReport | None = None  # Set when the request asked to fetch new data first.
    email: EmailReport | None = None  # Set when the request asked for an email.
    # The backtest run whose selected models ranked the lists (RECOMMENDER_MODEL_RUN); null for load_models().
    run: str | None = None
