"""The API's response bodies."""
from datetime import date, datetime
import math
from pydantic import BaseModel, field_validator


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
    # Its standard deviation of daily moves over the last 20 sessions, which a volatility_exit rule is measured in;
    # null with fewer sessions.
    daily_volatility: float | None = None

    @field_validator("daily_volatility")
    @classmethod
    def _known(cls, value: float | None) -> float | None:
        return None if value is None or math.isnan(value) else value  # JSON has no NaN.


class ExitStrategy(BaseModel):
    """When a model's bought stocks are sold (exits.py): fractions of the entry price, None for a rule it lacks."""
    hold_days: int  # Trading days to hold at most.
    stop_loss: float | None = None  # Sell at a close this far below the entry price (MAE limit).
    take_profit: float | None = None  # Or this far above it (MFE target).
    volatility_exit: float | None = None  # Or once a move is back within daily_volatility after one this many times it.
    summary: str  # The rule in words.


class AccountRule(BaseModel):
    """How an account trades the buy lists: as the served run's backtest account did, with one change (account.trade
    make_room). A guide: the API keeps no account."""
    capital: float  # The cash the account starts with.
    order_value: float  # The dollars of each stock to buy, bought at the close of the day it is listed.
    cost_bps: float  # Per side, on the value traded.
    # The served model whose oldest open trades an order the cash does not cover sells first, never one bought the same
    # day; None: such an order is skipped.
    make_room: str | None = None
    # Where the cash not in trades is held from each close to the next (account.trade sweep); an order spends it
    # first. None: it stays cash.
    idle_cash: str | None = None
    summary: str  # The rule in words.


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
    horizon_days: int  # The longest a listed stock is held: the largest hold_days among the models.
    exits: dict[str, ExitStrategy] = {}  # Each model's exit strategy, by the names in Recommendation.model.
    account: AccountRule | None = None  # With a served run: how its account trades the buy lists.
    model_version: str
    universe_size: int
    buy: list[Recommendation]  # Best scores first; with several selected models, each one's in turn.
    sell: list[Recommendation]  # Worst scores first; never a ticker that is also in its model's buy list.
    fetch: FetchReport | None = None  # Set when the request asked to fetch new data first.
    email: EmailReport | None = None  # Set when the request asked for an email.
    # The backtest run whose selected models ranked the lists (RECOMMENDER_MODEL_RUN); null for load_models().
    run: str | None = None
