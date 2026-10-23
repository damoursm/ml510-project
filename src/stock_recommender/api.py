"""HTTP API that serves the latest stock recommendations.

Run from the repository root, then open http://127.0.0.1:8000/docs:
    python -m uvicorn stock_recommender.api:app --app-dir src
Keep a single worker process: each one would run its own daily refresh.

GET /recommendations/latest returns the stocks to buy and sell at the next open. POST to the same path does
the same after optionally fetching new data (fetch=true) and before optionally emailing the lists (email=true).
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import threading
from fastapi import FastAPI, HTTPException, Query
from . import __version__, notify
from .config import Settings
from .data import DataUnavailable, recent_day_files
from .models import Models, load_models
from .recommend import Recommendations, buy_and_sell, recommend
from .refresh import DailyRefresh
from .schemas import FetchReport, RecommendationList


class LatestRecommendations:
    """Holds the last recommendations and runs the models again once the downloads add or rewrite a file they use.

    While `refreshing` is set, the downloads are rewriting files, so the held recommendations are served as is.
    """

    def __init__(self, settings: Settings, models: Models):
        self.settings = settings
        self.models = models
        self.refreshing = threading.Event()
        self._lock = threading.Lock()
        self._inputs = None
        self._result: tuple[Recommendations, datetime] | None = None

    def _input_files(self) -> tuple:
        # The news files in the window, plus slack for days stored after as_of (weekends, lagging bars).
        files = (recent_day_files(self.settings.bars_dir, self.models.bars_lookback)
                 + recent_day_files(self.settings.news_dir, self.models.news_lookback_days + 5))
        return tuple((str(path), path.stat().st_mtime_ns) for path in files)

    def get(self) -> tuple[Recommendations, datetime]:
        with self._lock:
            if self._result is None or not self.refreshing.is_set():
                inputs = self._input_files()
                if inputs != self._inputs:
                    self._result = recommend(self.settings, self.models), datetime.now(timezone.utc)
                    self._inputs = inputs
            return self._result

    def lists(self, limit: int) -> RecommendationList:
        """Return the `limit` stocks to buy and to sell; DataUnavailable becomes a 503."""
        try:
            result, generated_at = self.get()
        except DataUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        buy, sell = buy_and_sell(result.table, limit)
        return RecommendationList(
            as_of=result.as_of.date(),
            news_cutoff=result.news_cutoff.to_pydatetime(),
            news_through=result.news_through.to_pydatetime() if result.news_through is not None else None,
            generated_at=generated_at,
            horizon_days=self.models.horizon_days,
            model_version=self.models.version,
            universe_size=result.universe_size,
            buy=buy.to_dict("records"),
            sell=sell.to_dict("records"),
        )


def create_app(settings: Settings | None = None, models: Models | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    latest = LatestRecommendations(settings, models or load_models())
    refresh = DailyRefresh(settings, latest)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Compute first, so requests during a refresh that starts right away get these recommendations.
        try:
            latest.get()
        except DataUnavailable:
            pass
        refresh.start()
        yield
        refresh.stop()

    app = FastAPI(title="ML510 stock recommendations", version=__version__, lifespan=lifespan)
    limit_query = Query(settings.top_n, ge=1, le=500, description="Stocks in each of the buy and sell lists.")

    @app.get("/health")
    def health() -> dict:
        """Report that the API is up, with the daily refresh's schedule and last run."""
        return {"status": "ok", "refresh": refresh.status()}

    @app.get("/recommendations/latest", response_model=RecommendationList)
    def latest_recommendations(limit: int = limit_query) -> RecommendationList:
        """Return the stocks to buy and sell at the next open, from the data on disk."""
        return latest.lists(limit)

    @app.post("/recommendations/latest", response_model=RecommendationList)
    def run_recommendations(
        limit: int = limit_query,
        fetch: bool = Query(False, description="Download new bars and news first (about 20 s)."),
        email: bool = Query(False, description="Email the lists to the configured recipients."),
    ) -> RecommendationList:
        """Optionally fetch new data, run the models, and optionally email the stocks to buy and sell.

        A failed fetch or email is reported in the response's fetch and email fields; the lists still come from
        the data on disk.
        """
        report = FetchReport(errors=refresh.run_once()) if fetch else None
        try:
            body = latest.lists(limit)
        except HTTPException as error:
            if report and report.errors:
                error.detail = f"{error.detail} The fetch failed: {'; '.join(report.errors)}"
            raise
        body.fetch = report
        if email:
            body.email = notify.send(body, settings)
        return body

    return app


app = create_app()
