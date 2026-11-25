"""HTTP API that serves the latest stock recommendations.

Run from the repository root, then open http://127.0.0.1:8000/docs:
    python -m uvicorn stock_recommender.api:app --app-dir src
Keep a single worker process: two would download into the same files at once.

GET /recommendations/latest returns the stocks to buy and sell at today's close (16:00 New York). POST to the same path does the
same after optionally fetching new data (fetch=true) and before optionally emailing the lists to the given
addresses (email=you@example.com). The API runs nothing on its own: every weekday at 16:00 the email task
(scripts/email_recommendations.py) calls it with both. With RECOMMENDER_MODEL_RUN set, the lists come from that
backtest run's final selection, the best combination of models (served_models.py), instead of load_models(), and
`account` says how an account trades them (Selection.account_rule): the API keeps no account itself.
GET /dashboard shows the page that every call of the recommendations writes again (dashboard.py).
"""
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
import logging
import threading
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from . import __version__, notify
from .config import Settings
from .dashboard import Dashboard
from .data import DataUnavailable, recent_day_files
from .exits import exit_rule
from .models import Models, load_models
from .recommend import Recommendations, buy_and_sell, pick_count, recommend
from .refresh import Refresh
from .schemas import ExitStrategy, FetchReport, RecommendationList
from .served_models import Selection, load_selection


def exit_strategy(models) -> ExitStrategy:
    """A served model's exit strategy, as the lists report it."""
    rule = exit_rule(models)
    return ExitStrategy(**rule.as_dict(), summary=rule.describe())

log = logging.getLogger("uvicorn.error")


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
        # The news files in the window, plus slack for days stored after as_of (weekends, lagging bars), and
        # their LLM scores.
        news_days = self.models.news_lookback_days + 5
        files = (recent_day_files(self.settings.bars_dir, self.models.bars_lookback)
                 + recent_day_files(self.settings.news_dir, news_days)
                 + recent_day_files(self.settings.sentiment_dir, news_days))
        return tuple((str(path), path.stat().st_mtime_ns) for path in files)

    def get(self) -> tuple[Recommendations, datetime]:
        with self._lock:
            if self._result is None or not self.refreshing.is_set():
                inputs = self._input_files()
                if inputs != self._inputs:
                    self._result = recommend(self.settings, self.models), datetime.now(timezone.utc)
                    self._inputs = inputs
            return self._result

    def lists(self, limit: int | None) -> RecommendationList:
        """Return the `limit` stocks to buy and to sell (None: as many as the model lists, recommend.pick_count);
        DataUnavailable becomes a 503."""
        try:
            result, generated_at = self.get()
        except DataUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        buy, sell = buy_and_sell(result.table, limit if limit is not None else pick_count(self.models,
                                                                                            len(result.table)))
        return RecommendationList(
            as_of=result.as_of.date(),
            news_cutoff=result.news_cutoff.to_pydatetime(),
            news_through=result.news_through.to_pydatetime() if result.news_through is not None else None,
            generated_at=generated_at,
            horizon_days=exit_rule(self.models).hold_days,
            exits={self.models.version: exit_strategy(self.models)},
            model_version=self.models.version,
            universe_size=result.universe_size,
            buy=buy.to_dict("records"),
            sell=sell.to_dict("records"),
        )

    def served(self, generated_at: datetime) -> list | None:
        """The result behind lists computed at generated_at, for the dashboard: (name, version, Recommendations,
        models) per model; None once newer lists replaced it."""
        held = self._result
        if held is None or held[1] != generated_at:
            return None
        return [(self.models.version, self.models.version, held[0], self.models)]


class RunRecommendations(LatestRecommendations):
    """The best combination of models: the final selection of the backtest run in settings.model_run
    (served_models.py), loaded on first use, so building the app reads nothing. Each model ranks the stocks from the
    newest files, again once the downloads add or rewrite a file one of them reads, and the lists hold each one's
    `limit` best and worst, labelled with its name: by default as many as the model lists, the number the backtest
    chose for it (its picks, or its pick_share of the stocks it ranked)."""

    def __init__(self, settings: Settings, fallback: Models):
        self._selection: Selection | None = None
        self._loading = threading.Lock()
        super().__init__(settings, fallback)  # Sets models: the fallback, whose windows serve until a model loads.

    @property
    def models(self) -> Models:
        """What the downloads and the change check cover: the widest windows among the served models."""
        loaded = self.selection().loaded
        if not loaded:
            return self._fallback
        return replace(self._fallback, bars_lookback=max(model.bars_lookback for model in loaded),
                       news_lookback_days=max(model.news_lookback_days for model in loaded))

    @models.setter
    def models(self, value: Models) -> None:
        self._fallback = value

    def selection(self) -> Selection:
        """The run's selection, loaded once; a model that does not load is logged and left out."""
        with self._loading:
            if self._selection is None:
                self._selection = load_selection(self.settings.model_run)
                for problem in self._selection.errors:
                    log.error("Model run: %s", problem)
            return self._selection

    def get(self) -> tuple[list, datetime]:
        with self._lock:
            if self._result is None or not self.refreshing.is_set():
                inputs = self._input_files()
                if inputs != self._inputs:
                    selection = self.selection()
                    if not selection.loaded:
                        raise DataUnavailable(f"No model of the served run loaded: {'; '.join(selection.errors)} "
                                              "Fix RECOMMENDER_MODEL_RUN or the run, then restart the API.")
                    self._result = selection.recommend(self.settings), datetime.now(timezone.utc)
                    self._inputs = inputs
            return self._result

    def lists(self, limit: int | None) -> RecommendationList:
        """Return each served model's `limit` stocks to buy and to sell (None: as many as that model lists);
        DataUnavailable becomes a 503."""
        try:
            results, generated_at = self.get()
        except DataUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        buy, sell = [], []
        for model, result in results:
            count = limit if limit is not None else pick_count(model.models, len(result.table))
            best, worst = buy_and_sell(result.table, count)
            buy += [dict(row, model=model.name) for row in best.to_dict("records")]
            sell += [dict(row, model=model.name) for row in worst.to_dict("records")]
        first = results[0][1]  # Every model reads the same newest files, so they share as_of.
        throughs = [result.news_through for _, result in results if result.news_through is not None]
        served = [model for model, _ in results]
        return RecommendationList(
            as_of=first.as_of.date(),
            news_cutoff=first.news_cutoff.to_pydatetime(),
            news_through=max(throughs).to_pydatetime() if throughs else None,
            generated_at=generated_at,
            horizon_days=max(exit_rule(model.models).hold_days for model in served),
            exits={model.name: exit_strategy(model.models) for model in served},
            account=self.selection().account_rule(),
            model_version=" + ".join(dict.fromkeys(model.version for model in served)),
            universe_size=max(result.universe_size for _, result in results),
            buy=buy,
            sell=sell,
            run=self.selection().run,
        )

    def served(self, generated_at: datetime) -> list | None:
        """The result behind lists computed at generated_at, for the dashboard: (name, version, Recommendations,
        models) per selected model; None once newer lists replaced it."""
        held = self._result
        if held is None or held[1] != generated_at:
            return None
        return [(model.name, model.version, result, model.models) for model, result in held[0]]


def recipients(email: str) -> list[str]:
    """Return the addresses in the email parameter; 422 for anything that is not one, such as email=true."""
    addresses = [address.strip() for address in email.split(",") if address.strip()]
    if not addresses or any("@" not in address or any(c.isspace() or c in "<>" for c in address)
                            for address in addresses):
        raise HTTPException(status_code=422, detail="email takes the addresses to send the lists to, comma-separated, "
                                                    "e.g. email=you@example.com.")
    return addresses


def create_app(settings: Settings | None = None, models: Models | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    models = models or load_models()
    # With a backtest run set, its final selection ranks instead: the best combination of models.
    latest = RunRecommendations(settings, models) if settings.model_run else LatestRecommendations(settings, models)
    refresh = Refresh(settings, latest)
    board = Dashboard(settings) if settings.dashboard_dir else None

    def record(method: str, limit: int | None, body: RecommendationList | None = None, error: str | None = None,
               fetch: bool = False, email: bool = False) -> None:
        """Record a call of the recommendations on the dashboard. A failure is logged, never raised: the lists
        come first."""
        if board is None:
            return
        try:
            served = latest.served(body.generated_at) if body is not None else None
            board.record(method, limit, body=body, served=served, error=error, fetch=fetch, email=email)
        except Exception:
            log.exception("Dashboard: could not record the %s call", method)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Compute first, so the first request is fast.
        try:
            latest.get()
        except DataUnavailable:
            pass
        yield

    app = FastAPI(title="ML510 stock recommendations", version=__version__, lifespan=lifespan)
    app.state.refresh = refresh
    limit_query = Query(settings.top_n, ge=1, le=500,
                        description="Stocks in each of the buy and sell lists; by default each served model's own "
                                    "number, the one its backtest chose (its picks, or its pick_share of the stocks it "
                                    "ranked), unless RECOMMENDER_TOP_N sets one for all.")

    @app.get("/health")
    def health() -> dict:
        """Report that the API is up, with the last fetch, and the served run if any."""
        report = {"status": "ok", "refresh": refresh.status()}
        if isinstance(latest, RunRecommendations):
            selection = latest.selection()
            report["run"] = {"run": selection.run, "selection_sha256": selection.sha256, "error": selection.error,
                             "models": {model.name: "loaded" if model.models is not None else model.error
                                        for model in selection.models},
                             "code_changed_since_run": list(selection.code_changed)}
        return report

    @app.get("/recommendations/latest", response_model=RecommendationList)
    def latest_recommendations(background: BackgroundTasks, limit: int | None = limit_query) -> RecommendationList:
        """Return the stocks to buy and sell at today's close, from the data on disk."""
        try:
            body = latest.lists(limit)
        except HTTPException as error:
            record("GET", limit, error=str(error.detail))
            raise
        background.add_task(record, "GET", limit, body)  # After the response: explaining reads the files again.
        return body

    @app.post("/recommendations/latest", response_model=RecommendationList)
    def run_recommendations(
        background: BackgroundTasks,
        limit: int | None = limit_query,
        fetch: bool = Query(False, description="Download new bars and news first (about 20 s)."),
        email: str | None = Query(None, description="Email the lists to these addresses, comma-separated."),
    ) -> RecommendationList:
        """Optionally fetch new data, run the models, and optionally email the stocks to buy and sell.

        A failed fetch or email is reported in the response's fetch and email fields; the lists still come from
        the data on disk.
        """
        to = recipients(email) if email is not None else None  # Checked before a fetch, not after.
        report = FetchReport(errors=refresh.run_once()) if fetch else None
        try:
            body = latest.lists(limit)
        except HTTPException as error:
            if report and report.errors:
                error.detail = f"{error.detail} The fetch failed: {'; '.join(report.errors)}"
            record("POST", limit, error=str(error.detail), fetch=fetch, email=to is not None)
            raise
        body.fetch = report
        if to:
            body.email = notify.send(body, settings, to)
            if body.email.sent_to:
                log.info("Recommendations emailed to %d recipient(s)", len(body.email.sent_to))
        background.add_task(record, "POST", limit, body)
        return body

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        """The dashboard: the last lists explained, the served run's backtest results, and the latest calls; every
        call of /recommendations/latest writes it again."""
        if board is None:
            return HTMLResponse("<p>The dashboard is off: RECOMMENDER_DASHBOARD_DIR is empty.</p>", status_code=404)
        return HTMLResponse(board.page())

    return app


app = create_app()
