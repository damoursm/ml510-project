"""The best combination of models, as the API serves it: the final selection of a pinned backtest run.

A backtest (scripts/backtest.py) ends with its final choice, selection.json: one model per strategy with the
hyperparameters chosen on 2022-2025, fit on 2021-2025 and pickled under model/<name>.pkl, with the filters the run used.
With RECOMMENDER_MODEL_RUN naming that run, the API serves those models in place of load_models(): each one ranks the
stocks from the newest files through recommend(), with the run's filters and news inputs, and the lists hold each
one's best and worst. Nothing is refit here; serve a better model by pinning its run. The backtest never imports this
module (tests/test_backtest.py), since it loads models fit on the test years.
"""
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
import pandas as pd
from .backtest.window import FIRST_DAY
from .config import ROOT, Settings
from .exits import exit_rule
from .recommend import Recommendations, recommend
from .schemas import AccountRule

TEST_YEAR = 2026  # The reserved final test: a served selection was chosen and fit on the years before it.
# /health lists those changed since the run.
DECISION_PATH = ("recommend.py", "models.py", "data.py", "exits.py")
SIZES = ("horizon_days", "bars_lookback", "news_lookback_days", "picks")
IDLE_CASH = "SPY"  # The served rule holds the cash not in trades in SPY, so none of it sits idle (account.trade sweep).
MODEL_NAME = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class SelectedModel:
    """One model of the selection: its entry in selection.json, and its pickle once loaded."""
    name: str
    version: str
    horizon_days: int
    bars_lookback: int
    news_lookback_days: int
    picks: int
    source: str  # The saved_models entry, e.g. "model/most_liquid.pkl".
    sha256: str | None
    models: object | None  # The unpickled Models; None when it did not load.
    error: str | None
    pick_share: float | None = None  # In place of picks: the share of the stocks it ranks that it lists.


@dataclass(frozen=True)
class Selection:
    """A backtest run's final choice, as the API serves it; values it could not read are None."""
    run: str  # The run folder, posix, relative to the repository when inside it.
    sha256: str | None  # Of its selection.json.
    chosen_on_years: tuple[int, ...]
    training_end: str | None
    min_price: float | None
    min_dollar_volume: float | None
    sentiment_model: str | None  # None: the run read the news without LLM scores.
    models: tuple[SelectedModel, ...]
    code_changed: tuple[str, ...]  # DECISION_PATH files whose sha256 differs from the run's manifest.
    error: str | None
    capital: float | None = None  # The run's account: its starting cash, order value and costs.
    order_value: float | None = None
    cost_bps: float | None = None
    # The served account rule (_make_room): the model whose oldest trades an order the cash does not cover sells first,
    # and each loaded model's return per day it was chosen by, (name, value).
    make_room: str | None = None
    return_per_day: tuple = ()

    @property
    def loaded(self) -> tuple[SelectedModel, ...]:
        """The models that loaded: the ones the API ranks with."""
        return tuple(model for model in self.models if model.models is not None)

    @property
    def errors(self) -> list[str]:
        """The selection's error, then each model's that did not load."""
        return ([self.error] if self.error else []) + [f"{m.name} did not load ({m.error})" for m in self.models
                                                       if m.error]

    def settings(self, settings: Settings) -> Settings:
        """The API's settings with the run's filters and news inputs, as run_backtest gives them to its models:
        news without LLM scores when the run read none."""
        return replace(settings, min_price=self.min_price, min_dollar_volume=self.min_dollar_volume,
                       sentiment_dir=settings.sentiment_dir if self.sentiment_model else None,
                       ollama_model=self.sentiment_model or settings.ollama_model)

    def recommend(self, settings: Settings) -> list[tuple[SelectedModel, Recommendations]]:
        """Rank the stocks with each loaded model from the newest files on disk, in the selection's order."""
        served = self.settings(settings)
        return [(model, recommend(served, model.models)) for model in self.loaded]

    def account_rule(self) -> AccountRule | None:
        """How an account trades the buy lists, the guide the API states beside them (it keeps no account): the run's
        backtest account, buying each listed stock for order_value at the close of its day, each model's best first
        and the models in turn, and selling each by its model's exit, except that an order the cash does not cover
        first sells make_room's oldest trades (account.trade make_room), and that the cash not in trades is held in
        IDLE_CASH from each close to the next (account.trade sweep). None when the run records no account."""
        if not (self.capital and self.order_value and self.cost_bps is not None):
            return None
        if self.make_room:
            room = (f"When an order finds no cash, first sell {self.make_room}'s oldest trades, never one bought the "
                    f"same day: {self.make_room} earned the least per invested dollar per day in the backtest.")
        else:
            room = "When an order finds no cash, skip it."
        summary = (f"From ${self.capital:,.0f}, buy ${self.order_value:,.0f} of each stock to buy at today's close, "
                   f"each model's #1 first, then each one's #2, and so on, and sell each when its model's exit says. "
                   f"Hold the idle cash in {IDLE_CASH} from each close to the next: an order spends it first. {room} "
                   f"Costs: {self.cost_bps:g} bps a side, on {IDLE_CASH} bought and sold too.")
        return AccountRule(capital=self.capital, order_value=self.order_value, cost_bps=self.cost_bps,
                           make_room=self.make_room, idle_cash=IDLE_CASH, summary=summary)


def _make_room(folder: Path, names: tuple[str, ...]) -> tuple[str | None, tuple]:
    """The served account rule's model: of the served models, the one with the worst return per day trading alone
    over the run's test years, as the run measured its final choice (tables/picks.csv tuning_return_per_day), with
    each one's value. None with one model, or when the run lacks those values."""
    if len(names) < 2:
        return None, ()
    try:
        table = pd.read_csv(folder / "tables" / "picks.csv")
        final = table[(table.choice.astype(str) == "final") & (table.chosen.astype(str).str.lower() == "true")]
        found = {str(model): float(value) for model, value in zip(final.model, final.tuning_return_per_day)}
    except (OSError, ValueError, KeyError, AttributeError, pd.errors.ParserError):
        return None, ()
    if not all(name in found and math.isfinite(found[name]) for name in names):
        return None, ()
    scores = tuple((name, found[name]) for name in names)
    return min(scores, key=lambda pair: pair[1])[0], scores


def _shown(path: Path) -> str:
    """A path as the API shows it: posix, relative to the repository when inside it."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _load_model(folder: Path, name, entry, saved) -> SelectedModel:
    """Unpickle one selected model and check it is the model its entry describes; any failure is its error."""
    entry = entry if isinstance(entry, dict) else {}
    try:
        sizes = {key: int(entry[key]) for key in SIZES}
    except (KeyError, TypeError, ValueError):
        sizes = dict.fromkeys(SIZES, 0)
    share = entry.get("pick_share")  # Runs saved before picks were tuned record none: they list their picks.
    shared = share is None or (isinstance(share, (int, float)) and not isinstance(share, bool) and 0 < share < 1)
    model = SelectedModel(str(name), str(entry.get("version", "")), **sizes,
                          source=saved if isinstance(saved, str) else "", sha256=None, models=None, error=None,
                          pick_share=float(share) if shared and share is not None else None)
    if not MODEL_NAME.fullmatch(model.name):
        return replace(model, error=f"{model.name!r} is not a model name of letters, digits, '_', '.' or '-'")
    if min(model.horizon_days, model.bars_lookback, model.picks) < 1 or model.news_lookback_days < 0:
        return replace(model, error="its selection entry lacks a valid horizon_days, bars_lookback, "
                                    "news_lookback_days or picks")
    if not shared:
        return replace(model, error=f"its selection entry's pick_share {share!r} is not a share between 0 and 1")
    pickled = folder / saved if isinstance(saved, str) else None
    if not (pickled is not None and saved.endswith(".pkl") and not Path(saved).is_absolute()
            and pickled.resolve().is_relative_to(folder)):
        reason = re.fullmatch(r"not saved \((.+?)\).*", saved) if isinstance(saved, str) else None
        detail = reason.group(1) if reason else (f"{saved!r} is not a .pkl file in the run folder" if saved
                                                 else "no saved_models entry")
        return replace(model, error=f"not saved ({detail}); make it picklable and rerun scripts/backtest.py")
    try:
        content = pickled.read_bytes()
        loaded = pickle.loads(content)
        if not callable(getattr(loaded, "score", None)):
            raise TypeError(f"a {type(loaded).__name__} is not a Models")
        for key in ("version", *SIZES, "pick_share"):
            if getattr(loaded, key, None) != getattr(model, key):
                raise ValueError(f"its {key} is {getattr(loaded, key, None)!r}, not the selection's "
                                 f"{getattr(model, key)!r}")
        # Runs saved before exit settings existed record none; their models hold for their horizon.
        expected = entry.get("exit")
        if expected is not None and exit_rule(loaded).as_dict() != expected:
            raise ValueError(f"its exit strategy is {exit_rule(loaded).as_dict()}, not the selection's {expected}")
    except Exception as error:  # Missing, truncated, unimportable or another model: only this one is unavailable.
        return replace(model, error=f"{saved}: {type(error).__name__}: {error}")
    return replace(model, sha256=hashlib.sha256(content).hexdigest(), models=loaded)


def load_selection(spec: str) -> Selection:
    """Read a backtest run's final selection (artifacts/<label>/<stamp>, or its selection.json; relative to the
    repository) and unpickle its models. Never raises: a failure is the selection's error, or one model's."""
    text = str(spec or "").strip()
    folder = None
    if text:
        folder = (Path(text) if Path(text).is_absolute() else ROOT / text).resolve()
        folder = folder.parent if folder.name == "selection.json" else folder
    run = _shown(folder) if folder is not None else ""
    blank = Selection(run, None, (), None, None, None, None, (), (), None)
    if folder is None:
        return replace(blank, error="RECOMMENDER_MODEL_RUN is empty: set it to a backtest run, "
                                    "artifacts/<label>/<stamp>.")
    if not ((folder / "selection.json").is_file() and (folder / "manifest.json").is_file()
            and folder.parent.parent.name == "artifacts"):
        return replace(blank, error=f"{run} is not a backtest run: a folder artifacts/<label>/<stamp> holding "
                                    "selection.json and manifest.json.")
    try:
        content = (folder / "selection.json").read_bytes()
        selection = json.loads(content)
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        configuration = manifest.get("configuration") or {}
        known = replace(blank, sha256=hashlib.sha256(content).hexdigest(),
                        chosen_on_years=tuple(int(year) for year in selection.get("chosen_on_years") or ()),
                        training_end=selection.get("training_end"), min_price=float(selection["min_price"]),
                        min_dollar_volume=float(selection["min_dollar_volume"]),
                        sentiment_model=configuration.get("sentiment_model")
                        if configuration.get("sentiment_dir") else None,
                        **{key: float(selection[key]) if selection.get(key) is not None else None
                           for key in ("capital", "order_value", "cost_bps")})
        entries, saved = selection.get("models"), selection.get("saved_models") or {}
        sources = manifest.get("source_sha256") or {}
    except Exception as error:
        return replace(blank, error=f"cannot read the selection of {run} ({type(error).__name__}: {error}).")
    # A served run was chosen on test years within 2022-2025 and fit on data before 2026. The 2026 final test measures
    # that same choice, so a run it evaluated is still served.
    problems = []
    if not (isinstance(known.training_end, str) and known.training_end < f"{TEST_YEAR}-01-01"):
        problems.append(f"its models were fit through {known.training_end}, not before {TEST_YEAR}")
    allowed = range(FIRST_DAY.year + 1, TEST_YEAR)
    if not known.chosen_on_years or any(year not in allowed for year in known.chosen_on_years):
        problems.append(f"it was chosen on {list(known.chosen_on_years)}, not within "
                        f"{allowed.start}..{allowed.stop - 1}")
    if not (isinstance(entries, dict) and entries):
        problems.append("it holds no model")
    if problems:
        return replace(known, error=f"{run} cannot be served: {'; '.join(problems)}.")
    saved = saved if isinstance(saved, dict) else {}
    models = tuple(_load_model(folder, name, entry, saved.get(name)) for name, entry in entries.items())
    # RunArtifacts keys the sources by path relative to the repository, with backslashes on Windows.
    recorded = ({str(key).replace("\\", "/"): value for key, value in sources.items()}
                if isinstance(sources, dict) else {})
    changed = tuple(name for name in DECISION_PATH if (key := f"src/stock_recommender/{name}") in recorded
                    and _file_sha256(ROOT / key) != recorded[key])
    room, per_day = _make_room(folder, tuple(model.name for model in models if model.models is not None))
    return replace(known, models=models, code_changed=changed, make_room=room, return_per_day=per_day)
