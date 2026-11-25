"""Trade a backtest run's final choice in each of its test years under the served account rule, and save it beside the
run as account_rules.json, which the dashboard shows (dashboard.py ACCOUNT_RULES).

The served account rule (served_models.py, account.trade make_room) is the backtest's account with one change: an
order the cash does not cover first sells the oldest open trades of the served model with the worst return per day,
never one bought the same day, instead of being skipped. The worst return per day is each model's trading alone over
the run's test years, profit over the dollars its open lots cost at each close, added up (the run's picks.csv
tuning_return_per_day of its final choice). With one model there is nothing to make room with.

Each year's models are fit on the years before it, as the run fit them (walk-forward), with its costs and idle cash
kept as cash. The plain account, which skips such an order, and each model alone must reproduce the run's saved
yearly returns, or nothing is written. The choice was made on these years, so they are in sample.

The served rule's account and each model alone are also traded with their idle cash held in SPY instead
(account.trade sweep): an order spends that cash first, then sells trades. Those are the *_spy strategies.

The served rule's accounts also record parts_by_year: per year, the account's profit split into what each model's
lots made and what its idle cash made (account.contributions), gross, costs and net, over the capital. The nets add
up to the year's return.

  python scripts/account_rules.py [--run artifacts/cv/<stamp>]

The run is RECOMMENDER_MODEL_RUN from .env, the one the API serves, unless --run names another. It refits each test
year's models, so it takes about half an hour for most_liquid + news_boost.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from stock_recommender.backtest.account import contributions, sweep_growth, trade  # noqa: E402
from stock_recommender.backtest.registry import GRIDS, MODELS  # noqa: E402
from stock_recommender.backtest.run import (BacktestConfig, _candidates, _fit, _per_day, _trade_year,  # noqa: E402
                                            load_data, walk_forward)
from stock_recommender.config import Settings  # noqa: E402
from stock_recommender.dashboard import ACCOUNT_RULES  # noqa: E402

TOLERANCE = 1e-9  # Of a replayed yearly return against the run's.
IDLE = "SPY"  # Where the *_spy strategies hold their idle cash.
SETTINGS = ("test_years", "cost_bps", "capital", "order_value", "whole_shares", "min_price", "min_dollar_volume",
            "decide_every", "offset", "sentiment_model")


def config_of(manifest: dict, models: list[str]) -> BacktestConfig:
    """The run's trading settings, on the backtest's own data (2021-2025), for the models of its final choice."""
    saved = manifest["configuration"]
    values = {key: tuple(saved[key]) if isinstance(saved[key], list) else saved[key] for key in SETTINGS
              if key in saved}
    return BacktestConfig(models=tuple(models), **values)


def account_rules(folder: Path) -> dict:
    started = time.monotonic()
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    selection = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
    names = list(selection["combination"]["models"])
    final = {name: selection["models"][name]["candidate"] for name in names}
    config = config_of(manifest, names)
    _, panel, news = load_data(config)
    print(f"data loaded in {time.monotonic() - started:.0f}s", flush=True)
    settings = Settings(min_price=config.min_price, min_dollar_volume=config.min_dollar_volume)
    by_label = {candidate.label: candidate for candidate in _candidates(config, MODELS, GRIDS)}
    candidates = [by_label[final[name]] for name in names]
    books, alone, skip = {}, {name: {} for name in names}, {}
    for fold in walk_forward(panel.sessions, config.test_years):
        fitted, _ = _fit({c.label: (c.model, dict(c.params)) for c in candidates}, MODELS, panel, news, settings,
                         fold.train_end)
        runs = _trade_year(fold, candidates, fitted, panel, news, settings, config)
        books[fold.year] = [runs[final[name]].book for name in names]
        for name in names:
            alone[name][fold.year] = runs[final[name]].ledger
            saved = selection["models"][name]["return_by_year"][str(fold.year)]
            if abs(alone[name][fold.year].yearly_return - saved) > TOLERANCE:
                raise RuntimeError(f"{fold.year} {name}: replayed {alone[name][fold.year].yearly_return!r}, the run "
                                   f"saved {saved!r}.")
        skip[fold.year] = trade(panel.sessions, books[fold.year], config.cost_bps, config.whole_shares,
                                capital=config.capital)
        saved = selection["combination"]["account_return_by_year"][str(fold.year)]
        if abs(skip[fold.year].yearly_return - saved) > TOLERANCE:
            raise RuntimeError(f"{fold.year} account: replayed {skip[fold.year].yearly_return!r}, the run saved "
                               f"{saved!r}.")
        print(f"{fold.year} reproduced; {time.monotonic() - started:.0f}s", flush=True)
    years = sorted(books)
    per_day = {name: _per_day([alone[name][year] for year in years])["return_per_day"] for name in names}
    worst = min(per_day, key=per_day.get) if len(names) > 1 else None
    picks = folder / "tables/picks.csv"
    if worst is not None and picks.is_file():  # The run's own measure of it, on its final choice.
        table = pd.read_csv(picks)
        mine = table[(table.choice.astype(str) == "final") & table.chosen.astype(bool) & table.model.isin(names)]
        theirs = dict(zip(mine.model, mine.tuning_return_per_day))
        if set(theirs) == set(names) and min(theirs, key=theirs.get) != worst:
            raise RuntimeError(f"The run's picks.csv ranks {theirs}, this replay {per_day}.")
    spy = sweep_growth(panel, IDLE)
    if spy is None:
        raise RuntimeError(f"{IDLE} is not in the data: the idle cash cannot be held in it.")
    index = names.index(worst) if worst else None
    room, room_spy, alone_spy = {}, {}, {name: {} for name in names}
    for year in years:
        room[year] = trade(panel.sessions, books[year], config.cost_bps, config.whole_shares, record=True,
                           capital=config.capital, make_room=index)
        room_spy[year] = trade(panel.sessions, books[year], config.cost_bps, config.whole_shares, record=True,
                               capital=config.capital, make_room=index, sweep=spy)
        for name, item in zip(names, books[year]):
            plain = trade(panel.sessions, [item], config.cost_bps, config.whole_shares, capital=config.capital)
            if abs(plain.yearly_return - alone[name][year].yearly_return) > TOLERANCE:  # Its book is the run's.
                raise RuntimeError(f"{year} {name}: its book alone makes {plain.yearly_return!r}, the run "
                                   f"{alone[name][year].yearly_return!r}.")
            alone_spy[name][year] = trade(panel.sessions, [item], config.cost_bps, config.whole_shares,
                                          capital=config.capital, sweep=spy)

    def entry(label: str, by_year: dict, idle: str = "cash", extra: dict | None = None) -> dict:
        returns = {str(year): float(value) for year, value in by_year.items()}
        return {"label": label, "idle_cash": idle, "return_by_year": returns,
                "mean": float(np.mean(list(returns.values()))), **(extra or {})}

    def counts(ledgers: dict) -> dict:
        orders = {year: ledgers[year].orders for year in years}
        return {"filled": {str(y): int((orders[y].reason == "filled").sum()) for y in years},
                "no_cash": {str(y): int((orders[y].reason == "no_cash").sum()) for y in years},
                "sold_to_make_room": {str(y): int((orders[y].exit_rule == "make_room").sum()) for y in years}}

    def parts(ledgers: dict, sweep=None) -> dict:
        """Per year, what each model's lots and the idle cash made (gross, costs, net) over the capital."""
        by_year = {}
        for year in years:
            table = contributions(ledgers[year], panel.sessions, config.cost_bps, sweep) / config.capital
            if abs(table.profit.sum() - ledgers[year].yearly_return) > TOLERANCE:
                raise RuntimeError(f"{year}: the parts make {table.profit.sum()!r}, the account "
                                   f"{ledgers[year].yearly_return!r}.")
            by_year[str(year)] = {part: {"gross": float(row.gross), "costs": float(row.costs),
                                         "net": float(row.profit)} for part, row in table.iterrows()}
        return {"parts_by_year": by_year}

    rule = (f"An order the cash does not cover sells {worst}'s oldest open trades first" if worst
            else "An order the cash does not cover is skipped")
    in_spy = (f"Account, the served rule, its idle cash in {IDLE}: an order spends that cash first"
              + (f", then sells {worst}'s oldest open trades" if worst else ""))
    strategies = {
        "make_room": entry(f"Account, the served rule: {rule[0].lower() + rule[1:]}",
                           {year: room[year].yearly_return for year in years}, extra={**counts(room), **parts(room)}),
        "make_room_spy": entry(in_spy, {year: room_spy[year].yearly_return for year in years}, IDLE,
                               {**counts(room_spy), **parts(room_spy, spy)}),
        "skip": entry("Account, the backtest's rule: an order the cash does not cover is skipped",
                      {year: skip[year].yearly_return for year in years}),
    }
    for name in names:
        strategies[name] = entry(f"{name} alone", {year: alone[name][year].yearly_return for year in years})
        strategies[f"{name}_spy"] = entry(f"{name} alone, idle cash in {IDLE}",
                                          {year: alone_spy[name][year].yearly_return for year in years}, IDLE)
    strategies["spy_hold"] = entry("SPY, held", {int(year): value for year, value in
                                                 selection["nested_estimate"]["spy_hold_return"].items()
                                                 if int(year) in years}, "-")
    return {"run": folder.relative_to(ROOT).as_posix() if folder.is_relative_to(ROOT) else folder.as_posix(),
            "computed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "years": years,
            "cost_bps": config.cost_bps, "capital": config.capital, "order_value": config.order_value,
            "idle_cash": f"cash; in {IDLE} for the *_spy strategies", "make_room": worst, "return_per_day": per_day,
            "rule": rule + ".", "strategies": strategies}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="The run, artifacts/<label>/<stamp>; RECOMMENDER_MODEL_RUN by default.")
    args = parser.parse_args()
    run = args.run or Settings.from_env().model_run
    if not run:
        raise SystemExit("No run: pass --run or set RECOMMENDER_MODEL_RUN in .env.")
    folder = Path(run) if Path(run).is_absolute() else ROOT / run
    result = account_rules(folder.resolve())
    (folder / ACCOUNT_RULES).write_text(json.dumps(result, indent=2), encoding="utf-8")
    for key, item in result["strategies"].items():
        returns = "  ".join(f"{year} {value:+.1%}" for year, value in item["return_by_year"].items())
        print(f"{item['label']}: {returns}  mean {item['mean']:+.1%}")
    print(f"Saved to {folder / ACCOUNT_RULES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
