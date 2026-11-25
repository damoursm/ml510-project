"""Write a backtest's results to artifacts/ (through ml_lib's RunArtifacts) and format its summary."""
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import tempfile
import numpy as np
import pandas as pd
from ml_lib.experiment.artifacts import RunArtifacts
from ..config import ROOT
from .run import BacktestResult

# The reference categorical palette, in its fixed order (validated for adjacent lines).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SUMMARY_COLUMNS = ["strategy", "capital", "profit", "mean_yearly_return", "daily_return", "return_per_day",
                   "invested_share", "volatility", "sharpe", "max_drawdown", "avg_turnover", "ic_mean",
                   "excess_return_vs_spy_hold", "null_return_percentile"]


def _plain(value):
    """Make a value JSON-safe: NaN becomes null, dates and paths become strings."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (Path, pd.Timestamp)) or hasattr(value, "isoformat"):
        return str(value)
    return value


def summary_table(result: BacktestResult) -> pd.DataFrame:
    table = result.metrics[result.metrics.scope == "all"]
    return table[[column for column in SUMMARY_COLUMNS if column in table]].reset_index(drop=True)


def yearly_table(result: BacktestResult) -> pd.DataFrame:
    """Return each strategy's return in every test year (its profit over its capital), and their mean."""
    years = result.metrics[result.metrics.scope != "all"]
    order = list(dict.fromkeys(years.strategy))
    table = years.pivot(index="strategy", columns="scope", values="yearly_return").reindex(order)
    overall = result.metrics[result.metrics.scope == "all"].set_index("strategy")
    table["mean"] = overall.mean_yearly_return.reindex(order)
    return table.rename_axis(columns=None).reset_index()


def idle_cash_table(result: BacktestResult) -> pd.DataFrame:
    """Return the account's, then each strategy's, return in every test year with its idle cash kept as cash or held
    in each ticker of config.idle_cash, and their mean (NaN when a year has none)."""
    table = result.idle_cash
    if table is None or not len(table):
        return pd.DataFrame()
    wide = table.pivot_table(index=["strategy", "idle_cash"], columns="year", values="yearly_return", dropna=False)
    wide.columns = [str(year) for year in wide.columns]
    wide["mean"] = wide.mean(axis=1, skipna=False)
    if "max_drawdown" in table:  # The deepest fall within any one year, on the capital.
        wide["worst_drawdown"] = table.groupby(["strategy", "idle_cash"]).max_drawdown.min()
    strategies = list(dict.fromkeys(table.strategy))  # The account first, then the strategies in their order.
    places = list(dict.fromkeys(table.idle_cash))  # Cash first, then the tickers in config order.
    order = [(strategy, place) for strategy in strategies for place in places if (strategy, place) in wide.index]
    return wide.reindex(order).reset_index()


def per_day_table(result: BacktestResult) -> pd.DataFrame:
    """Return each strategy's return per day in every test year (its profit over the dollars it had invested at each
    close, added up: what an invested dollar earned per day held), and over all of them together."""
    if "return_per_day" not in result.metrics:
        return pd.DataFrame()
    years = result.metrics[result.metrics.scope != "all"]
    order = list(dict.fromkeys(years.strategy))
    table = years.pivot(index="strategy", columns="scope", values="return_per_day").reindex(order)
    table["all"] = result.metrics[result.metrics.scope == "all"].set_index("strategy").return_per_day.reindex(order)
    return table.rename_axis(columns=None).reset_index()


def picks_summary(result: BacktestResult) -> str:
    """Say how many stocks a day each model listed in each tuned test year and in the final choice: the list size
    with the best return per day on the years before (run.picks_table), beside the default's; then, for a test year,
    how both did in it and which list size did best there."""
    table = result.picks
    if table is None or not len(table):
        return ""
    per_day, on_capital, lines = _percent(3), _percent(1), []

    def named(row) -> str:
        if not row.picks.endswith("%"):
            return f"{row.picks} a day"
        return f"{row.picks} (~{row.stocks_per_list:.0f} a day)" if pd.notna(row.stocks_per_list) else row.picks

    for (choice, model), rows in table.groupby(["choice", "model"], sort=False):
        mine, default = rows[rows.chosen].iloc[0], rows.iloc[0]
        changed = mine.candidate != default.candidate
        name = "2026 test and the API" if choice == "final" else choice
        line = f"  {name} {model}: {named(mine)}, {per_day(mine.tuning_return_per_day)}/day on {mine.chosen_on}"
        line += f" ({named(default)}: {per_day(default.tuning_return_per_day)})" if changed else " (the default)"
        if choice != "final":
            line += (f"; in {choice} {per_day(mine.test_return_per_day)}/day, {on_capital(mine.test_yearly_return)} "
                     f"on the capital")
            if changed:
                line += (f" ({named(default)}: {per_day(default.test_return_per_day)}/day, "
                         f"{on_capital(default.test_yearly_return)})")
            tested = rows[rows.test_return_per_day.notna()]
            if len(tested):
                best = tested.loc[tested.test_return_per_day.astype(float).idxmax()]
                line += ("; the best size there" if best.candidate == mine.candidate
                         else f"; best there: {named(best)}, {per_day(best.test_return_per_day)}/day")
        lines.append(line)
    return "\n".join(lines)


def selection_table(result: BacktestResult) -> pd.DataFrame:
    """Return the final choice for the 2026 test: each model of the combination, its hyperparameters, how many stocks
    it lists (picks, or a pick_share of those it ranks in its place), its exit strategy and order value."""
    columns = ["model", "params", "picks", "pick_share", "horizon_days", "hold_days", "stop_loss", "take_profit",
               "volatility_exit", "order_value", "mean_yearly_return"]
    return pd.DataFrame([{"model": name, "params": json.dumps(choice["params"]), "picks": choice["picks"],
                          "pick_share": choice.get("pick_share"),
                          "horizon_days": choice["horizon_days"], **(choice.get("exit") or {}),
                          "order_value": choice["order_value"], "mean_yearly_return": choice["mean_yearly_return"]}
                         for name, choice in result.selection["models"].items()], columns=columns)


def standing_table(result: BacktestResult) -> pd.DataFrame:
    """Return how each strategy's trades stood against SPY bought with the same dollars on the same dates, over
    every test year: the final choice's test, which keeps every strategy that beats it."""
    columns = ["strategy", "mean_yearly_return", "spy_mirror_mean_return", "beats_spy", "in_combination"]
    final = (result.selection.get("combination") or {}).get("strategies") or {}
    return pd.DataFrame([{"strategy": name, **{key: entry.get(key) for key in columns[1:]}}
                         for name, entry in final.items()], columns=columns)


def combination_summary(result: BacktestResult) -> str:
    """Say which strategies the account traded in each test year, chosen on the test years before it, and which the
    final choice keeps for the 2026 test and the API."""
    def named(members) -> str:
        return ", ".join(members) if members else "nothing (no model's trades beat SPY)"

    lines = []
    for row in result.folds.itertuples():
        basis = f"chosen on {row.validation_years}" if row.validation_years else "every model: no earlier test year"
        lines.append(f"  {row.fold}: {named(json.loads(row.combination))} ({basis})")
    final = result.selection.get("combination") or {}
    years = final.get("chosen_on_years") or []
    lines.append(f"  2026 test and the API: {named(final.get('models') or [])} (chosen on "
                 f"{'-'.join(map(str, years)) or 'no year'})")
    return "\n".join(lines)


def _percent(digits: int):
    return lambda x: f"{x:+.{digits}%}" if pd.notna(x) else ""


def validation_summary(result: BacktestResult) -> str:
    """Say how each test year's account, its chosen candidates trading together, did in the years that chose it,
    and how the final choice did in every test year: in sample, since those years chose them, unlike the account's
    own row, which only ever trades a year its choice did not see."""
    percent, lines = _percent(1), []

    def line(name: str, members, returns: dict, mean, spy) -> str:
        held = f", SPY held {percent(spy)}" if pd.notna(spy) else ""  # Data without SPY has nothing to show.
        return (f"  {name} ({', '.join(members) or 'nothing'}): "
                + ", ".join(f"{year} {percent(value)}" for year, value in returns.items())
                + f"; mean {percent(mean)}{held}")

    for row in result.folds.itertuples():
        checked = json.loads(getattr(row, "account_validation_returns", None) or "{}")
        if checked:
            lines.append(line(str(row.fold), json.loads(row.combination), checked, row.account_validation_return,
                              row.spy_hold_validation_return))
    final = result.selection.get("combination") or {}
    returns = final.get("account_return_by_year") or {}
    if returns:
        held = [value for value in (result.selection.get("nested_estimate") or {}).get("spy_hold_return", {}).values()
                if value is not None and pd.notna(value)]
        lines.append(line("2026 test and the API", final.get("models") or [], returns,
                          final.get("account_mean_yearly_return"), float(np.mean(held)) if held else np.nan))
    return "\n".join(lines)


def format_summary(result: BacktestResult) -> str:
    table = summary_table(result).copy()
    for column in ("mean_yearly_return", "volatility", "max_drawdown", "excess_return_vs_spy_hold"):
        if column in table:
            table[column] = table[column].map(_percent(1))
    for column in ("daily_return", "return_per_day"):
        if column in table:
            table[column] = table[column].map(_percent(3))
    if "invested_share" in table:
        table["invested_share"] = table.invested_share.map(lambda x: f"{x:.0%}" if pd.notna(x) else "")
    for column in ("sharpe", "avg_turnover", "ic_mean", "null_return_percentile"):
        if column in table:
            table[column] = table[column].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    table["capital"] = table.capital.map(lambda x: f"{x:,.0f}")
    table["profit"] = table.profit.map(lambda x: f"{x:+,.0f}" if pd.notna(x) else "")
    years = yearly_table(result)
    for column in years.columns[1:]:
        years[column] = years[column].map(_percent(1))
    standing = standing_table(result)
    for column in ("mean_yearly_return", "spy_mirror_mean_return"):
        standing[column] = standing[column].map(_percent(1))
    against = (f"\n\nEach model's trades against SPY bought with the same dollars on the same dates, over every test "
               f"year (the final choice keeps those that beat it):\n{standing.to_string(index=False)}"
               if len(standing) else "")
    checked = validation_summary(result)
    validated = (f"\n\nEach test year's account evaluated in the years that chose it, and the final choice's in "
                 f"every test year (in sample: what the choice saw, not what to expect; the account's row above is the "
                 f"out-of-sample result):\n{checked}" if checked else "")
    days = per_day_table(result)
    for column in days.columns[1:]:
        days[column] = days[column].map(_percent(3))
    per_day = (f"\n\nReturn per day on the dollars invested (profit over what the open lots cost at each close, "
               f"added up; idle cash left out):\n{days.to_string(index=False)}" if len(days) else "")
    sizes = picks_summary(result)
    sized = (f"\n\nStocks a day, chosen by return per day on the test years before (the other settings chosen at the "
             f"default list size by mean yearly return), and how they did the year after:\n{sizes}" if sizes else "")
    idle = idle_cash_table(result)
    for column in idle.columns[2:]:
        idle[column] = idle[column].map(_percent(1))
    swept = (f"\n\nIdle cash kept as cash, or held in {' or '.join(result.config.idle_cash)} (BIL: Treasury bills, "
             f"for the interest), per test year; the same orders, out of sample, chosen as above:\n"
             f"{idle.to_string(index=False)}" if len(idle) else "")
    return (f"{table.to_string(index=False)}\n\nReturn per test year (last equity over the "
            f"${result.config.capital:,.0f} capital, minus 1; ${result.config.order_value:,.0f} per order):\n"
            f"{years.to_string(index=False)}{per_day}{sized}{swept}\n\n"
            f"Models the account traded (every model whose trades beat SPY bought with the same dollars on the same "
            f"dates, over the test years before):\n{combination_summary(result)}{against}{validated}")


def profit_figure(result: BacktestResult):
    """Plot each strategy's and reference's profit at every close, from its starting capital, test years added up."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    columns = list(result.profit.columns)[:len(SERIES)]  # A ninth series would need another hue: leave it out.
    figure, axis = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    axis.axhline(0, color=INK_SECONDARY, linewidth=1)
    for color, column in zip(SERIES, columns):
        series = result.profit[column].dropna()
        axis.plot(series.index, series, color=color, linewidth=2, label=column, solid_joinstyle="round",
                  solid_capstyle="round")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:+,.0f}" if value else "0"))
    axis.grid(True, which="major", color=GRID, linewidth=1)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(colors=INK_SECONDARY, length=0)
    # A bare $ starts matplotlib's math text: escape it.
    axis.set_title(f"Profit on \\${result.config.capital:,.0f} a test year in \\${result.config.order_value:,.0f} "
                   f"slices, test years added up, out of sample "
                   f"{result.config.test_years[0]}-{result.config.test_years[-1]}", loc="left", color=INK, fontsize=12)
    axis.set_ylabel("Profit at the close (\\$)", color=INK_SECONDARY)
    legend = axis.legend(frameon=False, loc="upper left", ncols=min(len(columns), 4))
    for text in legend.get_texts():
        text.set_color(INK)
    figure.tight_layout()
    return figure


def write_run(result: BacktestResult, label: str = "backtest", root: Path = ROOT) -> Path:
    """Save the run under artifacts/<label>/<UTC stamp>-<id>/ and return that folder."""
    with tempfile.TemporaryDirectory() as folder:
        inputs = Path(folder) / "inputs.json"
        inputs.write_text(json.dumps(_plain(result.guards), sort_keys=True), encoding="utf-8")
        config = _plain(asdict(result.config))
        run = RunArtifacts(root, label, config=config, dataset_path=inputs)
    selection = dict(result.selection)
    selection["saved_models"] = _save_models(run.directory / "model", result.final_models)
    run.record("selection", _plain(selection))
    run.record("guards", _plain(result.guards))
    run.record("versions", _plain(result.versions))
    run.save_table("summary", summary_table(result))
    run.save_table("yearly", yearly_table(result))
    run.save_table("metrics", result.metrics)
    run.save_table("scores", result.scores)
    run.save_table("against_spy", result.against_spy)
    run.save_table("picks", result.picks)
    run.save_table("idle_cash", result.idle_cash)
    run.save_table("profit", result.profit.rename_axis("date"), index=True)
    run.save_table("folds", result.folds)
    run.save_table("orders", result.orders)
    run.save_table("periods", result.periods)
    run.save_table("positions", result.positions)
    run.save_table("signals", result.signals)
    run.save_table("null", result.null)
    run.save_table("anomalies", result.anomalies)
    run.save_figure("profit", profit_figure(result), close=True)
    run.write_overview(
        "Backtest",
        f"Nested walk-forward backtest on 2021-2025 data only (never 2026), out of sample "
        f"{result.config.test_years[0]}-{result.config.test_years[-1]}: each year's models are trained on the "
        f"years before it, and its hyperparameters are chosen on the test years before it, at the default list size "
        f"by mean yearly return, then its list size by return per day (tables/picks.csv). The account starts each "
        f"test year with ${result.config.capital:,.0f} and buys the stocks recommended by the models chosen for the "
        f"year in slices of ${result.config.order_value:,.0f}, at the 16:00 close of the day they are recommended, "
        f"while its cash lasts: every model whose trades beat SPY bought with the same dollars on the same dates, on "
        f"the test years before it, is traded. A return is the last equity over the starting capital, minus 1."
        f"\n\n```\n{format_summary(result)}\n```\n\n"
        f"Final choice for the 2026 test and the API (selection.json):\n\n"
        f"```\n{selection_table(result).to_string(index=False)}\n```",
        inspect_first=["tables/summary.csv", "tables/yearly.csv", "selection.json", "tables/folds.csv",
                       "tables/picks.csv", "tables/idle_cash.csv", "tables/against_spy.csv", "figures/profit.png",
                       "guards.json",
                       "tables/scores.csv",
                       "tables/metrics.csv"])
    return run.directory


def _save_models(folder: Path, models: dict) -> dict:
    """Pickle the models fit on 2021-2025 for the 2026 test; one that does not pickle is refit from its params."""
    saved = {}
    for name, model in models.items():
        try:
            content = pickle.dumps(model)
        except Exception as error:  # Lambdas and local classes do not pickle.
            saved[name] = f"not saved ({type(error).__name__}); refit from the params"
            continue
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{name}.pkl").write_bytes(content)
        saved[name] = f"model/{name}.pkl"
    return saved
