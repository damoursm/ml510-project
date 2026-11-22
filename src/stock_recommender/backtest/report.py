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
SUMMARY_COLUMNS = ["strategy", "capital", "profit", "mean_yearly_return", "daily_return", "volatility", "sharpe",
                   "max_drawdown", "avg_turnover", "ic_mean", "excess_return_vs_spy_hold", "null_return_percentile"]


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


def selection_table(result: BacktestResult) -> pd.DataFrame:
    """Return the final choice for the 2026 test: each model of the combination, its hyperparameters, its exit
    strategy and order value."""
    columns = ["model", "params", "picks", "horizon_days", "hold_days", "stop_loss", "take_profit", "volatility_exit",
               "order_value", "mean_yearly_return"]
    return pd.DataFrame([{"model": name, "params": json.dumps(choice["params"]), "picks": choice["picks"],
                          "horizon_days": choice["horizon_days"], **(choice.get("exit") or {}),
                          "order_value": choice["order_value"], "mean_yearly_return": choice["mean_yearly_return"]}
                         for name, choice in result.selection["models"].items()], columns=columns)


def combination_summary(result: BacktestResult) -> str:
    """Say which strategies the account traded in each test year, chosen on the test years before it, and which the
    final choice keeps for the 2026 test and the API."""
    def named(members) -> str:
        return ", ".join(members) if members else "nothing (no model beat holding SPY)"

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


def format_summary(result: BacktestResult) -> str:
    table = summary_table(result).copy()
    for column in ("mean_yearly_return", "volatility", "max_drawdown", "excess_return_vs_spy_hold"):
        if column in table:
            table[column] = table[column].map(_percent(1))
    if "daily_return" in table:
        table["daily_return"] = table.daily_return.map(_percent(3))
    for column in ("sharpe", "avg_turnover", "ic_mean", "null_return_percentile"):
        if column in table:
            table[column] = table[column].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    table["capital"] = table.capital.map(lambda x: f"{x:,.0f}")
    table["profit"] = table.profit.map(lambda x: f"{x:+,.0f}" if pd.notna(x) else "")
    years = yearly_table(result)
    for column in years.columns[1:]:
        years[column] = years[column].map(_percent(1))
    return (f"{table.to_string(index=False)}\n\nReturn per test year (profit over capital, "
            f"${result.config.order_value:,.0f} per order):\n{years.to_string(index=False)}\n\n"
            f"Models the account traded (only those whose mean yearly return beat SPY held qualify; then the set of "
            f"them with the best account):\n{combination_summary(result)}")


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
    axis.set_title(f"Profit from \\$0 at \\${result.config.order_value:,.0f} per order, test years added up, out of "
                   f"sample {result.config.test_years[0]}-{result.config.test_years[-1]}", loc="left", color=INK,
                   fontsize=12)
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
    run.save_table("combinations", result.combinations)
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
        f"years before it, and its hyperparameters are chosen on the test years before it. The account starts each "
        f"test year with ${result.config.capital:,.0f} and buys the stocks recommended by the models chosen for the "
        f"year in slices of ${result.config.order_value:,.0f}, at the 16:00 close of the day they are recommended, "
        f"while its cash lasts: on the test years before it, only the models whose mean yearly return beat SPY held "
        f"qualify, and the set of them whose account did best is traded. A return is the last equity over the "
        f"starting capital, minus 1.\n\n```\n{format_summary(result)}\n```\n\n"
        f"Final choice for the 2026 test and the API (selection.json):\n\n"
        f"```\n{selection_table(result).to_string(index=False)}\n```",
        inspect_first=["tables/summary.csv", "tables/yearly.csv", "selection.json", "tables/folds.csv",
                       "tables/combinations.csv", "figures/profit.png", "guards.json", "tables/scores.csv",
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
