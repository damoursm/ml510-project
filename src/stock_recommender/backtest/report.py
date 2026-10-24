"""Write a backtest's results to artifacts/ (through ml_lib's RunArtifacts) and format its summary."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from ml_lib.experiment.artifacts import RunArtifacts
from ..config import ROOT
from .run import BacktestResult

# The reference categorical palette, in its fixed order (validated for adjacent lines).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SUMMARY_COLUMNS = ["strategy", "final_value", "total_return", "cagr", "volatility", "sharpe", "max_drawdown",
                   "avg_turnover", "ic_mean", "excess_return_vs_universe_ew", "null_sharpe_percentile"]


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


def format_summary(result: BacktestResult) -> str:
    table = summary_table(result).copy()
    for column in ("total_return", "cagr", "volatility", "max_drawdown", "excess_return_vs_universe_ew"):
        if column in table:
            table[column] = table[column].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "")
    for column in ("sharpe", "avg_turnover", "ic_mean", "null_sharpe_percentile"):
        if column in table:
            table[column] = table[column].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    table["final_value"] = table.final_value.map(lambda x: f"{x:,.0f}")
    return table.to_string(index=False)


def growth_figure(result: BacktestResult):
    """Plot each strategy's and benchmark's value at every open, on one log-scaled axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator

    columns = list(result.equity.columns)[:len(SERIES)]  # A ninth series would need another hue: leave it out.
    figure, axis = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    for color, column in zip(SERIES, columns):
        axis.plot(result.equity.index, result.equity[column], color=color, linewidth=2, label=column,
                  solid_joinstyle="round", solid_capstyle="round")
    axis.set_yscale("log")
    values = result.equity[columns].to_numpy()
    ticks = MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]).tick_values(np.nanmin(values), np.nanmax(values))
    axis.yaxis.set_major_locator(FixedLocator([tick for tick in ticks if tick > 0]))
    axis.yaxis.set_minor_locator(NullLocator())
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.grid(True, which="major", color=GRID, linewidth=1)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(colors=INK_SECONDARY, length=0)
    axis.set_title(f"Value of {result.config.capital:,.0f} invested, out of sample "
                   f"{result.config.test_years[0]}-{result.config.test_years[-1]}", loc="left", color=INK, fontsize=12)
    axis.set_ylabel("Value at the open (log scale)", color=INK_SECONDARY)
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
    run.record("guards", _plain(result.guards))
    run.record("versions", _plain(result.versions))
    run.save_table("summary", summary_table(result))
    run.save_table("metrics", result.metrics)
    run.save_table("equity", result.equity.rename_axis("date"), index=True)
    run.save_table("folds", result.folds)
    run.save_table("periods", result.periods)
    run.save_table("positions", result.positions)
    run.save_table("signals", result.signals)
    run.save_table("null", result.null)
    run.save_table("anomalies", result.anomalies)
    run.save_figure("growth", growth_figure(result), close=True)
    run.write_overview(
        "Backtest",
        f"Walk-forward backtest on 2021-2025 data only (never 2026), out of sample "
        f"{result.config.test_years[0]}-{result.config.test_years[-1]}.\n\n```\n{format_summary(result)}\n```",
        inspect_first=["tables/summary.csv", "figures/growth.png", "guards.json", "tables/metrics.csv"])
    return run.directory
