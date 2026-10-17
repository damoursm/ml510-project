"""Create figures for exploring data and prediction errors.

Each function returns a figure for the caller to save or display.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def use_course_theme():
    sns.set_theme(style="whitegrid", context="notebook")


def target_distribution(values, *, title="Training target distribution", unit="target units"):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(x=np.asarray(values), bins=40, color=sns.color_palette()[0], ax=ax)
    ax.set(title=title, xlabel=unit, ylabel="Observed hours")
    fig.tight_layout(); return fig


def feature_relationship(frame, feature, target):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=frame, x=feature, y=target, alpha=.25, s=14, color=sns.color_palette()[0], ax=ax)
    ax.set(title=f"Training data: {feature} and {target}")
    fig.tight_layout(); return fig


def calendar_heatmap(frame, target, *, unit="Mean target value"):
    table = frame.assign(hour=frame.index.hour, weekday=frame.index.dayofweek).pivot_table(
        index="weekday", columns="hour", values=target, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(11, 4))
    sns.heatmap(table.reindex(index=range(7), columns=range(24)), ax=ax,
                cbar_kws={"label": unit})
    ax.set(title="Training data: average hourly profile", ylabel="Weekday (Monday=0)", xlabel="Hour")
    fig.tight_layout(); return fig


def observed_vs_predicted(report):
    data = report.predictions
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(data=data, x="actual", y="prediction", alpha=.3, s=16, color=sns.color_palette()[0], ax=ax)
    low = float(min(data.actual.min(), data.prediction.min()))
    high = float(max(data.actual.max(), data.prediction.max()))
    ax.plot([low, high], [low, high], linestyle="--", label="Perfect prediction")
    ax.set(title=f"{report.model_name}: {report.split}", xlabel=f"Observed ({report.contract.target_unit})", ylabel=f"Predicted ({report.contract.target_unit})")
    ax.legend(); fig.tight_layout(); return fig


def residual_plot(report):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=report.predictions, x="prediction", y="residual", alpha=.3, s=16, color=sns.color_palette()[0], ax=ax)
    ax.axhline(0, linestyle="--")
    ax.set(title=f"{report.model_name}: residual structure ({report.split})", ylabel=f"Observed minus predicted\n({report.contract.target_unit})")
    fig.tight_layout(); return fig


def error_distribution(report):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=report.predictions, x="residual", bins=40, color=sns.color_palette()[0], ax=ax)
    ax.axvline(0, linestyle="--")
    ax.set(title=f"{report.model_name}: directional errors", xlabel=f"Residual ({report.contract.target_unit}; positive = underprediction)")
    fig.tight_layout(); return fig


def errors_by_slice(report, column="hour"):
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=report.predictions, x=column, y="absolute_error", showfliers=False, color=sns.color_palette()[0], ax=ax)
    ax.set(title=f"{report.model_name}: errors by {column} (outlier points hidden, not removed)", ylabel=f"Absolute error ({report.contract.target_unit})")
    fig.tight_layout(); return fig


def chronological_predictions(report, *, hours=168):
    data = report.predictions.sort_values("target_time")
    if data.empty: raise ValueError("No predictions to display.")
    end = data.target_time.min() + pd.Timedelta(hours=hours)
    data = data.loc[data.target_time < end, ["target_time", "actual", "prediction"]]
    # Reindex missing hours; avoid plotting them as consecutive observations.
    data = data.set_index("target_time").asfreq("h").reset_index()
    data["segment"] = data[["actual", "prediction"]].isna().any(axis=1).cumsum()
    long = data.melt(["target_time", "segment"], var_name="series", value_name="value").dropna(subset=["value"])
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=long, x="target_time", y="value", hue="series", units="segment", estimator=None, errorbar=None, ax=ax)
    ax.set(title=f"{report.model_name}: first {hours} elapsed hours", xlabel="Target hour", ylabel=report.contract.target_unit)
    fig.autofmt_xdate(); fig.tight_layout(); return fig


def training_curves(history):
    table = pd.DataFrame(history).rename_axis("epoch").reset_index()
    table["epoch"] += 1
    long = table.melt("epoch", var_name="split", value_name="loss_value")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=long, x="epoch", y="loss_value", hue="split", estimator=None, errorbar=None, ax=ax)
    ax.set(title="Sequence learning curves", ylabel="MSE in training-standardized target space")
    fig.tight_layout(); return fig
