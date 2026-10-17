"""Collect prediction errors and compare methods on the same observations."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from ml_lib.problems.contracts import PredictionContract
from ml_lib.evaluation.metrics import regression_metrics


@dataclass
class EvaluationReport:
    model_name: str
    contract: PredictionContract
    split: str
    dataset_fingerprint: str
    predictions: pd.DataFrame

    @property
    def metrics(self) -> dict:
        return regression_metrics(self.predictions.actual.to_numpy(), self.predictions.prediction.to_numpy())

    def by(self, column: str) -> pd.DataFrame:
        """Summarize errors within a group, retaining its sample count."""
        if column not in self.predictions:
            raise KeyError(column)
        rows = []
        for key, group in self.predictions.groupby(column, dropna=False, observed=True):
            rows.append({column: key, **regression_metrics(group.actual.to_numpy(), group.prediction.to_numpy())})
        return pd.DataFrame(rows)


def evaluate_predictions(
    y_true, y_pred, *, target_times, origins=None, contract: PredictionContract,
    model_name: str, split: str, dataset_fingerprint: str,
    metadata: pd.DataFrame | None = None,
) -> EvaluationReport:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation or test.")
    a, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    regression_metrics(a, p)  # Checks finite values, nonempty arrays and exact shapes.
    times = pd.DatetimeIndex(target_times)
    origin = pd.DatetimeIndex(origins if origins is not None else times)
    if len(times) != len(a) or len(origin) != len(a) or origin.hasnans or times.hasnans:
        raise ValueError("Invalid prediction timestamps.")
    if contract.task == "forecasting" and not np.array_equal(
        times.to_numpy(), origin.to_numpy() + np.timedelta64(1, "h")
    ):
        raise ValueError("Forecast targets must be the next hour after their origins.")
    pred = pd.DataFrame({"origin": origin, "target_time": times, "actual": a, "prediction": p})
    if pred.duplicated(["origin", "target_time"]).any():
        raise ValueError("Duplicate forecast opportunities would distort evaluation.")
    pred["residual"] = pred.actual - pred.prediction
    pred["absolute_error"] = pred.residual.abs()
    pred["hour"] = pred.target_time.dt.hour
    pred["day_of_week"] = pred.target_time.dt.dayofweek
    if metadata is not None:
        if not metadata.index.is_unique:
            raise ValueError("Slice metadata must have a unique target-time index.")
        extras = metadata.reindex(pd.DatetimeIndex(pred.target_time))
        for name in extras.columns:
            if name not in pred.columns:
                pred[name] = extras[name].to_numpy()
    return EvaluationReport(model_name, contract, split, dataset_fingerprint, pred)


def comparison_table(reports: list[EvaluationReport]) -> pd.DataFrame:
    """Reject incompatible contracts, splits, population ordering or labels."""
    if not reports:
        raise ValueError("No reports to compare.")
    ref = reports[0]
    keys = ["origin", "target_time", "actual"]
    for r in reports[1:]:
        if (r.contract.fingerprint, r.split, r.dataset_fingerprint) != (ref.contract.fingerprint, ref.split, ref.dataset_fingerprint):
            raise ValueError("Incompatible prediction contracts, split roles or dataset snapshots; do not rank these scores together.")
        if not r.predictions[keys].equals(ref.predictions[keys]):
            raise ValueError("Predictions must cover the exact same ordered opportunities and outcomes.")
    return pd.DataFrame([{"model": r.model_name, **r.metrics} for r in reports]).set_index("model")


def reference_comparisons(
    reports: list[EvaluationReport], reference_names: dict[str, str], metric: str,
) -> pd.DataFrame:
    """Compare each split with named references already chosen by the caller.

    ``reference_names`` maps roles (such as "constant") to model names. Every
    reference must be present in each split. Positive improvement favors the
    candidate; relative error reduction is reported as a fraction.
    This function reports evidence; it neither fits nor selects a model.
    """
    if metric not in {"mae", "rmse", "wape", "r2"}:
        raise ValueError("Unsupported comparison metric. Choose mae, rmse, wape, or r2.")
    if not reports or not reference_names:
        raise ValueError("Reports and named references must be nonempty.")
    rows = []
    for partition in dict.fromkeys(report.split for report in reports):
        candidates = [report for report in reports if report.split == partition]
        table = comparison_table(candidates)
        if not table.index.is_unique:
            raise ValueError("Model names must be unique within each split.")
        scores = table[metric].to_numpy(dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError(f"Undefined {metric} score in {partition} comparisons.")
        for role, name in reference_names.items():
            if name not in table.index:
                raise ValueError(f"Missing reference {name!r} in {partition} reports.")
            reference_score = float(table.loc[name, metric])
            for model_name, values in table.iterrows():
                candidate_score = float(values[metric])
                improvement = reference_score - candidate_score
                if metric == "r2":
                    improvement = -improvement
                relative = None
                if metric != "r2" and reference_score > 0:
                    relative = improvement / reference_score
                rows.append({
                    "split": partition, "reference_role": role, "reference": name,
                    "model": model_name, "n": int(values["n"]), "metric": metric,
                    "reference_score": reference_score, "candidate_score": candidate_score,
                    "improvement": improvement, "relative_error_reduction": relative,
                })
    return pd.DataFrame(rows)


def asymmetric_cost(report: EvaluationReport, *, under_cost: float, over_cost: float) -> float:
    """Compute average error cost using caller-supplied hypothetical weights."""
    if min(under_cost, over_cost) < 0:
        raise ValueError("Costs must be nonnegative.")
    e = report.predictions.residual.to_numpy()
    return float((under_cost * np.maximum(e, 0) + over_cost * np.maximum(-e, 0)).mean())
