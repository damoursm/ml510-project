"""Validation-only model selection with an explicit metric direction."""
import math


def select_model(validation_table, metric: str = "mae") -> dict:
    """Select from a metric-column table; ties keep the first candidate.

    Accepts a pandas DataFrame (or a mapping of metric to model-score mappings).
    Every candidate must have a finite score for the selected metric.
    """
    directions = {"mae": "minimize", "rmse": "minimize", "wape": "minimize", "r2": "maximize"}
    if metric not in directions:
        raise ValueError(f"Unsupported selection metric {metric!r}. Choose mae, rmse, wape, or r2.")
    if metric not in validation_table:
        raise ValueError(f"Validation scores do not include {metric!r}.")
    candidates = list(validation_table[metric].items())
    if not candidates:
        raise ValueError("No validation candidates to select.")
    scores = []
    for name, value in candidates:
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Undefined {metric} validation score for {name!r}.") from None
        if not math.isfinite(value):
            raise ValueError(f"Undefined {metric} validation score for {name!r}.")
        scores.append((name, value))
    choose = min if directions[metric] == "minimize" else max
    name, score = choose(scores, key=lambda candidate: candidate[1])
    return {"selected_model": name, "metric": metric, "direction": directions[metric],
            "validation_score": score, "rule": f"{directions[metric]} validation {metric}"}
