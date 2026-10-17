"""Reuse scikit-learn metrics; retain original units and error direction."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ml_lib.data.validation import finite_array


def regression_metrics(y_true, y_pred) -> dict[str, float | int | None]:
    actual = finite_array(y_true, name="y_true")
    prediction = finite_array(y_pred, name="y_pred")
    if actual.shape != prediction.shape:
        raise ValueError(f"Shape mismatch: actual {actual.shape}, prediction {prediction.shape}; no broadcasting allowed.")
    if actual.ndim != 1:
        raise ValueError("Provide one-dimensional targets and predictions.")
    residual = actual - prediction
    abs_error = np.abs(residual)
    total = np.abs(actual).sum()
    # Constant-target R2 is undefined here, rather than silently coerced to 0 or 1.
    r2 = float(r2_score(actual, prediction)) if len(actual) > 1 and np.var(actual) > 0 else None
    return {
        "n": int(len(actual)), "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))), "r2": r2,
        "mean_residual": float(residual.mean()), "p90_absolute_error": float(np.quantile(abs_error, .9)),
        "max_absolute_error": float(abs_error.max()),
        "underprediction_rate": float((residual > 0).mean()),
        "negative_prediction_rate": float((prediction < 0).mean()),
        "wape": float(abs_error.sum() / total) if total > 0 else None,
    }
