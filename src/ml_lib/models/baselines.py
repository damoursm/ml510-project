"""Forecast references use only outcomes that are observable at the origin."""
import numpy as np
import pandas as pd
from ml_lib.data.validation import require_time_index


def naive_forecast(series: pd.Series, origins, *, period: int | None = None) -> np.ndarray:
    """Predict the next hour from the last count or a timestamped seasonal count.

    Missing source observations produce NaN for the common-coverage check.
    """
    require_time_index(series)
    origins = pd.DatetimeIndex(origins)
    if period is not None and (type(period) is not int or period < 1):
        raise ValueError("Seasonal period must be a positive integer number of hours.")
    source = origins if period is None else origins + pd.Timedelta(hours=1 - period)
    return series.reindex(source).to_numpy(dtype=float)


def matched_baseline_splits(series, origins, masks, *, periods):
    """Restrict already-valid examples to common reference coverage before fitting.

    Returns full-length baseline arrays, restricted split masks, and per-split
    counts. Callers use the same restricted masks for model inputs and labels.
    Availability uses historical observations only, never prediction quality.
    """
    from ml_lib.data.splitting import require_nonempty_splits

    if not periods:
        raise ValueError("Provide at least one forecast reference.")
    predictions = {name: naive_forecast(series, origins, period=period)
                   for name, period in periods.items()}
    available = np.logical_and.reduce([np.isfinite(values)
                                       for values in predictions.values()])
    restricted, counts = {}, []
    for name, mask in masks.items():
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != available.shape:
            raise ValueError("Split masks must be boolean and match the forecast origins.")
        restricted[name] = mask & available
        before, after = int(mask.sum()), int(restricted[name].sum())
        counts.append({"split": name, "before_baselines": before,
                       "retained": after, "dropped_missing_baselines": before - after})
    require_nonempty_splits(restricted)
    return predictions, restricted, pd.DataFrame(counts)
