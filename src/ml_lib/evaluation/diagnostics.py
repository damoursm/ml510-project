"""Inspect how prediction errors vary over time."""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from ml_lib.data.validation import require_time_index


def lag_correlations(series: pd.Series, lags=(1, 2, 24, 168)) -> pd.DataFrame:
    """Exact timestamp alignment even when hours are absent.

    Report descriptive correlation only. IID significance tests would be
    inappropriate for many dependent time-series observations.
    """
    require_time_index(series)
    out = []
    for lag in lags:
        if lag < 1:
            raise ValueError("Lags must be positive.")
        older = series.reindex(series.index - pd.Timedelta(hours=lag)).to_numpy()
        now = series.to_numpy()
        valid = np.isfinite(now) & np.isfinite(older)
        r = None
        if valid.sum() >= 2 and np.var(now[valid]) > 0 and np.var(older[valid]) > 0:
            r = float(pearsonr(now[valid], older[valid]).statistic)
        out.append({"lag_hours": lag, "paired_observations": int(valid.sum()), "correlation": r})
    return pd.DataFrame(out)
