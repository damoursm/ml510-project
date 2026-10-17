"""Time construction is explicit; missing hourly buckets never get compressed."""
from collections.abc import Sequence
import numpy as np
import pandas as pd
from ml_lib.data.validation import require_columns, require_time_index
from ml_lib.problems.forecasting import SequenceDataset


def lag_features(frame: pd.DataFrame, column: str, lags: Sequence[int], rolling: Sequence[int] = ()) -> pd.DataFrame:
    """Index is TARGET hour s. Every output uses values from s-1 or earlier."""
    require_time_index(frame, regular=True)
    require_columns(frame, [column])
    if any(not isinstance(n, int) or n < 1 for n in [*lags, *rolling]):
        raise ValueError("Lags and rolling windows must be positive integer hours.")
    result = pd.DataFrame(index=frame.index)
    for lag in lags:
        result[f"{column}_lag_{lag}"] = frame[column].shift(lag)
    past = frame[column].shift(1)
    for window in rolling:
        result[f"{column}_mean_{window}"] = past.rolling(window, min_periods=window).mean()
    return result


def make_sequences(
    frame: pd.DataFrame, *, history_columns: Sequence[str], target: str,
    future_columns: Sequence[str] = (), lookback: int = 24,
) -> SequenceDataset:
    """Pair each complete history window with the next hour's inputs and target.

    The caller chooses which next-hour fields are known at prediction time.
    Examples with missing history, known inputs or labels are excluded.
    """
    require_time_index(frame, regular=True)
    require_columns(frame, [*history_columns, *future_columns, target])
    if not history_columns or type(lookback) is not int or lookback < 1:
        raise ValueError("Provide history features and a positive integer lookback.")
    if target in future_columns:
        raise ValueError("The future target cannot also be a known input.")
    hist = frame[list(history_columns)].to_numpy(dtype=np.float32)
    future = frame[list(future_columns)].to_numpy(dtype=np.float32)
    values = frame[target].to_numpy(dtype=np.float32)
    hh, ff, yy, oo, tt = [], [], [], [], []
    candidates = max(0, len(frame) - lookback)
    for end in range(lookback - 1, len(frame) - 1):
        x = hist[end - lookback + 1:end + 1]
        z, y = future[end + 1], values[end + 1]
        if not (np.isfinite(x).all() and np.isfinite(z).all() and np.isfinite(y)):
            continue
        hh.append(x); ff.append(z); yy.append(y)
        oo.append(frame.index[end]); tt.append(frame.index[end + 1])
    n = len(yy)
    return SequenceDataset(
        np.asarray(hh, dtype=np.float32).reshape(n, lookback, len(history_columns)),
        np.asarray(ff, dtype=np.float32).reshape(n, len(future_columns)),
        np.asarray(yy, dtype=np.float32), pd.DatetimeIndex(oo, name="origin"),
        np.asarray(tt, dtype="datetime64[ns]"), tuple(history_columns), tuple(future_columns),
        {"candidate_windows": candidates, "retained_windows": n,
         "dropped_incomplete_windows": candidates - n,
         "retained_fraction": n / candidates if candidates else 0.0,
         "gap_policy": "reject incomplete examples; never fill target"},
    )
