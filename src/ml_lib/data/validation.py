"""Check data shapes, values, and timestamps before modeling."""
from collections.abc import Iterable
import numpy as np
import pandas as pd


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame.columns.has_duplicates:
        raise ValueError("Duplicate column names are not allowed.")


def require_time_index(frame: pd.DataFrame | pd.Series, *, regular: bool = False) -> None:
    idx = frame.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.hasnans:
        raise ValueError("Expected a DatetimeIndex without NaT.")
    if len(idx) == 0:
        raise ValueError("Time-indexed input is empty.")
    if not idx.is_unique or not idx.is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and sorted in increasing order.")
    if not (idx == idx.floor("h")).all():
        raise ValueError("Expected exact hourly bucket labels.")
    if regular and len(idx) > 1:
        deltas = idx[1:] - idx[:-1]
        if not (deltas == pd.Timedelta(hours=1)).all():
            raise ValueError("Expected a complete hourly index. Reindex before positional lags/windows.")


def hourly_grid(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose absent hours as NaN; never infer that missing demand equals zero."""
    require_time_index(frame)
    return frame.reindex(pd.date_range(frame.index.min(), frame.index.max(), freq="h", name=frame.index.name))


def finite_array(values, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be non-empty and entirely finite.")
    return result
