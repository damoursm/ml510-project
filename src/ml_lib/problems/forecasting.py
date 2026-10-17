"""Sequence-shaped numerical prediction data. All timestamps remain explicit."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class SequenceDataset:
    history: np.ndarray                 # [sample, lookback, historical feature]
    future: np.ndarray                  # [sample, known-next-hour feature]
    y: np.ndarray                       # [sample]
    origins: pd.DatetimeIndex           # last fully observed bucket
    target_times: np.ndarray            # datetime64[ns], [sample]
    history_names: tuple[str, ...]
    future_names: tuple[str, ...]
    coverage: dict

    def __post_init__(self):
        if self.history.ndim != 3 or self.future.ndim != 2 or self.y.ndim != 1:
            raise ValueError("Expected 3D history, 2D next-hour inputs and 1D labels.")
        n = len(self.y)
        if len(self.origins) != n or self.history.shape[0] != n or self.future.shape[0] != n:
            raise ValueError("Sequence sample counts are inconsistent.")
        if self.target_times.shape != (n,):
            raise ValueError("target_times must match the one-dimensional labels.")
        if self.history.shape[2] != len(self.history_names) or self.future.shape[1] != len(self.future_names):
            raise ValueError("Feature names and array dimensions disagree.")
        if self.history.shape[1] < 1:
            raise ValueError("Lookback must be positive.")
        if n:
            for values in (self.history, self.future, self.y):
                if not np.isfinite(values).all():
                    raise ValueError("Sequences and labels must be finite.")
            origin = self.origins.to_numpy(dtype="datetime64[ns]")
            if self.origins.hasnans or not np.array_equal(self.target_times, origin + np.timedelta64(1, "h")):
                raise ValueError("Each target must be the next hour after its origin.")

    def subset(self, mask) -> "SequenceDataset":
        mask = np.asarray(mask)
        return SequenceDataset(
            self.history[mask], self.future[mask], self.y[mask], self.origins[mask],
            self.target_times[mask], self.history_names, self.future_names,
            {**self.coverage, "subset_samples": int(len(self.y[mask]))},
        )

    @property
    def lookback(self) -> int:
        return self.history.shape[1]
