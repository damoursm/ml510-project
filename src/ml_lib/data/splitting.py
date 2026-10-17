"""Time-aware split boundaries are selected before model fitting."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_end: str
    validation_end: str

    def __post_init__(self):
        try:
            train_end = pd.Timestamp(self.train_end)
            validation_end = pd.Timestamp(self.validation_end)
        except (TypeError, ValueError) as exc:
            raise ValueError("Split boundaries must be valid timestamps.") from exc
        if pd.isna(train_end) or pd.isna(validation_end):
            raise ValueError("Split boundaries must be valid timestamps, not NaT.")
        if train_end >= validation_end:
            raise ValueError("train_end must be earlier than validation_end.")

    def masks(self, target_times, origins=None) -> dict[str, np.ndarray]:
        """Split by target timestamp; earlier history may cross a split boundary."""
        targets = pd.DatetimeIndex(target_times)
        if targets.hasnans:
            raise ValueError("Invalid target timestamps.")
        if origins is not None:
            origin = pd.DatetimeIndex(origins)
            if len(origin) != len(targets) or origin.hasnans or (origin >= targets).any():
                raise ValueError("Forecast origins must precede their targets.")
        a, b = pd.Timestamp(self.train_end), pd.Timestamp(self.validation_end)
        return {
            "train": np.asarray(targets < a),
            "validation": np.asarray((targets >= a) & (targets < b)),
            "test": np.asarray(targets >= b),
        }



def split_counts(masks: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame([{"split": name, "samples": int(mask.sum())} for name, mask in masks.items()])


def require_nonempty_splits(masks: dict[str, np.ndarray], minimum: int = 1) -> None:
    too_small = {name: int(mask.sum()) for name, mask in masks.items() if mask.sum() < minimum}
    if too_small:
        raise ValueError(f"Not enough usable samples: {too_small}. Inspect coverage, dates and lookback; do not fill targets silently.")
