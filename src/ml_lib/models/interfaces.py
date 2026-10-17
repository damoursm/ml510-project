"""Model settings and the shared fit, predict, and save interfaces."""
from dataclasses import dataclass, field
from typing import Any, Protocol
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelSpec:
    family: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        controls = {
            "mean": (), "median": (), "linear": (), "ridge": ("alpha",),
            "tree": ("max_depth", "min_samples_leaf"),
            "forest": ("n_estimators", "max_depth", "min_samples_leaf", "max_features"),
            "boosting": ("learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf"),
        }
        if self.family not in controls:
            raise ValueError(f"Unknown model family {self.family!r}; choose {list(controls)}.")
        expected = set(controls[self.family])
        supplied = set(self.parameters)
        if supplied != expected:
            raise ValueError(f"{self.family} expects controls {sorted(expected)}; "
                             f"missing {sorted(expected - supplied)}, unsupported {sorted(supplied - expected)}.")


class Regressor(Protocol):
    def fit(self, X: pd.DataFrame, y) -> "Regressor": ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def describe(self) -> dict: ...
