"""Deterministic, auditable feature transformations with no learned state."""
import numpy as np
import pandas as pd


def add_cyclic_features(frame: pd.DataFrame, periods: dict[str, int]) -> pd.DataFrame:
    out = frame.copy()
    for name, period in periods.items():
        if period <= 0:
            raise ValueError("Cyclic periods must be positive.")
        phase = 2 * np.pi * out[name].to_numpy(dtype=float) / period
        out[f"{name}_sin"] = np.sin(phase)
        out[f"{name}_cos"] = np.cos(phase)
    return out
