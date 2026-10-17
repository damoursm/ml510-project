"""Set random seeds and record the execution environment."""
from importlib import metadata
import platform
import random
import sys
import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # TensorFlow is deliberately configured by its adapter only when requested.


def environment_snapshot() -> dict:
    names = ["ml-lib", "numpy", "scipy", "pandas", "scikit-learn", "seaborn", "matplotlib",
             "tensorflow", "tensorflow-cpu", "keras", "jupyterlab", "ipykernel"]
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "python_executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(),
            "packages": versions}
