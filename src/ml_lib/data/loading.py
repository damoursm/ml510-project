"""Read local data and record file fingerprints."""
from pathlib import Path
import hashlib
import pandas as pd


def read_local_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing local dataset: {path}. Download and extract it yourself; "
            "see the relevant lab guide under docs/ for dataset instructions. No network request has been made."
        )
    return pd.read_csv(path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
