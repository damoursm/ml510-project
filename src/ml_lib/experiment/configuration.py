"""Local TOML configuration via the Python standard library."""
from pathlib import Path
import os
import tomllib


def project_root(start: str | Path | None = None) -> Path:
    explicit = os.environ.get("ML510_PROJECT_ROOT")
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not (root / "pyproject.toml").is_file():
            raise FileNotFoundError("ML510_PROJECT_ROOT must point to the repository root.")
        return root
    current = Path(start or Path.cwd()).resolve()
    for parent in (current, *current.parents):
        if (parent / "pyproject.toml").is_file() and (parent / "configs").is_dir():
            return parent
    raise FileNotFoundError("Start Jupyter from the ml510-labs repository, or set ML510_PROJECT_ROOT.")


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("rb") as f:
        return tomllib.load(f)
