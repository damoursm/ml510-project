"""Save experiment settings, predictions, tables, figures, and models."""
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
import hashlib
import json
import re
import uuid
import numpy as np
import pandas as pd
from ml_lib.data.loading import sha256_file
from ml_lib.experiment.reproducibility import environment_snapshot


def _json_default(value):
    if is_dataclass(value): return asdict(value)
    if isinstance(value, (Path, pd.Timestamp)): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: str | Path, content) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, default=_json_default, allow_nan=False), encoding="utf-8")


class RunArtifacts:
    def __init__(self, root: Path, label: str, *, config: dict, dataset_path: Path):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
            raise ValueError("Run labels may contain only letters, numbers, underscore and hyphen.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.directory = Path(root) / "artifacts" / label / f"{stamp}-{uuid.uuid4().hex[:8]}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self._reports = {}
        excluded = {"__pycache__", ".ipynb_checkpoints", "corrections"}
        files = sorted(
            p for folder in ["src", "configs", "notebooks"]
            for p in (Path(root) / folder).rglob("*")
            if p.is_file() and p.suffix in {".py", ".toml", ".ipynb"}
            and not excluded.intersection(p.relative_to(root).parts)
            and not (folder == "notebooks" and any(
                part.endswith("_corrections") for part in p.relative_to(root).parts[:-1]
            ))
        )
        sources = {str(p.relative_to(root)): sha256_file(p) for p in files}
        self.dataset_fingerprint = sha256_file(dataset_path)
        write_json(self.directory / "manifest.json", {
            "created_at_utc": stamp, "label": label, "configuration": config,
            "dataset": {"filename": dataset_path.name, "sha256": self.dataset_fingerprint},
            "environment": environment_snapshot(), "source_sha256": sources,
            "source_tree_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        })

    def record(self, name: str, content) -> None:
        self._validate_name(name)
        write_json(self.directory / f"{name}.json", content)

    @staticmethod
    def _validate_name(name: str):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Artifact names must be safe path components.")

    def save_report(self, report, *, consolidated: bool = False):
        """Save a report; optionally maintain one metrics/predictions table per run.

        Consolidated reports replace the previous model/split entry when a cell
        is rerun. Training metrics are saved, but training rows are not exported.
        """
        if consolidated:
            self._reports[(report.model_name, report.split)] = report
            reports = list(self._reports.values())
            metrics = pd.DataFrame([{"model": r.model_name, "split": r.split, **r.metrics} for r in reports])
            self.save_table("metrics", metrics)
            rows = [r.predictions.assign(model=r.model_name, split=r.split)
                    for r in reports if r.split != "train"]
            if rows:
                self.save_table("predictions", pd.concat(rows, ignore_index=True))
            return
        name = f"{report.split}-{report.model_name}"
        self._validate_name(name)
        report.predictions.to_csv(self.directory / f"{name}-predictions.csv", index=False)
        self.record(name + "-metrics", {"contract": report.contract.to_dict(), "dataset_sha256": report.dataset_fingerprint,
                                      "split": report.split, "metrics": report.metrics})

    def save_table(self, name: str, table: pd.DataFrame, *, index: bool = False) -> Path:
        """Write a stable CSV filename; preserve a meaningful index explicitly."""
        self._validate_name(name)
        path = self.directory / "tables" / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=index)
        return path

    def save_text(self, name: str, text: str) -> Path:
        self._validate_name(name)
        path = self.directory / f"{name}.txt"
        path.write_text(text, encoding="utf-8")
        return path

    def save_figure(self, name: str, figure, *, close: bool = False) -> Path:
        self._validate_name(name)
        path = self.directory / "figures" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            figure.savefig(path, dpi=150, bbox_inches="tight")
        finally:
            if close:
                import matplotlib.pyplot as plt
                plt.close(figure)
        return path

    def write_overview(self, title: str, summary: str, *, inspect_first=()) -> Path:
        """Create a short Markdown entry point linking only files that exist."""
        lines = [f"# {title}", "", summary, ""]
        root = self.directory.resolve()
        prioritized = []
        for relative in inspect_first:
            path = self.directory / relative
            if Path(relative).is_absolute() or not path.resolve().is_relative_to(root):
                raise ValueError("Overview links must stay within the run directory.")
            if path.is_file() and path.name != "README.md" and path not in prioritized:
                prioritized.append(path)
        if prioritized:
            lines.extend(["## Inspect first", ""])
            for path in prioritized:
                relative = path.relative_to(self.directory).as_posix()
                lines.append(f"- [{path.name}]({quote(relative)})")
            lines.append("")
        for label, folder in [("Tables", "tables"), ("Figures", "figures"), ("Model", "model")]:
            files = sorted(p for p in (self.directory / folder).rglob("*")
                           if p.is_file() and p not in prioritized and p.resolve().is_relative_to(root))
            if files:
                lines.extend([f"## {label}", ""])
                for path in files:
                    relative = path.relative_to(self.directory).as_posix()
                    lines.append(f"- [{path.name}]({quote(relative)})")
                lines.append("")
        files = sorted(p for p in self.directory.iterdir()
                       if p.is_file() and p.name != "README.md" and p not in prioritized
                       and p.resolve().is_relative_to(root))
        if files:
            lines.extend(["## Run details", ""])
            lines.extend(f"- [{p.name}]({quote(p.name)})" for p in files)
        path = self.directory / "README.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
