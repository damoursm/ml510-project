"""Describe the target, input timing, and intended use of a prediction."""
from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class PredictionContract:
    name: str
    task: str
    unit: str
    target: str
    target_unit: str
    prediction_moment: str
    information_set: str
    action: str

    def __post_init__(self):
        if self.task not in {"regression", "forecasting"}:
            raise ValueError("This release implements regression and numerical forecasting only.")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
