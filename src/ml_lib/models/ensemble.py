"""Average fitted sequence models in original target units."""
import json
from pathlib import Path
import numpy as np
from ml_lib.models.deep_learning import SequenceForecaster


class SequenceEnsemble:
    """Keep every fitted member with equal weight; fit them explicitly in the notebook."""

    def __init__(self, members):
        self.members = tuple(members)
        if len(self.members) < 2:
            raise ValueError("Provide at least two fitted sequence models.")
        if any(member.model is None or member.signature is None for member in self.members):
            raise ValueError("Every ensemble member must be fitted or loaded.")
        if any(member.signature != self.members[0].signature for member in self.members[1:]):
            raise ValueError("Ensemble members must use the same input shape and feature order.")

    def predict(self, data):
        return np.mean([member.predict(data) for member in self.members], axis=0)

    def predict_inputs(self, history, future, *, history_names, future_names):
        predictions = [member.predict_inputs(history, future,
                       history_names=history_names, future_names=future_names)
                       for member in self.members]
        return np.mean(predictions, axis=0)

    def describe(self):
        members = [member.describe() for member in self.members]
        return {"aggregation": "equal mean in original target units", "members": members,
                "parameters": sum(member["parameters"] for member in members),
                "signature": self.members[0].signature}

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for number, member in enumerate(self.members, start=1):
            member.save(directory / f"member-{number:02d}")
        (directory / "ensemble.json").write_text(json.dumps({
            "aggregation": "equal_mean", "member_count": len(self.members),
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        metadata = json.loads((directory / "ensemble.json").read_text(encoding="utf-8"))
        count = metadata["member_count"]
        if metadata.get("aggregation") != "equal_mean" or type(count) is not int or count < 2:
            raise ValueError("Invalid equal-mean ensemble metadata.")
        return cls([SequenceForecaster.load(directory / f"member-{number:02d}")
                    for number in range(1, count + 1)])
