"""Fit scikit-learn regressors with training-fitted preprocessing."""
from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from ml_lib.features.preprocessing import PreprocessingConfig, make_preprocessor
from ml_lib.models.interfaces import ModelSpec
from ml_lib.problems.regression import FeatureSchema
from ml_lib.data.validation import finite_array


class TabularRegressor:
    def __init__(self, spec: ModelSpec, schema: FeatureSchema, *, preprocessing: PreprocessingConfig, seed: int = 42):
        self.spec = ModelSpec(spec.family, deepcopy(spec.parameters))
        self.schema, self.seed = schema, seed
        self.preprocessing = preprocessing
        p = dict(self.spec.parameters)
        factories = {
            "mean": lambda: DummyRegressor(strategy="mean", **p),
            "median": lambda: DummyRegressor(strategy="median", **p),
            "linear": lambda: LinearRegression(**p),
            "ridge": lambda: Ridge(**p),
            "tree": lambda: DecisionTreeRegressor(**{"random_state": seed, **p}),
            "forest": lambda: RandomForestRegressor(**{"random_state": seed, "n_jobs": 1, **p}),
            # Avoid random internal early-stopping validation in a temporal lab.
            "boosting": lambda: HistGradientBoostingRegressor(**{"random_state": seed, "early_stopping": False, **p}),
        }
        if spec.family not in factories:
            raise ValueError(f"Unknown model family {spec.family!r}; choose {list(factories)}.")
        self.pipeline = Pipeline([("prepare", make_preprocessor(schema, preprocessing)), ("model", factories[spec.family]())])
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y) -> "TabularRegressor":
        self.schema.validate(X)
        if self.preprocessing.missing_values == "error" and X[self.schema.columns].isna().any().any():
            raise ValueError("Missing selected inputs; investigate or explicitly choose imputation.")
        values = finite_array(y, name="training labels")
        if values.ndim != 1 or len(values) != len(X):
            raise ValueError("Expected one scalar label per row.")
        if isinstance(y, pd.Series) and not X.index.equals(y.index):
            raise ValueError("Training inputs and labels are not aligned.")
        self.pipeline.fit(X, values)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Fit on training data before prediction.")
        self.schema.validate(X)
        if self.preprocessing.missing_values == "error" and X[self.schema.columns].isna().any().any():
            raise ValueError("Missing selected inputs; investigate or explicitly choose imputation.")
        # Raw predictions are retained. Negative predictions are diagnosed, not silently clipped.
        return np.asarray(self.pipeline.predict(X), dtype=float)

    def describe(self) -> dict:
        return {"spec": asdict(self.spec), "schema": asdict(self.schema), "seed": self.seed, "backend": "scikit-learn",
                "preprocessing": asdict(self.preprocessing),
                "policies": {"numeric_imputation": "median" if self.preprocessing.missing_values == "impute" else None,
                             "categorical_imputation": "most_frequent" if self.preprocessing.missing_values == "impute" else None,
                             "categorical_encoding": "one_hot", "unknown_categories": "ignore",
                             "training_objective": "absolute_error" if self.spec.family == "median" else "squared_error",
                             "early_stopping": False if self.spec.family == "boosting" else None,
                             "n_jobs": 1 if self.spec.family == "forest" else None}}

    def coefficients(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        model = self.pipeline.named_steps["model"]
        if not hasattr(model, "coef_"):
            raise ValueError("This family has no linear coefficient vector.")
        return pd.DataFrame({
            "feature": self.pipeline.named_steps["prepare"].get_feature_names_out(),
            "coefficient": np.asarray(model.coef_).ravel(),
        }).assign(abs_coefficient=lambda x: x.coefficient.abs()).sort_values("abs_coefficient", ascending=False)

    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_regressor(path: str | Path, *, trusted: bool = False) -> TabularRegressor:
    """Only load your own trusted artifact in the same software environment."""
    if not trusted:
        raise ValueError("Pickle can execute code. Loading requires trusted=True for your own artifacts.")
    with Path(path).open("rb") as f:
        model = pickle.load(f)
    if not isinstance(model, TabularRegressor):
        raise TypeError("Artifact is not a TabularRegressor.")
    return model


def build_regressor(spec: ModelSpec, schema: FeatureSchema, *, preprocessing: PreprocessingConfig, seed: int = 42) -> TabularRegressor:
    return TabularRegressor(spec, schema, preprocessing=preprocessing, seed=seed)
