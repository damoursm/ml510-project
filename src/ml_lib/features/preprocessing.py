"""Explicit preprocessing choices, fitted inside the training pipeline."""
from dataclasses import dataclass
from typing import Literal
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from ml_lib.problems.regression import FeatureSchema


@dataclass(frozen=True)
class PreprocessingConfig:
    scale_numeric: bool
    missing_values: Literal["error", "impute"]

    def __post_init__(self):
        if type(self.scale_numeric) is not bool or self.missing_values not in {"error", "impute"}:
            raise ValueError("Choose scale_numeric=True/False and missing_values='error'/'impute'.")


def make_preprocessor(schema: FeatureSchema, config: PreprocessingConfig) -> ColumnTransformer:
    transformations = []
    if schema.numeric:
        steps = []
        if config.missing_values == "impute":
            steps.append(("impute", SimpleImputer(strategy="median")))
        if config.scale_numeric:
            steps.append(("scale", StandardScaler()))
        transformations.append(("numeric", Pipeline(steps) if steps else "passthrough", list(schema.numeric)))
    if schema.categorical:
        steps = []
        if config.missing_values == "impute":
            steps.append(("impute", SimpleImputer(strategy="most_frequent")))
        steps.append(("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
        transformations.append(("category", Pipeline(steps), list(schema.categorical)))
    return ColumnTransformer(transformations, remainder="drop", verbose_feature_names_out=True)
