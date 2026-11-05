"""The models a backtest can use, by name, and the hyperparameters it tunes for each.

A factory takes a model's own hyperparameters as keyword arguments and returns new, untrained models, e.g.
"gbm": lambda depth=3: replace(placeholder_models(), price=GBMPriceModel(depth), version=f"gbm-{depth}").
A grid may also name a Models field (picks, horizon_days, bars_lookback, news_lookback_days); the backtest sets
those itself, so the factory never sees them. picks only cuts each decision's ranking, so candidates that differ
only in picks share one fit.

The first value of each hyperparameter is its default: the one used when no earlier year can choose.
Never register load_models(): once it loads models trained on 2021-2025, a backtest would test on what they saw.
"""
from dataclasses import fields, replace
from itertools import product
from ..models import Models, RandomPriceModel, placeholder_models

MODEL_FIELDS = frozenset(field.name for field in fields(Models))


def random_models(seed: int = 0) -> Models:
    """Placeholder models whose price model scores at random: a no-skill baseline."""
    return replace(placeholder_models(), price=RandomPriceModel(seed), version=f"random-{seed}")


MODELS = {"placeholder": placeholder_models, "random": random_models}

# Picking the best of three random seeds must still look like no skill in the next year: a selection-bias check.
GRIDS = {"placeholder": {"picks": (20, 10)}, "random": {"seed": (0, 1, 2)}}


def configurations(grid: dict) -> list[dict]:
    """Return every combination of a grid's values, keys in sorted order, starting with the defaults."""
    keys = sorted(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[key] for key in keys))]


def build(factory, params: dict):
    """Return fresh models for one configuration: the factory gets its own hyperparameters, the backtest sets
    the Models fields."""
    own = {key: value for key, value in params.items() if key not in MODEL_FIELDS}
    model = factory(**own)
    shared = {key: value for key, value in params.items() if key in MODEL_FIELDS}
    return replace(model, **shared) if shared else model
