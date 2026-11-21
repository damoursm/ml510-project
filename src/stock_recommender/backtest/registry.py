"""The models a backtest can use, by name, and the hyperparameters it tunes for each.

The account (backtest/account.py) is the combination model: it buys every stock that each model registered here
recommends, for the same order value. To add a model, add its factory to MODELS and its grid to GRIDS; the backtest
then trades it in the account beside the others.

A factory takes a model's own hyperparameters as keyword arguments and returns new, untrained models, e.g.
"gbm": lambda depth=3: replace(placeholder_models(), price=GBMPriceModel(depth), version=f"gbm-{depth}").
A grid may also name a Models field (horizon_days, bars_lookback, news_lookback_days); the backtest sets those
itself, so the factory never sees them. A grid never names picks: every recommended stock is bought for the order
value, so how many stocks a model recommends is the model's own, not something the backtest sizes or tunes.

The first value of each hyperparameter is its default: the one used when no earlier year can choose.
Never register load_models(): once it loads models trained on 2021-2025, a backtest would test on what they saw.
"""
from dataclasses import fields, replace
from itertools import product
from ..forest import ForestPriceModel
from ..models import Models, placeholder_models

MODEL_FIELDS = frozenset(field.name for field in fields(Models))


def most_liquid_models() -> Models:
    """Recommend the most liquid stocks: the price model expects no move for any stock, and equal scores rank the
    larger dollar volume first (recommend.score_day)."""
    return replace(placeholder_models(), version="most_liquid")


def random_forest_models(min_samples_leaf: int = 1_000, max_features: float = 0.5) -> Models:
    """Recommend the stocks a random forest on the ten daily-bar features expects to beat the others most over the
    horizon (forest.py); no news. Its 20 best picks, held 5 sessions, from 60 sessions of bars."""
    price = ForestPriceModel(min_samples_leaf=min_samples_leaf, max_features=max_features)
    return replace(placeholder_models(), price=price,
                   version=f"random_forest(min_samples_leaf={min_samples_leaf},max_features={max_features})")


MODELS = {"most_liquid": most_liquid_models, "random_forest": random_forest_models}

GRIDS: dict[str, dict] = {  # most_liquid has no hyperparameters: it recommends its 20 most liquid stocks.
    "random_forest": {"min_samples_leaf": (1_000, 250), "max_features": (0.5, 1.0)},
}
UNTUNED = frozenset({"picks"})  # Fields no grid may name.


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
