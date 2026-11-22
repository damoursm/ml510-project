"""The models a backtest can use, by name, and the hyperparameters it tunes for each.

The account (backtest/account.py) is the combination model: it buys every stock recommended by the models chosen
for the year, for the same order value. On the test years before it, a model qualifies only when its mean yearly
return beats SPY held, and the set of qualifying models whose account did best is traded (run.choose_combination).
To add a model, add its factory to MODELS and its grid to GRIDS; the backtest then tries it in every set.

A factory takes a model's own hyperparameters as keyword arguments and returns new, untrained models, e.g.
"gbm": lambda depth=3: replace(placeholder_models(), price=GBMPriceModel(depth), version=f"gbm-{depth}").
A grid may also name a Models field (horizon_days, bars_lookback, news_lookback_days, and the exit settings
hold_days, stop_loss, take_profit and volatility_exit); the backtest sets those itself, so the factory never sees
them. A grid never names picks: every recommended stock is bought for the order value, so how many stocks a model
recommends is the model's own, not something the backtest sizes or tunes.

The first value of each hyperparameter is its default: the one used when no earlier year can choose.
Never register load_models(): once it loads models trained on 2021-2025, a backtest would test on what they saw.
"""
from dataclasses import fields, replace
from itertools import product
from ..forest import ForestPriceModel
from ..models import Models, placeholder_models
from ..news_boost import NewsBoostModel
from ..news_features import EXTENDED_LOOKBACK_DAYS
from ..news_forest import NewsForestModel, news_blend
from ..price_news_forest import PriceNewsForestModel

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


def news_forest_models(min_samples_leaf: int = 1_000, max_features: float = 0.5) -> Models:
    """Recommend the stocks whose last 7 days of news a random forest on the eight news features rates best
    (news_forest.py); no bars feature: its price model expects no move for any stock. Its 20 best picks among the
    stocks with news, held 5 sessions."""
    news = NewsForestModel(min_samples_leaf=min_samples_leaf, max_features=max_features)
    return replace(placeholder_models(), news=news, blend=news_blend,
                   version=f"news_forest(min_samples_leaf={min_samples_leaf},max_features={max_features})")


def price_news_forest_models(min_samples_leaf: int = 1_000, max_features: float = 0.5) -> Models:
    """Recommend the stocks a random forest on the ten daily-bar features and the eight news features together
    expects to rank best over the horizon (price_news_forest.py). It is a joint model, which gives the score
    itself: the price and news parts stay the placeholders. Its 20 best picks among every tradable stock, news or
    not, held 5 sessions, from 60 sessions of bars and 7 days of news."""
    joint = PriceNewsForestModel(min_samples_leaf=min_samples_leaf, max_features=max_features)
    return replace(placeholder_models(), joint=joint,
                   version=f"price_news_forest(min_samples_leaf={min_samples_leaf},max_features={max_features})")


def news_boost_models(num_leaves: int = 7, min_child_samples: int = 2_000) -> Models:
    """Recommend the stocks whose last 90 days of news a LightGBM model on the 55 extended news features rates best
    over the next 20 sessions (news_boost.py). A joint model that rates only the stocks with news in the 7 days
    (preferred shares excluded), so its 20 picks are always stocks it rated."""
    joint = NewsBoostModel(num_leaves=num_leaves, min_child_samples=min_child_samples)
    return replace(placeholder_models(), joint=joint, horizon_days=20, news_lookback_days=EXTENDED_LOOKBACK_DAYS,
                   version=f"news_boost(num_leaves={num_leaves},min_child_samples={min_child_samples})")


MODELS = {"most_liquid": most_liquid_models, "random_forest": random_forest_models,
          "news_forest": news_forest_models, "price_news_forest": price_news_forest_models,
          "news_boost": news_boost_models}

# Every model's exit strategy (exits.py), tuned with its own hyperparameters. The defaults hold 5 trading days with
# no early rule, as before exit settings existed. Settings that differ only here share one fit per year, so they
# cost trading time, not training.
EXITS = {
    "hold_days": (5, 10, 20),  # Trading days to hold at most; the forests still learn 5-day returns.
    "stop_loss": (None, 0.08),  # Sell after a close 8% below the entry price: the MAE limit.
    "take_profit": (None, 0.15),  # Sell after a close 15% above it: the MFE target.
    "volatility_exit": (None, 3.0),  # After a move of 3 daily volatilities, sell once a move is back within one.
}
FOREST = {"min_samples_leaf": (1_000, 250), "max_features": (0.5, 1.0)}
GRIDS: dict[str, dict] = {  # most_liquid has no hyperparameters of its own, only its exit strategy.
    "most_liquid": dict(EXITS),
    "random_forest": {**FOREST, **EXITS},
    "news_forest": {**FOREST, **EXITS},
    "price_news_forest": {**FOREST, **EXITS},
    # It learns 20-session returns, so it holds 20 sessions by default.
    "news_boost": {"num_leaves": (7, 15), "min_child_samples": (2_000, 500), **EXITS, "hold_days": (20, 10, 5)},
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
