"""The models a backtest can use, by name, and the hyperparameters it tunes for each.

The account (backtest/account.py) is the combination model: from one capital it buys the stocks recommended by the
models chosen for the year, a slice each, while its cash lasts. On the test years before it, a model is chosen only
when its trades beat SPY bought with the same dollars on the same dates, and every such model is traded
(run.choose_combination). To add a model, add its factory to MODELS and its grid to GRIDS; the backtest then tests
its trades against SPY every year.

A factory takes a model's own hyperparameters as keyword arguments and returns new, untrained models, e.g.
"gbm": lambda depth=3: replace(placeholder_models(), price=GBMPriceModel(depth), version=f"gbm-{depth}").
A grid may also name a Models field (horizon_days, bars_lookback, news_lookback_days, the exit settings hold_days,
stop_loss, take_profit and volatility_exit, and picks); the backtest sets those itself, so the factory never sees
them. picks is how many of a decision's ranked stocks a model trades (picking): a whole number N, its best N, or a
share such as "1%", its best 1% of the stocks it ranked that day, which a grid names through picks, never as
pick_share. The capital and the slice stay fixed, so picks spreads the same cash over more or fewer stocks.

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
# How many of each decision's ranked stocks every model trades, tuned with its other settings: its best 20 (the
# default, as before picks were tuned), 10 or 5, or its best 0.5% or 1% of the stocks it ranked that day (about 23
# or 47 of the ~4,700 tradable stocks; news_boost ranks only those with news, 650-2,150 a day). Like the exit
# settings, these share one fit per year.
PICKS = {"picks": (20, 10, 5, "0.5%", "1%")}
FOREST = {"min_samples_leaf": (1_000, 250), "max_features": (0.5, 1.0)}
GRIDS: dict[str, dict] = {  # most_liquid has no hyperparameters of its own, only its exit strategy and picks.
    "most_liquid": {**EXITS, **PICKS},
    "random_forest": {**FOREST, **EXITS, **PICKS},
    "news_forest": {**FOREST, **EXITS, **PICKS},
    "price_news_forest": {**FOREST, **EXITS, **PICKS},
    # It learns 20-session returns, so it holds 20 sessions by default.
    "news_boost": {"num_leaves": (7, 15), "min_child_samples": (2_000, 500), **EXITS, "hold_days": (20, 10, 5),
                   **PICKS},
}
UNTUNED = frozenset({"pick_share"})  # Fields no grid may name: a share is given as picks, e.g. "1%".


def configurations(grid: dict) -> list[dict]:
    """Return every combination of a grid's values, keys in sorted order, starting with the defaults."""
    keys = sorted(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[key] for key in keys))]


def picking(value) -> dict:
    """Return the Models fields a grid's picks value sets: a whole number N, a decision's best N stocks; a share such
    as "1%", its best 1% of the stocks it ranked that day (recommend.pick_count). Anything else raises ValueError."""
    if isinstance(value, str) and value.endswith("%"):
        try:
            share = float(value[:-1]) / 100
        except ValueError:
            share = -1.0
        if not 0 < share < 1:
            raise ValueError(f"picks {value!r} is not a share between 0% and 100%, such as \"1%\".")
        return {"pick_share": share}
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"picks {value!r} is neither a whole number of stocks, at least 1, nor a share such as "
                         "\"1%\".")
    return {"picks": value, "pick_share": None}


def configure(model, params: dict):
    """Return the model with the Models fields params name: picks through picking, the others as they are."""
    shared = {key: value for key, value in params.items() if key in MODEL_FIELDS and key != "picks"}
    if "picks" in params:
        shared.update(picking(params["picks"]))
    return replace(model, **shared) if shared else model


def build(factory, params: dict):
    """Return fresh models for one configuration: the factory gets its own hyperparameters, the backtest sets
    the Models fields."""
    return configure(factory(**{key: value for key, value in params.items() if key not in MODEL_FIELDS}), params)
