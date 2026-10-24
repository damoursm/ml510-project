"""The models a backtest can use, by name. Each factory returns new, untrained models.

Add trained models here, e.g. "gbm": lambda: replace(placeholder_models(), price=GBMPriceModel(), version="gbm").
Never register load_models(): once it loads models trained on 2021-2025, a backtest would test on what they saw.
"""
from dataclasses import replace
from ..models import Models, RandomPriceModel, placeholder_models


def random_models() -> Models:
    """Placeholder models whose price model scores at random: a no-skill baseline."""
    return replace(placeholder_models(), price=RandomPriceModel(0), version="random-0")


MODELS = {"placeholder": placeholder_models, "random": random_models}
