"""When a recommended stock is sold: each model's exit strategy, part of its settings like its horizon.

A stock bought at the 16:00 close of the day it is recommended is held at most hold_days trading days and sold at the
close then. Earlier rules, each off when None, sell it at the close where one fires (16:00, when the API's lists
come out):
- stop_loss: the close is this fraction below the entry price: the holding's maximum adverse excursion (MAE)
  reached the limit;
- take_profit: the close is this fraction above the entry price: its maximum favorable excursion (MFE) reached
  the target;
- volatility_exit: a close-to-close move of at least this many times the stock's daily volatility (its standard
  deviation over the 20 sessions through the decision day) happened while held, and a later move is back within
  one times it: the volatility an event provoked has settled.
A rule that would fire only at the last close before the scheduled exit changes nothing, and a stop wins a tie.
Excursions are measured at the closes, never intraday.
"""
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd

EXIT_FIELDS = ("hold_days", "stop_loss", "take_profit", "volatility_exit")  # Models fields; fitting ignores them.
VOLATILITY_SESSIONS = 20  # The daily volatility is measured over this many close-to-close moves.


@dataclass(frozen=True)
class ExitRule:
    hold_days: int  # Trading days from the entry close to the scheduled exit close.
    stop_loss: float | None = None
    take_profit: float | None = None
    volatility_exit: float | None = None

    def __post_init__(self):
        if isinstance(self.hold_days, bool) or int(self.hold_days) != self.hold_days or self.hold_days < 1:
            raise ValueError(f"hold_days must be a whole number of trading days, at least 1; got {self.hold_days!r}.")
        if self.stop_loss is not None and not 0 < self.stop_loss < 1:
            raise ValueError(f"stop_loss must be a fraction between 0 and 1; got {self.stop_loss!r}.")
        if self.take_profit is not None and not self.take_profit > 0:
            raise ValueError(f"take_profit must be a positive fraction; got {self.take_profit!r}.")
        if self.volatility_exit is not None and not self.volatility_exit > 1:
            raise ValueError("volatility_exit must exceed 1, the multiple of the daily volatility a move settles "
                             f"back within; got {self.volatility_exit!r}.")

    @property
    def early(self) -> bool:
        """Whether a rule can sell before the scheduled exit."""
        return any(value is not None for value in (self.stop_loss, self.take_profit, self.volatility_exit))

    def as_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        """The rule in words, for the email and the dashboard."""
        if not self.early:
            return f"hold {self.hold_days} trading days"
        rules = []
        if self.stop_loss is not None:
            rules.append(f"{100 * self.stop_loss:g}% below the entry price (stop loss)")
        if self.take_profit is not None:
            rules.append(f"{100 * self.take_profit:g}% above it (take profit)")
        if self.volatility_exit is not None:
            rules.append(f"with a daily move back within its daily volatility after one of at least "
                         f"{self.volatility_exit:g} times it (volatility settled)")
        listed = rules[0] if len(rules) == 1 else ", ".join(rules[:-1]) + " or " + rules[-1]
        return f"hold up to {self.hold_days} trading days; sell at a close {listed}"


def exit_rule(models) -> ExitRule:
    """A Models bundle's or an Ensemble's exit rule. Bundles pickled before the exit settings existed, and
    ensembles, hold for their horizon with no early rule."""
    return ExitRule(int(getattr(models, "hold_days", None) or models.horizon_days),
                    getattr(models, "stop_loss", None), getattr(models, "take_profit", None),
                    getattr(models, "volatility_exit", None))


def exit_points(rule: ExitRule, returns: np.ndarray, moves: np.ndarray,
                volatility: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return, per lot, the closes from its entry to its sale and the rule that sold it ('' when held to the end).

    returns and moves are lots x hold_days, one column per close after the entry (the first: the next session's):
    each close's return from the entry price, and its move from the close before (the first from the entry close).
    volatility is each lot's daily volatility before its decision; NaN, or 0, keeps the volatility rule off. A NaN
    return (a lot never filled) fires nothing. A rule firing at the k-th close after the entry sells there: k.
    """
    lots, hold = returns.shape
    sold, why = np.full(lots, hold, dtype=int), np.full(lots, "", dtype=object)

    def first(fires: np.ndarray) -> np.ndarray:  # Only the closes before the last can move the sale earlier.
        fires = fires[:, :hold - 1]
        return np.where(fires.any(axis=1), fires.argmax(axis=1) + 1, hold)

    fired = []
    with np.errstate(invalid="ignore", divide="ignore"):
        if rule.stop_loss is not None:
            fired.append(("stop_loss", first(returns <= -rule.stop_loss)))
        if rule.take_profit is not None:
            fired.append(("take_profit", first(returns >= rule.take_profit)))
        if rule.volatility_exit is not None:
            scale = np.where(volatility > 0, volatility, np.nan)
            size = np.abs(moves) / scale[:, None]
            spiked = np.logical_or.accumulate(size >= rule.volatility_exit, axis=1)
            before = np.zeros_like(spiked)
            before[:, 1:] = spiked[:, :-1]  # A spike at an earlier close than this one.
            fired.append(("volatility", first(before & (size <= 1.0))))
    for name, opens in fired:  # In this order: at the same open, a stop wins.
        earlier = opens < sold
        sold, why = np.where(earlier, opens, sold), np.where(earlier, name, why)
    return sold, why


def excursions(returns: np.ndarray, held: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each lot's maximum favorable and adverse excursions: its best close return while held, at least 0,
    and its worst, at most 0, over the closes after its entry up to its sale (the `held`-th). NaN when it was never
    filled."""
    inside = np.arange(returns.shape[1])[None, :] < held[:, None]
    with np.errstate(invalid="ignore"):
        best = np.where(inside, returns, -np.inf).max(axis=1)
        worst = np.where(inside, returns, np.inf).min(axis=1)
    return np.maximum(best, 0.0), np.minimum(worst, 0.0)


def daily_volatility(bars: pd.DataFrame, sessions: int = VOLATILITY_SESSIONS) -> pd.Series:
    """Each ticker's daily volatility in a window of bars: the standard deviation of its last `sessions`
    close-to-close moves; NaN with fewer. The API shows it beside each stock for volatility_exit."""
    if bars.empty:
        return pd.Series(dtype=float)
    closes = bars.pivot_table(index="date", columns="ticker", values="close", observed=True).sort_index()
    moves = closes.iloc[-(sessions + 1):].pct_change(fill_method=None).iloc[1:]
    return moves.std(ddof=1).where(moves.notna().sum() >= sessions)
