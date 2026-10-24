"""Simulate a portfolio through back-to-back holdings and value it at every open.

At each rebalance the portfolio trades from its drifted weights to the new target weights at the open, paying
cost_bps on the traded value (both ways), then holds without trading until the next rebalance. A ticker that
does not trade at the entry open keeps its weight in cash. Short positions pay borrow_bps a year.
"""
import numpy as np
import pandas as pd
from .panel import Panel
from .protocol import Period

SESSIONS_PER_YEAR = 252


def simulate(panel: Panel, periods: list[Period], weights: list[pd.Series], cost_bps: float, borrow_bps: float,
             capital: float, folds: list[int]) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Return the value at each open, one row per holding period, and one row per position.

    weights[k] holds period k's signed target weights by ticker (+ long, - short); the rest of the value is cash.
    """
    cost_rate, borrow_rate = cost_bps / 1e4, borrow_bps / 1e4 / SESSIONS_PER_YEAR
    value = capital
    drifted = pd.Series(dtype=float)
    curve = {panel.sessions[periods[0].entry]: capital} if periods else {}
    period_rows, position_rows = [], []
    for period, target, fold in zip(periods, weights, folds, strict=True):
        entry, exit_ = period.entry, period.exit
        returns = panel.forward_returns(period.decision, period.holding, target.index)
        filled = returns.exit_reason != "no_entry"
        weight = target.where(filled, 0.0)
        turnover = weight.sub(drifted, fill_value=0.0).abs().sum()
        cost = value * cost_rate * turnover
        start = value - cost
        held = weight[weight != 0]
        codes = panel.codes(held.index)
        received = panel.paid_out[entry + 1:exit_ + 1, codes] - panel.paid_out[entry, codes]  # Payouts held as cash.
        marks = (panel.mark_open[entry + 1:exit_ + 1, codes] + received) / panel.q_open[entry, codes] - 1
        short = -held[held < 0].sum()
        borrow = borrow_rate * short * np.arange(1, exit_ - entry + 1)
        path = start * (1 + marks @ held.to_numpy() - borrow)
        for session, mark in zip(range(entry + 1, exit_ + 1), path):
            curve[panel.sessions[session]] = mark
        gross = float(marks[-1] @ held.to_numpy()) if len(held) else 0.0
        end = float(path[-1]) if len(path) else start
        drifted = held * (1 + marks[-1]) * start / end if len(held) else pd.Series(dtype=float)
        period_rows.append({"fold": fold, "decision_date": panel.sessions[period.decision],
                            "entry_date": panel.sessions[entry], "exit_date": panel.sessions[exit_],
                            "selected": len(target), "filled": int(filled.sum()), "gross_return": gross,
                            "turnover": turnover, "cost": cost, "borrow": start * float(borrow[-1]) if len(borrow) else 0.0,
                            "value_before": value, "value_after": end, "net_return": end / value - 1})
        position_rows.append(returns.assign(fold=fold, decision_date=panel.sessions[period.decision],
                                            target_weight=target, weight=weight).rename_axis("ticker").reset_index())
        value = end
    if periods:
        liquidation = value * cost_rate * drifted.abs().sum()  # Sell everything at the last exit open.
        value -= liquidation
        curve[panel.sessions[periods[-1].exit]] = value
        period_rows[-1]["cost"] += liquidation
        period_rows[-1]["value_after"] = value
        period_rows[-1]["net_return"] = value / period_rows[-1]["value_before"] - 1
    positions = pd.concat(position_rows, ignore_index=True) if position_rows else pd.DataFrame()
    return pd.Series(curve, name="value").sort_index(), pd.DataFrame(period_rows), positions
