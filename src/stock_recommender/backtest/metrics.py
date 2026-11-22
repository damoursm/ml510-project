"""Return, risk and signal metrics of an account's ledger (backtest/account.py).

An account funds each order anew and reinvests nothing, so it is measured on its capital: its cumulative return
is its profit so far over that capital, and its daily return is the day's change in it (open to open, 252 a year,
risk-free rate 0).
"""
import numpy as np
import pandas as pd

YEAR = 252


def period_summary(periods: pd.DataFrame | None) -> dict:
    """Summarize a model's holding periods (account.holdings): their count, the share with a profit after costs,
    the turnover, the costs paid and the orders filled per holding."""
    if periods is None or not len(periods):
        return {}
    return {"periods": len(periods), "period_hit_rate": float((periods.net_profit > 0).mean()),
            "avg_turnover": float(periods.turnover.mean()), "total_fees": float(periods.fees.sum()),
            "avg_positions": float(periods.filled.mean())}


def ledger_performance(path: pd.Series, periods: pd.DataFrame | None = None) -> dict:
    """Summarize an account's cumulative return on its capital at every open: the mean daily return, risk, and
    the deepest drawdown, in points of capital (-0.1 is a fall of a tenth of the capital from the best point)."""
    if len(path) < 2:
        return period_summary(periods)
    daily = path.diff().dropna()
    volatility = float(daily.std(ddof=1) * np.sqrt(YEAR)) if len(daily) > 1 else np.nan
    downside = float(np.sqrt((np.minimum(daily, 0) ** 2).mean()) * np.sqrt(YEAR))
    drawdown = path - path.cummax()
    trough = drawdown.idxmin()
    peak = path.loc[:trough].idxmax()
    result = {
        "start": path.index[0].date(), "end": path.index[-1].date(), "daily_return": float(daily.mean()),
        "volatility": volatility,
        "sharpe": float(daily.mean() / daily.std(ddof=1) * np.sqrt(YEAR)) if volatility and volatility > 0 else np.nan,
        "sortino": float(daily.mean() * YEAR / downside) if downside > 0 else np.nan,
        "max_drawdown": float(drawdown.min()), "drawdown_peak": peak.date(), "drawdown_trough": trough.date(),
    }
    result.update(period_summary(periods))
    return result


def relative_daily(path: pd.Series, reference: pd.Series) -> dict:
    """Compare two cumulative returns on capital over the same opens: tracking error, information ratio and beta
    of the daily changes."""
    joined = pd.concat([path.diff(), reference.diff()], axis=1, join="inner").dropna()
    if len(joined) < 2:
        return {}
    active = joined.iloc[:, 0] - joined.iloc[:, 1]
    tracking = float(active.std(ddof=1) * np.sqrt(YEAR))
    variance = joined.iloc[:, 1].var(ddof=1)
    return {"tracking_error": tracking,
            "information_ratio": float(active.mean() * YEAR / tracking) if tracking > 0 else np.nan,
            "beta": float(joined.iloc[:, 0].cov(joined.iloc[:, 1]) / variance) if variance > 0 else np.nan}


def information_coefficient(scores: pd.Series, returns: pd.Series, minimum: int = 20) -> float:
    """Return the Spearman correlation of scores with realized returns across tickers; NaN when the scores are
    constant (as with the placeholders) or fewer than `minimum` tickers have both."""
    both = pd.concat([scores, returns], axis=1, join="inner").dropna()
    if len(both) < minimum or both.iloc[:, 0].nunique() < 2 or both.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(both.iloc[:, 0].rank().corr(both.iloc[:, 1].rank()))


def _independent(rows: pd.DataFrame) -> float:
    """How many independent observations the decisions are: each counts min(every, holding) / holding, from its
    `every` (sessions to the next decision) and `holding` columns; 1 each without them."""
    if not len(rows):
        return 0.0
    holding = rows["holding"].to_numpy(dtype=float) if "holding" in rows else np.ones(len(rows))
    every = rows["every"].to_numpy(dtype=float) if "every" in rows else holding
    return float(np.sum(np.minimum(every, holding) / holding))


def signal_summary(signals: pd.DataFrame) -> dict:
    """Summarize per-decision information coefficients and buy-minus-sell spreads.

    A decision's forward return spans its holding (the signal's `holding` column, in sessions; 1 when absent).
    Decisions closer together than that share sessions, as the daily decisions do, so each counts as `every /
    holding` of an independent observation for the t-statistics, where `every` is the sessions to the next one
    (at most the holding): back-to-back decisions count 1 each, daily ones of a 5-session holding 1/5.
    """
    result = {}
    for column in ("ic", "spread"):
        rows = signals[signals[column].notna()] if column in signals else signals.iloc[:0]
        values = rows[column] if column in rows else pd.Series(dtype=float)
        mean, std = values.mean(), values.std(ddof=1)
        observations = _independent(rows)
        result[f"{column}_mean"] = float(mean) if len(values) else np.nan
        result[f"{column}_t"] = (float(mean / std * np.sqrt(observations)) if len(values) > 1 and std > 0
                                 else np.nan)
        result[f"{column}_days"] = int(len(values))
    result["ic_positive_share"] = float((signals.ic.dropna() > 0).mean()) if signals.ic.notna().any() else np.nan
    return result

