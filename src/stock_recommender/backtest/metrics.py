"""Return, growth, risk and signal metrics of a simulated portfolio.

Return and risk come from the value at every open (open-to-open daily returns, 252 a year, risk-free rate 0).
"""
import numpy as np
import pandas as pd

YEAR = 252


def max_drawdown(curve: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    """Return the deepest fall from a running peak, with the peak's and the trough's dates."""
    if len(curve) < 2:
        return 0.0, None, None
    drawdown = curve / curve.cummax() - 1
    trough = drawdown.idxmin()
    peak = curve.loc[:trough].idxmax()
    return float(drawdown.min()), peak, trough


def performance(curve: pd.Series, periods: pd.DataFrame | None = None) -> dict:
    """Summarize a value curve (and its holding periods): growth, return, risk and trading."""
    if len(curve) < 2:
        return {"growth": np.nan}
    daily = curve.pct_change().dropna()
    growth = float(curve.iloc[-1] / curve.iloc[0])
    years = len(daily) / YEAR
    volatility = float(daily.std(ddof=1) * np.sqrt(YEAR)) if len(daily) > 1 else np.nan
    downside = float(np.sqrt((np.minimum(daily, 0) ** 2).mean()) * np.sqrt(YEAR))
    drawdown, peak, trough = max_drawdown(curve)
    cagr = growth ** (1 / years) - 1 if years > 0 and growth > 0 else np.nan
    result = {
        "start": curve.index[0].date(), "end": curve.index[-1].date(),
        "start_value": float(curve.iloc[0]), "final_value": float(curve.iloc[-1]),
        "growth": growth, "total_return": growth - 1, "cagr": cagr,
        "volatility": volatility,
        "sharpe": float(daily.mean() / daily.std(ddof=1) * np.sqrt(YEAR)) if volatility and volatility > 0 else np.nan,
        "sortino": float(daily.mean() * YEAR / downside) if downside > 0 else np.nan,
        "max_drawdown": drawdown, "drawdown_peak": peak.date() if peak is not None else None,
        "drawdown_trough": trough.date() if trough is not None else None,
        "calmar": cagr / -drawdown if drawdown < 0 and np.isfinite(cagr) else np.nan,
    }
    if periods is not None and len(periods):
        result.update({
            "periods": len(periods), "period_hit_rate": float((periods.net_return > 0).mean()),
            "avg_turnover": float(periods.turnover.mean()), "total_cost": float(periods.cost.sum()),
            "total_borrow": float(periods.borrow.sum()), "avg_positions": float(periods.filled.mean()),
        })
    return result


def relative(curve: pd.Series, benchmark: pd.Series) -> dict:
    """Compare a curve with a benchmark's over the same opens: excess return, tracking and beta."""
    joined = pd.concat([curve.pct_change(), benchmark.pct_change()], axis=1, join="inner").dropna()
    if len(joined) < 2:
        return {}
    active = joined.iloc[:, 0] - joined.iloc[:, 1]
    tracking = float(active.std(ddof=1) * np.sqrt(YEAR))
    variance = joined.iloc[:, 1].var(ddof=1)
    return {"excess_return": float(curve.iloc[-1] / curve.iloc[0] - benchmark.iloc[-1] / benchmark.iloc[0]),
            "tracking_error": tracking,
            "information_ratio": float(active.mean() * YEAR / tracking) if tracking > 0 else np.nan,
            "beta": float(joined.iloc[:, 0].cov(joined.iloc[:, 1]) / variance) if variance > 0 else np.nan}


def information_coefficient(scores: pd.Series, returns: pd.Series, minimum: int = 20) -> float:
    """Return the Spearman correlation of scores with realized returns across tickers; NaN when the scores are
    constant (as with the placeholders) or fewer than `minimum` tickers have both."""
    both = pd.concat([scores, returns], axis=1, join="inner").dropna()
    if len(both) < minimum or both.iloc[:, 0].nunique() < 2 or both.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(both.iloc[:, 0].rank().corr(both.iloc[:, 1].rank()))


def signal_summary(signals: pd.DataFrame) -> dict:
    """Summarize per-decision information coefficients and buy-minus-sell spreads.

    Holdings do not overlap, so each decision is one independent observation for the t-statistics.
    """
    result = {}
    for column in ("ic", "spread"):
        values = signals[column].dropna() if column in signals else pd.Series(dtype=float)
        mean, std = values.mean(), values.std(ddof=1)
        result[f"{column}_mean"] = float(mean) if len(values) else np.nan
        result[f"{column}_t"] = float(mean / std * np.sqrt(len(values))) if len(values) > 1 and std > 0 else np.nan
        result[f"{column}_days"] = int(len(values))
    result["ic_positive_share"] = float((signals.ic.dropna() > 0).mean()) if signals.ic.notna().any() else np.nan
    return result

