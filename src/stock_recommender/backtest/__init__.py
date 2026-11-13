"""Backtest models on 2021-2025 only, never 2026: nested walk-forward training and tuning, each model's trades,
the account (the combination model: from $0, $1,000 per recommended stock), SPY references, and return metrics.

Run: python scripts/backtest.py --help
"""
from .run import BacktestConfig, BacktestResult, run_backtest
from .window import FIRST_DAY, LAST_DAY, DataError, LeakageError
