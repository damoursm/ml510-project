"""Backtest models on 2021-2025 only, never 2026: walk-forward training, trading and return metrics.

Run: python scripts/backtest.py --help
"""
from .run import BacktestConfig, BacktestResult, run_backtest
from .window import FIRST_DAY, LAST_DAY, DataError, LeakageError
