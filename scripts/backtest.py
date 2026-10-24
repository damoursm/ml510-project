"""Backtest models walk-forward on 2021-2025 data, never 2026, and report return and growth.

Each test year is one fold: models are fit on every earlier year, then pick stocks at a session's close to buy
at the next open and hold for their horizon, rebalancing back to back. Every model is backtested alone, each
--ensemble as one more strategy, next to an equal-weight universe and SPY. Results go to artifacts/<label>/.

Needs the as-traded bars, splits and dividends (2021-2025 only), downloaded once:
  python scripts/download_daily_bars.py --unadjusted --start 2021-01-01 --end 2025-12-31 --out data/raw/polygon/daily_bars_unadjusted
  python scripts/download_splits.py --start 2021-01-01 --end 2025-12-31 --out data/raw/polygon/splits_2021_2025.parquet
  python scripts/download_dividends.py --start 2021-01-01 --end 2025-12-31 --out data/raw/polygon/dividends_2021_2025.parquet
"""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stock_recommender.backtest import BacktestConfig, DataError, LeakageError, run_backtest  # noqa: E402
from stock_recommender.backtest.registry import MODELS  # noqa: E402
from stock_recommender.backtest.report import format_summary, write_run  # noqa: E402


def ensemble(spec: str) -> tuple[str, tuple[tuple[str, float], ...]]:
    """Parse NAME=MODEL:WEIGHT,MODEL:WEIGHT (weights default to 1)."""
    name, _, members = spec.partition("=")
    if not name or not members:
        raise argparse.ArgumentTypeError(f"Expected NAME=MODEL:WEIGHT,MODEL:WEIGHT; got {spec!r}.")
    parsed = []
    for member in members.split(","):
        model, _, weight = member.partition(":")
        parsed.append((model.strip(), float(weight) if weight else 1.0))
    return name, tuple(parsed)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = BacktestConfig()
    parser.add_argument("--models", nargs="*", default=list(defaults.models), choices=sorted(MODELS),
                        help="Models to backtest one by one.")
    parser.add_argument("--ensemble", action="append", type=ensemble, metavar="NAME=MODEL:W,MODEL:W",
                        help="A weighted combination of models to backtest as one strategy; repeatable. "
                             "Without --models or --ensemble, the demo ensemble=placeholder:1,random:1 is added.")
    parser.add_argument("--normalize", default=defaults.normalize, choices=["rank", "zscore", "none"],
                        help="How an ensemble puts its members' scores on one scale.")
    parser.add_argument("--test-years", nargs="+", type=int, default=list(defaults.test_years),
                        help="Consecutive years within 2022-2025; each is trained on all earlier years.")
    parser.add_argument("--no-refit", action="store_true", help="Fit once, before the first test year.")
    parser.add_argument("--top-n", type=int, default=defaults.top_n, help="Positions per book.")
    parser.add_argument("--side", default=defaults.side, choices=["long", "long_short"])
    parser.add_argument("--rebalance-every", type=int, help="Sessions per holding; default the models' horizon.")
    parser.add_argument("--offset", type=int, default=defaults.offset, help="Sessions to delay the first entry.")
    parser.add_argument("--cost-bps", type=float, default=defaults.cost_bps, help="Per side, on the value traded.")
    parser.add_argument("--borrow-bps", type=float, default=defaults.borrow_bps, help="A year, on shorts.")
    parser.add_argument("--capital", type=float, default=defaults.capital)
    parser.add_argument("--min-price", type=float, default=defaults.min_price)
    parser.add_argument("--min-dollar-volume", type=float, default=defaults.min_dollar_volume)
    parser.add_argument("--null-seeds", type=int, default=defaults.null_seeds,
                        help="Random portfolios to rank the strategies against.")
    parser.add_argument("--label", default="backtest", help="Folder name under artifacts/.")
    args = parser.parse_args()
    try:
        config = BacktestConfig(
            models=tuple(args.models), ensembles=tuple(args.ensemble) if args.ensemble else
            defaults.ensembles if args.models == list(defaults.models) else (),
            normalize=args.normalize, test_years=tuple(args.test_years), refit=not args.no_refit, top_n=args.top_n,
            side=args.side, rebalance_every=args.rebalance_every, offset=args.offset, cost_bps=args.cost_bps,
            borrow_bps=args.borrow_bps, capital=args.capital, min_price=args.min_price,
            min_dollar_volume=args.min_dollar_volume, null_seeds=args.null_seeds)
        started = time.monotonic()
        result = run_backtest(config)
    except (LeakageError, DataError, ValueError) as error:
        raise SystemExit(f"backtest: {error}")
    print(result.folds.to_string(index=False), "\n")
    print(format_summary(result), "\n")
    guards = result.guards
    print(f"Read {guards['files_opened']} files dated {guards['first_file_day']}..{guards['last_file_day']}, "
          f"none outside 2021-2025; last exit {guards['max_exit_date']}.")
    print(f"Saved to {write_run(result, args.label)} in {time.monotonic() - started:.0f}s.")


if __name__ == "__main__":
    main()
