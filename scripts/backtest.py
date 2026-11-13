"""Backtest models with a nested walk-forward on 2021-2025 data, never 2026, and report their yearly returns.

Every model candidate (a model with one set of hyperparameters, from backtest/registry.py) is fit on all the years
before each test year, then trades that year: it ranks the stocks at a session's close, buys every stock on its
list for the order value ($1,000) at the next open, and holds it for its horizon, back to back. The account, the
combination model, starts at $0 and buys every model's list the same way; a year's return is its profit over its
capital, the most its open orders cost at one time. SPY held with that capital, and SPY bought for every order,
are the references. A test year's hyperparameters are chosen on the test years before it only, by mean yearly
return; the first test year runs the defaults. The choice made on every test year is saved for the 2026 final
test, which this never runs. Results go to artifacts/<label>/.

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
from stock_recommender.backtest.report import format_summary, selection_table, write_run  # noqa: E402


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
                        help="Models to backtest (backtest/registry.py); one with a grid there has its "
                             "hyperparameters chosen on earlier test years.")
    parser.add_argument("--ensemble", action="append", type=ensemble, metavar="NAME=MODEL:W,MODEL:W",
                        help="A weighted blend of models' scores, backtested as one more model; repeatable.")
    parser.add_argument("--normalize", default=defaults.normalize, choices=["rank", "zscore", "none"],
                        help="How an ensemble puts its members' scores on one scale.")
    parser.add_argument("--test-years", nargs="+", type=int, default=list(defaults.test_years),
                        help="Consecutive years within 2022-2025. Each is traded by models trained on all earlier "
                             "years, with hyperparameters chosen on the test years before it.")
    parser.add_argument("--offset", type=int, default=defaults.offset, help="Sessions to delay each year's first entry.")
    parser.add_argument("--cost-bps", type=float, default=defaults.cost_bps, help="Per side, on the value traded.")
    parser.add_argument("--order-value", type=float, default=defaults.order_value,
                        help="Dollars spent on every recommended stock, each order funded anew.")
    parser.add_argument("--whole-shares", action="store_true",
                        help="Buy whole shares only (rounded down from the list price); default fractional.")
    parser.add_argument("--no-final-fit", action="store_true", help="Skip fitting the final choice on 2021-2025.")
    parser.add_argument("--min-price", type=float, default=defaults.min_price)
    parser.add_argument("--min-dollar-volume", type=float, default=defaults.min_dollar_volume)
    parser.add_argument("--null-seeds", type=int, default=defaults.null_seeds,
                        help="Random lists of the same size, to rank the models against.")
    parser.add_argument("--label", default="backtest", help="Folder name under artifacts/.")
    args = parser.parse_args()
    try:
        config = BacktestConfig(
            models=tuple(args.models), ensembles=tuple(args.ensemble or ()), normalize=args.normalize,
            test_years=tuple(args.test_years), offset=args.offset, cost_bps=args.cost_bps,
            order_value=args.order_value, whole_shares=args.whole_shares, final_fit=not args.no_final_fit,
            min_price=args.min_price, min_dollar_volume=args.min_dollar_volume, null_seeds=args.null_seeds)
        started = time.monotonic()
        result = run_backtest(config)
    except (LeakageError, DataError, ValueError) as error:
        raise SystemExit(f"backtest: {error}")
    print(result.folds.drop(columns=["validation_return"]).to_string(index=False), "\n")
    print(format_summary(result), "\n")
    print("Final choice for the 2026 test (never evaluated here):")
    print(selection_table(result).to_string(index=False), "\n")
    guards = result.guards
    print(f"Read {guards['files_opened']} files dated {guards['first_file_day']}..{guards['last_file_day']}, "
          f"none outside 2021-2025; last exit {guards['max_exit_date']}.")
    print(f"Saved to {write_run(result, args.label)} in {time.monotonic() - started:.0f}s.")


if __name__ == "__main__":
    main()
