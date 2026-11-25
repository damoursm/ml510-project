"""Run the 2026 final test of a backtest run's final choice (stock_recommender/final_test.py).

The run is RECOMMENDER_MODEL_RUN from .env, the one the API serves, unless --run names another. The test needs the
2026 data in the form the backtest reads for 2021-2025: as-traded bars, splits, dividends, and the news with their LLM
scores (scripts/score_news.py). <last day> is the last finished trading day:

  python scripts/download_daily_bars.py --unadjusted --start 2026-01-01 --end <last day> --out data/raw/polygon/daily_bars_unadjusted
  python scripts/download_splits.py --start 2026-01-01 --end <last day> --out data/raw/polygon/splits_2026.parquet
  python scripts/download_dividends.py --start 2026-01-01 --end <last day> --out data/raw/polygon/dividends_2026.parquet

It first checks that data against the backtest's on 2021-2025 and stops unless they match, but for a few listed stocks
whose 2021-2025 history only 2026 data completes; it also re-trades 2025 from it, for comparison with the run.
Results go to artifacts/final/<stamp>/: results.csv, orders.csv (the account's), orders_make_room.csv (the same
account under the served rule), parts.csv (each account's profit by model and idle cash, in dollars), changed.csv
(those stocks), final_test.json and README.md.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_recommender.config import Settings  # noqa: E402
from stock_recommender.final_test import CHECK_YEAR, TEST_YEAR, final_test  # noqa: E402

NAMES = {"cash": "kept as cash", "BIL": "in T-bills (BIL)", "SPY": "in SPY", "-": "fully invested"}
COUNTS = ("filled", "no_cash", "made_room")


def table(results) -> str:
    shown = results.copy()
    for column in ("return", "volatility", "max_drawdown", "invested_share"):
        shown[column] = shown[column].map(lambda value: f"{value:+.1%}" if column != "invested_share"
                                          else f"{value:.0%}")
    shown["return_per_day"] = shown.return_per_day.map(lambda value: f"{value:+.3%}")
    shown["sharpe"] = shown.sharpe.map(lambda value: f"{value:.2f}")
    shown["profit"] = shown.profit.map(lambda value: f"{value:+,.0f}")
    shown["idle_cash"] = shown.idle_cash.map(NAMES)
    for column in COUNTS:
        shown[column] = shown[column].map(lambda value: "" if value != value else f"{int(value)}")
    return shown.to_string(index=False)


def parts_table(parts) -> str:
    shown = parts.copy()
    shown["idle_cash"] = shown.idle_cash.map(NAMES)
    for column in ("gross", "costs", "profit"):
        sign = "" if column == "costs" else "+"
        shown[column] = shown[column].map(lambda value: f"{round(value):{sign},}")
    return shown.to_string(index=False)


def legend(make_room: str | None) -> str:
    skip = "account: an order the cash does not cover is skipped (the backtest's rule)."
    if not make_room:
        return f"Strategies: {skip}"
    return (f"Strategies: {skip} make_room: the served rule; such an order first sells {make_room}'s oldest open "
            f"trades, never one bought the same day (made_room counts those sales). It was chosen after the first "
            f"{TEST_YEAR} test, so {TEST_YEAR} does not test it.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="The run to test, artifacts/<label>/<stamp>; RECOMMENDER_MODEL_RUN by default.")
    parser.add_argument("--no-check", action="store_true", help=f"Do not re-trade {CHECK_YEAR} first.")
    args = parser.parse_args()
    run = args.run or Settings.from_env().model_run
    if not run:
        raise SystemExit("No run to test: pass --run or set RECOMMENDER_MODEL_RUN in .env.")
    started = datetime.now(timezone.utc)
    result = final_test(run, check=not args.no_check)
    out = ROOT / "artifacts/final" / started.strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    result.results.to_csv(out / "results.csv", index=False)
    result.orders.to_csv(out / "orders.csv", index=False)
    if result.served_orders is not None:
        result.served_orders.to_csv(out / "orders_make_room.csv", index=False)
    result.changed.to_csv(out / "changed.csv", index=False)
    if result.parts is not None:
        result.parts.to_csv(out / "parts.csv", index=False)
    summary = {"run": result.run, "test_year": TEST_YEAR, "first_session": result.first_session,
               "last_session": result.last_session, "sessions": result.sessions, "started_utc": started.isoformat(),
               "make_room": result.make_room,
               "history_changed": result.changed.to_dict(orient="records"), f"retraded_{CHECK_YEAR}": result.check,
               "data": result.audit, "results": result.results.to_dict(orient="records"),
               "parts": result.parts.to_dict(orient="records") if result.parts is not None else None}
    (out / "final_test.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    text = table(result.results)
    checked = ", ".join(f"{name} {pair['replayed']:+.2%} (run {pair['saved']:+.2%})"
                        for name, pair in result.check.items())
    history = (f"The data matches the backtest's on 2021-2025 (sessions, news, LLM scores, prices) but for "
               f"{len(result.changed)} stocks whose history 2026 data completes (changed.csv): "
               f"{', '.join(result.changed.ticker)}." if args.no_check is False else "Data not checked.")
    retraded = (f"{CHECK_YEAR} re-trades from this data, its models re-fit on 2021-{CHECK_YEAR - 1}, to {checked}."
                if checked else "")
    split = ("Each account's profit in dollars, by part: what each model's lots and its idle cash made before costs "
             "(gross), the costs, and after them (profit). The profits add up to the account's.\n\n"
             f"```\n{parts_table(result.parts)}\n```\n" if result.parts is not None else "")
    (out / "README.md").write_text(
        f"# {TEST_YEAR} final test of {result.run}\n\n"
        f"Trading days {result.first_session}..{result.last_session} ({result.sessions}); models fit on 2021-2025, "
        f"never refit or tuned on {TEST_YEAR}.\n\n{history}\n\n{retraded}\n\n{legend(result.make_room)}\n\n"
        f"```\n{text}\n```\n\n{split}", encoding="utf-8")
    print(f"{TEST_YEAR} final test of {result.run}, {result.first_session}..{result.last_session} "
          f"({result.sessions} trading days)\n{history}\n{retraded}\n")
    print(legend(result.make_room) + "\n")
    print(text)
    if result.parts is not None:
        print("\n" + parts_table(result.parts))
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
