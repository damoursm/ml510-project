# ML510 Project - Stock Trading Recommendation

Project for the ML510 class, this is a basic stocks trading app that will give recommendations on which stocks to buy for a short term horizon. The recommendations will be produce by an AI model and will be served through an API. 

## Repository map
Based on the labs. 

```text
docs/              Report and documentation
src/ml_lib/        Shared ML teaching library
configs/           Lab data and execution settings
data/              Local datasets and explicitly generated data
artifacts/         Saved experiment results and models
scripts/           Environment, data, and execution tools
tests/             Shared-library unit tests
requirements.txt   Shared pinned course dependencies
```

## Recommendations API

`src/stock_recommender` serves the stocks to buy and to sell at the next open, ranked by models run
on the downloaded daily bars and news. The models are placeholders until trained ones are plugged into
`stock_recommender/models.py` (`load_models`). Download the history once, then start the API:

```bash
python scripts/download_daily_bars.py
python scripts/download_news.py
python -m uvicorn stock_recommender.api:app --app-dir src
```

Poll it at any time; `/docs` shows every parameter and the response schema:

```bash
# The latest recommendations, from the data on disk.
curl "http://127.0.0.1:8000/recommendations/latest?limit=20"
# Fetch new data first (about 20 s), run the models, then email the lists.
curl -X POST "http://127.0.0.1:8000/recommendations/latest?fetch=true&email=true&limit=20"
```

The response holds a `buy` list (best scores first) and a `sell` list (worst scores first), with the
bars date (`as_of`), the news cutoff and the model version; a POST also reports how the fetch and the
email went. The models see bars through the last trading day (`as_of`) and news published before its
16:00 New York close. Fetching before the close therefore only catches up on completed days, so `as_of`
advances once a day, after the close. The `as_of` bar and news fetched at 16:00 are a snapshot that a
later fetch replaces with their final versions, so the lists for the same `as_of` can still change;
`generated_at` shows when they were computed.

Every weekday at 16:00 New York time the API fetches the day's bars and news by itself, and at startup
if it was not running then. Each fetch re-downloads the last 90 weekdays of bars, because Polygon adjusts
bars for the splits it knows when they are downloaded; the download scripts also fetch again any day
stored before it ended. Once email is set up (below), each of these runs also emails the top `RECOMMENDER_TOP_N` buys and sells,
with a warning when the fetch failed and the lists come from older data; `RECOMMENDER_EMAIL_DAILY=false`
turns that off. `/health` shows the next and last run, and how the last email went. Use a single worker process, or each one
runs the downloads.

Settings are read from the environment or `.env` as `RECOMMENDER_<NAME>` (see
`stock_recommender/config.py`), e.g. `RECOMMENDER_REFRESH_TIME=off` to stop the daily fetch. To send
email, e.g. through Gmail with an [app password](https://support.google.com/accounts/answer/185833),
add to `.env`:

```bash
RECOMMENDER_SMTP_HOST=smtp.gmail.com
RECOMMENDER_SMTP_USER=you@gmail.com
RECOMMENDER_SMTP_PASSWORD=your-app-password
RECOMMENDER_EMAIL_TO=you@gmail.com,teammate@example.com
```

The API has no authentication: keep it on 127.0.0.1 (uvicorn's default), since anyone who can reach it
can trigger fetches and emails.

To keep the API running on Windows without starting it by hand, register a scheduled task once:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_task.ps1   # -Uninstall removes it
```

The task runs `scripts/serve.py`, without a window, when you log on and again at 15:55 New York time on
weekdays in case the API stopped; it exits at once if an API already listens on port 8000, e.g. one started
by hand. The API logs to `logs/api.log`. It runs the code checked out in this folder, so restart it after
pulling or switching branches (`Stop-ScheduledTask` then `Start-ScheduledTask` with
`-TaskName "ML510 stock recommender"`). It runs only while you are logged on and the computer is awake: one
asleep at 16:00 fetches and emails when it wakes. `-Python` picks another interpreter than Miniconda's
`pythonw.exe`.

## Backtesting (2021-2025 only)

`src/stock_recommender/backtest` replays the API's own decisions over history to measure the return and
growth of each model, and of a $10,000 account that trades every model's recommendations. It uses
2021-2025 data only and never anything from 2026, which is reserved for the final test:
- the bounds are constants;
- only the 2021..2025 folders are listed;
- every row read is checked against its file's day;
- models are trained on data cut before the year they trade.

The stored `daily_bars` were
split-adjusted in October 2026, so their 2021-2025 price levels carry 2026 splits; the backtest instead reads
prices as traded plus the 2021-2025 splits and dividends, downloaded once:

```bash
python scripts/download_daily_bars.py --unadjusted --start 2021-01-01 --end 2025-12-31 --out data/raw/polygon/daily_bars_unadjusted
python scripts/download_splits.py --start 2021-01-01 --end 2025-12-31 --out data/raw/polygon/splits_2021_2025.parquet
python scripts/download_dividends.py --start 2021-01-01 --end 2025-12-31 --out data/raw/polygon/dividends_2021_2025.parquet
python scripts/backtest.py                     # about 10 minutes; --help lists the options
```

- **Training and tuning: nested walk-forward.** Training data never comes after the year it is scored on.
  - A *candidate* is a model with one set of hyperparameters (its grid is in `backtest/registry.py`).
  - Every candidate is fit on all the years before each test year (2022-2025; 2021 only trains), with
    training labels ending before the year's first trade, and then trades that year.
  - That run is out of sample, and serves both as the year's test run and as validation for later years.
  - **Hyperparameters**: for test year T, each model runs its candidate with the best mean annualized log
    growth (`252·ln(end/start)` per day) over the test years before T. The first test year has no earlier
    year, so it runs the defaults (each grid's first values).
  - **Final choice**: the choice made on every test year 2022-2025 is saved for the 2026 test, together with
    models fit on all of 2021-2025. It is never evaluated here.
- **Model output**: at a session's close each model ranks the tradable stocks exactly as the API does, on its
  own bar window. Its recommendations (`score_day(...).trades`) are its best `picks` stocks with their price,
  the decision-day close.
- **Per-model trading**: each model buys its list equal-weight at the next open and sells at the open
  `horizon_days` later, when the next holding starts. Its holdings stay inside each test year. Reported per
  year: yearly growth, mean daily return (`daily_return`), CAGR, Sharpe, drawdown and IC.
- **The account**: $10,000 (`--capital`), long only, trading every model's list, each model on its own
  horizon. Each model's order size is a share of the account's value per recommended stock (`--sizes`).
  - **Order value**: computed from the account's value at the open of the decision day.
  - **Shares**: fractional, so each order buys exactly its value at the open. With `--whole-shares` the
    share count is rounded down from the list price, and a stock priced above its order is skipped.
  - **Cash**: orders that need more than the account holds are scaled down together. Cash never goes
    negative, so there is no leverage.
  - **Costs**: netted per ticker at each open, so a stock sold and bought back at the same open pays only on
    the difference.
  - **Dividends**: a lot's dividends are reinvested in the lot rather than paid out as cash.
  - **Tuning**: for test year T, the sizes are tuned on the test years before T by coordinate ascent
    (`--sweeps`), maximizing the account's mean annualized log growth. Each model's size moves in turn, and
    ties keep the smaller size. The first test year splits the account evenly.
- **Costs and returns**: costs are 5 bps per side (`--cost-bps`). Returns are total returns: dividends (converted to dollars when paid in
  another currency) are reinvested at the ex-day open, payouts of 15% of the price or more (specials,
  liquidations) are held as cash until the holding ends, a split during a holding changes nothing, and a
  holding across a rename continues in the new ticker. Splits and dividends that Polygon files under a
  company's later ticker are moved to the ticker that traded them (once, even when Polygon lists them under
  both). Two curated files fix what the records miss: `configs/corporate_actions_2021_2025.csv` (spin-offs and
  cash deals, each checked to leave the holder's value continuous) and `configs/ticker_changes_2021_2025.csv`
  (renames detection misses, false ones it finds, and events filed under another company's ticker). What
  remains doubtful is in `anomalies.csv`.
- **No look-ahead**:
  - Everything that decides a trade (the model inputs, the universe, the ranking, the order size) uses only
    data up to the decision day.
  - Models are trained on a copy of the data cut before the year they trade.
  - No choice for a year sees that year or a later one: `select()` raises `LeakageError` otherwise.
  - Only the targets (training labels, realized returns, metrics) look ahead.
- **Models**: `--models` lists the models to backtest; `--ensemble NAME=MODEL:W,MODEL:W` adds a model that
  blends others' rank-normalized scores. The models are still placeholders:
  - `placeholder` scores every stock 0, so it buys the most liquid ones; its grid tries 20 or 10 picks;
  - `random` is a no-skill baseline whose grid picks among three seeds, which checks the selection for bias.

  To add a model, implement `fit` and `predict` (see `stock_recommender/models.py`), then register a factory
  and its grid in `backtest/registry.py`. A grid can name the model's own hyperparameters, and also `picks`,
  `horizon_days` or `bars_lookback`. Candidates that differ only in `picks` share one fit.
- **Results** go to `artifacts/<label>/<stamp>/`:
  - `summary.csv` and `yearly.csv`: each model as chosen, the account, the equal-weight universe and SPY.
  - `metrics.csv`: per test year and chained, the growth, daily return, CAGR, volatility, Sharpe, drawdown,
    turnover, IC, and the comparison with random lists of the same size.
  - `scores.csv`: every candidate in every year, which is what the choices read.
  - `folds.csv`: what was chosen for each test year, on which years, and the account's growth.
  - `orders.csv`: every account order, with the reason it was filled, scaled or skipped.
  - `sizes.csv`: every order-size trial.
  - `selection.json` and `model/<name>.pkl`: the final choice for the 2026 test.
  - Also: the equity curve (`figures/growth.png`), every holding and position, `anomalies.csv`, and
    `guards.json` (the files read, the last date of every input, the years behind each choice, and
    `final_evaluated: false`).

## Shared environment and library

When changing reusable library behavior, run:

```bash
python -m unittest discover -s tests -t . -v
```

Tests maintain the shared library. Review notebook and lab-helper changes by running the affected notebooks.
