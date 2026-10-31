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

## Backtesting (2021-2025 only)

`src/stock_recommender/backtest` replays the API's own decisions over history to measure the return and
growth of one model or a combination of models. It uses 2021-2025 data only and never anything from 2026:
the bounds are constants, only the 2021..2025 folders are listed, every row read is checked against its
file's day, and models are trained on data cut at their fold's cutoff. The stored `daily_bars` were
split-adjusted in October 2026, so their 2021-2025 price levels carry 2026 splits; the backtest instead reads
prices as traded plus the 2021-2025 splits and dividends, downloaded once:

```bash
python scripts/download_daily_bars.py --unadjusted --start 2021-01-01 --end 2025-12-31 --out data/raw/polygon/daily_bars_unadjusted
python scripts/download_splits.py --start 2021-01-01 --end 2025-12-31 --out data/raw/polygon/splits_2021_2025.parquet
python scripts/download_dividends.py --start 2021-01-01 --end 2025-12-31 --out data/raw/polygon/dividends_2021_2025.parquet
python scripts/backtest.py                     # a few minutes; --help lists the options
```

- **Training and evaluation**: walk-forward. Each test year (2022-2025) is one fold whose models are fit on
  every earlier year (2021 only trains), and training labels end before the fold's first trade.
- **Trading**: at a session's close each model ranks the tradable stocks exactly as the API does, on its own
  bar window; the top 20 (`--top-n`) are bought equal-weight at the next open and sold at the open
  `horizon_days` later, when the next holding starts. `--side long_short` also shorts the sell list. Costs
  are 5 bps per side (`--cost-bps`). Returns are total returns: dividends (converted to dollars when paid in
  another currency) are reinvested at the ex-day open, payouts of 15% of the price or more (specials,
  liquidations) are held as cash until the holding ends, a split during a holding changes nothing, and a
  holding across a rename continues in the new ticker. Splits and dividends that Polygon files under a
  company's later ticker are moved to the ticker that traded them (once, even when Polygon lists them under
  both). Two curated files fix what the records miss: `configs/corporate_actions_2021_2025.csv` (spin-offs and
  cash deals, each checked to leave the holder's value continuous) and `configs/ticker_changes_2021_2025.csv`
  (renames detection misses, false ones it finds, and events filed under another company's ticker). What
  remains doubtful is in `anomalies.csv`.
- **No look-ahead**: everything that decides a trade (the model inputs, the universe, the ranking) uses only
  data up to the decision day's close, and models are trained on a copy of the data cut at their fold's
  cutoff; only the targets (training labels, realized returns, metrics) look ahead.
- **Models**: `--models` backtests each model alone; `--ensemble NAME=MODEL:W,MODEL:W` combines models by
  weighted, rank-normalized scores. The models are still placeholders (`placeholder` scores every stock 0,
  so it buys the most liquid ones; `random` is a no-skill baseline); a bare run also adds the demo ensemble
  of the two. To add a model, implement `fit` and `predict` (see `stock_recommender/models.py`) and register
  a factory in `backtest/registry.py`.
- **Results** go to `artifacts/<label>/<stamp>/`: growth, total return, CAGR, volatility, Sharpe, drawdown,
  turnover and information coefficient per strategy, overall and per fold, against an equal-weight universe
  and SPY, plus the equity curve (`figures/growth.png`), every holding and position, `anomalies.csv`, and
  `guards.json` (how many files were read, their first and last dates, and the last date of every input).

## Shared environment and library

When changing reusable library behavior, run:

```bash
python -m unittest discover -s tests -t . -v
```

Tests maintain the shared library. Review notebook and lab-helper changes by running the affected notebooks.
