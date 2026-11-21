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

`src/stock_recommender` serves the stocks to buy and to sell at the next open, ranked by models run on the
downloaded daily bars and news: the best combination of models from a backtest run (see *The served models*), or,
until one is set, the placeholders of `stock_recommender/models.py` (`load_models`). Download the history once, then
start the API:

```bash
python scripts/download_daily_bars.py
python scripts/download_news.py
python -m uvicorn stock_recommender.api:app --app-dir src
```

Poll it at any time; `/docs` shows every parameter and the response schema:

```bash
# The latest recommendations, from the data on disk.
curl "http://127.0.0.1:8000/recommendations/latest?limit=20"
# Fetch new data first (about 20 s), run the models, then email the lists to these addresses.
curl -X POST "http://127.0.0.1:8000/recommendations/latest?fetch=true&email=you@example.com&limit=20"
```

The response holds a `buy` list (best scores first) and a `sell` list (worst scores first), with the bars date
(`as_of`), the news cutoff and the model version. A POST also reports how the fetch and the email went.

**What the models see.** Bars through the last trading day (`as_of`), and news published before its 16:00 New York
close. A fetch before the close only catches up on completed days, so `as_of` advances once a day, after the close.

**Snapshots.** The `as_of` bar and news fetched at 16:00 are a snapshot that a later fetch replaces with their final
versions, so the lists for the same `as_of` can still change; `generated_at` shows when they were computed.

**Fetching.** Each fetch re-downloads the last 90 weekdays of bars, because Polygon adjusts bars for the splits it
knows when they are downloaded; the download scripts also fetch again any day stored before it ended. The API does
nothing on its own: the email task below fetches and emails every weekday. `/health` shows the last fetch. Use a
single worker process: two would download into the same files at once.

### The served models

With `RECOMMENDER_MODEL_RUN` naming a backtest run, the API serves that run's final selection in place of
`load_models()`. That selection is the best combination of models and hyperparameters, chosen on 2022-2025 and fit on
2021-2025 (`selection.json` and `model/<name>.pkl`). Pin it in `.env`, then restart the API task:

```bash
RECOMMENDER_MODEL_RUN=artifacts/cv/20261113T174213Z-8f850acf
```

When a backtest finds a better combination, point this line at its run and restart the task. A pinned run never
changes by itself.
- **Ranking.** Each selected model ranks the stocks from the newest files through the backtest's decision path
  (`score_day`). It uses the pickle the run saved (never refit), and the run's price and dollar-volume filters and news
  inputs.
- **Lists.** They hold each model's top `limit` buys and worst `limit` sells. Each item carries its `model`, `run`
  names the run, and `model_version` joins the models' versions.
- **Health.** `/health` shows whether each model loaded. If none loads, the API answers 503 and the error goes to
  `logs/api.log`.

### Settings, email and the scheduled tasks

Settings are read from the environment or `.env` as `RECOMMENDER_<NAME>` (see `stock_recommender/config.py`). To send
email, e.g. through Gmail with an [app password](https://support.google.com/accounts/answer/185833), add to `.env`:

```bash
RECOMMENDER_SMTP_HOST=smtp.gmail.com
RECOMMENDER_SMTP_USER=you@gmail.com
RECOMMENDER_SMTP_PASSWORD=your-app-password
RECOMMENDER_EMAIL_TO=you@gmail.com,teammate@example.com
```

The API sends through that server to the addresses each request gives (`email=`), and `RECOMMENDER_EMAIL_TO` is where
the email task asks it to send. The API has no authentication: keep it on 127.0.0.1 (uvicorn's default), since anyone
who can reach it can trigger fetches and emails.

On Windows, two scheduled tasks run all of this without starting anything by hand:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_task.ps1         # the API; -Uninstall removes it
powershell -ExecutionPolicy Bypass -File scripts/install_email_task.ps1   # the 16:00 email; -Uninstall removes it
```

- **"ML510 stock recommender"** runs `scripts/serve.py` without a window.
  - It starts when you log on, and again at 15:55 New York time on weekdays in case the API stopped. It exits at once
    if an API already listens on port 8000, e.g. one started by hand.
  - The API logs to `logs/api.log`.
  - It runs the code checked out in this folder and reads `.env` once, at startup. Restart it after pulling or
    switching branches, or after changing `.env`: `Stop-ScheduledTask` then `Start-ScheduledTask`, with
    `-TaskName "ML510 stock recommender"`.
- **"ML510 recommendations email"** runs `scripts/email_recommendations.py` every weekday at 16:00 New York time.
  - It calls `POST /recommendations/latest?fetch=true&email=<RECOMMENDER_EMAIL_TO>`, so the API fetches the day's bars
    and news, ranks, and emails the lists.
  - The email carries a warning when the fetch failed and the lists come from older data.
  - If the computer is asleep or off at 16:00, it runs when the computer is available again.
  - It logs to `logs/email.log`.

Both tasks run only while you are logged on. `-Python` picks another interpreter than Miniconda's `pythonw.exe`.
`python scripts/preview_email.py` shows the email from the files on disk without sending it.

## Dashboard

`stock_recommender/dashboard.py` writes a page that explains the recommendations, again at every call of the API: every
GET or POST of `/recommendations/latest`, by hand or by the 16:00 email task. Open
http://127.0.0.1:8000/dashboard (it refreshes every minute), or `data/processed/dashboard/index.html` directly. It has
two tabs; `/dashboard#backtest` opens the second, and the minute's refresh keeps the tab you chose.

The **Recommendations** tab:
- **The lists.** Each served model's buy and sell lists. Beside each stock: its value of the features its model's
  ranking follows most, with its percentile among the tradable stocks.
- **SHAP and LIME values.** Hover over a stock, or Tab to it, to see how much each feature moves it up or down the
  ranking, in points of rank percentile (100 is the best stock, and the average stock sits at 50).
  - The models rank from raw bars and news, not from a row of features, so both methods explain a surrogate: a
    gradient-boosted tree that predicts each scored stock's rank percentile from the 18 features and its dollar volume
    (the tie-break between equal scores). Its R² on the stocks held out of its fit, shown under each model, says how
    much the values tell about the model itself.
  - SHAP values are a permutation estimate. Walks start from 100 stocks spread along the ranking and switch the
    features to the stock's values one at a time, in a random order and its reverse. The values add up to the stock's
    predicted percentile minus the average stock's.
  - LIME fits a weighted ridge regression to 5,000 perturbations that keep each feature at the stock's value or take
    another stock's. LIME's kernel weighs them by how close they stay to the stock, so its values describe nearby
    changes and run smaller than SHAP's, which compare with the average stock.
  - `explain.py` computes both, without the shap or lime packages.
- **What drives each model's ranking.** For each of the ten price features (`features.py`) and the eight news features
  (`news_features.py`), the Spearman correlation between the feature and the model's ranking, across the stocks it
  scored (`explain.py`). This describes any model, a rule such as `most_liquid` as well as a fitted one. A price or news
  part with a `feature_importances()` method returning `{feature: importance}` also shows the importances it measures.
  The news features that read LLM scores stay n/a until the day's articles are scored.
- **Calls.** The latest calls, with their result, fetch and email: a count, never the addresses.

The **Backtest simulation** tab shows the served run's out-of-sample results, 2022-2025, from its tables. Each model,
the account and the SPY references get their mean yearly return, Sharpe, drawdown, profitable periods, IC and spread,
then their return each year and the cumulative profit. How the recommendations do in 2026 is never shown: 2026 is
reserved for the one final evaluation.

Explaining a new day's lists takes a few seconds and runs after the response, so it never delays a call. A failure is
logged and shown on the page; it never fails the call, and a failed call keeps the last lists on the page.
`RECOMMENDER_DASHBOARD_DIR` moves the folder (default `data/processed/dashboard`), and an empty value turns the
dashboard off. The folder holds `index.html`, `latest.json` (the last lists, with their explanation) and `calls.jsonl`
(one line per call).

## News sentiment from a local LLM

`stock_recommender/sentiment.py` scores each article with a model served by [Ollama](https://ollama.com) on this
computer (`qwen3:8b` by default). For each article tagged with 1 to 5 companies, it rates every one of them:
- event (earnings, guidance, analyst, deal, product, legal, management, capital, price_move, opinion, macro, other);
- relevance (0 mentioned, 1 a main subject, 2 about it);
- impact (0 nothing new, 1 routine, 2 major news);
- sentiment (-2 to 2).

Each company counts once (`CMCS.A` and `CMCSA` are one). Securities other than common shares don't count: preferred
shares, warrants, rights, units and when-issued shares. Polygon names them `JPMpC`, `IONQ.WS`, `SOULr`, `AIIA.U` and
`DDw` on NYSE. On Nasdaq they are a fifth letter after the issuer's ticker (`AGNCP`, `SOUNW`, `PRSRU`), dropped when
that ticker is tagged too; `GOOGL` and `LILAK` are share classes and stay. Polygon tags 675 of 2021-2025's JPM
articles with its six preferreds, which would otherwise push them past the limit. Articles with more companies
(roundups, lists) are not scored.

Only the stocks the recommender may trade are scored: a tagged stock that passes its filters (price and median dollar
volume, `min_price` and `min_dollar_volume`) on a session in the 8 days from the article's day, the days the news
features read it. That leaves out 15% of the 2021-2025 articles and a share of the companies of the others; those
companies are still hidden from the model. The filter reads the daily bars as traded
(`data/raw/polygon/daily_bars_unadjusted`, then the API's bars for the days it lacks), and on the latest day it gives
the recommender's universe exactly. Lowering the filters later only needs another `score_news.py` run, which scores
the stocks that became tradable; `--all-tickers` scores every tagged company.

The model sees only the title and summary, with summaries cut at 12,000 characters. Its answer is forced into a JSON
schema, with no thinking, at temperature 0, and asked for on one line: generating the answer is nearly all the time
a request takes, and pretty-printed JSON doubled its tokens. Requests use an 8,192-token context: that matches how
the llm_trader project runs the shared Ollama server, and Ollama would reload the model for another value.

The model does not see which company an article is about, or when it was written. Qwen3 was released in April
2025, so it may know what happened after most of the 2021-2025 news to a company it recognizes, or after a date it
reads. `stock_recommender/anonymize.py` rewrites each article before the model sees it:

- The scored companies become Company A, Company B...: their tickers, wherever they are written (`NYSE: F`, `$F`,
  `(F)`, `TECK.B`, `F stock`, "T, F and GM"), and their names. The two share classes of one company get one
  placeholder and one score.
- Every other company the text names gets the next placeholders: a ticker written after a market or a dollar sign,
  a name written before a ticker in brackets ("Li Auto Inc. (NASDAQ: LI)"), and the names and tickers the table
  marks as safe to hide in any article (Morgan Stanley, Apple, NVDA). The model is told to label only the listed
  companies.
- Dates go: "March 15, 2023", "15 mars 2023" and "3/15/2023" become `[date]`, "in March" `in [month]`, 2023 `20XX`
  (FY23 `FY 20XX`, '21 `'XX`), and the dateline of a press release ("DUNMORE, Pa., April 21, 2023 (GLOBE NEWSWIRE)
  --") is dropped. Relative times stay: last year, next quarter, Q3, Monday.
- Web addresses, e-mails, social handles and security codes (ISIN, CUSIP, WKN) become `[link]`, `[email]`,
  `[handle]` and `[code]`: they carry names and tickers.

"Ford (F) beat Q3 2023 estimates on March 15" reaches the model as "Company A beat Q3 20XX estimates on [date]",
and the prompt asks what the text says, not for a forecast. A scored company the text never names gets relevance 0.

The names to hide come from a table of aliases, built once from two sources:

```bash
python scripts/download_ticker_names.py    # every name Polygon gave each ticker since 2021, exchanges and OTC
python scripts/build_ticker_aliases.py     # the aliases the 2021-2025 news confirm, into data/processed
```

An alias is Polygon's name without its legal ending (Ford Motor), its first words (Ford), its other words (Disney),
its initials (TSMC, P&G), a name the news write before the ticker in parentheses ("Coke (KO)"), or the ticker
itself. The news decide which are kept. "American" is not hidden for American Airlines, nor "Solar" for First Solar:
most articles that contain them are about something else. "Target" is hidden except beside the words it forms other
names with in other articles ("Price Target"). A name its own articles write small (lululemon) is hidden written
small too. An alias applies to its ticker's articles from the day Polygon first gave that name, or the news first
wrote it before the ticker, since tickers pass from one company to another. The article itself adds names: the words
before its ticker in brackets, and the short names a press release defines ("Evergold" or the "Company"). Scoring
refuses to start without the table, or with one built for an older version of the masking.

Polygon tags some articles about a foreign company with a US ticker it shares: Boralex (TSX: BLX) with Bladex,
Saputo (TSX: SAP) with SAP. When the scored ticker is written only after a foreign market and the article never
names the scored company, the foreign company gets its own placeholder, so the scored one gets relevance 0.

What stays: a ticker that is also a word where the text uses the word ("artificial intelligence (AI)", "Houston,
TX"); a single letter where it is one (Plan B, Class C, F-150); another company's name after a first name
("Harrison Ford") or inside a place ("New Brunswick"); products (iPhone), people (Elon Musk), places and well-known
events; companies listed only abroad, which Polygon does not name (Commonwealth Bank of Australia, SK Hynix);
brokers' brands (RBC Capital Markets, BMO, TD Cowen). Rebuild the table now and then for new listings; scores
already stored keep the aliases they were made with.

A malformed or partial answer is asked again with sampling, with seeds that differ from run to run. A ticker still
unscored after 3 requests gets no score, and the next run asks again. One article the server can't answer doesn't
stop the run. A server that is down stops it, and so does one that leaves 12 articles in a row unanswered.

Scores go to `data/processed/news_sentiment/<year>/<day>.parquet`, one file per news day. Each row records the model,
its digest (the exact weights) and the prompt version. Changing the prompt means bumping `PROMPT_VERSION`: the stored
scores become stale and everything is scored again, but the old rows stay in the files, as do other models' rows.

`model_digests.json` in that directory records the weights the scores come from. If `qwen3:8b` is pulled again, every
run refuses to start, the API's fetches included, until you restore the old weights or accept the new ones once with
`python scripts/score_news.py --allow-new-digest --start <today> --end <today>`. Use a recent range, so no older
article is scored with the new weights. A run also stops if the weights change while it runs.

```bash
python scripts/score_news.py --dry-run             # what 2021-2025 holds to score, and how long it would take
python scripts/score_news.py --retry-minutes 10    # score 2021-2025: days of GPU time; resumes where it stopped
```

With `--retry-minutes`, a run left alone waits and goes on when Ollama restarts, instead of stopping.

Each of the API's fetches scores the new articles after downloading them (`RECOMMENDER_SCORE_NEWS=false` turns that
off), so Ollama must be running at 16:00, when the email task fetches; if it is not, the fetch reports the error and
those articles stay unscored until the next one. `RECOMMENDER_OLLAMA_URL`, `RECOMMENDER_OLLAMA_MODEL`, `RECOMMENDER_SENTIMENT_MAX_TICKERS` and
`RECOMMENDER_SENTIMENT_WORKERS` change the server, the model, the ticker limit and the requests sent at once.

`stock_recommender/news_features.py` turns the scores into eight features per stock and decision. They read the
articles published in the 7 days before the decision's 16:00 close, and "1d" ones those since the previous weekday's
close:
- article counts over 7 days and 1 day;
- sentiment weighted by relevance, over 7 days and 1 day;
- the sentiment of major news;
- the highest impact;
- whether there was earnings or guidance news;
- the sentiment of analyst actions.

`decision_news_features(news, tickers, cutoff)` computes them for a decision. `training_news_features(data)` computes
them in `fit`, next to the labels. Both give the same values for the same day and read no later article. The
backtest reads the scores through its 2021-2025 store, like the news, and `guards.json` shows the share of each
year's scorable articles (1 to 5 companies) that have a score for every company: 1.0 once a year is fully scored.

One caveat: an LLM can know what happened after an article. Hiding the companies and the years removes most of what
it could recognize, not all of it, and nothing here measures what is left. A backtest of 2021-2024 can therefore
still look better than live trading; late 2025 and the 2026 final test are cleaner, since Qwen3 cannot have seen them.

## Backtesting (2021-2025 only)

`src/stock_recommender/backtest` replays the API's own decisions over history to measure the yearly return of
each model, and of the account: the combination model, which starts at $0 and buys every stock every model
recommends for $1,000. It uses 2021-2025 data only and never anything from 2026, which is reserved for the
final test:
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
python scripts/backtest.py                     # about 2 minutes per model; --help lists the options
```

- **Training and tuning: nested walk-forward.** Training data never comes after the year it is scored on.
  - A *candidate* is a model with one set of hyperparameters (its grid is in `backtest/registry.py`).
  - Every candidate is fit on all the years before each test year (2022-2025; 2021 only trains), with
    training labels ending before the year's first trade, and then trades that year.
  - That run is out of sample, and serves both as the year's test run and as validation for later years.
  - **Hyperparameters**: for test year T, each model runs its candidate with the best mean yearly return
    over the test years before T. The first test year has no earlier year, so it runs the defaults (each
    grid's first values). Nothing that sizes the money is tuned: every order is $1,000, and how many stocks a
    model recommends (`picks`) is the model's own.
  - **Final choice**: the choice made on every test year 2022-2025 is saved for the 2026 test, together with
    models fit on all of 2021-2025. It is never evaluated here.
- **Model output**: at a session's close each model ranks the tradable stocks exactly as the API does, on its
  own bar window. Its recommendations (`score_day(...).trades`) are its best `picks` stocks with their price,
  the decision-day close.
- **Trading: $1,000 per recommended stock, from $0.** Each model, and the account that combines them, buys
  every stock on a list for `--order-value` ($1,000) at the next open and sells it at the open
  `horizon_days` later, when the next holding starts. Holdings stay inside each test year.
  - **The account, the combination model**: every backtested model's list, each model on its own horizon.
    With one model, the account is that model.
  - **No cash limit, nothing reinvested**: each order is funded with new money, so no order is ever scaled
    down, and what one trade earns or loses never changes another.
  - **Return**: a year's profit over its capital, the most the open orders cost at one time ($20,000 for 20
    picks). A year's mean daily return, volatility, Sharpe and drawdown are measured on that capital too.
    Over several years, each year counts as funded anew: the yearly returns are averaged, and the profits
    added up.
  - **Shares**: fractional, so each order buys exactly $1,000 at the open. With `--whole-shares` the share
    count is rounded down from the list price, and a stock priced above $1,000 is skipped.
  - **Costs**: netted per ticker at each open, so a stock sold and bought back at the same open pays only on
    the difference.
  - **Dividends**: a lot's dividends are reinvested in the lot rather than paid out as cash.
- **References**, with the account's money, fractional shares and the same costs:
  - `spy_hold`: SPY bought with the account's capital at its first entry and held to its last exit;
  - `spy_mirror`: SPY bought for every order the account filled, with the same dollars and dates;
  - `universe_ew`: the same dollars spread equally over the tradable universe of each order's decision day.
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
  - Everything that decides a trade (the model inputs, the universe, the ranking, the share count) uses only
    data up to the decision day.
  - Models are trained on a copy of the data cut before the year they trade.
  - No choice for a year sees that year or a later one: `select()` raises `LeakageError` otherwise.
  - Only the targets (training labels, realized returns, metrics) look ahead.
- **Models**: `--models` lists the models to backtest, by default every model in `backtest/registry.py`;
  `--ensemble NAME=MODEL:W,MODEL:W` adds a model that blends others' rank-normalized scores. There are two models:
  - `most_liquid` expects no move for any stock, so it buys the 20 most liquid ones. It has no hyperparameters, so
    nothing is tuned for it.
  - `random_forest` (`stock_recommender/forest.py`) is a random forest on the ten daily-bar features below, with no
    news. It learns each stock's 5-session forward return minus the day's average over the tradable stocks,
    clipped at the 1st and 99th percentiles, so it learns the order of the stocks rather than the market's move. It
    trains on one decision day in every 5 sessions, so the holdings never overlap. It buys its 20 best predictions.
    Its grid tunes the stocks a leaf averages at least (`min_samples_leaf`: 1,000 by default, or 250) and the share
    of the features each split tries (`max_features`: 0.5 by default, or 1.0). Its own feature importances show on
    the dashboard.

  To add a model, implement `fit` and `predict` (see `stock_recommender/models.py`), then register a factory
  and its grid in `backtest/registry.py`. A grid can name the model's own hyperparameters, and also
  `horizon_days` or `bars_lookback`, but not `picks`: a model sets how many stocks it recommends itself, and
  the backtest refuses a grid that tunes it.

  `stock_recommender/features.py` has ten features from the daily bars for a price model such as a random forest:
  returns over 1, 5 and 20 sessions, momentum over 60 sessions without the last week, the overnight gap, where the
  day closed in its range, 20-session volatility, the distance below the 60-session high, recent over usual volume,
  and dollar volume. `decision_features(bars)` computes them in `predict`, and `training_features(data)` in `fit`,
  next to the labels. Both give the same values for the same day, using no bar after it, and a split changes none
  of them. They read the 60 sessions models get by default (`LOOKBACK`); `training_features` refuses a model with a
  shorter `bars_lookback`.
- **Results** go to `artifacts/<label>/<stamp>/`:
  - `summary.csv` and `yearly.csv`: each model as chosen, the account and the references; the yearly returns
    and their mean.
  - `metrics.csv`: per test year and over all of them, the capital, profit, return, daily return, volatility,
    Sharpe, drawdown, turnover, IC, the comparison with each reference, and with random lists of the same
    size.
  - `scores.csv`: every candidate in every year, which is what the choices read.
  - `folds.csv`: what was chosen for each test year, on which years, and the account's capital, profit and
    return beside SPY's.
  - `orders.csv`: every account order, with the reason it was filled or skipped.
  - `periods.csv` and `positions.csv`: each model as chosen, trading alone; per holding period, what its orders
    cost, its profit before and after costs, and its turnover (the dollars traded at its entry over the dollars
    invested); and every order it placed, with its return and how it exited.
  - `profit.csv` and `figures/profit.png`: each strategy's profit at every open, from $0, test years added up.
  - `selection.json` and `model/<name>.pkl`: the final choice for the 2026 test. This is what the API serves
    once `RECOMMENDER_MODEL_RUN` names the run; a model that does not pickle cannot be served.
  - Also: `signals.csv`, `null.csv`, `anomalies.csv`, and `guards.json` (the files read, the last date of every
    input, the years behind each choice, and `final_evaluated: false`).

## Shared environment and library

When changing reusable library behavior, run:

```bash
python -m unittest discover -s tests -t . -v
```

Tests maintain the shared library. Review notebook and lab-helper changes by running the affected notebooks.
