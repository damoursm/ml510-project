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
run refuses to start, the daily refresh included, until you restore the old weights or accept the new ones once with
`python scripts/score_news.py --allow-new-digest --start <today> --end <today>`. Use a recent range, so no older
article is scored with the new weights. A run also stops if the weights change while it runs.

```bash
python scripts/score_news.py --dry-run             # what 2021-2025 holds to score, and how long it would take
python scripts/score_news.py --retry-minutes 10    # score 2021-2025: days of GPU time; resumes where it stopped
```

With `--retry-minutes`, a run left alone waits and goes on when Ollama restarts, instead of stopping.

The daily refresh scores the new articles after downloading them (`RECOMMENDER_SCORE_NEWS=false` turns that off), so
Ollama must be running at 16:00; if it is not, the refresh reports the error and those articles stay unscored until
the next run. `RECOMMENDER_OLLAMA_URL`, `RECOMMENDER_OLLAMA_MODEL`, `RECOMMENDER_SENTIMENT_MAX_TICKERS` and
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
  `--ensemble NAME=MODEL:W,MODEL:W` adds a model that blends others' rank-normalized scores. So far there is
  one model, `most_liquid`: it expects no move for any stock, so it buys the 20 most liquid ones. It has no
  hyperparameters, so nothing is tuned for it.

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
  - `selection.json` and `model/<name>.pkl`: the final choice for the 2026 test.
  - Also: `signals.csv`, `null.csv`, `anomalies.csv`, and `guards.json` (the files read, the last date of every
    input, the years behind each choice, and `final_evaluated: false`).

## Shared environment and library

When changing reusable library behavior, run:

```bash
python -m unittest discover -s tests -t . -v
```

Tests maintain the shared library. Review notebook and lab-helper changes by running the affected notebooks.
