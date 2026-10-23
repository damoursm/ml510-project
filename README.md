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
stored before it ended. `/health` shows the next and last run. Use a single worker process, or each one
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

## Shared environment and library

When changing reusable library behavior, run:

```bash
python -m unittest discover -s tests -t . -v
```

Tests maintain the shared library. Review notebook and lab-helper changes by running the affected notebooks.
