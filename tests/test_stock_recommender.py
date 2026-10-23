from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import unittest.mock
import warnings
import pandas as pd

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*install `httpx2`")  # The httpx client still works.
    from fastapi.testclient import TestClient
from stock_recommender.api import create_app
from stock_recommender.config import ROOT, Settings
from stock_recommender.models import load_models


def write_bars(directory: Path, day: str, rows: dict[str, tuple[float, float]]):
    """rows maps ticker -> (close, volume)."""
    frame = pd.DataFrame([{"date": pd.Timestamp(day), "ticker": ticker, "open": close, "high": close, "low": close,
                           "close": close, "volume": volume, "vwap": close, "transactions": 1}
                          for ticker, (close, volume) in rows.items()])
    path = directory / day[:4] / f"{day}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def write_news(directory: Path, day: str, articles: list[tuple[str, list[str]]]):
    """articles lists (published_utc, tickers)."""
    frame = pd.DataFrame([{"published_utc": pd.Timestamp(published), "id": f"{published}-{tickers}", "title": "",
                           "tickers": tickers} for published, tickers in articles])
    path = directory / day[:4] / f"{day}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


class FixedPriceModel:
    def __init__(self, returns: dict[str, float]):
        self.returns = pd.Series(returns)

    def predict(self, bars, tickers):
        return self.returns


class MentionNewsModel:
    """Scores 1 for every ticker in the articles it sees, so the tests can tell which articles it saw."""

    def predict(self, news, tickers):
        return pd.Series(1.0, index=news.tickers.explode().unique())


class RecommendationsApiTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.settings = Settings(bars_dir=root / "bars", news_dir=root / "news", top_n=10, min_price=5.0,
                                 min_dollar_volume=1_000.0, refresh_time="")
        # BIG trades more dollars than ACME but sorts after it; PENNY is too cheap; THIN too illiquid;
        # GONE stopped trading; ZVZZT is an exchange test symbol.
        for day in ("2026-10-14", "2026-10-15"):
            write_bars(self.settings.bars_dir, day, {"BIG": (100, 1_000), "ACME": (50, 100), "PENNY": (1, 1e6),
                                                     "THIN": (10, 1), "GONE": (20, 1_000), "ZVZZT": (100, 1_000)})
        write_bars(self.settings.bars_dir, "2026-10-16", {"BIG": (100, 1_000), "ACME": (50, 100), "PENNY": (1, 1e6),
                                                          "THIN": (10, 1), "ZVZZT": (100, 1_000)})
        # The models see news published before the 16:00 New York close of 2026-10-16, 20:00 UTC.
        write_news(self.settings.news_dir, "2026-10-01", [("2026-10-01T12:00:00Z", ["THIN"])])
        write_news(self.settings.news_dir, "2026-10-15", [("2026-10-15T12:00:00Z", ["ACME"])])
        write_news(self.settings.news_dir, "2026-10-16", [("2026-10-16T19:59:00Z", ["ACME", "BIG"]),
                                                          ("2026-10-16T20:00:00Z", ["BIG"])])
        write_news(self.settings.news_dir, "2026-10-18", [("2026-10-18T23:00:00Z", ["ACME"])])

    def tearDown(self):
        self.directory.cleanup()

    def get(self, models=None, **params):
        client = TestClient(create_app(self.settings, models or load_models()))
        return client.get("/recommendations/latest", params=params)

    def test_placeholders_rank_tradable_tickers_by_dollar_volume(self):
        response = self.get()
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["as_of"], "2026-10-16")
        self.assertEqual(body["news_cutoff"], "2026-10-16T20:00:00Z")
        self.assertEqual(body["news_through"], "2026-10-16T19:59:00Z")
        self.assertEqual(body["model_version"], "placeholder")
        self.assertEqual(body["universe_size"], 2)
        rows = body["buy"]
        self.assertEqual([row["ticker"] for row in rows], ["BIG", "ACME"])
        self.assertEqual([row["rank"] for row in rows], [1, 2])
        self.assertEqual([row["news_count"] for row in rows], [1, 2])
        self.assertEqual(rows[0]["close"], 100.0)
        self.assertEqual(rows[0]["dollar_volume"], 100_000.0)

    def test_models_scores_set_the_ranking(self):
        models = replace(load_models(), price=FixedPriceModel({"BIG": 0.01, "ACME": 0.03}),
                         news=MentionNewsModel(), blend=lambda expected_return, sentiment: expected_return + sentiment)
        rows = self.get(models).json()["buy"]
        self.assertEqual([row["ticker"] for row in rows], ["ACME", "BIG"])
        self.assertAlmostEqual(rows[0]["score"], 1.03)
        self.assertEqual([row["news_sentiment"] for row in rows], [1.0, 1.0])

    def test_tickers_the_price_model_leaves_out_are_not_recommended(self):
        models = replace(load_models(), price=FixedPriceModel({"ACME": 0.02, "PENNY": 0.5}))
        rows = self.get(models).json()["buy"]
        self.assertEqual([row["ticker"] for row in rows], ["ACME"])

    def test_blend_sees_only_the_scored_tickers(self):
        def blend(expected_return, sentiment):
            self.assertFalse(expected_return.isna().any() or sentiment.isna().any())
            return (expected_return + sentiment).to_numpy()  # A fitted model may return a bare array.

        models = replace(load_models(), price=FixedPriceModel({"ACME": 0.02}), blend=blend)
        rows = self.get(models).json()["buy"]
        self.assertEqual([(row["ticker"], row["score"]) for row in rows], [("ACME", 0.02)])

    def test_news_model_gets_the_news_columns_when_there_is_no_news(self):
        class TitleNewsModel:
            def predict(self, news, tickers):
                return pd.Series(len(news.title), index=tickers, dtype=float)

        self.settings = replace(self.settings, news_dir=self.settings.news_dir / "missing")
        body = self.get(replace(load_models(), news=TitleNewsModel())).json()
        self.assertIsNone(body["news_through"])
        self.assertEqual([row["news_count"] for row in body["buy"]], [0, 0])

    def test_news_outside_the_lookback_is_not_seen(self):
        models = replace(load_models(), news=MentionNewsModel(), news_lookback_days=1)
        rows = self.get(models).json()["buy"]
        self.assertEqual({row["ticker"]: row["news_count"] for row in rows}, {"ACME": 1, "BIG": 1})

    def test_limit_applies_to_each_list_and_no_ticker_is_in_both(self):
        body = self.get(limit=1).json()
        self.assertEqual([row["ticker"] for row in body["buy"]], ["BIG"])
        self.assertEqual([row["ticker"] for row in body["sell"]], ["ACME"])
        self.assertEqual(self.get(limit=10).json()["sell"], [])  # Both tickers are already buys.
        self.assertEqual(self.get(limit=0).status_code, 422)

    def test_sell_list_starts_with_the_worst_score(self):
        write_bars(self.settings.bars_dir, "2026-10-16", {"BIG": (100, 1_000), "ACME": (50, 100), "CORE": (20, 500),
                                                          "DUD": (30, 500)})
        models = replace(load_models(), price=FixedPriceModel({"BIG": 0.01, "ACME": 0.03, "CORE": -0.01, "DUD": -0.05}))
        body = self.get(models, limit=2).json()
        self.assertEqual([row["ticker"] for row in body["buy"]], ["ACME", "BIG"])
        self.assertEqual([(row["rank"], row["ticker"]) for row in body["sell"]], [(1, "DUD"), (2, "CORE")])
        self.assertIsNone(body["fetch"])
        self.assertIsNone(body["email"])

    def test_new_bars_refresh_the_recommendations(self):
        client = TestClient(create_app(self.settings, load_models()))
        self.assertEqual(client.get("/recommendations/latest").json()["as_of"], "2026-10-16")
        write_bars(self.settings.bars_dir, "2026-10-19", {"ACME": (50, 100)})
        body = client.get("/recommendations/latest").json()
        self.assertEqual(body["as_of"], "2026-10-19")
        self.assertEqual([row["ticker"] for row in body["buy"]], ["ACME"])

    def test_missing_bars_is_service_unavailable(self):
        self.settings = replace(self.settings, bars_dir=self.settings.bars_dir / "missing")
        response = self.get()
        self.assertEqual(response.status_code, 503)
        self.assertIn("download_daily_bars.py", response.json()["detail"])

    def test_settings_from_environment(self):
        absolute = str(Path(self.directory.name).resolve())
        with unittest.mock.patch.dict("os.environ", {"RECOMMENDER_TOP_N": "3", "RECOMMENDER_NEWS_DIR": "elsewhere",
                                                     "RECOMMENDER_BARS_DIR": absolute}):
            settings = Settings.from_env()
        self.assertEqual(settings.top_n, 3)
        self.assertEqual(settings.news_dir, ROOT / "elsewhere")  # The download scripts write there too.
        self.assertEqual(settings.bars_dir, Path(absolute))


if __name__ == "__main__":
    unittest.main()
