from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
import pandas as pd
from stock_recommender.api import create_app
from stock_recommender.config import DASHBOARD_DIR, ROOT, Settings
from stock_recommender.dashboard import render
from stock_recommender.explain import DESCRIPTIONS, MIN_STOCKS, correlations, lime_values, shap_values
from stock_recommender.features import FEATURES
from stock_recommender.models import load_models
from stock_recommender.news_features import NEWS_FEATURES
from .test_notify import TO, FakeSMTP
from .test_served_models import make_run
from .test_stock_recommender import TestClient, write_news

DAYS = pd.bdate_range(end="2026-10-16", periods=65)  # More than the 60 sessions the price features read.
TICKERS = [f"T{number:03d}" for number in range(120)]  # Enough for correlations and for the surrogate to learn.


def write_history(directory: Path, seed: int = 0) -> None:
    """Write DAYS of bars for TICKERS: random walks, each ticker trading more shares than the one before."""
    rng = np.random.default_rng(seed)
    closes = 20 + 80 * rng.random(len(TICKERS))
    for day in DAYS:
        closes = closes * (1 + rng.normal(0, 0.02, len(TICKERS)))
        volume = 1_000 * np.arange(1, len(TICKERS) + 1) * (1 + 0.1 * rng.random(len(TICKERS)))
        frame = pd.DataFrame({"date": day, "ticker": TICKERS, "open": closes * 0.99, "high": closes * 1.01,
                              "low": closes * 0.98, "close": closes, "volume": volume, "vwap": closes,
                              "transactions": 1})
        path = directory / str(day.year) / f"{day.date()}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)


class MomentumModel:
    """Expects each stock's return of the last 20 sessions to go on, so its ranking is ret_20d's."""

    def predict(self, bars, tickers):
        closes = bars.pivot(index="date", columns="ticker", values="close").sort_index()
        return (closes.iloc[-1] / closes.iloc[-21] - 1).reindex(tickers)


class MeasuredModel(MomentumModel):
    """A momentum model that also reports importances, as a fitted forest would."""

    def feature_importances(self):
        return {"ret_20d": 0.7, "volume_ratio": 0.3}


class DashboardTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.settings = Settings(bars_dir=self.root / "bars", news_dir=self.root / "news",
                                 sentiment_dir=self.root / "sentiment", min_dollar_volume=0.0, score_news=False,
                                 dashboard_dir=self.root / "dashboard", smtp_host="smtp.example.com",
                                 smtp_user="bot@example.com", smtp_password="secret")
        write_history(self.settings.bars_dir)
        write_news(self.settings.news_dir, "2026-10-16", [("2026-10-16T15:00:00Z", ["T029", "T003"])])
        self.board = self.settings.dashboard_dir

    def tearDown(self):
        self.directory.cleanup()

    def client(self, models=None, **changes) -> TestClient:
        return TestClient(create_app(replace(self.settings, **changes), models or load_models()))

    def snapshot(self) -> dict:
        return json.loads((self.board / "latest.json").read_text(encoding="utf-8"))

    def calls(self) -> list[dict]:
        return [json.loads(line) for line in (self.board / "calls.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_every_call_writes_the_page_again(self):
        client = self.client()
        empty = client.get("/dashboard").text
        self.assertIn("No call of /recommendations/latest has returned lists yet", empty)
        self.assertIn(">Backtest simulation</button>", empty)
        body = client.get("/recommendations/latest", params={"limit": 3}).json()
        buys = [item["ticker"] for item in body["buy"]]
        page = client.get("/dashboard").text
        self.assertEqual(page, (self.board / "index.html").read_text(encoding="utf-8"))  # Readable without the API.
        for ticker in buys:
            self.assertIn(f"<b>{ticker}</b>", page)
        self.assertIn("What drives each model", page)
        client.post("/recommendations/latest", params={"limit": 2})
        self.assertEqual([(call["method"], call["error"]) for call in self.calls()], [("GET", None), ("POST", None)])
        self.assertEqual(self.calls()[0]["buy"], {"placeholder": buys})
        self.assertEqual(len(self.snapshot()["lists"]["models"]["placeholder"]["buy"]), 2)  # The last call's lists.
        page = client.get("/dashboard").text
        self.assertLess(page.index(">POST<"), page.index(">GET<"))  # The newest call first.

    def test_the_drivers_name_what_the_ranking_follows(self):
        self.client().get("/recommendations/latest")
        (model,) = self.snapshot()["explanation"]["models"]
        # The placeholders give every stock the same score, so the dollar volume ranks them.
        outputs = {row["feature"]: row["correlation"] for row in model["outputs"]}
        self.assertEqual((outputs["score"], outputs["dollar_volume"]), (None, 1.0))
        self.assertEqual(model["drivers"][0]["feature"], "dollar_volume_20d")
        self.assertGreater(model["drivers"][0]["correlation"], 0.9)
        self.assertEqual((model["scored"], model["importances"]), (len(TICKERS), None))
        self.assertIn("rank by dollar volume", self.client().get("/dashboard").text)

        momentum = replace(load_models(), price=MomentumModel(), version="momentum")
        self.client(momentum).get("/recommendations/latest", params={"limit": 2})
        (model,) = self.snapshot()["explanation"]["models"]
        self.assertEqual(model["drivers"][0]["feature"], "ret_20d")
        self.assertAlmostEqual(model["drivers"][0]["correlation"], 1.0)
        self.assertEqual(model["columns"][0], "ret_20d")
        listed = self.snapshot()["lists"]["models"]["momentum"]
        for item in listed["buy"] + listed["sell"]:  # Each listed stock's value of its model's drivers.
            self.assertAlmostEqual(model["profile"][item["ticker"]]["ret_20d"][0], item["expected_return"])
        # And its percentile among the scored stocks: the best buy has the highest ret_20d, the first sell the lowest.
        self.assertEqual(model["profile"][listed["buy"][0]["ticker"]]["ret_20d"][1], 1.0)
        self.assertAlmostEqual(model["profile"][listed["sell"][0]["ticker"]]["ret_20d"][1], 1 / len(TICKERS))

    def test_hovering_a_stock_shows_its_shap_and_lime_values(self):
        momentum = replace(load_models(), price=MomentumModel(), version="momentum")
        self.client(momentum).get("/recommendations/latest", params={"limit": 3})
        (model,) = self.snapshot()["explanation"]["models"]
        self.assertGreater(model["fidelity"], 0.8)  # The surrogate learns a ranking by ret_20d.
        listed = self.snapshot()["lists"]["models"]["momentum"]
        self.assertEqual(set(model["local"]), {item["ticker"] for item in listed["buy"] + listed["sell"]})
        best, worst = model["local"][listed["buy"][0]["ticker"]], model["local"][listed["sell"][0]["ticker"]]
        for stock, sign in ((best, 1), (worst, -1)):
            name, shap, lime, value = stock["features"][0]  # The strongest feature by SHAP.
            self.assertEqual(name, "ret_20d")
            self.assertGreater(sign * shap, 0.2)  # Points of rank percentile, as fractions: 0.2 is 20 points.
            self.assertGreater(sign * lime, 0.1)
            self.assertTrue(value.endswith("%"))
            self.assertAlmostEqual(stock["base"], 0.5, delta=0.05)  # The start stocks spread along the ranking.
            self.assertAlmostEqual(stock["predicted"], stock["actual"], delta=0.15)
            self.assertAlmostEqual(sum(feature[1] for feature in stock["features"]) + stock["rest"],
                                   stock["predicted"] - stock["base"])  # The SHAP values add up.
        page = (self.board / "index.html").read_text(encoding="utf-8")
        for item in listed["buy"] + listed["sell"]:
            self.assertIn(f'data-model="momentum" data-ticker="{item["ticker"]}" tabindex="0"', page)
        data = json.loads(page.split('id="local-explanations">')[1].split("</script>")[0])
        self.assertEqual(data["momentum"]["stocks"], model["local"])
        self.assertIn('id="tip"', page)
        self.assertIn(f"an R² of {model['fidelity']:.2f} on the stocks held out of its fit", page)

    def test_the_placeholders_stocks_rank_by_their_dollar_volume(self):
        self.client().get("/recommendations/latest", params={"limit": 2})
        (model,) = self.snapshot()["explanation"]["models"]
        for stock in model["local"].values():
            self.assertEqual(stock["features"][0][0], "dollar_volume")  # The tie-break that ranks them.

    def test_shap_and_lime_failing_leaves_the_rest_explained(self):
        with (mock.patch("stock_recommender.dashboard.local_explanations", side_effect=RuntimeError("no tree")),
              self.assertLogs("uvicorn.error", "ERROR")):
            response = self.client().get("/recommendations/latest", params={"limit": 2})
        self.assertEqual(response.status_code, 200)
        (model,) = self.snapshot()["explanation"]["models"]
        self.assertEqual((model["local"], model["local_error"]), ({}, "RuntimeError: no tree"))
        page = (self.board / "index.html").read_text(encoding="utf-8")
        self.assertIn("No SHAP and LIME values: RuntimeError: no tree", page)
        self.assertNotIn("data-ticker=", page)
        self.assertIn("What drives each model", page)

    def test_a_snapshot_from_before_shap_and_lime_still_renders(self):
        self.client().get("/recommendations/latest", params={"limit": 2})
        snapshot = self.snapshot()
        for model in snapshot["explanation"]["models"]:
            for key in ("local", "local_error", "fidelity"):
                del model[key]
        page = render(snapshot, [])
        self.assertNotIn("data-ticker=", page)
        self.assertIn('id="local-explanations">{}</script>', page)

    def test_a_models_own_importances_are_shown(self):
        measured = replace(load_models(), price=MeasuredModel(), version="measured")
        self.client(measured).get("/recommendations/latest")
        (model,) = self.snapshot()["explanation"]["models"]
        self.assertEqual(model["importances"], {"ret_20d": 0.7, "volume_ratio": 0.3})
        self.assertIn("Importances the model measures", (self.board / "index.html").read_text(encoding="utf-8"))

    def test_email_addresses_are_never_recorded(self):
        with mock.patch("stock_recommender.notify.smtplib.SMTP", FakeSMTP):
            FakeSMTP.sent = []
            body = self.client().post("/recommendations/latest", params={"limit": 2, "email": TO}).json()
        self.assertEqual(len(body["email"]["sent_to"]), 2)
        self.assertEqual(self.calls()[-1]["email"], {"sent": 2, "error": None})
        for name in ("calls.jsonl", "latest.json", "index.html"):
            self.assertNotIn("example.com", (self.board / name).read_text(encoding="utf-8"), name)
        self.assertIn("sent to 2", (self.board / "index.html").read_text(encoding="utf-8"))

    def test_a_failed_call_is_recorded_and_the_last_lists_stay(self):
        self.client().get("/recommendations/latest", params={"limit": 2})
        listed = self.snapshot()["lists"]
        response = self.client(bars_dir=self.root / "missing").post("/recommendations/latest", params={"fetch": False})
        self.assertEqual(response.status_code, 503)
        failed = self.calls()[-1]
        self.assertEqual(failed["method"], "POST")
        self.assertIn("download_daily_bars.py", failed["error"])
        self.assertEqual(self.snapshot()["lists"], listed)  # latest.json keeps the last lists.
        page = (self.board / "index.html").read_text(encoding="utf-8")
        self.assertIn("failed: No daily bars", page)
        self.assertIn("The lists below are from the last call that returned some", page)
        self.assertIn(f"<b>{listed['models']['placeholder']['buy'][0]['ticker']}</b>", page)

    def test_an_explanation_that_fails_never_fails_the_call(self):
        with (mock.patch("stock_recommender.dashboard.explain", side_effect=RuntimeError("boom")),
              self.assertLogs("uvicorn.error", "ERROR")):
            response = self.client().get("/recommendations/latest", params={"limit": 2})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.snapshot()["explanation"], {"error": "RuntimeError: boom"})
        page = (self.board / "index.html").read_text(encoding="utf-8")
        self.assertIn("The lists are not explained: RuntimeError: boom", page)
        self.assertIn(f"<b>{response.json()['buy'][0]['ticker']}</b>", page)

    def test_the_served_runs_backtest_is_shown_without_2026(self):
        folder = make_run(self.root, {"liquid": load_models()})
        tables = folder / "tables"
        tables.mkdir()
        pd.DataFrame([{"strategy": "liquid", "scope": "all", "mean_yearly_return": 0.15, "sharpe": 0.58},
                      {"strategy": "liquid", "scope": "2022", "mean_yearly_return": -0.4},
                      {"strategy": "spy_hold", "scope": "all", "mean_yearly_return": 0.125}]).to_csv(
            tables / "metrics.csv", index=False)
        pd.DataFrame([{"strategy": "liquid", "2022": -0.4, "2025": 0.35, "2026": 0.9, "mean": 0.15}]).to_csv(
            tables / "yearly.csv", index=False)
        pd.DataFrame({"date": ["2022-01-03", "2025-12-24", "2026-01-05"], "liquid": [-10.0, 12_361.0, 99_999.0],
                      "spy_hold": [-10.0, 10_031.0, 88_888.0]}).to_csv(tables / "profit.csv", index=False)
        self.client(model_run=str(folder)).get("/recommendations/latest")
        backtest = self.snapshot()["backtest"]
        self.assertEqual([row["strategy"] for row in backtest["strategies"]], ["liquid", "spy_hold"])
        self.assertEqual((backtest["years"], backtest["profit"]["dates"]), (["2022", "2025"],
                                                                           ["2022-01-03", "2025-12-24"]))
        page = (self.board / "index.html").read_text(encoding="utf-8")
        self.assertIn("liquid (served)", page)
        self.assertIn("+15.0%", page)
        self.assertIn("<polyline", page)
        self.assertIn("2026 is reserved for the one final evaluation", page)
        for later in ("<th>2026</th>", "+90.0%", "99,999", "88,888"):
            self.assertNotIn(later, page)

    def test_the_page_has_a_recommendations_tab_and_a_backtest_simulation_tab(self):
        folder = make_run(self.root, {"liquid": load_models()})
        self.client(model_run=str(folder)).get("/recommendations/latest")
        page = (self.board / "index.html").read_text(encoding="utf-8")
        self.assertLess(page.index(">Recommendations</button>"), page.index(">Backtest simulation</button>"))
        recommendations, backtest = page.index('id="panel-recommendations"'), page.index('id="panel-backtest"')
        for section in ('id="lists"', 'id="drivers"', 'id="calls"'):
            self.assertTrue(recommendations < page.index(section) < backtest, section)
        self.assertGreater(page.index('id="backtest-results"'), backtest)
        # A tab's address is its name, so no element may carry it as id: choosing a tab would scroll to it.
        for name in ("recommendations", "backtest"):
            self.assertIn(f'data-tab="{name}"', page)
            self.assertNotIn(f'id="{name}"', page)
        self.assertIn("location.reload()", page)  # The minute's reload keeps the tab in the address.

    def test_without_a_dashboard_folder_nothing_is_written(self):
        client = self.client(dashboard_dir=None)
        self.assertEqual(client.get("/recommendations/latest").status_code, 200)
        self.assertEqual(client.get("/dashboard").status_code, 404)
        self.assertFalse(self.board.exists())

    def test_the_api_writes_the_dashboard_by_default(self):
        self.assertIsNone(Settings().dashboard_dir)  # Settings built in code, as in tests, write nothing.
        with mock.patch.dict("os.environ", {"RECOMMENDER_DASHBOARD_DIR": "elsewhere"}):
            self.assertEqual(Settings.from_env().dashboard_dir, ROOT / "elsewhere")
        with mock.patch.dict("os.environ", {"RECOMMENDER_DASHBOARD_DIR": ""}):
            self.assertIsNone(Settings.from_env().dashboard_dir)
        with mock.patch.dict("os.environ"):
            os.environ.pop("RECOMMENDER_DASHBOARD_DIR", None)
            self.assertEqual(Settings.from_env().dashboard_dir, DASHBOARD_DIR)


class CorrelationsTest(unittest.TestCase):
    def test_correlations_follow_the_ranking(self):
        ranked = pd.Index([f"S{number}" for number in range(MIN_STOCKS)])  # Best first.
        values = pd.DataFrame({"follows": np.arange(MIN_STOCKS, 0, -1.0), "against": np.arange(MIN_STOCKS, dtype=float),
                               "same": 1.0, "rare": [1.0] + [np.nan] * (MIN_STOCKS - 1)}, index=ranked)
        values = values.iloc[::-1]  # Matched to the ranking by ticker, whatever the rows' order.
        found = {row["feature"]: row["correlation"] for row in correlations(ranked, values)}
        self.assertAlmostEqual(found["follows"], 1.0)
        self.assertAlmostEqual(found["against"], -1.0)
        self.assertEqual((found["same"], found["rare"]), (None, None))

    def test_every_feature_is_described(self):
        self.assertEqual(set(DESCRIPTIONS), set(FEATURES + NEWS_FEATURES))


class ShapLimeTest(unittest.TestCase):
    weights = np.array([2.0, -1.0, 0.5, 0.0])
    x = np.array([1.0, 3.0, -2.0, 5.0])
    start = np.array([0.5, 1.0, 1.0, -1.0])

    def linear(self, rows):
        return rows @ self.weights + 3.0

    def test_shap_values_add_up_to_the_prediction(self):
        def curved(rows):
            return rows[:, 0] * rows[:, 1] + np.sin(rows[:, 2]) + rows[:, 3] ** 2

        starts = np.random.default_rng(1).normal(size=(50, 4))
        values, base = shap_values(curved, self.x, starts, np.random.default_rng(2))
        self.assertAlmostEqual(base + values.sum(), curved(self.x[None, :])[0], places=10)
        self.assertAlmostEqual(base, curved(starts).mean(), places=10)

    def test_shap_values_of_a_linear_model_are_its_weights_times_the_change(self):
        values, base = shap_values(self.linear, self.x, np.tile(self.start, (10, 1)), np.random.default_rng(3))
        np.testing.assert_allclose(values, self.weights * (self.x - self.start), atol=1e-12)
        self.assertAlmostEqual(base, self.linear(self.start[None, :])[0])

    def test_lime_values_of_a_linear_model_are_its_weights_times_the_change(self):
        background = np.tile(self.start, (20, 1))
        values, intercept, fit = lime_values(self.linear, self.x, background, np.ones(4), np.random.default_rng(4))
        np.testing.assert_allclose(values, self.weights * (self.x - self.start), rtol=0.01, atol=0.01)
        self.assertAlmostEqual(intercept, self.linear(self.start[None, :])[0], delta=0.05)
        self.assertGreater(fit, 0.999)


if __name__ == "__main__":
    unittest.main()
