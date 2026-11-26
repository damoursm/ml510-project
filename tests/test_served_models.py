"""The best combination of models: a backtest run's final selection, loaded for the API to serve."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import uuid
import pandas as pd
from stock_recommender.backtest.report import write_run
from stock_recommender.config import Settings
from stock_recommender.models import RandomPriceModel
from stock_recommender.recommend import recommend
from stock_recommender.served_models import load_selection
from .backtest_fixtures import small_models
from .test_backtest import Store


def make_run(root: Path, models: dict, manifest: dict | None = None, **fields) -> Path:
    """Write a minimal backtest run, root/artifacts/t/<stamp>/: selection.json, manifest.json and the pickles."""
    folder = root / "artifacts" / "t" / f"20261117T000000Z-{uuid.uuid4().hex[:8]}"
    (folder / "model").mkdir(parents=True)
    selection = {"chosen_on_years": [2022, 2023, 2024, 2025], "training_end": "2025-12-31", "order_value": 1000.0,
                 "cost_bps": 5.0, "whole_shares": False, "min_price": 0.0, "min_dollar_volume": 0.0, "models": {},
                 "final_evaluated": False, "saved_models": {}}
    for name, model in models.items():
        selection["models"][name] = {"params": {}, "candidate": name, "version": model.version,
                                     "horizon_days": model.horizon_days, "bars_lookback": model.bars_lookback,
                                     "news_lookback_days": model.news_lookback_days, "picks": model.picks,
                                     "order_value": 1000.0}
        (folder / "model" / f"{name}.pkl").write_bytes(pickle.dumps(model))
        selection["saved_models"][name] = f"model/{name}.pkl"
    selection.update(fields)
    (folder / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    manifest = manifest or {"configuration": {"offset": 0, "sentiment_dir": None}, "source_sha256": {}}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


def write_picks(folder: Path, per_day: dict) -> None:
    """Write the run's tables/picks.csv with each model's final choice at its return per day, plus rows the account
    rule must not read: a list size not chosen, and an earlier year's choice."""
    rows = [{"choice": "final", "model": name, "chosen": True, "tuning_return_per_day": value}
            for name, value in per_day.items()]
    rows += [{"choice": "final", "model": name, "chosen": False, "tuning_return_per_day": -1.0} for name in per_day]
    rows += [{"choice": "2025", "model": name, "chosen": True, "tuning_return_per_day": -2.0} for name in per_day]
    (folder / "tables").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / "tables" / "picks.csv", index=False)


def edit_selection(folder: Path, change) -> None:
    selection = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
    change(selection)
    (folder / "selection.json").write_text(json.dumps(selection), encoding="utf-8")


class SelectionTest(Store):
    def test_a_backtest_run_loads(self):
        result = self.run_backtest()
        folder = write_run(result, "served", root=self.root)
        for spec in (str(folder), str(folder / "selection.json")):
            selection = load_selection(spec)
            self.assertIsNone(selection.error)
            (model,) = selection.models
            self.assertEqual(selection.loaded, (model,))
            self.assertIsNone(model.error)
            self.assertEqual((model.name, model.source, model.horizon_days, model.bars_lookback, model.picks),
                             ("flat", "model/flat.pkl", 2, 3, 2))
            self.assertEqual(model.models.version, result.final_models["flat"].version)
            self.assertEqual(model.sha256, hashlib.sha256((folder / "model/flat.pkl").read_bytes()).hexdigest())
            self.assertEqual(selection.sha256, hashlib.sha256((folder / "selection.json").read_bytes()).hexdigest())
            self.assertEqual((selection.sentiment_model, selection.code_changed, selection.errors), (None, (), []))
            self.assertEqual((selection.chosen_on_years, selection.training_end), ((2025,), "2025-03-31"))
            self.assertEqual(selection.run, folder.resolve().as_posix())

    def test_a_model_that_lists_a_share_of_its_stocks_loads_with_it(self):
        folder = make_run(self.root, {"flat": replace(small_models(), pick_share=0.1)})
        edit_selection(folder, lambda selection: selection["models"]["flat"].update(pick_share=0.1))
        (model,) = load_selection(str(folder)).models
        self.assertEqual((model.error, model.pick_share, model.models.pick_share), (None, 0.1, 0.1))
        for share in (0.2, "10%", None):  # Another share, not a share, or none while the pickle lists one.
            edit_selection(folder, lambda selection: selection["models"]["flat"].update(pick_share=share))
            self.assertIn("pick_share", load_selection(str(folder)).models[0].error, share)

    def test_the_selected_models_rank_as_the_backtest_decides(self):
        """Each loaded model ranks the newest files exactly as recommend() does with the run's filters."""
        noisy = small_models(price=RandomPriceModel(3), version="noise", horizon_days=3)
        folder = make_run(self.root, {"flat": small_models(), "noise": noisy}, min_price=45.0, min_dollar_volume=1e6)
        settings = Settings(bars_dir=self.root / "bars", news_dir=self.root / "news",
                            sentiment_dir=self.root / "sentiment", model_run=str(folder))
        results = load_selection(settings.model_run).recommend(settings)
        self.assertEqual([model.name for model, _ in results], ["flat", "noise"])
        served = replace(settings, min_price=45.0, min_dollar_volume=1e6, sentiment_dir=None)
        for (model, result), expected in zip(results, (small_models(), noisy)):
            direct = recommend(served, expected)
            self.assertEqual(result.as_of, direct.as_of)
            self.assertEqual(result.table.ticker.tolist(), direct.table.ticker.tolist())
            self.assertTrue((result.table.close >= 45.0).all())  # The run's filters, not the API's defaults.

    def test_a_bad_selection_is_an_error_not_an_exception(self):
        model = small_models()
        for fields in ({"training_end": "2026-01-05"}, {"chosen_on_years": [2026]}, {"chosen_on_years": [2021, 2022]}):
            selection = load_selection(str(make_run(self.root, {"flat": model}, **fields)))
            self.assertTrue(selection.error, fields)
            self.assertEqual((selection.models, selection.loaded), ((), ()), fields)
        folder = make_run(self.root, {"flat": model})
        (folder / "selection.json").unlink()
        self.assertIn("not a backtest run", load_selection(str(folder)).error)
        elsewhere = self.root / "runs" / "flat"  # Not shaped artifacts/<label>/<stamp>.
        shutil.copytree(make_run(self.root, {"flat": model}), elsewhere)
        self.assertIn("not a backtest run", load_selection(str(elsewhere)).error)
        self.assertIn("RECOMMENDER_MODEL_RUN is empty", load_selection("").error)
        (folder / "selection.json").write_text("{", encoding="utf-8")
        self.assertIn("cannot read", load_selection(str(folder)).error)

    def test_the_account_rule_makes_room_with_the_model_that_earned_least_per_day(self):
        models = {"liquid": small_models(), "news": small_models(version="news")}
        folder = make_run(self.root, models, capital=10_000.0, order_value=100.0)
        write_picks(folder, {"liquid": 0.0010, "news": 0.0007})
        selection = load_selection(str(folder))
        self.assertEqual((selection.make_room, selection.return_per_day),
                         ("news", (("liquid", 0.0010), ("news", 0.0007))))
        rule = selection.account_rule()
        self.assertEqual((rule.capital, rule.order_value, rule.cost_bps, rule.make_room, rule.idle_cash),
                         (10_000.0, 100.0, 5.0, "news", "SPY"))
        self.assertIn("From $10,000, buy $100 of each stock to buy at today's close", rule.summary)
        self.assertIn("first sell news's oldest trades, never one bought the same day", rule.summary)
        # No cash sits idle: it is held in SPY, and an order spends it before selling a trade.
        self.assertIn("Hold the idle cash in SPY from each close to the next: an order spends it first.", rule.summary)
        self.assertNotIn("Idle cash stays cash", rule.summary)
        # Without the run's measure, or with one model, there is no model to make room with: such an order is skipped.
        (folder / "tables" / "picks.csv").unlink()
        self.assertIsNone(load_selection(str(folder)).make_room)
        self.assertIn("When an order finds no cash, skip it.", load_selection(str(folder)).account_rule().summary)
        alone = make_run(self.root, {"liquid": small_models()}, capital=10_000.0, order_value=100.0)
        write_picks(alone, {"liquid": 0.0010})
        self.assertIsNone(load_selection(str(alone)).make_room)
        # A run that records no account states no rule.
        self.assertIsNone(load_selection(str(make_run(self.root, models))).account_rule())

    def test_the_choice_the_final_test_evaluated_is_still_served(self):
        # The 2026 test measures the combination the API serves; it does not retire it.
        selection = load_selection(str(make_run(self.root, {"flat": small_models()}, final_evaluated=True)))
        self.assertIsNone(selection.error)
        self.assertEqual([model.name for model in selection.loaded], ["flat"])

    def test_a_model_that_does_not_load_is_left_out(self):
        good, other = small_models(), small_models(horizon_days=3, version="other")

        def saved_as(entry):
            return lambda folder: edit_selection(folder, lambda selection: selection["saved_models"].update(bad=entry))

        def outside(folder):
            shutil.copy(folder / "model/bad.pkl", folder.parent / "x.pkl")  # It exists, but outside the run.
            saved_as("../x.pkl")(folder)

        changes = {
            "missing": lambda folder: (folder / "model/bad.pkl").unlink(),
            "not saved": saved_as("not saved (PicklingError); refit from the params"),
            "truncated": lambda folder: (folder / "model/bad.pkl").write_bytes(
                (folder / "model/bad.pkl").read_bytes()[:40]),
            "outside": outside,
            "horizon": lambda folder: edit_selection(folder, lambda selection: selection["models"]["bad"].update(
                horizon_days=4)),
            "not a model": lambda folder: (folder / "model/bad.pkl").write_bytes(pickle.dumps({"score": 1})),
        }
        reasons = {}
        for label, change in changes.items():
            with self.subTest(label):
                folder = make_run(self.root, {"flat": good, "bad": other})
                change(folder)
                selection = load_selection(str(folder))
                self.assertIsNone(selection.error)
                models = {model.name: model for model in selection.models}
                self.assertEqual([model.name for model in selection.loaded], ["flat"])
                self.assertEqual(models["flat"].models.version, "placeholder")
                self.assertIsNone(models["bad"].models)
                self.assertIsNone(models["bad"].sha256)
                self.assertEqual(selection.errors, [f"bad did not load ({models['bad'].error})"])
                reasons[label] = models["bad"].error
        self.assertIn("FileNotFoundError", reasons["missing"])
        self.assertEqual(reasons["not saved"], "not saved (PicklingError); make it picklable and rerun "
                                               "scripts/backtest.py")
        self.assertIn("'../x.pkl' is not a .pkl file in the run folder", reasons["outside"])
        self.assertIn("horizon_days is 3, not the selection's 4", reasons["horizon"])
        self.assertIn("a dict is not a Models", reasons["not a model"])
        self.assertTrue(reasons["truncated"].startswith("model/bad.pkl: "))

    def test_news_inputs_and_filters_follow_the_run(self):
        scored = make_run(self.root, {"flat": small_models()}, manifest={
            "configuration": {"sentiment_dir": "data/processed/news_sentiment", "sentiment_model": "m"},
            "source_sha256": {}}, min_price=5.0, min_dollar_volume=1e6)
        settings = Settings(bars_dir=self.root / "bars", news_dir=self.root / "news",
                            sentiment_dir=self.root / "sentiment", min_price=1.0, min_dollar_volume=10.0)
        served = load_selection(str(scored)).settings(settings)
        self.assertEqual((served.ollama_model, served.sentiment_dir), ("m", self.root / "sentiment"))
        self.assertEqual((served.min_price, served.min_dollar_volume), (5.0, 1e6))
        plain = load_selection(str(make_run(self.root, {"flat": small_models()}))).settings(settings)
        self.assertIsNone(plain.sentiment_dir)  # The run read no LLM scores, so neither does the API for it.
        self.assertEqual(plain.ollama_model, settings.ollama_model)
