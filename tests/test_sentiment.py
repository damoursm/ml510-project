from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
import pandas as pd
import requests
from stock_recommender import sentiment
from stock_recommender.anonymize import ALIAS_COLUMNS, Aliases
from stock_recommender.backtest.run import _scored_by_year
from stock_recommender.backtest.window import LeakageError, WindowedStore
from stock_recommender.config import ROOT
from stock_recommender.data import read_news
from stock_recommender.sentiment import (EVENTS, OllamaScorer, ScorerUnavailable, attach, check_digest, eligible,
                                         others, parse, read_day, score_day, targets, to_frame, write_day)
from .backtest_fixtures import write_news

DIGEST = "500a1f067a9f" + "0" * 52
FIELDS = ["company", "event", "relevance", "impact", "sentiment"]
A, B = "Company A", "Company B"  # What the LLM gets in place of an article's first and second company.
MASK = sentiment.mask


def unmasked(title, description, tickers, *_):
    """Stands in for anonymize.mask where a test is about the store: the fake LLM answers with the tickers."""
    return title, description, list(tickers)


def alias_table(*rows) -> pd.DataFrame:
    """An alias table of (ticker, alias, kind) rows, all used."""
    return pd.DataFrame([(ticker, alias, kind, 0, 0, 0, 0, "always", pd.NaT, False, "", "")
                         for ticker, alias, kind in rows], columns=ALIAS_COLUMNS)


def answer(*scores) -> str:
    return json.dumps({"scores": [dict(zip(FIELDS, score)) for score in scores]})


def values(event, relevance, impact, sentiment_) -> dict:
    return {"event": event, "relevance": relevance, "impact": impact, "sentiment": sentiment_}


class Reply:
    def __init__(self, content: str | None = None, status: int = 200, models: list | None = None):
        self.status_code, self.text = status, content or ""
        self._body = {"models": models} if models is not None else {"message": {"content": content}}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(self.status_code)


class FakeSession:
    """Answers chat requests from a list of replies (an exception is raised; the last reply repeats), and records
    the bodies sent. models maps the server's model names to their digests; down makes every request fail, and
    after_post(session) runs after each chat request is answered."""

    def __init__(self, replies=(), models=None, down=False, after_post=None):
        self.replies, self.bodies, self.down, self.after_post = list(replies), [], down, after_post
        self.models = {"qwen3:8b": DIGEST} if models is None else models

    def get(self, url, timeout):
        if self.down:
            raise requests.ConnectionError("refused")
        return Reply(models=[{"name": name, "digest": digest} for name, digest in self.models.items()])

    def post(self, url, json, timeout):
        self.bodies.append(json)
        if self.down:
            raise requests.ConnectionError("refused")
        if self.after_post:
            self.after_post(self)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, Reply) else Reply(reply)


class ParseTest(unittest.TestCase):
    def test_scores_by_ticker_whatever_the_case_or_punctuation(self):
        scores = parse(answer(("msft", "deal", 1, 1, 1), ("AAPL", "earnings", 2, 2, -2), ("ZZZ", "other", 0, 0, 0),
                              ("BRK-B", "capital", 1, 1, 1)), ["AAPL", "MSFT", "GOOG", "BRK.B"])
        self.assertEqual(scores, {"MSFT": values("deal", 1, 1, 1), "AAPL": values("earnings", 2, 2, -2),
                                  "BRK.B": values("capital", 1, 1, 1)})  # GOOG is left out, ZZZ was not asked.

    def test_malformed_answers_and_items(self):
        for content in ("not json", None, '{"other": []}', '{"scores": 3}'):
            self.assertEqual(parse(content, ["AAPL"]), {}, content)
        wrong = answer(("AAPL", "rumor", 1, 1, 1), ("AAPL", "deal", 3, 1, 1), ("AAPL", "deal", 1, 1, -3),
                       ("MSFT", "deal", 1, 1, 1))
        self.assertEqual(parse(wrong, ["AAPL"]), {})
        # A bad item spoils nothing else, and the first valid score of a ticker counts.
        mixed = json.dumps({"scores": [{"company": "AAPL"}, "junk", dict(zip(FIELDS, ("AAPL", "deal", 1, 1, 1))),
                                       dict(zip(FIELDS, ("AAPL", "legal", 2, 2, -2)))]})
        self.assertEqual(parse(mixed, ["AAPL"]), {"AAPL": values("deal", 1, 1, 1)})


class TargetsTest(unittest.TestCase):
    def test_each_common_share_once(self):
        jpm = ["JPM", "AMJB", "JPMpC", "JPMpD", "JPMpJ", "JPMpK", "JPMpL", "JPMpM"]
        self.assertEqual(targets(jpm), ["JPM", "AMJB"])
        self.assertEqual(targets(np.array(["ACB", "SMG", "SQ", "CGC", "ACB"])), ["ACB", "SMG", "SQ", "CGC"])
        self.assertEqual(targets(["BRK.B", "BRK.A", "IONQ", "IONQ.WS", "IONQ.WS.A", "DDw", "SOULr", "AIIA.U", "Ap",
                                  "AP"]), ["BRK.B", "BRK.A", "IONQ", "AP"])
        self.assertEqual(targets(None), [])
        # Nasdaq's fifth letter, when the issuer is tagged too: preferreds, notes, warrants, units, rights.
        self.assertEqual(targets(["AGNC", "AGNCL", "AGNCM", "AGNCN", "AGNCO", "AGNCP", "NLY"]), ["AGNC", "NLY"])
        self.assertEqual(targets(["RILY", "RILYK", "RILYB", "OPEN", "OPENL"]), ["RILY", "OPEN"])
        self.assertEqual(targets(["SOUN", "SOUNW", "PRSRU", "PRSR", "BRLIR", "BRLI", "TMUS", "TMUSI", "TMUSZ"]),
                         ["SOUN", "PRSR", "BRLI", "TMUS"])
        self.assertEqual(targets(["SOUNW"]), ["SOUNW"])  # Alone, it may be a company of its own.
        self.assertEqual(targets(["GOOG", "GOOGL", "LILA", "LILAK", "METC", "METCB", "PARA", "PARAA", "ZG", "Z"]),
                         ["GOOG", "GOOGL", "LILA", "LILAK", "METC", "METCB", "PARA", "PARAA", "ZG", "Z"])  # Classes.
        # One spelling per company: the one without punctuation.
        self.assertEqual(targets(["CCZ", "CMCS.A", "CMCSA", "VZ"]), ["CCZ", "CMCSA", "VZ"])
        self.assertEqual(targets(["CYCC", "CYCC.P", "CYCCP"]), ["CYCC"])
        # The tags left out that belong to a scored ticker's company are hidden with it.
        self.assertEqual(others(jpm), {tag: "JPM" for tag in jpm[2:]})
        self.assertEqual(others(["SOUN", "SOUNW", "CMCS.A", "CMCSA", "D", "DDw", "AAPL", "BRK.A", "BRK.B"]),
                         {"SOUNW": "SOUN", "CMCS.A": "CMCSA"})  # DDw is DD's, which is not scored: not D's.
        self.assertEqual((others(None), others(["JPMpC"])), ({}, {}))
        news = pd.DataFrame({"tickers": [jpm, ["ACB"] * 6, ["JPMpC"], None, [f"T{i}" for i in range(6)]]})
        self.assertEqual(list(eligible(news, 5)), [True, True, False, False, False])


class ScorerTest(unittest.TestCase):
    def setUp(self):
        sleep = mock.patch.object(sentiment.time, "sleep")
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)

    def test_request_is_deterministic_without_thinking_and_constrained_to_the_schema(self):
        session = FakeSession([answer((A, "product", 2, 1, 1))])
        scores = OllamaScorer(session=session).score_article("Apple ships", "A new phone.", ["AAPL"])
        self.assertEqual(scores, {"AAPL": values("product", 2, 1, 1)})
        body = session.bodies[0]
        self.assertEqual((body["model"], body["think"], body["stream"]), ("qwen3:8b", False, False))
        self.assertEqual(body["options"], {"temperature": 0, "seed": 0, "num_ctx": sentiment.NUM_CTX,
                                           "num_predict": sentiment.MAX_ANSWER_TOKENS})
        self.assertNotIn("keep_alive", body)  # The shared server keeps its own setting.
        scores = body["format"]["properties"]["scores"]  # One score per company asked, named as asked.
        self.assertEqual((scores["items"]["properties"]["company"]["enum"], scores["minItems"], scores["maxItems"]),
                         ([A], 1, 1))
        self.assertNotIn("enum", sentiment.ANSWER_SCHEMA["properties"]["scores"]["items"]["properties"]["company"])
        self.assertEqual(body["messages"][0]["content"], sentiment.SYSTEM_PROMPT)
        # Without a table of names, the ticker is hidden and the name is not.
        self.assertEqual(body["messages"][1]["content"], f"Title: Apple ships\nSummary: A new phone.\nCompanies: {A}")

    def test_the_llm_gets_the_article_without_its_companies_or_dates(self):
        aliases = Aliases(alias_table(("AAPL", "apple", "name"), ("AAPL", "AAPL", "ticker"),
                                      ("GOOGL", "alphabet", "name"), ("GOOG", "alphabet", "name"),
                                      ("GOOGL", "google", "brand"), ("GOOGL", "GOOGL", "ticker")))
        session = FakeSession([answer((A, "deal", 2, 1, 1), (B, "deal", 1, 1, -1))])
        scorer = OllamaScorer(session=session, aliases=aliases)
        scores = scorer.score_article("Apple (AAPL) and Google sign 2024 search deal",
                                      "CUPERTINO, Calif., March 10, 2024 (GLOBE NEWSWIRE) -- Apple's deal with "
                                      "Alphabet Inc. (NASDAQ: GOOGL, GOOG) runs to FY26; Samsung Electronics Co. "
                                      "(KRX: 005930) lost it on March 8. AAPL rose in March. See apple.com.",
                                      ["AAPL", "GOOGL", "GOOG"], pd.Timestamp("2024-03-10", tz="UTC"))
        body = session.bodies[0]
        # The dateline goes; a company the article is not scored for gets the next placeholder.
        self.assertEqual(body["messages"][1]["content"],
                         f"Title: {A} and {B} sign 20XX search deal\n"
                         f"Summary: {A}'s deal with {B} runs to FY 20XX; Company C lost it on [date]. {A} rose in "
                         f"[month]. See [link].\nCompanies: {A}, {B}")
        self.assertEqual(body["format"]["properties"]["scores"]["items"]["properties"]["company"]["enum"], [A, B])
        # The two classes of Alphabet's shares are one company: one score, for both tickers.
        self.assertEqual(scores, {"AAPL": values("deal", 2, 1, 1), "GOOGL": values("deal", 1, 1, -1),
                                  "GOOG": values("deal", 1, 1, -1)})

    def test_a_long_summary_is_cut_to_fit_the_context(self):
        session = FakeSession([answer((A, "other", 0, 0, 0))])
        OllamaScorer(session=session).score_article("t" * 5_000, "x" * 50_000, ["AAPL"])
        content = session.bodies[0]["messages"][1]["content"]
        self.assertIn("x" * sentiment.MAX_SUMMARY_CHARS + f"\nCompanies: {A}", content)
        self.assertNotIn("x" * (sentiment.MAX_SUMMARY_CHARS + 1), content)
        self.assertNotIn("t" * (sentiment.MAX_TITLE_CHARS + 1), content)

    def test_a_malformed_or_partial_answer_is_resampled_and_the_answers_merge(self):
        session = FakeSession(["oops", answer((A, "other", 1, 0, 0)), answer((B, "deal", 2, 1, 1))])
        first = OllamaScorer(session=session)
        scores = first.score_article("t", None, ["AAPL", "MSFT"])
        self.assertEqual(scores, {"AAPL": values("other", 1, 0, 0), "MSFT": values("deal", 2, 1, 1)})
        retry = sentiment.RETRY_TEMPERATURE
        self.assertEqual([(body["options"]["temperature"], body["options"]["seed"]) for body in session.bodies],
                         [(0, 0), (retry, first.retry_seed + 1), (retry, first.retry_seed + 2)])
        session = FakeSession([answer((A, "other", 1, 0, 0))])
        scorer = OllamaScorer(session=session)
        self.assertEqual(scorer.score_article("t", None, ["AAPL", "MSFT"]),
                         {"AAPL": values("other", 1, 0, 0)})  # MSFT stays unscored after 3 requests.
        self.assertEqual(len(session.bodies), 3)
        self.assertNotEqual(first.retry_seed, scorer.retry_seed)  # A later run does not repeat the same samples.
        self.sleep.assert_not_called()

    def test_an_article_the_server_cannot_answer_gets_no_score(self):
        for reply, posts in ((requests.ReadTimeout("slow"), 3), (Reply("", status=500), 3),
                             (Reply("bad input", status=400), 1)):
            session = FakeSession([reply])
            self.assertEqual(OllamaScorer(session=session).score_article("t", None, ["AAPL"]), {}, reply)
            self.assertEqual(len(session.bodies), posts)
            # A request that failed in transit is sent again as it was, not resampled, after 5 s then 15 s.
            self.assertTrue(all(body["options"]["seed"] == 0 for body in session.bodies))
        self.assertEqual([call.args[0] for call in self.sleep.call_args_list], [5, 15, 5, 15])

    def test_a_server_that_answers_no_article_stops_the_scoring(self):
        for reply in (Reply("runner died", status=500), Reply("bad", status=400)):
            scorer = OllamaScorer(session=FakeSession([reply]))
            for _ in range(sentiment.MAX_FAILURES_IN_A_ROW - 1):
                self.assertEqual(scorer.score_article("t", None, ["AAPL"]), {})
            with self.assertRaisesRegex(ScorerUnavailable, "in a row"):
                scorer.score_article("t", None, ["AAPL"])
        # An answer, even a partial one, starts the count again.
        session = FakeSession([Reply("bad", status=400)])
        scorer = OllamaScorer(session=session)
        for _ in range(sentiment.MAX_FAILURES_IN_A_ROW - 1):
            scorer.score_article("t", None, ["AAPL"])
        session.replies = [answer((A, "other", 1, 0, 0))]
        scorer.score_article("t", float("nan"), ["AAPL", "MSFT"])
        session.replies = [Reply("bad", status=400)]
        for _ in range(sentiment.MAX_FAILURES_IN_A_ROW - 1):
            scorer.score_article("t", None, ["AAPL"])

    def test_a_model_whose_answers_score_nothing_stops_the_scoring(self):
        scorer = OllamaScorer(session=FakeSession(["", "not json", answer((B, "other", 1, 0, 0))]))
        for _ in range(sentiment.MAX_FAILURES_IN_A_ROW - 1):
            self.assertEqual(scorer.score_article("t", None, ["AAPL"]), {})  # It answers, with nothing usable.
        with self.assertRaisesRegex(ScorerUnavailable, "in a row"):
            scorer.score_article("t", None, ["AAPL"])

    def test_an_unreachable_server_stops_the_scoring(self):
        with self.assertRaises(ScorerUnavailable):
            OllamaScorer(session=FakeSession(down=True)).score_article("t", None, ["AAPL"])
        with self.assertRaises(ScorerUnavailable):
            OllamaScorer(session=FakeSession([Reply("not found", status=404)])).score_article("t", None, ["AAPL"])
        with self.assertRaisesRegex(ScorerUnavailable, "ollama pull qwen3:8b"):
            OllamaScorer(session=FakeSession(models={"llama3:8b": DIGEST})).check()

    def test_new_weights_under_the_same_name_stop_the_scoring(self):
        session = FakeSession()
        scorer = OllamaScorer(session=session)
        scorer.check()
        self.assertEqual(scorer.digest, DIGEST)
        scorer.check()
        session.models = {"qwen3:8b": "f" * 64}
        with self.assertRaisesRegex(ScorerUnavailable, "changed"):
            scorer.check()
        latest = OllamaScorer(session=FakeSession(models={"qwen3:8b:latest": DIGEST}))
        latest.check()
        self.assertEqual(latest.digest, DIGEST)


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.path = self.root / "2025" / "2025-03-10.parquet"
        for patch in (mock.patch.object(sentiment.time, "sleep"), mock.patch.object(sentiment, "mask", unmasked)):
            patch.start()
            self.addCleanup(patch.stop)
        self.news = pd.DataFrame({
            "published_utc": pd.to_datetime(["2025-03-10 14:00", "2025-03-10 15:00", "2025-03-10 16:00"], utc=True),
            "id": ["a", "b", "c"], "title": ["A up", "B and C", "Roundup"], "description": [None, "x", "y"],
            "tickers": [["AAA"], ["BBB", "CCC"], [f"T{i}" for i in range(6)]]})

    def tearDown(self):
        self.directory.cleanup()

    def scorer(self, *scores):
        scores = scores or (("AAA", "deal", 2, 1, 1), ("BBB", "legal", 2, 2, -2), ("CCC", "legal", 0, 0, 0))
        return OllamaScorer(session=FakeSession([answer(*scores)]))

    def test_only_unscored_eligible_articles_are_sent_and_stale_scores_are_redone(self):
        self.assertEqual(list(eligible(self.news, 5)), [True, True, False])  # Six tickers: a roundup.
        scorer = self.scorer()
        self.assertEqual(score_day(self.news.iloc[:1], self.path, scorer, 5), (1, 0))
        self.assertEqual(score_day(self.news, self.path, scorer, 5), (1, 0))  # Only b is new.
        self.assertEqual(score_day(self.news, self.path, scorer, 5), (0, 0))
        self.assertEqual(len(scorer.session.bodies), 2)
        stored = read_day(self.path, "qwen3:8b")
        self.assertEqual(list(zip(stored.id, stored.ticker)), [("a", "AAA"), ("b", "BBB"), ("b", "CCC")])
        self.assertEqual(stored.sentiment.tolist(), [1, -2, 0])
        self.assertEqual(stored.model_digest.tolist(), [DIGEST] * 3)  # Which weights scored them.
        self.assertEqual(len(read_day(self.path, "llama3:8b")), 0)  # Another model's scores are not these.
        with mock.patch.object(sentiment, "PROMPT_VERSION", sentiment.PROMPT_VERSION + 1):
            self.assertEqual(len(read_day(self.path, "qwen3:8b")), 0)
            self.assertEqual(score_day(self.news, self.path, scorer, 5), (2, 0))  # A new prompt scores all again.

    def test_a_ticker_left_unscored_is_asked_again_on_the_next_run(self):
        partial = self.scorer(("AAA", "deal", 2, 1, 1), ("BBB", "legal", 2, 2, -2))
        self.assertEqual(score_day(self.news, self.path, partial, 5), (2, 1))  # b's CCC has no score.
        stored = read_day(self.path, "qwen3:8b")
        self.assertEqual(list(zip(stored.id, stored.ticker)), [("a", "AAA"), ("b", "BBB")])
        later = self.scorer(("BBB", "other", 1, 0, 1), ("CCC", "legal", 0, 0, 0))
        self.assertEqual(score_day(self.news, self.path, later, 5), (1, 0))  # Only b is sent.
        stored = read_day(self.path, "qwen3:8b")
        self.assertEqual(list(zip(stored.id, stored.ticker, stored.event)),
                         [("a", "AAA", "deal"), ("b", "BBB", "legal"), ("b", "CCC", "legal")])  # BBB's score stays.
        self.assertEqual(score_day(self.news, self.path, later, 5), (0, 0))

    def test_an_article_the_server_refuses_does_not_stop_its_day(self):
        session = FakeSession([Reply("bad input", status=400), answer(("BBB", "legal", 2, 2, -2),
                                                                      ("CCC", "legal", 0, 0, 0))])
        self.assertEqual(score_day(self.news, self.path, OllamaScorer(session=session), 5, workers=1), (2, 1))
        self.assertEqual(read_day(self.path, "qwen3:8b").id.tolist(), ["b", "b"])
        self.assertTrue(pd.read_parquet(self.path).event.notna().all())  # No placeholder rows.

    def test_nothing_is_written_when_the_server_fails_mid_day(self):
        def go_down(session):
            session.down = True  # After answering article a.
        session = FakeSession([answer(("AAA", "deal", 2, 1, 1))], after_post=go_down)
        with self.assertRaises(ScorerUnavailable):
            score_day(self.news, self.path, OllamaScorer(session=session), 5, workers=1)
        self.assertEqual(len(session.bodies), 4)  # a answered, then b three times.
        self.assertFalse(self.path.exists())

    def test_weights_pulled_again_mid_day_write_nothing(self):
        def pull(session):
            session.models = {"qwen3:8b": "f" * 64}
        session = FakeSession([answer(("AAA", "deal", 2, 1, 1), ("BBB", "legal", 2, 2, -2),
                                      ("CCC", "legal", 0, 0, 0))], after_post=pull)
        with self.assertRaisesRegex(ScorerUnavailable, "changed"):
            score_day(self.news, self.path, OllamaScorer(session=session), 5, workers=1)
        self.assertFalse(self.path.exists())

    def test_another_models_scores_stay_in_the_file(self):
        self.assertEqual(score_day(self.news, self.path, self.scorer(), 5), (2, 0))
        other = OllamaScorer(model="qwen3:14b", session=FakeSession([answer(("AAA", "other", 1, 0, 0))],
                                                                    models={"qwen3:14b": "e" * 64}))
        self.assertEqual(score_day(self.news.iloc[:1], self.path, other, 5), (1, 0))
        self.assertEqual(len(read_day(self.path, "qwen3:8b")), 3)
        self.assertEqual(read_day(self.path, "qwen3:14b").event.tolist(), ["other"])
        with mock.patch.object(sentiment, "PROMPT_VERSION", sentiment.PROMPT_VERSION + 1):
            self.assertEqual(score_day(self.news.iloc[:1], self.path, self.scorer(), 5), (1, 0))
        self.assertEqual(len(read_day(self.path, "qwen3:8b")), 3)  # The earlier prompt's scores are kept.
        self.assertEqual(len(pd.read_parquet(self.path)), 5)

    def test_the_store_keeps_to_the_weights_it_was_scored_with(self):
        session = FakeSession()
        scorer = OllamaScorer(session=session)
        scorer.check()
        check_digest(self.root, scorer)
        self.assertEqual(json.loads((self.root / sentiment.DIGESTS_FILE).read_text()), {"qwen3:8b": DIGEST})
        check_digest(self.root, scorer)
        session.models = {"qwen3:8b": "f" * 64}
        pulled = OllamaScorer(session=session)
        pulled.check()
        for _ in range(2):  # Every run, the daily refresh's too, until someone decides.
            with self.assertRaisesRegex(ScorerUnavailable, "allow-new-digest"):
                check_digest(self.root, pulled)
        check_digest(self.root, pulled, allow_new=True)
        check_digest(self.root, pulled)  # The new weights are the store's now.
        self.assertEqual(json.loads((self.root / sentiment.DIGESTS_FILE).read_text()), {"qwen3:8b": "f" * 64})

    def test_preferred_shares_and_repeated_tags_are_left_out_of_the_prompt(self):
        news = pd.DataFrame({
            "published_utc": pd.to_datetime(["2025-03-10 14:00", "2025-03-10 15:00"], utc=True), "id": ["j", "d"],
            "title": ["JPM earnings", "Cannabis"], "description": ["x", "y"],
            "tickers": [["JPM", "JPMpC", "JPMpD", "JPMpJ", "JPMpK", "JPMpL", "JPMpM"], ["ACB", "SMG", "ACB"]]})
        session = FakeSession([answer(("JPM", "earnings", 2, 2, 1), ("ACB", "other", 2, 0, 0),
                                      ("SMG", "other", 1, 0, 0))])
        self.assertEqual(score_day(news, self.path, OllamaScorer(session=session), 5, workers=1), (2, 0))
        prompts = [body["messages"][1]["content"].rsplit("\n", 1)[1] for body in session.bodies]
        self.assertEqual(prompts, ["Companies: JPM", "Companies: ACB, SMG"])
        stored = read_day(self.path, "qwen3:8b")
        self.assertEqual(list(zip(stored.id, stored.ticker)), [("j", "JPM"), ("d", "ACB"), ("d", "SMG")])
        attached = attach(news, stored)
        self.assertEqual([[score["ticker"] for score in cell] for cell in attached.llm_scores],
                         [["JPM"], ["ACB", "SMG"]])
        news["published_utc"] = pd.to_datetime(["2024-03-10 14:00", "2025-03-10 15:00"], utc=True)
        self.assertEqual(_scored_by_year(attach(news, stored), 5), {"2024": 1.0, "2025": 1.0})
        self.assertEqual(_scored_by_year(attach(news, stored[stored.ticker != "SMG"]), 5),
                         {"2024": 1.0, "2025": 0.0})  # d lacks SMG's score: not scored.
        roundup = pd.DataFrame({"published_utc": pd.to_datetime(["2025-03-11"], utc=True), "id": ["r"],
                                "tickers": [[f"T{i}" for i in range(6)]], "llm_scores": [None]})
        everything = pd.concat([attach(news, stored), roundup], ignore_index=True)
        self.assertEqual(_scored_by_year(everything, 5), {"2024": 1.0, "2025": 1.0})  # A roundup is never sent.

    def test_scores_are_stored_under_the_tickers_the_placeholders_stand_for(self):
        news = pd.DataFrame({
            "published_utc": pd.to_datetime(["2025-03-10 14:00", "2025-03-10 15:00"], utc=True), "id": ["j", "d"],
            "title": ["JPMorgan (JPM) beats", "Aurora and Scotts"], "description": ["JPMpC and JPM rose.", None],
            "tickers": [["JPM", "JPMpC"], ["ACB", "SMG"]]})
        session = FakeSession([answer((A, "earnings", 2, 2, 1), (B, "other", 1, 0, -1))])
        scorer = OllamaScorer(session=session, aliases=Aliases(alias_table(("JPM", "jpmorgan", "start"),
                                                                                ("JPM", "JPM", "ticker"))))
        with mock.patch.object(sentiment, "mask", MASK):
            self.assertEqual(score_day(news, self.path, scorer, 5, workers=1), (2, 0))
        prompts = [body["messages"][1]["content"] for body in session.bodies]
        self.assertEqual(prompts[0], f"Title: {A} beats\nSummary: {A} and {A} rose.\nCompanies: {A}")  # JPMpC is JPM's.
        stored = read_day(self.path, "qwen3:8b")
        self.assertEqual(list(zip(stored.id, stored.ticker, stored.sentiment)),
                         [("j", "JPM", 1), ("d", "ACB", 1), ("d", "SMG", -1)])

    def test_attach_puts_each_articles_scores_in_its_tickers_order(self):
        scores = self.scorer().score(self.news.iloc[1:2])
        attached = attach(self.news, scores)
        self.assertIsNone(attached.llm_scores[0])
        self.assertEqual([score["ticker"] for score in attached.llm_scores[1]], ["BBB", "CCC"])
        self.assertEqual(attached.llm_scores[1][0], {"ticker": "BBB", "event": "legal", "relevance": 2, "impact": 2,
                                                     "sentiment": -2})
        self.assertNotIn("llm_scores", self.news)

    def test_the_api_and_the_backtest_attach_the_stored_scores(self):
        write_news(self.root / "news", [("2025-03-10 14:00", ["AAA"]), ("2025-03-10 15:00", ["BBB"]),
                                        ("2026-01-05 15:00", ["AAA"])])
        ids = pd.read_parquet(self.root / "news/2025/2025-03-10.parquet").id.tolist()
        rows = [{"id": ids[0], "published_utc": pd.Timestamp("2025-03-10 14:00", tz="UTC"), "ticker": "AAA",
                 "event": "deal", "relevance": 2, "impact": 1, "sentiment": 1, "model": "qwen3:8b",
                 "model_digest": DIGEST, "prompt_version": sentiment.PROMPT_VERSION}]
        write_day(to_frame(rows), self.root / "sentiment/2025/2025-03-10.parquet")
        later = dict(rows[0], id="later", published_utc=pd.Timestamp("2026-01-05 15:00", tz="UTC"))
        write_day(to_frame([later]), self.root / "sentiment/2026/2026-01-05.parquet")
        api = read_news(self.root / "news", pd.Timestamp("2025-03-10", tz="UTC"), pd.Timestamp("2025-03-11", tz="UTC"),
                        self.root / "sentiment", "qwen3:8b")
        self.assertEqual(api.llm_scores[0][0]["event"], "deal")
        self.assertIsNone(api.llm_scores[1])
        store = WindowedStore(self.root / "bars", self.root / "news", self.root / "splits.parquet",
                              self.root / "sentiment", "qwen3:8b")
        backtest = store.read_news()
        self.assertEqual(len(backtest), 2)  # 2026 is never read, news or scores.
        self.assertEqual(backtest.llm_scores[0][0]["sentiment"], 1)
        self.assertFalse(any("2026" in str(path) for path in store.opened))
        misfiled = dict(rows[0], published_utc=pd.Timestamp("2025-03-11 15:00", tz="UTC"))
        write_day(to_frame([misfiled]), self.root / "sentiment/2025/2025-03-10.parquet")
        with self.assertRaises(LeakageError):  # A score must be of an article published on its file's day.
            store.read_news()


class ScoreNewsScriptTest(unittest.TestCase):
    """scripts/score_news.py against a fake Ollama."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        spec = importlib.util.spec_from_file_location("score_news", ROOT / "scripts/score_news.py")
        self.script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.script)
        self.session = FakeSession([answer((A, "deal", 2, 1, 1))])
        patch = mock.patch.object(sentiment.requests, "Session", lambda: self.session)
        patch.start()
        self.addCleanup(patch.stop)
        self.aliases = self.root / "aliases.parquet"
        alias_table(("AAA", "aaa corp", "name")).to_parquet(self.aliases)

    def tearDown(self):
        self.directory.cleanup()

    def run_script(self, *args) -> tuple[int, str]:
        """Run the script on the test's news; every target is scored unless the test gives bars (--bars-dir)."""
        every = [] if "--bars-dir" in args else ["--all-tickers"]
        argv = ["score_news.py", "--news-dir", str(self.root / "news"), "--out", str(self.root / "sentiment"),
                "--aliases", str(self.aliases), *every, *args]
        output = io.StringIO()
        with mock.patch("sys.argv", argv), redirect_stdout(output), redirect_stderr(output):
            return self.script.main(), output.getvalue()

    def recorded(self) -> dict:
        return json.loads((self.root / "sentiment" / sentiment.DIGESTS_FILE).read_text())

    def test_new_weights_are_refused_until_accepted_even_once_everything_is_scored(self):
        write_news(self.root / "news", [("2025-03-10 14:00", ["AAA"])])
        self.assertEqual(self.run_script("--start", "2025-03-10", "--end", "2025-03-10")[0], 0)
        self.assertEqual((self.recorded(), len(self.session.bodies)), ({"qwen3:8b": DIGEST}, 1))
        self.session.models = {"qwen3:8b": "f" * 64}  # Pulled again.
        write_news(self.root / "news", [("2025-03-11 14:00", ["BBB"])])
        code, output = self.run_script("--start", "2025-03-10", "--end", "2025-03-11")
        self.assertEqual((code, len(self.session.bodies)), (2, 1))
        self.assertIn("--allow-new-digest", output)
        # Accepting them works over a range with nothing to score, and scores nothing older.
        code, output = self.run_script("--allow-new-digest", "--start", "2025-03-12", "--end", "2025-03-12")
        self.assertEqual((code, self.recorded(), len(self.session.bodies)), (0, {"qwen3:8b": "f" * 64}, 1))
        self.assertEqual(self.run_script("--start", "2025-03-10", "--end", "2025-03-11")[0], 0)
        self.assertEqual(read_day(self.root / "sentiment/2025/2025-03-11.parquet", "qwen3:8b").model_digest.tolist(),
                         ["f" * 64])
        self.assertEqual(read_day(self.root / "sentiment/2025/2025-03-10.parquet", "qwen3:8b").model_digest.tolist(),
                         [DIGEST])


    def test_nothing_is_sent_without_the_table_of_names_to_hide(self):
        write_news(self.root / "news", [("2025-03-10 14:00", ["AAA"])])
        self.aliases.unlink()
        code, output = self.run_script("--start", "2025-03-10", "--end", "2025-03-10")
        self.assertEqual((code, len(self.session.bodies)), (2, 0))
        self.assertIn("build_ticker_aliases.py", output)
        self.assertEqual(self.run_script("--dry-run", "--start", "2025-03-10", "--end", "2025-03-10")[0], 0)

    def test_only_the_stocks_the_recommender_may_trade_are_scored(self):
        # AAA passes the filters ($5, $1M a day); BBB is a penny stock; CCC trades too little. All three are hidden.
        days = pd.bdate_range("2025-01-02", "2025-03-14")
        bars = pd.DataFrame([(day, ticker, close, volume) for day in days for ticker, close, volume in
                             (("AAA", 50.0, 100_000), ("BBB", 2.0, 10_000_000), ("CCC", 20.0, 1_000))],
                            columns=["date", "ticker", "close", "volume"])
        for day, group in bars.groupby("date"):
            path = self.root / "bars" / str(day.year) / f"{day.date()}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            group.to_parquet(path)
        write_news(self.root / "news", [("2025-03-10 14:00", ["AAA", "BBB", "CCC"]), ("2025-03-10 15:00", ["BBB"])])
        code, output = self.run_script("--start", "2025-03-10", "--end", "2025-03-10", "--bars-dir",
                                       str(self.root / "bars"))
        self.assertEqual((code, len(self.session.bodies)), (0, 1))  # The article of BBB alone is not sent.
        body = self.session.bodies[0]
        self.assertEqual(body["format"]["properties"]["scores"]["items"]["properties"]["company"]["enum"], [A])
        self.assertTrue(body["messages"][1]["content"].endswith(f"Companies: {A}"))
        stored = read_day(self.root / "sentiment/2025/2025-03-10.parquet", "qwen3:8b")
        self.assertEqual(list(stored.ticker), ["AAA"])
        self.assertEqual(self.run_script("--start", "2025-03-10", "--end", "2025-03-10", "--bars-dir",
                                         str(self.root / "bars"))[0], 0)
        self.assertEqual(len(self.session.bodies), 1)  # Nothing left to score: BBB and CCC are not asked again.

    def test_the_tradable_stocks_of_a_news_day_are_those_of_the_week_it_is_read(self):
        days = pd.bdate_range("2025-01-02", "2025-03-14")
        bars = pd.DataFrame([(day, "AAA", 50.0 if day < pd.Timestamp("2025-03-12") else 4.0, 100_000) for day in days]
                            + [(day, "BBB", 6.0 if day >= pd.Timestamp("2025-03-12") else 4.0, 1_000_000)
                               for day in days], columns=["date", "ticker", "close", "volume"])
        found = sentiment.tradable_days(bars, [pd.Timestamp("2025-03-03").date(), pd.Timestamp("2025-03-13").date(),
                                               pd.Timestamp("2025-03-16").date()], 5.0, 1_000_000)
        self.assertEqual(found[pd.Timestamp("2025-03-03").date()], {"AAA"})  # BBB is above $5 only after 8 days.
        self.assertEqual(found[pd.Timestamp("2025-03-13").date()], {"BBB"})
        self.assertEqual(found[pd.Timestamp("2025-03-16").date()], {"BBB"})  # No session yet: the last one.

    def test_a_table_of_names_built_for_an_older_masking_is_refused(self):
        write_news(self.root / "news", [("2025-03-10 14:00", ["AAA"])])
        alias_table(("AAA", "aaa corp", "name")).drop(columns=["wide"]).to_parquet(self.aliases)
        code, output = self.run_script("--start", "2025-03-10", "--end", "2025-03-10")
        self.assertEqual((code, self.session.bodies), (2, []))
        self.assertIn("build_ticker_aliases.py", output)

    def test_a_run_left_alone_waits_for_the_server_and_goes_on(self):
        write_news(self.root / "news", [("2025-03-10 14:00", ["AAA"]), ("2025-03-11 14:00", ["BBB"])])
        self.session.down = True

        def back_up(seconds):
            self.session.down = False  # The server is back while the script waits.
        with mock.patch.object(self.script.time, "sleep", side_effect=back_up) as sleep:
            code, output = self.run_script("--start", "2025-03-10", "--end", "2025-03-11", "--retry-minutes", "10")
        self.assertEqual((code, [call.args[0] for call in sleep.call_args_list]), (0, [600]))
        self.assertIn("Trying again in 10 minutes", output)
        self.assertEqual(len(read_day(self.root / "sentiment/2025/2025-03-11.parquet", "qwen3:8b")), 1)
        self.session.down = True  # Without it, the run stops.
        write_news(self.root / "news", [("2025-03-12 14:00", ["AAA"])])
        self.assertEqual(self.run_script("--start", "2025-03-10", "--end", "2025-03-12")[0], 2)


class PromptTest(unittest.TestCase):
    def test_the_prompt_names_every_event_and_the_schema_allows_only_them(self):
        for event in EVENTS:
            self.assertIn(event, sentiment.SYSTEM_PROMPT)
        for forecast in ("next few trading days", "Ignore anything you know"):  # It labels; it does not predict.
            self.assertNotIn(forecast, sentiment.SYSTEM_PROMPT)
        for told in ("Company A", "[date]", "[month]", "20XX", "label only the listed companies", "single line"):
            self.assertIn(told, sentiment.SYSTEM_PROMPT)  # What stands in the text for what mask() hid.
        item = sentiment.ANSWER_SCHEMA["properties"]["scores"]["items"]["properties"]
        self.assertEqual(item["event"]["enum"], EVENTS)
        self.assertEqual(item["sentiment"]["enum"], [-2, -1, 0, 1, 2])


if __name__ == "__main__":
    unittest.main()
