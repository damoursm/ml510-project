from dataclasses import replace
from email.message import EmailMessage
from pathlib import Path
import smtplib
import subprocess
import tempfile
import unittest
from unittest import mock
from stock_recommender.api import create_app
from stock_recommender.config import Settings
from stock_recommender.models import load_models
from .test_served_models import make_run
from .test_stock_recommender import FixedPriceModel, TestClient, write_bars

TO = "me@example.com, team@example.com"  # What the 16:00 email task passes as email=.


class FakeSMTP:
    """Records what the email code does with an SMTP server that offers STARTTLS unless told otherwise."""
    sent: list[EmailMessage] = []
    calls: list[str] = []
    tls = True

    def __init__(self, host, port, timeout):
        FakeSMTP.calls.append(f"connect {host}:{port}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def ehlo(self):
        pass

    def has_extn(self, name):
        return self.tls and name == "starttls"

    def starttls(self, context):
        FakeSMTP.calls.append("starttls")

    def login(self, user, password):
        FakeSMTP.calls.append(f"login {user}")

    def send_message(self, message):
        FakeSMTP.sent.append(message)


class EmailTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.settings = Settings(bars_dir=root / "bars", news_dir=root / "news", sentiment_dir=root / "sentiment",
                                 min_dollar_volume=0.0, smtp_host="smtp.example.com", smtp_user="bot@example.com",
                                 smtp_password="secret", score_news=False)
        write_bars(self.settings.bars_dir, "2026-10-16", {"BIG": (100, 1_000), "DUD": (30, 500)})
        FakeSMTP.sent, FakeSMTP.calls, FakeSMTP.tls = [], [], True

    def tearDown(self):
        self.directory.cleanup()

    def post(self, settings=None, **params):
        models = replace(load_models(), price=FixedPriceModel({"BIG": 0.02, "DUD": -0.04}))
        client = TestClient(create_app(settings or self.settings, models))
        with mock.patch("stock_recommender.notify.smtplib.SMTP", FakeSMTP):
            return client.post("/recommendations/latest", params={"limit": 1, **params}).json()

    def test_email_lists_the_buys_and_sells(self):
        body = self.post(email=TO)
        self.assertEqual(body["email"], {"sent_to": ["me@example.com", "team@example.com"], "error": None})
        self.assertEqual(FakeSMTP.calls, ["connect smtp.example.com:587", "starttls", "login bot@example.com"])
        (message,) = FakeSMTP.sent
        self.assertEqual(message["Subject"], "Stock recommendations as of 2026-10-16: 1 buy, 1 sell")
        self.assertEqual((message["From"], message["To"]), ("bot@example.com", "me@example.com, team@example.com"))
        self.assertTrue(message["Date"] and message["Message-ID"])
        text = message.get_body("plain").get_content()
        self.assertLess(text.index("BUY"), text.index("BIG"))
        self.assertLess(text.index("SELL"), text.index("DUD"))
        self.assertIn("-4.00%", text)
        self.assertIn("<td align=\"right\">DUD</td>", message.get_body("html").get_content())

    def test_no_email_unless_asked(self):
        self.assertIsNone(self.post()["email"])
        self.assertEqual(FakeSMTP.sent, [])

    def test_email_takes_the_recipients_addresses(self):
        downloads = []

        def run(command, **kwargs):
            downloads.append(command)
            return subprocess.CompletedProcess(command, 0, "done", "")

        for email in ("true", " , ", "me@example.com\nBcc: everyone@example.com", "Me <me@example.com>"):
            with mock.patch("stock_recommender.refresh.subprocess.run", run):
                body = self.post(fetch=True, email=email)
            self.assertIn("email takes the addresses", body["detail"], email)
        self.assertEqual((downloads, FakeSMTP.sent), ([], []))  # Refused before fetching anything.

    def test_unconfigured_email_is_reported(self):
        body = self.post(replace(self.settings, smtp_host=""), email=TO)
        self.assertEqual(body["email"]["sent_to"], [])
        self.assertIn("RECOMMENDER_SMTP_HOST", body["email"]["error"])
        self.assertEqual([row["ticker"] for row in body["buy"]], ["BIG"])  # The lists are still returned.

    def test_password_is_never_sent_unencrypted(self):
        FakeSMTP.tls = False
        with self.assertLogs("uvicorn.error", "ERROR"):
            body = self.post(email=TO)
        self.assertIn("refusing to send the password unencrypted", body["email"]["error"])
        self.assertEqual(FakeSMTP.sent, [])

    def test_server_errors_are_reported(self):
        def refuse(host, port, timeout):
            raise smtplib.SMTPConnectError(421, "busy")

        models = replace(load_models(), price=FixedPriceModel({"BIG": 0.02, "DUD": -0.04}))
        client = TestClient(create_app(self.settings, models))
        with mock.patch("stock_recommender.notify.smtplib.SMTP", refuse), self.assertLogs("uvicorn.error", "ERROR"):
            body = client.post("/recommendations/latest", params={"email": TO}).json()
        self.assertEqual(body["email"]["sent_to"], [])
        self.assertIn("SMTPConnectError", body["email"]["error"])

    def test_invalid_settings_are_reported_without_the_password(self):
        def login(smtp, user, password):
            password.encode("ascii")  # As smtplib does.

        settings = replace(self.settings, smtp_password="motdepassé")
        with mock.patch.object(FakeSMTP, "login", login), self.assertLogs("uvicorn.error", "ERROR"):
            body = self.post(settings, email=TO)
        self.assertIn("invalid email settings", body["email"]["error"])
        self.assertNotIn("motdepass", body["email"]["error"])
        self.assertEqual([row["ticker"] for row in body["buy"]], ["BIG"])
        self.assertEqual(FakeSMTP.sent, [])

    def test_the_email_warns_of_a_failed_fetch(self):
        def run(command, **kwargs):  # Each download fails, as without a Polygon key.
            return subprocess.CompletedProcess(command, 1, "", "no key")

        with mock.patch("stock_recommender.refresh.subprocess.run", run), self.assertLogs("uvicorn.error", "ERROR"):
            body = self.post(fetch=True, email=TO)
        self.assertEqual(len(body["fetch"]["errors"]), 2)
        self.assertEqual(body["email"]["sent_to"], ["me@example.com", "team@example.com"])
        self.assertIn("Warning: fetching new data failed", FakeSMTP.sent[0].get_body(("plain",)).get_content())

    def test_password_is_kept_out_of_the_settings_repr(self):
        self.assertNotIn("secret", repr(self.settings))


class RunEmailTest(unittest.TestCase):
    """With a backtest run set, the email still sends the lists the API returns, now its selection's."""
    setUp, tearDown = EmailTest.setUp, EmailTest.tearDown

    def app(self, **models):
        folder = make_run(Path(self.directory.name), models)
        return create_app(replace(self.settings, model_run=str(folder), top_n=1))

    @staticmethod
    def fixed(returns: dict, version: str):
        return replace(load_models(), price=FixedPriceModel(returns), version=version)

    def test_the_email_sends_the_runs_lists(self):
        app = self.app(fixed=self.fixed({"BIG": 0.02, "DUD": -0.04}, "fixed"))

        def run(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, "", "done")

        with (mock.patch("stock_recommender.refresh.subprocess.run", run),
              mock.patch("stock_recommender.notify.smtplib.SMTP", FakeSMTP)):
            with TestClient(app) as client:  # As the 16:00 email task calls it.
                sent = client.post("/recommendations/latest", params={"fetch": True, "email": TO}).json()
                body = client.get("/recommendations/latest", params={"limit": 1}).json()
        (message,) = FakeSMTP.sent
        text = message.get_body(("plain",)).get_content()
        self.assertEqual((sent["buy"], sent["sell"]), (body["buy"], body["sell"]))
        self.assertEqual(message["Subject"], "Stock recommendations as of 2026-10-16: 1 buy, 1 sell")
        self.assertIn(f"Model fixed from backtest run {body['run']}, 5-day horizon.", text)
        self.assertEqual(([item["ticker"] for item in body["buy"]], [item["ticker"] for item in body["sell"]]),
                         (["BIG"], ["DUD"]))
        bought, sold = text.split("\nSELL\n")
        self.assertIn("BIG", bought.split("\nBUY\n")[1])
        self.assertIn("DUD", sold)
        self.assertNotIn("<th>Model</th>", message.get_body(("html",)).get_content())  # One model: the usual columns.

    def test_several_models_add_a_model_column(self):
        app = self.app(fixed=self.fixed({"BIG": 0.02, "DUD": -0.04}, "fixed"),
                       flipped=self.fixed({"BIG": -0.01, "DUD": 0.01}, "flipped"))
        with mock.patch("stock_recommender.notify.smtplib.SMTP", FakeSMTP):
            body = TestClient(app).post("/recommendations/latest", params={"limit": 1, "email": TO}).json()
        self.assertEqual(body["email"]["sent_to"], ["me@example.com", "team@example.com"])
        text = FakeSMTP.sent[0].get_body(("plain",)).get_content()
        rows = [line.split()[:3] for line in text.splitlines()]
        self.assertIn(["#", "Model", "Ticker"], rows)
        self.assertIn(["1", "fixed", "BIG"], rows)
        self.assertIn(["1", "flipped", "DUD"], rows)
        self.assertIn("<th>Model</th>", FakeSMTP.sent[0].get_body(("html",)).get_content())


if __name__ == "__main__":
    unittest.main()
