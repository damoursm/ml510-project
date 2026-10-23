from dataclasses import replace
from email.message import EmailMessage
from pathlib import Path
import smtplib
import tempfile
import unittest
from unittest import mock
from stock_recommender.api import create_app
from stock_recommender.config import Settings
from stock_recommender.models import load_models
from .test_stock_recommender import FixedPriceModel, TestClient, write_bars


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
        self.settings = Settings(bars_dir=root / "bars", news_dir=root / "news", min_dollar_volume=0.0,
                                 refresh_time="off", smtp_host="smtp.example.com", smtp_user="bot@example.com",
                                 smtp_password="secret", email_to="me@example.com, team@example.com")
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
        body = self.post(email=True)
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

    def test_unconfigured_email_is_reported(self):
        body = self.post(replace(self.settings, smtp_host=""), email=True)
        self.assertEqual(body["email"]["sent_to"], [])
        self.assertIn("RECOMMENDER_SMTP_HOST", body["email"]["error"])
        self.assertEqual([row["ticker"] for row in body["buy"]], ["BIG"])  # The lists are still returned.

    def test_password_is_never_sent_unencrypted(self):
        FakeSMTP.tls = False
        with self.assertLogs("uvicorn.error", "ERROR"):
            body = self.post(email=True)
        self.assertIn("refusing to send the password unencrypted", body["email"]["error"])
        self.assertEqual(FakeSMTP.sent, [])

    def test_server_errors_are_reported(self):
        def refuse(host, port, timeout):
            raise smtplib.SMTPConnectError(421, "busy")

        models = replace(load_models(), price=FixedPriceModel({"BIG": 0.02, "DUD": -0.04}))
        client = TestClient(create_app(self.settings, models))
        with mock.patch("stock_recommender.notify.smtplib.SMTP", refuse), self.assertLogs("uvicorn.error", "ERROR"):
            body = client.post("/recommendations/latest", params={"email": True}).json()
        self.assertEqual(body["email"]["sent_to"], [])
        self.assertIn("SMTPConnectError", body["email"]["error"])

    def test_invalid_settings_are_reported_without_the_password(self):
        def login(smtp, user, password):
            password.encode("ascii")  # As smtplib does.

        settings = replace(self.settings, smtp_password="motdepass\u00e9")
        with mock.patch.object(FakeSMTP, "login", login), self.assertLogs("uvicorn.error", "ERROR"):
            body = self.post(settings, email=True)
        self.assertIn("invalid email settings", body["email"]["error"])
        self.assertNotIn("motdepass", body["email"]["error"])
        self.assertEqual([row["ticker"] for row in body["buy"]], ["BIG"])
        with self.assertLogs("uvicorn.error", "ERROR"):
            body = self.post(replace(self.settings, email_to="me@example.com\nBcc: everyone@example.com"), email=True)
        self.assertIn("invalid email settings", body["email"]["error"])
        self.assertEqual(FakeSMTP.sent, [])

    def test_password_is_kept_out_of_the_settings_repr(self):
        self.assertNotIn("secret", repr(self.settings))


if __name__ == "__main__":
    unittest.main()
