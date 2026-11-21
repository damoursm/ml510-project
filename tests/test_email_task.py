"""The 16:00 email task's script (scripts/email_recommendations.py), with the API stubbed out."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.error
import urllib.parse
from stock_recommender.config import ROOT

TO = "me@example.com, team@example.com"
BODY = {"as_of": "2026-11-20", "buy": [{}] * 20, "sell": [{}] * 20, "fetch": {"errors": []},
        "email": {"sent_to": ["me@example.com", "team@example.com"], "error": None}}


def load_script():
    spec = importlib.util.spec_from_file_location("email_recommendations",
                                                  ROOT / "scripts" / "email_recommendations.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Answer(io.BytesIO):
    """What urlopen returns: a context manager holding the API's JSON answer."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class EmailTaskTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.script = load_script()
        self.script.LOG = Path(self.directory.name) / "email.log"
        self.requests = []

    def tearDown(self):
        self.directory.cleanup()

    def run_script(self, answer, *argv) -> tuple[int, str]:
        """Run the script against an API that gives `answer`, a JSON body or an exception; return the exit code and
        the log."""
        def urlopen(request, timeout):
            self.requests.append(request)
            if isinstance(answer, Exception):
                raise answer
            return Answer(json.dumps(answer).encode())

        with (mock.patch.object(self.script.urllib.request, "urlopen", urlopen),
              mock.patch.object(self.script.time, "sleep")):
            code = self.script.main(["--to", TO, *argv])
        return code, self.script.LOG.read_text(encoding="utf-8")

    def test_it_asks_the_api_to_fetch_and_email(self):
        code, log = self.run_script(BODY)
        self.assertEqual(code, 0)
        (request,) = self.requests
        url = urllib.parse.urlsplit(request.full_url)
        self.assertEqual((request.get_method(), url.netloc, url.path),
                         ("POST", "127.0.0.1:8000", "/recommendations/latest"))
        self.assertEqual(urllib.parse.parse_qs(url.query), {"fetch": ["true"], "email": [TO]})
        self.assertIn("Emailed the lists as of 2026-11-20 (20 buy, 20 sell) to 2 recipient(s).", log)
        self.assertNotIn("me@example.com", log)  # The log keeps no address.

    def test_a_failed_fetch_is_logged_and_the_email_still_goes(self):
        code, log = self.run_script(dict(BODY, fetch={"errors": ["download_news.py exited with 1: no key"]}))
        self.assertEqual(code, 0)
        self.assertIn("The fetch failed, so the email says the lists may be stale: download_news.py exited", log)

    def test_an_email_the_api_did_not_send_fails(self):
        code, log = self.run_script(dict(BODY, email={"sent_to": [], "error": "SMTPAuthenticationError: bad"}))
        self.assertEqual(code, 1)
        self.assertIn("Not sent: SMTPAuthenticationError: bad", log)

    def test_an_api_error_fails_without_asking_again(self):
        error = urllib.error.HTTPError("http://127.0.0.1:8000/recommendations/latest", 503, "Service Unavailable", {},
                                       io.BytesIO(b'{"detail": "No daily bars"}'))
        code, log = self.run_script(error)
        self.assertEqual((code, len(self.requests)), (1, 1))
        self.assertIn("Not sent: the API answered 503", log)
        self.assertIn("No daily bars", log)

    def test_an_api_not_listening_is_tried_again_then_fails(self):
        refused = urllib.error.URLError(ConnectionRefusedError(10061, "refused"))
        with mock.patch.object(self.script.time, "monotonic", side_effect=[0, 10, 200]):  # WAIT is 180 s.
            code, log = self.run_script(refused)
        self.assertEqual((code, len(self.requests)), (1, 2))
        self.assertIn("is the 'ML510 stock recommender' task running?", log)

    def test_a_request_that_times_out_is_not_sent_again(self):  # It would fetch and email twice.
        code, log = self.run_script(TimeoutError("timed out"))
        self.assertEqual((code, len(self.requests)), (1, 1))
        self.assertIn("Not sent: no answer from the API", log)

    def test_no_address_fails(self):
        self.assertEqual(self.script.main(["--to", " "]), 1)
        self.assertIn("no address", self.script.LOG.read_text(encoding="utf-8"))

    def test_the_address_defaults_to_the_setting(self):
        with mock.patch.dict("os.environ", {"RECOMMENDER_EMAIL_TO": "you@example.com"}):
            self.assertEqual(self.script.default_recipients(), "you@example.com")


if __name__ == "__main__":
    unittest.main()
