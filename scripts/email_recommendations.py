"""The 16:00 email task: ask the recommendations API to fetch the day's data and email its lists.

The scheduled task that scripts/install_email_task.ps1 registers runs this every weekday at 16:00 New York time, and
when the computer is available again if it was asleep or off then. It calls the API, which must be running (the
"ML510 stock recommender" task of scripts/install_task.ps1 keeps it up):

    POST http://127.0.0.1:<port>/recommendations/latest?fetch=true&email=<RECOMMENDER_EMAIL_TO>

and appends the outcome to logs/email.log. It exits with 1 when the API cannot be reached, answers an error, or did
not send the email.

    python scripts/email_recommendations.py [--port 8000] [--to you@example.com]
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs/email.log"
WAIT = 180  # Seconds to keep trying an API that is not listening yet, e.g. one starting after logon.
TIMEOUT = 2 * 3600  # The fetch takes about 20 s, but scoring the new news with the LLM can take up to an hour.


def log(message: str) -> None:
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as file:
        file.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}\n")


def default_recipients() -> str:
    """RECOMMENDER_EMAIL_TO, read from the environment or the repository's .env as the API reads its settings."""
    sys.path.insert(0, str(ROOT / "src"))
    from stock_recommender.config import Settings
    return Settings.from_env().email_to


def post(url: str) -> dict:
    """POST to the API and return its JSON answer, waiting up to WAIT seconds for it to listen. A request it
    received is never sent again: that would fetch and email twice."""
    deadline = time.monotonic() + WAIT
    while True:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="POST"), timeout=TIMEOUT) as response:
                return json.load(response)
        except urllib.error.HTTPError:
            raise
        except urllib.error.URLError as error:
            if not isinstance(error.reason, ConnectionRefusedError) or time.monotonic() >= deadline:
                raise
        time.sleep(10)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--to", help="Comma-separated addresses (default: RECOMMENDER_EMAIL_TO).")
    args = parser.parse_args(argv)
    to = args.to if args.to is not None else default_recipients()
    if not to.strip():
        log("Not sent: no address; set RECOMMENDER_EMAIL_TO in .env or pass --to.")
        return 1
    url = (f"http://127.0.0.1:{args.port}/recommendations/latest?"
           + urllib.parse.urlencode({"fetch": "true", "email": to}))
    try:
        body = post(url)
    except urllib.error.HTTPError as error:  # E.g. 503 without data, or 422 for a malformed address.
        log(f"Not sent: the API answered {error.code}: {error.read().decode('utf-8', 'replace')[:2000]}")
        return 1
    except (urllib.error.URLError, OSError) as error:
        log(f"Not sent: no answer from the API at 127.0.0.1:{args.port} ({error}); is the 'ML510 stock "
            "recommender' task running?")
        return 1
    for error in (body.get("fetch") or {}).get("errors") or []:
        log(f"The fetch failed, so the email says the lists may be stale: {error[:500]}")
    email = body.get("email") or {}
    if not email.get("sent_to"):
        log(f"Not sent: {email.get('error')}")
        return 1
    log(f"Emailed the lists as of {body['as_of']} ({len(body['buy'])} buy, {len(body['sell'])} sell) to "
        f"{len(email['sent_to'])} recipient(s).")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # pythonw has no console: the log is the only place a failure shows.
        log("Failed: " + traceback.format_exc().replace("\n", " | "))
        sys.exit(1)
