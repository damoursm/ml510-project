"""Preview the email the API would send now, from the files on disk, without sending it.

It builds the lists exactly as the API does (create_app, with the settings from the environment and .env; --run
serves another backtest run's final selection) and composes them with notify.compose. It never sends, never
downloads, and writes only preview.txt and preview.html.

    python scripts/preview_email.py [--run artifacts/<label>/<stamp>] [--out DIR]
"""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import HTTPException  # noqa: E402
from stock_recommender import notify  # noqa: E402
from stock_recommender.api import create_app  # noqa: E402
from stock_recommender.config import Settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", help="The backtest run to serve, as RECOMMENDER_MODEL_RUN (default: the setting).")
    parser.add_argument("--out", type=Path, help="Folder for preview.txt and preview.html (default: a new temporary "
                                                 "folder).")
    args = parser.parse_args()
    settings = Settings.from_env()
    # No email server: the app only computes the lists.
    settings = replace(settings, smtp_host="", model_run=settings.model_run if args.run is None else args.run)
    try:
        body = create_app(settings).state.refresh.latest.lists(settings.top_n)
    except HTTPException as error:
        print(f"No lists: {error.detail}")
        return 1
    message = notify.compose(body, "preview@localhost", ["preview@localhost"])
    text = f"Subject: {message['Subject']}\n\n{message.get_body(('plain',)).get_content()}"
    out = args.out or Path(tempfile.mkdtemp(prefix="email-preview-"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "preview.txt").write_text(text, encoding="utf-8")
    (out / "preview.html").write_text(message.get_body(("html",)).get_content(), encoding="utf-8")
    print(f"{text}\nWritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
