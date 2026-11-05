"""Run the recommendations API without a console, logging to logs/api.log; the scheduled task that
scripts/install_task.ps1 registers runs this with pythonw.

Exits at once if something already listens on the port, e.g. an API started by hand, so that a single
instance runs the daily refresh.

    pythonw scripts/serve.py [--port 8000]
"""
import argparse
import copy
from datetime import datetime
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs/api.log"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    port = parser.parse_args().port
    LOG.parent.mkdir(exist_ok=True)
    if LOG.exists() and LOG.stat().st_size > 10_000_000:
        LOG.replace(LOG.with_name("api.log.1"))
    # pythonw has no console; uvicorn's logging and any traceback go to sys.stdout and sys.stderr.
    sys.stdout = sys.stderr = open(LOG, "a", encoding="utf-8", errors="replace", buffering=1)
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Port {port} is in use, so an API is already running; exiting.")
            return
    sys.path.insert(0, str(ROOT / "src"))
    import uvicorn
    from uvicorn.config import LOGGING_CONFIG
    log_config = copy.deepcopy(LOGGING_CONFIG)
    for formatter in log_config["formatters"].values():
        formatter["fmt"] = "%(asctime)s " + formatter["fmt"]
    # A single worker, so a single daily refresh.
    uvicorn.run("stock_recommender.api:app", host="127.0.0.1", port=port, log_config=log_config)


if __name__ == "__main__":
    main()
