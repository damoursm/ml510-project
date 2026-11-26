"""The dashboard: a page that explains the API's recommendations, written again at every call: the lists, what drives
each model's ranking, and the calls. Hovering a listed stock shows its SHAP and LIME values (explain.py), from data the
page carries as JSON.

Every GET or POST of /recommendations/latest, by hand or by the 16:00 email task, is recorded (Dashboard.record): the
lists it returned, how the fetch and the email went (counts, never the addresses), and the explanation of the lists
(explain.py): what each model's ranking follows, and where the listed stocks stand on those features. A call that
fails is recorded with its error, and the page keeps the last lists.

The files, in settings.dashboard_dir: index.html, the page (served at /dashboard, and readable without the API);
latest.json, the last call that returned lists, with their explanation; calls.jsonl, one line per call.
"""
from datetime import datetime, timezone
from html import escape
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
import pandas as pd
from .config import Settings
from .explain import DESCRIPTIONS, OUTPUTS, explain, local_explanations
from .schemas import RecommendationList

log = logging.getLogger("uvicorn.error")
HISTORY = 30  # Calls the page lists.
PROFILE = 5  # A model's strongest drivers shown beside each listed stock.
SHOWN = 6  # Hovering a stock shows its strongest features by SHAP and by LIME, this many of each.
RANKED_BY = ("score", "dollar_volume")  # The outputs the drivers section correlates with the ranking.
CSS = """
:root { --bg: #ffffff; --fg: #1f2328; --muted: #59636e; --line: #d1d9e0; --card: #f6f8fa; --pos: #1a7f37;
        --neg: #cf222e; --accent: #0969da; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --line: #3d444d; --card: #151b23; --pos: #3fb950;
          --neg: #f85149; --accent: #4493f8; } }
body { margin: 0; background: var(--bg); color: var(--fg);
       font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 32px 0 8px; padding-bottom: 4px; border-bottom: 1px solid var(--line); }
h3 { font-size: 14px; margin: 18px 0 6px; }
.meta, .note, .pct, .muted { color: var(--muted); }
.note { font-size: 13px; max-width: 900px; }
.pct { font-size: 11px; margin-left: 4px; }
.banner { margin: 12px 0; padding: 8px 12px; border-radius: 6px; background: var(--card);
          border-left: 4px solid var(--accent); }
.banner.bad { border-left-color: var(--neg); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; margin: 6px 0 14px; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 7px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
th:first-child, td:first-child, .text { text-align: left; }
th { color: var(--muted); font-weight: 600; vertical-align: bottom; }
.pos { color: var(--pos); } .neg { color: var(--neg); }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 4px 28px; }
svg text { fill: var(--fg); font-size: 11px; } svg .muted { fill: var(--muted); }
svg .axis { stroke: var(--line); } svg .up { fill: var(--pos); } svg .down { fill: var(--neg); }
tr[data-ticker] { cursor: help; }
tr[data-ticker]:hover, tr[data-ticker]:focus { background: var(--card); outline: none; }
.tip { position: fixed; z-index: 10; max-width: 660px; padding: 10px 12px; border: 1px solid var(--line);
       border-radius: 8px; background: var(--bg); box-shadow: 0 8px 28px rgba(0, 0, 0, 0.22); font-size: 12px; }
.tip table { margin: 6px 0 4px; }
.tip th, .tip td { padding: 2px 6px; }
.bar { position: relative; display: inline-block; width: 96px; height: 10px; vertical-align: middle;
       background: linear-gradient(var(--line), var(--line)) center / 1px 100% no-repeat; }
.bar i { position: absolute; top: 1px; bottom: 1px; border-radius: 2px; }
.bar i.up { background: var(--pos); } .bar i.down { background: var(--neg); }
"""
SCRIPT = """<script>
(function () {
  var tip = document.getElementById("tip");
  setTimeout(function () { location.reload(); }, 60000);

  // Hovering a stock, or focusing it with Tab, shows its SHAP and LIME values.
  var payload = document.getElementById("local-explanations");
  var explained = payload ? JSON.parse(payload.textContent) : {};
  var entities = {"&": "&amp;", "<": "&lt;", ">": "&gt;", "\\"": "&quot;"};
  function text(value) { return String(value).replace(/[&<>"]/g, function (c) { return entities[c]; }); }
  function points(value) { return (value >= 0 ? "+" : "") + (100 * value).toFixed(1); }
  function bar(value, largest) {
    var width = largest > 0 ? 48 * Math.abs(value) / largest : 0;
    return "<span class=\\"bar\\"><i class=\\"" + (value >= 0 ? "up" : "down") + "\\" style=\\"" +
      (value >= 0 ? "left" : "right") + ":50%;width:" + width.toFixed(1) + "%\\"></i></span>";
  }
  function describe(row) {
    var model = explained[row.getAttribute("data-model")];
    var stock = model && model.stocks[row.getAttribute("data-ticker")];
    if (!stock) { return ""; }
    var largest = Math.abs(stock.rest);
    stock.features.forEach(function (f) { largest = Math.max(largest, Math.abs(f[1]), Math.abs(f[2])); });
    var lines = stock.features.map(function (f) {
      return "<tr><td class=\\"text\\">" + text(f[0]) + "</td><td class=\\"text\\">" + text(f[3]) + "</td><td>" +
        bar(f[1], largest) + "</td><td>" + points(f[1]) + "</td><td>" + bar(f[2], largest) + "</td><td>" +
        points(f[2]) + "</td></tr>";
    });
    lines.push("<tr><td class=\\"text muted\\">Other features</td><td></td><td>" + bar(stock.rest, largest) +
      "</td><td>" + points(stock.rest) + "</td><td></td><td></td></tr>");
    var fidelity = model.fidelity === null ? "n/a" : model.fidelity.toFixed(2);
    return "<b>" + text(row.getAttribute("data-ticker")) + "</b> <span class=\\"muted\\">" +
      text(row.getAttribute("data-model")) + "</span><div class=\\"muted\\">Rank percentile " +
      (100 * stock.actual).toFixed(1) + " · surrogate " + (100 * stock.predicted).toFixed(1) +
      " · average stock " + (100 * stock.base).toFixed(1) + "</div><table><tr><th class=\\"text\\">Feature</th>" +
      "<th class=\\"text\\">Value</th><th colspan=\\"2\\">SHAP</th><th colspan=\\"2\\">LIME</th></tr>" +
      lines.join("") + "</table><div class=\\"muted\\">In points of rank percentile (100 the best stock). The SHAP " +
      "values add up to the surrogate's " + (100 * stock.predicted).toFixed(1) + " from the average stock's " +
      (100 * stock.base).toFixed(1) + ". Surrogate R² " + fidelity + " on held-out stocks; LIME's local fit R² " +
      stock.lime_fit.toFixed(2) + ".</div>";
  }
  function place(row) {
    var box = row.getBoundingClientRect();
    tip.hidden = false;
    var top = box.bottom + 6;
    if (top + tip.offsetHeight > window.innerHeight - 6) { top = Math.max(6, box.top - tip.offsetHeight - 6); }
    tip.style.top = top + "px";
    tip.style.left = Math.max(6, Math.min(box.left + 24, window.innerWidth - tip.offsetWidth - 6)) + "px";
  }
  document.querySelectorAll("tr[data-ticker]").forEach(function (row) {
    function open() { var html = describe(row); if (html) { tip.innerHTML = html; place(row); } }
    function close() { tip.hidden = true; }
    row.addEventListener("mouseenter", open);
    row.addEventListener("focus", open);
    row.addEventListener("mouseleave", close);
    row.addEventListener("blur", close);
  });
  window.addEventListener("scroll", function () { tip.hidden = true; }, {passive: true});
})();
</script>"""
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<noscript><meta http-equiv="refresh" content="60"></noscript>
<title>ML510 recommendations</title>
<style>{css}</style>
</head>
<body><main>
{body}
</main>
{script}
</body>
</html>
"""


class Dashboard:
    """Records the calls of the recommendations in settings.dashboard_dir and writes the page again after each."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.directory = Path(settings.dashboard_dir)
        self._lock = threading.Lock()
        self._explained = None  # (generated_at, explanation, error) of the last result explained.
        self._local: dict = {}  # That result's SHAP and LIME values shown on hover, by (model, ticker).

    def record(self, method: str, limit: int, body: RecommendationList | None = None, served: list | None = None,
               error: str | None = None, fetch: bool = False, email: bool = False) -> None:
        """Record one call and write the page again. body is what the call returned, and served the result behind
        it ((name, version, Recommendations, models) per model, from the API); a failed call gives its error."""
        with self._lock:
            call = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "method": method,
                    "limit": limit, "error": error, "fetch": _fetch(body, fetch), "email": _email(body, email)}
            snapshot = self._snapshot(call, body, served) if body is not None else None
            self.directory.mkdir(parents=True, exist_ok=True)
            with open(self.directory / "calls.jsonl", "a", encoding="utf-8") as calls:
                calls.write(json.dumps({**call, **(_summary(snapshot["lists"]) if snapshot else {})}) + "\n")
            if snapshot is not None:
                _write(self.directory / "latest.json", json.dumps(snapshot, indent=1))
            else:
                snapshot = self._last()
            _write(self.directory / "index.html", render(snapshot, self._history(), call if error else None))

    def page(self) -> str:
        """The page as the last call wrote it, or one saying that no call came yet."""
        try:
            return (self.directory / "index.html").read_text(encoding="utf-8")
        except FileNotFoundError:
            return render(None, [])

    def _snapshot(self, call: dict, body: RecommendationList, served: list | None) -> dict:
        lists = _lists(body)
        explained, problem = self._explanation(body, served)
        return {"call": call, "lists": lists, "explanation": _explanation(explained, problem, lists, self._local)}

    def _explanation(self, body: RecommendationList, served: list | None):
        """The explanation of the result behind body, computed once per result: (explanation, error)."""
        if self._explained is not None and self._explained[0] == body.generated_at:
            return self._explained[1:]
        if not served:  # Another call computed newer lists meanwhile; the next call explains those.
            return None, "the lists were computed again while this call was recorded"
        try:
            explained, problem = explain(self.settings, served), None
        except Exception as error:  # The lists were served; the page says why they are not explained.
            log.exception("Dashboard: could not explain the lists")
            explained, problem = None, f"{type(error).__name__}: {error}"
        self._explained, self._local = (body.generated_at, explained, problem), {}
        return explained, problem

    def _last(self) -> dict | None:
        try:
            return json.loads((self.directory / "latest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _history(self) -> list[dict]:
        """The last HISTORY calls, newest first."""
        try:
            lines = (self.directory / "calls.jsonl").read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        calls = []
        for line in reversed(lines):
            try:
                calls.append(json.loads(line))
            except ValueError:
                continue
            if len(calls) == HISTORY:
                break
        return calls


def _write(path: Path, text: str, attempts: int = 10) -> None:
    """Replace a file at once, so a reader never sees half of it; retry while one holds it open on Windows."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    for attempt in range(attempts):
        try:
            return os.replace(temporary, path)
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.2)


def _number(value) -> float | None:
    """A finite float, or None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fetch(body: RecommendationList | None, asked: bool) -> dict | None:
    """How the call's fetch went: None when it asked for none; errors None when its outcome is unknown."""
    if body is not None and body.fetch is not None:
        return {"errors": len(body.fetch.errors)}
    return {"errors": None} if asked else None


def _email(body: RecommendationList | None, asked: bool) -> dict | None:
    """How the call's email went, as a count and the error's kind: never the addresses, which an error can quote."""
    if body is not None and body.email is not None:
        error = body.email.error.split(":", 1)[0] if body.email.error else None
        return {"sent": len(body.email.sent_to), "error": error}
    return {"sent": None, "error": None} if asked else None


def _lists(body: RecommendationList) -> dict:
    """The call's lists, by model."""
    models: dict[str, dict] = {}
    for side in ("buy", "sell"):
        for item in getattr(body, side):
            name = item.model or body.model_version
            models.setdefault(name, {"buy": [], "sell": []})[side].append(item.model_dump(exclude={"model"}))
    return {"as_of": body.as_of.isoformat(), "news_cutoff": body.news_cutoff.isoformat(),
            "news_through": body.news_through.isoformat() if body.news_through else None,
            "generated_at": body.generated_at.isoformat(), "horizon_days": body.horizon_days,
            "exits": {name: strategy.summary for name, strategy in body.exits.items()},
            "model_version": body.model_version, "universe_size": body.universe_size, "run": body.run,
            "models": models}


def _explanation(explained, problem: str | None, lists: dict, cache: dict | None = None) -> dict:
    """The explanation as JSON: each model's drivers; for the stocks it listed, its strongest PROFILE drivers' values
    and percentiles; and their SHAP and LIME values from the result's surrogate, kept in cache by (model, ticker)
    so that a stock is explained once per result."""
    cache = {} if cache is None else cache
    if explained is None:
        return {"error": problem}
    models = []
    for model in explained.models:
        columns = [row["feature"] for row in model.drivers if row["correlation"] is not None][:PROFILE]
        listed = lists["models"].get(model.name, {"buy": [], "sell": []})
        tickers = [item["ticker"] for item in listed["buy"] + listed["sell"]]
        profile = {}
        for ticker in tickers:
            known = ticker in explained.features.index
            profile[ticker] = {name: [_number(explained.features.at[ticker, name]) if known else None,
                                      _number(explained.percentiles.at[ticker, name]) if known else None]
                               for name in columns}
        local_error = None
        if model.surrogate is not None:
            try:
                missing = [ticker for ticker in tickers if (model.name, ticker) not in cache]
                for ticker, found in local_explanations(model.surrogate, missing).items():
                    cache[(model.name, ticker)] = _shown(found)
            except Exception as error:  # The lists stay explained by their drivers.
                log.exception("Dashboard: could not compute the SHAP and LIME values of %s", model.name)
                local_error = f"{type(error).__name__}: {error}"
        local = {ticker: cache[(model.name, ticker)] for ticker in tickers if (model.name, ticker) in cache}
        models.append({"name": model.name, "version": model.version, "scored": model.scored,
                       "drivers": model.drivers, "outputs": model.outputs, "importances": model.importances,
                       "columns": columns, "profile": profile, "local": local, "local_error": local_error,
                       "fidelity": model.surrogate.fidelity if model.surrogate is not None else None})
    return {"error": None, "as_of": explained.as_of.date().isoformat(), "models": models}


def _shown(found: dict) -> dict:
    """What hovering a stock shows: of its SHOWN strongest features by SHAP and by LIME, those worth at least 5% of
    the strongest value (three at least), by SHAP first, with their values; and rest, the SHAP values of the other
    features, so that the shown ones and rest add up."""
    shap, lime = found["shap"], found["lime"]
    strongest = (sorted(shap, key=lambda name: -abs(shap[name]))[:SHOWN]
                 + sorted(lime, key=lambda name: -abs(lime[name]))[:SHOWN])
    names = sorted(set(strongest), key=lambda name: -abs(shap[name]))
    largest = max(max(abs(shap[name]), abs(lime[name])) for name in names)
    kept = [name for name in names if max(abs(shap[name]), abs(lime[name])) >= 0.05 * largest] if largest else []
    names = kept if len(kept) >= 3 else names[:3]
    rest = found["predicted"] - found["base"] - sum(shap[name] for name in names)
    return {"actual": found["actual"], "predicted": found["predicted"], "base": found["base"], "rest": rest,
            "lime_fit": found["lime_fit"],
            "features": [[name, shap[name], lime[name], _feature_text(name, found["values"].get(name))]
                         for name in names]}


def _summary(lists: dict) -> dict:
    """What calls.jsonl keeps of a call's lists: the date, and each model's first five buys and sells."""
    return {"as_of": lists["as_of"], "generated_at": lists["generated_at"], "run": lists["run"],
            "buy": {name: [item["ticker"] for item in sides["buy"][:5]] for name, sides in lists["models"].items()},
            "sell": {name: [item["ticker"] for item in sides["sell"][:5]] for name, sides in lists["models"].items()}}


# ---- The page ----

def _et(stamp: str | None, form: str = "%Y-%m-%d %H:%M") -> str:
    """A UTC time as New York time."""
    if not stamp:
        return "—"
    return pd.Timestamp(stamp).tz_convert("America/New_York").strftime(form) + " ET"


def _format(value, form: str) -> str:
    """A number for the page: signed and share as percents, sentiment and ratio as decimals, money in dollars."""
    value = _number(value)
    if value is None:
        return '<span class="muted">—</span>'
    if form == "signed":
        text, sign = f"{value:+.1%}", value
    elif form == "share":
        text, sign = f"{value:.0%}", 0
    elif form == "money":
        text, sign = _money(value), 0
    elif form == "sentiment":
        text, sign = f"{value:+.2f}", value
    else:
        text, sign = f"{value:,.2f}", 0
    color = "pos" if sign > 0 else "neg" if sign < 0 else ""
    return f'<span class="{color}">{escape(text)}</span>' if color else escape(text)


def _money(value: float) -> str:
    return (f"${value / 1e9:,.2f}B" if abs(value) >= 1e9 else f"${value / 1e6:,.1f}M" if abs(value) >= 1e6
            else f"${value:,.0f}")


FEATURE_FORMS = {"ret_1d": "signed", "ret_5d": "signed", "ret_20d": "signed", "momentum_60d": "signed",
                 "gap_1d": "signed", "high_60d": "signed", "volatility_20d": "share", "close_location": "ratio",
                 "volume_ratio": "ratio", "dollar_volume_20d": "log-money", "news_count_7d": "count",
                 "news_count_1d": "count", "impact_7d": "ratio", "earnings_7d": "flag", "dollar_volume": "money"}


def _feature_text(name: str, value) -> str:
    """A feature's value as plain text, as hovering a stock shows it."""
    value = _number(value)
    form = FEATURE_FORMS.get(name, "sentiment")
    if value is None:
        return "n/a"
    if form == "log-money":
        value, form = math.exp(value), "money"
    texts = {"money": _money, "signed": "{:+.1%}".format, "share": "{:.1%}".format, "count": "{:,.0f}".format,
             "ratio": "{:,.2f}".format, "sentiment": "{:+.2f}".format}
    return ("yes" if value else "no") if form == "flag" else texts[form](value)


def _feature(name: str, value, percentile) -> str:
    """A feature's value as its kind reads best, with its percentile among the scored stocks."""
    value = _number(value)
    form = FEATURE_FORMS.get(name, "sentiment")
    if value is None:
        shown = '<span class="muted">—</span>'
    elif form == "log-money":
        shown = _format(math.exp(value), "money")
    elif form == "count":
        shown = f"{value:,.0f}"
    elif form == "flag":
        shown = "yes" if value else "no"
    else:
        shown = _format(value, form)
    percentile = _number(percentile)
    rank = f'<span class="pct">p{round(percentile * 100)}</span>' if percentile is not None else ""
    return shown + rank


def driver_chart(drivers: list[dict]) -> str:
    """A bar per feature, from -1 to +1: its correlation with the ranking."""
    label, plot, row, top = 150, 300, 18, 16
    width, height = label + plot + 56, top + row * len(drivers) + 4
    middle = label + plot / 2
    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
             f'aria-label="Correlation of each feature with the ranking">',
             f'<text class="muted" x="{label}" y="10">-1</text>',
             f'<text class="muted" x="{middle}" y="10" text-anchor="middle">0</text>',
             f'<text class="muted" x="{label + plot}" y="10" text-anchor="end">+1</text>',
             f'<line class="axis" x1="{middle}" y1="{top - 2}" x2="{middle}" y2="{height}"/>']
    for index, driver in enumerate(drivers):
        y = top + index * row
        name = driver["feature"]
        parts.append(f'<text x="{label - 8}" y="{y + 12}" text-anchor="end"><title>'
                     f'{escape(DESCRIPTIONS.get(name, name))}</title>{escape(name)}</text>')
        value = driver["correlation"]
        if value is None:
            parts.append(f'<text class="muted" x="{label + plot + 8}" y="{y + 12}">n/a</text>')
            continue
        length = abs(value) * plot / 2
        start = middle if value >= 0 else middle - length
        parts.append(f'<rect class="{"up" if value >= 0 else "down"}" x="{start:.1f}" y="{y + 3}" '
                     f'width="{max(length, 0.5):.1f}" height="{row - 6}" rx="2"/>')
        parts.append(f'<text x="{label + plot + 8}" y="{y + 12}">{value:+.2f}</text>')
    return "".join(parts) + "</svg>"


def _call_text(call: dict) -> tuple[str, str, str]:
    """A call's result, fetch and email, in words."""
    result = "ok" if not call.get("error") else f"failed: {call['error']}"
    fetch = call.get("fetch")
    fetched = ("—" if fetch is None else "asked" if fetch.get("errors") is None
               else "ok" if fetch["errors"] == 0 else f"{fetch['errors']} download(s) failed")
    email = call.get("email")
    if email is None:
        emailed = "—"
    elif email.get("sent") is None:
        emailed = "asked"
    else:
        emailed = f"sent to {email['sent']}" + (f" ({email['error']})" if email.get("error") else "")
    return result, fetched, emailed


def _correlation(value: float | None) -> str:
    """A correlation with the ranking, or why there is none."""
    return '<span class="muted">same for every stock</span>' if value is None else f"{value:+.2f}"


def _summary_line(model: dict) -> str:
    """One sentence on what a model's ranking follows."""
    outputs = {row["feature"]: row["correlation"] for row in model["outputs"]}
    drivers = [row for row in model["drivers"] if row["correlation"] is not None]
    if outputs.get("score") is None and outputs.get("dollar_volume") is not None:
        lead = ("Every stock gets the same score, so the stocks rank by dollar volume, the tie-break: the model "
                "recommends the most traded stocks.")
    else:
        lead = "The ranking follows the model's score."
    if drivers:
        top = ", ".join(f"{escape(row['feature'])} ({row['correlation']:+.2f})" for row in drivers[:3])
        lead += f" Strongest features: {top}."
    return lead


def _lists_section(snapshot: dict) -> str:
    lists, explanation = snapshot["lists"], snapshot["explanation"]
    explained = {model["name"]: model for model in explanation.get("models") or []}
    exits = lists.get("exits") or {}
    selling = (f"hold {lists['horizon_days']} trading days" if not exits else next(iter(exits.values()))
               if len(exits) == 1 else "; ".join(f"{name}: {summary}" for name, summary in exits.items()))
    parts = ['<section id="lists"><h2>Recommendations</h2>',
             f'<p class="note">To buy at today\'s close (16:00); then {escape(selling)}. Beside each stock: '
             f'its value of the features its model\'s ranking follows most, with its percentile among the '
             f'{lists["universe_size"]:,} tradable stocks (p99: higher than 99% of them). Hover over a stock, or Tab '
             f'to it, for its SHAP and LIME values: how much each feature moves it up or down the ranking.</p>']
    if explanation.get("error"):
        parts.append(f'<p class="banner bad">The lists are not explained: {escape(explanation["error"])}.</p>')
    for name, sides in lists["models"].items():
        model = explained.get(name)
        columns = model["columns"] if model else []
        correlation = {row["feature"]: row["correlation"] for row in model["drivers"]} if model else {}
        parts.append(f"<h3>{escape(name)}</h3>")
        for side, title in (("buy", "Buy"), ("sell", "Sell")):
            head = "".join(f'<th title="{escape(DESCRIPTIONS.get(column, column))}">{escape(column)}<br>'
                           f'<span class="pct">{correlation[column]:+.2f}</span></th>' for column in columns)
            rows = []
            for item in sides[side]:
                profile = model["profile"].get(item["ticker"], {}) if model else {}
                features = "".join(f"<td>{_feature(column, *profile.get(column, [None, None]))}</td>"
                                   for column in columns)
                hover = (f' data-model="{escape(name)}" data-ticker="{escape(item["ticker"])}" tabindex="0"'
                         if model and item["ticker"] in (model.get("local") or {}) else "")
                rows.append(f'<tr{hover}><td>{item["rank"]}</td><td class="text"><b>{escape(item["ticker"])}</b></td>'
                            f'<td>{item["news_count"]}</td>'
                            f'<td>{_format(item["close"], "ratio")}</td>'
                            f'<td>{_format(item["dollar_volume"], "money")}</td>{features}</tr>')
            empty = f'<tr><td colspan="{5 + len(columns)}" class="muted">None</td></tr>'
            parts.append(f'<div class="scroll"><b>{title}</b><table><tr><th>#</th><th class="text">Ticker</th>'
                         f'<th>News</th><th>Close</th>'
                         f'<th>$ volume/day</th>{head}</tr>{"".join(rows) or empty}</table></div>')
    return "".join(parts) + "</section>"


def _drivers_section(snapshot: dict) -> str:
    explanation = snapshot["explanation"]
    parts = ['<section id="drivers"><h2>What drives each model\'s ranking</h2>']
    if explanation.get("error"):
        return "".join(parts) + f'<p class="banner bad">Not explained: {escape(explanation["error"])}.</p></section>'
    parts.append(f'<p class="note">For each feature, the Spearman correlation between its value and the model\'s '
                 f'ranking, across the stocks the model scored on {escape(explanation["as_of"])}: +1 when the ranking '
                 f'follows the feature exactly (a higher value ranks first), -1 when it goes the other way, n/a when '
                 f'the value is missing for most stocks or the same for all. The news features that read LLM scores '
                 f'stay n/a until the day\'s articles are scored. Hover a feature for its meaning.</p>')
    for model in explanation["models"]:
        # The score and its tie-break; the lists no longer show expected_return or news_sentiment.
        outputs = "".join(f'<tr><td class="text" title="{escape(OUTPUTS.get(row["feature"], ""))}">'
                          f'{escape(row["feature"])}</td><td>{_correlation(row["correlation"])}</td></tr>'
                          for row in model["outputs"] if row["feature"] in RANKED_BY)
        if model["importances"]:
            ranked = sorted(model["importances"].items(), key=lambda pair: -abs(pair[1]))
            measured = ("<b>Importances the model measures</b><table>"
                        + "".join(f'<tr><td class="text">{escape(name)}</td><td>{value:.3f}</td></tr>'
                                  for name, value in ranked) + "</table>")
        else:
            measured = ('<p class="note">The model measures no importances of its own; a fitted price, news or joint '
                        'part with a feature_importances() method would list them here.</p>')
        parts.append(f'<h3>{escape(model["name"])} <span class="meta">({model["scored"]:,} stocks scored)</span></h3>'
                     f'<p>{_summary_line(model)}</p>{_surrogate_note(model)}'
                     f'<div class="grid"><div>{driver_chart(model["drivers"])}</div>'
                     f'<div><b>The score and its tie-break</b><table>{outputs}</table>{measured}</div></div>')
    return "".join(parts) + "</section>"


def _surrogate_note(model: dict) -> str:
    """What the SHAP and LIME values of a model's stocks explain, and how faithfully."""
    if model.get("local_error"):
        return f'<p class="banner bad">No SHAP and LIME values: {escape(model["local_error"])}.</p>'
    if not model.get("local"):
        return ""
    fidelity = _number(model.get("fidelity"))
    fit = (f"an R² of {fidelity:.2f} on the stocks held out of its fit" if fidelity is not None
           else "too few stocks to measure its fit")
    return (f'<p class="note">Hovering its stocks shows the SHAP and LIME values of a surrogate of this ranking: a '
            f'gradient-boosted tree that predicts each stock\'s rank percentile from the features below and its '
            f'dollar volume, with {fit}. The closer it is to 1, the more the values say about the model itself.</p>')


def _finite(value):
    """value with every float that is not finite as None, for JSON that JavaScript reads."""
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _hover_data(snapshot: dict | None) -> str:
    """The listed stocks' SHAP and LIME values by model, as the JSON the page's script reads on hover."""
    models = {}
    explanation = (snapshot or {}).get("explanation") or {}
    for model in explanation.get("models") or []:
        if model.get("local"):
            models[model["name"]] = {"fidelity": model.get("fidelity"), "stocks": model["local"]}
    data = json.dumps(_finite(models), allow_nan=False).replace("</", "<\\/")
    return f'<script type="application/json" id="local-explanations">{data}</script>'


def _history_section(history: list[dict]) -> str:
    parts = ['<section id="calls"><h2>Calls of the API</h2>',
             '<p class="note">Every GET or POST of /recommendations/latest, by hand or by the 16:00 email task, '
             'newest first. Email addresses are never recorded.</p>']
    if not history:
        return "".join(parts) + '<p class="muted">None yet.</p></section>'
    rows = []
    for call in history:
        result, fetched, emailed = _call_text(call)
        buys = "; ".join(f"{escape(model)}: {escape(', '.join(tickers))}" for model, tickers in
                         (call.get("buy") or {}).items())
        status = "pos" if not call.get("error") else "neg"
        rows.append(f'<tr><td>{escape(_et(call["at"], "%Y-%m-%d %H:%M:%S"))}</td><td class="text">'
                    f'{escape(call["method"])}</td><td class="text {status}">{escape(result)}</td>'
                    f'<td class="text">{escape(fetched)}</td><td class="text">{escape(emailed)}</td>'
                    f'<td>{escape(call.get("as_of") or "—")}</td><td class="text">{buys or "—"}</td></tr>')
    return "".join(parts) + ('<div class="scroll"><table><tr><th>When</th><th class="text">Call</th>'
                             '<th class="text">Result</th><th class="text">Fetch</th><th class="text">Email</th>'
                             '<th>Bars through</th><th class="text">First buys</th></tr>'
                             f'{"".join(rows)}</table></div></section>')


def render(snapshot: dict | None, history: list[dict], failed: dict | None = None) -> str:
    """The page: the last lists, their explanation and the latest calls. failed is a call that just failed, shown
    above them."""
    parts = ["<header><h1>ML510 recommendations</h1>"]
    if snapshot is not None:
        lists = snapshot["lists"]
        call = snapshot["call"]
        _, fetched, emailed = _call_text(call)
        fetched = "no fetch" if fetched == "—" else f"fetch {fetched}"
        emailed = "no email" if emailed == "—" else f"email {emailed}"
        parts.append(f'<p class="meta">Last lists: {escape(call["method"])} call at {escape(_et(call["at"]))} '
                     f'({escape(fetched)}, {escape(emailed)}) · bars through {escape(lists["as_of"])} · '
                     f'news before {escape(_et(lists["news_cutoff"]))} · computed {escape(_et(lists["generated_at"]))}'
                     f' · {escape(lists["model_version"])}'
                     + (f' from run <code>{escape(lists["run"])}</code>' if lists["run"] else "") + "</p>")
    parts.append("</header>")
    if failed is not None:
        parts.append(f'<p class="banner bad">The last call, {escape(failed["method"])} at {escape(_et(failed["at"]))}, '
                     f'failed: {escape(failed["error"] or "")}'
                     + (' The lists below are from the last call that returned some.' if snapshot else "") + "</p>")
    if snapshot is None:
        parts.append('<p class="banner">No call of /recommendations/latest has returned lists yet. This page is '
                     'written again at every call: by hand, or by the 16:00 email task.</p>')
    else:
        parts += [_lists_section(snapshot), _drivers_section(snapshot)]
    parts.append(_history_section(history))
    parts.append('<p class="note">This page refreshes every minute. Every call of the API writes it again, as '
                 'index.html in the dashboard folder (RECOMMENDER_DASHBOARD_DIR, by default data/processed/dashboard); '
                 'the API serves it at /dashboard.</p>')
    parts.append('<div id="tip" class="tip" role="tooltip" hidden></div>')
    parts.append(_hover_data(snapshot))
    return PAGE.format(css=CSS, body="\n".join(parts), script=SCRIPT)
