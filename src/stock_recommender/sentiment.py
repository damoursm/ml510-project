"""Score news articles with a local LLM served by Ollama, and store the scores beside the news.

For each article (title and summary) and each company it is tagged with, the LLM gives:
- event: what the article is about for that company (EVENTS);
- relevance: 0 only mentioned, 1 one of the main subjects, 2 the article is about it;
- impact: 0 no new information, 1 routine news, 2 major news likely to move the stock;
- sentiment: -2 very negative to 2 very positive for that company's stock.
It sees only the article's own text, so a score holds nothing from after the article was published. The model's
own training can still know what happened later to a company it recognizes, or after a date it reads, so the text it
gets hides the companies and the dates (anonymize.mask): Ford and F become Company A, the other companies named get
the next placeholders, "March 15, 2023" becomes [date] and 2023 becomes 20XX, and the prompt asks what the text
says, not for a forecast. Products, people, places and companies that nothing names as one stay recognizable.

The tickers scored are an article's tags, each company once, without other securities than common shares
(targets()): Polygon tags an article about JPM with its six preferred shares too (JPMpC...), AGNC's with AGNCM to
AGNCP, and SOUN's with its warrants (SOUNW). Articles with no ticker left, or more than max_tickers (roundups and
lists), are not scored.

Scores are stored like the news: one parquet file per UTC publication day, <sentiment_dir>/<year>/<day>.parquet,
with one row per article id and scored ticker, the model, its digest (the weights the model's tag named) and the
PROMPT_VERSION that produced it. Only valid scores are stored: a ticker the LLM failed to score has no row, and the
next run asks again. Changing the prompt or the model makes the stored scores stale: readers ignore them (they stay
in the files) and score_news.py scores the articles again. <sentiment_dir>/model_digests.json holds the weights each
model's scores come from (check_digest).
"""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import random
import re
import threading
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from .anonymize import Aliases, mask

# Bump on any change to SYSTEM_PROMPT, user_prompt(), targets(), the answer schema or how anonymize.mask() hides.
# 2: no ticker or name of any company, no date. 3: a compact answer on one line (45% fewer tokens to generate).
PROMPT_VERSION = 3
EVENTS = ["earnings", "guidance", "analyst", "deal", "product", "legal", "management", "capital", "price_move",
          "opinion", "macro", "other"]
SYSTEM_PROMPT = """You label financial news for a stock-trading model. You get one article's title and summary and the companies to label it for. Company names and tickers are replaced by placeholders (Company A, Company B, ...) and dates are removed: [date] stands for a day, [month] for a month and 20XX for a year; web addresses and security codes are [link] and [code]. The text may hold more placeholders than the list: label only the listed companies. For each listed company, label what the text itself reports: use only the title and the summary, and do not guess which company it is, when the article was written or what happened afterwards. A listed company whose placeholder is not in the text may still be its subject ("this stock" in an article with one listed company); if nothing in the text refers to it, its relevance is 0.

For each listed company give:
- event: what the article is about for that company: earnings (results), guidance (outlook), analyst (rating or price-target change), deal (merger, acquisition, partnership, contract), product (launch, approval, trial result), legal (lawsuit, investigation, regulation), management (executive change), capital (offering, buyback, dividend, split), price_move (explains why the stock moved, e.g. "Why Company A stock soared today"), opinion (buy or sell commentary, stock picks, lists), macro (economy, sector or market news), other.
- relevance: 0 the company is only mentioned or listed, 1 it is one of the main subjects, 2 the article is about it.
- impact: how much new information the article brings, whatever its direction. 0 none (commentary, opinions, recaps, lists, explanations of past price moves), 1 new but routine information (filings, share-buyback transactions, regular dividends, conference presentations, small contracts or partnerships, product updates), 2 major news that usually moves a stock several percent (earnings surprises, guidance changes, merger or acquisition agreements, regulatory decisions, large lawsuits or settlements, chief executive departures). Most articles are 0 or 1. Articles explaining a past price move (price_move) or giving an opinion (opinion) are always 0. Law-firm notices that invite investors to join a class action or remind them of a deadline repeat an existing case: legal, impact 0, sentiment -1.
- sentiment: the article's tone for that company's stock, whatever its impact: -2 very negative, -1 negative, 0 neutral or mixed, 1 positive, 2 very positive. Routine or opinion articles still have a tone: a dividend increase, a bullish pick or a report of rising shares is positive.

Write the answer as compact JSON on a single line, with no spaces or line breaks."""
ANSWER_SCHEMA = {"type": "object", "properties": {"scores": {"type": "array", "items": {
    "type": "object", "properties": {
        "company": {"type": "string"}, "event": {"type": "string", "enum": EVENTS},
        "relevance": {"type": "integer", "enum": [0, 1, 2]}, "impact": {"type": "integer", "enum": [0, 1, 2]},
        "sentiment": {"type": "integer", "enum": [-2, -1, 0, 1, 2]}},
    "required": ["company", "event", "relevance", "impact", "sentiment"]}}}, "required": ["scores"]}
SCORES = ["event", "relevance", "impact", "sentiment"]
COLUMNS = ["id", "published_utc", "ticker", *SCORES, "model", "model_digest", "prompt_version"]
SCHEMA = pa.schema([("id", pa.string()), ("published_utc", pa.timestamp("ms", tz="UTC")), ("ticker", pa.string()),
                    ("event", pa.string()), ("relevance", pa.int8()), ("impact", pa.int8()),
                    ("sentiment", pa.int8()), ("model", pa.string()), ("model_digest", pa.string()),
                    ("prompt_version", pa.int16())])
RANGES = {"relevance": (0, 2), "impact": (0, 2), "sentiment": (-2, 2)}
# Tags of securities other than common shares, in Polygon's notation: preferred shares (JPMpC, Ap, CYCC.P), warrants
# (IONQ.WS, IONQ.WS.A), when-issued shares (DDw), rights (SOULr) and units (AIIA.U, GRP.U). Classes (BRK.B) are kept.
OTHER_SECURITY = re.compile(r"(p[A-Z]*|w|r|\.WS(\.[A-Z])?|\.U|\.P)$")
# Nasdaq names an issuer's other securities with a fifth letter: preferred shares (AGNCP, AGNCL, WHLRD), notes (OXLCZ,
# TMUSI, RILYK...), rights (BRLIR), units (PRSRU), warrants (SOUNW) and foreign shares (BABAF). Such a tag is dropped
# when the issuer's 4-letter ticker is tagged too. A and C name share classes; so do L, K and B in SHARE_CLASSES.
NASDAQ_OTHER = frozenset("BDFGHIJKLMNOPRSTUWZ")
SHARE_CLASSES = frozenset({"GOOGL", "LILAK", "UONEK", "METCB", "AMTBB"})  # Common shares tagged beside GOOG, LILA...
MAX_FAILURES_IN_A_ROW = 12  # Articles in a row without an answer while /api/tags answers: the runner is failing.
# Tokens of context per request. Ollama reloads the model when a request asks for another context than the one it
# has loaded, so this matches the server's OLLAMA_CONTEXT_LENGTH: the llm_trader project shares the server and runs
# it at 8192. A prompt longer than the context loses its start, the instructions, silently: see MAX_SUMMARY_CHARS.
NUM_CTX = 8192
MAX_SUMMARY_CHARS = 12_000  # Longer summaries are cut, so every prompt fits NUM_CTX with room for the answer.
MAX_TITLE_CHARS = 1_000
MAX_ANSWER_TOKENS = 512  # The answer for 5 tickers takes about 200 tokens; a runaway answer is cut, then asked again.
RETRY_TEMPERATURE = 0.5  # A retry samples, with its own seed, so it does not repeat the same malformed answer.
# A scorer may use a smaller context than NUM_CTX, on a server run with more, smaller slots for a backfill. It then
# sends an article only if its prompt surely fits with room for the answer: at most MIN_CHARS_PER_TOKEN characters a
# token (prose has about 4, but Qwen writes each digit as a token) and TEMPLATE_TOKENS for the chat template. The
# other articles stay unscored until a run at NUM_CTX.
MIN_CHARS_PER_TOKEN = 2.0
TEMPLATE_TOKENS = 32


class ScorerUnavailable(RuntimeError):
    """The LLM server cannot be reached, does not serve the model, or now serves other weights under its name."""


def targets(tickers) -> list[str]:
    """Return the tickers an article is scored for: its tags, in order, without other securities than common shares
    (OTHER_SECURITY, NASDAQ_OTHER), and each company once: of two spellings of one ticker (CMCS.A, CMCSA), the one
    without punctuation."""
    if tickers is None or isinstance(tickers, float):
        return []
    tags = list(dict.fromkeys(str(ticker) for ticker in tickers))
    present, kept = set(tags), {}
    for tag in tags:
        if OTHER_SECURITY.search(tag) or (len(tag) == 5 and tag[4] in NASDAQ_OTHER and tag[:4] in present
                                          and tag not in SHARE_CLASSES):
            continue
        key = _key(tag)
        if key not in kept or (tag.isalnum() and not kept[key].isalnum()):
            kept[key] = tag
    return list(kept.values())


def others(tickers) -> dict[str, str]:
    """The tags targets() leaves out that name a scored ticker's company, each with that ticker: its other
    securities (JPMpC with JPM, SOUNW with SOUN) and its other spellings (CMCS.A with CMCSA)."""
    if tickers is None or isinstance(tickers, float):
        return {}
    kept = {_key(ticker): ticker for ticker in targets(tickers)}
    found = {}
    for tag in dict.fromkeys(str(ticker) for ticker in tickers):
        if tag in kept.values():
            continue
        root = (OTHER_SECURITY.sub("", tag) if OTHER_SECURITY.search(tag)
                else tag[:4] if len(tag) == 5 and tag[4] in NASDAQ_OTHER else tag)
        if _key(root) in kept:
            found[tag] = kept[_key(root)]
    return found


def eligible(news: pd.DataFrame, max_tickers: int) -> np.ndarray:
    """Whether each article is scored: it has 1 to max_tickers targets."""
    count = news.tickers.map(lambda tickers: len(targets(tickers))).to_numpy(dtype=np.int64)
    return (count >= 1) & (count <= max_tickers)


def answer_schema(companies) -> dict:
    """ANSWER_SCHEMA for one article: one score for each of the companies asked, named as asked."""
    schema = copy.deepcopy(ANSWER_SCHEMA)
    scores = schema["properties"]["scores"]
    scores["items"]["properties"]["company"]["enum"] = list(companies)
    scores["minItems"] = scores["maxItems"] = len(companies)
    return schema


def user_prompt(title: str | None, description: str | None, companies) -> str:
    """The article as the LLM gets it: its masked title and summary, and the placeholders to score."""
    return f"Title: {title or ''}\nSummary: {description or ''}\nCompanies: {', '.join(companies)}"


def _key(ticker) -> str:
    """A ticker or a placeholder as the LLM may write it: BRK-B, brk.b and BRK.B are one key."""
    return re.sub(r"[^A-Z0-9]", "", str(ticker).upper())


def parse(content: str | None, tickers) -> dict[str, dict]:
    """Return the valid scores the LLM's answer gives for the names asked (placeholders, or tickers), by name; {}
    if it is malformed.

    A name matches whatever its case or punctuation. An item that is incomplete, out of range or for a name not
    asked is ignored, and so is a second score for the same name.
    """
    try:
        answer = json.loads(content)["scores"]
    except (ValueError, KeyError, TypeError):
        return {}
    if not isinstance(answer, list):
        return {}
    asked = {}
    for ticker in tickers:
        asked.setdefault(_key(ticker), ticker)
    scores = {}
    for item in answer:
        try:
            ticker = asked.get(_key(item["company"]))
            values = {"event": item["event"], **{name: int(item[name]) for name in RANGES}}
        except (KeyError, TypeError, ValueError):
            continue
        if (ticker is None or ticker in scores or values["event"] not in EVENTS
                or any(not low <= values[name] <= high for name, (low, high) in RANGES.items())):
            continue
        scores[ticker] = values
    return scores


class OllamaScorer:
    """Scores articles through Ollama's chat API: deterministic (temperature 0, fixed seed) on the first request,
    thinking off, and the answer constrained to ANSWER_SCHEMA. The LLM gets each article with its companies and
    dates hidden (anonymize.mask); aliases are the names to hide, and without them only the tickers and the names
    the article writes beside them are.

    check() records the digest of the weights the server has under the model's name; a later check() that finds
    other weights raises ScorerUnavailable, so one run never mixes two models' scores. Retries sample with seeds
    that differ from run to run, so a later run does not repeat the answers that failed.
    """

    def __init__(self, url: str = "http://localhost:11434", model: str = "qwen3:8b", timeout: float = 300,
                 attempts: int = 3, session: requests.Session | None = None, aliases: Aliases | None = None,
                 num_ctx: int = NUM_CTX):
        self.url, self.model, self.timeout, self.attempts = url.rstrip("/"), model, timeout, attempts
        self.num_ctx = num_ctx  # Must match the server's context, or it reloads the model.
        self.session = session or requests.Session()
        self.aliases = aliases or Aliases()
        self.digest: str | None = None
        self.retry_seed = random.SystemRandom().randrange(1, 1 << 30)
        self._lock, self._failures = threading.Lock(), 0

    def _answered(self, answered: bool) -> None:
        """Count the articles in a row the server left without a usable answer (none, or one that scores no
        company), and stop at MAX_FAILURES_IN_A_ROW."""
        with self._lock:
            self._failures = 0 if answered else self._failures + 1
            failures = self._failures
        if failures >= MAX_FAILURES_IN_A_ROW:
            raise ScorerUnavailable(f"{failures} articles in a row got no score from {self.model} at {self.url}, "
                                    f"though the server is up: its model runner may be failing.")

    def check(self) -> None:
        """Raise ScorerUnavailable unless the server answers, has the model, and has the same weights as at the
        first check."""
        try:
            response = self.session.get(f"{self.url}/api/tags", timeout=10)
            response.raise_for_status()
            models = {model.get("name"): model for model in response.json().get("models", [])}
        except (requests.RequestException, ValueError, AttributeError) as error:
            raise ScorerUnavailable(f"Ollama is not reachable at {self.url}: {error}") from error
        found = models.get(self.model) or models.get(f"{self.model}:latest")
        if found is None:
            raise ScorerUnavailable(f"Ollama at {self.url} does not have {self.model}; it has {sorted(models)}. "
                                    f"Run: ollama pull {self.model}")
        digest = found.get("digest")
        if self.digest is None:
            self.digest = digest
        elif digest != self.digest:
            raise ScorerUnavailable(f"Ollama's {self.model} changed from digest {self.digest[:12]} to "
                                    f"{str(digest)[:12]} (pulled again?); scores of the two would mix.")

    def body(self, title: str | None, description: str | None, companies, retry: int = 0) -> dict:
        """The request for a masked article and its placeholders; retry counts the answers already found malformed
        or partial, which a resample replaces."""
        options = {"temperature": 0, "seed": 0, "num_ctx": self.num_ctx, "num_predict": MAX_ANSWER_TOKENS}
        if retry:
            options.update(temperature=RETRY_TEMPERATURE, seed=self.retry_seed + retry)
        # No keep_alive: the server's own setting keeps the model loaded for the other programs that use it.
        return {"model": self.model, "stream": False, "think": False, "format": answer_schema(companies),
                "options": options,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": user_prompt(title, description, companies)}]}

    def score_article(self, title: str | None, description: str | None, tickers, published=None,
                      tags=None, wanted=None) -> dict[str, dict]:
        """Return the scores of the tickers, by ticker: all of them, or those the LLM scored in `attempts` requests.
        The LLM gets the article masked: published and tags, the article's day and all its tags, tell mask() what
        to hide. Tickers of one company get the same score. With wanted, a part of the tickers, the LLM is asked
        for those only: the others are still hidden behind their placeholders.

        A malformed or partial answer is asked again (sampling, with another seed); an article the server refuses
        (HTTP 400) or does not answer while it answers check() gets the scores it has, maybe none. A server that
        stops answering, or leaves MAX_FAILURES_IN_A_ROW articles in a row without an answer, raises
        ScorerUnavailable.
        """
        tickers = list(tickers)
        title, description = (text if isinstance(text, str) else "" for text in (title, description))
        title, description, labels = mask(title[:MAX_TITLE_CHARS], description[:MAX_SUMMARY_CHARS], tickers,
                                          self.aliases, published, others(tags))
        wanted = tickers if wanted is None else [ticker for ticker in tickers if ticker in set(wanted)]
        asked = {ticker: company for ticker, company in zip(tickers, labels) if ticker in wanted}
        companies = list(dict.fromkeys(asked.values()))
        if not asked or not self.fits(title, description, companies):
            return {}  # An article too long for a reduced context is asked by a later run at NUM_CTX.
        found = self._ask(title, description, companies)
        return {ticker: found[company] for ticker, company in asked.items() if company in found}

    def fits(self, title: str, description: str, companies) -> bool:
        """Whether a masked article's prompt surely fits the context with room for the answer. At NUM_CTX every
        prompt does: MAX_TITLE_CHARS and MAX_SUMMARY_CHARS cut the article to fit."""
        if self.num_ctx >= NUM_CTX:
            return True
        characters = len(SYSTEM_PROMPT) + len(user_prompt(title, description, companies))
        return characters / MIN_CHARS_PER_TOKEN + TEMPLATE_TOKENS + MAX_ANSWER_TOKENS <= self.num_ctx

    def _ask(self, title: str, description: str, tickers: list[str]) -> dict[str, dict]:
        """Return the scores the LLM gives a masked article's placeholders, by placeholder."""
        scores, error, answers = {}, None, 0
        for attempt in range(self.attempts):
            if error is not None:
                time.sleep(min(5 * 3 ** (attempt - 1), 60))
            try:
                response = self.session.post(f"{self.url}/api/chat", json=self.body(title, description, tickers,
                                                                                    answers), timeout=self.timeout)
            except requests.RequestException as exc:
                error = exc
                continue
            if response.status_code >= 500 or response.status_code == 429:
                error = f"HTTP {response.status_code}: {response.text[:300]}"
                continue
            if response.status_code == 400:
                self._answered(bool(scores))
                return scores  # This request is refused (e.g. an input the model cannot take), not the server down.
            if response.status_code != 200:
                raise ScorerUnavailable(f"Ollama answered HTTP {response.status_code}: {response.text[:300]}")
            error, answers = None, answers + 1
            try:
                content = response.json()["message"]["content"]
            except (ValueError, KeyError, TypeError):
                content = ""
            for ticker, values in parse(content, tickers).items():
                scores.setdefault(ticker, values)
            if len(scores) == len(tickers):
                self._answered(True)
                return scores
        if error is not None:
            self.check()  # Raises if the server is down; otherwise only this article went unanswered.
        self._answered(bool(scores))  # An answer that scores nothing, 12 articles in a row, is a failing model too.
        return scores

    def score(self, articles: pd.DataFrame, workers: int = 4, keep=None) -> pd.DataFrame:
        """Return COLUMNS rows for the targets of the articles (columns id, published_utc, title, description,
        tickers) that the LLM scored, asking up to `workers` at once; a target it failed to score has no row. With
        keep, a set of tickers, only the targets in it are scored (wanted())."""
        if self.digest is None:
            self.check()
        rows = list(articles[["id", "published_utc", "title", "description", "tickers"]].itertuples(index=False))
        pool = ThreadPoolExecutor(max(1, workers))
        try:
            answers = list(pool.map(lambda row: self.score_article(
                row.title, row.description, targets(row.tickers), row.published_utc, row.tickers,
                wanted(row.tickers, keep)), rows))
        finally:
            pool.shutdown(cancel_futures=True)  # A failure stops the articles not sent yet.
        if rows:
            self.check()  # Weights pulled again while these were scored would be stored under the old digest.
        records = [{"id": row.id, "published_utc": row.published_utc, "ticker": ticker, **scores[ticker],
                    "model": self.model, "model_digest": self.digest, "prompt_version": PROMPT_VERSION}
                   for row, scores in zip(rows, answers) for ticker in wanted(row.tickers, keep) if ticker in scores]
        return to_frame(records)


def to_frame(records) -> pd.DataFrame:
    """Return score records as a COLUMNS frame with SCHEMA's types (scores missing as NaN or None)."""
    table = pa.Table.from_pylist(list(records), schema=SCHEMA)
    return table.to_pandas()


def current(scores: pd.DataFrame, model: str) -> pd.DataFrame:
    """Return the valid scores by `model` under the current PROMPT_VERSION; other rows are stale."""
    keep = (scores.model == model) & (scores.prompt_version == PROMPT_VERSION) & scores.event.notna()
    return scores[keep].reset_index(drop=True)


def read_day(path: Path, model: str) -> pd.DataFrame:
    """Return a day file's current scores by `model`."""
    return current(pq.read_table(path, schema=SCHEMA).to_pandas(), model)


def write_day(scores: pd.DataFrame, path: Path, attempts: int = 10) -> None:
    """Write a day's scores atomically, retrying while a reader holds the file open on Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")  # Parquet readers skip dot-files.
    table = pa.Table.from_pandas(scores[COLUMNS].sort_values(["published_utc", "id", "ticker"], kind="stable"),
                                 schema=SCHEMA, preserve_index=False)
    pq.write_table(table, temporary, compression="zstd")
    for attempt in range(attempts):
        try:
            return os.replace(temporary, path)
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.5)


def wanted(tickers, keep=None) -> list[str]:
    """The targets an article is scored for: all of them, or with keep (a set of tickers) those in it."""
    return [ticker for ticker in targets(tickers) if keep is None or ticker in keep]


def _missing(articles: pd.DataFrame, have: set, keep=None) -> np.ndarray:
    """Whether each article has a wanted target without a score in `have`, a set of (id, ticker)."""
    return np.array([any((article, ticker) not in have for ticker in wanted(tickers, keep))
                     for article, tickers in zip(articles.id, articles.tickers)], dtype=bool)


def pending(news: pd.DataFrame, path: Path, model: str, max_tickers: int,
            keep=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a day's current scores and the eligible articles of its news with a target they do not score yet;
    with keep, a set of tickers, only those targets count (the stocks the recommender may trade, tradable_days)."""
    done = read_day(path, model) if path.exists() else to_frame([])
    news = news[eligible(news, max_tickers)]
    return done, news[_missing(news, set(zip(done.id, done.ticker)), keep)]


def score_day(news: pd.DataFrame, path: Path, scorer: OllamaScorer, max_tickers: int,
              workers: int = 4, keep=None) -> tuple[int, int]:
    """Score a day's eligible articles that have a target without a current score, add the new scores to its file,
    and return how many articles were sent and how many of them still miss a score (the next run asks again).
    With keep, a set of tickers, only those targets are scored. The file keeps its other rows: other models' and
    other prompt versions'. Nothing is written if the server stops answering (ScorerUnavailable)."""
    done, todo = pending(news, path, scorer.model, max_tickers, keep)
    if not len(todo):
        return 0, 0
    have = set(zip(done.id, done.ticker))
    new = scorer.score(todo, workers, keep)
    new = new[np.array([pair not in have for pair in zip(new.id, new.ticker)], dtype=bool)]
    if len(new):
        other = pq.read_table(path, schema=SCHEMA).to_pandas() if path.exists() else to_frame([])
        other = other[~((other.model == scorer.model) & (other.prompt_version == PROMPT_VERSION))]
        write_day(pd.concat([frame for frame in (other, done, new) if len(frame)], ignore_index=True), path)
    return len(todo), int(_missing(todo, have | set(zip(new.id, new.ticker)), keep).sum())


def tradable_days(bars: pd.DataFrame, days, min_price: float, min_dollar_volume: float,
                  lookback: int = 60, ahead: int = 8) -> dict:
    """Return, for each news day (UTC dates), the tickers the recommender may trade while it reads that day's
    articles: those that pass its filters (recommend.tradable: the day's close >= min_price, the median close *
    volume of the last `lookback` sessions >= min_dollar_volume) on a session in the `ahead` days that start on
    the news day. The news features read 7 days of articles. A day with no such session yet (today's news, before
    the next session) takes the last session before it. bars holds the columns date, ticker, close and volume, as
    traded then: their closes are those of the day, split-adjusted or not."""
    from .recommend import TEST_SYMBOLS
    bars = bars[["date", "ticker", "close", "volume"]].copy()
    bars["date"] = pd.to_datetime(bars.date).dt.normalize()
    bars["ticker"] = bars.ticker.astype(str)
    bars = bars[~bars.ticker.str.contains(TEST_SYMBOLS)].sort_values(["date", "ticker"], kind="stable")
    sessions = pd.Index(sorted(bars.date.unique()))
    if not len(sessions):
        return {pd.Timestamp(day).date(): set() for day in days}
    # The median dollar volume of each ticker's sessions within the last `lookback` sessions, as tradable() takes it.
    bars["dollar_volume"] = bars.close * bars.volume
    bars["session"] = sessions.get_indexer(bars.date)
    bars = bars.sort_values(["ticker", "session"], kind="stable")
    rolled = (bars.set_index(pd.to_datetime(bars.session, unit="D")).groupby("ticker").dollar_volume
              .rolling(f"{lookback}D").median())
    bars["median"] = rolled.to_numpy()
    passing = bars[(bars.close >= min_price) & (bars["median"] >= min_dollar_volume)]
    by_session = passing.groupby("date").ticker.agg(frozenset)
    found = {}
    for day in days:
        day = pd.Timestamp(day).normalize()
        window = sessions[(sessions >= day) & (sessions < day + pd.Timedelta(days=ahead + 1))]
        if not len(window):
            window = sessions[sessions <= day][-1:]
        found[day.date()] = set().union(*(by_session.get(session, frozenset()) for session in window))
    return found


DIGESTS_FILE = "model_digests.json"  # In the scores' directory: the weights each model's scores come from.


def check_digest(directory: Path, scorer: OllamaScorer, allow_new: bool = False) -> None:
    """Hold a store of scores to one set of weights per model: the digest its first scores came from, kept in
    <directory>/model_digests.json. Raise ScorerUnavailable if the server (scorer.check() first) now has other
    weights under the model's name, unless allow_new, which makes the new weights the store's from now on; the
    rows scored before keep their own digest."""
    path = Path(directory) / DIGESTS_FILE
    known = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    stored = known.get(scorer.model)
    if scorer.digest is None or stored == scorer.digest:
        return
    if stored is not None and not allow_new:
        raise ScorerUnavailable(
            f"The scores in {directory} come from {scorer.model} digest {stored[:12]}, but Ollama now has "
            f"{scorer.digest[:12]} (pulled again?). Restore those weights, or accept the new ones from now on with "
            f"scripts/score_news.py --allow-new-digest --start <today> --end <today> (a recent range, so that no "
            f"older article is scored with them).")
    known[scorer.model] = scorer.digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(known, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def attach(news: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Return the news with an llm_scores column: for each article, one dict per scored tagged ticker (ticker,
    event, relevance, impact, sentiment), in its tickers' order and each ticker once, or None if none is scored."""
    news = news.copy()
    if not len(scores):
        news["llm_scores"] = None
        return news
    by_article = {}
    for row in scores[["id", "ticker", *SCORES]].itertuples(index=False):
        by_article.setdefault(row.id, {})[row.ticker] = {
            "ticker": row.ticker, "event": row.event,
            **{name: None if pd.isna(value) else int(value)
               for name, value in zip(RANGES, (row.relevance, row.impact, row.sentiment))}}
    cells = []
    for article, tickers in zip(news.id, news.tickers):
        scored = by_article.get(article)
        cells.append(None if scored is None or tickers is None
                     else [scored[t] for t in dict.fromkeys(tickers) if t in scored])
    news["llm_scores"] = cells
    return news
