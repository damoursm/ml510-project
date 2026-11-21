"""Explain the API's recommendations: the features of the stocks each model scored, and which of them its ranking
follows.

For one result of the API, explain() computes the ten price features (features.py) and the eight news features
(news_features.py) of every stock the served models scored, on the decision day (as_of), from the files on disk: the
bars through as_of and the articles published before its 16:00 New York close, with their LLM scores. A news feature
that needs LLM scores is NaN until the articles are scored.

A model's drivers are the Spearman correlations, across the stocks it scored, between each feature and its ranking:
+1 when the ranking follows the feature exactly (a higher value ranks higher), -1 when it goes the other way. They
describe any model, a rule such as most_liquid as well as a fitted one. A model that measures its own importances
also shows them: a price or news part with a feature_importances() method returning {feature: importance}.

Each listed stock also gets local explanations, SHAP and LIME values: how much each feature moves its rank percentile
(1 the best stock, 0 the worst) away from the average stock's. The models rank from raw bars and news, not from a
row of features, so both explain a surrogate: a gradient-boosted tree fit to predict the model's rank percentile from
the scored stocks' LOCAL_FEATURES, the features plus the dollar volume that breaks ties between equal scores. Its R² on
stocks held out of the fit tells how faithful it is.
- SHAP values (shap_values): a permutation estimate of interventional Shapley values. Walks start from STARTS scored
  stocks spread evenly along the ranking, each with a random order of the features and its reverse, and switch the
  features to the stock's values one at a time; a feature's value is the mean change in prediction when it switches.
  They add up to the stock's prediction minus the base, the mean prediction of the start stocks: about 0.5 for all.
- LIME (lime_values): LIME_SAMPLES perturbations keep each feature at the stock's value or take a random scored
  stock's, half and half; LIME's exponential kernel weighs them by their standardized distance to the stock; and a
  weighted ridge regression of the predictions on which features kept the stock's value gives each feature's effect.
Nothing here changes a recommendation.
"""
from dataclasses import dataclass
import logging
import math
import warnings
import zlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from .config import Settings
from .data import read_bars, read_news
from .features import FEATURES, LOOKBACK, decision_features
from .news_features import NEWS_FEATURES, NEWS_LOOKBACK_DAYS, decision_news_features
from .recommend import news_cutoff

DESCRIPTIONS = {
    "ret_1d": "Return on the decision day, close to close",
    "ret_5d": "Return over the last 5 sessions",
    "ret_20d": "Return over the last 20 sessions",
    "momentum_60d": "Return from 59 to 5 sessions back",
    "gap_1d": "The decision day's open over the previous close",
    "close_location": "Where the day closed in its range: 0 at the low, 1 at the high",
    "volatility_20d": "Standard deviation of the last 20 daily log returns",
    "high_60d": "Close relative to the highest high of 60 sessions",
    "volume_ratio": "Mean volume of the last 5 sessions over that of 60",
    "dollar_volume_20d": "Median dollars traded over 20 sessions (log)",
    "news_count_7d": "Articles tagged with the stock in the last 7 days",
    "news_count_1d": "Articles since the previous weekday's close",
    "sentiment_7d": "Relevance-weighted LLM sentiment over 7 days, -1 to 1",
    "sentiment_1d": "The same since the previous weekday's close",
    "major_sentiment_7d": "Sentiment of the relevant articles rated major news",
    "impact_7d": "Highest LLM impact of the relevant articles, 0 to 1",
    "earnings_7d": "1 if a relevant article is about earnings or guidance",
    "analyst_sentiment_7d": "Sentiment of the relevant analyst articles",
}
# The ranking table's own columns: the model's outputs, and the dollar volume that breaks ties between equal scores.
OUTPUTS = {
    "score": "The model's score, which sets the ranking",
    "expected_return": "The price model's expected return",
    "news_sentiment": "The news model's sentiment",
    "dollar_volume": "Median dollars traded over the bars window; breaks ties between equal scores",
}
KINDS = {**dict.fromkeys(FEATURES, "price"), **dict.fromkeys(NEWS_FEATURES, "news"), **dict.fromkeys(OUTPUTS, "output")}
MIN_STOCKS = 20  # Fewest stocks with a value for a correlation, or a surrogate, to be computed.
BARS_SLACK = 10  # Day files read beyond the features' window, for days stored after as_of.
LOCAL_FEATURES = FEATURES + NEWS_FEATURES + ["dollar_volume"]  # What the surrogate reads; SHAP and LIME explain.
STARTS = 100  # Stocks along the ranking that a stock's SHAP walks start from, two orders each.
LIME_SAMPLES = 5_000  # Perturbations per stock for LIME.
KERNEL_WIDTH = 0.75 * math.sqrt(len(LOCAL_FEATURES))  # LIME's default kernel width.
log = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class Surrogate:
    """A gradient-boosted tree that predicts a model's rank percentile (1 the best, 0 the worst) from the scored
    stocks' LOCAL_FEATURES: what SHAP and LIME explain, in place of the model."""
    model: HistGradientBoostingRegressor
    tickers: pd.Index  # The scored stocks, best first.
    inputs: np.ndarray  # Their LOCAL_FEATURES as the tree reads them: the dollar volume as its log.
    values: pd.DataFrame  # The same as they are, indexed by ticker.
    ranks: np.ndarray  # Their rank percentiles.
    fidelity: float | None  # R² on the fifth of the stocks held out of a fit; None with too few stocks.
    scale: np.ndarray  # Each input's standard deviation, for LIME's distances.


@dataclass(frozen=True)
class ModelExplanation:
    name: str
    version: str
    scored: int  # Stocks the model scored.
    drivers: list[dict]  # One {"feature", "kind", "correlation", "stocks"} per feature, strongest first.
    outputs: list[dict]  # The same for the ranking table's columns (OUTPUTS).
    importances: dict[str, float] | None  # What the model measures itself, by feature; None if nothing.
    surrogate: Surrogate | None = None  # For the listed stocks' SHAP and LIME values; None if it could not be fit.


@dataclass(frozen=True)
class Explanation:
    as_of: pd.Timestamp
    features: pd.DataFrame  # FEATURES and NEWS_FEATURES of the scored stocks, indexed by ticker.
    percentiles: pd.DataFrame  # Each value's percentile among the scored stocks, 0 to 1.
    models: tuple[ModelExplanation, ...]


def stock_features(settings: Settings, as_of: pd.Timestamp, tickers: pd.Index) -> pd.DataFrame:
    """Return the price and news features of the tickers for a decision at as_of's close, from the files on disk."""
    bars = read_bars(settings.bars_dir, LOOKBACK + BARS_SLACK)
    bars = bars[(bars.date <= as_of) & bars.ticker.isin(set(tickers))]
    price = decision_features(bars).reindex(tickers)
    cutoff = news_cutoff(as_of)
    news = read_news(settings.news_dir, cutoff - pd.Timedelta(days=NEWS_LOOKBACK_DAYS), cutoff,
                     settings.sentiment_dir, settings.ollama_model)
    return price.join(decision_news_features(news, tickers, cutoff))[FEATURES + NEWS_FEATURES]


def correlations(ranked: pd.Index, values: pd.DataFrame) -> list[dict]:
    """Return the Spearman correlation of each column of values with a best-first ranking of tickers, strongest
    first; None where fewer than MIN_STOCKS have a value or the value never changes."""
    position = pd.Series(np.arange(len(ranked), 0, -1, dtype=np.float64), index=ranked)  # The best gets the most.
    rows = []
    for name in values.columns:
        column = values[name].reindex(ranked)
        known = column.notna().to_numpy()
        stocks = int(known.sum())
        correlation = None
        if stocks >= MIN_STOCKS and column[known].nunique() > 1:
            correlation = float(column[known].rank().corr(position[known].rank()))
        rows.append({"feature": str(name), "kind": KINDS.get(name, "other"), "correlation": correlation,
                     "stocks": stocks})
    rows.sort(key=lambda row: -abs(row["correlation"]) if row["correlation"] is not None else 1.0)
    return rows


def importances(model) -> dict[str, float] | None:
    """Return the importances a model measures itself, by feature: its price and news parts' feature_importances(),
    or an ensemble's members' weighted by their weights; None when no part measures any."""
    members = getattr(model, "members", None)
    if members is not None:  # An Ensemble.
        total, weights = {}, sum(weight for _, weight in members) or 1.0
        for member, weight in members:
            for name, value in (importances(member) or {}).items():
                total[name] = total.get(name, 0.0) + weight / weights * value
        return total or None
    found = {}
    for part in (getattr(model, "price", None), getattr(model, "news", None)):
        measure = getattr(part, "feature_importances", None)
        if callable(measure):
            for name, value in dict(measure()).items():
                found[str(name)] = found.get(str(name), 0.0) + float(value)
    return found or None


def fit_surrogate(table: pd.DataFrame, features: pd.DataFrame) -> Surrogate | None:
    """Fit the surrogate of a model's ranking: its table (best first, with dollar_volume) and the scored stocks'
    features. None with fewer than MIN_STOCKS stocks."""
    tickers = pd.Index(table.ticker.astype(str), dtype=object)
    count = len(tickers)
    if count < MIN_STOCKS:
        return None
    values = features.reindex(tickers)[FEATURES + NEWS_FEATURES].assign(
        dollar_volume=table.dollar_volume.to_numpy(dtype=np.float64))
    inputs = values.to_numpy(dtype=np.float64).copy()
    volume = inputs[:, -1]
    inputs[:, -1] = np.log(np.where(volume > 0, volume, np.nan))
    ranks = 1 - np.arange(count) / (count - 1)
    leaf = max(2, min(20, count // 50))  # Scikit-learn's 20 stocks a leaf, fewer for a small universe.

    def tree() -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(max_iter=200, min_samples_leaf=leaf, random_state=0)

    held = np.random.default_rng(0).random(count) < 0.2
    fidelity = None
    if held.sum() >= 2 and (~held).sum() >= MIN_STOCKS:
        fidelity = float(tree().fit(inputs[~held], ranks[~held]).score(inputs[held], ranks[held]))
    with warnings.catch_warnings():  # A column with no value (news not scored yet) has no spread: scale 1.
        warnings.simplefilter("ignore", RuntimeWarning)
        scale = np.nanstd(inputs, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    return Surrogate(tree().fit(inputs, ranks), tickers, inputs, values.set_axis(tickers), ranks, fidelity, scale)


def shap_values(predict, x: np.ndarray, starts: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Return the permutation estimate of predict's interventional SHAP values at x, walking from each row of starts
    in a random order of the features and in its reverse (which lowers the noise), and their base: the mean
    prediction of starts. base + the values == predict(x), exactly."""
    features = len(x)
    order = np.array([rng.permutation(features) for _ in range(len(starts))])
    order = np.concatenate([order, order[:, ::-1]])
    starts = np.concatenate([starts, starts])
    position = np.argsort(order, axis=1)  # position[k, j]: the step at which order k switches feature j.
    switched = position[:, None, :] < np.arange(features + 1)[None, :, None]
    walks = np.where(switched, x[None, None, :], starts[:, None, :])
    predictions = predict(walks.reshape(-1, features)).reshape(len(order), features + 1)
    walk = np.arange(len(order))[:, None]
    changes = predictions[walk, position + 1] - predictions[walk, position]
    return changes.mean(axis=0), float(predictions[:, 0].mean())


def lime_values(predict, x: np.ndarray, background: np.ndarray, scale: np.ndarray, rng: np.random.Generator,
                samples: int = LIME_SAMPLES) -> tuple[np.ndarray, float, float]:
    """Return LIME's effect of each feature of x on predict, the fit's intercept and its weighted R²."""
    features = len(x)
    kept = rng.random((samples, features)) < 0.5
    kept[0] = True  # x itself.
    perturbed = np.where(kept, x, background[rng.integers(len(background), size=samples)])
    predictions = predict(perturbed)
    distance = np.sqrt((np.nan_to_num((perturbed - x) / scale) ** 2).sum(axis=1))
    weights = np.sqrt(np.exp(-distance ** 2 / KERNEL_WIDTH ** 2))  # LIME's kernel.
    design = np.column_stack([np.ones(samples), kept.astype(np.float64)])
    penalty = np.eye(features + 1)
    penalty[0, 0] = 0.0  # A ridge of alpha 1, as LIME's, on everything but the intercept.
    weighted = design * weights[:, None]
    coefficients = np.linalg.solve(design.T @ weighted + penalty, weighted.T @ predictions)
    mean = np.average(predictions, weights=weights)
    spread = np.sum(weights * (predictions - mean) ** 2)
    fit = 1 - np.sum(weights * (predictions - design @ coefficients) ** 2) / spread if spread > 0 else 1.0
    return coefficients[1:], float(coefficients[0]), float(fit)


def local_explanations(surrogate: Surrogate, tickers) -> dict[str, dict]:
    """Return, for each of the tickers that the surrogate's model scored, its SHAP and LIME values by feature (in
    rank percentile), its actual and predicted rank percentiles, the SHAP base, LIME's local R² and its features'
    values. Each stock draws from its own seed, so a result's explanations never change."""
    rows = {ticker: index for index, ticker in enumerate(surrogate.tickers)}
    # The same start stocks for every stock, spread evenly along the ranking: one base, about 0.5, for all.
    starts = surrogate.inputs[np.unique(np.linspace(0, len(surrogate.tickers) - 1, STARTS).round().astype(int))]
    found = {}
    for ticker in tickers:
        if ticker not in rows:
            continue
        index = rows[ticker]
        x = surrogate.inputs[index]
        rng = np.random.default_rng([20261120, zlib.crc32(str(ticker).encode())])
        shap, base = shap_values(surrogate.model.predict, x, starts, rng)
        lime, _, fit = lime_values(surrogate.model.predict, x, surrogate.inputs, surrogate.scale, rng)
        found[ticker] = {"actual": float(surrogate.ranks[index]),
                         "predicted": float(surrogate.model.predict(x[None, :])[0]), "base": base,
                         "shap": dict(zip(LOCAL_FEATURES, shap.tolist())),
                         "lime": dict(zip(LOCAL_FEATURES, lime.tolist())), "lime_fit": fit,
                         "values": surrogate.values.loc[ticker].to_dict()}
    return found


def explain(settings: Settings, served: list) -> Explanation:
    """Explain one result of the API from served: (name, version, Recommendations, models) for each served model,
    ranked from the same files (one as_of)."""
    as_of = served[0][2].as_of
    tickers = pd.Index(sorted({str(ticker) for _, _, result, _ in served for ticker in result.table.ticker}),
                       dtype=object, name="ticker")
    features = stock_features(settings, as_of, tickers)
    models = []
    for name, version, result, model in served:
        table = result.table.set_index(result.table.ticker.astype(str))
        ranked = pd.Index(table.index, dtype=object)
        outputs = table[[column for column in OUTPUTS if column in table]]
        try:
            surrogate = fit_surrogate(result.table, features)
        except Exception:  # The rest of the explanation stands without the stocks' SHAP and LIME values.
            log.exception("Dashboard: could not fit the surrogate of %s", name)
            surrogate = None
        models.append(ModelExplanation(str(name), str(version), len(ranked), correlations(ranked, features),
                                       correlations(ranked, outputs), importances(model), surrogate))
    return Explanation(as_of, features, features.rank(pct=True), tuple(models))
