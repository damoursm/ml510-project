"""Ten features of each stock from its daily OHLCV bars, for a price model such as a random forest.

A stock's features on decision day t use only its bars through t's close, counted in trading sessions: ret_5d
compares t's close with the close 5 sessions earlier. A feature is NaN when a bar it reads is missing (a halt, a
recent listing, a window shorter than LOOKBACK); scikit-learn's random forests accept NaN. Prices are compared on one
split basis, so a split changes no feature. Every feature but dollar_volume_20d is a ratio of prices or of volumes,
which the price level does not change; dollar_volume_20d is the log of the dollars traded. Dividends are not added
back, as in the split-adjusted bars PriceModel.predict receives.

The two entry points compute the same features:
- decision_features(bars): from a decision's bars window (what PriceModel.predict receives), the features of the
  tickers on its last day;
- training_features(data): from a TrainingData, its labels with the features of each label's decision day.
Every feature fits in the LOOKBACK sessions a model reads by default (placeholder_models' bars_lookback); with a
shorter bars_lookback, decision time would lack bars training has, so training_features refuses it.
"""
import numpy as np
import pandas as pd

FEATURES = [
    "ret_1d",  # The decision day's return, close to close.
    "ret_5d",  # The return over the last 5 sessions (a week).
    "ret_20d",  # The return over the last 20 sessions (a month).
    "momentum_60d",  # The return from 59 to 5 sessions back: momentum over the 60 sessions, leaving out the last week.
    "gap_1d",  # The decision day's open over the previous close: the overnight move.
    "close_location",  # Where the decision day closed in its range: 0 at the low, 1 at the high (0.5 if no range).
    "volatility_20d",  # The standard deviation of the last 20 daily log returns.
    "high_60d",  # The close relative to the highest high of the last 60 sessions: 0 at the high, below 0 under it.
    "volume_ratio",  # The mean volume of the last 5 sessions over that of the last 60: unusual activity.
    "dollar_volume_20d",  # The log of the median close * volume of the last 20 sessions: liquidity, size.
]
LOOKBACK = 60  # Sessions through the decision day that the features read: the decision day and the 59 before.
PRICES = ["open", "high", "low", "close"]
_CHUNK = 500  # Tickers training_features computes at once, to bound memory.


def compute(open_, high, low, close, volume) -> dict[str, np.ndarray]:
    """Return each feature as a sessions x tickers array, from sessions x tickers arrays of prices on one split
    basis and of volumes, NaN where a ticker has no bar. Row t reads rows t - 60 to t only."""
    o, h, l, c, v = (pd.DataFrame(np.asarray(values, dtype=np.float64)) for values in (open_, high, low, close, volume))
    o, h, l, c = (prices.where(prices > 0) for prices in (o, h, l, c))  # A price of 0 or less is no price.
    v = v.where(v >= 0)
    span = h - l
    location = ((c - l) / span.where(span > 0)).where(span > 0, 0.5).where(c.notna() & span.notna())
    with np.errstate(divide="ignore", invalid="ignore"):  # 0 / 0 and log(0), e.g. from zero volume: NaN below.
        features = {
            "ret_1d": c / c.shift(1) - 1,
            "ret_5d": c / c.shift(5) - 1,
            "ret_20d": c / c.shift(20) - 1,
            "momentum_60d": c.shift(5) / c.shift(59) - 1,
            "gap_1d": o / c.shift(1) - 1,
            "close_location": location,
            # A rolling window is NaN unless it has every bar.
            "volatility_20d": np.log(c / c.shift(1)).rolling(20).std(),
            "high_60d": c / h.rolling(60).max() - 1,
            "volume_ratio": v.rolling(5).mean() / v.rolling(60).mean(),
            "dollar_volume_20d": np.log((c * v).rolling(20).median()),
        }
        arrays = {name: frame.to_numpy() for name, frame in features.items()}
        return {name: np.where(np.isfinite(values), values, np.nan) for name, values in arrays.items()}


def _scatter(rows: np.ndarray, columns: np.ndarray, values: list[np.ndarray], shape: tuple) -> list[np.ndarray]:
    """Return one array of `shape` per values, holding values[k][i] at (rows[i], columns[i]) and NaN elsewhere."""
    arrays = []
    for column in values:
        array = np.full(shape, np.nan)
        array[rows, columns] = column
        arrays.append(array)
    return arrays


_decided = None  # The newest (bars, features): a day's models read one snapshot's bars (recommend.rank).


def decision_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Return the features, indexed by ticker, of each ticker that has a bar on the last day of a bars window.

    bars has columns date, ticker, open, high, low, close and volume, split-adjusted as of its last day (as
    Snapshot.bars and the API's downloaded bars are), and a bar for every trading day a ticker traded. The
    features of the newest bars table are kept, so the models deciding on the same snapshot compute them once.
    """
    global _decided
    if bars.empty:
        return pd.DataFrame(columns=FEATURES, index=pd.Index([], dtype=object, name="ticker"), dtype=np.float64)
    held = _decided
    if held is not None and held[0] is bars:  # Held, so its id cannot be another table's.
        return held[1].copy()
    days = pd.DatetimeIndex(np.sort(bars.date.unique()))[-LOOKBACK:]
    window = bars[bars.date >= days[0]]
    names = window.ticker.astype(str)
    tickers = pd.Index(names[window.date == days[-1]].unique(), dtype=object, name="ticker")
    rows, columns = days.get_indexer(window.date), tickers.get_indexer(names)
    keep = columns >= 0
    values = [window[name].to_numpy(dtype=np.float64)[keep] for name in PRICES + ["volume"]]
    arrays = compute(*_scatter(rows[keep], columns[keep], values, (len(days), len(tickers))))
    features = pd.DataFrame({name: arrays[name][-1] for name in FEATURES}, index=tickers)
    _decided = (bars, features)
    return features.copy()


def training_features(data) -> pd.DataFrame:
    """Return data.labels() with the features of each row's ticker on its decision day, from data.bars().

    Prices times split_factor, and volumes divided by it, put all of a ticker's days on one split basis. A row's
    features read only bars through its decision day, which are all on or before data.cutoff.
    """
    lookback = getattr(data, "bars_lookback", LOOKBACK)
    if lookback < LOOKBACK:
        raise ValueError(f"The features read {LOOKBACK} sessions of bars, but this model's predict gets {lookback}: "
                         f"set its bars_lookback to at least {LOOKBACK}.")
    labels = data.labels().reset_index(drop=True)
    features = np.full((len(labels), len(FEATURES)), np.nan)
    if len(labels):
        bars = data.bars()
        days = pd.DatetimeIndex(np.sort(bars.date.unique()))
        tickers = pd.Index(labels.ticker.astype(str).unique(), dtype=object)
        bar_codes = tickers.get_indexer(bars.ticker.astype(str))
        kept = np.flatnonzero(bar_codes >= 0)
        kept = kept[np.argsort(bar_codes[kept], kind="stable")]  # Grouped by ticker, so a chunk is a slice.
        bar_codes, bar_rows = bar_codes[kept], days.get_indexer(pd.DatetimeIndex(bars.date.to_numpy()[kept]))
        factor = bars.split_factor.to_numpy(dtype=np.float64)[kept]
        inputs = [bars[name].to_numpy(dtype=np.float64)[kept] * factor for name in PRICES]
        inputs.append(bars.volume.to_numpy(dtype=np.float64)[kept] / factor)
        rows, codes = days.get_indexer(labels.decision_date), tickers.get_indexer(labels.ticker.astype(str))
        for start in range(0, len(tickers), _CHUNK):
            stop = min(start + _CHUNK, len(tickers))
            lo, hi = np.searchsorted(bar_codes, [start, stop])
            arrays = compute(*_scatter(bar_rows[lo:hi], bar_codes[lo:hi] - start, [x[lo:hi] for x in inputs],
                                       (len(days), stop - start)))
            selected = (codes >= start) & (codes < stop)
            for j, name in enumerate(FEATURES):
                features[selected, j] = arrays[name][rows[selected], codes[selected] - start]
    return labels.join(pd.DataFrame(features, columns=FEATURES, index=labels.index))
