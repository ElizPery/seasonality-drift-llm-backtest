"""Base forecasting model for stream B (RQ3): a LightGBM day-ahead forecaster on lag and calendar
features, the weekly seasonal naive benchmark and MASE.

Forecast protocol: at every origin (00:00 of a day) the model predicts the next 24 hours. Every feature
of hour t uses values at t-24 and earlier, so one model forecasts the whole next day (direct strategy)
and nothing from the forecast day leaks into its own features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WEEK = 168
DAY = 24
LAGS = (24, 48, 168, 336)
BASE_FEATURES = ("hour", "dow", "lag_24", "lag_48", "lag_168", "lag_336", "mean24_lag24", "mean168_lag24")
YEARLY_FEATURES = ("doy_sin", "doy_cos")
PROFILE_FEATURE = "profile_4w"

# native LightGBM parameters; `rounds` is the number of boosting rounds.
# Chosen with `run_backtest --tune` on validation windows (synthetic control series and ENTSO-E 2019).
LGBM_PARAMS = dict(
    objective="regression", rounds=200, learning_rate=0.05, num_leaves=31, min_data_in_leaf=20,
    bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.9, lambda_l2=1.0, max_bin=127,
    seed=0, num_threads=1, deterministic=True, verbose=-1,
)


def _shift(y: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(y), np.nan)
    out[k:] = y[:-k]
    return out


def _lagged_mean(y: np.ndarray, width: int, lag: int) -> np.ndarray:
    """Mean of y[t-lag-width+1 .. t-lag] for every t (NaN where the window is incomplete)."""
    c = np.concatenate([[0.0], np.cumsum(y)])
    t = np.arange(len(y))
    hi = t - lag + 1
    lo = hi - width
    out = np.full(len(y), np.nan)
    ok = lo >= 0
    out[ok] = (c[hi[ok]] - c[lo[ok]]) / width
    return out


def feature_table(y: np.ndarray, hour: np.ndarray, dow: np.ndarray, doy: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """All candidate features for every hour. `y` must have no gaps (interpolate first)."""
    F = {"hour": hour.astype(float), "dow": dow.astype(float)}
    for lag in LAGS:
        F[f"lag_{lag}"] = _shift(y, lag)
    F["mean24_lag24"] = _lagged_mean(y, 24, 24)
    F["mean168_lag24"] = _lagged_mean(y, 168, 24)
    # seasonal feature: mean of the same hour of the week over the last 4 weeks
    F[PROFILE_FEATURE] = np.mean([_shift(y, k * WEEK) for k in (1, 2, 3, 4)], axis=0)
    if doy is not None:
        F["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        F["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return F


def calendar_from_index(n: int, start_hour_of_week: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Hour of day and day of week for a series that starts at the given hour of the week (0 = Monday 00:00)."""
    how = (np.arange(n) + start_hour_of_week) % WEEK
    return how % DAY, how // DAY


@dataclass
class Model:
    features: tuple[str, ...]
    regressor: object = None
    bias: float = 0.0  # additive level correction set by the `recalibrate` action

    def predict(self, F: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
        X = np.column_stack([F[f][rows] for f in self.features])
        # one thread: experiments run many series in parallel processes
        return self.regressor.predict(X, num_threads=1) + self.bias


def fit_model(F: dict[str, np.ndarray], y: np.ndarray, rows: np.ndarray, features: tuple[str, ...],
              weights: np.ndarray | None = None, params: dict | None = None) -> Model:
    import lightgbm as lgb

    X = np.column_stack([F[f][rows] for f in features])
    target = y[rows]
    ok = np.isfinite(X).all(axis=1) & np.isfinite(target)
    p = dict(params or LGBM_PARAMS)
    rounds = p.pop("rounds")
    data = lgb.Dataset(X[ok], target[ok], weight=None if weights is None else weights[ok],
                       params={"max_bin": p["max_bin"], "verbose": -1})
    booster = lgb.train(p, data, num_boost_round=rounds)
    return Model(features=tuple(features), regressor=booster)


def mase_scale(y: np.ndarray, m: int = WEEK) -> float:
    """In-sample mean absolute error of the seasonal naive forecast y[t-m] (Hyndman, Koehler 2006)."""
    d = np.abs(y[m:] - y[:-m])
    return float(np.nanmean(d))


def seasonal_naive(y: np.ndarray, rows: np.ndarray, m: int = WEEK) -> np.ndarray:
    return y[rows - m]
