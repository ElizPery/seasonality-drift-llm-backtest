"""Series uploaded through the API: the way to run the loop of the service on data that are not part of the experiments.

An uploaded series is a standard frame (`timestamp`, `value`: hourly values without gaps) with the moment at which the
forecasting model goes to work (`deploy_at`). It is stored next to the exported series, in its own folder with its own
manifest, and gets the id `user/<name>/uploaded`; from then on a session plays it like any exported series. The
features use the hour and the weekday of the stamps as they are given, as for the seismic noise.

The detectors have no setting for such a series, so each of them is calibrated on the history of the series itself: on
the hours between `CONTROL_FROM` and the deployment, to the target of the experiments (one false alarm per 14 days).
Only values before the deployment are used. One series gives a short control period, so the setting is rougher than
the settings of the experiments, which are calibrated on many series of a source; a session can still be given its own
`detector_value`.
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from src.recommendations.alarms import DETECTORS, TARGET_DAYS, calibrate
from src.recommendations.backtester import VAL_WEEKS
from src.recommendations.forecast import BASE_FEATURES, DAY, WEEK, feature_table, mase_scale

SOURCE, SCENARIO = "user", "uploaded"
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
HOUR = pd.Timedelta(hours=1)
CONTROL_FROM = 5 * WEEK  # the detectors need the first weeks to warm up (4 weeks of profile for ADWIN)
MIN_HISTORY = 8 * WEEK  # before the deployment: the training data of the model and the control period of the detectors
MIN_AFTER = (VAL_WEEKS + 1) * WEEK  # after the deployment: the acceptance rule needs VAL_WEEKS of own forecasts
MAX_VALUES = 50_000  # more than five years of hourly data
MAX_CSV_BYTES = 8_000_000


def series_id(name: str) -> str:
    return f"{SOURCE}/{name}/{SCENARIO}"


def _position(stamp: str, start: pd.Timestamp, what: str) -> int:
    try:
        t = pd.Timestamp(stamp)
    except (ValueError, TypeError):
        raise ValueError(f"{what} {stamp!r} is not a timestamp") from None
    if (t.tzinfo is None) != (start.tzinfo is None):
        raise ValueError(f"{what} and the timestamps of the series must both have a time zone or both have none")
    hours = (t - start) / HOUR
    if hours != int(hours):
        raise ValueError(f"{what} {stamp!r} is not on the hourly grid of the series, which starts at {start}")
    return int(hours)


def build(name: str, start, values, deploy_at: str, change_at: str | None = None) -> dict:
    """The series dictionary of the backtest from uploaded hourly values; raises ValueError with the reason if the
    data cannot be used."""
    if not NAME.fullmatch(name or ""):
        raise ValueError("name: 1 to 64 letters, digits, '_', '-' or '.', starting with a letter or a digit")
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or len(y) > MAX_VALUES:
        raise ValueError(f"values: one list of at most {MAX_VALUES} hourly values")
    if not np.isfinite(y).all():
        raise ValueError("values must be finite hourly values without gaps (fill the gaps before the upload)")
    try:
        start = pd.Timestamp(start)
    except (ValueError, TypeError):
        raise ValueError(f"start {start!r} is not a timestamp") from None
    t_dep = _position(deploy_at, start, "deploy_at")
    if t_dep < MIN_HISTORY:
        raise ValueError(f"the series needs at least {MIN_HISTORY // WEEK} weeks of history before deploy_at "
                         f"({MIN_HISTORY} hourly values; it has {max(t_dep, 0)})")
    if len(y) - t_dep < MIN_AFTER:
        raise ValueError(f"the series needs at least {MIN_AFTER // WEEK} weeks from deploy_at on "
                         f"({MIN_AFTER} hourly values; it has {max(len(y) - t_dep, 0)})")
    t_chg = None
    if change_at is not None:
        t_chg = _position(change_at, start, "change_at")
        if not t_dep < t_chg < len(y):
            raise ValueError("change_at must be after deploy_at and inside the series")
    scale = mase_scale(y[:t_dep])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("the history before deploy_at repeats itself exactly from week to week: the error scale is zero")
    stamps = pd.date_range(start, periods=len(y), freq="1h")
    F = feature_table(y, stamps.hour.to_numpy(), stamps.dayofweek.to_numpy())
    return dict(source=SOURCE, series=name, client=name, seed=-1, scenario=SCENARIO, magnitude=np.nan, y=y, F=F,
                t_deploy=t_dep, t_change=t_chg, t_stop=len(y), base_features=BASE_FEATURES, start=str(start))


def read_csv(body: bytes) -> tuple[pd.Timestamp, np.ndarray]:
    """Start and values of a standard frame sent as CSV (columns `timestamp` and `value`, one row per hour)."""
    if len(body) > MAX_CSV_BYTES:
        raise ValueError(f"the file is larger than {MAX_CSV_BYTES // 1_000_000} MB")
    try:
        df = pd.read_csv(io.BytesIO(body), float_precision="round_trip")
    except Exception as e:  # any parser error is an error of the request
        raise ValueError(f"not a CSV file: {e}") from None
    if not {"timestamp", "value"} <= set(df.columns):
        raise ValueError("the CSV needs the columns `timestamp` and `value`")
    if len(df) < 2:
        raise ValueError("the CSV has fewer than two rows")
    try:
        stamps = pd.to_datetime(df["timestamp"])
        values = df["value"].to_numpy(dtype=float)
    except (ValueError, TypeError) as e:
        raise ValueError(f"cannot read the columns: {e}") from None
    if not pd.api.types.is_datetime64_any_dtype(stamps):
        raise ValueError("the timestamps have different UTC offsets: give them in UTC or without a time zone")
    steps = stamps.diff().iloc[1:]
    if not (steps == HOUR).all():
        bad = stamps[1:][(steps != HOUR).to_numpy()].iloc[0]
        raise ValueError(f"the rows must be one hour apart without gaps or repeats (see the row at {bad})")
    return stamps.iloc[0], values


def calibrate_on_history(s: dict, target_days: int = TARGET_DAYS) -> list[dict]:
    """Setting of every detector for an uploaded series: `alarms.calibrate` on the control period of the series itself,
    the hours from `CONTROL_FROM` to the deployment. The detectors see only the values before the deployment."""
    t_dep = s["t_deploy"]
    history = np.ascontiguousarray(s["y"][:t_dep])
    window = [(history, CONTROL_FROM, t_dep)]
    out = []
    for method, (_, param, _) in DETECTORS.items():
        value, rate = calibrate(method, window, target_days)
        out.append(dict(method=method, param=param, value=float(value), far_control_per_1000=round(rate * 1000, 3),
                        target_per_1000=round(1000 / (target_days * DAY), 3), control_hours=t_dep - CONTROL_FROM))
    return out
