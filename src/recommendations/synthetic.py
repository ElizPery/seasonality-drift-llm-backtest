"""Synthetic series for stream B, built from UCI ElectricityLoadDiagrams 2011-2014 clients.

Raw file: LD2011_2014.txt, 370 clients, 15-minute kW values, timestamps mark the end of each interval.
Known quirks (from the dataset description): clients that joined later have zeros before their start;
on the March DST day the 01:00-02:00 values are zero for all clients; on the October DST day the
01:00-02:00 values hold two hours of consumption.

Control series without changes: a real hourly series is decomposed with MSTL (periods 24 and 168). The
weekly seasonal profile is averaged over the weeks, the trend is replaced by a constant level, the
residuals are cleaned of outliers and week-to-week variance changes, and then resampled with a seasonal
block bootstrap (daily blocks aligned to the weekday). The result keeps the real seasonal shape and the
noise structure inside a week but has no changes by construction. Changes of known type, moment and
magnitude are then injected into the control series.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL

from src.recommendations.forecast import BASE_FEATURES, DAY, WEEK, calendar_from_index, feature_table


RAW_FILE = Path("data/raw/LD2011_2014.txt")
HOURLY_FILE = Path("data/processed/uci_hourly.parquet")


def _last_sunday(year: int, month: int) -> pd.Timestamp:
    last_day = calendar.monthrange(year, month)[1]
    d = pd.Timestamp(year=year, month=month, day=last_day)
    return d - pd.Timedelta(days=(d.weekday() - 6) % 7)


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",", index_col=0, parse_dates=True)
    df.index.name = "time"
    # timestamps mark the end of the 15-minute interval; move to the start
    df.index = df.index - pd.Timedelta(minutes=15)
    df = df.astype("float32")
    return df


def fix_dst(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    years = sorted(set(df.index.year))
    for y in years:
        march = _last_sunday(y, 3)
        hour = (df.index >= march + pd.Timedelta(hours=1)) & (df.index < march + pd.Timedelta(hours=2))
        df.loc[hour] = np.nan
        october = _last_sunday(y, 10)
        hour = (df.index >= october + pd.Timedelta(hours=1)) & (df.index < october + pd.Timedelta(hours=2))
        df.loc[hour] = df.loc[hour] / 2.0
    return df


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    hourly = df.resample("1h").mean()
    hourly = hourly.interpolate(limit=3, limit_direction="both")
    return hourly


def prepare_hourly(raw: Path = RAW_FILE, out: Path = HOURLY_FILE) -> pd.DataFrame:
    hourly = to_hourly(fix_dst(load_raw(raw)))
    out.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(out)
    return hourly


def load_hourly(path: Path = HOURLY_FILE) -> pd.DataFrame:
    if not path.exists():
        return prepare_hourly(out=path)
    return pd.read_parquet(path)


def select_clients(
    hourly: pd.DataFrame,
    n: int,
    start_before: str = "2013-01-01",
    check_window: tuple[str, str] = ("2013-01-01", "2013-12-31"),
    max_zero_frac: float = 0.005,
) -> list[str]:
    """Clients that were already active before `start_before` and have almost
    no zero readings inside `check_window`. Sorted by client id."""
    chosen = []
    win = hourly.loc[check_window[0] : check_window[1]]
    for col in hourly.columns:
        s = hourly[col]
        nz = s[s > 0]
        if nz.empty or nz.index[0] >= pd.Timestamp(start_before):
            continue
        w = win[col]
        if (w <= 0).mean() > max_zero_frac or w.isna().any():
            continue
        chosen.append(col)
        if len(chosen) == n:
            break
    return chosen


# Autumn 2013: no Portuguese public holidays (Oct 5 and Nov 1 were suspended
# in 2013-2015), 12 full weeks starting on a Monday.
SOURCE_WINDOW = ("2013-09-02 00:00", "2013-11-24 23:00")


def source_series(hourly: pd.DataFrame, client: str, window=SOURCE_WINDOW) -> pd.Series:
    s = hourly.loc[window[0] : window[1], client].astype(float)
    assert s.index[0].weekday() == 0, "source window must start on Monday"
    assert len(s) % WEEK == 0, "source window must hold whole weeks"
    return s


@dataclass
class SeasonalModel:
    client: str
    profile: np.ndarray  # length 168, hour-of-week seasonal profile (mean zero)
    level: float  # constant level replacing the trend
    resid: np.ndarray  # residuals of the source series
    sigma: float  # std of residuals

    @property
    def amplitude(self) -> float:
        return float(self.profile.max() - self.profile.min())


def clean_residuals(resid: np.ndarray, winsor: float = 4.0) -> np.ndarray:
    """Make the residual process free of changes by construction: clip gross
    outliers at `winsor` robust sigmas and give every week the same spread
    (the median weekly standard deviation). Autocorrelation and the
    hour-of-week variance pattern inside a week are kept."""
    r = resid - np.median(resid)
    robust_sigma = 1.4826 * np.median(np.abs(r))
    r = np.clip(r, -winsor * robust_sigma, winsor * robust_sigma)
    n_weeks = len(r) // WEEK
    weeks = r[: n_weeks * WEEK].reshape(n_weeks, WEEK)
    weeks = weeks - weeks.mean(axis=1, keepdims=True)
    sd = weeks.std(axis=1, ddof=1)
    weeks = weeks * (np.median(sd) / sd)[:, None]
    out = weeks.ravel()
    return out - out.mean()


def fit_seasonal_model(y: pd.Series, client: str = "") -> SeasonalModel:
    values = y.to_numpy(dtype=float)
    res = MSTL(values, periods=(DAY, WEEK), stl_kwargs={"robust": True}).fit()
    seasonal = np.asarray(res.seasonal).sum(axis=1) if np.ndim(res.seasonal) == 2 else np.asarray(res.seasonal)
    trend = np.asarray(res.trend)
    resid = clean_residuals(values - seasonal - trend)
    n_weeks = len(values) // WEEK
    profile = seasonal[: n_weeks * WEEK].reshape(n_weeks, WEEK).mean(axis=0)
    profile = profile - profile.mean()
    level = float(np.median(trend))
    return SeasonalModel(client=client, profile=profile, level=level, resid=resid, sigma=float(resid.std(ddof=1)))


def block_bootstrap(resid: np.ndarray, n: int, rng: np.random.Generator, start_day: int = 0) -> np.ndarray:
    """Seasonal block bootstrap with daily blocks aligned to the day of the
    week: day d of the output is a randomly chosen source day with the same
    weekday (Politis 2001; Dudek et al. 2014). Keeps intra-day autocorrelation
    and the hour-of-week variance pattern, so whole weeks of the output have
    the same composition."""
    n_weeks = len(resid) // WEEK
    n_days = int(np.ceil(n / DAY))
    weeks = rng.integers(0, n_weeks, size=n_days)
    out = np.concatenate([resid[w * WEEK + ((d + start_day) % 7) * DAY :][:DAY] for d, w in enumerate(weeks)])
    return out[:n]


def make_control(model: SeasonalModel, n: int, seed: int, block_len: int = DAY) -> np.ndarray:
    """Series of length `n` starting at Monday 00:00 with no changes."""
    rng = np.random.default_rng(seed)
    how = np.arange(n) % WEEK
    return model.level + model.profile[how] + block_bootstrap(model.resid, n, rng)


@dataclass
class Change:
    kind: str  # level_shift | trend_ramp | variance
    magnitude: float  # in units of residual sigma (variance: multiplier of sigma)
    t0: int  # index of the first changed point


def inject(x: np.ndarray, model: SeasonalModel, change: Change, ramp_len: int = WEEK, seed: int = 0) -> np.ndarray:
    y = x.copy()
    t = np.arange(len(y))
    after = t >= change.t0
    delta = change.magnitude * model.sigma
    if change.kind == "level_shift":
        y[after] += delta
    elif change.kind == "trend_ramp":
        ramp = np.clip((t - change.t0 + 1) / ramp_len, 0, 1)
        y += delta * ramp * after
    elif change.kind == "variance":
        rng = np.random.default_rng(seed + 10_000)
        extra = block_bootstrap(model.resid, int(after.sum()), rng, start_day=change.t0 // DAY)
        y[after] += extra * np.sqrt(max(change.magnitude**2 - 1.0, 0.0))
    else:
        raise ValueError(change.kind)
    return y


def fit_models(n_clients: int = 20) -> list[SeasonalModel]:
    """Seasonal models of the first `n_clients` clients that pass `select_clients`, on the autumn 2013 window."""
    hourly = load_hourly()
    return [fit_seasonal_model(source_series(hourly, c), client=c) for c in select_clients(hourly, n_clients)]


# ------------------------------------------------------------------ synthetic series (UCI clients)
SCENARIOS = (("none", 0.0), ("level_shift", 2.0), ("trend_ramp", 2.0), ("variance", 2.0))
SYN_WEEKS, SYN_DEPLOY_WEEK, SYN_CHANGE_WEEK = 24, 10, 13


def synthetic_series(models, seeds, scenarios=SCENARIOS) -> list[dict]:
    n = SYN_WEEKS * WEEK
    hour, dow = calendar_from_index(n)
    out = []
    for m in models:
        for seed in seeds:
            control = make_control(m, n, seed + 500)  # seeds apart from the detector experiment
            for kind, mag in scenarios:
                y = control if kind == "none" else inject(control, m, Change(kind, mag, SYN_CHANGE_WEEK * WEEK), seed=seed)
                out.append(dict(source="synthetic", series=f"{m.client}/s{seed}", client=m.client, seed=seed, scenario=kind,
                                magnitude=mag, y=y, F=feature_table(y, hour, dow), t_deploy=SYN_DEPLOY_WEEK * WEEK,
                                t_change=SYN_CHANGE_WEEK * WEEK, base_features=BASE_FEATURES))
    return out
