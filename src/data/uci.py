"""UCI ElectricityLoadDiagrams 2011-2014: 370 clients, 15-minute kW values, timestamps mark the end of each interval.

Known quirks (from the dataset description): clients that joined later have zeros before their start; on the
March DST day the 01:00-02:00 values are zero for all clients; on the October DST day the 01:00-02:00 values
hold two hours of consumption. `load_hourly` returns the cleaned hourly table (kW, interval start, DST hours
fixed, gaps up to 3 h interpolated) and caches it as parquet.
"""
from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.frames import frame_from_series


RAW_FILE = Path("data/raw/LD2011_2014.txt")
HOURLY_FILE = Path("data/processed/uci_hourly.parquet")
WEEK = 168


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


def load_hourly(path: Path = HOURLY_FILE, raw: Path = RAW_FILE) -> pd.DataFrame:
    if not path.exists():
        return prepare_hourly(raw, out=path)
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


def client_frame(client: str, freq: str = "1h", hourly: pd.DataFrame | None = None) -> pd.DataFrame:
    """Standard frame of one client from the hourly table; a coarser `freq` averages the hours."""
    hourly = load_hourly() if hourly is None else hourly
    s = hourly[client].astype("float64")
    if freq and pd.Timedelta(pd.tseries.frequencies.to_offset(freq)) != pd.Timedelta(hours=1):
        s = s.resample(freq).mean()
    return frame_from_series(s)
