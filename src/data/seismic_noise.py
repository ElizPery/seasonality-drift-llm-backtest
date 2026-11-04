"""Seismic noise level from the companion data of Lecocq et al. (2020), Science 369(6509):1338-1343
(github.com/ThomasLecocq/2020_Science_GlobalQuieting): displacement RMS every 15 minutes per station. The
4-14 Hz band is dominated by human activity (traffic, industry). The change is the date of the first measures
at each station from the paper's Table S1 (`lockdown_stations.csv`)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.frames import frame_from_series


SEISMIC_DIR = Path("data/raw/seismic_noise")
SEISMIC_STATIONS = ("BE.UCC", "IV.MILN", "IV.FIR", "FR.CURIE", "GB.EDI", "NL.ALK1", "GE.STU", "GR.BUG", "CA.ICJA",
                    "CH.SZUZ", "OE.VIE", "LD.CPNY", "AM.R091F", "AM.R9F13", "AM.S0C34")
SEISMIC_BAND = "4.0-14.0"
SEISMIC_DEPLOY, SEISMIC_END = "2020-02-03", "2020-04-26 23:00"


def station_table(folder: str | Path = SEISMIC_DIR) -> pd.DataFrame:
    """Table S1 of the paper indexed by station code: city, time zone, dates of the measures."""
    meta = pd.read_csv(Path(folder) / "lockdown_stations.csv", encoding="latin-1", dtype=str)
    return meta.drop_duplicates("Station_Code").set_index("Station_Code")


def load_station(folder: str | Path, code: str, tz: str, deploy: str = SEISMIC_DEPLOY, end: str = SEISMIC_END,
                 band: str = SEISMIC_BAND) -> pd.Series:
    """Hourly log10 noise level (nm) on the local wall clock, so the daily cycle of human activity does not
    move by an hour at the DST switch. The series starts on the first Monday; gaps up to 6 h are interpolated,
    longer ones are filled with the hour-of-week median of the part before `deploy`."""
    folder = Path(folder)
    s = pd.read_csv(next(folder.glob(code + ".*.csv")), index_col=0, parse_dates=True)[band].sort_index()
    # gaps are stored as tiny placeholder values (about 1e-157 m, and 1.6e-13 m at CH.SZUZ); real 4-14 Hz
    # levels at these stations are above 7e-11 m
    s = s[~s.index.duplicated() & (s > 1e-11)]
    s.index = s.index.tz_localize("UTC").tz_convert(tz).tz_localize(None)
    h = np.log10(s * 1e9).resample("1h").median()
    first = h.index[0].normalize() + pd.Timedelta(days=1)
    first += pd.Timedelta(days=(7 - first.dayofweek) % 7)
    h = h[first:end].interpolate(limit=6, limit_area="inside")
    train = h[: pd.Timestamp(deploy) - pd.Timedelta(hours=1)]
    profile = train.groupby([train.index.dayofweek, train.index.hour]).median()
    fill = profile.reindex(pd.MultiIndex.from_arrays([h.index.dayofweek, h.index.hour])).to_numpy()
    return h.fillna(pd.Series(fill, index=h.index)).astype("float64")


def station_frame(folder: str | Path, code: str, meta: pd.DataFrame | None = None, **kwargs) -> pd.DataFrame:
    """Standard frame of one station."""
    meta = station_table(folder) if meta is None else meta
    return frame_from_series(load_station(folder, code, meta.loc[code, "TimeZone"], **kwargs))
