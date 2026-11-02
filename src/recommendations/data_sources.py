"""Real series for stream B: national electricity load (ENTSO-E via Open Power System Data) and the seismic
noise level (Lecocq et al., 2020), both with the known date of the March 2020 measures as the change."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.recommendations.forecast import BASE_FEATURES, YEARLY_FEATURES, feature_table


# ------------------------------------------------------------------ real series (ENTSO-E, events of March 2020)
ENTSOE_EVENTS = {"IT": "2020-03-09", "PL": "2020-03-13", "ES": "2020-03-14", "NL": "2020-03-15",
                 "PT": "2020-03-16", "FR": "2020-03-17", "DE": "2020-03-22", "GB_GBN": "2020-03-23"}
ENTSOE_TZ = {"IT": "Europe/Rome", "PL": "Europe/Warsaw", "ES": "Europe/Madrid", "NL": "Europe/Amsterdam",
             "PT": "Europe/Lisbon", "FR": "Europe/Paris", "DE": "Europe/Berlin", "GB_GBN": "Europe/London"}
ENTSOE_START, ENTSOE_DEPLOY, ENTSOE_END = "2017-01-02", "2020-01-06", "2020-06-28 23:00"


def entsoe_series(path: str, countries=tuple(ENTSOE_EVENTS), start: str = ENTSOE_START, deploy: str = ENTSOE_DEPLOY,
                  end: str = ENTSOE_END) -> list[dict]:
    cols = {c: f"{c}_load_actual_entsoe_transparency" for c in countries}
    usecols = ["utc_timestamp"] + list(cols.values())
    head = pd.read_csv(path, nrows=0).columns
    if "PT_load_MW" in head:  # the 8-country extract shared with the team
        cols = {c: f"{c}_load_MW" for c in countries}
        usecols = ["utc_timestamp"] + list(cols.values())
    df = pd.read_csv(path, usecols=usecols, parse_dates=["utc_timestamp"]).set_index("utc_timestamp")
    df = df.loc[start:end]
    t_dep = int(df.index.get_indexer([pd.Timestamp(deploy, tz="UTC")])[0])
    out = []
    for c, col in cols.items():
        y = df[col].interpolate(limit_direction="both").to_numpy(dtype=float)
        local = df.index.tz_convert(ENTSOE_TZ[c])
        F = feature_table(y, local.hour.to_numpy(), local.dayofweek.to_numpy(), local.dayofyear.to_numpy())
        ev = pd.Timestamp(ENTSOE_EVENTS[c], tz="UTC")
        t_chg = int(df.index.get_indexer([ev])[0]) if df.index[0] <= ev <= df.index[-1] else None
        out.append(dict(source="entsoe", series=c, client=c, seed=-1, scenario="lockdown_2020", magnitude=np.nan,
                        y=y, F=F, t_deploy=t_dep, t_change=t_chg, t_stop=len(y), base_features=BASE_FEATURES + YEARLY_FEATURES))
    return out


# ------------------------------------------------------------------ real series (seismic noise, lockdowns of 2020)
# Displacement RMS every 15 minutes from the companion data of Lecocq et al. (2020), Science 369(6509):1338-1343
# (github.com/ThomasLecocq/2020_Science_GlobalQuieting). The 4-14 Hz band is dominated by human activity
# (traffic, industry). The change is the date of the first measures at each station from the paper's Table S1.
SEISMIC_STATIONS = ("BE.UCC", "IV.MILN", "IV.FIR", "FR.CURIE", "GB.EDI", "NL.ALK1", "GE.STU", "GR.BUG", "CA.ICJA",
                    "CH.SZUZ", "OE.VIE", "LD.CPNY", "AM.R091F", "AM.R9F13", "AM.S0C34")
SEISMIC_BAND = "4.0-14.0"
SEISMIC_DEPLOY, SEISMIC_END = "2020-02-03", "2020-04-26 23:00"


def seismic_series(folder: str, stations=SEISMIC_STATIONS, deploy: str = SEISMIC_DEPLOY, end: str = SEISMIC_END) -> list[dict]:
    """Hourly log10 noise level (nm) on the local wall clock, so the daily cycle of human activity does not
    move by an hour at the DST switch. Series start on the first Monday; gaps up to 6 h are interpolated,
    longer ones are filled with the hour-of-week median of the training part."""
    folder = Path(folder)
    meta = pd.read_csv(folder / "lockdown_stations.csv", encoding="latin-1", dtype=str)
    meta = meta.drop_duplicates("Station_Code").set_index("Station_Code")
    out = []
    for code in stations:
        s = pd.read_csv(next(folder.glob(code + ".*.csv")), index_col=0, parse_dates=True)[SEISMIC_BAND].sort_index()
        # gaps are stored as tiny placeholder values (about 1e-157 m, and 1.6e-13 m at CH.SZUZ); real 4-14 Hz
        # levels at these stations are above 7e-11 m
        s = s[~s.index.duplicated() & (s > 1e-11)]
        s.index = s.index.tz_localize("UTC").tz_convert(meta.loc[code, "TimeZone"]).tz_localize(None)
        h = np.log10(s * 1e9).resample("1h").median()
        first = h.index[0].normalize() + pd.Timedelta(days=1)
        first += pd.Timedelta(days=(7 - first.dayofweek) % 7)
        h = h[first:end].interpolate(limit=6, limit_area="inside")
        train = h[: pd.Timestamp(deploy) - pd.Timedelta(hours=1)]
        profile = train.groupby([train.index.dayofweek, train.index.hour]).median()
        fill = profile.reindex(pd.MultiIndex.from_arrays([h.index.dayofweek, h.index.hour])).to_numpy()
        y = h.fillna(pd.Series(fill, index=h.index)).to_numpy(dtype=float)
        F = feature_table(y, h.index.hour.to_numpy(), h.index.dayofweek.to_numpy())
        t_dep = int(h.index.get_indexer([pd.Timestamp(deploy)])[0])
        t_chg = int(h.index.get_indexer([pd.to_datetime(meta.loc[code, "Date_LD1"], dayfirst=True)])[0])
        if t_dep < 0 or t_chg < 0:
            raise ValueError(f"{code}: deployment or change date outside {h.index[0]} .. {h.index[-1]}")
        out.append(dict(source="seismic", series=code, client=meta.loc[code, "City"], seed=-1, scenario="lockdown_2020",
                        magnitude=np.nan, y=y, F=F, t_deploy=t_dep, t_change=t_chg, t_stop=len(y), base_features=BASE_FEATURES))
    return out
