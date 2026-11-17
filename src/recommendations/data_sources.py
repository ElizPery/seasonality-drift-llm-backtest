"""Real series for stream B: national electricity load (ENTSO-E via Open Power System Data) and the seismic
noise level (Lecocq et al., 2020), both with the known date of the March 2020 measures as the change.
The loading itself lives in `src.data`; here the series get the feature table and the experiment labels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path

from src.data.entsoe import ENTSOE_DEPLOY, ENTSOE_END, ENTSOE_EVENTS, ENTSOE_START, ENTSOE_TZ, load_entsoe
from src.data.seismic_noise import SEISMIC_DEPLOY, SEISMIC_END, SEISMIC_STATIONS, load_station, station_table
from src.recommendations.forecast import BASE_FEATURES, YEARLY_FEATURES, calendar_from_index, feature_table

EXPORT_DIR = Path("data/processed/series")  # written by `python -m src.data.export`


# ------------------------------------------------------------------ real series (ENTSO-E, events of March 2020)
def entsoe_series(path: str, countries=tuple(ENTSOE_EVENTS), start: str = ENTSOE_START, deploy: str = ENTSOE_DEPLOY,
                  end: str = ENTSOE_END) -> list[dict]:
    df = load_entsoe(path, countries, start, end)
    t_dep = int(df.index.get_indexer([pd.Timestamp(deploy, tz="UTC")])[0])
    out = []
    for c in countries:
        y = df[c].to_numpy(dtype=float)
        local = df.index.tz_convert(ENTSOE_TZ[c])
        F = feature_table(y, local.hour.to_numpy(), local.dayofweek.to_numpy(), local.dayofyear.to_numpy())
        ev = pd.Timestamp(ENTSOE_EVENTS[c], tz="UTC")
        t_chg = int(df.index.get_indexer([ev])[0]) if df.index[0] <= ev <= df.index[-1] else None
        out.append(dict(source="entsoe", series=c, client=c, seed=-1, scenario="lockdown_2020", magnitude=np.nan,
                        y=y, F=F, t_deploy=t_dep, t_change=t_chg, t_stop=len(y), base_features=BASE_FEATURES + YEARLY_FEATURES,
                        start=str(df.index[0])))
    return out


# ------------------------------------------------------------------ real series (seismic noise, lockdowns of 2020)
def seismic_series(folder: str, stations=SEISMIC_STATIONS, deploy: str = SEISMIC_DEPLOY, end: str = SEISMIC_END) -> list[dict]:
    """Hourly log10 noise level on the local wall clock (see `src.data.seismic_noise.load_station`); the change
    is the date of the first measures at the station (Date_LD1 of Table S1)."""
    meta = station_table(folder)
    out = []
    for code in stations:
        h = load_station(folder, code, meta.loc[code, "TimeZone"], deploy, end)
        y = h.to_numpy(dtype=float)
        F = feature_table(y, h.index.hour.to_numpy(), h.index.dayofweek.to_numpy())
        t_dep = int(h.index.get_indexer([pd.Timestamp(deploy)])[0])
        t_chg = int(h.index.get_indexer([pd.to_datetime(meta.loc[code, "Date_LD1"], dayfirst=True)])[0])
        if t_dep < 0 or t_chg < 0:
            raise ValueError(f"{code}: deployment or change date outside {h.index[0]} .. {h.index[-1]}")
        out.append(dict(source="seismic", series=code, client=meta.loc[code, "City"], seed=-1, scenario="lockdown_2020",
                        magnitude=np.nan, y=y, F=F, t_deploy=t_dep, t_change=t_chg, t_stop=len(y), base_features=BASE_FEATURES,
                        start=str(h.index[0])))
    return out


# ------------------------------------------------------------------ series exported as standard frames (src.data.export)
def read_manifest(folder: str | Path = EXPORT_DIR) -> pd.DataFrame:
    return pd.read_csv(Path(folder) / "manifest.csv")


def series_from_frame(row, frame: pd.DataFrame) -> dict:
    """The series of the experiments from its standard frame and manifest row: the feature table is rebuilt from the
    values with the calendar of the source (ENTSO-E: UTC stamps, features on the local clock of the country, yearly
    terms; seismic noise: the local wall clock; synthetic: the nominal calendar from Monday 00:00), so the result is
    the same series that `entsoe_series`, `seismic_series` and `synthetic_series` give."""
    y = frame["value"].to_numpy(dtype=float)
    base = BASE_FEATURES
    if row["source"] == "entsoe":
        local = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(ENTSOE_TZ[row["series"]])
        F = feature_table(y, local.dt.hour.to_numpy(), local.dt.dayofweek.to_numpy(), local.dt.dayofyear.to_numpy())
        base = BASE_FEATURES + YEARLY_FEATURES
    elif row["source"] == "seismic":
        stamps = pd.to_datetime(frame["timestamp"])
        F = feature_table(y, stamps.dt.hour.to_numpy(), stamps.dt.dayofweek.to_numpy())
    else:
        hour, dow = calendar_from_index(len(y))
        F = feature_table(y, hour, dow)
    t_chg = row["t_change"]
    return dict(source=row["source"], series=row["series"], client=row["client"], seed=int(row["seed"]),
                scenario=row["scenario"], magnitude=float(row["magnitude"]), y=y, F=F, t_deploy=int(row["t_deploy"]),
                t_change=None if pd.isna(t_chg) else int(t_chg), t_stop=int(row["t_stop"]), base_features=base,
                start=str(row["start"]))


def read_series(series_id: str, folder: str | Path = EXPORT_DIR, manifest: pd.DataFrame | None = None) -> dict:
    """One exported series by its id (`source/series/scenario`)."""
    manifest = read_manifest(folder) if manifest is None else manifest
    rows = manifest[manifest["id"] == series_id]
    if rows.empty:
        raise KeyError(f"no series {series_id!r} in {folder}")
    row = rows.iloc[0]
    frame = pd.read_csv(Path(folder) / row["file"], float_precision="round_trip")
    return series_from_frame(row, frame)
