"""Real series for stream B: national electricity load (ENTSO-E via Open Power System Data) and the seismic
noise level (Lecocq et al., 2020), both with the known date of the March 2020 measures as the change.
The loading itself lives in `src.data`; here the series get the feature table and the experiment labels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.entsoe import ENTSOE_DEPLOY, ENTSOE_END, ENTSOE_EVENTS, ENTSOE_START, ENTSOE_TZ, load_entsoe
from src.data.seismic_noise import SEISMIC_DEPLOY, SEISMIC_END, SEISMIC_STATIONS, load_station, station_table
from src.recommendations.forecast import BASE_FEATURES, YEARLY_FEATURES, feature_table


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
