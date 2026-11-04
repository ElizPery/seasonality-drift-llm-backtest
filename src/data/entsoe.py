"""National electricity load from the ENTSO-E transparency platform (hourly, UTC) as published by Open Power
System Data, with the date of the first March 2020 measures in each country as the known change."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.frames import frame_from_series


ENTSOE_FILE = Path("data/raw/opsd/time_series_60min_singleindex.csv")
ENTSOE_EVENTS = {"IT": "2020-03-09", "PL": "2020-03-13", "ES": "2020-03-14", "NL": "2020-03-15",
                 "PT": "2020-03-16", "FR": "2020-03-17", "DE": "2020-03-22", "GB_GBN": "2020-03-23"}
ENTSOE_TZ = {"IT": "Europe/Rome", "PL": "Europe/Warsaw", "ES": "Europe/Madrid", "NL": "Europe/Amsterdam",
             "PT": "Europe/Lisbon", "FR": "Europe/Paris", "DE": "Europe/Berlin", "GB_GBN": "Europe/London"}
ENTSOE_START, ENTSOE_DEPLOY, ENTSOE_END = "2017-01-02", "2020-01-06", "2020-06-28 23:00"


def load_entsoe(path: str | Path = ENTSOE_FILE, countries=tuple(ENTSOE_EVENTS), start: str = ENTSOE_START,
                end: str = ENTSOE_END) -> pd.DataFrame:
    """Hourly load in MW, one column per country code, UTC index, gaps interpolated."""
    cols = {c: f"{c}_load_actual_entsoe_transparency" for c in countries}
    head = pd.read_csv(path, nrows=0).columns
    if "PT_load_MW" in head:  # the 8-country extract shared with the team
        cols = {c: f"{c}_load_MW" for c in countries}
    usecols = ["utc_timestamp"] + list(cols.values())
    df = pd.read_csv(path, usecols=usecols, parse_dates=["utc_timestamp"]).set_index("utc_timestamp")
    df = df.loc[start:end].rename(columns={v: k for k, v in cols.items()})[list(countries)]
    return df.interpolate(limit_direction="both").astype("float64")


def country_frame(load: pd.DataFrame, country: str) -> pd.DataFrame:
    """Standard frame of one country from the table of `load_entsoe`."""
    return frame_from_series(load[country])
