"""The standard frame shared by both streams, and the manifest of the stream B series."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


COLUMNS = ("timestamp", "value")
FREQ = "1h"  # all stream B series are hourly without gaps


def frame_from_series(s: pd.Series) -> pd.DataFrame:
    """Standard frame from a Series with a DatetimeIndex."""
    return pd.DataFrame({"timestamp": s.index, "value": s.to_numpy(dtype="float64")})


def series_frame(s: dict) -> pd.DataFrame:
    """Standard frame of one stream B series. `start` is the timestamp of the first hour (a nominal calendar
    for the synthetic series, UTC for ENTSO-E, the local wall clock for the seismic noise)."""
    stamps = pd.date_range(s["start"], periods=len(s["y"]), freq=FREQ)
    return pd.DataFrame({"timestamp": stamps, "value": np.asarray(s["y"], dtype="float64")})


def series_id(s: dict) -> str:
    """Unique name of a series: `source/series/scenario` (the synthetic scenarios share `series`)."""
    return f"{s['source']}/{s['series']}/{s['scenario']}"


def series_table(series: list[dict]) -> pd.DataFrame:
    """One manifest row per series: `id`, labels, length, start and the deployment / change points both as
    positions (`t_deploy`, `t_change`, `t_stop`) and as timestamps (`deploy_at`, `change_at`)."""
    rows = []
    for s in series:
        start = pd.Timestamp(s["start"])
        t_chg = s.get("t_change")
        rows.append(dict(id=series_id(s), source=s["source"], series=s["series"], scenario=s["scenario"],
                         client=s["client"], seed=s["seed"], magnitude=s["magnitude"], n=len(s["y"]), start=start,
                         t_deploy=s["t_deploy"], t_change=t_chg, t_stop=s.get("t_stop", len(s["y"])),
                         deploy_at=start + pd.Timedelta(hours=int(s["t_deploy"])),
                         change_at=start + pd.Timedelta(hours=int(t_chg)) if t_chg is not None else pd.NaT))
    return pd.DataFrame(rows)


def series_file(s: dict) -> str:
    return series_id(s).replace("/", "__") + ".csv"


def write_series(series: list[dict], out_dir: Path) -> pd.DataFrame:
    """Write every series as a standard-frame CSV and the manifest as `manifest.csv` (with a `file` column)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = series_table(series)
    table.insert(1, "file", [series_file(s) for s in series])
    for s, file in zip(series, table["file"]):
        series_frame(s).to_csv(out_dir / file, index=False)
    table.to_csv(out_dir / "manifest.csv", index=False)
    return table
