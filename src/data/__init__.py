"""Data loading shared by both streams.

Every loader returns the same standard frame: a DataFrame with the columns `timestamp` (start of the
interval) and `value` (float64), one row per point in time order. Stream A runs its detectors on these
frames; stream B builds its experiment series (dicts with `y`, `F`, `t_deploy`, ...) on top of the same
loaders in `src/recommendations`, and `frames.series_frame` / `frames.series_table` turn those series back
into standard frames and manifest rows, so the detectors of stream A can run on them too.

Loaders: `uci` (ElectricityLoadDiagrams), `entsoe` (national load), `seismic_noise` (Lecocq et al. 2020),
`alibaba` (cluster trace 2018), `tcpd` (TCPDBench series with annotations), `iquique` (SeisBench traces).
`loader.DataLoader` keeps the class interface used by stream A scripts and notebooks.
"""
from src.data.frames import COLUMNS, frame_from_series, series_frame, series_table, write_series
from src.data.loader import DataLoader

__all__ = ["COLUMNS", "DataLoader", "frame_from_series", "series_frame", "series_table", "write_series"]
