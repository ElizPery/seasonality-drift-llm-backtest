"""Alibaba Cluster Trace 2018, the pre-aggregated 300-second slice from Zenodo (zenodo.org/records/14564935,
`machine_usage_days_1_to_8_grouped_300_seconds.csv`): 8 days of machine usage every 5 minutes."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ALIBABA_FILE = Path("data/raw/alibaba_cluster_trace.csv")


def load_alibaba_trace(path: str | Path = ALIBABA_FILE, metric: str = "cpu_util") -> pd.DataFrame:
    """Standard frame of one metric averaged over the machines at each timestamp."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Alibaba trace file not found at {path}. Place 'alibaba_cluster_trace.csv' there.")

    df = pd.read_csv(path)

    # Handle timestamps for pre-grouped CSVs
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"])
    else:
        # Fallback for 300-second (5-minute) grouped intervals
        df["timestamp"] = pd.date_range(start="2018-01-01", periods=len(df), freq="5min")

    # Select the target metric column (fallback to the first numeric column if not found)
    val_col = metric if metric in df.columns else (df.columns[1] if len(df.columns) > 1 else df.columns[0])

    ts_df = df.groupby("timestamp")[val_col].mean().reset_index().rename(columns={val_col: "value"})
    ts_df["value"] = ts_df["value"].astype("float64")
    return ts_df
