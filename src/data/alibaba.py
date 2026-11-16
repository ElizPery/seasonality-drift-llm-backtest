"""Alibaba Cluster Trace 2018, the pre-aggregated 300-second slice from Zenodo (zenodo.org/records/14564935,
`machine_usage_days_1_to_8_grouped_300_seconds.csv`): 8 days of machine usage every 5 minutes."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np


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

def load_alibaba_batch(path: str | Path = ALIBABA_FILE, n_series: int = 5, seed: int = 42):
    """Loads N distinct time series from the Alibaba cluster trace dataset
    by extracting different resource utilization metrics.
    """
    file_path = Path(path)
    if not file_path.exists():
        # Fallback if working directory is root instead of src/ or data/
        raise FileNotFoundError(f"Alibaba trace file not found at {path}. Place 'alibaba_cluster_trace.csv' there.")

    df_raw = pd.read_csv(file_path)
    
    # Available numeric metric columns in the trace
    metric_cols = [col for col in df_raw.columns if col in [
        'net_in', 'mem_util_percent'
    ]]
    if not metric_cols:
        metric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

    batch_dict = {}
    period = 288  # e.g., 288 steps per day for 5-minute intervals (24 * 60 / 5)

    # Generate multiple series by mapping each metric column (and optionally sub-segments)
    series_counter = 0
    for col in metric_cols:
        if series_counter >= n_series:
            break
            
        # Construct standard format: [timestamp, value]
        df_sub = pd.DataFrame()
        df_sub["timestamp"] = pd.date_range(start="2018-01-01", periods=len(df_raw), freq="5min")
        df_sub["value"] = df_raw[col].astype("float64")
        
        series_id = f"alibaba_{col}"
        batch_dict[series_id] = (df_sub, period)
        series_counter += 1

    # If n_series requests more than available metrics, create segmented/resampled variants
    if n_series > len(metric_cols):
        raise ValueError(f"Requested {n_series} series, but Alibaba only has {len(metric_cols)} metric columns.")

    return batch_dict
