"""
Data Loader Module for Seasonality-Aware Drift Detection System.

This module handles loading, parsing, and standardizing time-series datasets:
1. Alibaba Cluster Trace (Infrastructure CPU/Memory load)
2. UCI ElectricityLoadDiagrams 2011-2014 (Energy consumption)

All datasets are returned as standardized pandas DataFrames with a unified format:
- 'timestamp': DatetimeIndex or RangeIndex
- 'value': float64 continuous target time-series signal
"""

import os
import json
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


class DataLoader:
    """Centralized Data Loader class for experimental time-series datasets."""

    def __init__(self, data_dir: Union[str, Path] = "data"):
        """
        Initialize DataLoader with project data paths.

        :param data_dir: Base directory path for storing raw and processed datasets.
        """
        self.base_dir = Path(data_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Alibaba Cluster Trace Loader
    # -------------------------------------------------------------------------
    def load_alibaba_standard_trace(
    self, file_path: Optional[Union[str, Path]] = None, metric: str = "cpu_util"
    ) -> pd.DataFrame:
        """
        Load time-series from standard Alibaba Cluster Trace 2018 (grouped 300-second format).
        
        :param file_path: Path to the grouped Alibaba trace CSV file.
        :param metric: Target metric column name (e.g., 'cpu_util', 'mem_util', or 'value').
        :return: Standardized DataFrame with columns ['timestamp', 'value'].
        """
        path = Path(file_path) if file_path else self.raw_dir / "alibaba_cluster_trace.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"[DataLoader] Alibaba standard trace file not found at {path}. "
                f"Please place 'alibaba_cluster_trace.csv' into '{self.raw_dir}' directory."
            )

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

        # Group by timestamp to generate a clean global infrastructure time-series
        ts_df = (
            df.groupby("timestamp")[val_col]
            .mean()
            .reset_index()
            .rename(columns={val_col: "value"})
        )
        ts_df["value"] = ts_df["value"].astype("float64")

        return ts_df

    # -------------------------------------------------------------------------
    # 2. UCI ElectricityLoadDiagrams Loader
    # -------------------------------------------------------------------------
    def load_uci_electricity(
        self,
        file_path: Optional[Union[str, Path]] = None,
        client_id: str = "MT_001",
        resample_freq: str = "1h",
    ) -> pd.DataFrame:
        """
        Load electricity consumption time-series from UCI ElectricityLoadDiagrams 2011-2014 dataset.

        :param file_path: Path to 'LD2011_2014.txt' or CSV file.
        :param client_id: Specific client column to load (e.g., 'MT_001' to 'MT_370').
        :param resample_freq: Resampling frequency (e.g., '1h' for hourly aggregation).
        :return: Standardized DataFrame with columns ['timestamp', 'value'].
        """
        path = Path(file_path) if file_path else self.raw_dir / "LD2011_2014.txt"

        if not path.exists():
            path_csv = self.raw_dir / "electricity_load.csv"
            if path_csv.exists():
                path = path_csv
            else:
                raise FileNotFoundError(
                    f"[DataLoader] UCI Electricity dataset not found at {path}. "
                    f"Please download 'LD2011_2014.txt' and place it into '{self.raw_dir}'."
                )

        # UCI file uses semicolon separator and comma for decimals
        df = pd.read_csv(
            path,
            sep=";",
            decimal=",",
            parse_dates=[0],
            low_memory=False,
        )

        date_col = df.columns[0]
        df = df.rename(columns={date_col: "timestamp"})

        if client_id not in df.columns:
            # Fallback to first available client column
            client_id = df.columns[1]

        res_df = df[["timestamp", client_id]].rename(columns={client_id: "value"})
        res_df["value"] = pd.to_numeric(res_df["value"], errors="coerce").fillna(0.0)

        # Resample to hourly/daily frequency if specified
        if resample_freq:
            res_df = (
                res_df.set_index("timestamp")
                .resample(resample_freq)
                .mean()
                .reset_index()
            )

        return res_df

    # -------------------------------------------------------------------------
    # 3. Time-Series Change Point Detection (TCPD) benchmark datasets Loader
    # -------------------------------------------------------------------------
    def load_tcpd_series(self, dataset_name: str = "jfk_passengers", frequency: str = "1min") -> pd.DataFrame:
        """
        Load Time-Series Change Point Detection (TCPD) benchmark datasets.
        
        :param dataset_name: Name of the TCPD series ('jfk_passengers', etc.)
        :param frequency: Frequency of the time-series data.
        :return: Standardized DataFrame with columns ['timestamp', 'value'].
        """
        file_path = f"{self.base_dir}/raw/tcpd/{dataset_name}.csv"
    
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"TCPD dataset file not found at {file_path}")

        # Ensure standardized columns
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.date_range(start="2020-01-01", periods=len(df), freq=frequency)
        if "value" not in df.columns:
            # Fallback to the first numeric column if 'value' is missing
            numeric_cols = df.select_dtypes(include='number').columns
            df["value"] = df[numeric_cols[0]]
            
        return df[["timestamp", "value"]]

    def load_tcpd_annotated_series(
        self, 
        dataset_name: str = "jfk_passengers", 
        frequency: str = "1min", 
        annotator_key: str = None
    ) -> Tuple[pd.DataFrame, List[int]]:
        """
        Load TCPD dataset along with its human ground-truth change-point annotations 
        from 'data/raw/tcpd/annotations.json'.
        
        :param dataset_name: Name of the dataset (e.g., 'jfk_passengers').
        :param frequency: Time frequency for the datetime index.
        :param annotator_key: Specific annotator version key (e.g. '9'). 
                              If None, computes the union of all annotators or defaults to a consensus list.
        :return: (DataFrame with ['timestamp', 'value'], list of change point index positions)
        """
        # 1. Load the time series
        df = self.load_tcpd_series(dataset_name, frequency)

        # 2. Load annotations.json from data/raw/
        annotation_path = self.raw_dir / "tcpd" / "annotations.json"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotations file not found at {annotation_path}")

        with open(annotation_path, "r") as f:
            all_annotations = json.load(f)

        if dataset_name not in all_annotations:
            raise ValueError(f"Dataset '{dataset_name}' not found in annotations.json")

        dataset_annotations = all_annotations[dataset_name]

        # 3. Extract change point indices
        if annotator_key and annotator_key in dataset_annotations:
            # Pick a specific annotator profile
            ground_truth_cp = dataset_annotations[annotator_key]
        else:
            # Default behavior: Combine all unique indices across all annotators (Union consensus)
            cp_set = set()
            for key, points in dataset_annotations.items():
                if isinstance(points, list):
                    cp_set.update(points)
            ground_truth_cp = sorted(list(cp_set))

        return df, ground_truth_cp