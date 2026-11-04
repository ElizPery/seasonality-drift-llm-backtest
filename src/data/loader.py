"""The `DataLoader` class interface of stream A, kept for its scripts and notebooks. Every method is a thin
wrapper around the functions of this package and returns the standard frame (`timestamp`, `value`)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd

from src.data import alibaba, iquique, tcpd, uci


class DataLoader:
    """Loaders of the experimental datasets, resolved against one data directory (`raw/` and `processed/`)."""

    def __init__(self, data_dir: Union[str, Path] = "data"):
        self.base_dir = Path(data_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_alibaba_standard_trace(self, file_path: Optional[Union[str, Path]] = None, metric: str = "cpu_util") -> pd.DataFrame:
        """Alibaba cluster trace 2018 (grouped 300-second format), one metric averaged over the machines."""
        return alibaba.load_alibaba_trace(file_path or self.raw_dir / "alibaba_cluster_trace.csv", metric)

    def load_uci_electricity(self, file_path: Optional[Union[str, Path]] = None, client_id: str = "MT_001",
                             resample_freq: str = "1h") -> pd.DataFrame:
        """One client of UCI ElectricityLoadDiagrams from the cleaned hourly table (`uci.load_hourly`:
        timestamps moved to the interval start, DST hours fixed, kW averaged per hour). A coarser
        `resample_freq` averages the hours; the hourly table is cached in `processed/uci_hourly.parquet`."""
        raw = Path(file_path) if file_path else self.raw_dir / "LD2011_2014.txt"
        if not raw.exists() and not (self.processed_dir / "uci_hourly.parquet").exists():
            raise FileNotFoundError(f"UCI Electricity dataset not found at {raw}. Download 'LD2011_2014.txt' there.")
        hourly = uci.load_hourly(self.processed_dir / "uci_hourly.parquet", raw)
        if client_id not in hourly.columns:
            client_id = hourly.columns[0]  # fallback to the first client, as before
        return uci.client_frame(client_id, resample_freq, hourly)

    def load_tcpd_series(self, dataset_name: str = "jfk_passengers", frequency: str = "1min") -> pd.DataFrame:
        return tcpd.load_tcpd_series(dataset_name, frequency, self.raw_dir / "tcpd")

    def load_tcpd_annotated_series(self, dataset_name: str = "jfk_passengers", frequency: str = "1min",
                                   annotator_key: str = None) -> Tuple[pd.DataFrame, List[int]]:
        return tcpd.load_tcpd_annotated_series(dataset_name, frequency, annotator_key, self.raw_dir / "tcpd")

    def load_seismic_dataset(self, seed: int = None) -> Tuple[pd.DataFrame, List[int]]:
        return iquique.load_iquique_trace(seed)
