"""The `DataLoader` class interface of stream A, kept for its scripts and notebooks. Every method is a thin
wrapper around the functions of this package and returns the standard frame (`timestamp`, `value`)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import numpy as np

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

    def load_alibaba_batch(self, file_path: Optional[Union[str, Path]] = None, n_series: int = 5, seed: int = 42):
        """Loads N distinct Alibaba CPU/memory traces."""
        return alibaba.load_alibaba_batch(file_path or self.raw_dir / "alibaba_cluster_trace.csv", n_series, seed)

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

    def load_uci_batch(self, n_series: int = 10, resample_freq: str = "1h", check_window: tuple[str, str] | None = ("2013-01-01", "2013-12-31"), seed: int | None = None) -> dict[str, tuple[pd.DataFrame, int]]:
        """Loads N distinct client frames from the UCI Electricity dataset."""
        raw = self.raw_dir / "LD2011_2014.txt"
        out = self.processed_dir / "uci_hourly.parquet"
        if not raw.exists() and not out.exists():
            raise FileNotFoundError(f"UCI Electricity dataset not found at {raw}. Download 'LD2011_2014.txt' there.")
        return uci.load_uci_batch(n_series=n_series, resample_freq=resample_freq, check_window=check_window, raw=raw, out=out, random_state=seed)

    def load_tcpd_series(self, dataset_name: str = "jfk_passengers", frequency: str = "1min") -> pd.DataFrame:
        return tcpd.load_tcpd_series(dataset_name, frequency, self.raw_dir / "tcpd")

    def load_tcpd_annotated_series(self, dataset_name: str = "jfk_passengers", frequency: str = "1min",
                                   annotator_key: str = None) -> Tuple[pd.DataFrame, List[int]]:
        return tcpd.load_tcpd_annotated_series(dataset_name, frequency, annotator_key, self.raw_dir / "tcpd")

    def load_tcpd_batch(
            self, 
            series_names: list[str] = [
                "jfk_passengers", 
                "lga_passengers", 
                "construction", 
                "iceland_tourism", 
                "seatbelts", 
                "us_population", 
                "occupancy", 
                "measles"
            ],
            frequency: str | dict[str, str] = "1min",
            period: int | dict[str, int | None] | None = None,
            annotator_key: str | dict[str, str | None] | None = None,
        ) -> dict[str, tuple[pd.DataFrame, int | None, list[int]]]:
        """Loads a batch of annotated TCPD series with their respective seasonal periods."""
        return tcpd.load_tcpd_batch(
            series_names=series_names,
            frequency=frequency,
            period=period,
            annotator_key=annotator_key,
            raw_dir=self.raw_dir / "tcpd"
        )

    def load_seismic_dataset(self, seed: int = None) -> Tuple[pd.DataFrame, List[int]]:
        return iquique.load_iquique_trace(seed)

    def load_seisbench_iquique_batch(self, n_series: int = 5, seed: int | None = None) -> dict[str, tuple[pd.DataFrame, List[int]]]:
        """Loads N distinct seismic traces from the SeisBench Iquique dataset."""
        return iquique.load_iquique_batch(n_series=n_series, seed=seed)
