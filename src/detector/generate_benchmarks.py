"""
Benchmark Generation Script for Seasonality-Aware Drift Detection System.

Applies the Semi-Synthetic Drift Injection Framework (Levels 1 & 2) 
primarily on Alibaba Cluster Trace and UCI Electricity datasets.

Level 1: Drift-free series (Block Bootstrap on residuals) for FAR evaluation.
Level 2: Semi-synthetic drift injection across multiple magnitudes (0.5σ to 3σ) and drift types.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from src.detector.data_loader import DataLoader
from src.detector.drift_generator import DriftGenerator


def generate_all_benchmarks(data_dir: str = "data"):
    """
    Generate Level 1 and Level 2 benchmarks for primary datasets (Alibaba & UCI Electricity).
    
    :param data_dir: Path to base data directory containing raw and processed folders.
    """
    loader = DataLoader(data_dir=data_dir)
    processed_dir = Path(data_dir) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=== Starting Benchmark Generation (Level 1 & Level 2) ===")

    # Target parameters for Level 2 semi-synthetic drift injection
    drift_types = ["abrupt", "gradual", "seasonal_amplitude"]
    magnitudes = [0.5, 1.0, 2.0, 3.0]  # Magnitudes in sigma standard deviations

    metadata_catalog: List[Dict[str, Any]] = []

    # Map target dataset names to their loader methods and seasonality periods
    datasets_to_process = [
        {
            "name": "alibaba_cluster_trace",
            "seasonality_period": 288,  # 5-min intervals over 24h (288 steps per day)
            "loader_func": lambda: loader.load_alibaba_standard_trace(metric="cpu_util"),
        },
        {
            "name": "uci_electricity_MT200",
            "seasonality_period": 24,   # Hourly frequency (24 steps per day)
            "loader_func": lambda: loader.load_uci_electricity(client_id="MT_200", resample_freq="1h"),
        },
    ]

    for ds_info in datasets_to_process:
        ds_name = ds_info["name"]
        period = ds_info["seasonality_period"]

        print(f"\nProcessing primary dataset: '{ds_name}'...")

        try:
            raw_df = ds_info["loader_func"]()
        except FileNotFoundError as err:
            print(f"  [Skipped] {err}")
            print(f"  Falling back to TCPD sample datasets for testing pipeline...")
            # Fallback to TCPD dataset if local raw file isn't placed in data/raw/ yet
            ds_name = "tcpd_occupancy"
            period = 90
            raw_df = loader.load_tcpd_series("occupancy")

        generator = DriftGenerator(raw_df, seasonality_period=period)

        # ---------------------------------------------------------------------
        # Level 1: Generate Drift-Free Baseline Series (FAR Evaluation)
        # ---------------------------------------------------------------------
        level1_df = generator.generate_level1_baseline()
        level1_filename = f"level1_clean_{ds_name}.csv"
        level1_path = processed_dir / level1_filename
        level1_df.to_csv(level1_path, index=False)

        metadata_catalog.append({
            "dataset_name": ds_name,
            "level": 1,
            "filename": level1_filename,
            "drift_type": "none",
            "magnitude_sigma": 0.0,
            "change_point_idx": None,
            "seasonality_period": period
        })
        print(f"  [Level 1] Saved drift-free series -> {level1_filename}")

        # ---------------------------------------------------------------------
        # Level 2: Controlled Semi-Synthetic Drift Injections
        # ---------------------------------------------------------------------
        change_idx = len(raw_df) // 2  # Change point placed at 50% length

        for d_type in drift_types:
            for mag in magnitudes:
                injected_df, drift_meta = generator.inject_level2_drift(
                    drift_type=d_type,
                    change_point_idx=change_idx,
                    magnitude_sigma=mag
                )

                level2_filename = f"level2_{ds_name}_{d_type}_mag{mag}sig.csv"
                level2_path = processed_dir / level2_filename
                injected_df.to_csv(level2_path, index=False)

                metadata_catalog.append({
                    "dataset_name": ds_name,
                    "level": 2,
                    "filename": level2_filename,
                    "drift_type": d_type,
                    "magnitude_sigma": mag,
                    "change_point_idx": drift_meta["change_point_idx"],
                    "change_point_timestamp": str(drift_meta["change_point_timestamp"]),
                    "absolute_delta": drift_meta["absolute_delta"],
                    "seasonality_period": period
                })
                print(f"  [Level 2] Saved {d_type} drift ({mag}σ) -> {level2_filename}")

    # Save benchmark metadata manifest
    manifest_path = processed_dir / "benchmark_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(metadata_catalog, f, indent=4)

    print(f"\nSuccessfully generated benchmarks! Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    generate_all_benchmarks()