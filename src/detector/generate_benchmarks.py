"""
Benchmark Generation Script for Seasonality-Aware Drift Detection System.

Supports a 3-Level Benchmark Architecture:

- Level 1: Drift-free clean baselines (Alibaba, UCI Electricity, TCPD, SeisBench) for FAR evaluation.
- Level 2: Controlled Semi-Synthetic Drift Injections (Alibaba & UCI Electricity)
- Level 3: Real-world TCPD and SeisBench benchmarks with consensus ground-truth annotations
"""

import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from src.detector.data_loader import DataLoader
from src.detector.drift_generator import DriftGenerator


def generate_all_benchmarks(data_dir: str = "data"):
    """
    Generate Level 1, Level 2, and Level 3 benchmarks.
    
    :param data_dir: Path to base data directory containing raw and processed folders.
    """
    loader = DataLoader(data_dir=data_dir)
    processed_dir = Path(data_dir) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=== Starting 3-Level Benchmark Generation ===")

    # Target parameters for Level 2 semi-synthetic drift injection
    drift_types = ["abrupt", "gradual", "seasonal_amplitude"]
    magnitudes = [0.5, 1.0, 2.0, 3.0]  # Magnitudes in sigma standard deviations

    metadata_catalog: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Levels 1 & 2: Alibaba Cluster Trace & UCI Electricity (Synthetic Framework)
    # -------------------------------------------------------------------------
    # Map target dataset names to their loader methods and seasonality periods
    synthetic_datasets = [
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

    for ds_info in synthetic_datasets:
        ds_name = ds_info["name"]
        period = ds_info["seasonality_period"]

        print(f"\n[Levels 1 & 2] Processing dataset: '{ds_name}'...")

        try:
            raw_df = ds_info["loader_func"]()
        except FileNotFoundError as err:
            print(f"  [Skipped] {err}")
            continue

        generator = DriftGenerator(raw_df, seasonality_period=period)

        # Level 1: Generate Drift-Free Baseline Series (FAR Evaluation)
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

        # Level 2: Controlled Semi-Synthetic Drift Injections
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

    # -------------------------------------------------------------------------
    # Level 1 ONLY: TCPD & SeisBench Clean Baselines (For FAR Evaluation)
    # -------------------------------------------------------------------------
    level1_only_datasets = [
        {
            "name": "tcpd_jfk_passengers",
            "seasonality_period": 12,
            "loader_func": lambda: loader.load_tcpd_annotated_series("jfk_passengers", annotator_key="10")[0],
        },
        {
            "name": "seismic_iquique_trace",
            "seasonality_period": 50,
            "loader_func": lambda: loader.load_seismic_dataset(seed=42)[0],
        },
    ]

    for ds_info in level1_only_datasets:
        ds_name = ds_info["name"]
        period = ds_info["seasonality_period"]

        print(f"\n[Level 1 Only] Processing clean baseline for: '{ds_name}'...")
        try:
            raw_df = ds_info["loader_func"]()
            generator = DriftGenerator(raw_df, seasonality_period=period)
            
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
            print(f"  [Level 1] Saved clean baseline -> {level1_filename}")
        except Exception as err:
            print(f"  [Skipped Level 1 for {ds_name}] {err}")

    # -------------------------------------------------------------------------
    # Level 3: TCPD Real-World Benchmarks with Consensus Annotations
    # -------------------------------------------------------------------------
    print("\n[Level 3] Processing TCPD Real-World Annotated Datasets...")
    
    tcpd_targets = ["jfk_passengers"]  # TCPD benchmark series names

    for tcpd_name in tcpd_targets:
        try:
            tcpd_df, ground_truth_cps = loader.load_tcpd_annotated_series(tcpd_name, annotator_key="10")
            period = "Not defined"  # Estimated domain periodicity or configured default

            level3_filename = f"level3_tcpd_{tcpd_name}.csv"
            level3_path = processed_dir / level3_filename
            tcpd_df.to_csv(level3_path, index=False)

            # Map timestamps for ground truth change point indices
            gt_timestamps = [str(tcpd_df.loc[idx, "timestamp"]) for idx in ground_truth_cps if idx < len(tcpd_df)]

            metadata_catalog.append({
                "dataset_name": f"tcpd_{tcpd_name}",
                "level": 3,
                "filename": level3_filename,
                "drift_type": "real_consensus_annotation",
                "magnitude_sigma": None,
                "change_point_indices": ground_truth_cps,
                "change_point_timestamps": gt_timestamps,
                "seasonality_period": period
            })
            print(f"  -> Level 3 saved [TCPD {tcpd_name}] with {len(ground_truth_cps)} ground-truth change points.")

        except Exception as e:
            print(f"  [Skipped TCPD '{tcpd_name}'] {e}")

    # -------------------------------------------------------------------------
    # Level 3: Seismic Datasets (P & S Wave Change Points)
    # -------------------------------------------------------------------------
    print("\n[Level 3] Processing SeisBench Iquique Benchmark...")
    
    try:
        seismic_df, gt_cps = loader.load_seismic_dataset(seed=42)
        
        level3_filename = "level3_seismic_iquique_random.csv"
        level3_path = processed_dir / level3_filename
        seismic_df.to_csv(level3_path, index=False)

        gt_timestamps = [str(seismic_df.loc[idx, "timestamp"]) for idx in gt_cps if idx < len(seismic_df)]

        metadata_catalog.append({
            "dataset_name": "seismic_iquique_random",
            "level": 3,
            "filename": level3_filename,
            "drift_type": "seismic_single_wave_arrival",
            "magnitude_sigma": None,
            "change_point_indices": gt_cps,
            "change_point_timestamps": gt_timestamps,
            "seasonality_period": "Not applicable"
        })
        print(f"  -> Saved [Seismic Iquique] benchmark with change point index: {gt_cps}")

    except Exception as e:
        print(f"  [Skipped Seismic Iquique Benchmark] {e}")

    # Save benchmark metadata manifest
    manifest_path = processed_dir / "benchmark_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(metadata_catalog, f, indent=4)

    print(f"\n3-Level Benchmark Generation Complete! Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    generate_all_benchmarks()