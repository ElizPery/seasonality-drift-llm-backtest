"""
Baseline Evaluation Script.
Runs classic detectors (KS, PSI, Wasserstein, ADWIN, Page-Hinkley, Kswin) on raw data 
across a 3-Level Benchmark Architecture (Level 1, Level 2, and Level 3).
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from src.detector.baseline_detectors import BaselineDetectorWrapper
from src.detector.scoring import score_alerts_for_series


def run_all_baseline_detectors(data_dir: str = "data", manifest_file: str = "benchmark_manifest.json", output_file: str = "baseline_evaluation_results.csv"):
    processed_dir = Path(data_dir) / "processed"
    manifest_path = processed_dir / manifest_file

    if not manifest_path.exists():
        print(f"[Error] Benchmark manifest not found at {manifest_path}. Run generate_benchmarks.py first!")
        return

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Configure baseline detectors to test
    detectors_config = [
        # Batch Two-Sample Tests
        {"method": "ks", "window_size": 100, "threshold": 0.01},
        {"method": "psi", "window_size": 100, "threshold": 0.1},
        {"method": "wasserstein", "window_size": 100, "threshold": 0.5},
        # Streaming / Online Detectors (requires river)
        {"method": "adwin", "threshold": 0.002},
        {"method": "page_hinkley", "threshold": 50.0},
        {"method": "kswin", "window_size": 100, "threshold": 0.01}
    ]

    results = []

    print("=== Starting Baseline Detector Evaluation (3-Level Benchmark) ===")

    for item in manifest:
        file_path = processed_dir / item["filename"]
        if not file_path.exists():
            continue

        df = pd.read_csv(file_path)
        level = item["level"]
        dataset_name = item["dataset_name"]
        series_id = item.get("series_id", "default_series")
        seas_strength = item.get("seasonality_strength", np.nan)
        seas_group = item.get("seasonality_group", "unknown")
        n = len(df)

        for config in detectors_config:
            method_name = config["method"]
            try:
                wrapper = BaselineDetectorWrapper(**config)
                alerts = wrapper.run(df)
            except Exception as e:
                print(f"Error running {method_name} on {item['filename']}: {e}")
                continue

            alert_indices = [a["index"] for a in alerts]

            if level == 1:
                gt_cps = None
                drift_type = "none"
                magnitude = 0.0
            elif level == 2:
                gt_cps = item.get("change_point_idx")
                drift_type = item["drift_type"]
                magnitude = item["magnitude_sigma"]
            elif level == 3:
                gt_cps = item.get("change_point_indices", [])
                drift_type = item["drift_type"]
                magnitude = 0.0
            else:
                continue

            metrics = score_alerts_for_series(
                detected_indices=alert_indices,
                level=level,
                n_samples=n,
                ground_truth_cps=gt_cps,
                tolerance_window=50,
            )

            results.append({
                "dataset": dataset_name,
                "series_id": series_id,
                "seasonality_strength": seas_strength,
                "seasonality_group": seas_group,
                "level": level,
                "drift_type": drift_type,
                "magnitude": magnitude,
                "method": method_name,
                **metrics,
            })


    # Save results summary
    results_df = pd.DataFrame(results)

    group_cols = ["dataset", "series_id", "level", "drift_type", "magnitude", "method"]
    metric_cols = [
        "false_alarms",
        "far_per_1000",
        "precision",
        "recall",
        "f1_score",
        "detection_delay",
        "detected",
    ]

    # Ensure all target metric columns exist even if a specific level wasn't present in the manifest
    for col in metric_cols:
        if col not in results_df.columns:
            results_df[col] = np.nan

    # Group by experimental configuration and average across series (NaNs are ignored by .mean())
    summary_df = (
        results_df.groupby(group_cols, as_index=False, dropna=False)[metric_cols]
        .mean()
        .round(4)
    )

    results_path = processed_dir / output_file
    results_df.to_csv(results_path, index=False)
    print(f"\nEvaluation complete! Baseline results saved to: {results_path}")
    return results_df


if __name__ == "__main__":
    run_all_baseline_detectors()