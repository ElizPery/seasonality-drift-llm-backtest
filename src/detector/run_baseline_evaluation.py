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


def run_all_baseline_detectors(data_dir: str = "data"):
    processed_dir = Path(data_dir) / "processed"
    manifest_path = processed_dir / "benchmark_manifest.json"

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
                # Level 1: Clean series (no drift anywhere, Y_true is all 0)
                far = (len(alert_indices) / n) * 1000

                results.append({
                    "dataset": dataset_name,
                    "level": 1,
                    "drift_type": "none",
                    "magnitude": 0.0,
                    "method": method_name,
                    "false_alarms": len(alerts),
                    "far_per_1000": round(far, 2),
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "detection_delay": None
                })
                print(f"[Level 1 FAR] {method_name} on {dataset_name}: {far:.2f} alarms/1000 pts")

            elif level == 2:
                # Level 2: Event-based Precision, Recall, F1, and Delay evaluation
                change_idx = item["change_point_idx"]
                tolerance_window = 40  # Max steps allowed to catch the drift

                # FP: Alerts triggered BEFORE the change point
                fp_alerts = [idx for idx in alert_indices if idx < change_idx]
                fp = len(fp_alerts)

                # TP: Alert triggered within [change_idx, change_idx + tolerance_window]
                post_change_alerts = sorted([idx for idx in alert_indices if change_idx <= idx])
                
                valid_alerts = [
                    idx for idx in post_change_alerts 
                    if idx <= change_idx + tolerance_window
                ]

                tp = 1 if valid_alerts else 0
                fn = 0 if tp == 1 else 1

                # Event-based Precision and Recall
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                delay = (post_change_alerts[0] - change_idx) if post_change_alerts else None
                detected = 1 if post_change_alerts else 0

                results.append({
                    "dataset": dataset_name,
                    "level": 2,
                    "drift_type": item["drift_type"],
                    "magnitude": item["magnitude_sigma"],
                    "method": method_name,
                    "detected": detected,
                    "detection_delay": delay,
                    "precision": round(precision, 2),
                    "recall": round(recall, 2),
                    "f1_score": round(f1, 2)
                })

            elif level == 3:
                ground_truth_cps = item.get("change_point_indices", [])
                tolerance_window = 40  # Tolerance margin around each ground truth point

                tp = 0
                fp = 0
                matched_gt = set()
                delays = []

                for det in sorted(alert_indices):
                    if not ground_truth_cps:
                        fp += 1
                        continue

                    # Find valid matches where detection occurs AFTER or AT the change point
                    # within the forward-looking tolerance window [gt, gt + tolerance_window]
                    valid_matches = []
                    for gt in ground_truth_cps:
                        if gt not in matched_gt and 0 <= (det - gt) <= tolerance_window:
                            valid_matches.append((gt, det - gt))

                    if valid_matches:
                        # Pick the closest valid ground truth point
                        valid_matches.sort(key=lambda x: x[1])
                        matched_gt_point, delay = valid_matches[0]
                        
                        tp += 1
                        matched_gt.add(matched_gt_point)
                        delays.append(delay)
                    else:
                        fp += 1

                fn = len(ground_truth_cps) - len(matched_gt)

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                mean_delay = float(np.mean(delays)) if delays else None

                detected = 1 if len(matched_gt) > 0 else 0

                results.append({
                    "dataset": dataset_name,
                    "level": 3,
                    "drift_type": item["drift_type"],
                    "magnitude": 0.0,
                    "method": method_name,
                    "precision": round(precision, 2),
                    "recall": round(recall, 2),
                    "f1_score": round(f1, 2),
                    "detected": detected,
                    "detection_delay": round(mean_delay, 2) if mean_delay is not None else None
                })
                print(f"[Level 3 TCPD] {method_name} on {dataset_name}: F1={f1:.2f}, Precision={precision:.2f}, Recall={recall:.2f}")


    # Save results summary
    results_df = pd.DataFrame(results)
    results_path = processed_dir / "baseline_evaluation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nEvaluation complete! Baseline results saved to: {results_path}")
    return results_df


if __name__ == "__main__":
    run_all_baseline_detectors()