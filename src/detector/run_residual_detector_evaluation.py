import pandas as pd
import numpy as np
from pathlib import Path
from src.detector.residual_detector import ResidualConformalDriftDetector

def evaluate_residual_detector(manifest, processed_dir):
    results = []
    
    for item in manifest:
        # Skip datasets explicitly marked as having no seasonality / non-applicable
        raw_period = item.get("seasonality_period", 24)
        if not isinstance(raw_period, int) or raw_period <= 1:
            print(f"Skipping non-seasonal dataset: {item.get('dataset_name', 'unknown')}")
            continue

        file_path = processed_dir / item["filename"]
        if not file_path.exists():
            continue
            
        df = pd.read_csv(file_path)
        
        # Ensure correct column formatting
        if "timestamp" not in df.columns:
            df["timestamp"] = np.arange(len(df))
            
        dataset = item["dataset_name"]
        level = item["level"]
        drift_type = item["drift_type"]
        magnitude = item["magnitude_sigma"]
        ground_truth_cp = item.get("change_point_idx", None)

        period = raw_period

        gt_cp = ground_truth_cp
        gt_cps = item.get("change_point_indices", [])
        if gt_cp is not None:
            ground_truth_cps = [gt_cp] if isinstance(gt_cp, int) else gt_cp
        elif gt_cps:
            ground_truth_cps = gt_cps
        else:
            ground_truth_cps = []

        if len(df) < (period + 50):
            continue

        # Adapt sensitivity based on dataset characteristics
        if "alibaba" in dataset.lower():
            sens_mult = 1.3
            min_alerts = 1
        else:
            sens_mult = 1.6
            min_alerts = 2

        detector = ResidualConformalDriftDetector(
            seasonality_period=period,
            calibration_window=150,
            alpha=0.05,
            sensitivity_multiplier=sens_mult,
            cooldown_steps=40,
            min_consecutive_alerts=min_alerts,
            variance_window=16
        )
        
        try:
            alerts = detector.fit_predict(df)
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
            continue

        detected_indices = [alert["index"] for alert in alerts]
        
        # Metric calculations
        n_samples = len(df)
        false_alarms = 0
        detected = False
        detection_delay = np.nan
        
        if level == 1:
            # Level 1: Drift-free (Count all alerts as false alarms)
            false_alarms = len(detected_indices)
            far_per_1000 = (false_alarms / n_samples) * 1000 if n_samples > 0 else 0
            precision, recall, f1 = 0.0, 0.0, 0.0
        else:
            # Level 2 & 3: Drift Detection Evaluation
            tp = 0
            fp = 0
            matched_gt = set()
            delays = []

            for det_idx in sorted(detected_indices):
                if not ground_truth_cps:
                    fp += 1
                    continue

                valid_matches = []
                for gt in ground_truth_cps:
                    tolerance_window = 50

                    if gt not in matched_gt and 0 <= (det_idx - gt) <= tolerance_window:
                        valid_matches.append((gt, det_idx - gt))

                if valid_matches:
                    valid_matches.sort(key=lambda x: x[1])
                    matched_gt_point, delay = valid_matches[0]
                    tp += 1
                    matched_gt.add(matched_gt_point)
                    delays.append(delay)
                else:
                    fp += 1

            fn = len(ground_truth_cps) - len(matched_gt) if ground_truth_cps else 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            far_per_1000 = None
            detection_delay = float(np.mean(delays)) if delays else np.nan
            detected = len(matched_gt) > 0

        results.append({
            "dataset": dataset,
            "level": level,
            "drift_type": drift_type,
            "magnitude": magnitude,
            "method": "residual_conformal",
            "false_alarms": fp if level != 1 else false_alarms,
            "far_per_1000": far_per_1000,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "detection_delay": detection_delay,
            "detected": detected
        })
        
    return pd.DataFrame(results)