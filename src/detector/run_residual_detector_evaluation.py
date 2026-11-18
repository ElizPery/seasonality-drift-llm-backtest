import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from src.detector.residual_detector import ResidualConformalDriftDetector


# Separate parameter dictionaries for Single-Series vs. Multi-Series evaluation
RESIDUAL_DETECTOR_CONFIGS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "single": {
        "default": {
            "calibration_window": 150,
            "alpha": 0.05,
            "sensitivity_multiplier": 1.6,
            "cooldown_steps": 40,
            "min_consecutive_alerts": 2,
            "variance_window": 16,
            "tolerance_window": 50,
            "dynamic_windows": False,
        },
        "alibaba": {
            "sensitivity_multiplier": 1.3,
            "min_consecutive_alerts": 1,
        },
    },
    "multi": {
        "default": {
            "calibration_window": 150,
            "alpha": 0.05,
            "sensitivity_multiplier": 1.6,
            "cooldown_steps": 40,
            "min_consecutive_alerts": 2,
            "variance_window": 16,
            "tolerance_window": 50,
            "dynamic_windows": True,  # Scales windows to period & series length
        },
        "alibaba": {
            "sensitivity_multiplier": 1.25,
            "min_consecutive_alerts": 1,
        },
        "tcpd": {
            "sensitivity_multiplier": 1.35,
            "min_consecutive_alerts": 1,
        },
        "uci": {
            "sensitivity_multiplier": 1.6,
            "min_consecutive_alerts": 2,
        },
    },
}


def get_detector_params(
    dataset_name: str,
    period: int,
    n_samples: int,
    mode: str = "single",
    custom_configs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolves detector and evaluation window parameters for a given mode and dataset."""
    configs = custom_configs or RESIDUAL_DETECTOR_CONFIGS
    mode_cfg = configs.get(mode, configs["single"])

    # Start with mode defaults and override with dataset-specific keys if matched
    params = mode_cfg["default"].copy()
    ds_lower = dataset_name.lower()
    for key, overrides in mode_cfg.items():
        if key != "default" and key in ds_lower:
            params.update(overrides)
            break

    # Apply dynamic window scaling if enabled (primarily for single-series heterogeneity)
    if params.pop("dynamic_windows", False):
        params["calibration_window"] = int(np.clip(max(3 * period, 36), 36, min(150, n_samples // 3)))
        params["variance_window"] = int(np.clip(period // 2, 6, 24))
        params["cooldown_steps"] = int(np.clip(period, 10, 40))
        params["tolerance_window"] = int(np.clip(max(period, 15), 15, 50))

    return params


def evaluate_residual_detector(
    manifest,
    processed_dir: Path,
    mode: str = "single",
    custom_configs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Evaluates ResidualConformalDriftDetector across a benchmark manifest.
    
    :param manifest: Loaded benchmark_manifest.json list.
    :param processed_dir: Path to directory containing benchmark CSVs.
    :param mode: 'single' for single-series fixed params, 'multi' for multi-series batch params.
    :param custom_configs: Optional override dictionary matching RESIDUAL_DETECTOR_CONFIGS schema.
    """
    results = []

    for item in manifest:
        raw_period = item.get("seasonality_period", 24)
        if not isinstance(raw_period, int) or raw_period <= 1:
            print(f"Skipping non-seasonal dataset: {item.get('dataset_name', 'unknown')}")
            continue

        file_path = processed_dir / item["filename"]
        if not file_path.exists():
            continue

        df = pd.read_csv(file_path)
        if "timestamp" not in df.columns:
            df["timestamp"] = np.arange(len(df))

        dataset = item["dataset_name"]
        series_id = item.get("series_id", "default_series")
        level = item["level"]
        drift_type = item["drift_type"]
        magnitude = item["magnitude_sigma"]
        ground_truth_cp = item.get("change_point_idx", None)

        period = raw_period
        n_samples = len(df)

        gt_cp = ground_truth_cp
        gt_cps = item.get("change_point_indices", [])
        if gt_cp is not None:
            ground_truth_cps = [gt_cp] if isinstance(gt_cp, int) else gt_cp
        elif gt_cps:
            ground_truth_cps = gt_cps
        else:
            ground_truth_cps = []

        # Fetch resolved parameters for current mode & dataset
        params = get_detector_params(
            dataset_name=dataset,
            period=period,
            n_samples=n_samples,
            mode=mode,
            custom_configs=custom_configs,
        )
        tolerance_window = params.pop("tolerance_window")

        if n_samples < (period + params["calibration_window"]):
            print(
                f"Skipping short series {item['filename']} "
                f"(len={n_samples}, req={period + params['calibration_window']})"
            )
            continue

        detector = ResidualConformalDriftDetector(
            seasonality_period=period,
            **params,
        )

        try:
            alerts = detector.fit_predict(df)
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
            continue

        detected_indices = [alert["index"] for alert in alerts]

        false_alarms = 0
        detected = False
        detection_delay = np.nan

        if level == 1:
            false_alarms = len(detected_indices)
            far_per_1000 = (false_alarms / n_samples) * 1000 if n_samples > 0 else 0
            precision, recall, f1 = 0.0, 0.0, 0.0
        else:
            tp = 0
            fp = 0
            matched_gt = set()
            delays = []

            for det_idx in sorted(detected_indices):
                if not ground_truth_cps:
                    fp += 1
                    continue

                valid_matches = [
                    (gt, det_idx - gt)
                    for gt in ground_truth_cps
                    if gt not in matched_gt and 0 <= (det_idx - gt) <= tolerance_window
                ]

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
            "series_id": series_id,
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
            "detected": detected,
        })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        return results_df

    results_df["detected"] = results_df["detected"].astype(float)

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

    summary_df = (
        results_df.groupby(group_cols, as_index=False, dropna=False)[metric_cols]
        .mean()
        .round(4)
    )

    return results_df