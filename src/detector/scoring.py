import numpy as np
import pandas as pd


def score_alerts_for_series(
    detected_indices,
    level: int,
    n_samples: int,
    ground_truth_cps=None,
    tolerance_window: int = 50,
):
    """
    Unified event-based scoring across Level 1, Level 2, and Level 3.
    - Level 1: all alerts are false alarms; FAR is per 1000 samples.
    - Level 2 / Level 3: one-to-one forward matching in [gt, gt + tolerance_window],
      counting unmatched detections (pre-drift or outside tolerance) as FP,
      unmatched ground-truth change points as FN, and averaging delay over matched TPs.
    """
    sorted_dets = sorted(detected_indices)

    if level == 1:
        false_alarms = len(sorted_dets)
        far_per_1000 = (false_alarms / n_samples) * 1000.0 if n_samples > 0 else 0.0
        return {
            "false_alarms": false_alarms,
            "far_per_1000": far_per_1000,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "detection_delay": np.nan,
            "detected": 0,
        }

    gt_list = []
    if ground_truth_cps is not None:
        if isinstance(ground_truth_cps, (list, tuple, np.ndarray, pd.Series)):
            gt_list = [int(cp) for cp in ground_truth_cps if pd.notna(cp)]
        elif pd.notna(ground_truth_cps):
            gt_list = [int(ground_truth_cps)]

    tp = 0
    fp = 0
    matched_gt = set()
    delays = []

    for det_idx in sorted_dets:
        if not gt_list:
            fp += 1
            continue

        valid_matches = [
            (gt, int(det_idx) - gt)
            for gt in gt_list
            if gt not in matched_gt and 0 <= (int(det_idx) - gt) <= tolerance_window
        ]

        if valid_matches:
            valid_matches.sort(key=lambda x: x[1])
            matched_gt_point, delay = valid_matches[0]
            tp += 1
            matched_gt.add(matched_gt_point)
            delays.append(delay)
        else:
            fp += 1

    fn = (len(gt_list) - len(matched_gt)) if gt_list else 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    detection_delay = float(np.mean(delays)) if delays else np.nan
    detected = 1 if len(matched_gt) > 0 else 0

    return {
        "false_alarms": fp,
        "far_per_1000": np.nan,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "detection_delay": detection_delay,
        "detected": detected,
    }