"""
Baseline Detectors Module for Seasonality-Aware Drift Detection System.

Implements traditional, off-the-shelf drift detectors operating directly on RAW 
time-series data (WITHOUT seasonality decomposition) to record baseline metrics 
(FAR, Precision, Recall, Delay).

Supported Detectors:
1. Two-Sample Window/Batch Tests:
   - Kolmogorov-Smirnov (KS-test)
   - Population Stability Index (PSI)
   - Wasserstein Distance / Earth Mover's Distance
2. Streaming/Online Detectors (via River):
   - ADWIN (Adaptive Windowing)
   - Page-Hinkley (CUSUM-based)
   - KSWIN (Kolmogorov-Smirnov Windowing)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Any, Union, Optional

# Try importing streaming detectors from river
try:
    from river import drift
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False


def calculate_psi(reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) between reference and current windows.
    
    :param reference: Array of baseline reference values.
    :param current: Array of current window values.
    :param num_bins: Number of quantiles/bins for histogram calculation.
    :return: Float PSI value (> 0.1 indicates moderate shift, > 0.25 significant shift).
    """
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    # Determine bin edges based on reference distribution
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(reference, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Calculate proportions in each bin
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    curr_counts, _ = np.histogram(current, bins=bin_edges)

    ref_props = ref_counts / len(reference)
    curr_props = curr_counts / len(current)

    # Avoid zero counts for log stability
    eps = 1e-4
    ref_props = np.where(ref_props == 0, eps, ref_props)
    curr_props = np.where(curr_props == 0, eps, curr_props)

    # Calculate PSI formula
    psi_value = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
    return float(psi_value)


class BaselineDetectorWrapper:
    """Unified wrapper for executing classic drift detectors on raw time-series data."""

    def __init__(
        self,
        method: str = "ks",
        window_size: int = 100,
        ref_window_size: Optional[int] = None,
        threshold: float = 0.05,
    ):
        """
        Initialize the baseline detector wrapper.

        :param method: Algorithm name ('ks', 'psi', 'wasserstein', 'adwin', 'page_hinkley', 'kswin').
        :param window_size: Size of sliding current window (or min_window length for streaming).
        :param ref_window_size: Size of reference window for batch tests (defaults to window_size).
        :param threshold: Significance alpha threshold for statistical tests or drift threshold.
        """
        self.method = method.lower()
        self.window_size = window_size
        self.ref_window_size = ref_window_size if ref_window_size else window_size
        self.threshold = threshold

        if self.method in ["adwin", "page_hinkley", "kswin"] and not RIVER_AVAILABLE:
            raise ImportError(
                "The 'river' package is required for streaming detectors. "
                "Please install it using 'pip install river'."
            )

    def detect_sliding_batch(self, series: Union[pd.Series, np.ndarray]) -> List[int]:
        """
        Execute two-sample batch tests (KS, PSI, Wasserstein) using a sliding reference & current window.

        :param series: 1D numerical time-series values.
        :return: List of 0-based indices where drift alerts were triggered.
        """
        values = np.asarray(series, dtype=np.float64)
        n = len(values)
        drift_indices = []

        stride = max(1, self.window_size // 4)  # Step size for sliding check
        min_required = self.ref_window_size + self.window_size

        for i in range(min_required, n, stride):
            ref_data = values[i - self.ref_window_size - self.window_size : i - self.window_size]
            curr_data = values[i - self.window_size : i]

            is_drift = False

            if self.method == "ks":
                # Kolmogorov-Smirnov test (p-value < threshold indicates drift)
                stat, p_val = stats.ks_2samp(ref_data, curr_data)
                is_drift = p_val < self.threshold

            elif self.method == "psi":
                # Population Stability Index (PSI > threshold indicates drift, default ~0.1 or 0.2)
                psi_val = calculate_psi(ref_data, curr_data)
                is_drift = psi_val > self.threshold

            elif self.method == "wasserstein":
                # Wasserstein distance
                dist = stats.wasserstein_distance(ref_data, curr_data)
                is_drift = dist > self.threshold

            else:
                raise ValueError(f"Unsupported batch method: '{self.method}'. Use 'ks', 'psi', or 'wasserstein'.")

            if is_drift:
                drift_indices.append(i)

        return drift_indices

    def detect_streaming(self, series: Union[pd.Series, np.ndarray]) -> List[int]:
        """
        Execute online/streaming drift detection (ADWIN, Page-Hinkley, KSWIN) sequentially.

        :param series: 1D numerical time-series values.
        :return: List of 0-based indices where streaming drift was flagged.
        """
        values = np.asarray(series, dtype=np.float64)
        drift_indices = []

        # Instantiate River streaming detector
        if self.method == "adwin":
            detector = drift.ADWIN(delta=self.threshold)
        elif self.method == "page_hinkley":
            detector = drift.PageHinkley(threshold=self.threshold)
        elif self.method == "kswin":
            detector = drift.KSWIN(alpha=self.threshold, window_size=self.window_size)
        else:
            raise ValueError(f"Unsupported streaming method: '{self.method}'. Use 'adwin', 'page_hinkley', or 'kswin'.")

        for idx, val in enumerate(values):
            detector.update(val)
            if detector.drift_detected:
                drift_indices.append(idx)

        return drift_indices

    def run(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Run the configured detector on a standardized DataFrame.

        :param df: Standardized DataFrame with columns ['timestamp', 'value'].
        :return: List of records containing timestamp and index of flagged drift alerts.
        """
        values = df["value"].values

        if self.method in ["ks", "psi", "wasserstein"]:
            flagged_indices = self.detect_sliding_batch(values)
        elif self.method in ["adwin", "page_hinkley", "kswin"]:
            flagged_indices = self.detect_streaming(values)
        else:
            raise ValueError(f"Unknown detection method '{self.method}'.")

        alerts = []
        for idx in flagged_indices:
            alerts.append({
                "index": idx,
                "timestamp": df.loc[idx, "timestamp"] if "timestamp" in df.columns else idx,
                "method": self.method,
            })

        return alerts