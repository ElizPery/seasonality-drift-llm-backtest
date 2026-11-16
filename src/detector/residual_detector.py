import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from typing import List, Dict, Any


class ResidualConformalDriftDetector:
    """
    Detects concept/data drift on time-series residuals after STL decomposition
    using adaptive conformal prediction intervals that monitor:
      1. Point-wise residual magnitude shifts (abrupt shifts)
      2. Rolling mean residual shifts (gradual/ramp drifts)
      3. Local residual spread/IQR changes (seasonal amplitude/noise shifts)
    """

    def __init__(
        self,
        seasonality_period: int,
        calibration_window: int = 120,
        alpha: float = 0.05,
        robust: bool = True,
        sensitivity_multiplier: float = 1.4,
        cooldown_steps: int = 30,
        min_consecutive_alerts: int = 1,
        variance_window: int = 12,
        causal_step: int | None = None,
    ):
        self.period = max(2, int(seasonality_period))
        self.cal_window = calibration_window
        self.alpha = alpha
        self.robust = robust
        self.multiplier = sensitivity_multiplier
        self.cooldown_steps = cooldown_steps
        self.min_consecutive = min_consecutive_alerts
        self.var_window = max(3, variance_window)
        # Re-fit STL causally every `causal_step` points to prevent future drift leakage into trend
        self.causal_step = causal_step if causal_step is not None else self.period

    def _compute_causal_residuals(self, values: np.ndarray) -> np.ndarray:
        """
        Computes STL residuals causally in blocks of `causal_step` using only past data
        up to the current block, preventing post-drift points from distorting pre-drift
        or early post-drift LOESS trend estimates.
        """
        n = len(values)
        min_stl_len = max(2 * self.period, self.cal_window)
        if n <= min_stl_len:
            return STL(values, period=self.period, robust=self.robust).fit().resid

        residuals = np.zeros(n, dtype=float)
        # Initial warm-up decomposition on the first `min_stl_len` points
        init_res = STL(values[:min_stl_len], period=self.period, robust=self.robust).fit().resid
        residuals[:min_stl_len] = init_res

        # Max history window for STL keeps runtime O(N) even on 35k-point UCI series
        max_stl_history = max(6 * self.period, 3 * self.cal_window)

        for start in range(min_stl_len, n, self.causal_step):
            end = min(start + self.causal_step, n)
            hist_start = max(0, end - max_stl_history)
            segment = values[hist_start:end]
            seg_resid = STL(segment, period=self.period, robust=self.robust).fit().resid
            residuals[start:end] = seg_resid[-(end - start):]

        return residuals

    def fit_predict(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        values = df["value"].astype(float).values
        timestamps = df["timestamp"].values if "timestamp" in df.columns else np.arange(len(values))
        n = len(values)

        if n < (self.period + self.cal_window):
            raise ValueError(
                f"Series length ({n}) is too short for period ({self.period}) "
                f"and calibration window ({self.cal_window})."
            )

        # 1. Causal STL Decomposition to prevent trend absorption of gradual drifts
        residuals = self._compute_causal_residuals(values)

        alerts = []
        consecutive_violations = 0
        last_alert_idx = -self.cooldown_steps

        # 2. Sequential conformal evaluation
        for i in range(self.cal_window, n):
            cal_residuals = residuals[i - self.cal_window : i]
            median_res = np.median(cal_residuals)

            # Non-conformity score 1: Point-wise absolute deviation
            abs_devs = np.abs(cal_residuals - median_res)
            q_idx = int(np.ceil((1.0 - self.alpha) * (len(abs_devs) + 1))) - 1
            q_idx = np.clip(q_idx, 0, len(abs_devs) - 1)
            base_threshold = np.sort(abs_devs)[q_idx]
            adjusted_threshold = base_threshold * self.multiplier

            current_res = residuals[i]
            current_score = np.abs(current_res - median_res)

            # Non-conformity score 2 & 3: Local rolling mean (gradual drift) & IQR (amplitude drift)
            if i >= self.cal_window + self.var_window:
                local_window = residuals[i - self.var_window + 1 : i + 1]
                local_mean_shift = np.abs(np.mean(local_window) - median_res)
                # Threshold for mean of `var_window` residuals scales by ~1/sqrt(var_window)
                mean_threshold = (adjusted_threshold / np.sqrt(self.var_window)) * 1.35

                local_iqr = np.percentile(local_window, 75) - np.percentile(local_window, 25)
                baseline_iqr = np.percentile(cal_residuals, 75) - np.percentile(cal_residuals, 25)
                iqr_score = local_iqr / (baseline_iqr + 1e-8)
            else:
                local_mean_shift = 0.0
                mean_threshold = np.inf
                iqr_score = 0.0

            is_violation = (
                (current_score > adjusted_threshold)
                or (local_mean_shift > mean_threshold)
                or (iqr_score > (1.8 * self.multiplier))
            )

            if is_violation:
                consecutive_violations += 1
            else:
                consecutive_violations = 0

            if (consecutive_violations >= self.min_consecutive) and ((i - last_alert_idx) >= self.cooldown_steps):
                alerts.append({
                    "index": i,
                    "timestamp": str(timestamps[i]),
                    "residual": float(current_res),
                    "score": float(current_score),
                    "threshold": float(adjusted_threshold),
                    "iqr_score": float(iqr_score),
                })
                last_alert_idx = i
                consecutive_violations = 0  # Reset after firing so bursts don't double-fire

        return alerts