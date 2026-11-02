import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from typing import List, Dict, Any, Optional

class ResidualConformalDriftDetector:
    """
    Detects concept/data drift on time-series residuals after STL decomposition
    using adaptive conformal prediction intervals that monitor both residual 
    magnitude shifts (level shifts) and residual spread/variance changes (amplitude/noise shifts).
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
        variance_window: int = 12
    ):
        """
        :param seasonality_period: Period of seasonality for STL decomposition (e.g., 24, 288).
        :param calibration_window: Historical window to calibrate non-conformity quantiles.
        :param alpha: Conformal significance level (e.g., 0.05 for 95% coverage).
        :param robust: Whether to use robust LOESS smoothing in STL.
        :param sensitivity_multiplier: Multiplier for conformal threshold (1.4 - 1.5 recommended).
        :param cooldown_steps: Minimum timesteps required between consecutive drift alerts.
        :param min_consecutive_alerts: Number of consecutive violations to trigger an alert.
        :param variance_window: Short rolling window to monitor local residual spread/IQR.
        """
        self.period = seasonality_period
        self.cal_window = calibration_window
        self.alpha = alpha
        self.robust = robust
        self.multiplier = sensitivity_multiplier
        self.cooldown_steps = cooldown_steps
        self.min_consecutive = min_consecutive_alerts
        self.var_window = variance_window

    def fit_predict(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        values = df["value"].values
        timestamps = df["timestamp"].values if "timestamp" in df.columns else np.arange(len(values))
        n = len(values)

        if n < (self.period + self.cal_window):
            raise ValueError(
                f"Series length ({n}) is too short for period ({self.period}) "
                f"and calibration window ({self.cal_window})."
            )

        # 1. STL Decomposition to isolate seasonal residuals
        stl = STL(values, period=self.period, robust=self.robust)
        stl_result = stl.fit()
        residuals = stl_result.resid

        alerts = []
        consecutive_violations = 0
        last_alert_idx = -self.cooldown_steps

        # 2. Sequential conformal evaluation with Variance/Spread awareness
        for i in range(self.cal_window, n):
            # Baseline calibration window of past residuals
            cal_residuals = residuals[i - self.cal_window : i]
            median_res = np.median(cal_residuals)
            
            # Non-conformity score calculation (Absolute deviation)
            abs_devs = np.abs(cal_residuals - median_res)
            
            # Conformal threshold (1 - alpha quantile)
            q_idx = int(np.ceil((1.0 - self.alpha) * (len(abs_devs) + 1))) - 1
            q_idx = np.clip(q_idx, 0, len(abs_devs) - 1)
            sorted_scores = np.sort(abs_devs)
            base_threshold = sorted_scores[q_idx]
            adjusted_threshold = base_threshold * self.multiplier

            # Evaluation score 1: Direct residual deviation
            current_res = residuals[i]
            current_score = np.abs(current_res - median_res)

            # Evaluation score 2: Local residual dispersion / Interquartile Range (IQR)
            # Detects seasonal amplitude shifts where residual volatility increases
            if i >= self.cal_window + self.var_window:
                local_window = residuals[i - self.var_window + 1 : i + 1]
                local_iqr = np.percentile(local_window, 75) - np.percentile(local_window, 25)
                
                # Baseline calibration IQR distribution
                cal_iqrs = np.abs(cal_residuals - np.percentile(cal_residuals, 50))
                baseline_iqr = np.median(cal_iqrs) * 2.0
                iqr_score = local_iqr / (baseline_iqr + 1e-8)
            else:
                iqr_score = 0.0

            # Violation trigger: Either absolute offset breach OR residual dispersion spike (> 2.0x baseline)
            is_violation = (current_score > adjusted_threshold) or (iqr_score > (1.8 * self.multiplier))

            if is_violation:
                consecutive_violations += 1
            else:
                consecutive_violations = 0

            # Alert confirmation logic
            if (consecutive_violations >= self.min_consecutive) and ((i - last_alert_idx) >= self.cooldown_steps):
                alerts.append({
                    "index": i,
                    "timestamp": str(timestamps[i]),
                    "residual": float(current_res),
                    "score": float(current_score),
                    "threshold": float(adjusted_threshold),
                    "iqr_score": float(iqr_score)
                })
                last_alert_idx = i

        return alerts