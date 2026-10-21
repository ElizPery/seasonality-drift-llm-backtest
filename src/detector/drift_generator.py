"""
Drift Generator Module for Seasonality-Aware Drift Detection System.
Implements the 3-level validation methodology:
- Level 1: Block bootstrap on residuals to generate drift-free series (FAR evaluation).
- Level 2: Semi-synthetic drift injection (abrupt, gradual, seasonal amplitude) with 
  controlled magnitudes in standard deviations of residuals (sigma).
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from typing import Tuple, Dict, Any

class DriftGenerator:
    """Generates synthetic drift and clean baseline series based on project specifications."""

    def __init__(self, series: pd.DataFrame, seasonality_period: int):
        """
        Initialize the generator with a clean reference time series.
        
        :param series: Standardized DataFrame with columns ['timestamp', 'value']
        :param seasonality_period: Period of seasonality (e.g., 24 for hourly daily cycle, 7 for daily weekly cycle)
        """
        self.df = series.copy().reset_index(drop=True)
        self.period = seasonality_period
        
        # Decompose the series using STL to isolate trend, seasonal, and residuals
        stl = STL(self.df["value"], period=self.period, robust=True)
        self.result = stl.fit()
        
        self.trend = self.result.trend
        self.seasonal = self.result.seasonal
        self.residuals = self.result.resid
        self.sigma = np.std(self.residuals)

    def generate_level1_baseline(self, block_size: int = None) -> pd.DataFrame:
        """
        Level 1: Series without changes (FAR Evaluation).
        Retains the stable seasonal profile and trend level, but resamples residuals 
        using block bootstrap to preserve autocorrelation while removing any real drift.
        
        :param block_size: Size of blocks for block bootstrap (defaults to seasonality period).
        :return: Standardized DataFrame with synthetic drift-free time series ['timestamp', 'value'].
        """
        if block_size is None:
            block_size = self.period

        n = len(self.residuals)
        res_array = self.residuals.values
        bootstrap_resid = []

        # Block bootstrap resampling loop
        while len(bootstrap_resid) < n:
            start_idx = np.random.randint(0, n - block_size)
            block = res_array[start_idx : start_idx + block_size]
            bootstrap_resid.extend(block)

        bootstrap_resid = np.array(bootstrap_resid[:n])

        # Reconstruct series: Trend level + Stable Seasonal Profile + Bootstrapped Residuals
        # Replace trend with its mean/median level or keep stable trend to test false alarms cleanly
        stable_trend = np.full_like(self.trend, fill_value=np.median(self.trend))
        synthetic_values = stable_trend + self.seasonal.values + bootstrap_resid

        res_df = self.df.copy()
        res_df["value"] = synthetic_values.astype("float64")
        return res_df

    def inject_level2_drift(
        self, 
        drift_type: str = "abrupt", 
        change_point_idx: int = None, 
        magnitude_sigma: float = 1.0
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Level 2: Controlled Semi-Synthetic Drift Injection.
        Injects known changes into the original or baseline series with precise parameters.
        
        :param drift_type: Type of drift ('abrupt', 'gradual', 'seasonal_amplitude').
        :param change_point_idx: Index where drift begins (defaults to 50% of the series length).
        :param magnitude_sigma: Magnitude multiplier in standard deviations (e.g., 0.5, 1.0, 2.0, 3.0).
        :return: Tuple of (Modified DataFrame, Metadata dictionary containing ground truth).
        """
        if change_point_idx is None:
            change_point_idx = len(self.df) // 2

        df_out = self.df.copy()
        values = df_out["value"].values.copy()
        delta = magnitude_sigma * self.sigma

        metadata = {
            "drift_type": drift_type,
            "change_point_idx": change_point_idx,
            "change_point_timestamp": df_out.loc[change_point_idx, "timestamp"] if "timestamp" in df_out.columns else None,
            "magnitude_sigma": magnitude_sigma,
            "absolute_delta": delta
        }

        if drift_type == "abrupt":
            # Abrupt level shift: sudden constant offset starting from change point
            values[change_point_idx:] += delta

        elif drift_type == "gradual":
            # Gradual linear trend drift spanning from change point to the end
            n_duration = len(values) - change_point_idx
            ramp = np.linspace(0, delta, n_duration)
            values[change_point_idx:] += ramp

        elif drift_type == "seasonal_amplitude":
            # Change in seasonal amplitude
            # Multiplies the seasonal component amplitude after the change point
            future_indices = np.arange(change_point_idx, len(values))
            values[future_indices] += self.seasonal.values[future_indices] * magnitude_sigma

        else:
            raise ValueError(f"Unknown drift type '{drift_type}'. Supported: 'abrupt', 'gradual', 'seasonal_amplitude'.")

        df_out["value"] = values.astype("float64")
        return df_out, metadata