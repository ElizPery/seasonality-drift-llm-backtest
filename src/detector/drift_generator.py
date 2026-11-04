"""
Drift Generator Module for Seasonality-Aware Drift Detection System.
Implements the 3-level validation methodology:
- Level 1: Control series without changes (FAR evaluation): a stable seasonal profile (daily and weekly for
  hourly data), a constant level instead of the trend, and residuals resampled with a seasonal block bootstrap.
- Level 2: Semi-synthetic drift injection into the control series (abrupt, gradual, seasonal amplitude,
  variance) with controlled magnitudes in standard deviations of residuals (sigma).
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL, STL
from typing import Tuple, Dict, Any, Optional, Sequence


def clean_residuals(resid: np.ndarray, cycle: int, winsor: float = 4.0) -> np.ndarray:
    """
    Make the residual process free of changes by construction: clip gross outliers at `winsor` robust sigmas
    and give every cycle (a week for hourly data) the same spread, the median spread over the cycles.
    Autocorrelation and the variance pattern inside a cycle are kept. Only whole cycles are returned.
    """
    r = resid - np.median(resid)
    robust_sigma = 1.4826 * np.median(np.abs(r))
    r = np.clip(r, -winsor * robust_sigma, winsor * robust_sigma)
    n_cycles = len(r) // cycle
    cycles = r[: n_cycles * cycle].reshape(n_cycles, cycle)
    cycles = cycles - cycles.mean(axis=1, keepdims=True)
    sd = cycles.std(axis=1, ddof=1)
    cycles = cycles * (np.median(sd) / sd)[:, None]
    out = cycles.ravel()
    return out - out.mean()


class DriftGenerator:
    """Generates synthetic drift and clean baseline series based on project specifications."""

    def __init__(
        self,
        series: pd.DataFrame,
        seasonality_period: int,
        periods: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the generator with a reference time series.

        :param series: Standardized DataFrame with columns ['timestamp', 'value']
        :param seasonality_period: Period of seasonality (e.g., 24 for hourly daily cycle, 7 for daily weekly cycle)
        :param periods: All seasonal periods, e.g. (24, 168) for hourly data with a daily and a weekly cycle
            (MSTL). Defaults to (seasonality_period,) (STL). The longest period is the cycle of the seasonal
            profile, the shortest one is the bootstrap block.
        :param seed: Seed of the random generator used when a method gets no seed of its own.
        """
        self.df = series.copy().reset_index(drop=True)
        self.period = seasonality_period
        self.periods = tuple(periods) if periods else (seasonality_period,)
        self.cycle = max(self.periods)
        self.block = min(self.periods)
        if self.cycle % self.block:
            raise ValueError(f"The longest period {self.cycle} must be a multiple of the shortest {self.block}.")
        self.rng = np.random.default_rng(seed)

        # Decompose the series to isolate trend, seasonal, and residuals
        values = self.df["value"].to_numpy(dtype=float)
        if len(self.periods) > 1:
            result = MSTL(values, periods=self.periods, stl_kwargs={"robust": True}).fit()
            seasonal = np.asarray(result.seasonal)
            seasonal = seasonal.sum(axis=1) if seasonal.ndim == 2 else seasonal
        else:
            result = STL(values, period=self.period, robust=True).fit()
            seasonal = np.asarray(result.seasonal)
        self.result = result
        trend = np.asarray(result.trend)

        self.trend = pd.Series(trend)
        self.seasonal = pd.Series(seasonal)
        # Residuals without outliers and spread changes (whole cycles only)
        residuals = clean_residuals(values - seasonal - trend, self.cycle)
        self.residuals = pd.Series(residuals)
        self.sigma = float(residuals.std(ddof=1))

        # Stable seasonal profile: the seasonal component averaged over the cycles (mean zero),
        # and a constant level instead of the trend
        n_cycles = len(values) // self.cycle
        if n_cycles < 2:
            raise ValueError(f"The series must hold at least two cycles of {self.cycle} points.")
        profile = seasonal[: n_cycles * self.cycle].reshape(n_cycles, self.cycle).mean(axis=0)
        self.profile = profile - profile.mean()
        self.level = float(np.median(trend))

        self.baseline: Optional[pd.DataFrame] = None

    def _block_bootstrap(self, n: int, rng: np.random.Generator, block_size: int, start: int = 0) -> np.ndarray:
        """
        Seasonal block bootstrap: block d of the output is the block at the same position inside the cycle
        (the same weekday for daily blocks in a weekly cycle) of a randomly chosen source cycle
        (Politis 2001; Dudek et al. 2014). Keeps the autocorrelation inside a block and the variance
        pattern inside a cycle. `start` is the index of the first output block inside the cycle.
        """
        res_array = self.residuals.to_numpy()
        n_cycles = len(res_array) // self.cycle
        per_cycle = self.cycle // block_size
        n_blocks = int(np.ceil(n / block_size))
        picks = rng.integers(0, n_cycles, size=n_blocks)
        blocks = [
            res_array[c * self.cycle + ((d + start) % per_cycle) * block_size :][:block_size]
            for d, c in enumerate(picks)
        ]
        return np.concatenate(blocks)[:n]

    def _timestamps(self, n: int) -> pd.Series:
        """Timestamps of an output of length `n` starting where the reference series starts."""
        if n == len(self.df):
            return self.df["timestamp"].copy()
        ts = pd.to_datetime(self.df["timestamp"])
        return pd.Series(pd.date_range(ts.iloc[0], periods=n, freq=ts.diff().median()))

    def generate_level1_baseline(
        self, block_size: int = None, n: Optional[int] = None, seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Level 1: Series without changes (FAR Evaluation).
        Stable seasonal profile + constant level + residuals resampled with a seasonal block bootstrap,
        so the series has no changes by construction.

        :param block_size: Size of blocks for block bootstrap (defaults to the shortest period); must divide the cycle.
        :param n: Length of the output (defaults to the length of the reference series).
        :param seed: Seed of this baseline (defaults to the generator's own random state).
        :return: Standardized DataFrame with synthetic drift-free time series ['timestamp', 'value'].
        """
        block_size = block_size or self.block
        if self.cycle % block_size:
            raise ValueError(f"block_size {block_size} must divide the cycle {self.cycle}.")
        n = n or len(self.df)
        rng = self.rng if seed is None else np.random.default_rng(seed)

        position = np.arange(n) % self.cycle
        bootstrap_resid = self._block_bootstrap(n, rng, block_size)
        synthetic_values = self.level + self.profile[position] + bootstrap_resid

        res_df = pd.DataFrame({"timestamp": self._timestamps(n), "value": synthetic_values.astype("float64")})
        self.baseline = res_df
        return res_df

    def inject_level2_drift(
        self,
        drift_type: str = "abrupt",
        change_point_idx: int = None,
        magnitude_sigma: float = 1.0,
        base: Optional[pd.DataFrame] = None,
        ramp_len: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Level 2: Controlled Semi-Synthetic Drift Injection.
        Injects one known change into a control series, so the injected change is the only change in it.

        :param drift_type: 'abrupt' (level shift), 'gradual' (ramp to the new level over `ramp_len` points),
            'seasonal_amplitude' (a larger seasonal profile; the added part has standard deviation
            magnitude_sigma * sigma) or 'variance' (residual noise magnitude_sigma times stronger).
        :param change_point_idx: Index where drift begins (defaults to 50% of the series length).
        :param magnitude_sigma: Magnitude in standard deviations of residuals (e.g., 0.5, 1.0, 2.0, 3.0);
            for 'variance' the multiplier of the residual standard deviation (e.g., 2.0, 3.0).
        :param base: Series to inject into: a Level 1 baseline. Defaults to the reference series.
        :param ramp_len: Length of the 'gradual' ramp (defaults to one cycle, a week for hourly data).
        :param seed: Seed of the extra residuals of 'variance' (defaults to the generator's own random state).
        :return: Tuple of (Modified DataFrame, Metadata dictionary containing ground truth).
        """
        df_out = (self.df if base is None else base).copy().reset_index(drop=True)
        if change_point_idx is None:
            change_point_idx = len(df_out) // 2
        ramp_len = ramp_len or self.cycle

        values = df_out["value"].to_numpy(dtype=float).copy()
        t = np.arange(len(values))
        after = t >= change_point_idx
        delta = magnitude_sigma * self.sigma

        metadata = {
            "drift_type": drift_type,
            "change_point_idx": int(change_point_idx),
            "change_point_timestamp": df_out.loc[change_point_idx, "timestamp"] if "timestamp" in df_out.columns else None,
            "magnitude_sigma": magnitude_sigma,
            "absolute_delta": delta,
        }

        if drift_type == "abrupt":
            # Abrupt level shift: sudden constant offset starting from change point
            values[after] += delta

        elif drift_type == "gradual":
            # Linear ramp to the new level over `ramp_len` points, then the level stays
            ramp = np.clip((t - change_point_idx + 1) / ramp_len, 0, 1)
            values += delta * ramp * after
            metadata["ramp_len"] = int(ramp_len)

        elif drift_type == "seasonal_amplitude":
            # Larger seasonal profile after the change point, scaled so the added part has std = delta
            values[after] += self.profile[t[after] % self.cycle] * (delta / self.profile.std())

        elif drift_type == "variance":
            # Residual noise magnitude_sigma times stronger: extra bootstrapped residuals after the change point
            rng = self.rng if seed is None else np.random.default_rng(seed)
            extra = self._block_bootstrap(int(after.sum()), rng, self.block, start=change_point_idx // self.block)
            values[after] += extra * np.sqrt(max(magnitude_sigma**2 - 1.0, 0.0))
            metadata["absolute_delta"] = None

        else:
            raise ValueError(
                f"Unknown drift type '{drift_type}'. Supported: 'abrupt', 'gradual', 'seasonal_amplitude', 'variance'."
            )

        df_out["value"] = values.astype("float64")
        return df_out, metadata
