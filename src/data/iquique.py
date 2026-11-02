"""One waveform of the SeisBench Iquique dataset (Woollam et al., 2019): aftershocks of the 2014 Iquique
earthquake, 100 Hz, with the picked P and S arrivals as the change points. `seisbench` is imported on
first use, so the rest of the package loads without it."""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_iquique_trace(seed: int | None = None) -> tuple[pd.DataFrame, list[int]]:
    """A random trace with valid P and S arrivals, clipped before the S arrival, so the P arrival is the
    single change point. Returns the standard frame and the list with that change point position."""
    from seisbench.data import Iquique

    if seed is not None:
        np.random.seed(seed)

    dataset = Iquique()
    p_col = "trace_P_arrival_sample"
    s_col = "trace_S_arrival_sample"

    valid_mask = dataset.metadata[p_col].notna() & dataset.metadata[s_col].notna()
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        raise ValueError("No traces with both valid P and S arrivals found in the Iquique dataset.")

    chosen_idx = np.random.choice(valid_indices)
    row = dataset.metadata.iloc[chosen_idx]
    waveform = dataset.get_waveforms(chosen_idx)  # (components, samples) or (samples,)
    signal_values = waveform[0] if waveform.ndim > 1 else waveform  # first component (vertical)

    p_idx = int(row[p_col])
    s_idx = int(row[s_col])
    chosen_wave_type = np.random.choice(["P"])

    start_time_col = "trace_start_time" if "trace_start_time" in row else None
    delta_seconds = 0.01  # 100 Hz

    if chosen_wave_type == "P":
        # Clip from the start of the series to the S arrival: the P arrival stays at its index
        signal_values = signal_values[:s_idx]
        change_point_idx = p_idx
        start_dt = pd.to_datetime(row[start_time_col]) if (start_time_col and pd.notna(row[start_time_col])) else pd.Timestamp("2026-01-01")
    else:
        # Clip from the P arrival to the end: the S arrival moves to index s - p
        signal_values = signal_values[p_idx:]
        change_point_idx = s_idx - p_idx
        base_dt = pd.to_datetime(row[start_time_col]) if (start_time_col and pd.notna(row[start_time_col])) else pd.Timestamp("2026-01-01")
        start_dt = base_dt + pd.Timedelta(seconds=p_idx * delta_seconds)

    timestamps = pd.date_range(start=start_dt, periods=len(signal_values), freq=f"{int(delta_seconds * 1000)}ms")
    df = pd.DataFrame({"timestamp": timestamps, "value": signal_values})

    print(f"Loaded SeisBench Iquique trace #{chosen_idx}. Focus: {chosen_wave_type}-wave. Ground-truth CP index: {change_point_idx}")
    return df, [change_point_idx]
