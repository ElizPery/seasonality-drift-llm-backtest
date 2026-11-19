"""One evaluation function for every detector of stream A.

The two runners of stream A score the alarms in different ways. `run_residual_detector_evaluation` uses a tolerance
window of 50 points and counts every unmatched alarm as a false positive. `run_baseline_evaluation` uses 40 points,
on level 2 counts only the alarms before the change as false positives, and measures the delay to the first alarm
after the change, also when it falls outside the window. `score_alarms` applies one rule to every detector:

* a series without changes: the number of alarms and the false alarms per 1000 points;
* a series with one change: the change is detected if an alarm falls into [change, change + tolerance]; the delay is
  the distance from the change to the first such alarm; every other alarm, before or after the change, is a false
  positive.

The module also holds the detector calls that the experiments of `run_common_evaluation` share:

* `residual_alarms`: the residual detector with its settings of the single-series mode, with another multiplier or
  with a subset of its three scores. The residuals do not depend on the multiplier or on the scores, so they are
  computed once per series and passed in.
* `baseline_alarms`: the baseline detectors with the settings of `run_baseline_evaluation`. KSWIN of River tests a
  random sample of its window, and its alarms change from run to run: on the generated series its recall lies between
  0.05 and 0.58 depending on the seed. It therefore gets a seed here and is run with every seed of KSWIN_SEEDS; the
  summary tables average over them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.detector.baseline_detectors import BaselineDetectorWrapper
from src.detector.residual_detector import ResidualConformalDriftDetector
from src.detector.run_residual_detector_evaluation import get_detector_params

TOLERANCE = 40  # points after the change in which an alarm counts as the detection
SCORES = ("point", "mean", "spread")  # the three conditions of the residual detector, in the order of its code
KSWIN_SEEDS = tuple(range(20))
BASELINES = (  # the settings of run_baseline_evaluation
    {"method": "ks", "window_size": 100, "threshold": 0.01},
    {"method": "psi", "window_size": 100, "threshold": 0.1},
    {"method": "wasserstein", "window_size": 100, "threshold": 0.5},
    {"method": "adwin", "threshold": 0.002},
    {"method": "page_hinkley", "threshold": 50.0},
    {"method": "kswin", "window_size": 100, "threshold": 0.01},
)


def score_alarms(alarms, n: int, change: int | None = None, tolerance: int = TOLERANCE) -> dict:
    """Metrics of one detector on one series from the positions of its alarms.

    `n` is the length of the series. Without a change the result has `alarms` and `far` (alarms per 1000 points).
    With a change at position `change` it has `alarms`, `precision`, `recall`, `f1`, `delay` and `false_alarms`:
    the first alarm in [change, change + tolerance] is the detection, every other alarm is a false positive.
    """
    alarms = sorted(int(i) for i in alarms)
    if change is None:
        return {"alarms": len(alarms), "far": 1000 * len(alarms) / n}
    hits = [i for i in alarms if 0 <= i - change <= tolerance]
    tp = 1 if hits else 0
    fp = len(alarms) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = float(tp)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"alarms": len(alarms), "precision": precision, "recall": recall, "f1": f1,
            "delay": float(hits[0] - change) if hits else np.nan, "false_alarms": fp}


def detector_params(period: int, n: int, dataset: str = "uci_electricity") -> dict:
    """Settings of the residual detector in its single-series mode (the mode of the comparison notebook)."""
    params = get_detector_params(dataset, period, n, mode="single")
    params.pop("tolerance_window")
    return params


class CachedResidualDetector(ResidualConformalDriftDetector):
    """The residual detector on residuals that are already computed (they do not depend on the multiplier)."""

    def __init__(self, residuals: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self._residuals = residuals

    def _compute_causal_residuals(self, values: np.ndarray) -> np.ndarray:
        return self._residuals


def causal_residuals(df: pd.DataFrame, period: int, dataset: str = "uci_electricity") -> np.ndarray:
    """Causal STL residuals of the series, as the residual detector computes them."""
    detector = ResidualConformalDriftDetector(seasonality_period=period, **detector_params(period, len(df), dataset))
    return detector._compute_causal_residuals(df["value"].to_numpy(float))


def score_violations(detector: ResidualConformalDriftDetector, residuals: np.ndarray, i: int) -> tuple[bool, bool, bool]:
    """Which conditions of `fit_predict` hold at position i: point deviation, mean shift, spread (see SCORES)."""
    cal = residuals[i - detector.cal_window:i]
    median = np.median(cal)
    deviations = np.abs(cal - median)
    q = int(np.clip(int(np.ceil((1.0 - detector.alpha) * (len(deviations) + 1))) - 1, 0, len(deviations) - 1))
    threshold = np.sort(deviations)[q] * detector.multiplier
    point = bool(np.abs(residuals[i] - median) > threshold)
    if i < detector.cal_window + detector.var_window:
        return point, False, False
    local = residuals[i - detector.var_window + 1:i + 1]
    mean = bool(np.abs(np.mean(local) - median) > (threshold / np.sqrt(detector.var_window)) * 1.35)
    local_iqr = np.percentile(local, 75) - np.percentile(local, 25)
    baseline_iqr = np.percentile(cal, 75) - np.percentile(cal, 25)
    spread = bool(local_iqr / (baseline_iqr + 1e-8) > 1.8 * detector.multiplier)
    return point, mean, spread


def residual_alarms(df: pd.DataFrame, period: int, residuals: np.ndarray | None = None, multiplier: float | None = None,
                    scores=SCORES, dataset: str = "uci_electricity", use_detector: bool = True) -> list[int]:
    """Alarm positions of the residual detector.

    With all three scores the alarms come from `fit_predict` of the detector itself. With a subset of the scores
    (or with `use_detector=False`) they come from the same loop written here, in which only the chosen conditions
    can start an alarm; the consecutive violations and the cooldown work as in the detector.
    """
    params = detector_params(period, len(df), dataset)
    if multiplier is not None:
        params["sensitivity_multiplier"] = multiplier
    if residuals is None:
        residuals = causal_residuals(df, period, dataset)
    detector = CachedResidualDetector(residuals, seasonality_period=period, **params)
    unknown = set(scores) - set(SCORES)
    if unknown:
        raise ValueError(f"Unknown scores {sorted(unknown)}; choose from {SCORES}.")
    if use_detector and set(scores) == set(SCORES):
        return [a["index"] for a in detector.fit_predict(df)]
    use = [name in scores for name in SCORES]
    alarms, consecutive, last = [], 0, -detector.cooldown_steps
    for i in range(detector.cal_window, len(residuals)):
        violated = any(v and u for v, u in zip(score_violations(detector, residuals, i), use))
        consecutive = consecutive + 1 if violated else 0
        if consecutive >= detector.min_consecutive and i - last >= detector.cooldown_steps:
            alarms.append(i)
            last, consecutive = i, 0
    return alarms


def baseline_alarms(df: pd.DataFrame, config: dict, seed: int = 0) -> list[int]:
    """Alarm positions of one baseline detector (an entry of BASELINES) on the raw series; `seed` is used by KSWIN."""
    if config["method"] == "kswin":  # the wrapper of stream A builds KSWIN without a seed
        from river import drift

        detector = drift.KSWIN(alpha=config["threshold"], window_size=config["window_size"], seed=seed)
        alarms = []
        for i, value in enumerate(df["value"].to_numpy(float)):
            detector.update(value)
            if detector.drift_detected:
                alarms.append(i)
        return alarms
    return [a["index"] for a in BaselineDetectorWrapper(**config).run(df)]


def baseline_runs(df: pd.DataFrame):
    """Every baseline detector on the series: (method, seed, alarm positions), KSWIN once per seed of KSWIN_SEEDS."""
    for config in BASELINES:
        for seed in KSWIN_SEEDS if config["method"] == "kswin" else (np.nan,):
            yield config["method"], seed, baseline_alarms(df, config, seed=0 if np.isnan(seed) else seed)


def mean_over_seeds(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """One row per `keys`: the metrics averaged over the seeds of KSWIN (the other detectors have one row already)."""
    table = table.drop(columns="seed", errors="ignore")
    return table.groupby(keys, dropna=False, sort=False).mean(numeric_only=True).reset_index()
