"""Tests for the evaluation of the stream A detector: the position from which an alarm is known, the F1 score of the
TCPD benchmark and level 0 (the series of level 2 without the change). Run from the repository root:
python -m pytest tests"""
import numpy as np
import pandas as pd

from src.detector.residual_detector import ResidualConformalDriftDetector
from src.detector.run_residual_detector_evaluation import evaluate_residual_detector
from src.detector.scoring import tcpd_f1

PERIOD = 24


def seasonal_series(n=2400, shift_at=None, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 10 + 3 * np.sin(2 * np.pi * t / PERIOD) + rng.normal(0, 0.5, n)
    if shift_at is not None:
        y[shift_at:] += shift
    return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="h"), "value": y})


def detector():
    return ResidualConformalDriftDetector(seasonality_period=PERIOD, calibration_window=150, alpha=0.05,
                                          sensitivity_multiplier=1.6, cooldown_steps=40, min_consecutive_alerts=2,
                                          variance_window=16)


def test_an_alarm_is_known_at_the_end_of_its_block():
    d = detector()  # warm-up block: the first max(2 * 24, 150) = 150 positions, then blocks of 24
    assert d.known_index(10, 2400) == 149 and d.known_index(149, 2400) == 149
    assert d.known_index(150, 2400) == 173 and d.known_index(173, 2400) == 173 and d.known_index(174, 2400) == 197
    assert d.known_index(2390, 2400) == 2399  # the last block ends with the series
    assert d.known_index(50, 120) == 119      # a series shorter than the warm-up block is one fit


def test_alarms_known_by_a_position_do_not_depend_on_later_values():
    df = seasonal_series(shift_at=1200, shift=4.0)
    alerts = detector().fit_predict(df)
    assert alerts and all(a["known_index"] >= a["index"] for a in alerts)
    k = next(a["known_index"] for a in alerts if a["index"] >= 1200)
    later = df.copy()
    later.loc[k + 1:, "value"] += 50.0  # change everything after the position at which that alarm is known
    known = lambda al: [(a["index"], a["known_index"]) for a in al if a["known_index"] <= k]
    assert known(detector().fit_predict(later)) == known(alerts)


def test_tcpd_f1_reproduces_the_examples_of_tcpdbench():
    assert tcpd_f1([10, 20], {1: [10, 20], 2: [11, 20], 3: [10], 4: [0, 5]}) == 1.0
    assert abs(tcpd_f1([10], {1: [], 2: [10], 3: [50]}) - 0.9090909090909091) < 1e-12
    assert abs(tcpd_f1([], {1: [], 2: [10], 3: [50]}) - 0.8) < 1e-12


def test_level_0_is_scored_at_the_position_of_the_change_and_alarms_count_from_their_block_end(tmp_path):
    for name, df in (("placebo.csv", seasonal_series()), ("shift.csv", seasonal_series(shift_at=1200, shift=4.0))):
        df.to_csv(tmp_path / name, index=False)
    common = dict(dataset_name="toy", series_id="s", seasonality_period=PERIOD, change_point_idx=1200)
    manifest = [dict(common, level=0, filename="placebo.csv", drift_type="none", magnitude_sigma=0.0),
                dict(common, level=2, filename="shift.csv", drift_type="abrupt", magnitude_sigma=4.0)]
    res = evaluate_residual_detector(manifest, tmp_path).set_index("level")
    assert set(res.index) == {0, 2} and np.isnan(res.loc[0, "far_per_1000"])  # scored like level 2, not level 1
    alerts = detector().fit_predict(seasonal_series(shift_at=1200, shift=4.0))
    first = min(a["known_index"] for a in alerts if a["known_index"] >= 1200)
    if res.loc[2, "detected"] == 1:
        assert res.loc[2, "detection_delay"] == first - 1200
