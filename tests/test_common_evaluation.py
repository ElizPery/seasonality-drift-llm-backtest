"""Tests of the common evaluation function of stream A and of the detector calls around it. Run from the repository
root: python -m pytest tests"""
import numpy as np
import pandas as pd
import pytest

from src.detector.common_evaluation import (BASELINES, KSWIN_SEEDS, SCORES, TOLERANCE, baseline_alarms, baseline_runs,
                                            causal_residuals, detector_params, mean_over_seeds, residual_alarms,
                                            score_alarms)
from src.detector.residual_detector import ResidualConformalDriftDetector
from src.detector.run_common_evaluation import benchmark_table, real_rows, real_table, scores_table

PERIOD = 24


def seasonal_series(n: int = 720, change: int | None = 480, seed: int = 0) -> pd.DataFrame:
    """Hourly series with a daily profile and noise; from `change` on the noise is three times wider."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    noise = rng.normal(0, 1, n)
    if change is not None:
        noise[change:] *= 3
    value = 50 + 10 * np.sin(2 * np.pi * t / PERIOD) + noise
    return pd.DataFrame({"timestamp": pd.date_range("2013-01-01", periods=n, freq="h"), "value": value})


def test_series_without_changes_gives_alarms_per_1000_points():
    assert score_alarms([10, 500, 900], n=2000) == {"alarms": 3, "far": 1.5}
    assert score_alarms([], n=2000) == {"alarms": 0, "far": 0.0}


def test_first_alarm_in_the_window_is_the_detection_and_every_other_alarm_is_a_false_positive():
    change = 1000
    r = score_alarms([100, change + 7, change + 30, change + 300], n=2000, change=change)
    assert r["recall"] == 1.0 and r["delay"] == 7.0
    assert r["false_alarms"] == 3  # before the change, the second alarm in the window, after the window
    assert r["precision"] == pytest.approx(0.25) and r["f1"] == pytest.approx(0.4)


def test_window_includes_both_ends_and_no_alarm_before_the_change_detects_it():
    change = 1000
    assert score_alarms([change], 2000, change)["delay"] == 0.0
    assert score_alarms([change + TOLERANCE], 2000, change)["recall"] == 1.0
    late = score_alarms([change + TOLERANCE + 1], 2000, change)
    assert late["recall"] == 0.0 and np.isnan(late["delay"]) and late["false_alarms"] == 1 and late["f1"] == 0.0
    early = score_alarms([change - 1], 2000, change)
    assert early["recall"] == 0.0 and early["precision"] == 0.0 and early["false_alarms"] == 1
    silent = score_alarms([], 2000, change)
    assert silent["recall"] == 0.0 and silent["precision"] == 0.0 and silent["alarms"] == 0
    assert score_alarms([change + 60], 2000, change, tolerance=60)["recall"] == 1.0


def test_score_does_not_depend_on_the_order_of_the_alarms():
    assert score_alarms([1040, 5, 1010], 2000, 1000) == score_alarms([5, 1010, 1040], 2000, 1000)


def test_loop_with_all_three_scores_gives_the_alarms_of_the_detector():
    df = seasonal_series()
    params = detector_params(PERIOD, len(df))
    expected = [a["index"] for a in ResidualConformalDriftDetector(seasonality_period=PERIOD, **params).fit_predict(df)]
    assert len(expected) > 0
    assert residual_alarms(df, PERIOD) == expected
    residuals = causal_residuals(df, PERIOD)
    assert residual_alarms(df, PERIOD, residuals) == expected
    assert residual_alarms(df, PERIOD, residuals, use_detector=False) == expected


def test_stored_residuals_with_another_multiplier_equal_a_new_detector_with_it():
    df = seasonal_series()
    residuals = causal_residuals(df, PERIOD)
    for mult in (1.2, 3.0):
        params = dict(detector_params(PERIOD, len(df)), sensitivity_multiplier=mult)
        expected = [a["index"] for a in ResidualConformalDriftDetector(seasonality_period=PERIOD, **params).fit_predict(df)]
        assert residual_alarms(df, PERIOD, residuals, multiplier=mult) == expected
        assert residual_alarms(df, PERIOD, residuals, multiplier=mult, use_detector=False) == expected


def test_one_score_at_a_time():
    df = seasonal_series()
    residuals = causal_residuals(df, PERIOD)
    assert residual_alarms(df, PERIOD, residuals, scores=()) == []
    full = residual_alarms(df, PERIOD, residuals)
    single = {name: residual_alarms(df, PERIOD, residuals, scores=(name,)) for name in SCORES}
    # the full detector raises an alarm when any score is violated, so its first alarm is never later than theirs
    assert all(alarms and alarms[0] >= full[0] for alarms in single.values())
    assert full[0] >= 480  # the wider noise starts at 480
    with pytest.raises(ValueError):
        residual_alarms(df, PERIOD, residuals, scores=("level",))


def test_baselines_repeat_exactly_and_kswin_depends_on_its_seed_only():
    df = seasonal_series(n=1500, change=None)
    for config in BASELINES:
        assert baseline_alarms(df, config) == baseline_alarms(df, config)
    kswin = next(c for c in BASELINES if c["method"] == "kswin")
    seeds = {tuple(baseline_alarms(df, kswin, seed=seed)) for seed in range(5)}
    assert len(seeds) > 1  # without a fixed seed two runs of KSWIN differ


def test_every_baseline_runs_once_and_kswin_once_per_seed():
    df = seasonal_series(n=600, change=None)
    runs = list(baseline_runs(df))
    methods = [method for method, _, _ in runs]
    assert methods.count("kswin") == len(KSWIN_SEEDS) and len(runs) == len(BASELINES) - 1 + len(KSWIN_SEEDS)
    assert [seed for method, seed, _ in runs if method == "kswin"] == list(KSWIN_SEEDS)
    assert all(np.isnan(seed) for method, seed, _ in runs if method != "kswin")
    rows = pd.DataFrame([dict(series="a", method=m, mult=np.nan, seed=s, alarms=len(a)) for m, s, a in runs])
    table = mean_over_seeds(rows, ["series", "method", "mult"])
    assert len(table) == len(BASELINES) and "seed" not in table.columns
    kswin = rows[rows["method"] == "kswin"]["alarms"].mean()
    assert table.set_index("method").loc["kswin", "alarms"] == pytest.approx(kswin)


def test_real_series_rows_count_alarms_from_deployment_only():
    row = dict(source="entsoe", series="XX", t_deploy=1000, t_change=2000, t_stop=3000)
    r = real_rows([10, 1100, 1900, 2050, 2100, 2500, 3500], row, "adwin")
    assert r["alarms"] == 5 and r["pre_hours"] == 1000
    assert r["far_pre"] == pytest.approx(2.0)  # two alarms in the 1000 hours before the change
    assert r["recall"] == 1.0 and r["delay"] == 50.0 and r["first_after"] == 50.0
    assert r["week_after"] == 2  # 2050 and 2100
    late = real_rows([1500, 2400], row, "adwin")
    assert late["recall"] == 0.0 and np.isnan(late["delay"]) and late["first_after"] == 400.0 and late["week_after"] == 0


def test_summary_tables():
    rows = [dict(series="a", level=1, kind="none", mag=0.0, draw=d, method="adwin", mult=np.nan, far=f)
            for d, f in enumerate([1.0, 3.0])]
    rows += [dict(series="a", level=2, kind="abrupt", mag=m, draw=0, method="adwin", mult=np.nan, recall=r, precision=p,
                  f1=p, delay=d) for m, r, p, d in [(1.0, 1.0, 0.5, 10.0), (2.0, 0.0, 0.0, np.nan)]]
    t = benchmark_table(pd.DataFrame(rows)).iloc[0]
    assert t["far"] == 2.0 and t["far_std"] == pytest.approx(np.sqrt(2))
    assert t["recall"] == 0.5 and t["precision"] == 0.25 and t["delay"] == 10.0  # delay over the detected changes

    real = pd.DataFrame([dict(source="entsoe", series=s, method="adwin", mult=np.nan, far_pre=f, recall=r, delay=d,
                              week_after=w) for s, f, r, d, w in [("a", 10.0, 1.0, 20.0, 2), ("b", 20.0, 0.0, np.nan, 4)]])
    t = real_table(real).iloc[0]
    assert t["far_pre"] == 15.0 and t["detected"] == 1.0 and t["series"] == 2 and t["delay"] == 20.0
    assert t["ratio"] == pytest.approx(3 / (15.0 / 1000 * 168))
    silent = real_table(real.assign(far_pre=0.0)).iloc[0]
    assert np.isnan(silent["ratio"])  # no alarms before the change: the ratio is not defined

    far = [("full", 5.0), ("point", 3.0), ("mean", 2.0), ("spread", 4.0)]
    recall = [("full", 1.0), ("point", 0.0), ("mean", 1.0), ("spread", 0.0)]
    scores = pd.DataFrame([dict(variant=v, level=1, kind="none", far=f, recall=np.nan) for v, f in far]
                          + [dict(variant=v, level=2, kind="abrupt", far=np.nan, recall=r) for v, r in recall])
    t = scores_table(scores)
    assert list(t.index) == ["full", "point", "mean", "spread"]
    assert t.loc["mean", "abrupt"] == 1.0 and t.loc["spread", "far"] == 4.0
