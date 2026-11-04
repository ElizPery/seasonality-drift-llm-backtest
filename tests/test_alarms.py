"""Tests of the alarm strategies (review on detector alarms) and of the stand-in detectors and alerts file."""
import numpy as np
import pandas as pd
import pytest

from src.data.frames import series_table
from src.recommendations.alarms import (DETECTORS, adwin_alarms, calibrate, conformal_alarms, control_windows, detect_all,
                                        false_alarm_rate, profile_residuals, read_alerts)
from src.recommendations.backtester import ALARM_STRATEGIES, STRATEGIES, backtest, evaluate_series
from src.recommendations.forecast import BASE_FEATURES, DAY, WEEK
from tests.test_backtester import N, T_CHANGE, T_DEPLOY, toy

BY_NAME = {st.name: st for st in STRATEGIES + ALARM_STRATEGIES}


def series_dict(y, F, start="2013-09-02 00:00", **labels):
    base = dict(source="synthetic", series="toy/s0", client="toy", seed=0, scenario="level_shift", magnitude=2.0,
                y=y, F=F, t_deploy=T_DEPLOY, t_change=T_CHANGE, t_stop=N, base_features=BASE_FEATURES, start=start)
    return dict(base, **labels)


def test_alarm_strategy_without_alarms_is_the_untouched_model():
    y, F = toy(shift=30)
    nothing = backtest(y, F, T_DEPLOY, BY_NAME["nothing"], BASE_FEATURES)
    r = backtest(y, F, T_DEPLOY, BY_NAME["alarm_retrain"], BASE_FEATURES, alarms=[], detector="x")
    assert r["retrains"] == 0 and r["applied"] == [] and r["alarms"] == 0
    assert np.allclose(r["pred"], nothing["pred"], equal_nan=True)


def test_alarm_retrain_reviews_at_the_first_origin_after_an_alarm_and_respects_the_cooldown():
    y, F = toy(shift=30)
    alarms = [T_DEPLOY - 5, T_CHANGE + 5, T_CHANGE + 30, T_CHANGE + 10 * DAY + 3]  # before deployment, change, cooldown, later
    r = backtest(y, F, T_DEPLOY, BY_NAME["alarm_retrain"], BASE_FEATURES, alarms=alarms, detector="x")
    assert r["applied"] == [T_CHANGE + DAY, T_CHANGE + 11 * DAY]  # the second alarm falls inside the 7-day cooldown
    assert r["retrains"] == 2 and r["alarms"] == 3
    # the model changes from the review on: forecasts before it equal the untouched model's
    nothing = backtest(y, F, T_DEPLOY, BY_NAME["nothing"], BASE_FEATURES)
    assert np.allclose(r["pred"][: T_CHANGE + DAY], nothing["pred"][: T_CHANGE + DAY], equal_nan=True)
    assert not np.allclose(r["pred"][T_CHANGE + DAY : T_CHANGE + 2 * DAY], nothing["pred"][T_CHANGE + DAY : T_CHANGE + 2 * DAY])


def test_alarm_gate_decides_only_with_a_check_window_and_counts_the_review():
    y, F = toy(shift=30)
    early = backtest(y, F, T_DEPLOY, BY_NAME["alarm_gate"], BASE_FEATURES, alarms=[T_DEPLOY + 2], detector="x")
    assert early["decisions"] == 0 and early["retrains"] == 0  # no two weeks of own forecasts to check against yet
    r = backtest(y, F, T_DEPLOY, BY_NAME["alarm_gate"], BASE_FEATURES, alarms=[T_CHANGE + 3 * DAY], detector="x")
    assert r["decisions"] == 1 and r["proposed"] == 1 and r["check_fits"] > 0
    assert r["accepted"] == len(r["applied"]) <= 1


def test_evaluate_series_runs_an_alarm_strategy_once_per_detector():
    y, F = toy(shift=30)
    s = series_dict(y, F)
    strategies = (BY_NAME["nothing"], BY_NAME["alarm_retrain"])
    rows = pd.DataFrame(evaluate_series(s, strategies=strategies, alerts={"a": np.array([T_CHANGE + 5]), "b": np.array([])}))
    assert sorted(rows.strategy) == ["alarm_retrain", "alarm_retrain", "naive", "nothing"]
    a = rows[(rows.strategy == "alarm_retrain") & (rows.detector == "a")].iloc[0]
    b = rows[(rows.strategy == "alarm_retrain") & (rows.detector == "b")].iloc[0]
    assert a.retrains == 1 and a.alarms == 1 and a.acts_pre == 0 and a.reaction_h == DAY
    assert b.retrains == 0 and b.alarms == 0 and np.isnan(b.reaction_h)
    assert rows[rows.strategy == "nothing"].detector.isna().all()
    assert "start" not in rows.columns


def test_job_runs_every_detector_on_every_series_even_without_alarms():
    from src.recommendations.backtester import _job

    y, F = toy(shift=30)
    s = series_dict(y, F)
    alerts = {"synthetic/toy/s0/level_shift": {"a": np.array([T_CHANGE + 5])}}  # detector "b" has no alarms here
    rows = pd.DataFrame(_job(s, strategies=(BY_NAME["nothing"], BY_NAME["alarm_retrain"]), alerts=alerts, detectors=("a", "b")))
    on = rows[rows.strategy == "alarm_retrain"].set_index("detector")
    assert sorted(on.index) == ["a", "b"] and on.loc["b", "retrains"] == 0 and on.loc["a", "retrains"] == 1


def test_profile_conformal_uses_only_the_past_and_fires_after_a_shift():
    y, F = toy(shift=40)
    r = profile_residuals(y)
    assert np.isnan(r[: 4 * WEEK]).all() and np.isfinite(r[4 * WEEK :]).all()
    alarms = conformal_alarms(r, T_DEPLOY, mult=2.0)
    after = alarms["index"][alarms["index"] >= T_CHANGE]
    assert len(after) > 0 and after.min() - T_CHANGE < 2 * DAY
    assert (alarms["score"] > alarms["threshold"]).all()
    assert (np.diff(alarms["index"].to_numpy()) >= DAY).all()  # one-day cooldown
    y2 = y.copy()
    y2[T_CHANGE + DAY :] += 1000.0
    alarms2 = conformal_alarms(profile_residuals(y2), T_DEPLOY, mult=2.0)
    cut = T_CHANGE + DAY
    assert alarms[alarms["index"] < cut].equals(alarms2[alarms2["index"] < cut])


def test_adwin_stand_in_fires_after_a_shift():
    pytest.importorskip("river")
    y, F = toy(shift=40)
    alarms = adwin_alarms(profile_residuals(y), T_DEPLOY, delta=0.002)
    after = alarms["index"][alarms["index"] >= T_CHANGE]
    assert len(after) > 0 and after.min() - T_CHANGE < 3 * DAY


def test_calibration_meets_the_target_false_alarm_rate_on_control_windows():
    controls = [(toy(seed=k)[0], T_DEPLOY, N) for k in range(3)]
    value, rate = calibrate("profile_conformal", controls, target_days=14)
    assert rate == false_alarm_rate("profile_conformal", controls, value)
    assert rate <= 1 / (14 * DAY) or value == DETECTORS["profile_conformal"][2][-1]
    assert false_alarm_rate("profile_conformal", controls, 1.0) >= rate  # a more sensitive setting alarms at least as often
    # alarms outside the window do not count: the window before the shift of a changed series is clean
    y, _ = toy(shift=40)
    before = false_alarm_rate("profile_conformal", [(y, T_DEPLOY, T_CHANGE)], 2.0)
    whole = false_alarm_rate("profile_conformal", [(y, T_DEPLOY, N)], 2.0)
    assert before <= whole


def test_control_windows_per_source():
    y, F = toy()
    series = [series_dict(y, F, scenario="none", magnitude=0.0), series_dict(y, F),
              series_dict(y, F, source="seismic", series="BE.UCC", scenario="lockdown_2020")]
    w = control_windows(series)
    assert sorted(w) == ["seismic", "synthetic"] and len(w["synthetic"]) == 1  # the changed synthetic series is not a control
    assert w["synthetic"][0][1:] == (T_DEPLOY, N) and w["seismic"][0][1:] == (T_DEPLOY, T_CHANGE)


def test_alerts_file_round_trip_and_timestamp_matching(tmp_path):
    y, F = toy(shift=40)
    utc = series_dict(y, F, start="2017-01-02 00:00:00+00:00", source="entsoe", series="IT", scenario="lockdown_2020")
    naive = series_dict(y, F)
    manifest = series_table([utc, naive])
    alerts = detect_all([utc, naive], {"entsoe": {"profile_conformal": 2.0}, "synthetic": {"profile_conformal": 2.0}})
    assert set(alerts.columns) == {"id", "source", "series", "scenario", "method", "timestamp", "index", "score", "threshold"}
    assert alerts["timestamp"].iloc[0] == pd.Timestamp("2017-01-02 00:00:00+00:00") + pd.Timedelta(hours=int(alerts["index"].iloc[0]))
    path = tmp_path / "alerts.csv"
    alerts.to_csv(path, index=False)
    back = read_alerts(path, manifest)
    for s in (utc, naive):
        sid = f"{s['source']}/{s['series']}/{s['scenario']}"
        expect = alerts[alerts["id"] == sid]["index"].to_numpy()
        assert np.array_equal(back[sid]["profile_conformal"], expect)
    # rows with an index only, outside the series, or with an unknown id
    extra = pd.DataFrame([dict(id="entsoe/IT/lockdown_2020", method="m", timestamp=None, index=7),
                          dict(id="entsoe/IT/lockdown_2020", method="m", timestamp="2030-01-01 00:00:00+00:00", index=None),
                          dict(id="nope/x/y", method="m", timestamp="2017-01-02 05:00:00+00:00", index=None),
                          dict(id="synthetic/toy/s0/level_shift", method="m", timestamp="2013-09-02 05:00", index=None)])
    extra.to_csv(path, index=False)
    back = read_alerts(path, manifest)
    assert np.array_equal(back["entsoe/IT/lockdown_2020"]["m"], [7]) and "nope/x/y" not in back
    assert np.array_equal(back["synthetic/toy/s0/level_shift"]["m"], [5])
