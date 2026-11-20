"""Tests of the alarm strategies (review on detector alarms), of the detectors and of the alerts file."""
import numpy as np
import pandas as pd
import pytest

from src.data.frames import series_table
from src.recommendations.alarms import (DETECTORS, adwin_alarms, calibrate, control_windows, detect_all,
                                        false_alarm_rate, profile_residuals, read_alerts)
from src.recommendations.backtester import (ALARM_STRATEGIES, FOLLOW_UP_STRATEGIES, STRATEGIES, StrategyRun, backtest,
                                            evaluate_series)
from src.recommendations.forecast import BASE_FEATURES, DAY, WEEK
from src.recommendations.llm_client import Recommendation
from tests.test_backtester import N, T_CHANGE, T_DEPLOY, toy

BY_NAME = {st.name: st for st in STRATEGIES + ALARM_STRATEGIES + FOLLOW_UP_STRATEGIES}


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


def test_profile_residuals_use_only_the_past():
    y, _ = toy(shift=40)
    r = profile_residuals(y)
    assert np.isnan(r[: 4 * WEEK]).all() and np.isfinite(r[4 * WEEK :]).all()
    y2 = y.copy()
    y2[T_CHANGE + DAY :] += 1000.0
    cut = T_CHANGE + DAY
    assert np.array_equal(profile_residuals(y2)[:cut], r[:cut], equal_nan=True)


def test_adwin_fires_after_a_shift():
    pytest.importorskip("river")
    y, F = toy(shift=40)
    alarms = adwin_alarms(profile_residuals(y), T_DEPLOY, delta=0.002)
    after = alarms["index"][alarms["index"] >= T_CHANGE]
    assert len(after) > 0 and after.min() - T_CHANGE < 3 * DAY


@pytest.mark.parametrize("method,sensitive,setting", [("residual_conformal", 1.0, 2.0), ("adwin", 0.3, 0.002)])
def test_calibration_meets_the_target_false_alarm_rate_on_control_windows(method, sensitive, setting):
    if method == "adwin":
        pytest.importorskip("river")
    controls = [(toy(seed=k)[0], T_DEPLOY, N) for k in range(3)]
    value, rate = calibrate(method, controls, target_days=14)
    assert rate == false_alarm_rate(method, controls, value)
    assert rate <= 1 / (14 * DAY) or value == DETECTORS[method][2][-1]
    assert false_alarm_rate(method, controls, sensitive) >= rate  # the most sensitive setting alarms at least as often
    # alarms outside the window do not count: the window before the shift of a changed series is clean
    y, _ = toy(shift=40)
    before = false_alarm_rate(method, [(y, T_DEPLOY, T_CHANGE)], setting)
    whole = false_alarm_rate(method, [(y, T_DEPLOY, N)], setting)
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
    alerts = detect_all([utc, naive], {"entsoe": {"residual_conformal": 1.6}, "synthetic": {"residual_conformal": 1.6}})
    assert set(alerts.columns) == {"id", "source", "series", "scenario", "method", "timestamp", "index", "score", "threshold"}
    assert alerts["timestamp"].iloc[0] == pd.Timestamp("2017-01-02 00:00:00+00:00") + pd.Timedelta(hours=int(alerts["index"].iloc[0]))
    path = tmp_path / "alerts.csv"
    alerts.to_csv(path, index=False)
    back = read_alerts(path, manifest)
    for s in (utc, naive):
        sid = f"{s['source']}/{s['series']}/{s['scenario']}"
        expect = alerts[alerts["id"] == sid]["index"].to_numpy()
        assert np.array_equal(back[sid]["residual_conformal"], expect)
    # rows with an index only, outside the series, or with an unknown id
    extra = pd.DataFrame([dict(id="entsoe/IT/lockdown_2020", method="m", timestamp=None, index=7),
                          dict(id="entsoe/IT/lockdown_2020", method="m", timestamp="2030-01-01 00:00:00+00:00", index=None),
                          dict(id="nope/x/y", method="m", timestamp="2017-01-02 05:00:00+00:00", index=None),
                          dict(id="synthetic/toy/s0/level_shift", method="m", timestamp="2013-09-02 05:00", index=None)])
    extra.to_csv(path, index=False)
    back = read_alerts(path, manifest)
    assert np.array_equal(back["entsoe/IT/lockdown_2020"]["m"], [7]) and "nope/x/y" not in back
    assert np.array_equal(back["synthetic/toy/s0/level_shift"]["m"], [5])


@pytest.mark.parametrize("mult", [1.0, 1.6, 3.0])
def test_residual_conformal_alarms_are_those_of_the_stream_a_detector_dated_at_the_block_end(mult):
    from src.recommendations.alarms import block_end, residual_conformal_alarms, residual_detector

    y, _ = toy(shift=12, seed=3)
    d = residual_detector(mult)
    hers = d.fit_predict(pd.DataFrame({"value": y}))
    first = max(2 * d.period, d.cal_window)
    known = np.array([block_end(a["index"], first, d.causal_step) for a in hers])  # the last hour of the STL block
    detected = np.array([a["index"] for a in hers])
    assert len(hers) > 0 and (known >= detected).all() and (known - detected < d.causal_step).all()
    inside = known < len(y)  # an alarm of the last, incomplete block is not known by the end of the series
    ours = residual_conformal_alarms(y, 0, mult)
    assert ours["index"].tolist() == known[inside].tolist()
    assert ours["score"].tolist() == [a["score"] for a, ok in zip(hers, inside) if ok]          # the same floats
    assert ours["threshold"].tolist() == [a["threshold"] for a, ok in zip(hers, inside) if ok]
    late = residual_conformal_alarms(y, T_DEPLOY, mult)
    assert late["index"].tolist() == known[(known >= T_DEPLOY) & inside].tolist()
    assert len(residual_conformal_alarms(y[: d.period + d.cal_window - 1], 0, mult)) == 0  # too short for the detector


def test_residual_conformal_alarms_do_not_depend_on_later_values():
    from src.recommendations.alarms import residual_conformal_alarms, residual_detector

    y, _ = toy(shift=12, seed=3)
    full = residual_conformal_alarms(y, 0, 1.0)["index"].to_numpy()
    assert len(full) > 3
    for k in (int(full[1]), int(full[len(full) // 2]), int(full[-2]) + 7, T_CHANGE + 100):
        cut = residual_conformal_alarms(y[: k + 1], 0, 1.0)["index"].to_numpy()  # only the values up to hour k
        assert cut.tolist() == full[full <= k].tolist()
    # why the alarm is dated at the end of its block: inside a block that is not complete yet, the residuals of
    # the detector still change when the later values of the block arrive
    d = residual_detector(1.0)
    first, k = max(2 * d.period, d.cal_window), T_CHANGE + 100
    complete = first + ((k + 1 - first) // d.causal_step) * d.causal_step
    assert complete < k + 1
    r_full, r_cut = d._compute_causal_residuals(y), d._compute_causal_residuals(y[: k + 1])
    assert np.array_equal(r_cut[:complete], r_full[:complete])
    assert not np.allclose(r_cut[complete:], r_full[complete: k + 1])


class FixedAdvisor:
    """Proposes the same action at every review."""

    name = "fixed"

    def __init__(self, action):
        self.action = action

    def recommend(self, ctx, repeat=0):
        return Recommendation(action=self.action, reason="fixed")


def reviews_of(name, y, F, alarms, action="retrain_2w"):
    run = StrategyRun(y, F, T_DEPLOY, BY_NAME[name], BASE_FEATURES, advisors={"rules": FixedAdvisor(action)}, detector="x")
    for T in run.origins:
        run.step(T, np.asarray(alarms))
    return run


def test_follow_up_reviews_repeat_an_unconfirmed_proposal_until_the_check_confirms_it():
    y, F = toy(shift=30)
    alarm = T_CHANGE + 5  # the only alarm, five hours after the change
    plain = reviews_of("alarm_rules_gate", y, F, [alarm])
    # one day after the change the check cannot confirm the retraining, and without another alarm nothing follows
    assert [r["origin"] for r in plain.records] == [T_CHANGE + DAY] and plain.applied == []
    follow = reviews_of("alarm_rules_gate_follow", y, F, [alarm])
    origins = [r["origin"] for r in follow.records]
    assert origins == [T_CHANGE + DAY, T_CHANGE + 8 * DAY, T_CHANGE + 15 * DAY]
    assert [r["trigger"] for r in follow.records] == ["alarm", "follow_up", "follow_up"]
    assert [r["alarm"]["hours_ago"] for r in follow.records] == [T - alarm for T in origins]
    assert [r["changed"] for r in follow.records] == [False, False, True]
    assert follow.applied == [T_CHANGE + 15 * DAY] and follow.decisions == 3 and follow.accepted == 1
    changed = follow.applied[0]  # the same forecasts until the model is changed, better ones after it
    assert np.array_equal(follow.pred[:changed], plain.pred[:changed], equal_nan=True)
    rows = np.arange(changed, follow.end)
    assert np.abs(y[rows] - follow.pred[rows]).mean() < np.abs(y[rows] - plain.pred[rows]).mean()


def test_follow_up_reviews_stop_after_three_weeks_and_a_later_alarm_starts_over():
    y, F = toy()  # no change: no retraining is confirmed
    first = T_CHANGE + 5
    alarms = [first, T_CHANGE + 3 * DAY, T_CHANGE + 10 * DAY, T_CHANGE + 30 * DAY + 5]  # two fall into the pause after a review
    follow = reviews_of("alarm_rules_gate_follow", y, F, alarms)
    days = [(r["origin"] - T_CHANGE) // DAY for r in follow.records]
    assert days == [1, 8, 15, 22, 31]
    assert [r["trigger"] for r in follow.records] == ["alarm", "follow_up", "follow_up", "follow_up", "alarm"]
    assert follow.records[3]["alarm"]["hours_ago"] == T_CHANGE + 22 * DAY - first  # a follow-up belongs to its alarm
    assert follow.applied == [] and follow.accepted == 0
    nothing = backtest(y, F, T_DEPLOY, BY_NAME["nothing"], BASE_FEATURES)
    assert np.array_equal(follow.pred, nothing["pred"], equal_nan=True)
    # the same alarms without the follow-up: the strategy of the experiments reviews on alarms only
    plain = reviews_of("alarm_rules_gate", y, F, alarms)
    assert [(r["origin"] - T_CHANGE) // DAY for r in plain.records] == [1, 11, 31]
    assert {r["trigger"] for r in plain.records} == {"alarm"}


def test_follow_up_needs_a_proposal_that_the_check_did_not_confirm():
    y, F = toy(shift=30)
    keep = reviews_of("alarm_rules_gate_follow", y, F, [T_CHANGE + 5], action="keep")
    assert len(keep.records) == 1 and keep.records[0]["checks"] == []  # the advisor keeps the model: nothing to follow up
    assert {st.name for st in FOLLOW_UP_STRATEGIES}.isdisjoint(st.name for st in ALARM_STRATEGIES)  # not run by the experiments
    assert all(st.follow_up_days is None for st in STRATEGIES + ALARM_STRATEGIES)
