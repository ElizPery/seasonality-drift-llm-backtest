"""Tests of the engine of the monitoring service: the online detector and `Monitor` must give exactly what the batch
code of the experiments gives (alarms of `alarms_in`, forecasts and decisions of `backtest`), and the exported series
must come back as the series of the experiments."""
import numpy as np
import pandas as pd
import pytest

from src.data.frames import series_frame, series_table
from src.recommendations.alarms import OnlineDetector, alarms_in
from src.recommendations.backtester import backtest
from src.recommendations.data_sources import series_from_frame
from src.recommendations.forecast import DAY
from src.recommendations.llm_client import RuleAdvisor
from src.recommendations.monitor import STRATEGY_BY_NAME, Monitor
from tests.test_alarms import series_dict
from tests.test_backtester import T_CHANGE, T_DEPLOY, toy


@pytest.mark.parametrize("method,value", [("profile_conformal", 1.2), ("profile_conformal", 1.0), ("adwin", 0.003)])
def test_online_detector_raises_the_alarms_of_the_batch_detector(method, value):
    y, _ = toy(shift=12, seed=3)
    batch = alarms_in(method, y, T_DEPLOY, value)
    det = OnlineDetector(method, value, T_DEPLOY)
    online = [a for a in (det.update(v) for v in y) if a is not None]
    assert [a["index"] for a in online] == batch["index"].tolist()
    assert len(online) > 0
    if method == "profile_conformal":  # same scores and thresholds, not only the same positions
        assert np.array_equal([a["score"] for a in online], batch["score"].to_numpy())
        assert np.array_equal([a["threshold"] for a in online], batch["threshold"].to_numpy())


@pytest.mark.parametrize("name", ["scheduled_gate", "alarm_rules_gate", "alarm_catalog_gate"])
def test_monitor_played_to_the_end_equals_the_backtest(name):
    y, F = toy(shift=20, seed=1)
    s = series_dict(y, F)
    st = STRATEGY_BY_NAME[name]
    advisors = {"rules": RuleAdvisor()}
    detector = "profile_conformal" if st.trigger == "alarm" else None
    m = Monitor(s, st, detector=detector, detector_value=1.2, advisors=advisors)
    events = []
    while not m.done:
        events += m.advance(29)  # chunks that do not line up with the days
    alarms = alarms_in("profile_conformal", y, T_DEPLOY, 1.2)["index"].to_numpy() if detector else None
    ref = backtest(y, F, T_DEPLOY, st, s["base_features"], advisors=advisors, alarms=alarms, detector=detector)
    got = m.run.result(None if alarms is None else np.asarray(m.alarms))
    assert np.array_equal(got["pred"], ref["pred"], equal_nan=True)
    for k in ("retrains", "picks", "decisions", "proposed", "accepted", "check_fits", "applied", "alarms", "end"):
        assert got[k] == ref[k], k
    decisions = [e for e in events if e["type"] == "decision"]
    assert [d["index"] for d in decisions if d["changed"]] == ref["applied"]
    assert len(decisions) >= ref["decisions"] > 0
    nothing = backtest(y, F, T_DEPLOY, STRATEGY_BY_NAME["nothing"], s["base_features"])
    assert np.array_equal(m.reference.pred, nothing["pred"], equal_nan=True)


def test_monitor_uses_only_values_read_so_far():
    y, F = toy(shift=20, seed=1)
    s = series_dict(y, F)
    m = Monitor(s, "alarm_gate", detector="profile_conformal", detector_value=1.2)
    events = m.advance(T_CHANGE - T_DEPLOY + 5 * DAY)
    y2 = y.copy()
    y2[m.now:] += 1000.0  # the future changes completely
    s2 = series_dict(y2, F)
    m2 = Monitor(s2, "alarm_gate", detector="profile_conformal", detector_value=1.2)
    events2 = m2.advance(T_CHANGE - T_DEPLOY + 5 * DAY)
    def strip(evs):  # the time a review took differs between runs; the forecasts of the next day are checked below
        return [{k: v for k, v in e.items() if k != "seconds" and not (e["type"] == "forecast" and k == "forecast")} for e in evs]

    assert strip(events) == strip(events2)
    assert any(e["type"] == "decision" for e in events)
    # and the error of the hours read is the scaled absolute error of the forecast made before them
    obs = [e for e in events if e["type"] == "observation"]
    assert all(e["error"] == pytest.approx(abs(e["value"] - e["forecast"]) / m.scale) for e in obs)


def test_monitor_needs_a_detector_and_an_advisor_where_the_strategy_uses_them():
    y, F = toy()
    with pytest.raises(ValueError):
        Monitor(series_dict(y, F), "alarm_retrain")
    with pytest.raises(ValueError):
        Monitor(series_dict(y, F), "rules_gate")


def test_exported_frame_gives_back_the_series():
    y, F = toy(shift=5)
    s = series_dict(y, F)
    row = series_table([s]).iloc[0]
    back = series_from_frame(row, series_frame(s))
    assert np.array_equal(back["y"], y)
    for f in F:
        assert np.array_equal(back["F"][f], F[f], equal_nan=True), f
    for k in ("t_deploy", "t_change", "base_features", "source", "scenario"):
        assert back[k] == s[k], k


def test_exported_entsoe_frame_uses_the_local_clock():
    n = 6 * 7 * DAY
    stamps = pd.date_range("2020-03-01", periods=n, freq="1h", tz="UTC")
    frame = pd.DataFrame({"timestamp": stamps, "value": np.arange(n, dtype=float)})
    row = dict(source="entsoe", series="DE", client="DE", seed=-1, scenario="lockdown_2020", magnitude=np.nan,
               t_deploy=4 * 7 * DAY, t_change=5 * 7 * DAY, t_stop=n, start=str(stamps[0]))
    s = series_from_frame(row, frame)
    local = stamps.tz_convert("Europe/Berlin")
    assert np.array_equal(s["F"]["hour"], local.hour.to_numpy().astype(float))  # summer time from 29 March: UTC+2
    assert "doy_sin" in s["base_features"]
