"""Tests of the monitoring service API (FastAPI TestClient, no database): sessions, reviews, recommendations, detection
and metrics on a small exported series; a session played to its end gives the numbers of the backtest. A series
uploaded through the API is calibrated on its own history and played like an exported one."""
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.sessions import Manager
from src.api.uploads import CONTROL_FROM
from src.data.frames import series_frame, write_series
from src.recommendations.alarms import alarms_in, calibrate
from src.recommendations.backtester import backtest, mase
from src.recommendations.forecast import WEEK, mase_scale
from src.recommendations.llm_client import RuleAdvisor
from src.recommendations.monitor import STRATEGY_BY_NAME
from tests.test_alarms import series_dict
from tests.test_backtester import T_CHANGE, T_DEPLOY, toy

SID = "synthetic/toy/s0/level_shift"
START = pd.Timestamp("2013-09-02 00:00")  # a Monday, so the calendar of the stamps is the nominal calendar of `toy`
UPLOADED = "user/shop/uploaded"


def at(position: int) -> str:
    return str(START + pd.Timedelta(hours=position))


@pytest.fixture()
def client(tmp_path):
    y, F = toy(shift=20, seed=1)
    write_series([series_dict(y, F)], tmp_path)
    pd.DataFrame([dict(source="synthetic", method="residual_conformal", param="mult", value=1.6),
                  dict(source="synthetic", method="adwin", param="delta", value=0.003)]).to_csv(tmp_path / "detectors.csv", index=False)
    app.state.manager = Manager(tmp_path, tmp_path / "detectors.csv", database_url="", grafana_url="http://grafana.test",
                                upload_dir=tmp_path / "uploads")
    with TestClient(app) as c:
        yield c, y, F
    app.state.manager = None


def test_session_played_by_hand_gives_the_numbers_of_the_backtest(client):
    c, y, F = client
    assert c.get("/health").json()["series"] == 1
    r = c.post("/sessions", json=dict(series_id=SID, strategy="alarm_rules_gate", detector="adwin", paused=True,
                                      skip_to_days_before_change=None))
    assert r.status_code == 201, r.text
    info = r.json()
    assert info["state"] == "paused" and info["hours_read"] == 0 and info["detector_value"] == 0.003
    assert "var-session=1" in info["grafana_url"]
    info = c.post("/sessions/1/advance", json=dict(hours=len(y))).json()
    assert info["state"] == "done" and info["now"] is None
    alarms = alarms_in("adwin", y, T_DEPLOY, 0.003)["index"].to_numpy()
    ref = backtest(y, F, T_DEPLOY, STRATEGY_BY_NAME["alarm_rules_gate"], series_dict(y, F)["base_features"],
                   advisors={"rules": RuleAdvisor()}, alarms=alarms, detector="adwin")
    scale = mase_scale(y[:T_DEPLOY])
    assert info["mase_after_change"]["strategy"] == pytest.approx(mase(y, ref["pred"], T_CHANGE, ref["end"], scale), abs=1e-12)
    assert info["model_changes"] == len(ref["applied"]) and info["alarms"] == len(alarms[alarms < ref["end"]])
    reviews = c.get("/sessions/1/reviews").json()
    assert len(reviews) >= ref["decisions"] > 0
    assert {rv["outcome"] for rv in reviews} <= {"applied", "rejected", "keep", "skipped"}
    assert all(rv["summary"] for rv in reviews)
    rec = c.post("/recommendations", json=dict(session_id=1)).json()
    assert rec["proposal"] in {"keep", "retrain_full", "retrain_8w", "retrain_2w", "reweight_1w", "add_profile", "recalibrate_level"}
    assert rec["verdict"]
    text = c.get("/metrics").text
    assert 'drift_alarms_total{detector="adwin",series="synthetic/toy/s0/level_shift",session="1"' in text
    assert "drift_review_outcomes_total" in text and "drift_api_requests_total" in text
    assert c.delete("/sessions/1").status_code == 204
    assert c.get("/sessions/1").status_code == 404
    assert 'drift_value{detector="adwin",series="synthetic/toy/s0/level_shift",session="1"' not in c.get("/metrics").text


def test_session_plays_in_the_background(client):
    c, y, _ = client
    info = c.post("/sessions", json=dict(series_id=SID, strategy="scheduled_gate", detector=None, speed=5000)).json()
    assert info["state"] == "playing"
    deadline = time.time() + 60
    while time.time() < deadline and c.get(f"/sessions/{info['id']}").json()["state"] == "playing":
        time.sleep(0.2)
    done = c.get(f"/sessions/{info['id']}").json()
    assert done["state"] == "done" and done["reviews"] > 0


def test_detect_on_a_series_and_on_posted_values(client):
    c, y, _ = client
    got = c.post("/detect", json=dict(series_id=SID, method="adwin")).json()
    assert [a["index"] for a in got["alarms"]] == alarms_in("adwin", y, T_DEPLOY, 0.003)["index"].tolist()
    posted = c.post("/detect", json=dict(values=y.tolist(), start="2020-01-06", method="adwin", value=0.003, from_index=T_DEPLOY))
    assert posted.status_code == 200
    assert [a["index"] for a in posted.json()["alarms"]] == alarms_in("adwin", y, T_DEPLOY, 0.003)["index"].tolist()
    assert c.post("/detect", json=dict(values=y.tolist(), start="2020-01-06")).status_code == 422  # no setting given
    stream_a = c.post("/detect", json=dict(values=y.tolist(), start="2020-01-06", method="residual_conformal", value=1.6,
                                           from_index=T_DEPLOY))
    assert stream_a.status_code == 200
    expected = alarms_in("residual_conformal", y, T_DEPLOY, 1.6)
    assert len(expected) > 0 and [a["index"] for a in stream_a.json()["alarms"]] == expected["index"].tolist()


def test_bad_requests(client):
    c, _, _ = client
    assert c.post("/sessions", json=dict(series_id="nope/x/y")).status_code == 404
    assert c.post("/sessions", json=dict(series_id=SID, strategy="nope")).status_code == 422
    assert c.post("/sessions", json=dict(series_id=SID, strategy="alarm_retrain", detector=None)).status_code == 422
    assert c.post("/sessions", json=dict(series_id=SID, strategy="llm_gate")).status_code == 422  # no llm_model
    assert c.get("/sessions/42").status_code == 404


def test_uploaded_series_is_calibrated_on_its_history_and_played_like_an_exported_one(client, tmp_path):
    pytest.importorskip("river")
    c, _, _ = client
    y, F = toy(shift=25, seed=3)
    r = c.post("/series", json=dict(name="shop", start=str(START), values=y.tolist(), deploy_at=at(T_DEPLOY),
                                    change_at=at(T_CHANGE)))
    assert r.status_code == 201, r.text
    info = r.json()
    assert info["id"] == UPLOADED and info["n"] == len(y) and info["history_hours"] == T_DEPLOY
    assert info["deploy_at"] == at(T_DEPLOY) and info["change_at"] == at(T_CHANGE)
    # the settings are those of `calibrate` on the hours before the deployment only
    control = [(y[:T_DEPLOY].copy(), CONTROL_FROM, T_DEPLOY)]
    settings = {d["method"]: d for d in info["detectors"]}
    for method in ("residual_conformal", "adwin"):
        value, rate = calibrate(method, control)
        assert settings[method]["value"] == value
        assert settings[method]["false_alarms_per_1000_hours"] == pytest.approx(rate * 1000, abs=1e-3)
        assert settings[method]["control_hours"] == T_DEPLOY - CONTROL_FROM
    later = y.copy()
    later[T_DEPLOY:] += 1000.0  # values after the deployment do not move the settings
    again = c.post("/series", json=dict(name="later", start=str(START), values=later.tolist(), deploy_at=at(T_DEPLOY))).json()
    assert [d["value"] for d in again["detectors"]] == [d["value"] for d in info["detectors"]]
    assert c.get("/health").json()["series"] == 3
    assert UPLOADED in [row["id"] for row in c.get("/series", params=dict(source="user")).json()]
    # a session on the uploaded series takes the calibrated setting and gives the numbers of the backtest
    delta = settings["adwin"]["value"]
    s = c.post("/sessions", json=dict(series_id=UPLOADED, strategy="alarm_rules_gate", detector="adwin", paused=True,
                                      skip_to_days_before_change=None)).json()
    assert s["detector_value"] == delta and s["change_at"].startswith(at(T_CHANGE).replace(" ", "T"))
    done = c.post(f"/sessions/{s['id']}/advance", json=dict(hours=len(y))).json()
    alarms = alarms_in("adwin", y, T_DEPLOY, delta)["index"].to_numpy()
    ref = backtest(y, F, T_DEPLOY, STRATEGY_BY_NAME["alarm_rules_gate"], series_dict(y, F)["base_features"],
                   advisors={"rules": RuleAdvisor()}, alarms=alarms, detector="adwin")
    scale = mase_scale(y[:T_DEPLOY])
    assert done["state"] == "done" and done["model_changes"] == len(ref["applied"])
    assert done["mase_after_change"]["strategy"] == pytest.approx(mase(y, ref["pred"], T_CHANGE, ref["end"], scale), abs=1e-12)
    got = c.post("/detect", json=dict(series_id=UPLOADED, method="adwin")).json()
    assert got["value"] == delta and [a["index"] for a in got["alarms"]] == alarms.tolist()
    # the series survives a restart of the service, and only uploaded series can be deleted
    restarted = Manager(tmp_path, tmp_path / "detectors.csv", database_url="", grafana_url="", upload_dir=tmp_path / "uploads")
    assert restarted.is_uploaded(UPLOADED) and restarted.setting("user", "adwin", None, UPLOADED) == delta
    assert np.array_equal(restarted.read(UPLOADED)["y"], y)
    assert c.delete(f"/series/{SID}").status_code == 403
    assert c.delete(f"/series/{UPLOADED}").status_code == 204
    assert c.delete(f"/series/{UPLOADED}").status_code == 404
    assert c.post("/sessions", json=dict(series_id=UPLOADED)).status_code == 404


def test_series_uploaded_as_csv(client):
    c, _, _ = client
    y, _ = toy(shift=25, seed=3)
    frame = series_frame(dict(start=str(START), y=y)).to_csv(index=False)
    query = dict(name="shop", deploy_at=at(T_DEPLOY))
    r = c.post("/series/csv", params=query, content=frame, headers={"content-type": "text/csv"})
    assert r.status_code == 201, r.text
    assert np.array_equal(app.state.manager.read(UPLOADED)["y"], y)  # the values come back exactly
    assert r.json()["change_at"] is None
    assert c.post("/series/csv", params=query, content=frame).status_code == 409
    assert c.post("/series/csv", params=dict(query, replace=True), content=frame).status_code == 201
    assert c.get("/health").json()["series"] == 2
    gap = "\n".join(line for i, line in enumerate(frame.splitlines()) if i != 500)
    r = c.post("/series/csv", params=dict(query, name="gap"), content=gap)
    assert r.status_code == 422 and "one hour apart" in r.json()["detail"]
    assert c.post("/series/csv", params=dict(query, name="x"), content="a,b\n1,2\n3,4\n").status_code == 422


def test_bad_uploads(client):
    c, y, _ = client
    ok = dict(name="shop", start=str(START), values=y.tolist(), deploy_at=at(T_DEPLOY))
    for change, reason in [(dict(name="no/slash"), "name"),
                           (dict(deploy_at=at(T_DEPLOY - WEEK)), "weeks of history"),
                           (dict(deploy_at=at(len(y) - WEEK)), "from deploy_at on"),
                           (dict(deploy_at=str(START + pd.Timedelta(hours=T_DEPLOY, minutes=30))), "hourly grid"),
                           (dict(deploy_at=at(T_DEPLOY) + "+00:00"), "time zone"),
                           (dict(change_at=at(T_DEPLOY - 1)), "change_at"),
                           (dict(start="yesterday-ish"), "not a timestamp"),
                           (dict(values=[5.0] * len(y)), "error scale")]:
        r = c.post("/series", json=dict(ok, **change))
        assert r.status_code == 422 and reason in r.json()["detail"], (change, r.text)
    assert c.get("/health").json()["series"] == 1
