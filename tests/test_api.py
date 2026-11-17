"""Tests of the monitoring service API (FastAPI TestClient, no database): sessions, reviews, recommendations, detection
and metrics on a small exported series; a session played to its end gives the numbers of the backtest."""
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.sessions import Manager
from src.data.frames import write_series
from src.recommendations.alarms import alarms_in
from src.recommendations.backtester import backtest, mase
from src.recommendations.forecast import mase_scale
from src.recommendations.llm_client import RuleAdvisor
from src.recommendations.monitor import STRATEGY_BY_NAME
from tests.test_alarms import series_dict
from tests.test_backtester import T_CHANGE, T_DEPLOY, toy

SID = "synthetic/toy/s0/level_shift"


@pytest.fixture()
def client(tmp_path):
    y, F = toy(shift=20, seed=1)
    write_series([series_dict(y, F)], tmp_path)
    pd.DataFrame([dict(source="synthetic", method="profile_conformal", param="mult", value=1.2),
                  dict(source="synthetic", method="adwin", param="delta", value=0.003)]).to_csv(tmp_path / "detectors.csv", index=False)
    app.state.manager = Manager(tmp_path, tmp_path / "detectors.csv", database_url="", grafana_url="http://grafana.test")
    with TestClient(app) as c:
        yield c, y, F
    app.state.manager = None


def test_session_played_by_hand_gives_the_numbers_of_the_backtest(client):
    c, y, F = client
    assert c.get("/health").json()["series"] == 1
    r = c.post("/sessions", json=dict(series_id=SID, strategy="alarm_rules_gate", detector="profile_conformal", paused=True,
                                      skip_to_days_before_change=None))
    assert r.status_code == 201, r.text
    info = r.json()
    assert info["state"] == "paused" and info["hours_read"] == 0 and info["detector_value"] == 1.2
    assert "var-session=1" in info["grafana_url"]
    info = c.post("/sessions/1/advance", json=dict(hours=len(y))).json()
    assert info["state"] == "done" and info["now"] is None
    alarms = alarms_in("profile_conformal", y, T_DEPLOY, 1.2)["index"].to_numpy()
    ref = backtest(y, F, T_DEPLOY, STRATEGY_BY_NAME["alarm_rules_gate"], series_dict(y, F)["base_features"],
                   advisors={"rules": RuleAdvisor()}, alarms=alarms, detector="profile_conformal")
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
    assert 'drift_alarms_total{detector="profile_conformal",series="synthetic/toy/s0/level_shift",session="1"' in text
    assert "drift_review_outcomes_total" in text and "drift_api_requests_total" in text
    assert c.delete("/sessions/1").status_code == 204
    assert c.get("/sessions/1").status_code == 404
    assert 'drift_value{detector="profile_conformal",series="synthetic/toy/s0/level_shift",session="1"' not in c.get("/metrics").text


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
    got = c.post("/detect", json=dict(series_id=SID, method="profile_conformal")).json()
    assert [a["index"] for a in got["alarms"]] == alarms_in("profile_conformal", y, T_DEPLOY, 1.2)["index"].tolist()
    posted = c.post("/detect", json=dict(values=y.tolist(), start="2020-01-06", method="adwin", value=0.003, from_index=T_DEPLOY))
    assert posted.status_code == 200
    assert [a["index"] for a in posted.json()["alarms"]] == alarms_in("adwin", y, T_DEPLOY, 0.003)["index"].tolist()
    assert c.post("/detect", json=dict(values=y.tolist(), start="2020-01-06")).status_code == 422  # no setting given


def test_bad_requests(client):
    c, _, _ = client
    assert c.post("/sessions", json=dict(series_id="nope/x/y")).status_code == 404
    assert c.post("/sessions", json=dict(series_id=SID, strategy="nope")).status_code == 422
    assert c.post("/sessions", json=dict(series_id=SID, strategy="alarm_retrain", detector=None)).status_code == 422
    assert c.post("/sessions", json=dict(series_id=SID, strategy="llm_gate")).status_code == 422  # no llm_model
    assert c.get("/sessions/42").status_code == 404
