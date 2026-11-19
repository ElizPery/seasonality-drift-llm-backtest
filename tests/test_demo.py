"""Tests of the demo on one series (`data/demo`, `src/recommendations/demo.py`). Run from the repository root:
python -m pytest tests"""
import io
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from src.api.uploads import calibrate_on_history
from src.recommendations import demo
from src.recommendations.forecast import WEEK
from src.recommendations.llm_client import LLMAdvisor, RuleAdvisor
from src.recommendations.monitor import Monitor


class NoClient:
    """Stands in for the LLM client: a request means that a stored answer is missing."""

    def create_with_completion(self, **kwargs):
        raise RuntimeError("the demo must not send requests")


@pytest.fixture(scope="module")
def series():
    return demo.load_series()


@pytest.fixture(scope="module")
def multiplier(series):
    return next(c["value"] for c in calibrate_on_history(series) if c["method"] == "residual_conformal")


def test_demo_series_keeps_its_level_and_gets_a_larger_weekly_swing(series):
    y, t, deploy, change = series["y"], demo.stamps(series), series["t_deploy"], series["t_change"]
    assert len(y) == 24 * WEEK and deploy == 10 * WEEK and change == 13 * WEEK
    assert str(t[deploy].date()) == demo.DEPLOY_AT and str(t[change].date()) == demo.CHANGE_AT
    before, after = y[change - 3 * WEEK:change], y[change:change + 3 * WEEK]
    assert abs(after.mean() - before.mean()) < 0.1 * before.std()
    swing = lambda v: np.ptp(v.reshape(-1, WEEK).mean(axis=0))  # range of the mean weekly profile
    assert swing(after) / swing(before) == pytest.approx(demo.FACTOR, rel=0.1)


def test_generator_gives_the_stored_series():
    if not Path("data/processed/series/manifest.csv").exists():
        pytest.skip("needs the exported series (python -m src.data.export)")
    buffer = io.StringIO()
    demo.make_series().to_csv(buffer, index=False)
    assert buffer.getvalue() == demo.DEMO_CSV.read_text()


def test_loop_with_the_stored_llm_answers_needs_no_request_and_lowers_the_error(series, multiplier, tmp_path, monkeypatch):
    monkeypatch.setattr("src.recommendations.llm_client.time.sleep", lambda seconds: None)
    answers = tmp_path / "answers"  # a copy: the advisor writes failed requests into its folder
    shutil.copytree(demo.LLM_ANSWERS, answers)
    advisor = LLMAdvisor(demo.LLM_MODEL, cache_dir=answers, client=NoClient())
    monitor = Monitor(series, "alarm_llm_gate", detector="residual_conformal", detector_value=multiplier,
                      advisors={"llm": advisor})
    events = monitor.advance(len(series["y"]))
    assert advisor.calls == 0 and advisor.errors == 0
    change = series["t_change"]
    assert not [a for a in monitor.alarms if a < change] and monitor.alarms[0] - change < 24
    reviews = demo.review_table(events, series)
    assert list(reviews["check"]) == ["not passed", "passed"]
    assert list(reviews["applied"]) == ["keep", "retrain_8w"]
    with_system, without = monitor.mase(change)
    assert without > 3 and with_system < 0.6 * without

    rules = Monitor(series, "alarm_rules_gate", detector="residual_conformal", detector_value=multiplier,
                    advisors={"rules": RuleAdvisor()})
    rules.advance(len(series["y"]))
    assert rules.mase(change)[0] < 0.6 * without

    figure = demo.plot_errors(series, {"model left as deployed": monitor.reference.pred, "system": monitor.run.pred},
                              monitor.scale, alarms=monitor.alarms, applied=monitor.run.applied)
    assert len(figure.axes[0].lines) >= 2 + len(monitor.alarms) + 1
    daily = demo.daily_error(series, monitor.run.pred, monitor.scale)
    assert len(daily) == (monitor.end - series["t_deploy"]) // 24 and np.isfinite(daily).all()


def test_figures_of_the_series(series):
    assert len(demo.plot_series(series).axes) == 2
    figure = demo.plot_alarms(series, pd.DataFrame({"index": [series["t_change"] + 23]}))
    assert len(figure.axes[0].lines) >= 2
    assert len(demo.plot_forecast(series, series["y"]).axes[0].lines) >= 2


def test_session_start_without_a_service_gives_a_hint(capsys):
    assert demo.start_session(base_url="http://127.0.0.1:9") is None
    assert "docker compose up" in capsys.readouterr().out
