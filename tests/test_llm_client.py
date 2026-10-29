"""Tests for the LLM advisor of stream B: diagnostics without look-ahead, prompt, answer schema, answer cache,
and advised strategies in the backtest. No API key is needed: the LLM client is replaced by a stub.
Run from the repository root: python -m pytest tests"""
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from src.recommendations.backtester import ADVISED_STRATEGIES, STRATEGIES, backtest
from src.recommendations.catalog import CATALOG
from src.recommendations.forecast import BASE_FEATURES, WEEK
from src.recommendations.llm_client import LLMAdvisor, Recommendation, RuleAdvisor, diagnostics, render_prompt
from tests.test_backtester import T_DEPLOY, toy


class StubClient:
    """Stands in for the Instructor client: always answers with the same action."""

    def __init__(self, action):
        self.action, self.calls = action, 0

    def create_with_completion(self, response_model, messages, **kwargs):
        self.calls += 1
        usage = SimpleNamespace(input_tokens=500, output_tokens=40)
        return response_model(action=self.action, reason="stub"), SimpleNamespace(usage=usage)


class Fixed:
    def __init__(self, action):
        self.action = action

    def recommend(self, ctx, repeat=0, with_diagnostics=True):
        return Recommendation(action=self.action, reason="fixed")


def test_diagnostics_use_data_before_the_decision_only():
    y, _ = toy(shift=15)
    T = T_DEPLOY + 4 * WEEK
    pred = y + 3.0
    ctx = diagnostics(y, pred, T, T_DEPLOY, 5.0, T_DEPLOY)
    y2, pred2 = y.copy(), pred.copy()
    y2[T:] += 1000.0
    pred2[T:] = np.nan
    assert diagnostics(y2, pred2, T, T_DEPLOY, 5.0, T_DEPLOY) == ctx
    assert len(ctx["mase_last_weeks"]) == 3 and ctx["bias_last_week"] == -0.6


def test_prompt_lists_the_catalog_and_the_bare_prompt_has_no_numbers():
    y, _ = toy()
    ctx = diagnostics(y, y + 1.0, T_DEPLOY + 3 * WEEK, T_DEPLOY, 5.0, T_DEPLOY)
    system, user = render_prompt(ctx)
    assert all(f"- {a.name}:" in system for a in CATALOG)
    assert str(ctx["mase_first_weeks"]) in user
    _, bare = render_prompt(ctx, with_diagnostics=False)
    assert not any(ch.isdigit() for ch in bare)


def test_answer_schema_accepts_catalog_actions_only():
    assert Recommendation(action="retrain_2w", reason="ok").action == "retrain_2w"
    with pytest.raises(ValidationError):
        Recommendation(action="retrain_everything", reason="not in the catalog")


def test_answers_are_cached_and_a_rerun_needs_no_client(tmp_path):
    y, _ = toy()
    ctx = diagnostics(y, y + 1.0, T_DEPLOY + 3 * WEEK, T_DEPLOY, 5.0, T_DEPLOY)
    stub = StubClient("recalibrate_level")
    first = LLMAdvisor("anthropic/test-model", cache_dir=tmp_path, client=stub)
    assert first.recommend(ctx).action == "recalibrate_level"
    assert first.recommend(ctx).action == "recalibrate_level" and stub.calls == 1  # second answer from the cache
    first.recommend(ctx, repeat=1)
    assert stub.calls == 2  # a repeat is a new request
    rerun = LLMAdvisor("anthropic/test-model", cache_dir=tmp_path)  # no client: it must not be needed
    assert rerun.recommend(ctx).action == "recalibrate_level" and rerun.calls == 0


def test_advised_strategies_in_the_backtest():
    y, F = toy(shift=15)
    by = {s.name: s for s in STRATEGIES + ADVISED_STRATEGIES}
    cache = {}
    # an advisor that always proposes the weekly full retraining, with the check, is the scheduled_gate strategy
    same = backtest(y, F, T_DEPLOY, by["llm_gate"], BASE_FEATURES, cache=cache, advisors={"llm": Fixed("retrain_full")})
    ref = backtest(y, F, T_DEPLOY, by["scheduled_gate"], BASE_FEATURES, cache=cache)
    assert np.allclose(same["pred"], ref["pred"], equal_nan=True) and same["retrains"] == ref["retrains"]
    keep = backtest(y, F, T_DEPLOY, by["llm"], BASE_FEATURES, cache=cache, advisors={"llm": Fixed("keep")})
    nothing = backtest(y, F, T_DEPLOY, by["nothing"], BASE_FEATURES, cache=cache)
    assert np.allclose(keep["pred"], nothing["pred"], equal_nan=True) and keep["proposed"] == 0 < keep["decisions"]
    rules = {k: backtest(y, F, T_DEPLOY, by[k], BASE_FEATURES, cache=cache, advisors={"rules": RuleAdvisor()})
             for k in ("rules", "rules_gate")}
    for r in rules.values():
        assert r["accepted"] <= r["proposed"] <= r["decisions"] == ref["decisions"]
    assert rules["rules"]["accepted"] == rules["rules"]["proposed"] > 0  # without the check every proposal is applied
