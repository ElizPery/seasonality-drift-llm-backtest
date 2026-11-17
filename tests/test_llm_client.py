"""Tests for the LLM advisor of stream B: diagnostics without look-ahead, prompt, answer schema, answer cache,
and advised strategies in the backtest. No API key is needed: the LLM client is replaced by a stub.
Run from the repository root: python -m pytest tests"""
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from src.recommendations.backtester import ADVISED_STRATEGIES, STRATEGIES, backtest
from src.recommendations.catalog import CATALOG
from src.recommendations.forecast import BASE_FEATURES, DAY, WEEK
from src.recommendations.llm_client import (EXAMPLES, PROMPTS, LLMAdvisor, Recommendation, RuleAdvisor, diagnostics,
                                            render_prompt)
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

    def recommend(self, ctx, repeat=0):
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
    _, bare = render_prompt(ctx, "bare")
    assert not any(ch.isdigit() for ch in bare)


def test_prompt_variants_extend_the_base_system_prompt():
    y, _ = toy()
    ctx = diagnostics(y, y + 1.0, T_DEPLOY + 3 * WEEK, T_DEPLOY, 5.0, T_DEPLOY)
    msgs = {v: render_prompt(ctx, v) for v in PROMPTS}
    base_system, base_user = msgs["base"]
    assert msgs["instructions"][1] == msgs["examples"][1] == base_user
    assert msgs["instructions"][0].startswith(base_system) and msgs["examples"][0].startswith(msgs["instructions"][0])
    assert "Example" not in msgs["instructions"][0] and f"Example {len(EXAMPLES)}." in msgs["examples"][0]
    assert {a for _, a, _ in EXAMPLES} <= {a.name for a in CATALOG}
    with pytest.raises(ValueError):
        render_prompt(ctx, "few-shot")


def test_rich_prompt_adds_daily_errors_and_the_previous_decision():
    y, _ = toy()
    T = T_DEPLOY + 3 * WEEK
    ctx = diagnostics(y, y + 1.0, T, T_DEPLOY, 5.0, T_DEPLOY, history=[(T - 2 * WEEK, "retrain_2w")], kind="a toy series")
    assert len(ctx["daily_errors"]) == len(ctx["daily_bias"]) == 14 and len(ctx["recent_hours"]) == 48
    assert ctx["last_decision"]["action"] == "retrain_2w" and ctx["last_decision"]["weeks_ago"] == 2.0
    assert ctx["last_decision"]["mase_week_before"] == ctx["last_decision"]["mase_week_after"] == 0.2
    _, user = render_prompt(ctx, "rich")
    base_user = render_prompt(ctx)[1]
    assert user.startswith(base_user.rsplit("\n\n", 1)[0]) and user.endswith("Which action do you choose?")
    assert "a toy series" in user and "retrain_2w 2.0 weeks ago" in user and "/" in user
    # a decision too recent to be judged, or none at all, is reported as "none yet"
    ctx2 = diagnostics(y, y + 1.0, T, T_DEPLOY, 5.0, T_DEPLOY, history=[(T - DAY, "keep")])
    assert ctx2["last_decision"] is None and "none yet" in render_prompt(ctx2, "rich")[1]
    assert render_prompt(ctx2)[1] == render_prompt(diagnostics(y, y + 1.0, T, T_DEPLOY, 5.0, T_DEPLOY))[1]


def test_the_base_prompt_still_matches_the_stored_pilot_answers():
    # the pilot answers were cached before the prompt variants existed: "base" must render the same messages
    import json
    from pathlib import Path

    from src.recommendations.llm_client import CACHE_DIR, cache_key

    records = [json.loads(line) for f in sorted(Path(CACHE_DIR).glob("answers_*.jsonl")) for line in f.read_text().splitlines()]
    pilot = [r for r in records if r["model"] == "ollama/qwen2.5:7b-instruct" and r.get("prompt", "base") == "base"]
    if not pilot:
        pytest.skip("no stored pilot answers")
    y, _ = toy()
    system, _ = render_prompt(diagnostics(y, y + 1.0, T_DEPLOY + 3 * WEEK, T_DEPLOY, 5.0, T_DEPLOY))  # does not depend on the series
    assert all(cache_key(r["model"], system, r["user"], r["repeat"]) == r["key"] for r in pilot)


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


def test_bare_advisor_gets_a_tagged_name_and_a_prompt_without_numbers(tmp_path):
    y, _ = toy()
    ctx = diagnostics(y, y + 1.0, T_DEPLOY + 3 * WEEK, T_DEPLOY, 5.0, T_DEPLOY)
    stub = StubClient("keep")
    bare = LLMAdvisor("ollama/test-model", cache_dir=tmp_path, client=stub, prompt="bare")
    assert bare.name == "ollama/test-model|bare"
    bare.recommend(ctx)
    record = next(iter(bare._cache.values()))
    assert record["with_diagnostics"] is False and record["prompt"] == "bare" and "MASE" not in record["user"]
    full = LLMAdvisor("ollama/test-model", cache_dir=tmp_path, client=stub)
    full.recommend(ctx)
    assert stub.calls == 2  # the bare and the full prompt are different cache keys
    examples = LLMAdvisor("ollama/test-model", cache_dir=tmp_path, client=stub, prompt="examples")
    assert examples.name == "ollama/test-model|examples"
    examples.recommend(ctx)
    assert stub.calls == 3  # same user message, another system prompt: a new request


def test_job_repeats_the_advised_strategies_and_tags_the_model():
    import pandas as pd

    from src.recommendations.backtester import _job
    from tests.test_alarms import series_dict

    y, F = toy(shift=15)
    strategies = (STRATEGIES[0],) + tuple(st for st in ADVISED_STRATEGIES if st.advisor == "rules")
    rows = pd.DataFrame(_job(series_dict(y, F), strategies=strategies, advisors=("rules",), repeats=2))
    assert (rows.strategy == "nothing").sum() == 1 and (rows.strategy == "naive").sum() == 1
    rules = rows[rows.strategy == "rules"]
    assert sorted(rules.repeat) == [0, 1] and (rules.model == "rules").all()
    assert rows[rows.strategy == "nothing"].model.isna().all()
