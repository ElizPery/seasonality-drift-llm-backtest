"""LLM recommendations for stream B (RQ3): the diagnostic context, the prompt, the answer schema and the
advisors that turn a decision point into one action of the catalog.

At a decision origin T the advisor sees a short description of the series computed from data before T
only: how the error of the deployed model changed week by week, its bias, how the level, the daily
profile and the noise moved, and how long ago the model was last changed. It answers with one action
name from the catalog and a short reason. The answer schema is a Pydantic model; Instructor validates
the reply and asks again if it does not fit (JSON mode: newer Claude models reject a forced tool call,
which the default tools mode of Instructor sends).

Every LLM answer is stored in a JSON lines cache keyed by the model, the prompt and the repeat number, so
a rerun of the experiments reads the stored answers and needs no API key. `RuleAdvisor` applies fixed
rules to the same numbers: a non-LLM baseline that also lets the whole pipeline run without a key.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from src.recommendations.catalog import CATALOG
from src.recommendations.forecast import DAY, WEEK

ACTION_NAMES = tuple(a.name for a in CATALOG)
DEFAULT_MODEL = "anthropic/claude-sonnet-5-5"
CACHE_DIR = Path("data/llm_answers")  # tracked by git: a rerun from the stored answers needs no API key
# the API key is read from this file (one line ANTHROPIC_API_KEY=...), not from the shell environment
KEY_FILE = Path(os.environ.get("LLM_KEY_FILE", "~/.config/diploma/anthropic.env")).expanduser()


class Recommendation(BaseModel):
    action: Literal[ACTION_NAMES] = Field(description="name of exactly one action from the catalog")
    reason: str = Field(max_length=400, description="one or two sentences: which numbers led to this choice")


def diagnostics(y: np.ndarray, pred: np.ndarray, T: int, t_deploy: int, scale: float, last_change: int,
                alarm: dict | None = None) -> dict:
    """Numbers the advisor sees at the decision origin T. Only data before T is used; errors are those of
    the forecasts the deployed strategy actually made (`pred`). Values are in MASE scale units."""
    last = np.arange(T - WEEK, T)
    before = np.arange(T - 5 * WEEK, T - WEEK)  # the 4 weeks before the last one
    first = np.arange(t_deploy, min(t_deploy + 2 * WEEK, T - WEEK))

    def mase(rows):
        return float(np.mean(np.abs(y[rows] - pred[rows])) / scale)

    def amplitude(rows):  # spread of the mean daily profile
        return float(np.std(y[rows].reshape(-1, DAY).mean(axis=0)))

    def noise(rows):  # change against the same hour of the previous day
        return float(np.std(y[rows] - y[rows - DAY]))

    weeks = [mase(np.arange(o, o + WEEK)) for o in range(max(t_deploy, T - 3 * WEEK), T, WEEK)]
    return dict(
        mase_first_weeks=round(mase(first), 2),
        mase_last_weeks=[round(w, 2) for w in weeks],
        bias_last_week=round(float(np.mean(y[last] - pred[last])) / scale, 2),
        level_change=round(float(np.mean(y[last]) - np.mean(y[before])) / scale, 2),
        amplitude_ratio=round(amplitude(last) / amplitude(before), 2),
        noise_ratio=round(noise(last) / noise(before), 2),
        weeks_since_change=round((T - last_change) / WEEK, 1),
        weeks_of_history=round(T / WEEK, 1),
        alarm=alarm,
    )


SYSTEM = """You maintain an hourly forecasting model in production. Every day at 00:00 it forecasts the next
24 hours (gradient boosting on lagged values and calendar features). At a review point you get diagnostics
computed from the data up to now and choose exactly one action from the catalog. Every retraining has a
cost, so choose "keep" when the evidence for a change is weak.

Catalog:
{catalog}

Answer with the action name and a short reason."""

USER_FULL = """Diagnostics (errors are MASE: 1.0 = error of the weekly seasonal naive forecast on the training data):
- forecast error in the first weeks after deployment: {mase_first_weeks}
- forecast error in the last weeks, oldest first: {weeks_text}
- bias last week: {bias_last_week:+} (positive: actual values above the forecast), in the same units
- level of the series last week against the 4 weeks before: {level_change:+}, in the same units
- daily profile amplitude, last week / 4 weeks before: {amplitude_ratio}
- noise (change against the same hour of the previous day), last week / 4 weeks before: {noise_ratio}
- the model was last changed {weeks_since_change} weeks ago; history available: {weeks_of_history} weeks
- {alarm_text}

Which action do you choose?"""

USER_BARE = """{alarm_text}

Which action do you choose?"""


def render_prompt(ctx: dict, with_diagnostics: bool = True) -> tuple[str, str]:
    """System and user messages. Without diagnostics (ablation) the advisor only knows that a review is due."""
    catalog = "\n".join(f'- {a.name}: {a.title_en} ({"retrains the model" if a.kind == "retrain" else "no retraining"})'
                        for a in CATALOG)
    alarm = ctx.get("alarm")
    alarm_text = (f"a drift detector ({alarm['method']}) raised an alarm {alarm['hours_ago']} hours ago" if alarm
                  else "scheduled weekly review, no detector alarm attached")
    weeks_text = ", ".join(str(w) for w in ctx["mase_last_weeks"])
    user = (USER_FULL.format(**ctx, alarm_text=alarm_text, weeks_text=weeks_text) if with_diagnostics
            else USER_BARE.format(alarm_text=alarm_text[0].upper() + alarm_text[1:]))
    return SYSTEM.format(catalog=catalog), user


def cache_key(model: str, system: str, user: str, repeat: int) -> str:
    return hashlib.sha256(json.dumps([model, system, user, repeat]).encode()).hexdigest()[:24]


class RuleAdvisor:
    """Fixed rules on the diagnostics: the error grew by at least 20% against the first weeks -> act;
    a large bias without extra noise -> recalibrate the level; noise grew a lot -> keep (retraining
    cannot remove noise); otherwise retrain on the recent window if the model is at least 2 weeks old."""

    name = "rules"

    def recommend(self, ctx: dict, repeat: int = 0, with_diagnostics: bool = True) -> Recommendation:
        worse = ctx["mase_last_weeks"][-1] / ctx["mase_first_weeks"] if ctx["mase_first_weeks"] > 0 else 1.0
        if worse < 1.2:
            return Recommendation(action="keep", reason=f"error ratio {worse:.2f} below 1.2")
        if ctx["noise_ratio"] >= 1.5:
            return Recommendation(action="keep", reason=f"noise ratio {ctx['noise_ratio']}: retraining cannot remove noise")
        if abs(ctx["bias_last_week"]) >= 0.5:
            return Recommendation(action="recalibrate_level", reason=f"bias {ctx['bias_last_week']:+} with stable noise")
        if ctx["weeks_since_change"] >= 2:
            return Recommendation(action="retrain_2w", reason=f"error ratio {worse:.2f}, model {ctx['weeks_since_change']} weeks old")
        return Recommendation(action="keep", reason="the model was changed recently")


class LLMAdvisor:
    """One LLM through Instructor with a JSON lines cache of answers. The client is created only when an
    answer is missing from the cache, so a rerun from stored answers needs no API key."""

    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: Path = CACHE_DIR, mode=None, max_retries: int = 2,
                 max_tokens: int = 2048, effort: str | None = "low", client=None):
        self.name = model
        self.model, self.mode, self.max_retries, self.max_tokens, self.effort = model, mode, max_retries, max_tokens, effort
        self.cache_dir = Path(cache_dir)
        self._client = client
        self.calls = 0  # answers requested from the API in this process (not read from the cache)
        self._cache = {}
        for f in sorted(self.cache_dir.glob("answers_*.jsonl")):
            for line in f.read_text().splitlines():
                rec = json.loads(line)
                self._cache[rec["key"]] = rec

    def _get_client(self):
        if self._client is None:
            import instructor

            key = None
            if KEY_FILE.exists():
                for line in KEY_FILE.read_text().splitlines():
                    if line.strip().startswith("ANTHROPIC_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"')
            self._client = instructor.from_provider(self.model, mode=self.mode or instructor.Mode.JSON, api_key=key)
        return self._client

    def recommend(self, ctx: dict, repeat: int = 0, with_diagnostics: bool = True) -> Recommendation:
        system, user = render_prompt(ctx, with_diagnostics)
        key = cache_key(self.model, system, user, repeat)
        if key in self._cache:
            return Recommendation.model_validate(self._cache[key]["answer"])
        extra = {"extra_body": {"output_config": {"effort": self.effort}}} if self.effort and self.model.startswith("anthropic/") else {}
        rec, completion = self._get_client().create_with_completion(
            response_model=Recommendation, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_retries=self.max_retries, max_tokens=self.max_tokens, **extra)
        self.calls += 1
        usage = getattr(completion, "usage", None)
        record = dict(key=key, model=self.model, repeat=repeat, with_diagnostics=with_diagnostics, user=user,
                      answer=rec.model_dump(), time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                      input_tokens=getattr(usage, "input_tokens", None), output_tokens=getattr(usage, "output_tokens", None))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cache_dir / f"answers_{os.getpid()}.jsonl", "a") as f:  # one file per process: no write races
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._cache[key] = record
        return rec
