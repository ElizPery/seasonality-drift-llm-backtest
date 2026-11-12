"""LLM recommendations for stream B (RQ3): the diagnostic context, the prompt, the answer schema and the
advisors that turn a decision point into one action of the catalog.

At a decision origin T the advisor sees a short description of the series computed from data before T
only: how the error of the deployed model changed week by week, its bias, how the level, the daily
profile and the noise moved, and how long ago the model was last changed. It answers with one action
name from the catalog and a short reason. Four prompt variants (`PROMPTS`) differ in what the advisor is
told: the system prompt alone ("base"), with instructions on how to read the diagnostics ("instructions"),
with the instructions and worked examples ("examples"), or no diagnostics at all ("bare", the ablation). The answer schema is a Pydantic model; Instructor validates
the reply and asks again if it does not fit (JSON mode: newer Claude models reject a forced tool call,
which the default tools mode of Instructor sends).

Every LLM answer is stored in a JSON lines cache keyed by the model, the prompt and the repeat number, so
a rerun of the experiments reads the stored answers and needs no API key. The model string names the
provider first: "openrouter/mistralai/...", "anthropic/...", "ollama/...". `RuleAdvisor` applies fixed
rules to the same numbers: a non-LLM baseline that also lets the whole pipeline run without a key.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

from src.recommendations.catalog import CATALOG
from src.recommendations.forecast import DAY, WEEK

ACTION_NAMES = tuple(a.name for a in CATALOG)
DEFAULT_MODEL = "anthropic/claude-sonnet-5-5"
CACHE_DIR = Path("data/llm_answers")  # tracked by git: a rerun from the stored answers needs no API key
# API keys are read from files, not from the shell environment: <provider>.env in this folder with one line
# <PROVIDER>_API_KEY=..., e.g. openrouter.env with OPENROUTER_API_KEY=...
KEY_DIR = Path(os.environ.get("LLM_KEY_DIR", "~/.config/diploma")).expanduser()
PROMPTS = ("base", "instructions", "examples", "bare")


class Recommendation(BaseModel):
    action: Literal[ACTION_NAMES] = Field(description="name of exactly one action from the catalog")
    reason: str = Field(max_length=400, description="one or two sentences: which numbers led to this choice")

    @field_validator("reason", mode="before")
    @classmethod
    def _cut(cls, v):
        # a reason longer than the limit is cut, not re-asked: some models write long reasons at every attempt,
        # and the action, not the wording, is what the experiment evaluates
        return v[:400] if isinstance(v, str) and len(v) > 400 else v


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

# "instructions" and "examples": how to read the diagnostics, appended to the system prompt. The thresholds are
# those of `RuleAdvisor`; the instructions also say when each of the other catalog actions fits
INSTRUCTIONS = """How to read the diagnostics:
- First compare the error of the last weeks with the error in the first weeks. If it is at most about 20%
  higher, the model still works: choose keep.
- If the noise ratio is about 1.5 or higher while the level and the daily profile hardly moved, the series
  became noisier. No action removes noise, and a model retrained on a noisy window gets worse: choose keep.
- If the series jumped once to a new level (a large bias of the same sign as the level change, noise and
  amplitude about the same, the error rose and then stopped growing), its shape did not change:
  recalibrate_level shifts the forecast by the bias of the last week without retraining.
- If the error grows week after week while the level keeps moving, the series is still changing: retrain on
  a recent window. retrain_2w follows the change fastest, retrain_8w is steadier, reweight_1w keeps all
  history but gives the last weeks the most weight.
- If the daily profile amplitude changed (a ratio far from 1) and the error grew, the shape of the day
  changed: add_profile gives the model the mean profile of the last 4 weeks.
- retrain_full suits a slow drift without a clear break when the model is many weeks old.
- If the model was changed less than 2 weeks ago, give the change time: prefer keep unless the error is
  clearly growing.
- A detector alarm alone is not a reason to act: check whether the error actually grew."""

# "examples": worked decisions at weekly reviews of the synthetic series MT_001/s9 (UCI client MT_001,
# seed 9; the experiments use seeds 0-2, so no example comes from an evaluated series), deployed model
# never changed. In each the chosen action had the lowest MASE over the following week among the catalog
# actions (keep 0.796; recalibrate_level 0.862 against keep 1.093; retrain_2w 0.844 against keep 1.390;
# keep 1.755 against 1.79 and more for any change)
EXAMPLES = (
    (dict(mase_first_weeks=0.83, mase_last_weeks=[0.85, 0.79, 0.79], bias_last_week=-0.13, level_change=-0.09,
          amplitude_ratio=1.16, noise_ratio=0.94, weeks_since_change=5.0, weeks_of_history=15.0),
     "keep", "The error of the last weeks (0.79) is not above the first weeks (0.83); bias, level and noise are "
             "small. Nothing to fix."),
    (dict(mase_first_weeks=0.83, mase_last_weeks=[0.85, 1.63, 1.09], bias_last_week=0.88, level_change=1.38,
          amplitude_ratio=1.16, noise_ratio=0.91, weeks_since_change=5.0, weeks_of_history=15.0),
     "recalibrate_level", "The level jumped (+1.38) and the forecast is too low (bias +0.88), while noise is "
                          "stable and the error already fell from 1.63 to 1.09: a one-time shift, the shape is "
                          "unchanged, so a level correction is enough."),
    (dict(mase_first_weeks=0.83, mase_last_weeks=[0.78, 0.85, 1.2], bias_last_week=0.78, level_change=0.84,
          amplitude_ratio=1.13, noise_ratio=1.04, weeks_since_change=4.0, weeks_of_history=14.0),
     "retrain_2w", "The error grows every week (0.78, 0.85, 1.2) and the level is rising (+0.84) with stable "
                   "noise: the series is still changing, so the model should learn from the recent weeks."),
    (dict(mase_first_weeks=0.83, mase_last_weeks=[0.78, 0.85, 1.42], bias_last_week=0.3, level_change=0.36,
          amplitude_ratio=1.22, noise_ratio=1.7, weeks_since_change=4.0, weeks_of_history=14.0),
     "keep", "The error grew, but together with the noise (ratio 1.7) while the bias and the level change are "
             "small: the series became noisier, and retraining cannot remove noise."),
)


def _alarm_text(ctx: dict) -> str:
    alarm = ctx.get("alarm")
    return (f"a drift detector ({alarm['method']}) raised an alarm {alarm['hours_ago']} hours ago" if alarm
            else "scheduled weekly review, no detector alarm attached")


def _user_full(ctx: dict) -> str:
    weeks_text = ", ".join(str(w) for w in ctx["mase_last_weeks"])
    return USER_FULL.format(**ctx, alarm_text=_alarm_text(ctx), weeks_text=weeks_text)


def render_examples() -> str:
    parts = ["Examples of decisions on other series:"]
    for i, (ctx, action, reason) in enumerate(EXAMPLES, 1):
        diag = _user_full(ctx).rsplit("\n\n", 1)[0]  # without the closing question
        answer = json.dumps(dict(action=action, reason=reason))
        parts.append(f"Example {i}.\n{diag}\nAnswer: {answer}")
    return "\n\n".join(parts)


def render_prompt(ctx: dict, prompt: str = "base") -> tuple[str, str]:
    """System and user messages of a prompt variant (see `PROMPTS`). "bare" (ablation): the advisor only knows
    that a review is due; "instructions" and "examples" extend the system prompt of "base"."""
    if prompt not in PROMPTS:
        raise ValueError(f"unknown prompt variant {prompt!r}, expected one of {PROMPTS}")
    catalog = "\n".join(f'- {a.name}: {a.title} ({"retrains the model" if a.kind == "retrain" else "no retraining"})'
                        for a in CATALOG)
    system = SYSTEM.format(catalog=catalog)
    if prompt in ("instructions", "examples"):
        system += "\n\n" + INSTRUCTIONS
    if prompt == "examples":
        system += "\n\n" + render_examples()
    if prompt == "bare":
        alarm_text = _alarm_text(ctx)
        return system, USER_BARE.format(alarm_text=alarm_text[0].upper() + alarm_text[1:])
    return system, _user_full(ctx)


def cache_key(model: str, system: str, user: str, repeat: int) -> str:
    return hashlib.sha256(json.dumps([model, system, user, repeat]).encode()).hexdigest()[:24]


class RuleAdvisor:
    """Fixed rules on the diagnostics: the error grew by at least 20% against the first weeks -> act;
    a large bias without extra noise -> recalibrate the level; noise grew a lot -> keep (retraining
    cannot remove noise); otherwise retrain on the recent window if the model is at least 2 weeks old."""

    name = "rules"

    def recommend(self, ctx: dict, repeat: int = 0) -> Recommendation:
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
                 max_tokens: int = 2048, effort: str | None = "low", client=None, prompt: str = "base", timeout: float = 120.0):
        # rows of a prompt variant other than "base" carry the model name with the suffix "|<variant>", so all
        # variants can live in one results file
        if prompt not in PROMPTS:
            raise ValueError(f"unknown prompt variant {prompt!r}, expected one of {PROMPTS}")
        self.prompt = prompt
        self.name = model if prompt == "base" else f"{model}|{prompt}"
        self.model, self.mode, self.max_retries, self.max_tokens, self.effort = model, mode, max_retries, max_tokens, effort
        self.timeout = timeout  # seconds per request: a hung connection must not stall a worker process
        self.cache_dir = Path(cache_dir)
        self._client = client
        self.calls = 0  # answers requested from the API in this process (not read from the cache)
        self.errors = 0  # requests that failed after all retries (the model was kept, see `error` in the record)
        self._cache = {}
        for f in sorted(self.cache_dir.glob("answers_*.jsonl")):
            for line in f.read_text().splitlines():
                rec = json.loads(line)
                self._cache[rec["key"]] = rec

    def _get_client(self):
        if self._client is None:
            import instructor

            provider = self.model.split("/", 1)[0]
            key_file, key = KEY_DIR / f"{provider}.env", None
            if key_file.exists():
                for line in key_file.read_text().splitlines():
                    if line.strip().startswith(f"{provider.upper()}_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"')
            # providers without a key (a local Ollama server) get their own default when no key file is set
            kw = {"api_key": key} if key else {}
            self._client = instructor.from_provider(self.model, mode=self.mode or instructor.Mode.JSON, **kw)
        return self._client

    def recommend(self, ctx: dict, repeat: int = 0) -> Recommendation:
        system, user = render_prompt(ctx, self.prompt)
        key = cache_key(self.model, system, user, repeat)
        if key in self._cache:
            return Recommendation.model_validate(self._cache[key]["answer"])
        # low reasoning effort: the Anthropic API takes output_config, OpenRouter a reasoning block (ignored by models without reasoning)
        extra = {}
        if self.effort and self.model.startswith("anthropic/"):
            extra = {"extra_body": {"output_config": {"effort": self.effort}}}
        elif self.effort and self.model.startswith("openrouter/"):
            extra = {"extra_body": {"reasoning": {"effort": self.effort}}}
        error, usage = None, None
        for attempt in range(3):  # transport or validation failures after the retries of Instructor: wait and ask again
            try:
                rec, completion = self._get_client().create_with_completion(
                    response_model=Recommendation, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    max_retries=self.max_retries, max_tokens=self.max_tokens, timeout=self.timeout, **extra)
                usage = getattr(completion, "usage", None)
                break
            except Exception as e:  # noqa: BLE001 - the run must not die on one request; the failure is recorded
                error = f"{type(e).__name__}: {str(e)[:300]}"
                print(f"advisor {self.model}: attempt {attempt + 1} failed: {error}", file=sys.stderr, flush=True)
                time.sleep(5 * 3**attempt)
        else:
            # the advisor is unavailable at this review: keep the model, and the record says so (`error`)
            rec = Recommendation(action="keep", reason=f"advisor unavailable: {error}"[:400])
            self.errors += 1
        self.calls += 1
        # Anthropic reports input/output tokens, OpenAI-compatible providers (OpenRouter, Ollama) prompt/completion
        tokens_in = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
        tokens_out = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
        record = dict(key=key, model=self.model, repeat=repeat, prompt=self.prompt, with_diagnostics=self.prompt != "bare",
                      user=user, answer=rec.model_dump(), time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                      input_tokens=tokens_in, output_tokens=tokens_out, error=error if usage is None else None)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cache_dir / f"answers_{os.getpid()}.jsonl", "a") as f:  # one file per process: no write races
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._cache[key] = record
        return rec
