"""Probe of the prompt wording (RQ3): do the advisors avoid retraining on all history because the base system
message asks them to keep the model when the evidence is weak and says nothing about the cost of an action?

    python -m src.recommendations.prompt_probe                 # table from the stored answers (data/llm_probe)
    python -m src.recommendations.prompt_probe --models openai/gpt-5-mini --n 40   # new sample: asks where no answer is stored

Diagnostics that the models answered in the weekly runs with the base prompt (repeat 0) are sent again, unchanged,
with a system message that states the goal (the lowest error), that every retraining costs the same and that a
backtest checks the proposal before it is applied. Half of the sample are reviews at which the error of the last
week is at least 1.2 times the error of the first weeks. The table compares, per model, the actions chosen with
the base system message and with the clarified one on the same diagnostics.

The answers of the probe are stored in data/llm_probe (kept in git), apart from the cache of the experiments,
so the table is reproduced without an API key. The probe is a check of the wording, not an experiment of the
backtest: its answers do not enter runs.csv.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from src.recommendations.catalog import CATALOG
from src.recommendations.llm_client import CACHE_DIR, SYSTEM, LLMAdvisor, Recommendation, cache_key

PROBE_FILE = Path("data/llm_probe/answers_clarified.jsonl")
MODELS = ("mistralai/mistral-small-3.2-24b-instruct", "anthropic/claude-haiku-5.5", "openai/gpt-5-mini")
CLARIFIED = """You maintain an hourly forecasting model in production. Every day at 00:00 it forecasts the next
24 hours (gradient boosting on lagged values and calendar features). At a review point you get diagnostics
computed from the data up to now and choose exactly one action from the catalog.

The goal is the lowest forecast error over the following weeks. Every action that retrains the model costs
the same: one model fit, whatever the training window. Your proposal is not applied blindly: it is first
replayed on the last two weeks, and it is applied only if it would have reduced the error there. A rejected
proposal changes nothing, so propose the action you expect to reduce the error most, and "keep" only when
you expect no action to help.

Catalog:
{catalog}

Answer with the action name and a short reason."""


def catalog_text() -> str:
    return "\n".join(f'- {a.name}: {a.title} ({"retrains the model" if a.kind == "retrain" else "no retraining"})' for a in CATALOG)


def base_answers(cache_dir: Path = CACHE_DIR) -> dict[str, dict[str, str]]:
    """Action per user message and model in the weekly runs with the base prompt, repeat 0."""
    out: dict[str, dict[str, str]] = {}
    for f in sorted(Path(cache_dir).glob("answers_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("prompt") == "base" and r["repeat"] == 0 and "scheduled weekly review" in r["user"]:
                out.setdefault(r["user"], {})[r["model"]] = r["answer"]["action"]
    return out


def error_grew(user: str) -> bool:
    first = float(re.search(r"after deployment: ([\d.]+)", user).group(1))
    last = float(re.search(r"oldest first: (.*)", user).group(1).split(", ")[-1])
    return last >= 1.2 * first


def sample(base: dict, models: tuple[str, ...], n: int, seed: int = 11) -> list[str]:
    """`n` diagnostics that every model of the probe answered in the runs, half of them with a grown error."""
    users = sorted(u for u, d in base.items() if all(f"openrouter/{m}" in d for m in models))
    grew = [u for u in users if error_grew(u)]
    rest = [u for u in users if not error_grew(u)]
    rng = random.Random(seed)
    k = min(n // 2, len(grew))
    return rng.sample(grew, k) + rng.sample(rest, n - k)


def stored(path: Path = PROBE_FILE) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def ask(models: tuple[str, ...], users: list[str], path: Path = PROBE_FILE) -> None:
    """Send the clarified system message with each of `users` to each model; answers already stored are kept."""
    system = CLARIFIED.format(catalog=catalog_text())
    have = {r["key"] for r in stored(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    for m in models:
        advisor = LLMAdvisor(f"openrouter/{m}")
        for u in users:
            key = cache_key(advisor.model, system, u, 0)
            if key in have:
                continue
            rec, _ = advisor._get_client().create_with_completion(
                response_model=Recommendation, messages=[{"role": "system", "content": system}, {"role": "user", "content": u}],
                max_retries=2, max_tokens=2048, timeout=120, extra_body={"reasoning": {"effort": "low"}})
            with open(path, "a") as f:
                f.write(json.dumps(dict(key=key, model=advisor.model, user=u, answer=rec.model_dump(), hot=error_grew(u))) + "\n")


def table(records: list[dict], base: dict) -> list[dict]:
    """Per model and group of reviews: how often each action was chosen with the base and with the clarified message."""
    rows = []
    for model in sorted({r["model"] for r in records}):
        for grew in (True, False):
            rs = [r for r in records if r["model"] == model and r["hot"] == grew]
            if rs:
                rows.append(dict(model=model.replace("openrouter/", ""), error_grew=grew, n=len(rs),
                                 base=dict(Counter(base[r["user"]][model] for r in rs).most_common()),
                                 clarified=dict(Counter(r["answer"]["action"] for r in rs).most_common())))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--models", default=None, help="comma-separated models of the gateway to ask (default: only read the stored answers)")
    p.add_argument("--n", type=int, default=40, help="diagnostics per model")
    args = p.parse_args()
    base = base_answers()
    if args.models:
        models = tuple(args.models.split(","))
        ask(models, sample(base, models, args.n))
    records = stored()
    assert all(r["key"] == cache_key(r["model"], CLARIFIED.format(catalog=catalog_text()), r["user"], 0) for r in records), \
        "a stored answer does not belong to the clarified system message"
    assert SYSTEM.format(catalog=catalog_text()) != CLARIFIED.format(catalog=catalog_text())
    for row in table(records, base):
        print(f'{row["model"]}, error {"grew" if row["error_grew"] else "did not grow"}, n = {row["n"]}')
        print(f'   base:      {row["base"]}')
        print(f'   clarified: {row["clarified"]}')
    total = Counter(r["answer"]["action"] for r in records)
    print(f"{len(records)} answers with the clarified message, retrain_full chosen {total['retrain_full']} times")


if __name__ == "__main__":
    main()
