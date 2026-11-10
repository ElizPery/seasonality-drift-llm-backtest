"""Stream B (RQ3): backtest of adaptation strategies, reference strategies and the hindsight oracles.

A deployed LightGBM model forecasts every day the next 24 hours. A strategy decides when the model is
changed and how:

* nothing        - the model trained at deployment is never changed;
* always         - retrain on all history every day;
* scheduled      - retrain on all history once a week;
* scheduled_gate - once a week, retrain on all history only if the acceptance rule confirms the gain;
* catalog_gate   - once a week, check every catalog action with the acceptance rule and apply the one with
                   the highest lower bound of the gain (or keep the model if none passes): the deployable
                   counterpart of `oracle_weekly`, and the reference for LLM recommendations with a backtest;
* oracle_weekly  - once a week, apply the catalog action with the lowest error on the following week,
                   chosen in hindsight: the reference for weekly policies and for LLM recommendations that are
                   taken at a few points. It chooses greedily from the current model, so it is not a strict
                   bound: it was never worse than `scheduled` on the 263 series, but `always` beats it on 51;
* oracle         - the same every day with a one-day lookahead: the ideal ceiling (never worse than `always`).
The oracles use future data, so they cannot be deployed; they show how far a policy with the same decision points
could go.

The strategies with `trigger="alarm"` (`ALARM_STRATEGIES`) take the same decisions, but at the first origin
after an alarm of a drift detector instead of on a calendar, then ignore alarms for `cooldown_days`: this is
the link with stream A, and the cost of false alarms shows up in their retrainings and checks on series
without a change. They run once per detector whose alarms are given (see `alarms.py`).

Quality is MASE (scale: weekly seasonal naive on the training data), cost is the number of retrainings.
`mase_gain_ci` gives the gain Delta = MASE before - MASE after with a moving block bootstrap interval over
days. `check_action` is the acceptance rule of the backtest: an action is applied only if the lower bound
of the interval for its gain on the weeks before the decision is above delta.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src.data.frames import series_id
from src.recommendations.catalog import ACTIONS, CATALOG, apply_action
from src.recommendations.forecast import DAY, LGBM_PARAMS, WEEK, Model, mase_scale, seasonal_naive
from src.recommendations.llm_client import diagnostics

# acceptance rule: the gain of an action is checked on the last two weeks before the decision; daily gains
# are resampled in blocks of 3 days (about n^(1/3) for 14 days, 7-day blocks would leave too few distinct
# resamples); the action passes if the lower bound of the 95% interval is above DELTA (MASE units)
VAL_WEEKS = 2
GATE_BLOCK_DAYS = 3
DELTA = 0.0
SKIPPED = f"fewer than {VAL_WEEKS} weeks of own forecasts to check against"


@dataclass(frozen=True)
class Strategy:
    name: str
    title: str
    cadence_days: int | None  # how often a decision is taken; None = never
    policy: str  # fixed | oracle | gated | advised
    action: str | None = None  # gated: the only candidate; None = every catalog action except keep
    advisor: str | None = None  # advised: key of the advisor that proposes one action (see llm_client)
    gate: bool = False  # advised: apply the proposal only if the acceptance rule confirms it
    trigger: str = "calendar"  # calendar: every `cadence_days`; alarm: at the first origin after a detector alarm
    cooldown_days: int = 7  # alarm: alarms this soon after a review are ignored


STRATEGIES = (
    Strategy("nothing", "do nothing", None, "fixed"),
    Strategy("always", "always retrain (daily)", 1, "fixed", "retrain_full"),
    Strategy("scheduled", "scheduled retraining (weekly)", 7, "fixed", "retrain_full"),
    Strategy("scheduled_gate", "retrain weekly, only if the backtest check confirms a gain", 7, "gated", "retrain_full"),
    Strategy("catalog_gate", "weekly, the best catalog action among those that pass the backtest check", 7, "gated"),
    Strategy("oracle_weekly", "weekly oracle: the best catalog action in hindsight over the next week", 7, "oracle"),
    Strategy("oracle", "ideal oracle: the best catalog action in hindsight, daily", 1, "oracle"),
)
# strategies with an advisor: run only when the advisors are given (`advisors` of `backtest`)
ADVISED_STRATEGIES = (
    Strategy("rules", "rules on the diagnostics, no backtest check", 7, "advised", advisor="rules"),
    Strategy("rules_gate", "rules on the diagnostics with the backtest check", 7, "advised", advisor="rules", gate=True),
    Strategy("llm", "LLM, no backtest check", 7, "advised", advisor="llm"),
    Strategy("llm_gate", "LLM with the diagnostics and the backtest check", 7, "advised", advisor="llm", gate=True),
)
# strategies that review the model on a detector alarm: run only when alarms are given (`alerts` of `evaluate_series`);
# `cadence_days` is the lookahead window of the oracle; the advised ones also need their advisor
ALARM_STRATEGIES = (
    Strategy("alarm_retrain", "retrain on a detector alarm, no backtest check", 7, "fixed", "retrain_full", trigger="alarm"),
    Strategy("alarm_gate", "retrain on an alarm, only if the backtest check confirms a gain", 7, "gated", "retrain_full", trigger="alarm"),
    Strategy("alarm_catalog_gate", "on an alarm, the best catalog action among those that pass the backtest check", 7, "gated", trigger="alarm"),
    Strategy("alarm_oracle", "oracle on an alarm: the best catalog action in hindsight over the next week", 7, "oracle", trigger="alarm"),
    Strategy("alarm_rules", "on an alarm, rules on the diagnostics, no backtest check", 7, "advised", advisor="rules", trigger="alarm"),
    Strategy("alarm_rules_gate", "on an alarm, rules on the diagnostics with the backtest check", 7, "advised", advisor="rules", gate=True, trigger="alarm"),
    Strategy("alarm_llm", "on an alarm, LLM, no backtest check", 7, "advised", advisor="llm", trigger="alarm"),
    Strategy("alarm_llm_gate", "on an alarm, LLM with the diagnostics and the backtest check", 7, "advised", advisor="llm", gate=True, trigger="alarm"),
)


def make_actor(F: dict[str, np.ndarray], y: np.ndarray, base_features: tuple[str, ...], params: dict | None = None,
               cache: dict | None = None):
    """`act(action, model, origin) -> (model, fits)`. Retraining actions do not depend on the current model,
    so their fits are shared through `cache` (keyed by action and origin)."""
    cache = {} if cache is None else cache

    def act(a, model, T):
        if a.kind != "retrain":
            return apply_action(a, model, F, y, T, base_features, params)
        key = (a.name, T)
        if key not in cache:
            cache[key] = apply_action(a, model, F, y, T, base_features, params)[0]
        return cache[key], 1

    return act


def check_action(a, model: Model, y: np.ndarray, F: dict[str, np.ndarray], pred: np.ndarray, T: int, scale: float, act,
                 val_weeks: int = VAL_WEEKS, block_days: int = GATE_BLOCK_DAYS, delta: float = DELTA) -> dict:
    """Acceptance rule: backtest of one recommended action at the decision origin T, on data before T only.

    The action is replayed on the last `val_weeks` weeks with a weekly rolling origin: at each origin o its
    recipe is applied to data before o and forecasts the week [o, o + 7 days). The reference is what the
    deployed strategy actually forecast in those weeks (`pred`: day ahead, out of sample). Gain = MASE of
    the reference - MASE of the action, with a moving block bootstrap interval over days; the action is
    accepted only if the lower bound of the 95% interval is above `delta`. `fits` counts the model fits
    the check needed (shared fits included), the computational cost of the backtest."""
    if a.kind not in ("retrain", "recalibrate"):
        raise ValueError(f"{a.name} does not change point forecasts")
    start = T - val_weeks * WEEK
    cand = np.full(len(y), np.nan)
    fits = 0
    for o in range(start, T, WEEK):
        rows = np.arange(o, o + WEEK)
        if a.kind == "recalibrate":  # level correction from the reference errors of the week before o
            prev = np.arange(o - a.window, o)
            cand[rows] = pred[rows] + np.mean(y[prev] - pred[prev])
        else:
            m, n_fits = act(a, model, o)
            cand[rows] = m.predict(F, rows)
            fits += n_fits
    if not (np.isfinite(cand[start:T]).all() and np.isfinite(pred[start:T]).all()):
        return dict(action=a.name, gain=np.nan, lo=np.nan, hi=np.nan, accepted=False, fits=fits)
    g, lo, hi = mase_gain_ci(y, pred, cand, start, T, scale, block_days=block_days)
    return dict(action=a.name, gain=g, lo=lo, hi=hi, accepted=bool(lo > delta), fits=fits)


class StrategyRun:
    """One strategy on one series, one origin at a time. `step(T, alarms)` takes the decision due at origin T, if
    any, and then forecasts the day [T, T + 24). `backtest` drives it over the whole series, and the monitoring
    service (`src.api`) drives it as the data arrive, so both take exactly the same decisions. `alarms` are the
    alarm positions of the detector known at T (the backtest passes all of them; only those before T count).
    Every decision is also kept in `records`: when and why it was taken, what the advisor proposed and why, the
    checks of the acceptance rule and the action applied."""

    def __init__(self, y: np.ndarray, F: dict[str, np.ndarray], t_deploy: int, strategy: Strategy,
                 base_features: tuple[str, ...], params: dict | None = None, catalog=CATALOG, cache: dict | None = None,
                 advisors: dict | None = None, repeat: int = 0, detector: str | None = None):
        self.y, self.F, self.t_deploy, self.strategy, self.catalog = y, F, t_deploy, strategy, catalog
        self.advisors, self.repeat, self.detector = advisors, repeat, detector
        self.origins = list(range(t_deploy, len(y) - DAY + 1, DAY))
        self.end = self.origins[-1] + DAY
        self.cache = {} if cache is None else cache
        self.act = make_actor(F, y, base_features, params, self.cache)
        self.scale = mase_scale(y[:t_deploy])
        self.model, _ = self.act(ACTIONS["retrain_full"], Model(base_features), t_deploy)
        self.pred = np.full(len(y), np.nan)
        self.retrains, self.picks, self.applied, self.records = 0, [], [], []
        self.decisions = self.proposed = self.accepted = self.check_fits = 0
        self.last_change = t_deploy
        self.seen = None  # alarms consumed or ignored so far; set at the first step
        self.last_review = t_deploy - strategy.cooldown_days * DAY

    def step(self, T: int, alarms: np.ndarray | None = None) -> dict | None:
        """The decision due at origin T (returns its record, or None if no decision was due), then the forecast
        of the day [T, T + 24). An alarm strategy reviews the model at the first origin after an alarm (alarms
        before the origin are known, the origin itself is not) and then ignores alarms for `cooldown_days`;
        alarms before deployment are ignored."""
        st = self.strategy
        alarm = None
        if st.trigger == "alarm":
            alarms = np.asarray([] if alarms is None else alarms, dtype=int)
            if self.seen is None:
                self.seen = int(np.searchsorted(alarms, self.t_deploy))
            known = int(np.searchsorted(alarms, T))  # alarms before this origin
            if T - self.last_review < st.cooldown_days * DAY:
                self.seen = known
            due = known > self.seen
            if due:
                alarm = dict(method=self.detector, hours_ago=int(T - alarms[known - 1]))
                self.seen, self.last_review = known, T
        else:
            k = (T - self.t_deploy) // DAY
            due = bool(st.cadence_days) and k > 0 and k % st.cadence_days == 0
        record = self._decide(T, alarm) if due else None
        rows = np.arange(T, T + DAY)
        self.pred[rows] = self.model.predict(self.F, rows)
        return record

    def _decide(self, T: int, alarm: dict | None) -> dict:
        st, y, F, act = self.strategy, self.y, self.F, self.act
        rec = dict(origin=T, trigger=st.trigger, alarm=alarm, proposal=None, reason=None, checks=[], action="keep", fits=0)
        changed = False
        if st.policy == "fixed":
            self.model, fits = act(ACTIONS[st.action], self.model, T)
            self.retrains += fits
            rec.update(proposal=st.action, action=st.action, fits=fits)
            changed = True
        elif st.policy == "gated":
            if T - VAL_WEEKS * WEEK >= self.t_deploy:
                cands = [ACTIONS[st.action]] if st.action else [a for a in self.catalog if a.kind != "none"]
                checks = [check_action(a, self.model, y, F, self.pred, T, self.scale, act) for a in cands]
                self.check_fits += sum(c["fits"] for c in checks)
                ok = [c for c in checks if c["accepted"]]
                self.decisions += 1
                self.proposed += 1
                rec["checks"] = checks
                if ok:  # ties go to the earlier, cheaper action of the catalog
                    best = max(ok, key=lambda c: c["lo"])["action"]
                    self.model, fits = act(ACTIONS[best], self.model, T)
                    self.retrains += fits
                    self.accepted += 1
                    changed = True
                    rec.update(proposal=best, action=best, fits=fits)
                self.picks.append(best if ok else "keep")
            else:
                rec["skipped"] = SKIPPED
        elif st.policy == "advised":
            if T - VAL_WEEKS * WEEK >= self.t_deploy:
                ctx = diagnostics(y, self.pred, T, self.t_deploy, self.scale, self.last_change, alarm=alarm)
                answer = self.advisors[st.advisor].recommend(ctx, self.repeat)
                a = ACTIONS[answer.action]
                self.decisions += 1
                rec.update(proposal=a.name, reason=answer.reason, diagnostics=ctx)
                ok = a.kind != "none"
                if ok:
                    self.proposed += 1
                    if st.gate:
                        c = check_action(a, self.model, y, F, self.pred, T, self.scale, act)
                        self.check_fits += c["fits"]
                        ok = c["accepted"]
                        rec["checks"] = [c]
                if ok:
                    self.model, fits = act(a, self.model, T)
                    self.retrains += fits
                    self.accepted += 1
                    self.last_change = T
                    changed = True
                    rec.update(action=a.name, fits=fits)
                self.picks.append(a.name if ok else "keep")
            else:
                rec["skipped"] = SKIPPED
        else:  # hindsight oracle: best action on the next decision window
            rows = np.arange(T, min(T + st.cadence_days * DAY, self.end))
            best = None
            for a in self.catalog:  # `keep` first, so ties go to the cheapest action
                m_a, fits = act(a, self.model, T)
                err = float(np.nanmean(np.abs(y[rows] - m_a.predict(F, rows))))
                if best is None or err < best[0] - 1e-12:
                    best = (err, a.name, m_a, fits)
            _, name, self.model, fits = best
            self.retrains += fits
            self.picks.append(name)
            changed = name != "keep"
            rec.update(proposal=name, action=name, fits=fits)
        if changed:
            self.applied.append(T)
        rec["changed"] = changed
        self.records.append(rec)
        return rec

    def result(self, alarms: np.ndarray | None = None) -> dict:
        alarms = np.asarray([] if alarms is None else alarms, dtype=int)
        return dict(pred=self.pred, retrains=self.retrains, picks=self.picks, end=self.end, decisions=self.decisions,
                    proposed=self.proposed, accepted=self.accepted, check_fits=self.check_fits, applied=self.applied,
                    alarms=int(((alarms >= self.t_deploy) & (alarms < self.end)).sum()))


def backtest(y: np.ndarray, F: dict[str, np.ndarray], t_deploy: int, strategy: Strategy, base_features: tuple[str, ...],
             params: dict | None = None, catalog=CATALOG, cache: dict | None = None, prune_cache: bool = False,
             advisors: dict | None = None, repeat: int = 0, alarms=None, detector: str | None = None) -> dict:
    """Rolling day-ahead forecasts from `t_deploy` to the end of the series under one strategy (see `StrategyRun`).

    Retraining fits are shared through `cache` between strategies of the same series (see `make_actor`).
    With `prune_cache` (last strategy of a series) fits older than the check window are dropped as the run
    moves on, which bounds memory. Gated and advised strategies decide only once they have `VAL_WEEKS`
    weeks of their own forecasts to check against. `advisors` maps advisor keys to objects with
    `recommend(context, repeat)`; `repeat` numbers repeated runs of a non-deterministic advisor.
    `alarms` are the positions of the alarms of the detector `detector` on this series. Counters: `decisions`
    (review points), `proposed` (proposals of a change), `accepted` (changes applied), `applied` (origins at which
    the model was changed), `alarms` (alarms after deployment)."""
    run = StrategyRun(y, F, t_deploy, strategy, base_features, params, catalog, cache, advisors, repeat, detector)
    alarms = np.sort(np.asarray([] if alarms is None else alarms, dtype=int))
    for T in run.origins:
        run.step(T, alarms)
        if prune_cache:
            for key in [key for key in run.cache if key[1] <= T - VAL_WEEKS * WEEK]:
                del run.cache[key]
    return run.result(alarms)


def mase(y: np.ndarray, pred: np.ndarray, start: int, stop: int, scale: float) -> float:
    rows = np.arange(start, stop)
    return float(np.nanmean(np.abs(y[rows] - pred[rows])) / scale)


def mase_gain_ci(y: np.ndarray, pred_before: np.ndarray, pred_after: np.ndarray, start: int, stop: int, scale: float,
                 block_days: int = 7, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Gain = MASE(before) - MASE(after) on [start, stop) with a (1 - alpha) moving block bootstrap interval.

    Hourly scaled differences are averaged per day first (intra-day errors are strongly correlated), then
    whole blocks of `block_days` days are resampled, which keeps the day-to-day dependence."""
    rows = np.arange(start, stop)
    d = (np.abs(y[rows] - pred_before[rows]) - np.abs(y[rows] - pred_after[rows])) / scale
    days = np.nanmean(d[: len(d) // DAY * DAY].reshape(-1, DAY), axis=1)
    nb = len(days)
    L = max(1, min(block_days, nb))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, nb - L + 1, size=(n_boot, int(np.ceil(nb / L))))
    idx = (starts[..., None] + np.arange(L)).reshape(n_boot, -1)[:, :nb]
    boots = days[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(days.mean()), float(lo), float(hi)


def evaluate_series(s: dict, params: dict | None = None, strategies=STRATEGIES, advisors: dict | None = None,
                    repeat: int = 0, alerts: dict | None = None) -> list[dict]:
    """All strategies on one series. `s` holds y, F, t_deploy, t_change, base_features and labels. `alerts`
    maps a detector name to its alarm positions on this series; a strategy with `trigger="alarm"` runs once
    per detector (column `detector`) and is skipped without alerts. Besides MASE and retrainings, every row
    gets `acts_pre` (model changes before the change of the series, all of them on a series without one:
    the cost of false alarms or of a fixed calendar) and `reaction_h` (hours from the change to the first
    model change after it)."""
    y, F, t_dep, t_chg = s["y"], s["F"], s["t_deploy"], s["t_change"]
    scale = mase_scale(y[:t_dep])
    cache: dict = {}
    jobs = [(st, d) for st in strategies for d in ([None] if st.trigger == "calendar" else sorted(alerts or {}))]
    runs = {}
    for i, (st, d) in enumerate(jobs):
        runs[(st.name, d)] = backtest(y, F, t_dep, st, s["base_features"], params, cache=cache, prune_cache=(i == len(jobs) - 1),
                                      advisors=advisors, repeat=repeat, alarms=None if d is None else alerts[d], detector=d)
    cache.clear()
    end = runs[("nothing", None)]["end"]
    naive = np.full(len(y), np.nan)
    naive[t_dep:end] = seasonal_naive(y, np.arange(t_dep, end))
    runs[("naive", None)] = dict(pred=naive, retrains=0, picks=[], end=end)
    gated = {st.name for st in strategies if st.policy in ("gated", "advised")}
    # advised strategies record which advisor answered (`model`: the LLM model string, "|bare" for the ablation,
    # or "rules") and the number of the repeated run, so runs of several models and repeats share one file
    advised = {st.name: advisors[st.advisor].name for st in strategies if st.policy == "advised"}
    labels = {k: v for k, v in s.items() if k not in ("y", "F", "base_features", "start")}
    rows = []
    for (name, d), r in runs.items():
        applied = r.get("applied", [])
        row = dict(labels, strategy=name, detector=d, model=advised.get(name, np.nan),
                   repeat=repeat if name in advised else np.nan, scale=scale, retrains=r["retrains"],
                   mase_all=mase(y, r["pred"], t_dep, end, scale),
                   mase_pre=mase(y, r["pred"], t_dep, t_chg, scale) if t_chg and t_chg > t_dep else np.nan,
                   mase_post=mase(y, r["pred"], t_chg, end, scale) if t_chg else np.nan,
                   picks=" ".join(f"{a}:{c}" for a, c in Counter(r["picks"]).most_common()),
                   **{k: r[k] if name in gated else np.nan for k in ("decisions", "proposed", "accepted", "check_fits")},
                   alarms=r.get("alarms", np.nan) if d is not None else np.nan,
                   acts_pre=sum(1 for T in applied if not t_chg or T < t_chg) if name != "naive" else np.nan,
                   reaction_h=next((T - t_chg for T in applied if T >= t_chg), np.nan) if t_chg else np.nan)
        if name != "nothing":
            a = t_chg if t_chg else t_dep
            g, lo, hi = mase_gain_ci(y, runs[("nothing", None)]["pred"], r["pred"], a, end, scale)
            row.update(gain_vs_nothing=g, gain_lo=lo, gain_hi=hi)
        rows.append(row)
    return rows


_WORKER: dict = {}  # advisors of a worker process, created on its first series


def _job(s, strategies=STRATEGIES, advisors: tuple[str, ...] = (), llm_model: str | None = None, alerts: dict | None = None,
         detectors: tuple[str, ...] = (), repeats: int = 1, no_diagnostics: bool = False):
    if advisors and _WORKER.get("key") != (advisors, llm_model, no_diagnostics):
        from src.recommendations.llm_client import DEFAULT_MODEL, LLMAdvisor, RuleAdvisor

        made = {"rules": RuleAdvisor, "llm": lambda: LLMAdvisor(llm_model or DEFAULT_MODEL, with_diagnostics=not no_diagnostics)}
        _WORKER["advisors"] = {k: made[k]() for k in advisors}
        _WORKER["key"] = (advisors, llm_model, no_diagnostics)
    mine = None
    if alerts is not None:  # every detector runs on every series; one without alarms on this series gets an empty list
        mine = alerts.get(series_id(s), {})
        mine = {d: mine.get(d, np.array([], dtype=int)) for d in (detectors or sorted(mine))}
    rows = evaluate_series(s, LGBM_PARAMS, strategies, _WORKER.get("advisors"), alerts=mine)
    # repeated runs of a non-deterministic advisor: only the advised strategies are run again
    for r in range(1, repeats if advisors else 1):
        rows += [row for row in evaluate_series(s, LGBM_PARAMS, strategies, _WORKER.get("advisors"), repeat=r, alerts=mine)
                 if isinstance(row["model"], str)]
    return rows


def sort_runs(df: pd.DataFrame) -> pd.DataFrame:
    order = ["naive"] + [s.name for s in STRATEGIES + ADVISED_STRATEGIES + ALARM_STRATEGIES]
    if "detector" not in df:
        df = df.assign(detector=np.nan)
    df = df.assign(_o=df.strategy.map(order.index), _d=df.detector.fillna(""))
    return df.sort_values(["source", "scenario", "series", "_o", "_d"]).drop(columns=["_o", "_d"]).reset_index(drop=True)


def run_backtests(series: list[dict], processes: int | None = None, log=None, strategies=STRATEGIES,
                  advisors: tuple[str, ...] = (), llm_model: str | None = None, alerts: dict | None = None,
                  repeats: int = 1, no_diagnostics: bool = False) -> pd.DataFrame:
    """`alerts`: alarm positions per series id and detector (see `alarms.read_alerts`) for the alarm strategies.
    `repeats`: how many times the advised strategies run (column `repeat`); `no_diagnostics`: the LLM ablation."""
    rows = []
    detectors = tuple(sorted({d for per in (alerts or {}).values() for d in per}))
    job = partial(_job, strategies=strategies, advisors=advisors, llm_model=llm_model, alerts=alerts, detectors=detectors,
                  repeats=repeats, no_diagnostics=no_diagnostics)
    with Pool(processes) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, series, chunksize=1), 1):
            rows.extend(rs)
            if log is not None:
                print(f"{i}/{len(series)} series done ({rs[0]['source']} {rs[0]['series']} {rs[0]['scenario']})", file=log, flush=True)
    return sort_runs(pd.DataFrame(rows))


def tune_base_model(series: list[dict], configs: dict[str, dict], processes: int | None = None) -> pd.DataFrame:
    """Validation of base-model settings: a model trained at deployment and never changed, MASE on the
    weeks that follow (series and windows here must not overlap the test experiments)."""
    jobs = [(s, name, cfg) for s in series for name, cfg in configs.items()]
    with Pool(processes) as pool:
        rows = pool.map(_tune_job, jobs, chunksize=1)
    return pd.DataFrame(rows)


def _tune_job(args):
    s, name, cfg = args
    y, F, t_dep = s["y"], s["F"], s["t_deploy"]
    stop = s["t_stop"]
    scale = mase_scale(y[:t_dep])
    model, _ = apply_action(ACTIONS["retrain_full"], Model(s["base_features"]), F, y, t_dep, s["base_features"], cfg)
    rows = np.arange(t_dep, stop)
    return dict(source=s["source"], series=s["series"], config=name,
                mase=float(np.mean(np.abs(y[rows] - model.predict(F, rows))) / scale),
                mase_naive=float(np.mean(np.abs(y[rows] - seasonal_naive(y, rows))) / scale))


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean over series per source, scenario and strategy; regret = MASE after the change minus the ideal oracle's."""
    oracle = runs[runs.strategy == "oracle"].set_index(["source", "series", "scenario"]).mase_post
    r = runs.join(oracle.rename("oracle_post"), on=["source", "series", "scenario"])
    r["regret"] = r.mase_post - r.oracle_post
    r["gain_significant"] = (r.gain_lo > 0).astype(float).where(r.gain_lo.notna())
    for c in ("decisions", "proposed", "accepted", "check_fits", "detector", "alarms", "acts_pre", "reaction_h", "model", "repeat"):
        if c not in r:  # absent in results computed before the acceptance rule, the alarm strategies or the advisors
            r[c] = np.nan
    r["accept_rate"] = r.accepted / r.proposed.fillna(r.decisions).where(lambda x: x > 0)  # share of proposals applied
    # means over series and over the repeats of an advisor; `runs` counts rows (series x repeats)
    g = r.groupby(["source", "scenario", "strategy", r.detector.fillna(""), r.model.fillna("")], sort=False).agg(
        series=("series", "nunique"), runs=("series", "size"), mase_pre=("mase_pre", "mean"), mase_post=("mase_post", "mean"),
        regret=("regret", "mean"), retrains=("retrains", "mean"), gain_vs_nothing=("gain_vs_nothing", "mean"),
        share_gain_significant=("gain_significant", "mean"), accept_rate=("accept_rate", "mean"),
        check_fits=("check_fits", "mean"), alarms=("alarms", "mean"), acts_pre=("acts_pre", "mean"),
        reaction_h=("reaction_h", "mean")).reset_index()
    return g


def oracle_picks(runs: pd.DataFrame) -> pd.DataFrame:
    """Share of decisions in which each oracle chose each catalog action."""
    rows = []
    oracles = runs[runs.strategy.isin(["oracle_weekly", "oracle"])]
    for (st, src, sc), g in oracles.groupby(["strategy", "source", "scenario"], sort=False):
        c = Counter()
        for p in g.picks.dropna():
            for item in p.split():
                a, k = item.split(":")
                c[a] += int(k)
        total = sum(c.values())
        rows.append(dict(strategy=st, source=src, scenario=sc, decisions=total,
                         **{a.name: round(c[a.name] / total, 2) if total else 0 for a in CATALOG}))
    return pd.DataFrame(rows)
