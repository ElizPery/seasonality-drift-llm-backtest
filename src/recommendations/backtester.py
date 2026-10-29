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
                   chosen in hindsight: the ceiling for any weekly policy (it is never worse than
                   `scheduled`), and the reference for LLM recommendations that are taken at a few points;
* oracle         - the same every day with a one-day lookahead: the ideal ceiling (never worse than `always`).
The oracles use future data, so they cannot be deployed; they only bound what any policy can reach.

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

from src.recommendations.catalog import ACTIONS, CATALOG, apply_action
from src.recommendations.forecast import DAY, LGBM_PARAMS, WEEK, Model, mase_scale, seasonal_naive
from src.recommendations.llm_client import diagnostics

# acceptance rule: the gain of an action is checked on the last two weeks before the decision; daily gains
# are resampled in blocks of 3 days (about n^(1/3) for 14 days, 7-day blocks would leave too few distinct
# resamples); the action passes if the lower bound of the 95% interval is above DELTA (MASE units)
VAL_WEEKS = 2
GATE_BLOCK_DAYS = 3
DELTA = 0.0


@dataclass(frozen=True)
class Strategy:
    name: str
    title_uk: str
    cadence_days: int | None  # how often a decision is taken; None = never
    policy: str  # fixed | oracle | gated | advised
    action: str | None = None  # gated: the only candidate; None = every catalog action except keep
    advisor: str | None = None  # advised: key of the advisor that proposes one action (see llm_client)
    gate: bool = False  # advised: apply the proposal only if the acceptance rule confirms it


STRATEGIES = (
    Strategy("nothing", "нічого не робити", None, "fixed"),
    Strategy("always", "завжди перенавчати (щодня)", 1, "fixed", "retrain_full"),
    Strategy("scheduled", "перенавчання за розкладом (щотижня)", 7, "fixed", "retrain_full"),
    Strategy("scheduled_gate", "перенавчання щотижня, лише якщо бектест підтверджує виграш", 7, "gated", "retrain_full"),
    Strategy("catalog_gate", "щотижня найкраща дія каталогу з тих, що пройшли бектест", 7, "gated"),
    Strategy("oracle_weekly", "оракул щотижня: найкраща дія каталогу за фактом на наступному тижні", 7, "oracle"),
    Strategy("oracle", "ідеальний оракул: найкраща дія каталогу за фактом щодня", 1, "oracle"),
)
# strategies with an advisor: run only when the advisors are given (`advisors` of `backtest`)
ADVISED_STRATEGIES = (
    Strategy("rules", "правила на діагностиці, без бектесту", 7, "advised", advisor="rules"),
    Strategy("rules_gate", "правила на діагностиці з бектестом", 7, "advised", advisor="rules", gate=True),
    Strategy("llm", "LLM без бектесту", 7, "advised", advisor="llm"),
    Strategy("llm_gate", "LLM з діагностикою та бектестом", 7, "advised", advisor="llm", gate=True),
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


def backtest(y: np.ndarray, F: dict[str, np.ndarray], t_deploy: int, strategy: Strategy, base_features: tuple[str, ...],
             params: dict | None = None, catalog=CATALOG, cache: dict | None = None, prune_cache: bool = False,
             advisors: dict | None = None, repeat: int = 0) -> dict:
    """Rolling day-ahead forecasts from `t_deploy` to the end of the series under one strategy.

    Retraining fits are shared through `cache` between strategies of the same series (see `make_actor`).
    With `prune_cache` (last strategy of a series) fits older than the check window are dropped as the run
    moves on, which bounds memory. Gated and advised strategies decide only once they have `VAL_WEEKS`
    weeks of their own forecasts to check against. `advisors` maps advisor keys to objects with
    `recommend(context, repeat)`; `repeat` numbers repeated runs of a non-deterministic advisor.
    Counters: `decisions` (review points), `proposed` (proposals of a change), `accepted` (changes applied)."""
    n = len(y)
    origins = list(range(t_deploy, n - DAY + 1, DAY))
    end = origins[-1] + DAY
    cache = {} if cache is None else cache
    act = make_actor(F, y, base_features, params, cache)
    scale = mase_scale(y[:t_deploy])

    model, _ = act(ACTIONS["retrain_full"], Model(base_features), t_deploy)
    pred = np.full(n, np.nan)
    retrains, picks = 0, []
    decisions = proposed = accepted = check_fits = 0
    last_change = t_deploy
    for k, T in enumerate(origins):
        if strategy.cadence_days and k > 0 and k % strategy.cadence_days == 0:
            if strategy.policy == "fixed":
                model, fits = act(ACTIONS[strategy.action], model, T)
                retrains += fits
            elif strategy.policy == "gated":
                if T - VAL_WEEKS * WEEK >= t_deploy:
                    cands = [ACTIONS[strategy.action]] if strategy.action else [a for a in catalog if a.kind != "none"]
                    checks = [check_action(a, model, y, F, pred, T, scale, act) for a in cands]
                    check_fits += sum(c["fits"] for c in checks)
                    ok = [c for c in checks if c["accepted"]]
                    decisions += 1
                    proposed += 1
                    if ok:  # ties go to the earlier, cheaper action of the catalog
                        best = max(ok, key=lambda c: c["lo"])["action"]
                        model, fits = act(ACTIONS[best], model, T)
                        retrains += fits
                        accepted += 1
                    picks.append(best if ok else "keep")
            elif strategy.policy == "advised":
                if T - VAL_WEEKS * WEEK >= t_deploy:
                    ctx = diagnostics(y, pred, T, t_deploy, scale, last_change)
                    a = ACTIONS[advisors[strategy.advisor].recommend(ctx, repeat).action]
                    decisions += 1
                    ok = a.kind != "none"
                    if ok:
                        proposed += 1
                        if strategy.gate:
                            c = check_action(a, model, y, F, pred, T, scale, act)
                            check_fits += c["fits"]
                            ok = c["accepted"]
                    if ok:
                        model, fits = act(a, model, T)
                        retrains += fits
                        accepted += 1
                        last_change = T
                    picks.append(a.name if ok else "keep")
            else:  # hindsight oracle: best action on the next decision window
                rows = np.arange(T, min(T + strategy.cadence_days * DAY, end))
                best = None
                for a in catalog:  # `keep` first, so ties go to the cheapest action
                    m_a, fits = act(a, model, T)
                    err = float(np.nanmean(np.abs(y[rows] - m_a.predict(F, rows))))
                    if best is None or err < best[0] - 1e-12:
                        best = (err, a.name, m_a, fits)
                _, name, model, fits = best
                retrains += fits
                picks.append(name)
        rows = np.arange(T, T + DAY)
        pred[rows] = model.predict(F, rows)
        if prune_cache:
            for key in [key for key in cache if key[1] <= T - VAL_WEEKS * WEEK]:
                del cache[key]
    return dict(pred=pred, retrains=retrains, picks=picks, end=end, decisions=decisions, proposed=proposed,
                accepted=accepted, check_fits=check_fits)


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
                    repeat: int = 0) -> list[dict]:
    """All strategies on one series. `s` holds y, F, t_deploy, t_change, base_features and labels."""
    y, F, t_dep, t_chg = s["y"], s["F"], s["t_deploy"], s["t_change"]
    scale = mase_scale(y[:t_dep])
    cache: dict = {}
    runs = {st.name: backtest(y, F, t_dep, st, s["base_features"], params, cache=cache, prune_cache=(i == len(strategies) - 1),
                              advisors=advisors, repeat=repeat)
            for i, st in enumerate(strategies)}
    cache.clear()
    end = runs["nothing"]["end"]
    naive = np.full(len(y), np.nan)
    naive[t_dep:end] = seasonal_naive(y, np.arange(t_dep, end))
    runs["naive"] = dict(pred=naive, retrains=0, picks=[], end=end)
    gated = {st.name for st in strategies if st.policy in ("gated", "advised")}
    labels = {k: v for k, v in s.items() if k not in ("y", "F", "base_features")}
    rows = []
    for name, r in runs.items():
        row = dict(labels, strategy=name, scale=scale, retrains=r["retrains"],
                   mase_all=mase(y, r["pred"], t_dep, end, scale),
                   mase_pre=mase(y, r["pred"], t_dep, t_chg, scale) if t_chg and t_chg > t_dep else np.nan,
                   mase_post=mase(y, r["pred"], t_chg, end, scale) if t_chg else np.nan,
                   picks=" ".join(f"{a}:{c}" for a, c in Counter(r["picks"]).most_common()),
                   **{k: r[k] if name in gated else np.nan for k in ("decisions", "proposed", "accepted", "check_fits")})
        if name != "nothing":
            a = t_chg if t_chg else t_dep
            g, lo, hi = mase_gain_ci(y, runs["nothing"]["pred"], r["pred"], a, end, scale)
            row.update(gain_vs_nothing=g, gain_lo=lo, gain_hi=hi)
        rows.append(row)
    return rows


_WORKER: dict = {}  # advisors of a worker process, created on its first series


def _job(s, strategies=STRATEGIES, advisors: tuple[str, ...] = (), llm_model: str | None = None):
    if advisors and "advisors" not in _WORKER:
        from src.recommendations.llm_client import DEFAULT_MODEL, LLMAdvisor, RuleAdvisor

        made = {"rules": RuleAdvisor, "llm": lambda: LLMAdvisor(llm_model or DEFAULT_MODEL)}
        _WORKER["advisors"] = {k: made[k]() for k in advisors}
    return evaluate_series(s, LGBM_PARAMS, strategies, _WORKER.get("advisors"))


def sort_runs(df: pd.DataFrame) -> pd.DataFrame:
    order = ["naive"] + [s.name for s in STRATEGIES + ADVISED_STRATEGIES]
    df = df.assign(_o=df.strategy.map(order.index))
    return df.sort_values(["source", "scenario", "series", "_o"]).drop(columns="_o").reset_index(drop=True)


def run_backtests(series: list[dict], processes: int | None = None, log=None, strategies=STRATEGIES,
                  advisors: tuple[str, ...] = (), llm_model: str | None = None) -> pd.DataFrame:
    rows = []
    job = partial(_job, strategies=strategies, advisors=advisors, llm_model=llm_model)
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
    for c in ("decisions", "proposed", "accepted", "check_fits"):  # absent in results computed before the acceptance rule
        if c not in r:
            r[c] = np.nan
    r["accept_rate"] = r.accepted / r.proposed.fillna(r.decisions).where(lambda x: x > 0)  # share of proposals applied
    g = r.groupby(["source", "scenario", "strategy"], sort=False).agg(
        series=("series", "nunique"), mase_pre=("mase_pre", "mean"), mase_post=("mase_post", "mean"),
        regret=("regret", "mean"), retrains=("retrains", "mean"), gain_vs_nothing=("gain_vs_nothing", "mean"),
        share_gain_significant=("gain_significant", "mean"), accept_rate=("accept_rate", "mean"),
        check_fits=("check_fits", "mean")).reset_index()
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
