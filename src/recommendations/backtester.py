"""Stream B (RQ3): backtest of adaptation strategies, reference strategies and the hindsight oracles.

A deployed LightGBM model forecasts every day the next 24 hours. A strategy decides when the model is
changed and how:

* nothing       - the model trained at deployment is never changed;
* always        - retrain on all history every day;
* scheduled     - retrain on all history once a week;
* oracle_weekly - once a week, apply the catalog action with the lowest error on the following week,
                  chosen in hindsight: the ceiling for any weekly policy (it is never worse than
                  `scheduled`), and the reference for LLM recommendations that are taken at a few points;
* oracle        - the same every day with a one-day lookahead: the ideal ceiling (never worse than `always`).
The oracles use future data, so they cannot be deployed; they only bound what any policy can reach.

Quality is MASE (scale: weekly seasonal naive on the training data), cost is the number of retrainings.
`mase_gain_ci` gives the gain Delta = MASE before - MASE after with a moving block bootstrap interval over
days; it is the building block of the acceptance rule of the backtest (lower bound > delta).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src.recommendations.catalog import ACTIONS, CATALOG, apply_action
from src.recommendations.forecast import DAY, LGBM_PARAMS, Model, mase_scale, seasonal_naive


@dataclass(frozen=True)
class Strategy:
    name: str
    title_uk: str
    cadence_days: int | None  # how often a decision is taken; None = never
    policy: str  # fixed | oracle
    action: str | None = None


STRATEGIES = (
    Strategy("nothing", "нічого не робити", None, "fixed"),
    Strategy("always", "завжди перенавчати (щодня)", 1, "fixed", "retrain_full"),
    Strategy("scheduled", "перенавчання за розкладом (щотижня)", 7, "fixed", "retrain_full"),
    Strategy("oracle_weekly", "оракул щотижня: найкраща дія каталогу за фактом на наступному тижні", 7, "oracle"),
    Strategy("oracle", "ідеальний оракул: найкраща дія каталогу за фактом щодня", 1, "oracle"),
)


def backtest(y: np.ndarray, F: dict[str, np.ndarray], t_deploy: int, strategy: Strategy, base_features: tuple[str, ...],
             params: dict | None = None, catalog=CATALOG, cache: dict | None = None, prune_cache: bool = False) -> dict:
    """Rolling day-ahead forecasts from `t_deploy` to the end of the series under one strategy.

    Retraining actions do not depend on the current model, so their fits are shared through `cache`
    (keyed by action and origin) between strategies of the same series. With `prune_cache` (last strategy
    of a series) fits from past origins are dropped as the run moves on, which bounds memory."""
    n = len(y)
    origins = list(range(t_deploy, n - DAY + 1, DAY))
    end = origins[-1] + DAY
    cache = {} if cache is None else cache

    def act(a, model, T):
        if a.kind != "retrain":
            return apply_action(a, model, F, y, T, base_features, params)
        key = (a.name, T)
        if key not in cache:
            cache[key] = apply_action(a, model, F, y, T, base_features, params)[0]
        return cache[key], 1

    model, _ = act(ACTIONS["retrain_full"], Model(base_features), t_deploy)
    pred = np.full(n, np.nan)
    retrains, picks = 0, []
    for k, T in enumerate(origins):
        if strategy.cadence_days and k > 0 and k % strategy.cadence_days == 0:
            if strategy.policy == "fixed":
                model, fits = act(ACTIONS[strategy.action], model, T)
                retrains += fits
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
            for key in [key for key in cache if key[1] <= T]:
                del cache[key]
    return dict(pred=pred, retrains=retrains, picks=picks, end=end)


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


def evaluate_series(s: dict, params: dict | None = None, strategies=STRATEGIES) -> list[dict]:
    """All strategies on one series. `s` holds y, F, t_deploy, t_change, base_features and labels."""
    y, F, t_dep, t_chg = s["y"], s["F"], s["t_deploy"], s["t_change"]
    scale = mase_scale(y[:t_dep])
    cache: dict = {}
    runs = {st.name: backtest(y, F, t_dep, st, s["base_features"], params, cache=cache, prune_cache=(i == len(strategies) - 1))
            for i, st in enumerate(strategies)}
    cache.clear()
    end = runs["nothing"]["end"]
    naive = np.full(len(y), np.nan)
    naive[t_dep:end] = seasonal_naive(y, np.arange(t_dep, end))
    runs["naive"] = dict(pred=naive, retrains=0, picks=[], end=end)
    labels = {k: v for k, v in s.items() if k not in ("y", "F", "base_features")}
    rows = []
    for name, r in runs.items():
        row = dict(labels, strategy=name, scale=scale, retrains=r["retrains"],
                   mase_all=mase(y, r["pred"], t_dep, end, scale),
                   mase_pre=mase(y, r["pred"], t_dep, t_chg, scale) if t_chg and t_chg > t_dep else np.nan,
                   mase_post=mase(y, r["pred"], t_chg, end, scale) if t_chg else np.nan,
                   picks=" ".join(f"{a}:{c}" for a, c in Counter(r["picks"]).most_common()))
        if name != "nothing":
            a = t_chg if t_chg else t_dep
            g, lo, hi = mase_gain_ci(y, runs["nothing"]["pred"], r["pred"], a, end, scale)
            row.update(gain_vs_nothing=g, gain_lo=lo, gain_hi=hi)
        rows.append(row)
    return rows


def _job(s):
    return evaluate_series(s, LGBM_PARAMS)


def sort_runs(df: pd.DataFrame) -> pd.DataFrame:
    order = ["naive"] + [s.name for s in STRATEGIES]
    df = df.assign(_o=df.strategy.map(order.index))
    return df.sort_values(["source", "scenario", "series", "_o"]).drop(columns="_o").reset_index(drop=True)


def run_backtests(series: list[dict], processes: int | None = None, log=None) -> pd.DataFrame:
    rows = []
    with Pool(processes) as pool:
        for i, rs in enumerate(pool.imap_unordered(_job, series, chunksize=1), 1):
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
    g = r.groupby(["source", "scenario", "strategy"], sort=False).agg(
        series=("series", "nunique"), mase_pre=("mase_pre", "mean"), mase_post=("mase_post", "mean"),
        regret=("regret", "mean"), retrains=("retrains", "mean"), gain_vs_nothing=("gain_vs_nothing", "mean"),
        share_gain_significant=("gain_significant", "mean")).reset_index()
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
