"""Detector alarms for the strategies that review the model on alarm instead of on a calendar.

Alarms come from a CSV (`data/alerts/alerts.csv`): one row per alarm with `id` (source/series/scenario as in
the manifest of `src.data`), `method` and `timestamp` (or `index`, the position in the series). `read_alerts`
turns it into alarm positions per series and detector. The detector of stream A runs on the same series:

* residual_conformal: `ResidualConformalDriftDetector` of `src.detector` with the settings of its multi-series
  mode for a daily period. Its residual of an hour comes from an STL fit on a block that ends up to 23 hours
  later, so an alarm is dated at the end of its block, the first hour at which it is known.

Two stand-in detectors run next to it. Both use only the past: the residual of each hour is the value minus the
median of the same hour of the last `PROFILE_WEEKS` weeks, then

* profile_conformal: the deviation of the residual from the median of the last `CALIB` residuals is compared
  with a threshold, the 95% quantile of such deviations over the same window times a multiplier; an alarm
  needs two consecutive violations and is followed by a one-day cooldown (the rule of the stream A detector);
* adwin: ADWIN (Bifet and Gavalda, 2007) on the same residuals, checked at every point.

The multipliers and the ADWIN delta are calibrated per source on a control period without changes (synthetic:
the control series, scenario "none"; seismic noise: the weeks between deployment and the first measures; ENTSO-E:
the year 2019) to a target false alarm rate, one alarm per `TARGET_DAYS` days, so the methods are compared at the
same false alarm rate. Settings calibrated on one source do not carry over to another (checked on 02.11.2026:
the synthetic settings gave 6 times the target rate on the national load), hence the calibration per source.

    python -m src.recommendations.alarms        # calibrate, run all on all series, write data/alerts/alerts.csv
                                                # (about 18 minutes on 8 cores: the STL fits of the stream A detector)
"""
from __future__ import annotations

import argparse
import hashlib
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.frames import series_id, series_table
from src.recommendations.forecast import DAY, WEEK

ALERTS_DIR = Path("data/alerts")
PROFILE_WEEKS = 4
CALIB = 2 * WEEK
TARGET_DAYS = 14
MULT_GRID = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 5.0)
DELTA_GRID = (0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6)
RESIDUAL_GRID = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)


def profile_residuals(y: np.ndarray, weeks: int = PROFILE_WEEKS) -> np.ndarray:
    """Value minus the median of the same hour of the week over the last `weeks` weeks (NaN before that)."""
    past = np.full((weeks, len(y)), np.nan)
    for k in range(1, weeks + 1):
        past[k - 1, k * WEEK:] = y[: len(y) - k * WEEK]
    return y - np.median(past, axis=0)


def conformal_scores(r: np.ndarray, mult: float, calib: int = CALIB, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Score and threshold of the conformal rule at every position of the residuals `r`: the deviation of the residual
    from the median of the previous `calib` residuals, and `mult` times the (1 - alpha) quantile of the previous
    deviations. Both use only earlier positions, so the values at position i depend on r[i - 2 calib .. i] only."""
    s = pd.Series(r)
    med = s.rolling(calib, min_periods=calib // 2).median().shift(1)
    dev = (s - med).abs()
    thr = dev.rolling(calib, min_periods=calib // 2).quantile(1 - alpha).shift(1) * mult
    return dev.to_numpy(), thr.to_numpy()


def conformal_alarms(r: np.ndarray, start: int, mult: float, calib: int = CALIB, alpha: float = 0.05,
                     consecutive: int = 2, cooldown: int = DAY) -> pd.DataFrame:
    """Alarms of the conformal rule on residuals `r` from position `start` on: deviation of the residual from
    the median of the previous `calib` residuals against `mult` times the (1 - alpha) quantile of the previous
    deviations. Returns the positions with their score and threshold."""
    score, thr = conformal_scores(r, mult, calib, alpha)
    viol = (score > thr) & np.isfinite(thr)
    out, run, last = [], 0, -cooldown
    for i in range(start, len(r)):
        run = run + 1 if viol[i] else 0
        if run >= consecutive and i - last >= cooldown:
            out.append((i, float(score[i]), float(thr[i])))
            last = i
    return pd.DataFrame(out, columns=["index", "score", "threshold"])


def adwin_alarms(r: np.ndarray, start: int, delta: float) -> pd.DataFrame:
    """Alarms of ADWIN (River, checked at every point) on the residuals, from position `start` on."""
    from river import drift

    det = drift.ADWIN(delta=delta, clock=1)
    out = []
    first = int(np.argmax(np.isfinite(r)))
    for i in range(first, len(r)):
        det.update(float(r[i]))
        if det.drift_detected and i >= start:
            out.append((i, float(det.estimation), np.nan))
    return pd.DataFrame(out, columns=["index", "score", "threshold"])


def residual_detector(mult: float = 1.0):
    """The detector of stream A for hourly series with a daily period, with the settings of its multi-series mode."""
    from src.detector.residual_detector import ResidualConformalDriftDetector
    from src.detector.run_residual_detector_evaluation import get_detector_params

    params = get_detector_params("", DAY, 10 ** 6, mode="multi")
    params.pop("tolerance_window")
    params["sensitivity_multiplier"] = mult
    return ResidualConformalDriftDetector(seasonality_period=DAY, **params)


def residual_scores(window: np.ndarray, local: np.ndarray | None, value: float, alpha: float) -> tuple[float, float, float, float]:
    """The numbers the stream A detector compares at one position, computed as in its `fit_predict`: the deviation
    of the residual `value` from the median of the calibration `window`, the conformal quantile of the deviations
    inside the window (its threshold at multiplier 1), the shift of the mean of the `local` window from that median,
    and the interquartile range of the local window over the one of the calibration window. `local` is None while
    there are too few residuals for it; the last two rules are then off."""
    med = np.median(window)
    dev = np.abs(window - med)
    q = int(np.clip(int(np.ceil((1.0 - alpha) * (len(dev) + 1))) - 1, 0, len(dev) - 1))
    base = np.sort(dev)[q]
    score = np.abs(value - med)
    if local is None:
        return float(score), float(base), 0.0, 0.0
    shift = np.abs(np.mean(local) - med)
    local_iqr = np.percentile(local, 75) - np.percentile(local, 25)
    baseline_iqr = np.percentile(window, 75) - np.percentile(window, 25)
    return float(score), float(base), float(shift), float(local_iqr / (baseline_iqr + 1e-8))


def residual_violation(score, base, shift, iqr, mult: float, var_window: int):
    """The three rules of `fit_predict` at multiplier `mult`, with its operations in its order (floats or arrays)."""
    threshold = base * mult
    return (score > threshold) | (shift > (threshold / np.sqrt(var_window)) * 1.35) | (iqr > (1.8 * mult))


def block_end(i: int, first: int, step: int) -> int:
    """The last position of the STL block of the stream A detector that holds position `i`: the first `first`
    positions come from one warm-up fit, the later ones from fits on blocks of `step` positions."""
    return first - 1 if i < first else first + ((i - first) // step + 1) * step - 1


class ResidualStream:
    """The stream A detector fed one value at a time. It repeats `fit_predict` of the detector block by block: the
    residuals of a block of one period come from an STL fit on the history up to the end of that block, then every
    position of the block is checked. An alarm is therefore returned with the last value of its block (`index`),
    with the position that raised it in `detected`. On a whole series it gives the alarms of `fit_predict` that lie
    in complete blocks, with the same scores and thresholds (checked in the tests)."""

    def __init__(self, mult: float, start: int = 0):
        d = residual_detector(mult)
        self.d, self.mult, self.start = d, mult, start
        self.first = max(2 * d.period, d.cal_window)                 # the warm-up decomposition of the detector
        self.history = max(6 * d.period, 3 * d.cal_window)           # the history of its later decompositions
        if d.cooldown_steps < d.causal_step:
            raise ValueError("a block may hold two alarms: the cooldown is shorter than the block")
        self.y: list[float] = []
        self.r: list[float] = []
        self.checked = d.cal_window  # the next position to check
        self.run, self.last = 0, -d.cooldown_steps

    def _check(self, i: int) -> dict | None:
        d, r = self.d, self.r
        local = np.asarray(r[i - d.var_window + 1: i + 1]) if i >= d.cal_window + d.var_window else None
        score, base, shift, iqr = residual_scores(np.asarray(r[i - d.cal_window: i]), local, r[i], d.alpha)
        self.run = self.run + 1 if residual_violation(score, base, shift, iqr, self.mult, d.var_window) else 0
        if self.run >= d.min_consecutive and i - self.last >= d.cooldown_steps:
            self.last, self.run = i, 0
            return dict(detected=i, score=score, threshold=base * self.mult)
        return None

    def update(self, value: float) -> dict | None:
        from statsmodels.tsa.seasonal import STL

        d = self.d
        self.y.append(float(value))
        n = len(self.y)
        if n < self.first or (n - self.first) % d.causal_step:
            return None
        if n == self.first:
            self.r = list(STL(np.asarray(self.y), period=d.period, robust=d.robust).fit().resid)
        else:
            segment = np.asarray(self.y[max(0, n - self.history):])
            self.r.extend(STL(segment, period=d.period, robust=d.robust).fit().resid[-d.causal_step:])
        found = [a for a in (self._check(i) for i in range(self.checked, n)) if a is not None]
        self.checked = n
        if found and n - 1 >= self.start:
            return dict(index=n - 1, **found[0])
        return None


_SCORES: dict[str, np.ndarray] = {}  # scores of the series seen last: the calibration asks for them once per grid value


def _series_key(y: np.ndarray) -> str:
    return hashlib.blake2b(np.asarray(y, dtype=float).tobytes(), digest_size=16).hexdigest()


def warm_residual_scores(ys: list[np.ndarray], processes: int | None = None) -> None:
    """Score tables of the stream A detector for many series, computed in parallel processes and kept for the
    calls that follow: the STL fits take most of the time of a run (about 1 ms per hour of data on one core)."""
    todo = {_series_key(y): np.asarray(y, dtype=float) for y in ys}
    todo = sorted((y for k, y in todo.items() if k not in _SCORES), key=len, reverse=True)
    if not todo:
        return
    with Pool(processes) as pool:
        tables = pool.map(residual_score_table, todo, chunksize=1)
    for y, table in zip(todo, tables):
        _SCORES[_series_key(y)] = table


def residual_score_table(y: np.ndarray) -> np.ndarray:
    """`residual_scores` at every position of `y`, one row per position (NaN inside the first calibration window),
    from the causal STL residuals of the stream A detector."""
    y = np.asarray(y, dtype=float)
    key = _series_key(y)
    if key not in _SCORES:
        d = residual_detector()
        r = d._compute_causal_residuals(y)
        out = np.full((len(y), 4), np.nan)
        for i in range(d.cal_window, len(y)):
            local = r[i - d.var_window + 1: i + 1] if i >= d.cal_window + d.var_window else None
            out[i] = residual_scores(r[i - d.cal_window: i], local, r[i], d.alpha)
        while len(_SCORES) >= 512:
            _SCORES.pop(next(iter(_SCORES)))
        _SCORES[key] = out
    return _SCORES[key]


def residual_conformal_alarms(y: np.ndarray, start: int, mult: float) -> pd.DataFrame:
    """Alarms of the stream A detector on the series `y` at multiplier `mult`, each dated at the end of its STL
    block (see `ResidualStream`), with the score and the threshold of `fit_predict`; alarms known before `start`
    or only after the end of the series are dropped."""
    d = residual_detector(mult)
    if len(y) < d.period + d.cal_window:  # shorter than the detector accepts: nothing can be known yet
        return pd.DataFrame(columns=["index", "score", "threshold"])
    score, base, shift, iqr = residual_score_table(y).T
    with np.errstate(invalid="ignore"):
        viol = residual_violation(score, base, shift, iqr, mult, d.var_window)
    first = max(2 * d.period, d.cal_window)
    out, run, last = [], 0, -d.cooldown_steps
    for i in range(d.cal_window, len(y)):
        run = run + 1 if viol[i] else 0
        if run >= d.min_consecutive and i - last >= d.cooldown_steps:
            last, run = i, 0
            known = block_end(i, first, d.causal_step)
            if start <= known < len(y) and not (out and out[-1][0] == known):  # one alarm per block
                out.append((known, float(score[i]), float(base[i] * mult)))
    return pd.DataFrame(out, columns=["index", "score", "threshold"])


DETECTORS = {"residual_conformal": (residual_conformal_alarms, "mult", RESIDUAL_GRID),
             "profile_conformal": (conformal_alarms, "mult", MULT_GRID), "adwin": (adwin_alarms, "delta", DELTA_GRID)}
ON_VALUES = ("residual_conformal",)  # detectors that take the series itself; the others take the profile residuals


def alarms_in(method: str, y: np.ndarray, start: int, value: float) -> pd.DataFrame:
    """Alarms of one detector on `y` (the stand-ins: on its profile residuals) from position `start` on."""
    return DETECTORS[method][0](y if method in ON_VALUES else profile_residuals(y), start, value)


def alarms_of(method: str, s: dict, value: float) -> pd.DataFrame:
    return alarms_in(method, s["y"], s["t_deploy"], value)


class OnlineDetector:
    """A detector fed one value at a time, for the monitoring service. It raises the same alarms as
    `alarms_in(method, y, start, value)` on the whole series (checked in the tests): the profile residual of a new
    value needs the same hour of the last `PROFILE_WEEKS` weeks, the conformal rule the last 2 `CALIB` residuals,
    ADWIN keeps its own state, and the stream A detector runs as `ResidualStream`. Values before `start` only
    build up that history."""

    def __init__(self, method: str, value: float, start: int, weeks: int = PROFILE_WEEKS, calib: int = CALIB,
                 consecutive: int = 2, cooldown: int = DAY):
        if method not in DETECTORS:
            raise ValueError(f"unknown detector {method}; known: {', '.join(DETECTORS)}")
        self.method, self.value, self.start, self.weeks, self.calib = method, value, start, weeks, calib
        self.consecutive, self.cooldown = consecutive, cooldown
        self.y: list[float] = []
        self.r: list[float] = []
        self.run, self.last = 0, -cooldown  # state of the conformal rule (see `conformal_alarms`)
        self.adwin = None
        self.stream = ResidualStream(value, start) if method == "residual_conformal" else None

    def update(self, value: float) -> dict | None:
        """Add the next value of the series; returns the alarm at its position (index, score, threshold) or None."""
        if self.stream is not None:
            return self.stream.update(value)
        i = len(self.y)
        self.y.append(float(value))
        past = [self.y[i - k * WEEK] for k in range(1, self.weeks + 1)] if i >= self.weeks * WEEK else [np.nan]
        r = self.y[i] - float(np.median(past))
        self.r.append(r)
        if self.method == "adwin":
            if self.adwin is None:
                if not np.isfinite(r):
                    return None
                from river import drift

                self.adwin = drift.ADWIN(delta=self.value, clock=1)
            self.adwin.update(r)
            if self.adwin.drift_detected and i >= self.start:
                return dict(index=i, score=float(self.adwin.estimation), threshold=np.nan)
            return None
        if i < self.start:
            return None
        tail = np.asarray(self.r[max(0, i - 2 * self.calib):], dtype=float)
        score, thr = (float(v[-1]) for v in conformal_scores(tail, self.value, self.calib))
        self.run = self.run + 1 if (score > thr and np.isfinite(thr)) else 0
        if self.run >= self.consecutive and i - self.last >= self.cooldown:
            self.last = i
            return dict(index=i, score=score, threshold=thr)
        return None


def detector_settings(path: str | Path = ALERTS_DIR / "detectors.csv") -> dict[str, dict[str, float]]:
    """Calibrated setting of every stand-in detector per source, as written by `main` (source -> method -> value)."""
    df = pd.read_csv(path)
    return {src: dict(zip(g["method"], g["value"].astype(float))) for src, g in df.groupby("source")}


def control_windows(series: list[dict], entsoe_path: str | Path | None = None) -> dict[str, list[tuple]]:
    """Control periods without changes per source, as (y, start, stop): the synthetic control series from
    deployment to the end, the seismic series from deployment to the first measures, and for ENTSO-E the
    year 2019 of the same countries (loaded from `entsoe_path`; the test period is 2020)."""
    out: dict[str, list[tuple]] = {}
    for s in series:
        if s["source"] == "synthetic" and s["scenario"] == "none":
            out.setdefault("synthetic", []).append((s["y"], s["t_deploy"], len(s["y"])))
        elif s["source"] == "seismic":
            out.setdefault("seismic", []).append((s["y"], s["t_deploy"], s["t_change"]))
    if entsoe_path is not None and any(s["source"] == "entsoe" for s in series):
        from src.recommendations.data_sources import entsoe_series

        for s in entsoe_series(str(entsoe_path), deploy="2019-01-07", end="2019-12-01 23:00"):
            out.setdefault("entsoe", []).append((s["y"], s["t_deploy"], len(s["y"])))
    return out


def false_alarm_rate(method: str, windows: list[tuple], value: float) -> float:
    """Alarms per hour inside the control windows (y, start, stop)."""
    hours = sum(stop - start for _, start, stop in windows)
    alarms = 0
    for y, start, stop in windows:
        idx = alarms_in(method, y, start, value)["index"]
        alarms += int(((idx >= start) & (idx < stop)).sum())
    return alarms / hours


def calibrate(method: str, windows: list[tuple], target_days: int = TARGET_DAYS) -> tuple[float, float]:
    """The first grid value (from sensitive to strict) whose false alarm rate on the control windows is at most
    one alarm per `target_days` days; the strictest value if none reaches it. Returns (value, rate)."""
    _, _, grid = DETECTORS[method]
    rates = [(v, false_alarm_rate(method, windows, v)) for v in grid]
    for v, rate in rates:
        if rate <= 1 / (target_days * DAY):
            return v, rate
    return rates[-1]


def detect_all(series: list[dict], params: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Alarm table of every detector on every series; `params` maps source -> method -> calibrated value."""
    rows = []
    for s in series:
        stamps = pd.date_range(s["start"], periods=len(s["y"]), freq="1h")
        for method, value in params[s["source"]].items():
            a = alarms_of(method, s, value)
            a.insert(0, "timestamp", stamps[a["index"].to_numpy(dtype=int)])
            a.insert(0, "method", method)
            for col, val in (("scenario", s["scenario"]), ("series", s["series"]), ("source", s["source"]), ("id", series_id(s))):
                a.insert(0, col, val)
            if len(a):
                rows.append(a)
    return pd.concat(rows, ignore_index=True)


def _utc(values) -> np.ndarray:
    """Timestamps (strings or Timestamp objects, tz-aware or naive, mixed) as UTC datetime64; naive ones count as UTC,
    which is consistent as long as `start` of the series and its alarms are written the same way."""
    s = pd.Series(list(values))
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")
    ok = s.notna()
    out[ok] = pd.to_datetime(s[ok].astype(str), utc=True, format="mixed")
    return out.dt.tz_convert(None).to_numpy()  # datetime64 in UTC, so differences are plain timedeltas


def read_alerts(path: str | Path, manifest: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Alarm positions per series id and detector from an alerts CSV. A row needs `id`, `method` and either
    `index` or `timestamp` (matched to the series through `start` of the manifest); rows outside the series
    or with unknown ids are dropped."""
    df = pd.read_csv(path)
    info = manifest.set_index("id")[["start", "n"]]
    df = df[df["id"].isin(info.index)].copy()
    start = _utc(info.loc[df["id"], "start"])
    if "index" not in df:
        df["index"] = np.nan
    if "timestamp" in df:
        stamps = _utc(df["timestamp"])
        from_stamp = np.round((stamps - start) / np.timedelta64(1, "h"))
        df["index"] = df["index"].where(df["index"].notna(), from_stamp)
    df = df[df["index"].notna()]
    n = info.loc[df["id"], "n"].to_numpy()
    df = df[(df["index"] >= 0) & (df["index"] < n)]
    out: dict[str, dict[str, np.ndarray]] = {}
    for (sid, method), g in df.groupby(["id", "method"], sort=True):
        out.setdefault(sid, {})[method] = np.unique(g["index"].to_numpy(dtype=int))
    return out


def main():
    from src.data.entsoe import ENTSOE_FILE
    from src.data.seismic_noise import SEISMIC_DIR
    from src.recommendations.data_sources import entsoe_series, seismic_series
    from src.recommendations.synthetic import fit_models, synthetic_series

    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--out", default=str(ALERTS_DIR))
    p.add_argument("--clients", type=int, default=20)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--no-real", action="store_true")
    p.add_argument("--target-days", type=int, default=TARGET_DAYS, help="calibration target: one false alarm per this many days")
    p.add_argument("--processes", type=int, default=None, help="processes for the STL fits of the stream A detector (default: all cores)")
    args = p.parse_args()

    t = time.time()
    series = [] if args.no_real else entsoe_series(str(ENTSOE_FILE)) + seismic_series(str(SEISMIC_DIR))
    series += synthetic_series(fit_models(args.clients), list(range(args.seeds)))
    windows = control_windows(series, None if args.no_real else ENTSOE_FILE)
    warm_residual_scores([s["y"] for s in series] + [y for w in windows.values() for y, _, _ in w], args.processes)
    params, rows = {}, []
    for source, w in windows.items():
        params[source] = {}
        hours = sum(stop - start for _, start, stop in w)
        for method, (_, name, _) in DETECTORS.items():
            value, rate = calibrate(method, w, args.target_days)
            params[source][method] = value
            rows.append(dict(source=source, method=method, param=name, value=value, far_control_per_1000=round(rate * 1000, 3),
                             target_per_1000=round(1000 / (args.target_days * DAY), 3), control_windows=len(w), control_hours=hours))
            print(f"{source} / {method}: {name} = {value}, {rate * 1000:.2f} false alarms per 1000 h on {len(w)} control windows ({hours} h)")
    alerts = detect_all(series, params)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(out / "alerts.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "detectors.csv", index=False)
    table = series_table(series)
    hours = table.set_index("id")["n"] - table.set_index("id")["t_deploy"]
    per = alerts.groupby(["source", "scenario", "method"]).size().rename("alarms").reset_index()
    per["per_1000h"] = [1000 * a / hours[table.id[(table.source == s) & (table.scenario == c)]].sum()
                        for s, c, a in zip(per.source, per.scenario, per.alarms)]
    print(per.round(2).to_string(index=False))
    print(f"{len(alerts)} alarms on {len(series)} series written to {out} in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
