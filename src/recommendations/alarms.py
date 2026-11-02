"""Detector alarms for the strategies that review the model on alarm instead of on a calendar.

Alarms come from a CSV (`data/alerts/alerts.csv`): one row per alarm with `id` (source/series/scenario as in
the manifest of `src.data`), `method` and `timestamp` (or `index`, the position in the series). `read_alerts`
turns it into alarm positions per series and detector. Until the detectors of stream A deliver that file, two
stand-in detectors run on the same series. Both use only the past: the residual of each hour is the value
minus the median of the same hour of the last `PROFILE_WEEKS` weeks, then

* profile_conformal: the deviation of the residual from the median of the last `CALIB` residuals is compared
  with a threshold, the 95% quantile of such deviations over the same window times a multiplier; an alarm
  needs two consecutive violations and is followed by a one-day cooldown (the rule of the stream A detector);
* adwin: ADWIN (Bifet and Gavalda, 2007) on the same residuals, checked at every point.

The multiplier and the ADWIN delta are calibrated on the control series (scenario "none") to a target false
alarm rate, one alarm per `TARGET_DAYS` days, so the methods are compared at the same false alarm rate.

    python -m src.recommendations.alarms        # calibrate, run both on all series, write data/alerts/alerts.csv
"""
from __future__ import annotations

import argparse
import time
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


def profile_residuals(y: np.ndarray, weeks: int = PROFILE_WEEKS) -> np.ndarray:
    """Value minus the median of the same hour of the week over the last `weeks` weeks (NaN before that)."""
    past = np.full((weeks, len(y)), np.nan)
    for k in range(1, weeks + 1):
        past[k - 1, k * WEEK:] = y[: len(y) - k * WEEK]
    return y - np.median(past, axis=0)


def conformal_alarms(r: np.ndarray, start: int, mult: float, calib: int = CALIB, alpha: float = 0.05,
                     consecutive: int = 2, cooldown: int = DAY) -> pd.DataFrame:
    """Alarms of the conformal rule on residuals `r` from position `start` on: deviation of the residual from
    the median of the previous `calib` residuals against `mult` times the (1 - alpha) quantile of the previous
    deviations. Returns the positions with their score and threshold."""
    s = pd.Series(r)
    med = s.rolling(calib, min_periods=calib // 2).median().shift(1)
    dev = (s - med).abs()
    thr = dev.rolling(calib, min_periods=calib // 2).quantile(1 - alpha).shift(1) * mult
    score, thr = dev.to_numpy(), thr.to_numpy()
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


DETECTORS = {"profile_conformal": (conformal_alarms, "mult", MULT_GRID), "adwin": (adwin_alarms, "delta", DELTA_GRID)}


def alarms_of(method: str, s: dict, value: float) -> pd.DataFrame:
    fn = DETECTORS[method][0]
    return fn(profile_residuals(s["y"]), s["t_deploy"], value)


def false_alarm_rate(method: str, controls: list[dict], value: float) -> float:
    """Alarms per hour on series without changes, from deployment to the end."""
    hours = sum(len(s["y"]) - s["t_deploy"] for s in controls)
    return sum(len(alarms_of(method, s, value)) for s in controls) / hours


def calibrate(method: str, controls: list[dict], target_days: int = TARGET_DAYS) -> tuple[float, float]:
    """The first grid value (from sensitive to strict) whose false alarm rate on the control series is at most
    one alarm per `target_days` days; the strictest value if none reaches it. Returns (value, rate)."""
    _, _, grid = DETECTORS[method]
    rates = [(v, false_alarm_rate(method, controls, v)) for v in grid]
    for v, rate in rates:
        if rate <= 1 / (target_days * DAY):
            return v, rate
    return rates[-1]


def detect_all(series: list[dict], params: dict[str, float]) -> pd.DataFrame:
    """Alarm table of every detector in `params` (method -> calibrated value) on every series."""
    rows = []
    for s in series:
        stamps = pd.date_range(s["start"], periods=len(s["y"]), freq="1h")
        for method, value in params.items():
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
    args = p.parse_args()

    t = time.time()
    series = [] if args.no_real else entsoe_series(str(ENTSOE_FILE)) + seismic_series(str(SEISMIC_DIR))
    series += synthetic_series(fit_models(args.clients), list(range(args.seeds)))
    controls = [s for s in series if s["scenario"] == "none"]
    params, rows = {}, []
    for method, (_, name, _) in DETECTORS.items():
        value, rate = calibrate(method, controls, args.target_days)
        params[method] = value
        rows.append(dict(method=method, param=name, value=value, far_control_per_1000=round(rate * 1000, 3),
                         target_per_1000=round(1000 / (args.target_days * DAY), 3), control_series=len(controls)))
        print(f"{method}: {name} = {value}, {rate * 1000:.2f} false alarms per 1000 h on {len(controls)} control series")
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
