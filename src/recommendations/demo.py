"""The demo on one series: the series, the figures and the tables of `notebooks/demo_one_series.ipynb`, and the
start of the same series in the monitoring service.

The series (`data/demo/demo_seasonal_shift.csv`, a standard frame: `timestamp`, `value`, one row per hour) is the
synthetic control series of the UCI client MT_019 (seed 0) of the experiments, 24 weeks from 2 September 2013. The
forecasting model goes to work on 11 November; from 2 December on the weekly profile of the series is 2.5 times
larger, with the same level: a change of the seasonal amplitude that is easy to see.

The answers of the LLM advisor on this series are stored in `data/demo/llm_answers`, so the demo needs no API key:
`LLMAdvisor` reads a stored answer when the request is the same and asks the model only when it is not.

    python -m src.recommendations.demo series    # write the CSV again (needs python -m src.data.export)
    python -m src.recommendations.demo session   # upload the series to the running service and start a session
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.api import uploads
from src.recommendations.forecast import DAY, WEEK
from src.recommendations.plots import MUTED, STYLE

ROOT = Path(__file__).resolve().parents[2]
DEMO_CSV = ROOT / "data" / "demo" / "demo_seasonal_shift.csv"
LLM_ANSWERS = ROOT / "data" / "demo" / "llm_answers"
LLM_MODEL = "openrouter/anthropic/claude-sonnet-5.5"
NAME, DEPLOY_AT, CHANGE_AT = "demo_seasonal_shift", "2013-11-11", "2013-12-02"
SOURCE_SERIES, FACTOR = "synthetic/MT_019/s0/none", 2.5
ACTUAL, MODEL, SYSTEM, ALARM = "#b8b7b1", "#2a78d6", "#1baf7a", "#eb6834"


def make_series(factor: float = FACTOR) -> pd.DataFrame:
    """The demo series from the exported control series: from the change on, the mean weekly profile of the history
    (centred) is added `factor - 1` times, so the weekly swing becomes `factor` times larger."""
    from src.recommendations.data_sources import read_series
    from src.recommendations.synthetic import SYN_CHANGE_WEEK, SYN_START

    s = read_series(SOURCE_SERIES)
    y, t_deploy, t_change = s["y"].copy(), s["t_deploy"], SYN_CHANGE_WEEK * WEEK
    hour_of_week = np.arange(len(y)) % WEEK
    profile = np.array([y[:t_deploy][hour_of_week[:t_deploy] == h].mean() for h in range(WEEK)])
    profile -= profile.mean()
    y[t_change:] += (factor - 1) * profile[hour_of_week[t_change:]]
    return pd.DataFrame({"timestamp": pd.date_range(SYN_START, periods=len(y), freq="1h"), "value": y})


def load_series(path: str | Path = DEMO_CSV, name: str = NAME, deploy_at: str = DEPLOY_AT,
                change_at: str | None = CHANGE_AT) -> dict:
    """A series from a CSV in the standard frame, checked and prepared by the code behind `POST /series/csv`."""
    start, values = uploads.read_csv(Path(path).read_bytes())
    return uploads.build(name, start, values, deploy_at, change_at)


def stamps(s: dict) -> pd.DatetimeIndex:
    return pd.date_range(s["start"], periods=len(s["y"]), freq="1h")


def _lines(ax, s: dict, t: pd.DatetimeIndex, lo: int, hi: int, labels: bool = True) -> None:
    for name, pos in (("deployment", s["t_deploy"]), ("change", s["t_change"])):
        if pos is not None and lo <= pos < hi:
            ax.axvline(t[pos], color=MUTED, lw=1, ls=":" if name == "change" else "--")
            if labels:
                ax.annotate(name, (t[pos], 1.0), xycoords=("data", "axes fraction"), xytext=(3, -10),
                            textcoords="offset points", fontsize=8.5, color=MUTED)


def _dates(*axes) -> None:
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))


def _legend(ax, ncol: int = 4) -> None:
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, fontsize=8.5, borderaxespad=0.2)


def plot_series(s: dict, days: int = 14):
    """The whole series (top) and the days around the change (bottom)."""
    y, t, c = s["y"], stamps(s), s["t_change"]
    with plt.rc_context(STYLE):
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 5.6))
        top.plot(t, y, color=MODEL, lw=0.7)
        top.axvspan(t[0], t[s["t_deploy"]], color=ACTUAL, alpha=0.25, lw=0)
        top.annotate("history: training of the model, calibration of the detector", (t[0], 1.0),
                     xycoords=("data", "axes fraction"), xytext=(4, -10), textcoords="offset points", fontsize=8.5, color=MUTED)
        _lines(top, s, t, 0, len(y))
        top.set_title("Demo series: hourly values, 24 weeks", loc="left")
        lo, hi = c - days * DAY, c + days * DAY
        bottom.plot(t[lo:hi], y[lo:hi], color=MODEL, lw=1.2)
        _lines(bottom, s, t, lo, hi)
        bottom.set_title(f"{days} days before and after the change", loc="left")
        _dates(top, bottom)
        fig.tight_layout()
    plt.close(fig)  # the notebook shows the returned figure once
    return fig


def plot_forecast(s: dict, pred: np.ndarray, days_before: int = 7, days_after: int = 14, label: str = "forecast of the deployed model"):
    """The values and the day-ahead forecasts around the change."""
    y, t, c = s["y"], stamps(s), s["t_change"]
    lo, hi = c - days_before * DAY, c + days_after * DAY
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 3.4))
        ax.plot(t[lo:hi], y[lo:hi], color=ACTUAL, lw=1.8, label="actual")
        ax.plot(t[lo:hi], pred[lo:hi], color=MODEL, lw=1.3, label=label)
        _lines(ax, s, t, lo, hi)
        _legend(ax)
        _dates(ax)
        fig.tight_layout()
    plt.close(fig)  # the notebook shows the returned figure once
    return fig


def plot_alarms(s: dict, alarms: pd.DataFrame):
    """The series from the deployment on with the alarms of the detector."""
    y, t, lo = s["y"], stamps(s), s["t_deploy"]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 3.4))
        ax.plot(t[lo:], y[lo:], color=ACTUAL, lw=0.9, label="series")
        for k, i in enumerate(alarms["index"]):
            ax.axvline(t[i], color=ALARM, lw=1.6, label="alarm of the detector" if k == 0 else None)
        _lines(ax, s, t, lo, len(y))
        _legend(ax)
        _dates(ax)
        fig.tight_layout()
    plt.close(fig)  # the notebook shows the returned figure once
    return fig


def daily_error(s: dict, pred: np.ndarray, scale: float) -> pd.Series:
    """Scaled absolute error of the forecasts (MASE) per day, from the deployment on."""
    y, lo = s["y"], s["t_deploy"]
    hi = lo + (int(np.isfinite(pred[lo:]).sum()) // DAY) * DAY
    err = np.abs(y[lo:hi] - pred[lo:hi]).reshape(-1, DAY).mean(axis=1) / scale
    return pd.Series(err, index=stamps(s)[lo:hi:DAY])


def plot_errors(s: dict, preds: dict[str, np.ndarray], scale: float, alarms=(), applied=()):
    """Daily error of each run; alarms and model changes as vertical lines. `preds` maps a label to forecasts."""
    t = stamps(s)
    colors = [MODEL, SYSTEM, "#8c6bb1", "#eb6834"]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 3.8))
        for (label, pred), color in zip(preds.items(), colors):
            e = daily_error(s, pred, scale)
            ax.plot(e.index, e.to_numpy(), color=color, lw=1.8, label=label)
        for k, i in enumerate(alarms):
            ax.axvline(t[i], color=ALARM, lw=1.2, ls="-", alpha=0.8, label="alarm" if k == 0 else None)
        for k, i in enumerate(applied):
            ax.axvline(t[i], color=SYSTEM, lw=2.2, ls="--", label="model changed" if k == 0 else None)
        _lines(ax, s, t, s["t_deploy"], len(t), labels=False)
        ax.set_ylabel("daily error (MASE)")
        _legend(ax)
        _dates(ax)
        fig.tight_layout()
    plt.close(fig)  # the notebook shows the returned figure once
    return fig


def review_table(events: list[dict], s: dict) -> pd.DataFrame:
    """The reviews of the model in the events of `Monitor.advance`: when and why (an alarm, or a follow-up of a
    proposal that was not confirmed), what was proposed, what the check said (gain in MASE units with its 95%
    interval) and what was applied."""
    t, rows = stamps(s), []
    for e in events:
        if e["type"] != "decision":
            continue
        check = e["checks"][0] if e["checks"] else None
        rows.append(dict(review=t[e["index"]], trigger=e["trigger"], proposed=e["proposal"],
                         gain=None if check is None else round(check["gain"], 3),
                         interval=None if check is None else f"[{check['lo']:+.3f}, {check['hi']:+.3f}]",
                         check=None if check is None else ("passed" if check["accepted"] else "not passed"),
                         applied=e["action"]))
    return pd.DataFrame(rows)


def start_session(base_url: str = "http://localhost:8000", strategy: str = "alarm_llm_gate", speed: float = 60.0,
                  path: str | Path = DEMO_CSV, llm_model: str | None = LLM_MODEL) -> dict | None:
    """Upload the demo series to the running service and start a session on it; returns the answer of the service
    (with `grafana_url`), or None with a hint if the service does not answer."""

    def post(url: str, body: bytes, content_type: str) -> dict:
        request = urllib.request.Request(url, data=body, headers={"content-type": content_type}, method="POST")
        with urllib.request.urlopen(request, timeout=120) as answer:
            return json.load(answer)

    query = urllib.parse.urlencode(dict(name=NAME, deploy_at=DEPLOY_AT, change_at=CHANGE_AT, replace="true"))
    try:
        series = post(f"{base_url}/series/csv?{query}", Path(path).read_bytes(), "text/csv")
        body = dict(series_id=series["id"], strategy=strategy, detector="residual_conformal", speed=speed)
        if "llm" in strategy:
            body["llm_model"] = llm_model
        return post(f"{base_url}/sessions", json.dumps(body).encode(), "application/json")
    except urllib.error.HTTPError as e:
        print(f"The service refused the request: {e.code} {e.read().decode()[:300]}")
    except (urllib.error.URLError, OSError):
        print(f"No service at {base_url}. Start it from the root of the repository: docker compose up --build -d")
    return None


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "series":
        DEMO_CSV.parent.mkdir(parents=True, exist_ok=True)
        make_series().to_csv(DEMO_CSV, index=False)
        print(f"written {DEMO_CSV}")
    elif step == "session":
        session = start_session()
        if session:
            print(f"session {session['id']} is playing: {session['grafana_url']}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
