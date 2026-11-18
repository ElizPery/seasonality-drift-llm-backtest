"""Sessions of the monitoring service: one exported or uploaded series played hour by hour under one strategy
(`Monitor`), with its Prometheus metrics and, if a database is configured, its history for Grafana. A session plays in
the background at `speed` hours of data per second; it can be paused, advanced by hand, sped up or deleted. The manager
also keeps the series uploaded through the API (`src.api.uploads`) in their own folder."""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from src.api import uploads
from src.data.frames import series_file, series_frame, series_table
from src.monitoring import metrics as M
from src.recommendations.alarms import ALERTS_DIR, detector_settings
from src.recommendations.data_sources import EXPORT_DIR, read_manifest, read_series
from src.recommendations.forecast import DAY, WEEK
from src.recommendations.monitor import STRATEGY_BY_NAME, Monitor

log = logging.getLogger("drift.sessions")
TICK = 0.25  # seconds between two reads of a playing session
DASHBOARD = "d/drift-monitor/drift-monitor"
UPLOAD_DIR = Path("data/processed/uploads")  # series uploaded through the API, with their own manifest.csv
MANIFEST_COLUMNS = ["id", "file", "source", "series", "scenario", "client", "seed", "magnitude", "n", "start", "t_deploy",
                    "t_change", "t_stop", "deploy_at", "change_at"]


def advisors_for(strategy, llm_model: str | None) -> dict | None:
    if strategy.advisor == "rules":
        from src.recommendations.llm_client import RuleAdvisor

        return {"rules": RuleAdvisor()}
    if strategy.advisor == "llm":
        from src.recommendations.llm_client import LLMAdvisor

        if not llm_model:
            raise ValueError(f"{strategy.name} needs `llm_model`, an Instructor model string")
        return {"llm": LLMAdvisor(llm_model)}
    return None


def _num(x) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)


def review_summary(rec: dict, outcome: str) -> str:
    """One line for the dashboard: why the model was reviewed, what was proposed, what the check said, what was done."""
    a = rec.get("alarm")
    why = f"alarm of {a['method']} {a['hours_ago']} h ago" if a else "weekly review"
    if outcome == "skipped":
        return f"{why}: {rec['skipped']}"
    checks = rec["checks"]
    best = max(checks, key=lambda c: c["lo"] if np.isfinite(c["lo"]) else -np.inf) if checks else None
    if rec.get("reason") is not None:
        what = f"advisor proposed {rec['proposal']} ({rec['reason']})"
    elif len(checks) > 1:
        what = f"checked {len(checks)} catalog actions"
    else:
        what = f"proposed {rec['proposal']}"
    if best is not None:
        what += f"; check of {best['action']}: gain {best['gain']:+.3f}, 95% interval [{best['lo']:+.3f}, {best['hi']:+.3f}]"
    done = {"applied": f"applied {rec['action']}", "rejected": "no gain confirmed, model kept", "keep": "model kept"}[outcome]
    return f"{why}: {what}; {done}"


class Session:
    def __init__(self, sid: int, row: pd.Series, s: dict, monitor: Monitor, req, detector_value: float | None, store):
        self.id, self.row, self.s, self.monitor, self.store = sid, row, s, monitor, store
        self.series_id, self.strategy, self.detector = row["id"], monitor.strategy.name, req.detector if monitor.detector else None
        self.detector_value = detector_value
        self.llm_model = req.llm_model
        self.speed, self.paused = req.speed, req.paused
        self.state = "paused" if req.paused else "playing"
        self.error: str | None = None
        self.labels = dict(session=str(sid), series=self.series_id, strategy=self.strategy, detector=self.detector or "none")
        self.start = pd.Timestamp(row["start"])
        if self.start.tzinfo is None:  # local or nominal clock: shown as it is (the dashboard displays UTC)
            self.start = self.start.tz_localize("UTC")
        t_chg = s["t_change"]
        self.skip_to = None
        if req.skip_to_days_before_change is not None and t_chg is not None:
            self.skip_to = max(s["t_deploy"], int(t_chg - req.skip_to_days_before_change * DAY))
        self.lock = threading.Lock()
        self.carry = 0.0
        self.alarms = 0
        self.reviews: list[dict] = []
        self.errors = {"strategy": deque(maxlen=WEEK), "nothing": deque(maxlen=WEEK)}
        self.task: asyncio.Task | None = None

    def ts(self, i: int) -> pd.Timestamp:
        return self.start + pd.Timedelta(hours=int(i))

    def process(self, hours: int) -> int:
        """Read the next `hours` values: metrics, history and the list of reviews. Returns the hours read."""
        with self.lock:
            before = self.monitor.now
            events = self.monitor.advance(hours)
            lab = self.labels
            points, alarms, reviews = [], [], []
            for e in events:
                if e["type"] == "observation":
                    ts = self.ts(e["index"])
                    points.append((ts, e["value"], e["forecast"], e["forecast_nothing"], e["error"], e["error_nothing"]))
                    self.errors["strategy"].append(e["error"])
                    self.errors["nothing"].append(e["error_nothing"])
                elif e["type"] == "alarm":
                    self.alarms += 1
                    M.ALARMS.labels(**lab).inc()
                    alarms.append((self.ts(e["index"]), e["method"], e["score"], e["threshold"]))
                elif e["type"] == "decision":
                    reviews.append(self._review(e))
            if points:
                ts, value, f, q, _, _ = points[-1]
                M.HOURS.labels(**lab).inc(len(points))
                M.VALUE.labels(**lab).set(value)
                M.FORECAST.labels(**lab, model="strategy").set(f)
                M.FORECAST.labels(**lab, model="nothing").set(q)
                for model, errs in self.errors.items():
                    M.ERROR_7D.labels(**lab, model=model).set(float(np.mean(errs)))
                M.SERIES_TIME.labels(**lab).set(ts.timestamp())
            if self.store is not None and (points or alarms or reviews):
                self.store.write(self.id, points, alarms, reviews)
            if self.monitor.done and self.state in ("playing", "paused"):
                self.state = "done"
            return self.monitor.now - before

    def _review(self, e: dict) -> dict:
        lab = self.labels
        outcome = M.outcome(e)
        checks = e["checks"]
        check_fits = sum(c["fits"] for c in checks)
        M.REVIEWS.labels(**lab, trigger=e["trigger"]).inc()
        M.OUTCOMES.labels(**lab, outcome=outcome).inc()
        M.FITS.labels(**lab, purpose="apply").inc(e["fits"])
        M.FITS.labels(**lab, purpose="check").inc(check_fits)
        M.MODEL_CHANGES.labels(**lab).set(len(self.monitor.run.applied))
        M.REVIEW_SECONDS.labels(self.strategy).observe(e["seconds"])
        best = max(checks, key=lambda c: c["lo"] if np.isfinite(c["lo"]) else -np.inf) if checks else None
        if best is not None:
            M.LAST_LOWER_BOUND.labels(**lab).set(best["lo"])
        r = dict(ts=self.ts(e["index"]), trigger=e["trigger"], alarm=e["alarm"], outcome=outcome, proposal=e["proposal"],
                 reason=e.get("reason"), action=e["action"], gain=best["gain"] if best else None,
                 lower_bound=best["lo"] if best else None, upper_bound=best["hi"] if best else None, fits=int(e["fits"]),
                 check_fits=int(check_fits), seconds=float(e["seconds"]), summary=review_summary(e, outcome),
                 checks=[dict(action=c["action"], gain=_num(c["gain"]), lower_bound=_num(c["lo"]),
                              upper_bound=_num(c["hi"]), accepted=bool(c["accepted"]), fits=int(c["fits"])) for c in checks])
        self.reviews.append(r)
        return r

    def review_view(self, r: dict) -> dict:
        return dict(time=r["ts"].isoformat(), trigger=r["trigger"], alarm=r["alarm"], outcome=r["outcome"],
                    proposal=r["proposal"], reason=r["reason"], checks=r["checks"], action=r["action"], fits=r["fits"],
                    seconds=round(r["seconds"], 3), summary=r["summary"])

    def info(self, grafana: str | None) -> dict:
        m, s = self.monitor, self.s
        t_chg = s["t_change"]
        after = m.mase(t_chg) if t_chg is not None and m.now > t_chg else (None, None)
        since = m.mase() if m.now > s["t_deploy"] else (None, None)
        url = None
        if grafana:
            centre = self.ts(t_chg if t_chg is not None else s["t_deploy"])
            lo = max(self.ts(s["t_deploy"]) - pd.Timedelta(days=7), centre - pd.Timedelta(days=21))
            hi = min(self.ts(m.end), centre + pd.Timedelta(days=42))
            url = (f"{grafana.rstrip('/')}/{DASHBOARD}?var-session={self.id}&from={int(lo.timestamp() * 1000)}"
                   f"&to={int(hi.timestamp() * 1000)}&refresh=5s")
        return dict(id=self.id, series_id=self.series_id, strategy=self.strategy, detector=self.detector,
                    detector_value=self.detector_value, state=self.state, speed=self.speed,
                    now=self.ts(m.now).isoformat() if not m.done else None, deploy_at=self.ts(s["t_deploy"]).isoformat(),
                    change_at=self.ts(t_chg).isoformat() if t_chg is not None else None, end_at=self.ts(m.end).isoformat(),
                    hours_read=m.now - s["t_deploy"], alarms=self.alarms, reviews=len(self.reviews),
                    model_changes=len(m.run.applied),
                    mase_since_deploy=dict(strategy=_num(since[0]), nothing=_num(since[1])),
                    mase_after_change=dict(strategy=_num(after[0]), nothing=_num(after[1])),
                    last_review=self.review_view(self.reviews[-1]) if self.reviews else None, grafana_url=url,
                    error=self.error)


class Manager:
    def __init__(self, series_dir: str | Path | None = None, detectors_file: str | Path | None = None,
                 database_url: str | None = None, grafana_url: str | None = None, upload_dir: str | Path | None = None):
        self.series_dir = Path(series_dir or os.environ.get("SERIES_DIR", EXPORT_DIR))
        if (self.series_dir / "manifest.csv").exists():
            self.exported = read_manifest(self.series_dir)
        else:  # the service starts, but has no series until one is exported (`python -m src.data.export`) or uploaded
            log.warning("no exported series in %s: run python -m src.data.export", self.series_dir)
            self.exported = pd.DataFrame(columns=MANIFEST_COLUMNS)
        self.upload_dir = Path(upload_dir or os.environ.get("UPLOAD_DIR", UPLOAD_DIR))
        self.uploaded = (read_manifest(self.upload_dir) if (self.upload_dir / "manifest.csv").exists()
                         else pd.DataFrame(columns=MANIFEST_COLUMNS))
        self.uploaded_settings = pd.DataFrame(columns=["id", "method", "param", "value", "far_control_per_1000",
                                                       "target_per_1000", "control_hours"])
        if (self.upload_dir / "detectors.csv").exists():
            self.uploaded_settings = pd.read_csv(self.upload_dir / "detectors.csv")
        self.upload_lock = threading.Lock()
        path = Path(detectors_file or os.environ.get("DETECTORS_FILE", ALERTS_DIR / "detectors.csv"))
        self.settings = detector_settings(path) if path.exists() else {}
        self.grafana_url = grafana_url if grafana_url is not None else os.environ.get("GRAFANA_URL", "http://localhost:3000")
        self.store = None
        url = database_url if database_url is not None else os.environ.get("DATABASE_URL")
        if url:
            from src.api.store import Store

            self.store = Store(url)
        self.sessions: dict[int, Session] = {}
        self._next = 1

    def count_states(self) -> None:
        for state in ("playing", "paused", "done", "failed"):
            M.SESSIONS.labels(state).set(sum(1 for s in self.sessions.values() if s.state == state))

    @property
    def manifest(self) -> pd.DataFrame:
        """The exported series and the uploaded ones."""
        if self.uploaded.empty:
            return self.exported
        return self.uploaded if self.exported.empty else pd.concat([self.exported, self.uploaded], ignore_index=True)

    def is_uploaded(self, series_id: str) -> bool:
        return bool((self.uploaded["id"] == series_id).any())

    def read(self, series_id: str) -> dict:
        """One series by its id, from the folder of the exported or of the uploaded series (KeyError if unknown)."""
        if self.is_uploaded(series_id):
            return read_series(series_id, self.upload_dir, self.uploaded)
        return read_series(series_id, self.series_dir, self.exported)

    def setting(self, source: str, method: str, value: float | None, series_id: str | None = None) -> float:
        """Setting of a detector: the given value, else the one calibrated for the uploaded series, else for the source."""
        if value is not None:
            return float(value)
        own = self.uploaded_settings
        own = own[(own["id"] == series_id) & (own["method"] == method)]
        if not own.empty:
            return float(own["value"].iloc[0])
        try:
            return self.settings[source][method]
        except KeyError:
            raise ValueError(f"no calibrated setting of {method} for {source}: give it explicitly") from None

    def series_info(self, series_id: str) -> dict:
        row = self.manifest[self.manifest["id"] == series_id].iloc[0]
        own = self.uploaded_settings[self.uploaded_settings["id"] == series_id]
        change = row["change_at"]
        stamp = lambda x: str(pd.Timestamp(x))  # noqa: E731  (a manifest column of midnights is written as dates)
        return dict(id=series_id, source=row["source"], n=int(row["n"]), start=stamp(row["start"]),
                    deploy_at=stamp(row["deploy_at"]), change_at=None if pd.isna(change) else stamp(change),
                    history_hours=int(row["t_deploy"]),
                    detectors=[dict(method=r["method"], param=r["param"], value=float(r["value"]),
                                    false_alarms_per_1000_hours=float(r["far_control_per_1000"]),
                                    target_per_1000_hours=float(r["target_per_1000"]), control_hours=int(r["control_hours"]))
                               for _, r in own.iterrows()])

    def add_series(self, s: dict, replace: bool = False) -> dict:
        """Store an uploaded series (`uploads.build`) with the detector settings calibrated on its own history. Raises
        FileExistsError if the id is taken and `replace` is not set; an exported series is never replaced."""
        sid = uploads.series_id(s["series"])
        if (self.exported["id"] == sid).any() or (self.is_uploaded(sid) and not replace):
            raise FileExistsError(sid)
        calibrated = uploads.calibrate_on_history(s)  # outside the lock: it takes seconds on a long series
        row = series_table([s])
        row.insert(1, "file", [series_file(s)])
        with self.upload_lock:
            self.upload_dir.mkdir(parents=True, exist_ok=True)
            series_frame(s).to_csv(self.upload_dir / row["file"].iloc[0], index=False)
            self._forget(sid)
            own = pd.DataFrame([dict(id=sid, **c) for c in calibrated])
            self.uploaded = row if self.uploaded.empty else pd.concat([self.uploaded, row], ignore_index=True)
            self.uploaded_settings = own if self.uploaded_settings.empty else pd.concat([self.uploaded_settings, own],
                                                                                        ignore_index=True)
            self._save_uploads()
            self.uploaded = read_manifest(self.upload_dir)  # with the types of a manifest read from disk
        return self.series_info(sid)

    def remove_series(self, series_id: str) -> None:
        """Delete an uploaded series (KeyError if it is not one). Sessions that play it keep their copy of the data."""
        with self.upload_lock:
            rows = self.uploaded[self.uploaded["id"] == series_id]
            if rows.empty:
                raise KeyError(series_id)
            (self.upload_dir / rows["file"].iloc[0]).unlink(missing_ok=True)
            self._forget(series_id)
            self._save_uploads()

    def _forget(self, series_id: str) -> None:
        self.uploaded = self.uploaded[self.uploaded["id"] != series_id].reset_index(drop=True)
        self.uploaded_settings = self.uploaded_settings[self.uploaded_settings["id"] != series_id].reset_index(drop=True)

    def _save_uploads(self) -> None:
        self.uploaded.to_csv(self.upload_dir / "manifest.csv", index=False)
        self.uploaded_settings.to_csv(self.upload_dir / "detectors.csv", index=False)

    def create(self, req) -> Session:
        """Build the session (loads the series and trains the deployed model, a few seconds); not yet playing."""
        if req.strategy not in STRATEGY_BY_NAME:
            raise ValueError(f"unknown strategy {req.strategy}")
        st = STRATEGY_BY_NAME[req.strategy]
        rows = self.manifest[self.manifest["id"] == req.series_id]
        if rows.empty:
            raise KeyError(req.series_id)
        row = rows.iloc[0]
        s = self.read(req.series_id)
        detector = req.detector if (st.trigger == "alarm" or req.detector) else None
        value = self.setting(s["source"], detector, req.detector_value, req.series_id) if detector else None
        monitor = Monitor(s, st, detector=detector, detector_value=value, advisors=advisors_for(st, req.llm_model))
        if self.store is not None:
            start = pd.Timestamp(row["start"])
            start = start.tz_localize("UTC") if start.tzinfo is None else start
            at = lambda i: None if i is None else start + pd.Timedelta(hours=int(i))  # noqa: E731
            sid = self.store.new_session(req.series_id, st.name, detector, at(s["t_deploy"]), at(s["t_change"]), at(monitor.end))
        else:
            sid = self._next
        self._next = max(self._next, sid) + 1
        session = Session(sid, row, s, monitor, req, value, self.store)
        self.sessions[sid] = session
        return session

    async def play(self, session: Session) -> None:
        loop = asyncio.get_running_loop()
        try:
            while not session.monitor.done:
                if session.paused:
                    await asyncio.sleep(TICK)
                    continue
                if session.skip_to is not None and session.monitor.now < session.skip_to:
                    await asyncio.to_thread(session.process, min(session.skip_to - session.monitor.now, 7 * DAY))
                    continue
                t0 = loop.time()
                session.carry += session.speed * TICK
                n = int(session.carry)
                session.carry -= n
                if n:
                    await asyncio.to_thread(session.process, n)
                await asyncio.sleep(max(0.0, TICK - (loop.time() - t0)))
            session.state = "done"
        except asyncio.CancelledError:
            raise
        except Exception as e:  # keep the service up; the session shows the error
            log.exception("session %s failed", session.id)
            session.state, session.error = "failed", repr(e)
        finally:
            self.count_states()

    def start(self, session: Session) -> None:
        session.task = asyncio.get_running_loop().create_task(self.play(session))
        self.count_states()

    def get(self, sid: int) -> Session:
        if sid not in self.sessions:
            raise KeyError(sid)
        return self.sessions[sid]

    async def delete(self, sid: int) -> None:
        session = self.get(sid)
        if session.task is not None:
            session.task.cancel()
            try:
                await session.task
            except asyncio.CancelledError:
                pass
        del self.sessions[sid]
        M.forget(session.labels)
        if self.store is not None:
            await asyncio.to_thread(self.store.delete, sid)
        self.count_states()

    def set_paused(self, session: Session, paused: bool) -> None:
        session.paused = paused
        if session.state in ("playing", "paused"):
            session.state = "paused" if paused else "playing"
        self.count_states()
