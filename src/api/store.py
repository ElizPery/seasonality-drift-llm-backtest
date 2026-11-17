"""History of the sessions in PostgreSQL, read by the Grafana dashboard: the value and both forecasts of every hour, the
alarms and the reviews of the model, on the clock of the series (so the dashboard shows the real dates of the series,
not the time of the demo). The service works without it (no `DATABASE_URL`): the API and the Prometheus metrics stay,
only the history for Grafana is missing."""
from __future__ import annotations

import json
import threading

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY, series_id TEXT NOT NULL, strategy TEXT NOT NULL, detector TEXT, label TEXT NOT NULL,
    deploy_at TIMESTAMPTZ, change_at TIMESTAMPTZ, end_at TIMESTAMPTZ, created_at TIMESTAMPTZ DEFAULT now());
CREATE TABLE IF NOT EXISTS points (
    session_id INTEGER NOT NULL, ts TIMESTAMPTZ NOT NULL, value DOUBLE PRECISION, forecast DOUBLE PRECISION,
    forecast_nothing DOUBLE PRECISION, error DOUBLE PRECISION, error_nothing DOUBLE PRECISION,
    PRIMARY KEY (session_id, ts));
CREATE TABLE IF NOT EXISTS alarms (
    session_id INTEGER NOT NULL, ts TIMESTAMPTZ NOT NULL, method TEXT, score DOUBLE PRECISION, threshold DOUBLE PRECISION);
CREATE TABLE IF NOT EXISTS reviews (
    session_id INTEGER NOT NULL, ts TIMESTAMPTZ NOT NULL, trigger TEXT, outcome TEXT, proposal TEXT, reason TEXT,
    action TEXT, gain DOUBLE PRECISION, lower_bound DOUBLE PRECISION, upper_bound DOUBLE PRECISION, fits INTEGER,
    check_fits INTEGER, seconds DOUBLE PRECISION, summary TEXT, checks JSONB);
"""


def _num(x):
    return None if x is None or not np.isfinite(x) else float(x)


class Store:
    """One connection shared by the sessions (writes come from worker threads, hence the lock)."""

    def __init__(self, url: str):
        import psycopg

        self.conn = psycopg.connect(url, autocommit=True)
        self.lock = threading.Lock()
        with self.lock:
            self.conn.execute(SCHEMA)

    def new_session(self, series_id: str, strategy: str, detector: str | None, deploy_at, change_at, end_at) -> int:
        with self.lock:
            row = self.conn.execute(
                "INSERT INTO sessions (series_id, strategy, detector, label, deploy_at, change_at, end_at) "
                "VALUES (%s, %s, %s, '', %s, %s, %s) RETURNING id",
                (series_id, strategy, detector, deploy_at, change_at, end_at)).fetchone()
            sid = int(row[0])
            label = f"{sid}: {series_id} | {strategy}" + (f" | {detector}" if detector else "")
            self.conn.execute("UPDATE sessions SET label = %s WHERE id = %s", (label, sid))
        return sid

    def write(self, sid: int, points: list[tuple], alarms: list[tuple], reviews: list[dict]) -> None:
        with self.lock, self.conn.cursor() as cur:
            if points:
                cur.executemany("INSERT INTO points VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                                [(sid, ts, *map(_num, rest)) for ts, *rest in points])
            if alarms:
                cur.executemany("INSERT INTO alarms VALUES (%s, %s, %s, %s, %s)",
                                [(sid, ts, m, _num(score), _num(thr)) for ts, m, score, thr in alarms])
            for r in reviews:
                cur.execute("INSERT INTO reviews VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                            (sid, r["ts"], r["trigger"], r["outcome"], r["proposal"], r["reason"], r["action"],
                             _num(r["gain"]), _num(r["lower_bound"]), _num(r["upper_bound"]), r["fits"], r["check_fits"],
                             _num(r["seconds"]), r["summary"], json.dumps(r["checks"])))

    def delete(self, sid: int) -> None:
        with self.lock:
            for table in ("points", "alarms", "reviews"):
                self.conn.execute(f"DELETE FROM {table} WHERE session_id = %s", (sid,))
            self.conn.execute("DELETE FROM sessions WHERE id = %s", (sid,))
