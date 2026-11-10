"""Monitoring service of stream B (FastAPI): a deployed forecasting model watched as the data arrive, with the decisions
of the backtest (detector alarm, advisor, acceptance check, action) taken live.

    uvicorn src.api.main:app --port 8000          # docs at http://localhost:8000/docs, metrics at /metrics
    docker compose up --build                     # with PostgreSQL, Prometheus and Grafana (see README)

A session plays one exported series (`python -m src.data.export`) hour by hour from the deployment of the model. The
numbers are those of the experiments: the decisions are taken by the code of the backtest (`StrategyRun`), and the
stand-in detectors give the alarms of `data/alerts/alerts.csv`.

Environment: SERIES_DIR (exported series, default data/processed/series), DETECTORS_FILE (calibrated detector settings,
default data/alerts/detectors.csv), DATABASE_URL (PostgreSQL for the Grafana history; optional), GRAFANA_URL (address
of Grafana for the links in the answers, default http://localhost:3000; empty: no links).
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import make_asgi_app

from src.api import schemas
from src.api.sessions import Manager, advisors_for
from src.monitoring import metrics as M
from src.recommendations.alarms import DETECTORS, alarms_in
from src.recommendations.backtester import VAL_WEEKS, check_action
from src.recommendations.catalog import ACTIONS, CATALOG
from src.recommendations.data_sources import read_series
from src.recommendations.forecast import DAY, WEEK
from src.recommendations.llm_client import diagnostics
from src.recommendations.monitor import STRATEGY_BY_NAME


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "manager", None) is None:  # tests set their own
        app.state.manager = await asyncio.to_thread(Manager)
    yield
    for sid in list(app.state.manager.sessions):
        await app.state.manager.delete(sid) if app.state.manager.store is None else app.state.manager.sessions[sid].task.cancel()


app = FastAPI(title="Drift monitoring service (stream B)", version="0.1.0", lifespan=lifespan,
              description="Seasonality-aware drift detection and verified recommendations for a deployed forecasting model.")
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def count_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    path = getattr(route, "path", None) or ("/metrics" if request.url.path.startswith("/metrics") else "unmatched")
    M.REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    M.REQUEST_SECONDS.labels(path).observe(time.perf_counter() - t0)
    return response


def manager(request: Request) -> Manager:
    return request.app.state.manager


def session_or_404(request: Request, sid: int):
    try:
        return manager(request).get(sid)
    except KeyError:
        raise HTTPException(404, f"no session {sid}") from None


@app.get("/health")
def health(request: Request):
    m = manager(request)
    return dict(status="ok", series=len(m.manifest), sessions=len(m.sessions), database=m.store is not None)


@app.get("/series")
def list_series(request: Request, source: str | None = None, scenario: str | None = None, limit: int = 300):
    df = manager(request).manifest
    if source:
        df = df[df["source"] == source]
    if scenario:
        df = df[df["scenario"] == scenario]
    cols = ["id", "source", "series", "scenario", "n", "start", "deploy_at", "change_at"]
    return df[cols].head(limit).astype(object).where(df[cols].head(limit).notna(), None).to_dict("records")


@app.get("/strategies")
def list_strategies(request: Request):
    return dict(strategies=[dict(name=st.name, title=st.title, trigger=st.trigger, policy=st.policy, advisor=st.advisor,
                                 check=st.policy == "gated" or st.gate) for st in STRATEGY_BY_NAME.values()],
                detectors=list(DETECTORS), calibrated=manager(request).settings,
                catalog=[dict(name=a.name, title=a.title) for a in CATALOG])


@app.post("/sessions", status_code=201)
async def create_session(request: Request, req: schemas.SessionCreate) -> schemas.SessionInfo:
    m = manager(request)
    try:
        session = await asyncio.to_thread(m.create, req)
    except KeyError as e:
        raise HTTPException(404, f"no series {e}") from None
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    m.start(session)
    return session.info(m.grafana_url)


@app.get("/sessions")
def list_sessions(request: Request) -> list[schemas.SessionInfo]:
    m = manager(request)
    return [s.info(m.grafana_url) for s in m.sessions.values()]


@app.get("/sessions/{sid}")
def get_session(request: Request, sid: int) -> schemas.SessionInfo:
    return session_or_404(request, sid).info(manager(request).grafana_url)


@app.patch("/sessions/{sid}")
def update_session(request: Request, sid: int, req: schemas.SessionUpdate) -> schemas.SessionInfo:
    session = session_or_404(request, sid)
    if req.speed is not None:
        session.speed = req.speed
    if req.paused is not None:
        manager(request).set_paused(session, req.paused)
    return session.info(manager(request).grafana_url)


@app.post("/sessions/{sid}/advance")
async def advance_session(request: Request, sid: int, req: schemas.Advance) -> schemas.SessionInfo:
    """Read the next hours at once (for a paused session; a playing one goes on from there)."""
    session = session_or_404(request, sid)
    await asyncio.to_thread(session.process, req.hours)
    return session.info(manager(request).grafana_url)


@app.delete("/sessions/{sid}", status_code=204)
async def delete_session(request: Request, sid: int):
    session_or_404(request, sid)
    await manager(request).delete(sid)


@app.get("/sessions/{sid}/reviews")
def session_reviews(request: Request, sid: int, outcome: str | None = None) -> list[schemas.Review]:
    session = session_or_404(request, sid)
    return [session.review_view(r) for r in session.reviews if outcome is None or r["outcome"] == outcome]


@app.post("/detect")
def detect(request: Request, req: schemas.DetectRequest) -> schemas.DetectAnswer:
    """Alarms of a stand-in detector on an exported series or on posted hourly values (only the past is used)."""
    m = manager(request)
    if req.method not in DETECTORS:
        raise HTTPException(422, f"unknown method {req.method}; known: {', '.join(DETECTORS)}")
    if req.series_id is not None:
        try:
            s = read_series(req.series_id, m.series_dir, m.manifest)
        except KeyError:
            raise HTTPException(404, f"no series {req.series_id}") from None
        y, start, source = s["y"], pd.Timestamp(s["start"]), s["source"]
        first = s["t_deploy"] if req.from_index is None else req.from_index
    elif req.values and req.start:
        y, start, source = np.asarray(req.values, dtype=float), pd.Timestamp(req.start), None
        first = 5 * WEEK if req.from_index is None else req.from_index
        if not np.isfinite(y).all():
            raise HTTPException(422, "values must be finite hourly values without gaps")
    else:
        raise HTTPException(422, "give series_id, or values and start")
    if source is None and req.value is None:
        raise HTTPException(422, "give the setting `value` of the detector for posted values")
    try:
        value = m.setting(source, req.method, req.value) if source else float(req.value)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    a = alarms_in(req.method, y, first, value)
    return dict(method=req.method, value=value, alarms=[
        dict(index=int(i), time=(start + pd.Timedelta(hours=int(i))).isoformat(),
             score=None if not np.isfinite(sc) else float(sc), threshold=None if not np.isfinite(th) else float(th))
        for i, sc, th in zip(a["index"], a["score"], a["threshold"])])


@app.post("/recommendations")
async def recommend(request: Request, req: schemas.RecommendationRequest) -> schemas.RecommendationAnswer:
    """What the advisor proposes now for the model of a session, and whether the acceptance rule confirms it on the two
    weeks before the current day: a recommendation with its evidence. Nothing is applied."""
    session = session_or_404(request, req.session_id)

    def work():
        with session.lock:
            mon = session.monitor
            run, s = mon.run, session.s
            t_dep = s["t_deploy"]
            T = t_dep + ((min(mon.now, mon.end - 1) - t_dep) // DAY) * DAY  # 00:00 of the current day
            if T - VAL_WEEKS * WEEK < t_dep:
                raise ValueError(f"the session needs {VAL_WEEKS} weeks after deployment first")
            st = STRATEGY_BY_NAME["rules_gate" if req.advisor == "rules" else "llm_gate"]
            advisor = advisors_for(st, req.llm_model or session.llm_model)[st.advisor]
            ctx = diagnostics(s["y"], run.pred, T, t_dep, run.scale, run.last_change)
            answer = advisor.recommend(ctx)
            a = ACTIONS[answer.action]
            check = None
            if a.kind != "none":
                c = check_action(a, run.model, s["y"], s["F"], run.pred, T, run.scale, run.act)
                check = dict(action=c["action"], gain=c["gain"], lower_bound=c["lo"], upper_bound=c["hi"],
                             accepted=c["accepted"], fits=c["fits"])
            return T, ctx, answer, check

    try:
        T, ctx, answer, check = await asyncio.to_thread(work)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    if check is None:
        verdict = "keep the model: no change proposed"
    elif check["accepted"]:
        verdict = f"apply {check['action']}: the lower bound of the gain {check['lower_bound']:+.3f} is above 0"
    else:
        verdict = f"do not apply {check['action']}: the lower bound of the gain {check['lower_bound']:+.3f} is not above 0"
    return dict(session_id=session.id, origin=session.ts(T).isoformat(), diagnostics=ctx, proposal=answer.action,
                reason=answer.reason, check=check, verdict=verdict)
