"""Pydantic schemas of the API requests and answers."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    series_id: str = Field(description="id of an exported series, source/series/scenario (see GET /series)",
                           examples=["entsoe/PT/lockdown_2020"])
    strategy: str = Field("alarm_rules_gate", description="a strategy of the backtest (see GET /strategies)")
    detector: str | None = Field("profile_conformal", description="stand-in detector: profile_conformal or adwin; "
                                 "needed by the strategies that review the model on alarms")
    detector_value: float | None = Field(None, description="its setting (multiplier or ADWIN delta); "
                                         "default: the value calibrated for the source of the series")
    llm_model: str | None = Field(None, description="Instructor model string for the llm strategies")
    speed: float = Field(24.0, gt=0, le=10000, description="hours of data read per second of playing")
    skip_to_days_before_change: float | None = Field(7.0, ge=0, description="read the data from deployment up to this "
                                                     "many days before the change at once, then play; None: play from deployment")
    paused: bool = Field(False, description="create the session without starting to play")


class SessionUpdate(BaseModel):
    speed: float | None = Field(None, gt=0, le=10000)
    paused: bool | None = None


class Advance(BaseModel):
    hours: int = Field(24, ge=1, le=24 * 400, description="hours of data to read now")


class Check(BaseModel):
    action: str
    gain: float | None
    lower_bound: float | None
    upper_bound: float | None
    accepted: bool
    fits: int


class Review(BaseModel):
    time: str
    trigger: str
    alarm: dict | None = None
    outcome: Literal["applied", "rejected", "keep", "skipped"]
    proposal: str | None = None
    reason: str | None = None
    checks: list[Check] = []
    action: str
    fits: int
    seconds: float
    summary: str


class SessionInfo(BaseModel):
    id: int
    series_id: str
    strategy: str
    detector: str | None
    detector_value: float | None
    state: Literal["playing", "paused", "done", "failed"]
    speed: float
    now: str | None = Field(description="time of the next hour to read, on the clock of the series")
    deploy_at: str
    change_at: str | None
    end_at: str
    hours_read: int
    alarms: int
    reviews: int
    model_changes: int
    mase_since_deploy: dict[str, float | None]
    mase_after_change: dict[str, float | None]
    last_review: Review | None = None
    grafana_url: str | None = None
    error: str | None = None


class DetectRequest(BaseModel):
    series_id: str | None = Field(None, description="an exported series; or give `values` and `start`")
    values: list[float] | None = Field(None, description="hourly values without gaps")
    start: str | None = Field(None, description="time of the first value (for `values`)")
    method: str = "profile_conformal"
    value: float | None = Field(None, description="setting of the detector; default: calibrated for the source")
    from_index: int | None = Field(None, description="report alarms from this position on (default: deployment of the "
                                   "exported series, or 5 weeks into `values`)")


class Alarm(BaseModel):
    index: int
    time: str
    score: float | None
    threshold: float | None


class DetectAnswer(BaseModel):
    method: str
    value: float
    alarms: list[Alarm]


class RecommendationRequest(BaseModel):
    session_id: int
    advisor: str = Field("rules", description="rules, or llm (needs `llm_model` of the session or here)")
    llm_model: str | None = None


class RecommendationAnswer(BaseModel):
    session_id: int
    origin: str = Field(description="the decision point: 00:00 of the current day of the session")
    diagnostics: dict
    proposal: str
    reason: str
    check: Check | None = Field(None, description="the acceptance rule on the two weeks before the origin; None for keep")
    verdict: str
