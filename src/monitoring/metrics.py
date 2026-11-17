"""Prometheus metrics of the monitoring service (prometheus_client, default registry, served at /metrics).

Two groups: the service itself (requests, their latency, sessions playing) and what the watched models do (values and
forecasts of the last hour, the error of the last 7 days, alarms, reviews and their outcomes, model fits, the lower
bound of the last check, the time a review takes). The model metrics carry the labels of a session: `session` (its number), `series` (the id
`source/series/scenario`), `strategy` and `detector`; `model` is `strategy` (the model the strategy maintains) or
`nothing` (the model left as deployed)."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

SESSION_LABELS = ("session", "series", "strategy", "detector")

REQUESTS = Counter("drift_api_requests_total", "HTTP requests to the service", ("method", "path", "status"))
REQUEST_SECONDS = Histogram("drift_api_request_seconds", "Time to answer an HTTP request", ("path",))
SESSIONS = Gauge("drift_sessions", "Sessions by state", ("state",))

HOURS = Counter("drift_hours_read_total", "Hours of data read", SESSION_LABELS)
VALUE = Gauge("drift_value", "Last value read", SESSION_LABELS)
FORECAST = Gauge("drift_forecast", "Forecast of the last hour read", SESSION_LABELS + ("model",))
ERROR_7D = Gauge("drift_mase_7d", "MASE over the last 7 days read", SESSION_LABELS + ("model",))
SERIES_TIME = Gauge("drift_series_time_seconds", "Time of the last hour read, on the clock of the series", SESSION_LABELS)
ALARMS = Counter("drift_alarms_total", "Alarms of the drift detector", SESSION_LABELS)
REVIEWS = Counter("drift_reviews_total", "Reviews of the model", SESSION_LABELS + ("trigger",))
OUTCOMES = Counter("drift_review_outcomes_total", "Outcome of a review: applied, rejected (the check did not confirm a gain), "
                   "keep (no change proposed) or skipped (too little history to check)", SESSION_LABELS + ("outcome",))
FITS = Counter("drift_model_fits_total", "Model fits: to apply an action or to check one", SESSION_LABELS + ("purpose",))
MODEL_CHANGES = Gauge("drift_model_changes", "Changes of the model since deployment", SESSION_LABELS)
LAST_LOWER_BOUND = Gauge("drift_last_check_lower_bound", "Lower bound of the 95% interval of the MASE gain in the last check",
                         SESSION_LABELS)
REVIEW_SECONDS = Histogram("drift_review_seconds", "Time a review takes: advisor, checks and the action", ("strategy",),
                           buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64))

PER_SESSION = (HOURS, VALUE, FORECAST, ERROR_7D, SERIES_TIME, ALARMS, REVIEWS, OUTCOMES, FITS, MODEL_CHANGES, LAST_LOWER_BOUND)
EXTRA_LABEL_VALUES = {FORECAST: ("strategy", "nothing"), ERROR_7D: ("strategy", "nothing"), REVIEWS: ("alarm", "calendar"),
                      OUTCOMES: ("applied", "rejected", "keep", "skipped"), FITS: ("apply", "check")}


def outcome(rec: dict) -> str:
    """Outcome of a review from its record (see `backtester.StrategyRun`)."""
    if rec.get("skipped"):
        return "skipped"
    if rec["changed"]:
        return "applied"
    return "rejected" if rec["checks"] else "keep"


def forget(labels: dict) -> None:
    """Drop every series of one session (when the session is deleted)."""
    key = tuple(labels[k] for k in SESSION_LABELS)
    for metric in PER_SESSION:
        for extra in EXTRA_LABEL_VALUES.get(metric, (None,)):
            try:
                metric.remove(*key, *(() if extra is None else (extra,)))
            except KeyError:  # this combination was never set
                pass
