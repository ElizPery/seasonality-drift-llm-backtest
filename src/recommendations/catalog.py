"""Initial catalog of corrective actions for stream B (RQ3).

Every action is a complete, executable recipe: the LLM will later choose an action by name, and the
backtest can apply and measure any of them. Retraining actions do not depend on the current model, so
the outcome of an action is reproducible from the data alone. `keep` and `recalibrate_level` act on the
current model.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from src.recommendations.forecast import PROFILE_FEATURE, WEEK, Model, fit_model


@dataclass(frozen=True)
class Action:
    name: str
    title_uk: str
    title_en: str
    kind: str  # none | retrain | recalibrate | intervals
    window: int | None = None  # training window in hours (None = all history)
    half_life: int | None = None  # exponential sample weights, in hours
    add_feature: str | None = None


CATALOG: tuple[Action, ...] = (
    Action("keep", "нічого не робити", "do nothing", "none"),
    Action("retrain_full", "перенавчити на всій історії", "retrain on all history", "retrain"),
    Action("retrain_8w", "перенавчити на останньому вікні (8 тижнів)", "retrain on the last 8 weeks", "retrain", window=8 * WEEK),
    Action("retrain_2w", "змінити довжину вікна: перенавчити на 2 тижнях", "shorten the training window to 2 weeks", "retrain", window=2 * WEEK),
    Action("reweight_1w", "перезважити свіжі дані (напіврозпад ваги 1 тиждень)", "reweight recent data (weight half-life 1 week)", "retrain", half_life=WEEK),
    Action("add_profile", 'додати сезонну ознаку: профіль "година тижня" за 4 тижні', "add a seasonal feature: 4-week hour-of-week profile", "retrain", add_feature=PROFILE_FEATURE),
    Action("recalibrate_level", "перекалібрувати рівень прогнозу за останній тиждень", "recalibrate the forecast level on the last week", "recalibrate", window=WEEK),
)
# changes prediction intervals only, so it is not part of point-forecast (MASE) experiments yet
INTERVAL_ACTIONS: tuple[Action, ...] = (
    Action("recalibrate_intervals", "перекалібрувати інтервали прогнозу", "recalibrate prediction intervals", "intervals"),
)
ACTIONS = {a.name: a for a in CATALOG + INTERVAL_ACTIONS}


def apply_action(action: Action, model: Model, F: dict[str, np.ndarray], y: np.ndarray, origin: int,
                 base_features: tuple[str, ...], params: dict | None = None) -> tuple[Model, int]:
    """Apply `action` at `origin` using data before `origin`. Returns the new model and the number of fits."""
    if action.kind == "none":
        return model, 0
    if action.kind == "recalibrate":
        rows = np.arange(max(0, origin - action.window), origin)
        resid = y[rows] - model.predict(F, rows)
        return replace(model, bias=model.bias + float(np.nanmean(resid))), 0
    if action.kind != "retrain":
        raise ValueError(f"{action.name} does not change point forecasts")
    features = base_features + ((action.add_feature,) if action.add_feature else ())
    start = 0 if action.window is None else max(0, origin - action.window)
    rows = np.arange(start, origin)
    weights = None if action.half_life is None else 0.5 ** ((origin - 1 - rows) / action.half_life)
    return fit_model(F, y, rows, features, weights, params), 1
