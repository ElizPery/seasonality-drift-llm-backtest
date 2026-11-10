"""Watching a deployed forecasting model as the data arrive: the engine of the monitoring service (`src.api`).

`Monitor` plays one series hour by hour. At 00:00 of every day the deployed model forecasts the next 24 hours; every
hour the new value is compared with its forecast and fed to a stand-in drift detector (`alarms.OnlineDetector`); on an
alarm (or on the calendar, for the weekly strategies) the strategy reviews the model: the advisor proposes an action,
the acceptance rule checks it on the last two weeks, and the action is applied only if the check passes. The decisions
are taken by `StrategyRun`, the code of the backtest, so a series played to its end gives exactly the numbers of the
experiments. A run of `nothing` on the same series gives the error the model would have without any change.

Only data up to the current hour is used: the forecast of a day is made before its first value is read, and an alarm
raised at hour i is known to the decisions from the next origin on (as in the backtest).
"""
from __future__ import annotations

import time

import numpy as np

from src.recommendations.alarms import OnlineDetector
from src.recommendations.backtester import ADVISED_STRATEGIES, ALARM_STRATEGIES, STRATEGIES, Strategy, StrategyRun
from src.recommendations.forecast import DAY

STRATEGY_BY_NAME = {st.name: st for st in STRATEGIES + ADVISED_STRATEGIES + ALARM_STRATEGIES}


class Monitor:
    """One series under one strategy, one hour at a time. `advance(hours)` reads the next values and returns the
    events in time order: `forecast` (the day ahead of both models at 00:00), `decision` (a review of the model,
    with the record of `StrategyRun`), `observation` (the value and both forecasts of the hour) and `alarm`."""

    def __init__(self, s: dict, strategy: Strategy | str, detector: str | None = None, detector_value: float | None = None,
                 advisors: dict | None = None, params: dict | None = None):
        strategy = STRATEGY_BY_NAME[strategy] if isinstance(strategy, str) else strategy
        if strategy.trigger == "alarm" and detector is None:
            raise ValueError(f"{strategy.name} reviews the model on detector alarms: a detector is needed")
        if strategy.policy == "advised" and not (advisors and strategy.advisor in advisors):
            raise ValueError(f"{strategy.name} needs the advisor {strategy.advisor!r}")
        self.s, self.strategy = s, strategy
        y, t_dep = s["y"], s["t_deploy"]
        cache: dict = {}  # fits shared by the two runs, as between the strategies of the backtest
        self.run = StrategyRun(y, s["F"], t_dep, strategy, s["base_features"], params, cache=cache, advisors=advisors,
                               detector=detector)
        self.reference = StrategyRun(y, s["F"], t_dep, STRATEGY_BY_NAME["nothing"], s["base_features"], params, cache=cache)
        self.scale, self.end = self.run.scale, self.run.end
        self.detector = OnlineDetector(detector, detector_value, t_dep) if detector else None
        self.alarms: list[int] = []
        self.now = 0  # position of the next value to read
        if self.detector is not None:  # the history before deployment is known at deployment
            for v in y[:t_dep]:
                self.detector.update(v)
        self.now = t_dep

    @property
    def done(self) -> bool:
        return self.now >= self.end

    def advance(self, hours: int) -> list[dict]:
        """Read the next `hours` values (at most up to the end of the last forecast day)."""
        y, t_dep = self.s["y"], self.s["t_deploy"]
        events = []
        for i in range(self.now, min(self.now + max(0, int(hours)), self.end)):
            if (i - t_dep) % DAY == 0:  # an origin: decision (if due) and the forecast of the next day
                t0 = time.perf_counter()
                rec = self.run.step(i, np.asarray(self.alarms, dtype=int))
                if rec is not None:
                    events.append(dict(type="decision", index=i, seconds=time.perf_counter() - t0, **rec))
                self.reference.step(i)
                rows = slice(i, i + DAY)
                events.append(dict(type="forecast", index=i, forecast=self.run.pred[rows].tolist(),
                                   forecast_nothing=self.reference.pred[rows].tolist()))
            p, q = float(self.run.pred[i]), float(self.reference.pred[i])
            events.append(dict(type="observation", index=i, value=float(y[i]), forecast=p, forecast_nothing=q,
                               error=abs(y[i] - p) / self.scale, error_nothing=abs(y[i] - q) / self.scale))
            if self.detector is not None:
                alarm = self.detector.update(y[i])
                if alarm is not None:
                    self.alarms.append(i)
                    events.append(dict(type="alarm", method=self.detector.method, **alarm))
            self.now = i + 1
        return events

    def mase(self, start: int | None = None) -> tuple[float, float]:
        """MASE of the strategy and of `nothing` on the values read so far from `start` (default: deployment)."""
        start = self.s["t_deploy"] if start is None else start
        rows = np.arange(start, self.now)
        if len(rows) == 0:
            return float("nan"), float("nan")
        y = self.s["y"]
        return (float(np.mean(np.abs(y[rows] - self.run.pred[rows])) / self.scale),
                float(np.mean(np.abs(y[rows] - self.reference.pred[rows])) / self.scale))
