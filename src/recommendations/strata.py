"""Strata of series by seasonal strength, to report the stream B results per group (the supervisor's advice:
a method may help on strongly seasonal series and not on weakly seasonal ones).

Strength of a seasonal component (Wang, Smith and Hyndman 2006; Hyndman and Athanasopoulos, FPP3, 4.3):
F = max(0, 1 - Var(R) / Var(S + R)) from an MSTL decomposition with daily and weekly periods; 0 means no
seasonality, values near 1 a series that is almost all seasonal pattern. It is computed on the last 8 weeks
before deployment, so the stratum of a series is known before any decision is taken.

Groups are the thirds of the total strength over all series. The series were chosen for their seasonality and
the daily pattern is strong in almost all of them, so fixed thresholds (0.6, 0.85) put 212 of 263 series in
one group; the thirds were chosen from the distribution of the strength alone, not from results per group.
The real series fall in the lower two thirds, so the groups are also compared within the synthetic series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL

from src.recommendations.forecast import DAY, WEEK

STRENGTH_WEEKS = 8
GROUPS = ("low", "medium", "high")  # thirds of the total seasonal strength


def seasonal_strength(y: np.ndarray, t_deploy: int, weeks: int = STRENGTH_WEEKS) -> dict:
    x = y[t_deploy - weeks * WEEK: t_deploy]
    res = MSTL(x, periods=(DAY, WEEK), stl_kwargs={"robust": True}).fit()
    s = np.asarray(res.seasonal)
    r = np.asarray(res.resid)

    def f(comp):
        return float(max(0.0, 1.0 - np.var(r) / np.var(comp + r)))

    return dict(strength_daily=f(s[:, 0]), strength_weekly=f(s[:, 1]), strength=f(s.sum(axis=1)))


def strata_table(series: list[dict]) -> pd.DataFrame:
    """Seasonal strength and group of every series (one row per source, series and scenario)."""
    rows = [dict(source=s["source"], series=s["series"], scenario=s["scenario"], **seasonal_strength(s["y"], s["t_deploy"]))
            for s in series]
    return assign_groups(pd.DataFrame(rows))


def assign_groups(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(group=pd.qcut(df.strength, 3, labels=GROUPS).astype(str))
