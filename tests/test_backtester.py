"""Tests for stream B: features, action catalog, backtest invariants, bootstrap interval, seismic loader.
Run from the repository root: python -m pytest tests"""
import numpy as np
import pandas as pd

from src.recommendations.backtester import STRATEGIES, backtest, mase, mase_gain_ci
from src.recommendations.catalog import ACTIONS, CATALOG, apply_action
from src.recommendations.data_sources import seismic_series
from src.recommendations.forecast import BASE_FEATURES, DAY, WEEK, Model, calendar_from_index, feature_table, mase_scale

N, T_DEPLOY, T_CHANGE = 16 * WEEK, 8 * WEEK, 11 * WEEK


def toy(shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    y = 100 + 20 * np.sin(2 * np.pi * t / DAY) + 8 * np.sin(2 * np.pi * t / WEEK) + rng.normal(0, 3, N)
    y[T_CHANGE:] += shift
    hour, dow = calendar_from_index(N)
    return y, feature_table(y, hour, dow)


def test_features_do_not_see_the_forecast_day():
    y, F = toy()
    y2 = y.copy()
    y2[T_DEPLOY:] += 1000.0  # change everything from the origin on
    hour, dow = calendar_from_index(N)
    F2 = feature_table(y2, hour, dow)
    rows = np.arange(0, T_DEPLOY + DAY)  # the whole first forecast day must be unaffected
    for f in BASE_FEATURES:
        assert np.allclose(F[f][rows], F2[f][rows], equal_nan=True), f


def test_catalog_actions_run_and_retrain_count():
    y, F = toy(shift=15)
    model, _ = apply_action(ACTIONS["retrain_full"], Model(BASE_FEATURES), F, y, T_DEPLOY, BASE_FEATURES)
    for a in CATALOG:
        m, fits = apply_action(a, model, F, y, T_CHANGE + WEEK, BASE_FEATURES)
        assert fits == (1 if a.kind == "retrain" else 0)
        assert np.isfinite(m.predict(F, np.arange(T_CHANGE + WEEK, T_CHANGE + WEEK + DAY))).all()


def test_oracle_is_never_worse_than_scheduled_and_beats_nothing_after_a_shift():
    y, F = toy(shift=15)
    by = {s.name: s for s in STRATEGIES}
    scale = mase_scale(y[:T_DEPLOY])
    cache = {}
    r = {k: backtest(y, F, T_DEPLOY, by[k], BASE_FEATURES, cache=cache) for k in ("nothing", "scheduled", "oracle_weekly")}
    end = r["nothing"]["end"]
    m = {k: mase(y, v["pred"], T_DEPLOY, end, scale) for k, v in r.items()}
    assert m["oracle_weekly"] <= m["scheduled"] + 1e-9
    assert m["oracle_weekly"] < m["nothing"]
    assert r["scheduled"]["retrains"] == (end - T_DEPLOY) // DAY // 7 - (1 if ((end - T_DEPLOY) // DAY) % 7 == 0 else 0)


def test_mase_gain_ci_sign():
    y, _ = toy()
    biased = y + 5.0
    perfect = y.copy()
    g, lo, hi = mase_gain_ci(y, biased, perfect, T_DEPLOY, T_DEPLOY + 4 * WEEK, scale=mase_scale(y[:T_DEPLOY]))
    assert g > 0 and lo > 0 and lo <= g <= hi
    g, lo, hi = mase_gain_ci(y, perfect, biased, T_DEPLOY, T_DEPLOY + 4 * WEEK, scale=mase_scale(y[:T_DEPLOY]))
    assert g < 0 and hi < 0


def test_seismic_series_masks_gaps_and_keeps_the_local_daily_cycle(tmp_path):
    t = pd.date_range("2019-11-01", "2020-04-30 23:45", freq="15min")  # UTC, as in the published files
    local = t.tz_localize("UTC").tz_convert("Europe/Brussels")
    rms = np.where((local.hour >= 8) & (local.hour < 18), 1.5e-9, 1e-9)  # louder by day on the local clock
    rms[t.get_loc(pd.Timestamp("2020-01-15 14:45"))] = 2.7e-157  # placeholders of gaps in the published files
    rms[(t >= "2019-11-29 08:15") & (t <= "2019-11-29 08:45")] = 1.645e-13  # most of one hour
    keep = ~((t >= "2020-01-20 18:00") & (t < "2020-01-21 04:00"))  # a 10-hour night gap
    pd.DataFrame({"4.0-14.0": rms[keep]}, index=t[keep]).to_csv(tmp_path / "BE.UCC..HHZ.csv")
    pd.DataFrame({"Station_Code": ["BE.UCC"], "City": ["Brussels"], "TimeZone": ["Europe/Brussels"],
                  "Date_LD1": ["14/03/2020"]}).to_csv(tmp_path / "lockdown_stations.csv", index=False)
    (s,) = seismic_series(str(tmp_path), stations=("BE.UCC",))
    y, F = s["y"], s["F"]
    day = (F["hour"] >= 8) & (F["hour"] < 18)
    # log10 of the level in nm: the placeholders and the gap are gone, and the day stays on the same
    # local hours after the switch to summer time (29.03)
    assert np.allclose(y[day], np.log10(1.5)) and np.allclose(y[~day], 0.0)
    assert F["hour"][s["t_deploy"]] == 0 and F["dow"][s["t_deploy"]] == 0
    assert s["t_change"] - s["t_deploy"] == 40 * DAY
