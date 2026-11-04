"""DriftGenerator: control series without changes and changes injected into them."""
import numpy as np
import pandas as pd
import pytest

from src.detector.drift_generator import DriftGenerator

DAY, WEEK = 24, 168


def hourly_frame(weeks: int = 8, seed: int = 0) -> pd.DataFrame:
    """Hourly series starting on a Monday: a daily shape, lower weekends, noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(weeks * WEEK)
    daily = 10 * np.sin(2 * np.pi * (t % DAY) / DAY)
    weekend = np.where((t // DAY) % 7 >= 5, -15.0, 0.0)
    value = 100 + daily + weekend + rng.normal(0, 2, len(t))
    return pd.DataFrame({"timestamp": pd.date_range("2013-09-02", periods=len(t), freq="1h"), "value": value})


def test_old_calls_still_work():
    df = hourly_frame()
    gen = DriftGenerator(df, seasonality_period=24)
    base = gen.generate_level1_baseline()
    assert len(base) == len(df) and base["timestamp"].equals(df["timestamp"])
    out, meta = gen.inject_level2_drift(drift_type="abrupt", change_point_idx=len(df) // 2, magnitude_sigma=2.0)
    assert {"drift_type", "change_point_idx", "change_point_timestamp", "magnitude_sigma", "absolute_delta"} <= set(meta)
    assert np.allclose(out["value"].to_numpy()[len(df) // 2:] - df["value"].to_numpy()[len(df) // 2:], 2.0 * gen.sigma)


def test_control_is_reproducible_and_keeps_the_weekly_profile():
    gen = DriftGenerator(hourly_frame(), seasonality_period=24, periods=(24, WEEK))
    a = gen.generate_level1_baseline(n=24 * WEEK, seed=1)
    assert a.equals(gen.generate_level1_baseline(n=24 * WEEK, seed=1))
    assert not a["value"].equals(gen.generate_level1_baseline(n=24 * WEEK, seed=2)["value"])
    assert len(a) == 24 * WEEK and a["timestamp"].iloc[-1] == pd.Timestamp("2013-09-02") + pd.Timedelta(hours=24 * WEEK - 1)
    how = a["value"].to_numpy().reshape(24, WEEK).mean(axis=0)
    assert np.corrcoef(how, gen.profile)[0, 1] > 0.98
    weekday, weekend = how[: 5 * DAY].mean(), how[5 * DAY:].mean()
    assert weekday - weekend > 12  # the weekend drop of the source series is kept


@pytest.mark.parametrize("drift_type", ["abrupt", "gradual", "seasonal_amplitude", "variance"])
def test_change_goes_into_the_control_series_after_the_change_point(drift_type):
    gen = DriftGenerator(hourly_frame(), seasonality_period=24, periods=(24, WEEK))
    base = gen.generate_level1_baseline(n=24 * WEEK, seed=1)
    cp, mag = 13 * WEEK, 2.0
    out, meta = gen.inject_level2_drift(drift_type, cp, mag, base=base, seed=3)
    added = out["value"].to_numpy() - base["value"].to_numpy()
    assert meta["change_point_idx"] == cp and np.all(added[:cp] == 0)
    delta = mag * gen.sigma
    if drift_type == "abrupt":
        assert np.allclose(added[cp:], delta)
    elif drift_type == "gradual":
        assert np.allclose(added[cp + WEEK - 1:], delta) and np.all(np.diff(added[cp:cp + WEEK]) > 0)
    elif drift_type == "seasonal_amplitude":
        assert np.isclose(added[cp:cp + WEEK].std(), delta)
    else:
        noise_after = added[cp:]
        assert np.isclose(noise_after.std() / gen.sigma, np.sqrt(mag**2 - 1), rtol=0.25)


def test_a_single_period_series_works():
    rng = np.random.default_rng(0)
    t = np.arange(40 * 12)
    df = pd.DataFrame({"timestamp": pd.date_range("1980-01-01", periods=len(t), freq="MS"),
                       "value": 50 + 5 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, len(t))})
    gen = DriftGenerator(df, seasonality_period=12, seed=0)
    assert gen.cycle == gen.block == 12 and len(gen.residuals) % 12 == 0
    base = gen.generate_level1_baseline()
    out, _ = gen.inject_level2_drift("gradual", 240, 1.0, base=base)
    assert len(out) == len(df) and np.allclose(out["value"].to_numpy()[252:] - base["value"].to_numpy()[252:], gen.sigma)


def test_a_series_shorter_than_two_cycles_is_rejected():
    with pytest.raises(ValueError):
        DriftGenerator(hourly_frame(weeks=1), seasonality_period=24, periods=(24, WEEK))
