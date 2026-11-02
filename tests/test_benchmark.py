"""Tests of the common benchmark: the change types, the magnitude grid and the manifest read by stream A."""
import json

import numpy as np
import pandas as pd

from src.data.benchmark import MAGNITUDES, benchmark_manifest, grid_scenarios, level_of, write_benchmark
from src.data.frames import series_file
from src.recommendations.forecast import DAY, WEEK
from src.recommendations.synthetic import KINDS, Change, SeasonalModel, inject, make_control
from tests.test_alarms import series_dict
from tests.test_backtester import N, T_CHANGE, T_DEPLOY, toy


def model():
    rng = np.random.default_rng(1)
    how = np.arange(WEEK)
    profile = 20 * np.sin(2 * np.pi * how / DAY) + 8 * np.sin(2 * np.pi * how / WEEK)
    resid = rng.normal(0, 3, 12 * WEEK)
    return SeasonalModel(client="toy", profile=profile - profile.mean(), level=100.0, resid=resid, sigma=float(resid.std(ddof=1)))


def test_every_change_type_starts_at_t0_with_the_stated_magnitude():
    m = model()
    x = make_control(m, N, seed=0)
    for kind in KINDS:
        y = inject(x, m, Change(kind, 2.0, T_CHANGE), seed=0)
        assert np.array_equal(y[:T_CHANGE], x[:T_CHANGE]), kind
        assert not np.array_equal(y[T_CHANGE:], x[T_CHANGE:]), kind
    added = inject(x, m, Change("seasonal_amplitude", 2.0, T_CHANGE))[T_CHANGE:] - x[T_CHANGE:]
    assert abs(added.std() / m.sigma - 2.0) < 0.05  # the added seasonal part has the stated sigma scale
    assert abs(np.corrcoef(added, m.profile[np.arange(T_CHANGE, N) % WEEK])[0, 1] - 1) < 1e-9  # and the profile's shape


def test_grid_has_every_type_at_every_magnitude_except_variance_below_one():
    grid = grid_scenarios()
    assert grid[0] == ("none", 0.0) and len(grid) == 1 + 3 * len(MAGNITUDES) + 2
    assert ("variance", 0.5) not in grid and ("variance", 2.0) in grid and ("seasonal_amplitude", 0.5) in grid
    assert grid_scenarios(("level_shift",), (2.0,)) == (("none", 0.0), ("level_shift", 2.0))


def test_manifest_levels_and_fields_match_the_stream_a_scripts(tmp_path):
    y, F = toy(shift=30)
    control = series_dict(y, F, scenario="none", magnitude=0.0)
    shift = series_dict(y, F)
    real = series_dict(y, F, source="entsoe", series="IT", scenario="lockdown_2020", magnitude=np.nan,
                       start="2017-01-02 00:00:00+00:00")
    series = [control, shift, real]
    assert [level_of(s) for s in series] == [1, 2, 3]
    manifest = write_benchmark(series, tmp_path)
    back = json.load(open(tmp_path / "benchmark_manifest.json"))
    assert back == manifest and [e["level"] for e in back] == [1, 2, 3]
    c, s, r = back
    assert c["drift_type"] == "none" and c["change_point_idx"] is None and c["magnitude_sigma"] == 0.0
    assert s["drift_type"] == "abrupt" and s["change_point_idx"] == T_CHANGE and s["magnitude_sigma"] == 2.0
    assert r["drift_type"] == "lockdown_2020" and r["change_point_indices"] == [T_CHANGE] and r["magnitude_sigma"] is None
    assert all(e["seasonality_period"] == 24 and e["deploy_idx"] == T_DEPLOY for e in back)
    assert all((tmp_path / e["filename"]).exists() for e in back)
    frame = pd.read_csv(tmp_path / s["filename"])
    assert list(frame.columns) == ["timestamp", "value"] and len(frame) == N
    assert pd.read_csv(tmp_path / "manifest.csv")["file"].tolist() == [series_file(x) for x in series]
    assert benchmark_manifest(series, [series_file(x) for x in series]) == manifest
