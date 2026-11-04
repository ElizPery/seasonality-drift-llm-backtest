import importlib.util

import numpy as np
import pandas as pd
import pytest

from src.data import DataLoader, series_frame, series_table, write_series
from src.data.uci import WEEK, client_frame


def fake_series(n=3 * WEEK, start="2013-09-02 00:00"):
    return dict(source="synthetic", series="MT_001/s0", client="MT_001", seed=0, scenario="level_shift", magnitude=2.0,
                y=np.arange(n, dtype=float), F={}, t_deploy=WEEK, t_change=2 * WEEK, base_features=(), start=start)


def test_series_frame_and_manifest_row_share_one_calendar():
    s = fake_series()
    frame = series_frame(s)
    assert list(frame.columns) == ["timestamp", "value"] and len(frame) == 3 * WEEK
    assert frame["timestamp"].iloc[0] == pd.Timestamp("2013-09-02 00:00") and frame["timestamp"].iloc[0].weekday() == 0
    assert (frame["timestamp"].diff().dropna() == pd.Timedelta(hours=1)).all()
    row = series_table([s]).iloc[0]
    assert row["change_at"] == frame["timestamp"].iloc[s["t_change"]]
    assert row["deploy_at"] == frame["timestamp"].iloc[s["t_deploy"]]
    assert row["t_stop"] == 3 * WEEK and row["n"] == 3 * WEEK


def test_write_series_puts_every_series_and_the_manifest_in_the_folder(tmp_path):
    a, b = fake_series(), dict(fake_series(), scenario="none", magnitude=0.0, t_change=None)  # same client and seed
    table = write_series([a, b], tmp_path)
    files = ["synthetic__MT_001__s0__level_shift.csv", "synthetic__MT_001__s0__none.csv"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.csv"] + files
    assert list(table["file"]) == files and list(table["id"]) == ["synthetic/MT_001/s0/level_shift", "synthetic/MT_001/s0/none"]
    back = pd.read_csv(tmp_path / files[0], parse_dates=["timestamp"])
    assert np.array_equal(back["value"].to_numpy(), a["y"]) and back["timestamp"].iloc[-1] == pd.Timestamp("2013-09-22 23:00")
    manifest = pd.read_csv(tmp_path / "manifest.csv")
    assert manifest["t_change"].isna().tolist() == [False, True]


def test_client_frame_keeps_hours_and_averages_coarser_steps():
    idx = pd.date_range("2013-09-02", periods=48, freq="1h")
    hourly = pd.DataFrame({"MT_001": np.arange(48.0), "MT_002": np.ones(48)}, index=idx)
    hours = client_frame("MT_001", "1h", hourly)
    assert len(hours) == 48 and hours["value"].iloc[5] == 5.0 and list(hours.columns) == ["timestamp", "value"]
    days = client_frame("MT_001", "1D", hourly)
    assert len(days) == 2 and days["value"].tolist() == [11.5, 35.5]


def test_data_loader_keeps_the_stream_a_interface_without_seisbench(tmp_path):
    loader = DataLoader(tmp_path)
    assert (tmp_path / "raw").is_dir() and (tmp_path / "processed").is_dir()
    with pytest.raises(FileNotFoundError):
        loader.load_alibaba_standard_trace()
    with pytest.raises(FileNotFoundError):
        loader.load_uci_electricity()
    with pytest.raises(FileNotFoundError):
        loader.load_tcpd_series("jfk_passengers")
    if importlib.util.find_spec("seisbench") is None:
        with pytest.raises(ImportError):  # imported on first use only
            loader.load_seismic_dataset(seed=42)
