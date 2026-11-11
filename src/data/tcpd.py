"""Series of the Turing Change Point Dataset (van den Burg and Williams, 2020; github.com/alan-turing-institute/TCPD)
with the human annotations: `data/raw/tcpd/<name>.csv` and `data/raw/tcpd/annotations.json`."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


TCPD_DIR = Path("data/raw/tcpd")


def load_tcpd_series(name: str = "jfk_passengers", frequency: str = "1min", raw_dir: str | Path = TCPD_DIR) -> pd.DataFrame:
    """Standard frame of one TCPD series; a nominal calendar with step `frequency` when the file has no timestamps."""
    file_path = Path(raw_dir) / f"{name}.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"TCPD dataset file not found at {file_path}")

    if "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range(start="2020-01-01", periods=len(df), freq=frequency)
    if "value" not in df.columns:
        # Fallback to the first numeric column if 'value' is missing
        numeric_cols = df.select_dtypes(include="number").columns
        df["value"] = df[numeric_cols[0]]
    return df[["timestamp", "value"]]


def load_tcpd_annotated_series(name: str = "jfk_passengers", frequency: str = "1min", annotator_key: str | None = None,
                               raw_dir: str | Path = TCPD_DIR) -> tuple[pd.DataFrame, list[int]]:
    """The series and the change point positions of one annotator (`annotator_key`), or the union of all
    annotators when no key is given."""
    df = load_tcpd_series(name, frequency, raw_dir)

    annotation_path = Path(raw_dir) / "annotations.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotations file not found at {annotation_path}")
    with open(annotation_path, "r") as f:
        all_annotations = json.load(f)
    if name not in all_annotations:
        raise ValueError(f"Dataset '{name}' not found in annotations.json")
    dataset_annotations = all_annotations[name]

    if annotator_key and annotator_key in dataset_annotations:
        ground_truth_cp = dataset_annotations[annotator_key]
    else:
        cp_set = set()
        for key, points in dataset_annotations.items():
            if isinstance(points, list):
                cp_set.update(points)
        ground_truth_cp = sorted(cp_set)
    return df, ground_truth_cp
