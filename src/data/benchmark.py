"""The common benchmark of both streams: the series of stream B with the full grid of injected changes, written
as standard frames with the manifest of stream A (`benchmark_manifest.json`, the file read by
`run_baseline_evaluation` and `run_residual_detector_evaluation`) and the manifest of stream B (`manifest.csv`).

Levels: 1 = synthetic control series without changes; 2 = control series with one injected change of a known
type (abrupt level shift, one-week ramp, noise x2, larger seasonal amplitude) and magnitude in residual sigmas;
3 = real series with the dated measures of March 2020 (ENTSO-E national load, seismic noise).

    python -m src.data.benchmark                   # data/processed/benchmark/: 60 + 840 + 23 series
    python -m src.data.benchmark --magnitudes 2 --kinds level_shift,trend_ramp,variance   # the 263 series of the stream B backtests
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.frames import series_file, series_id, write_series
from src.recommendations.synthetic import KINDS

MAGNITUDES = (0.5, 1.0, 2.0, 3.0)  # for the variance type a multiplier of the residual sigma, so only values above 1 apply
DRIFT_TYPE = {"none": "none", "level_shift": "abrupt", "trend_ramp": "gradual", "variance": "variance",
              "seasonal_amplitude": "seasonal_amplitude"}
PERIODS = (24, 168)


def grid_scenarios(kinds=KINDS, magnitudes=MAGNITUDES) -> tuple[tuple[str, float], ...]:
    """The control scenario plus every change type at every magnitude (variance only at multipliers above 1)."""
    out = [("none", 0.0)]
    for k in kinds:
        for m in magnitudes:
            if k != "variance" or m > 1.0:
                out.append((k, float(m)))
    return tuple(out)


def level_of(s: dict) -> int:
    if s["source"] != "synthetic":
        return 3
    return 1 if s["scenario"] == "none" else 2


def benchmark_manifest(series: list[dict], files: list[str]) -> list[dict]:
    """One entry per series in the format of the stream A scripts: `dataset_name`, `level`, `filename`,
    `drift_type`, `magnitude_sigma`, `change_point_idx` (levels 2 and 3), `change_point_indices` (level 3),
    `seasonality_period`, plus `id`, `deploy_idx` and the seed for the link with stream B."""
    out = []
    for s, file in zip(series, files):
        level = level_of(s)
        t_chg = s.get("t_change")
        start = s["start"]
        entry = dict(id=series_id(s), dataset_name=f"{s['source']}_{s['client'] if s['source'] == 'synthetic' else s['series']}",
                     level=level, filename=file, drift_type=DRIFT_TYPE.get(s["scenario"], s["scenario"]),
                     magnitude_sigma=None if level == 3 else float(s["magnitude"]),
                     change_point_idx=None if level == 1 or t_chg is None else int(t_chg),
                     seasonality_period=PERIODS[0], seasonality_periods=list(PERIODS), deploy_idx=int(s["t_deploy"]),
                     seed=int(s["seed"]), n=len(s["y"]), start=str(start))
        if level == 3:
            entry["change_point_indices"] = [] if t_chg is None else [int(t_chg)]
        out.append(entry)
    return out


def write_benchmark(series: list[dict], out_dir: Path) -> list[dict]:
    """The series as standard-frame CSVs, `manifest.csv` (stream B) and `benchmark_manifest.json` (stream A)."""
    out_dir = Path(out_dir)
    write_series(series, out_dir)
    manifest = benchmark_manifest(series, [series_file(s) for s in series])
    with open(out_dir / "benchmark_manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)
    return manifest


def main():
    from src.data.entsoe import ENTSOE_FILE
    from src.data.seismic_noise import SEISMIC_DIR
    from src.recommendations.data_sources import entsoe_series, seismic_series
    from src.recommendations.synthetic import fit_models, synthetic_series

    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--out", default="data/processed/benchmark")
    p.add_argument("--clients", type=int, default=20)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--kinds", default=",".join(KINDS), help="comma-separated change types")
    p.add_argument("--magnitudes", default=",".join(str(m) for m in MAGNITUDES), help="comma-separated magnitudes in residual sigmas")
    p.add_argument("--no-real", action="store_true", help="skip the real series (level 3)")
    args = p.parse_args()

    kinds = tuple(args.kinds.split(","))
    magnitudes = tuple(float(m) for m in args.magnitudes.split(","))
    series = [] if args.no_real else entsoe_series(str(ENTSOE_FILE)) + seismic_series(str(SEISMIC_DIR))
    series += synthetic_series(fit_models(args.clients), list(range(args.seeds)), grid_scenarios(kinds, magnitudes))
    manifest = write_benchmark(series, Path(args.out))
    counts = {}
    for e in manifest:
        counts[(e["level"], e["drift_type"])] = counts.get((e["level"], e["drift_type"]), 0) + 1
    for (level, kind), n in sorted(counts.items()):
        print(f"level {level} {kind}: {n}")
    print(f"{len(manifest)} series written to {args.out}")


if __name__ == "__main__":
    main()
