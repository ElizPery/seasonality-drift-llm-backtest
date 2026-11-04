"""Write the stream B series as standard frames, so the detectors of stream A run on the same series.

    python -m src.data.export                      # 263 series: ENTSO-E, seismic noise, synthetic UCI
    python -m src.data.export --clients 2 --seeds 1 --no-real --out data/processed/series_small

Output: one CSV per series (`timestamp`, `value`) and `manifest.csv` with the labels, the deployment point and
the change point of every series, in `--out` (default data/processed/series).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.entsoe import ENTSOE_FILE
from src.data.frames import write_series
from src.data.seismic_noise import SEISMIC_DIR
from src.recommendations.data_sources import entsoe_series, seismic_series
from src.recommendations.synthetic import fit_models, synthetic_series


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/processed/series")
    p.add_argument("--clients", type=int, default=20, help="UCI clients for the synthetic series")
    p.add_argument("--seeds", type=int, default=3, help="control series per client")
    p.add_argument("--no-real", action="store_true", help="skip the real series (ENTSO-E and seismic noise)")
    args = p.parse_args()

    series = []
    if not args.no_real:
        series += entsoe_series(str(ENTSOE_FILE)) + seismic_series(str(SEISMIC_DIR))
    series += synthetic_series(fit_models(args.clients), list(range(args.seeds)))
    table = write_series(series, Path(args.out))
    print(table.groupby("source").size().to_string())
    print(f"{len(table)} series written to {args.out}")


if __name__ == "__main__":
    main()
