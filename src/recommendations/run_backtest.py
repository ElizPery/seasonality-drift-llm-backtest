"""
Stream B (RQ3) runner: base model validation, reference adaptation strategies and the hindsight oracles
on synthetic UCI series, ENTSO-E national load and seismic noise.

    python -m src.recommendations.run_backtest --tune              # base model validation
    python -m src.recommendations.run_backtest                     # all sources (about 40 min on 16 cores)
    python -m src.recommendations.run_backtest --sources seismic   # recompute one source, keep the other rows
    python -m src.recommendations.run_backtest --report            # rebuild the summaries without recomputing

Results go to data/processed/stream_b/ as CSV; the figures are drawn in notebooks/02_stream_b_baselines.ipynb.
Set OMP_NUM_THREADS=1: every process trains its own models, and extra threads only compete for the same cores.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from src.recommendations.backtester import STRATEGIES, backtest, oracle_picks, run_backtests, sort_runs, summarize, tune_base_model
from src.recommendations.data_sources import entsoe_series, seismic_series
from src.recommendations.forecast import LGBM_PARAMS, WEEK, mase_scale
from src.recommendations.plots import example_frame
from src.recommendations.synthetic import SYN_DEPLOY_WEEK, fit_models, synthetic_series

ENTSOE_FILE = Path("data/raw/opsd/time_series_60min_singleindex.csv")
SEISMIC_DIR = Path("data/raw/seismic_noise")
OUT_DIR = Path("data/processed/stream_b")
SOURCES = ("entsoe", "seismic", "synthetic")
TUNE_CONFIGS = {
    "A: 200 rounds, lr 0.05, 31 leaves": {},
    "B: 400 rounds, lr 0.05, 63 leaves": dict(rounds=400, num_leaves=63),
    "C: 300 rounds, lr 0.03, 15 leaves": dict(rounds=300, learning_rate=0.03, num_leaves=15),
    "D: 200 rounds, lr 0.10, 31 leaves": dict(learning_rate=0.10),
}


def run_tune(models, real: bool, processes: int | None, out: Path):
    """Validation only: other control seeds, and 2019 for ENTSO-E (the test period is 2020)."""
    tune = synthetic_series(models(), [900, 901], scenarios=(("none", 0.0),))
    for s in tune:
        s["t_stop"] = (SYN_DEPLOY_WEEK + 6) * WEEK
    if real:
        tune += entsoe_series(str(ENTSOE_FILE), start="2017-01-02", deploy="2019-09-02", end="2019-12-01 23:00")
    configs = {k: dict(LGBM_PARAMS, **v) for k, v in TUNE_CONFIGS.items()}
    res = tune_base_model(tune, configs, processes)
    res.to_csv(out / "tune.csv", index=False)
    print(res.groupby(["source", "config"]).agg(mase=("mase", "mean"), mase_naive=("mase_naive", "mean"),
                                                series=("series", "nunique")).round(3).to_string())


def write_examples(models, out: Path, real: bool, seismic: bool, recompute: set[str]):
    """Forecasts of three strategies around the change for one series of each source (synthetic level shift,
    Portugal 2020, Milan seismic noise 2020). Recomputed only for the sources in `recompute` or when missing."""

    def build(s, before, after):
        preds = {st.name: backtest(s["y"], s["F"], s["t_deploy"], st, s["base_features"])["pred"]
                 for st in STRATEGIES if st.name in ("nothing", "scheduled", "oracle")}
        return example_frame(s["y"], preds, s["t_change"], mase_scale(s["y"][: s["t_deploy"]]), before, after)

    csv = out / "example_level_shift.csv"
    if "synthetic" in recompute or not csv.exists():
        m = max(models(), key=lambda m: m.amplitude)  # the client with the largest seasonal amplitude
        df = build([x for x in synthetic_series([m], [0]) if x["scenario"] == "level_shift"][0], 7, 21)
        df.insert(0, "client", m.client)
        df.to_csv(csv, index=False)
    csv = out / "example_entsoe_PT.csv"
    if real and ("entsoe" in recompute or not csv.exists()):
        build(entsoe_series(str(ENTSOE_FILE), countries=("PT",))[0], 14, 35).to_csv(csv, index=False)
    csv = out / "example_seismic_MILN.csv"  # its MASE values are close to the 15-station means
    if seismic and ("seismic" in recompute or not csv.exists()):
        build(seismic_series(str(SEISMIC_DIR), stations=("IV.MILN",))[0], 14, 35).to_csv(csv, index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--clients", type=int, default=20, help="UCI clients for the synthetic series")
    p.add_argument("--seeds", type=int, default=3, help="control series per client")
    p.add_argument("--processes", type=int, default=None)
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--tune", action="store_true", help="validate base-model settings instead of running the strategies")
    p.add_argument("--no-real", action="store_true", help="skip the real series (ENTSO-E and seismic noise)")
    p.add_argument("--sources", default=None,
                   help="comma-separated subset of entsoe,seismic,synthetic to recompute; rows of the other sources in runs.csv are kept")
    p.add_argument("--report", action="store_true", help="rebuild the summaries from runs.csv without recomputing")
    args = p.parse_args()

    t = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = {}

    def models():
        if "m" not in cache:
            cache["m"] = fit_models(args.clients)
        return cache["m"]

    real = ENTSOE_FILE.exists() and not args.no_real
    seismic = SEISMIC_DIR.exists() and not args.no_real
    todo = set(args.sources.split(",")) if args.sources else set(SOURCES)
    if todo - set(SOURCES):
        raise SystemExit(f"unknown sources: {', '.join(sorted(todo - set(SOURCES)))}")
    if args.tune:
        run_tune(models, real, args.processes, out)
        return
    if not args.report:
        # the long ENTSO-E series go first, so they run alongside the shorter ones
        series = entsoe_series(str(ENTSOE_FILE)) if real and "entsoe" in todo else []
        series += seismic_series(str(SEISMIC_DIR)) if seismic and "seismic" in todo else []
        series += synthetic_series(models(), list(range(args.seeds))) if "synthetic" in todo else []
        if not series:
            raise SystemExit("no series to compute: the data of the listed sources is missing")
        runs = run_backtests(series, args.processes, log=sys.stderr)
        if args.sources and (out / "runs.csv").exists():  # only the listed sources were recomputed: keep the others
            prev = pd.read_csv(out / "runs.csv", float_precision="round_trip")  # unchanged rows stay byte-identical
            runs = sort_runs(pd.concat([prev[~prev.source.isin(runs.source.unique())], runs], ignore_index=True))
        runs.to_csv(out / "runs.csv", index=False)
    runs = pd.read_csv(out / "runs.csv")
    summ = summarize(runs)
    summ.to_csv(out / "summary.csv", index=False)
    oracle_picks(runs).to_csv(out / "oracle_picks.csv", index=False)
    write_examples(models, out, real, seismic, recompute=set() if args.report else todo)
    print(f"done in {time.time() - t:.0f} s -> {out}")
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
