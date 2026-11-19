"""Experiments of stream A on one evaluation function (`src.detector.common_evaluation`).

    python -m src.detector.run_common_evaluation benchmark   # generated UCI series: all detectors, multipliers, scores
    python -m src.detector.run_common_evaluation real        # 23 real hourly series with a documented change
    python -m src.detector.run_common_evaluation timing      # run time of every detector on one series
    python -m src.detector.run_common_evaluation report      # summary tables and the figure from the stored results
    python -m src.detector.run_common_evaluation             # all four steps

The results are written to data/processed/common_evaluation:

* benchmark.csv: five UCI clients (the batch of the comparison notebook: seed 42, the year 2013, hourly). For each
  client three series without changes (level 1, block bootstrap with the seeds 1000, 1001, 1002) and twelve series
  with one injected change in the middle (level 2: abrupt, gradual, seasonal amplitude at 0.5, 1, 2 and 3 sigma).
  Every baseline detector and the residual detector at the multipliers MULTIPLIERS are scored by `score_alarms`
  with a tolerance window of 40 points. KSWIN has one row per seed (column `seed`), the summary averages over them.
* scores.csv: the residual detector on the same series with all three scores and with one score at a time, at its
  default multiplier; for a detected change, the scores that were violated at the detecting alarm.
* real.csv: the national load of 8 countries (ENTSO-E) and the seismic noise of 15 stations, with the date of the
  first lockdown measures as the change (the series and the manifest of `python -m src.data.export`). The detectors
  run on the whole series; the alarms are counted from the deployment point to the end of the series. `far_pre` is
  the alarms per 1000 hours between deployment and change, a detection is an alarm within 168 hours after the
  change, `week_after` is the number of alarms in the 168 hours after the change. The residual detector also runs
  at the multipliers REAL_MULTIPLIERS.
* timing.csv: seconds per run on one series of 8760 points, three repeats.

The benchmark step takes about 20 minutes on 16 cores and the real series about 6 minutes: most of the time goes
into the STL fits and the loop of the residual detector.
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import DataLoader
from src.detector.common_evaluation import (BASELINES, SCORES, TOLERANCE, baseline_alarms, baseline_runs,
                                            causal_residuals, detector_params, mean_over_seeds, residual_alarms,
                                            score_alarms, score_violations)
from src.detector.drift_generator import DriftGenerator
from src.detector.residual_detector import ResidualConformalDriftDetector

OUT_DIR = Path("data/processed/common_evaluation")
MULTIPLIERS = (1.2, 1.4, 1.6, 2.0, 2.5, 3.0)
REAL_MULTIPLIERS = (3.0, 5.0, 8.0)
DRIFT_TYPES = ("abrupt", "gradual", "seasonal_amplitude")
MAGNITUDES = (0.5, 1.0, 2.0, 3.0)
DRAWS = 3
REAL_SOURCES = ("entsoe", "seismic")
REAL_TOLERANCE = 168  # hours: a detection within seven days after the first measures
VARIANTS = {"full": SCORES, "point": ("point",), "mean": ("mean",), "spread": ("spread",)}


def benchmark_series(data_dir: str | Path = "data", n_series: int = 5, draws: int = DRAWS) -> list[dict]:
    """The generated series: per UCI client `draws` series without changes and 12 series with one injected change."""
    batch = DataLoader(data_dir=data_dir).load_uci_batch(n_series=n_series, resample_freq="1h", seed=42,
                                                         check_window=("2013-01-01", "2013-12-31"))
    out = []
    for series, (df, period) in batch.items():
        generator = DriftGenerator(df, seasonality_period=period)
        for draw in range(draws):
            np.random.seed(1000 + draw)  # the block bootstrap of the generator draws from the global state of NumPy
            out.append(dict(series=series, level=1, kind="none", mag=0.0, draw=draw, change=None, period=period,
                            df=generator.generate_level1_baseline()))
        change = len(df) // 2
        for kind in DRIFT_TYPES:
            for mag in MAGNITUDES:
                injected, _ = generator.inject_level2_drift(kind, change, mag)
                out.append(dict(series=series, level=2, kind=kind, mag=mag, draw=0, change=change, period=period, df=injected))
    return out


def benchmark_job(job: dict) -> tuple[list[dict], list[dict]]:
    """Rows of benchmark.csv and of scores.csv for one generated series."""
    df, period, change = job["df"], job["period"], job["change"]
    labels = {k: job[k] for k in ("series", "level", "kind", "mag", "draw")}
    residuals = causal_residuals(df, period)
    rows = []
    for mult in MULTIPLIERS:
        alarms = residual_alarms(df, period, residuals, multiplier=mult)
        rows.append(dict(labels, method="residual_conformal", mult=mult, seed=np.nan, **score_alarms(alarms, len(df), change)))
    for method, seed, alarms in baseline_runs(df):
        rows.append(dict(labels, method=method, mult=np.nan, seed=seed, **score_alarms(alarms, len(df), change)))
    detector = ResidualConformalDriftDetector(seasonality_period=period, **detector_params(period, len(df)))
    by_score = []
    for variant, scores in VARIANTS.items():
        alarms = residual_alarms(df, period, residuals, scores=scores)
        row = dict(labels, variant=variant, **score_alarms(alarms, len(df), change))
        if variant == "full" and change is not None and row["recall"]:
            row.update(zip(SCORES, score_violations(detector, residuals, change + int(row["delay"]))))
        by_score.append(row)
    return rows, by_score


def run_benchmark(out_dir: Path, data_dir: str | Path = "data", processes: int | None = None) -> None:
    jobs = benchmark_series(data_dir)
    print(f"benchmark: {len(jobs)} generated series", flush=True)
    with Pool(processes) as pool:
        results = pool.map(benchmark_job, jobs)
    keys = ["series", "level", "kind", "mag", "draw"]
    benchmark = pd.DataFrame([r for rows, _ in results for r in rows]).sort_values(keys + ["method", "mult", "seed"])
    scores = pd.DataFrame([r for _, rows in results for r in rows]).sort_values(keys + ["variant"])
    benchmark.to_csv(out_dir / "benchmark.csv", index=False)
    scores.to_csv(out_dir / "scores.csv", index=False)


def real_rows(alarms, row, method: str, mult: float = np.nan, seed: float = np.nan) -> dict:
    """Alarms of one detector on one real series, counted from deployment to the end of the series."""
    start, change, stop = int(row["t_deploy"]), int(row["t_change"]), int(row["t_stop"])
    alarms = np.array(sorted(i for i in alarms if start <= i < stop), dtype=int)
    before, after = alarms[alarms < change], alarms[alarms >= change]
    scored = score_alarms(alarms, stop - start, change, tolerance=REAL_TOLERANCE)
    return dict(source=row["source"], series=row["series"], method=method, mult=mult, seed=seed,
                far_pre=1000 * len(before) / (change - start), recall=scored["recall"], precision=scored["precision"],
                f1=scored["f1"], delay=scored["delay"], first_after=float(after[0] - change) if len(after) else np.nan,
                alarms=len(alarms), week_after=int(((after - change) < REAL_TOLERANCE).sum()), pre_hours=change - start)


def real_job(row: dict) -> list[dict]:
    """Rows of real.csv for one real series (hourly, daily period); `row` is its row of the manifest with `path`."""
    frame = pd.read_csv(row["path"])
    df = pd.DataFrame({"timestamp": frame["timestamp"], "value": frame["value"]})
    period = 24
    residuals = causal_residuals(df, period, dataset=row["source"])
    default = detector_params(period, len(df), row["source"])["sensitivity_multiplier"]
    out = []
    for mult in (default,) + REAL_MULTIPLIERS:
        alarms = residual_alarms(df, period, residuals, multiplier=mult, dataset=row["source"])
        out.append(real_rows(alarms, row, "residual_conformal", mult))
    for method, seed, alarms in baseline_runs(df):
        out.append(real_rows(alarms, row, method, seed=seed))
    return out


def run_real(out_dir: Path, data_dir: str | Path = "data", processes: int | None = None) -> None:
    series_dir = Path(data_dir) / "processed" / "series"
    if not (series_dir / "manifest.csv").exists():
        raise FileNotFoundError(f"{series_dir / 'manifest.csv'} not found: write the series with python -m src.data.export")
    manifest = pd.read_csv(series_dir / "manifest.csv")
    rows = [dict(r, path=str(series_dir / r["file"])) for r in manifest[manifest["source"].isin(REAL_SOURCES)].to_dict("records")]
    print(f"real: {len(rows)} series", flush=True)
    with Pool(processes) as pool:
        results = pool.map(real_job, rows)
    real = pd.DataFrame([r for rows in results for r in rows]).sort_values(["source", "series", "method", "mult", "seed"])
    real.to_csv(out_dir / "real.csv", index=False)


def run_timing(out_dir: Path, data_dir: str | Path = "data", repeats: int = 3) -> None:
    """Run time of every detector on the first series with an abrupt change of 2 sigma (8760 points)."""
    job = next(j for j in benchmark_series(data_dir, n_series=1, draws=0) if j["kind"] == "abrupt" and j["mag"] == 2.0)
    df, period = job["df"], job["period"]
    rows = []
    for repeat in range(repeats):
        started = time.perf_counter()
        residual_alarms(df, period)
        rows.append(dict(method="residual_conformal", repeat=repeat, seconds=time.perf_counter() - started, n=len(df)))
        for config in BASELINES:
            started = time.perf_counter()
            baseline_alarms(df, config)
            rows.append(dict(method=config["method"], repeat=repeat, seconds=time.perf_counter() - started, n=len(df)))
    pd.DataFrame(rows).to_csv(out_dir / "timing.csv", index=False)


def benchmark_table(benchmark: pd.DataFrame) -> pd.DataFrame:
    """Per detector (and multiplier): false alarms per 1000 points on the series without changes (mean and standard
    deviation) and recall, precision, F1 and mean delay on the series with an injected change. The rows of KSWIN are
    averaged over its seeds per series first."""
    benchmark = mean_over_seeds(benchmark, ["series", "level", "kind", "mag", "draw", "method", "mult"])
    keys = ["method", "mult"]
    clean = benchmark[benchmark["level"] == 1].groupby(keys, dropna=False)["far"].agg(far="mean", far_std="std")
    drift = benchmark[benchmark["level"] == 2].groupby(keys, dropna=False)[["recall", "precision", "f1", "delay"]].mean()
    return clean.join(drift).reset_index()


def recall_by_type(table: pd.DataFrame, key: str) -> pd.DataFrame:
    """Recall by type of the injected change, one row per value of `key`."""
    drift = table[table["level"] == 2]
    return drift.pivot_table(index=key, columns="kind", values="recall", aggfunc="mean", dropna=False)


def real_table(real: pd.DataFrame) -> pd.DataFrame:
    """Per source, detector and multiplier: mean false alarms per 1000 hours before the change, the number of changes
    detected within seven days, the median delay, and `ratio`: the alarms in the week after the change divided by
    the alarms expected in a week at the rate before it (near 1: no reaction to the change). The rows of KSWIN are
    averaged over its seeds per series first, so its number of detected changes can be a fraction."""
    real = mean_over_seeds(real, ["source", "series", "method", "mult"])
    groups = real.groupby(["source", "method", "mult"], dropna=False)
    table = groups.agg(far_pre=("far_pre", "mean"), detected=("recall", "sum"), series=("recall", "size"),
                       delay=("delay", "median"), week_after=("week_after", "mean")).reset_index()
    expected = table["far_pre"] / 1000 * REAL_TOLERANCE
    table["ratio"] = (table["week_after"] / expected).where(expected > 0)
    return table


def scores_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Per variant of the residual detector: false alarms per 1000 points and recall by type of the change."""
    far = scores[scores["level"] == 1].groupby("variant")["far"].mean()
    return recall_by_type(scores, "variant").assign(far=far).loc[list(VARIANTS)]


def plot_far_recall(table: pd.DataFrame, path: Path) -> None:
    """False alarms against recall: the baselines as points, the residual detector as a line over its multipliers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"adwin": ("ADWIN", (6, 6), "left"), "kswin": ("KSWIN", (6, 6), "left"), "ks": ("KS test", (6, -14), "left"),
              "page_hinkley": ("Page-Hinkley", (-10, -16), "right"), "psi": ("PSI", (6, -14), "left"),
              "wasserstein": ("Wasserstein", (0, 10), "center")}
    blue = "#1f5fa8"
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    residual = table[table["method"] == "residual_conformal"].sort_values("mult")
    low, high = residual["mult"].min(), residual["mult"].max()
    ax.plot(residual["far"], residual["recall"], "-o", color=blue,
            label=f"Residual conformal, multiplier c from {low:g} to {high:g}")
    default = detector_params(24, 8760)["sensitivity_multiplier"]
    for mult, offset in ((low, (6, 8)), (default, (0, 12)), (high, (-14, -18))):
        r = residual[np.isclose(residual["mult"], mult)]
        if len(r):
            ax.annotate(f"c = {mult:g}", (r["far"].iloc[0], r["recall"].iloc[0]), textcoords="offset points",
                        xytext=offset, fontsize=8, color=blue)
    for _, r in table[table["method"] != "residual_conformal"].iterrows():
        name, offset, align = labels[r["method"]]
        ax.plot(r["far"], r["recall"], "o", color="#999999")
        ax.annotate(name, (r["far"], r["recall"]), textcoords="offset points", xytext=offset, ha=align, fontsize=8)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0, 1, 2, 5, 10, 20, 40], labels=["0", "1", "2", "5", "10", "20", "40"])
    ax.set_xlim(0, 60)
    ax.set_xlabel("False alarms per 1000 points (series without changes)")
    ax.set_ylabel("Recall on injected changes")
    ax.set_ylim(-0.03, 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def report(out_dir: Path) -> None:
    """Prints the summary tables from the stored results and draws far_recall.png."""
    pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_rows", 200)
    if (out_dir / "benchmark.csv").exists():
        benchmark = pd.read_csv(out_dir / "benchmark.csv")
        table = benchmark_table(benchmark)
        clean, drift = (benchmark["level"] == 1), (benchmark["level"] == 2)
        print(f"\nCommon evaluation: {benchmark[clean].groupby(['series', 'draw']).ngroups} series without changes, "
              f"{benchmark[drift].groupby(['series', 'kind', 'mag']).ngroups} injected changes, tolerance {TOLERANCE} points")
        print(table.round(3).to_string(index=False))
        kswin = benchmark[(benchmark["method"] == "kswin") & drift].groupby("seed")["recall"].mean()
        print(f"KSWIN: mean over {len(kswin)} seeds; recall by seed from {kswin.min():.2f} to {kswin.max():.2f}")
        print("\nRecall of the residual detector by type of change and multiplier")
        print(recall_by_type(benchmark[benchmark["method"] == "residual_conformal"], "mult").round(2).to_string())
        plot_far_recall(table, out_dir / "far_recall.png")
    if (out_dir / "scores.csv").exists():
        scores = pd.read_csv(out_dir / "scores.csv")
        print("\nResidual detector with one score at a time (far: false alarms per 1000 points)")
        print(scores_table(scores).round(2).to_string())
        detected = scores[(scores["variant"] == "full") & (scores["recall"] == 1)]
        print("\nScores violated at the detecting alarm of the full detector (number of detected changes)")
        fired = detected[list(SCORES)].astype(bool).groupby(detected["kind"]).sum()
        print(fired.assign(detected=detected.groupby("kind").size()).to_string())
    if (out_dir / "real.csv").exists():
        real = real_table(pd.read_csv(out_dir / "real.csv"))
        print(f"\nReal series: alarms before the change per 1000 hours, detections within {REAL_TOLERANCE} hours")
        print(real.round(2).to_string(index=False))
    if (out_dir / "timing.csv").exists():
        timing = pd.read_csv(out_dir / "timing.csv")
        print(f"\nRun time on one series of {int(timing['n'].iloc[0])} points, seconds (mean of the repeats)")
        print(timing.groupby("method")["seconds"].mean().round(3).to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("step", nargs="?", default="all", choices=["all", "benchmark", "real", "timing", "report"])
    p.add_argument("--data", default="data", help="data directory (raw/ and processed/ inside)")
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--processes", type=int, default=None)
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.step in ("all", "benchmark"):
        run_benchmark(out_dir, args.data, args.processes)
    if args.step in ("all", "real"):
        run_real(out_dir, args.data, args.processes)
    if args.step in ("all", "timing"):
        run_timing(out_dir, args.data)
    if args.step in ("all", "report"):
        report(out_dir)


if __name__ == "__main__":
    main()
