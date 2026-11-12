"""Report on the language model advisors (RQ3) on the 23 real series: paired tests against the reference
strategies, the actions the models choose, and the figure of error against retrainings.

    python -m src.recommendations.llm_report                 # tables and the figure from runs.csv and the answer cache
    python -m src.recommendations.llm_report --recalibrate   # first recompute the fixed weekly recalibration (needs the raw data)

A configuration is one advisor: a model, a prompt variant, the trigger (weekly or on the alarms of a detector)
and whether its advice is applied as proposed or checked by the acceptance rule. The error of a configuration
on a series is the mean over its repeats, so every series counts once. The tests are those of the statistical
comparison of the strategies: the paired Wilcoxon signed-rank test on the differences per series, zero
differences split between the signs, and the Holm correction over the configurations of one family.

Outputs: data/processed/stream_b/recalibrate_weekly.csv, llm_paired_tests.csv, llm_action_shares.csv and
docs/images/quality_cost_real_series.png.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.recommendations.catalog import CATALOG
from src.recommendations.llm_client import CACHE_DIR

OUT_DIR = Path("data/processed/stream_b")
FIGURE = Path("docs/images/quality_cost_real_series.png")
KEY = ["source", "series"]
REAL = ("entsoe", "seismic")
RECALIBRATE = "recalibrate_weekly"
MODEL = {"anthropic/claude-sonnet-5.5": "Claude Sonnet 5.5", "anthropic/claude-haiku-5.5": "Claude Haiku 5.5",
         "openai/gpt-5-mini": "GPT-5 mini", "google/gemini-2.5-flash": "Gemini 2.5 Flash",
         "mistralai/mistral-small-3.2-24b-instruct": "Mistral Small 3.2", "qwen/qwen3-235b-a22b-2507": "Qwen3 235B",
         "qwen/qwen3-14b": "Qwen3 14B"}


def recalibrate_weekly() -> pd.DataFrame:
    """Fixed reference without a language model: correct the level every week, never retrain."""
    from src.data.entsoe import ENTSOE_FILE
    from src.recommendations.backtester import Strategy, backtest, mase
    from src.recommendations.data_sources import entsoe_series, seismic_series
    from src.recommendations.forecast import mase_scale
    from src.recommendations.run_backtest import SEISMIC_DIR

    st = Strategy(RECALIBRATE, "recalibrate the level every week", 7, "fixed", "recalibrate_level")
    rows = []
    for s in entsoe_series(str(ENTSOE_FILE)) + seismic_series(str(SEISMIC_DIR)):
        y, t_dep, t_chg = s["y"], s["t_deploy"], s["t_change"]
        scale = mase_scale(y[:t_dep])
        r = backtest(y, s["F"], t_dep, st, s["base_features"])
        rows.append(dict(source=s["source"], series=s["series"], strategy=st.name, mase_post=mase(y, r["pred"], t_chg, r["end"], scale),
                         mase_pre=mase(y, r["pred"], t_dep, t_chg, scale), retrains=r["retrains"]))
    return pd.DataFrame(rows)


def configurations(runs: pd.DataFrame, recal: pd.DataFrame | None = None, metric: str = "mase_post") -> pd.DataFrame:
    """One row per real series, one column per strategy or advisor configuration, mean of `metric` over the repeats.
    Strategies on alarm carry the detector (`alarm_retrain [adwin]`), advisors the model and the prompt variant
    (`llm_gate|mistralai/mistral-small-3.2-24b-instruct|examples`); the ablation without diagnostics is left out."""
    df = runs[runs.source.isin(REAL)].copy()
    name = df.strategy.where(df.model.isna(), df.strategy + "|" + df.model.fillna("").str.replace("openrouter/", ""))
    df["name"] = name.where(df.detector.isna(), name + " [" + df.detector.fillna("").astype(str) + "]")
    df = df[~df.name.str.contains(r"\|bare")]
    table = df.pivot_table(index=KEY, columns="name", values=metric, aggfunc="mean")
    if recal is not None:
        table = table.join(recal.set_index(KEY)[metric].rename(RECALIBRATE))
    return table


def families(table: pd.DataFrame) -> dict[str, list[str]]:
    """Advisor configurations by trigger and by whether the advice is checked."""
    cols = [c for c in table.columns if "|" in c]
    return {"applied": [c for c in cols if c.startswith("llm|")], "checked": [c for c in cols if c.startswith("llm_gate|")],
            "on alarm": [c for c in cols if c.startswith("alarm_")]}


def holm(pvalues) -> np.ndarray:
    """Holm step-down adjusted p values."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.minimum(1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    out = np.empty(len(p))
    out[order] = adjusted
    return out


def wilcoxon_pairs(table: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Paired test per pair (a, b) on the differences a - b per series; `wins` counts series where a is lower."""
    rows = []
    for a, b in pairs:
        d = table[a].to_numpy() - table[b].to_numpy()
        p = 1.0 if np.all(d == 0) else float(stats.wilcoxon(d, zero_method="zsplit").pvalue)
        rows.append(dict(a=a, b=b, n=len(d), mean_a=table[a].mean(), mean_b=table[b].mean(), median_diff=float(np.median(d)),
                         wins=int(np.sum(d < 0)), ties=int(np.sum(d == 0)), losses=int(np.sum(d > 0)), p=p))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.to_numpy())
    return out


def paired_tests(table: pd.DataFrame) -> pd.DataFrame:
    """Every advisor configuration against doing nothing, and each family against the fixed strategies it can be
    replaced by: the weekly schedule and the weekly recalibration for advice applied as proposed, the checked
    schedule (a fixed advisor that always proposes retraining on all history) and the checked catalog for checked
    advice. The Holm correction runs over the configurations of one comparison."""
    f = families(table)
    plan = [("every configuration against doing nothing", f["applied"] + f["checked"] + f["on alarm"], "nothing"),
            ("applied against the weekly schedule", f["applied"], "scheduled"),
            ("applied against the weekly recalibration", f["applied"], RECALIBRATE),
            ("checked against the checked schedule", f["checked"], "scheduled_gate"),
            ("checked against the checked catalog", f["checked"], "catalog_gate")]
    out = [wilcoxon_pairs(table, [(c, b) for c in cols]).assign(comparison=name) for name, cols, b in plan if b in table]
    refs = [(a, b) for a, b in [("scheduled", "nothing"), ("scheduled_gate", "nothing"), ("catalog_gate", "nothing"),
                                (RECALIBRATE, "nothing"), ("catalog_gate", "scheduled_gate")] if a in table and b in table]
    out.append(wilcoxon_pairs(table, refs).assign(comparison="strategies without a language model"))
    return pd.concat(out, ignore_index=True)[["comparison", "a", "b", "n", "mean_a", "mean_b", "median_diff", "wins", "ties",
                                               "losses", "p", "p_holm"]]


def action_shares(cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Share of every catalog action among the stored answers of each hosted model and prompt variant."""
    counts: dict[tuple[str, str], Counter] = {}
    for f in sorted(Path(cache_dir).glob("answers_*.jsonl")):
        for line in f.read_text().splitlines():
            rec = json.loads(line)
            if rec["model"].startswith("openrouter/"):
                counts.setdefault((rec["model"].replace("openrouter/", ""), rec.get("prompt") or "base"), Counter())[rec["answer"]["action"]] += 1
    rows = [dict(model=m, prompt=p, answers=sum(c.values()), **{a.name: c[a.name] / sum(c.values()) for a in CATALOG})
            for (m, p), c in sorted(counts.items())]
    total = sum((c for c in counts.values()), Counter())
    rows.append(dict(model="all hosted models", prompt="all", answers=sum(total.values()),
                     **{a.name: total[a.name] / sum(total.values()) for a in CATALOG}))
    return pd.DataFrame(rows)


# where the label of a point goes on each panel: offset in points and alignment; far labels get a thin leader line
LABEL_AT = {
    "entsoe": {"nothing": (16, -40, "left"), RECALIBRATE: (16, -62, "left"), "rules": (0, 11, "center"),
               "scheduled_gate": (0, -12, "center"), "catalog_gate": (9, 7, "left"), "alarm_retrain [adwin]": (9, 9, "left"),
               "scheduled": (9, -8, "left"), "always": (0, -11, "center"), "oracle_weekly": (-9, 0, "right"), "top": (9, 0, "left")},
    "seismic": {"nothing": (9, 0, "left"), RECALIBRATE: (10, -74, "left"), "rules": (10, 26, "left"),
                "scheduled_gate": (-2, -15, "right"), "catalog_gate": (-2, -25, "right"), "alarm_retrain [adwin]": (44, 30, "left"),
                "scheduled": (9, 0, "left"), "always": (0, -11, "center"), "oracle_weekly": (-9, 0, "right"), "top": (44, 12, "left")},
}


def plot_quality_cost(runs: pd.DataFrame, recal: pd.DataFrame):
    """Mean MASE after the change against mean retrainings per series on the two real sources: the strategies
    without a language model (labelled), the weekly oracle (hollow) and the weekly advisor configurations,
    applied as proposed and checked. The horizontal axis is linear up to one retraining and logarithmic above."""
    import matplotlib.pyplot as plt

    from src.recommendations.plots import MUTED, STYLE, TEXT

    applied_color, checked_color, surface = "#eb6834", "#2a78d6", "#ffffff"
    named = [("nothing", "do nothing"), (RECALIBRATE, "level recalibration\nevery week"), ("rules", "rules, no check"),
             ("scheduled_gate", "checked schedule"), ("catalog_gate", "checked catalog"),
             ("alarm_retrain [adwin]", "retrain on alarm (ADWIN)"), ("scheduled", "weekly schedule"), ("always", "daily retraining")]
    error = configurations(runs, recal)
    cost = configurations(runs, recal, "retrains")
    fam = families(error)
    panels = [("entsoe", "National load, 8 countries"), ("seismic", "Seismic noise, 15 stations")]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
        handles = {}
        for ax, (source, title) in zip(axes, panels):
            e, c = error.xs(source).mean(), cost.xs(source).mean()
            handles["applied"], = ax.plot(c[fam["applied"]], e[fam["applied"]], "o", ms=6.5, mfc=applied_color, mec=surface, mew=0.9, zorder=3)
            handles["checked"], = ax.plot(c[fam["checked"]], e[fam["checked"]], "s", ms=6, mfc=checked_color, mec=surface, mew=0.9, zorder=3)
            top = max(fam["applied"], key=lambda k: c[k])  # the one advisor that retrains like a schedule
            labels = [(c[top], e[top], MODEL.get(top.split("|")[1], top) + ", applied", TEXT, "top")]
            for key, label in named:
                handles["strategy"], = ax.plot(c[key], e[key], "o", ms=7, mfc=TEXT, mec=surface, mew=0.9, zorder=4)
                labels.append((c[key], e[key], label, TEXT, key))
            handles["oracle"], = ax.plot(c["oracle_weekly"], e["oracle_weekly"], "o", ms=7, mfc=surface, mec=TEXT, mew=1.5, zorder=4)
            labels.append((c["oracle_weekly"], e["oracle_weekly"], "weekly oracle", MUTED, "oracle_weekly"))
            for x, y, text, color, key in labels:
                dx, dy, ha = LABEL_AT[source][key]
                far = abs(dx) + abs(dy) > 24
                ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=8, color=color, zorder=5,
                            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=1, shrinkB=4) if far else None)
            ax.set_xscale("symlog", linthresh=1, linscale=0.7)
            ax.set_xticks([0, 1, 3, 10, 30, 100])
            ax.set_xticklabels(["0", "1", "3", "10", "30", "100"])
            ax.minorticks_off()
            ax.set_xlim(-0.12, 520)
            values = pd.concat([e[fam["applied"] + fam["checked"]], e[[k for k, _ in named] + ["oracle_weekly"]]])
            ax.set_ylim(values.min() - 0.08 * (values.max() - values.min()), values.max() + 0.08 * (values.max() - values.min()))
            ax.set_title(title, loc="left", fontsize=10)
            ax.set_xlabel("retrainings per series")
        axes[0].set_ylabel("MASE after the change")
        n = len(fam["applied"])
        fig.tight_layout(rect=(0, 0.11, 1, 1))
        fig.legend([handles["strategy"], handles["applied"], handles["oracle"], handles["checked"]],
                   ["strategy without a language model", f"language model, advice applied as proposed ({n} configurations)",
                    "weekly oracle (hindsight)", f"language model, advice checked by the backtest ({n} configurations)"],
                   loc="lower center", ncol=2, fontsize=8.5, handletextpad=0.4, columnspacing=2.5)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--figure", default=str(FIGURE))
    p.add_argument("--recalibrate", action="store_true", help="recompute the fixed weekly recalibration (needs the raw data)")
    args = p.parse_args()
    out = Path(args.out)
    if args.recalibrate or not (out / "recalibrate_weekly.csv").exists():
        recalibrate_weekly().to_csv(out / "recalibrate_weekly.csv", index=False)
    recal = pd.read_csv(out / "recalibrate_weekly.csv")
    runs = pd.read_csv(out / "runs.csv")
    table = configurations(runs, recal)
    tests = paired_tests(table)
    tests.to_csv(out / "llm_paired_tests.csv", index=False)
    shares = action_shares()
    shares.to_csv(out / "llm_action_shares.csv", index=False)
    fig = plot_quality_cost(runs, recal)
    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=200, facecolor="white")
    for name, g in tests.groupby("comparison", sort=False):
        print(f"{name}: {len(g)} pairs, wins of a {g.wins.min()}-{g.wins.max()} of {g.n.iloc[0]}, p {g.p.min():.2g}-{g.p.max():.2g}, "
              f"Holm {g.p_holm.min():.2g}-{g.p_holm.max():.2g}, significant after Holm {(g.p_holm < 0.05).sum()}")
    medians = table.groupby(level="source").median().T.round(3)
    print(medians.loc[[c for c in ("nothing", RECALIBRATE, "scheduled_gate", "catalog_gate", "scheduled", "always") if c in medians.index]].to_string())
    print(shares.tail(1).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
