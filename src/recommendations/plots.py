"""Figures for stream B (RQ3): quality against cost of the adaptation strategies, and example series.
The functions return the figure, so the notebook shows it inline."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.recommendations.forecast import DAY

TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
STYLE = {
    "figure.dpi": 110, "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "axes.titlecolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
}

LINE = {"nothing": "#2a78d6", "scheduled": "#eb6834", "oracle": "#1baf7a"}  # validated categorical slots 1-3
LABEL = {"naive": "seasonal naive", "nothing": "do nothing", "always": "always retrain (daily)",
         "scheduled": "scheduled (weekly)", "scheduled_gate": "scheduled + backtest check",
         "catalog_gate": "best checked action (weekly)", "rules": "rules, no check", "rules_gate": "rules + backtest check",
         "llm": "LLM, no check", "llm_gate": "LLM + backtest check", "oracle_weekly": "oracle, weekly",
         "oracle": "oracle, daily (ideal)"}
PANEL = {("synthetic", "none"): "Synthetic: no change", ("synthetic", "level_shift"): "Synthetic: level shift 2 sigma",
         ("synthetic", "trend_ramp"): "Synthetic: 1-week ramp 2 sigma", ("synthetic", "variance"): "Synthetic: noise x2",
         ("entsoe", "lockdown_2020"): "ENTSO-E, 8 countries: March 2020",
         ("seismic", "lockdown_2020"): "Seismic noise, 15 stations: 2020"}


def _place_labels(ax, points):
    """Direct labels next to points; a label that would overlap an earlier one tries other positions."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # the other markers are obstacles too (a marker is about 7 points wide)
    r = 4 * fig.dpi / 72
    markers = {(x, y): matplotlib.transforms.Bbox.from_bounds(*(ax.transData.transform((x, y)) - r), 2 * r, 2 * r) for x, y, _, _ in points}
    placed = []
    offsets = [(7, 0, "left"), (7, 9, "left"), (7, -9, "left"), (7, 18, "left"), (7, -18, "left"),
               (-7, 9, "right"), (-7, -9, "right"), (-7, 0, "right")]
    frame = ax.get_window_extent(renderer)
    for x, y, text, color in sorted(points, key=lambda p: -p[1]):
        obstacles = placed + [b for k, b in markers.items() if k != (x, y)]
        for dx, dy, ha in offsets:
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center", ha=ha, fontsize=7.5, color=color)
            box = t.get_window_extent(renderer)
            inside = frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0 and box.y1 <= frame.y1
            if inside and not any(box.overlaps(b) for b in obstacles):
                placed.append(box)
                break
            t.remove()
        else:
            ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7.5, color=color)


def plot_quality_cost(summary: pd.DataFrame, ncols: int = 3):
    """One panel per scenario: mean retrainings per series against mean MASE after the change."""
    with plt.rc_context(STYLE):
        return _quality_cost(summary, ncols)


def _quality_cost(summary: pd.DataFrame, ncols: int):
    panels = [k for k in PANEL if ((summary.source == k[0]) & (summary.scenario == k[1])).any()]
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.2 * nrows), squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    labels = []
    for ax, key in zip(axes.flat, panels):
        d = summary[(summary.source == key[0]) & (summary.scenario == key[1])].set_index("strategy")
        pts = []
        for name, r in d.iterrows():
            hollow = name.startswith("oracle")
            ax.plot(r.retrains, r.mase_post, "o", ms=7, mfc="white" if hollow else TEXT, mec=TEXT, mew=1.5, zorder=3)
            pts.append((r.retrains, r.mase_post, LABEL.get(name, name), MUTED if hollow else TEXT))
        ax.set_title(PANEL[key], loc="left", fontsize=9.5)
        ax.set_xlabel("retrainings per series")
        ax.set_xlim(-8, d.retrains.max() * 1.9 + 10)
        lo, hi = d.mase_post.min(), d.mase_post.max()
        ax.set_ylim(lo - 0.12 * (hi - lo) - 0.01, hi + 0.12 * (hi - lo) + 0.01)
        labels.append((ax, pts))
    for row in axes:
        row[0].set_ylabel("MASE after the change")
    fig.tight_layout()
    for ax, pts in labels:  # after the layout is final, so the overlap checks see the real geometry
        _place_labels(ax, pts)
    fig.text(0.01, -0.02, "Lower is better. Filled: deployable strategies. Hollow: hindsight oracles (use future data, upper bounds only).",
             fontsize=8, color=MUTED)
    return fig


def example_frame(y: np.ndarray, preds: dict[str, np.ndarray], t_change: int, scale: float, before_days: int, after_days: int) -> pd.DataFrame:
    """Hours around the change with the actual series and the forecasts of each strategy (stored as CSV)."""
    lo, hi = t_change - before_days * DAY, t_change + after_days * DAY
    df = pd.DataFrame({"hours_from_change": np.arange(lo, hi) - t_change, "actual": y[lo:hi]})
    for name, p in preds.items():
        df[name] = p[lo:hi]
    df["scale"] = scale
    return df


def plot_example(df: pd.DataFrame, title: str, unit: str = "load"):
    """Actual series and forecasts around the change (top) and the daily MASE of each strategy (bottom)."""
    with plt.rc_context(STYLE):
        return _example(df, title, unit)


def _example(df: pd.DataFrame, title: str, unit: str):
    t = df.hours_from_change.to_numpy() / DAY
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
    top.plot(t, df.actual, color="#b8b7b1", lw=1.6, label="actual")
    for name, color in LINE.items():
        top.plot(t, df[name], color=color, lw=1.2, label=LABEL[name])
    top.set_ylabel(unit)
    top.set_title(title, loc="left")
    top.legend(loc="upper left", ncol=4, fontsize=8.5)
    day = np.floor(df.hours_from_change.to_numpy() / DAY).astype(int)
    for name, color in LINE.items():
        err = (df[name] - df.actual).abs() / df.scale
        daily = err.groupby(day).mean()
        bottom.plot(daily.index + 0.5, daily.to_numpy(), color=color, lw=2, marker="o", ms=4, label=LABEL[name])
    bottom.set_ylabel("daily MASE")
    bottom.set_xlabel("days from the change")
    for ax in (top, bottom):
        ax.axvline(0, color=MUTED, lw=1, ls=":")
    fig.tight_layout()
    return fig
