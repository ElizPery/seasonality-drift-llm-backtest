"""Statistical comparison of the stream B strategies over many series, following Demsar (2006).

Every series is a block and every strategy a treatment. Per series the strategies are ranked by their error
(rank 1 = the lowest MASE, ties share the average rank), so series of different sources and scales can be pooled.
The Friedman test checks whether the average ranks differ more than chance allows; Demsar recommends its
Iman-Davenport F form, which is less conservative. If the test rejects, the Nemenyi post-hoc test calls two
strategies different when their average ranks differ by at least the critical difference

    CD = q_alpha * sqrt(k (k + 1) / (6 N)),

with k strategies, N series and q_alpha the studentized range quantile divided by sqrt(2). For a few chosen
pairs the paired Wilcoxon signed-rank test on the differences of MASE per series is stronger than Nemenyi;
when several pairs are tested on the same series, their p values get the Holm step-down correction. Zero
differences (two strategies with the same forecasts on a series) are split between the two signs, as Demsar
suggests (`zero_method="zsplit"`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

KEY = ["source", "series", "scenario"]


def tagged(runs: pd.DataFrame) -> pd.Series:
    """Strategy names with the detector in brackets for the strategies that review on alarms
    (`alarm_retrain [adwin]`), the same tags as `plots.tag_detector`."""
    det = runs.detector.fillna("").astype(str)
    return runs.strategy.where(det == "", runs.strategy + " [" + det + "]")


def wide(runs: pd.DataFrame, strategies: list[str], metric: str = "mase_post") -> pd.DataFrame:
    """One row per series, one column per strategy (tagged names), values of `metric`. Raises if a series
    lacks a strategy: the Friedman test needs complete blocks."""
    df = runs.assign(name=tagged(runs))
    df = df[df.name.isin(strategies)]
    table = df.pivot_table(index=KEY, columns="name", values=metric, aggfunc="first")
    missing = [s for s in strategies if s not in table]
    if missing or table.isna().any().any():
        raise ValueError(f"incomplete blocks: missing strategies {missing} or missing values")
    return table[strategies]


def average_ranks(table: pd.DataFrame) -> pd.Series:
    """Mean rank of every column over the rows; rank 1 = the lowest value, ties get the average rank."""
    ranks = np.apply_along_axis(stats.rankdata, 1, table.to_numpy())
    return pd.Series(ranks.mean(axis=0), index=table.columns)


def friedman(table: pd.DataFrame) -> dict:
    """Friedman test over the rows of `table` (blocks) for its columns (treatments), with the correction for
    ties, and the Iman-Davenport F statistic. Returns k, N, the average ranks, both statistics and p values."""
    x = table.to_numpy()
    n, k = x.shape
    ranks = np.apply_along_axis(stats.rankdata, 1, x)
    r = ranks.mean(axis=0)
    chi2 = 12 * n / (k * (k + 1)) * (np.sum(r**2) - k * (k + 1) ** 2 / 4)
    ties = sum(np.sum(c**3 - c) for c in (np.unique(row, return_counts=True)[1] for row in ranks))
    correction = 1 - ties / (n * k * (k**2 - 1))
    chi2 = chi2 / correction if correction > 0 else np.nan
    p = float(stats.chi2.sf(chi2, k - 1))
    ff, p_ff = iman_davenport(chi2, n, k)
    return dict(k=k, n=n, ranks=pd.Series(r, index=table.columns), chi2=float(chi2), p=p, ff=ff, p_ff=p_ff)


def iman_davenport(chi2: float, n: int, k: int) -> tuple[float, float]:
    """F_F = (N - 1) chi2 / (N (k - 1) - chi2), F distributed with k - 1 and (k - 1)(N - 1) degrees of freedom."""
    denom = n * (k - 1) - chi2
    if denom <= 0:  # every block ranks the treatments the same way
        return np.inf, 0.0
    ff = (n - 1) * chi2 / denom
    return float(ff), float(stats.f.sf(ff, k - 1, (k - 1) * (n - 1)))


def critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference of average ranks for k treatments on n blocks."""
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6 * n)))


def cliques(ranks: pd.Series, cd: float) -> list[list[str]]:
    """Maximal groups of strategies whose average ranks all differ by less than `cd` (not significantly
    different by Nemenyi), as drawn by the thick bars of a critical difference diagram."""
    r = ranks.sort_values()
    names, values = list(r.index), r.to_numpy()
    groups = []
    for i in range(len(values)):
        j = i
        while j + 1 < len(values) and values[j + 1] - values[i] < cd:
            j += 1
        if j > i and not any(g[0] <= i and j <= g[1] for g in groups):
            groups.append((i, j))
    return [names[i:j + 1] for i, j in groups]


def holm(pvalues) -> np.ndarray:
    """Holm step-down adjusted p values: the i-th smallest of m is multiplied by m - i + 1, then made monotone."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out = np.empty(m)
    out[order] = adjusted
    return out


def wilcoxon_pairs(table: pd.DataFrame, pairs: list[tuple[str, str]], extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test per pair (a, b) on the per-series differences a - b, with Holm-adjusted
    p values over the pairs. `wins` counts series where a has the lower value. Columns of the pairs may also come
    from `extra` (same rows), so a strategy outside the ranking (an oracle) can be compared too."""
    data = table if extra is None else table.join(extra[[c for c in extra if c not in table]])
    rows = []
    for a, b in pairs:
        d = data[a].to_numpy() - data[b].to_numpy()
        if np.all(d == 0):
            stat, p = np.nan, 1.0
        else:
            res = stats.wilcoxon(d, zero_method="zsplit")
            stat, p = float(res.statistic), float(res.pvalue)
        rows.append(dict(a=a, b=b, n=len(d), mean_a=data[a].mean(), mean_b=data[b].mean(), median_diff=float(np.median(d)),
                         wins=int(np.sum(d < 0)), ties=int(np.sum(d == 0)), losses=int(np.sum(d > 0)), statistic=stat, p=p))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.to_numpy())
    return out


def plot_cd(ranks: pd.Series, cd: float, title: str = "", label=lambda s: s):
    """Critical difference diagram (Demsar 2006, figure 1): average ranks on an axis (best on the left), the
    length of CD above it, and thick bars joining strategies that are not significantly different. The better
    half of the strategies is labeled on the left, the worse half on the right."""
    import matplotlib.pyplot as plt

    from src.recommendations.plots import MUTED, STYLE, TEXT

    r = ranks.sort_values()
    k = len(r)
    half = (k + 1) // 2
    groups = cliques(r, cd)
    top = 0.5 + 0.3 * len(groups)  # label rows start below the bars of the groups
    rows = max(half, k - half)
    span = 2.0 + top + rows  # data units on the vertical axis, about 0.28 inch each
    with plt.rc_context(dict(STYLE, **{"axes.grid": False})):
        fig = plt.figure(figsize=(11, span * 0.28 / 0.86 + 0.2))
        ax = fig.add_axes([0.3, 0.02, 0.4, 0.86])  # the rank axis in the middle, the labels on both sides
        lo, hi = 1, k
        ax.set_xlim(lo - 0.3, hi + 0.3)
        ax.set_ylim(-(top + rows), 2.0)
        ax.axis("off")
        ax.plot([lo, hi], [0, 0], color=TEXT, lw=1)
        for t in range(lo, hi + 1):
            ax.plot([t, t], [0, 0.18], color=TEXT, lw=1)
            ax.text(t, 0.3, str(t), ha="center", va="bottom", fontsize=8, color=MUTED)
        ax.plot([lo, lo + cd], [1.3, 1.3], color=TEXT, lw=1.5)
        for x in (lo, lo + cd):
            ax.plot([x, x], [1.2, 1.4], color=TEXT, lw=1.5)
        ax.text(lo + cd / 2, 1.5, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8.5, color=TEXT)
        for j, group in enumerate(groups):
            y = -(0.35 + 0.3 * j)
            ax.plot([r[group[0]] - 0.08, r[group[-1]] + 0.08], [y, y], color=TEXT, lw=3, solid_capstyle="round", zorder=3)
        for i, (name, value) in enumerate(r.items()):
            left = i < half
            y = -(top + (i if left else k - 1 - i))
            edge = lo - 0.3 if left else hi + 0.3
            ax.plot([value, value, edge], [0, y, y], color=MUTED, lw=0.8)
            text = f"{label(name)}  {value:.2f}" if left else f"{value:.2f}  {label(name)}"
            ax.text(edge + (-0.15 if left else 0.15), y, text, ha="right" if left else "left", va="center", fontsize=8.5,
                    color=TEXT, clip_on=False)
        if title:
            fig.text(0.02, 1.0, title, ha="left", va="bottom", fontsize=10, color=TEXT)
    return fig
