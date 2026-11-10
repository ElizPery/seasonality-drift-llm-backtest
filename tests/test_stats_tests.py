"""Tests for the statistical comparison of the strategies (Demsar 2006). Run from the repository root:
python -m pytest tests"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.recommendations.stats_tests import (average_ranks, cliques, critical_difference, friedman, holm, iman_davenport,
                                             plot_cd, tagged, wide, wilcoxon_pairs)


def test_critical_difference_matches_the_worked_example_of_demsar():
    # Demsar (2006), section 3.2.4: four variants of C4.5 on 14 data sets, CD = 1.25 at alpha 0.05 (q = 2.569)
    # and CD = 1.12 at alpha 0.10 (q = 2.291)
    q = stats.studentized_range.ppf(0.95, 4, np.inf) / np.sqrt(2)
    assert q == pytest.approx(2.569, abs=1e-3)
    assert critical_difference(4, 14) == pytest.approx(1.25, abs=5e-3)
    assert critical_difference(4, 14, alpha=0.10) == pytest.approx(1.12, abs=5e-3)


def test_iman_davenport_matches_the_worked_example_of_demsar():
    # same example: average ranks 3.143, 2.000, 2.893, 1.964 give chi2_F = 9.28 and F_F = 3.69
    r = np.array([3.143, 2.000, 2.893, 1.964])
    n, k = 14, 4
    chi2 = 12 * n / (k * (k + 1)) * (np.sum(r**2) - k * (k + 1) ** 2 / 4)
    assert chi2 == pytest.approx(9.28, abs=0.01)
    ff, p = iman_davenport(chi2, n, k)
    assert ff == pytest.approx(3.69, abs=0.01)
    assert p < 0.05 < stats.f.sf(2.0, k - 1, (k - 1) * (n - 1))


def test_ranks_with_ties_and_friedman_agrees_with_scipy():
    table = pd.DataFrame({"a": [0.5, 1.0, 2.0, 3.0], "b": [0.5, 2.0, 1.0, 2.5], "c": [0.7, 3.0, 3.0, 1.0]})
    r = average_ranks(table)
    # row 1: a and b tie for the first place -> 1.5 each, c gets 3
    assert r.tolist() == pytest.approx([(1.5 + 1 + 2 + 3) / 4, (1.5 + 2 + 1 + 2) / 4, (3 + 3 + 3 + 1) / 4])
    res = friedman(table)
    ref = stats.friedmanchisquare(*[table[c] for c in table])  # scipy applies the same correction for ties
    assert res["chi2"] == pytest.approx(ref.statistic)
    assert res["p"] == pytest.approx(ref.pvalue)
    rng = np.random.default_rng(0)
    big = pd.DataFrame(rng.normal(size=(30, 5)) + np.arange(5) * 0.4, columns=list("abcde"))
    assert friedman(big)["chi2"] == pytest.approx(stats.friedmanchisquare(*[big[c] for c in big]).statistic)


def test_holm_on_a_small_example():
    p = [0.01, 0.04, 0.03, 0.005]
    # sorted 0.005, 0.01, 0.03, 0.04 -> 4 x 0.005, 3 x 0.01, 2 x 0.03, 1 x 0.04 = 0.02, 0.03, 0.06, 0.04 -> monotone 0.06
    assert holm(p).tolist() == pytest.approx([0.03, 0.06, 0.06, 0.02])
    from statsmodels.stats.multitest import multipletests

    assert holm(p).tolist() == pytest.approx(multipletests(p, method="holm")[1].tolist())


def test_cliques_join_strategies_closer_than_cd():
    ranks = pd.Series({"a": 1.0, "b": 1.5, "c": 3.0, "d": 3.2})
    assert cliques(ranks, 1.0) == [["a", "b"], ["c", "d"]]
    assert cliques(ranks, 2.1) == [["a", "b", "c"], ["b", "c", "d"]]  # d - a = 2.2 is not below 2.1
    assert cliques(ranks, 2.5) == [["a", "b", "c", "d"]]
    assert cliques(ranks, 0.1) == []


def test_wilcoxon_pairs_counts_and_direction():
    table = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "b": [2.0, 3.0, 3.0, 5.0, 7.0, 6.5]})
    res = wilcoxon_pairs(table, [("a", "b"), ("b", "b")]).set_index("a")
    # differences a - b: -1, -1, 0, -1, -2, -0.5
    assert (res.loc["a", "wins"], res.loc["a", "ties"], res.loc["a", "losses"]) == (5, 1, 0)
    assert res.loc["a", "median_diff"] < 0  # a has the lower error
    assert res.loc["b", "p"] == 1.0  # identical strategies: no test, p = 1
    assert (res.p_holm >= res.p).all()


def test_wide_table_from_runs_with_detectors():
    runs = pd.DataFrame(dict(source="s", series=["x", "x", "x", "y", "y", "y"], scenario="c",
                             strategy=["nothing", "alarm_retrain", "alarm_retrain"] * 2,
                             detector=[np.nan, "adwin", "profile_conformal"] * 2, mase_post=[1.0, 0.9, 0.8, 1.2, 1.1, 1.3]))
    assert tagged(runs).tolist()[:3] == ["nothing", "alarm_retrain [adwin]", "alarm_retrain [profile_conformal]"]
    t = wide(runs, ["nothing", "alarm_retrain [adwin]", "alarm_retrain [profile_conformal]"])
    assert t.shape == (2, 3) and t.loc[("s", "y", "c"), "alarm_retrain [profile_conformal]"] == 1.3
    with pytest.raises(ValueError):
        wide(runs.iloc[:5], ["nothing", "alarm_retrain [adwin]", "alarm_retrain [profile_conformal]"])


def test_cd_diagram_draws():
    import matplotlib

    matplotlib.use("Agg")
    fig = plot_cd(pd.Series({"a": 1.2, "b": 2.1, "c": 2.4, "d": 3.9, "e": 5.4}), 1.5, "toy")
    assert len(fig.axes) == 1
