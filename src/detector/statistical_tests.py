import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats
import seaborn as sns


def evaluate_detectors_statistically(results_df: pd.DataFrame, metric_name: str = "Metric"):
    """Performs the Friedman test and Nemenyi post-hoc test to compare drift

    detectors across multiple benchmarks.

    Parameters:
    ----------
    results_df : pd.DataFrame
        A dataframe where rows represent benchmarks (test scenarios)
        and columns represent different detectors.
        Values are performance metrics.
    metric_name : str
        Name of the metric being analyzed (for titles and logs).
    """
    print(f"\n=== Statistical Analysis for Metric: {metric_name} ===")
    
    # Convert dataframe columns to list of arrays for scipy
    detector_names = results_df.columns.tolist()
    data_arrays = [results_df[col].values for col in detector_names]

    # --- Friedman Test ---
    # Null Hypothesis (H0): All detectors have equivalent performance.
    stat, p_value = stats.friedmanchisquare(*data_arrays)

    print(f"Friedreg/Chi-square Statistic: {stat:.4f}")
    print(f"p-value: {p_value:.4e}")

    alpha = 0.05
    if p_value < alpha:
        print(
            "Conclusion: Null hypothesis rejected. Statistically significant"
            " differences exist among the detectors."
        )
    else:
        print(
            "Conclusion: No sufficient evidence to reject the null hypothesis"
            " (no significant differences found)."
        )
        return None

    # --- Nemenyi Post-hoc Test ---
    print("\n--- Nemenyi Post-hoc Test p-value Matrix ---")
    nemenyi_p_values = sp.posthoc_nemenyi_friedman(results_df)

    # Rename columns and index for clarity
    nemenyi_p_values.columns = detector_names
    nemenyi_p_values.index = detector_names
    print(nemenyi_p_values)

    # --- Visualization ---
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        nemenyi_p_values,
        annot=True,
        fmt=".3f",
        cmap="coolwarm_r",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "p-value"},
    )
    plt.title(
        f"Pairwise Detector Comparisons (Nemenyi Test) for {metric_name}\n(p < 0.05 indicates significant difference)"
    )
    plt.tight_layout()
    plt.show()

    return nemenyi_p_values