from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from stattool.empirical_design import calculate_fpr_FWER


def plot_rate(ax, p_values: np.array, alpha: Optional[float] = 0.05, error_type: Optional[str] = "I error"):
    height = ax.hist(p_values, bins=20, density=True)
    rate = 100 * sum(np.array(p_values) < alpha) / len(p_values)

    ax.axhline(1.0, 0, max(height[0]), color="red")

    if error_type == "I error":
        ax.axhline(1.0, 0, max(height[0]), color="red")
        ax.set_title("FPR per test=%2.1f%%" % (rate))

    elif error_type == "II error":
        ax.set_title("Sensitivity per test=%2.1f%%" % (rate))
    else:
        raise Exception('Uknown error type. Use "I error", "II error"')

    ax.axvline(alpha, 0, max(height[0]), color="red")
    ax.set_xlabel("p-value")
    ax.set_xlabel("Frequency")


def plot_metric_distribution(ax, metric: np.array):
    ax.hist(metric, bins=50, density=True)
    ax.set_xlabel("metric value")
    ax.set_xlabel("Frequency")


def plot_cdf(
    p_values: np.array,
    metric: np.array,
    report_name: str,
    alpha: Optional[float] = 0.05,
    error_type: Optional[str] = "I error",
):
    if p_values.ndim == 1:
        fig, ax = plt.subplots(1, ncols=2, figsize=(25, 10))
        fig.suptitle(report_name + f" (size: {len(metric)})", fontsize=16)

        plot_rate(ax[0], p_values, alpha=alpha, error_type=error_type)
        plot_metric_distribution(ax[1], metric)

    else:
        fig, ax = plt.subplots(p_values.shape[1], ncols=2, figsize=(15, p_values.shape[1] * 5))

        if error_type == "I error":
            fig.suptitle(
                "%s. FWER for metric %2.1f%%"
                % (report_name + f" (size: {len(metric)})", calculate_fpr_FWER(p_values, alpha=alpha)),
                fontsize=16,
            )
        elif error_type == "II error":
            fig.suptitle(
                "%s." % (report_name + f" (size: {len(metric)})"),
                fontsize=16,
            )
        else:
            raise Exception("Unkown error_type. Possible values: I error, II error")

        for i in range(p_values.shape[1]):
            plot_rate(ax[i][0], p_values[::, i], alpha=alpha, error_type=error_type)
            plot_metric_distribution(ax[i][1], metric)