import re
from collections import namedtuple

import numpy as np
from scipy.stats import norm, ttest_ind
from statsmodels.stats.proportion import proportions_ztest
from typing import Optional

result = namedtuple("result", ["statistics", "pvalue"])


# TODO: add CI intervals
def bucketization(x: np.array, y: np.array, bucket=200):
    # получившиеся бакеты могут иметь разное количество наблюдений, поэтому счтаем так
    x = [np.mean(subarray) for subarray in np.array_split(x, bucket)]
    y = [np.mean(subarray) for subarray in np.array_split(y, bucket)]

    return x, y


def bootstrap_bucket(x: np.array, y: np.array, bucket: int = 200, n_samples: int = 1000, p: float = 0.5):
    x_bucket, y_bucket = bucketization(x, y, bucket)

    return bootstrap(x_bucket, y_bucket, n_samples, p=p)


def bootstrap(x: np.array, y: np.array, n_samples: int = 1000, p: float = 0.5, **kwargs):
    mean_diffs = []
    for i in range(n_samples):
        mean_1 = np.percentile(np.random.choice(x, size=len(y)), p)
        mean_2 = np.percentile(np.random.choice(y, size=len(y)), p)
        mean_diffs.append(mean_2 - mean_1)

    p_1 = norm.cdf(x=0, loc=np.mean(mean_diffs), scale=np.std(mean_diffs))
    p_2 = norm.cdf(x=0, loc=-np.mean(mean_diffs), scale=np.std(mean_diffs))

    p_value = min(p_1, p_2) * 2
    # conf_interval = [round(np.percentile(mean_diffs, 2.5), 4), round(np.percentile(mean_diffs, 97.5), 4)]

    return result(np.mean(mean_diffs), p_value)


def linearization(x_0: np.array, y_0: np.array, x_1: np.array, y_1: np.array, k=None):
    if k is None:
        k = x_0.sum() / y_0.sum()
    l_0 = x_0 - k * y_0
    l_1 = x_1 - k * y_1
    return l_0, l_1


def proportion_ztest(x_0: np.array, y_0: np.array, x_1: np.array, y_1: np.array, **kwargs):
    return result(*proportions_ztest([x_0.sum(), x_1.sum()], [y_0.sum(), y_1.sum()], **kwargs))


def calc_t_test_lin(x_0: np.array, y_0: np.array, x_1: np.array, y_1: np.array, **kwargs):
    l0, l1 = linearization(x_0, y_0, x_1, y_1, k=kwargs.get("k", None))
    kwargs.pop("k", None)
    return ttest_ind(l0, l1, **kwargs)


def deltamethod(x_0: np.array, y_0: np.array, x_1: np.array, y_1: np.array, **kwargs):
    n_0 = y_0.shape[0]
    n_1 = y_1.shape[0]

    rto_0, std_0 = get_metric_stats(x_0, y_0)
    rto_1, std_1 = get_metric_stats(x_1, y_1)

    statistic = (rto_1 - rto_0) / np.sqrt(std_0 ** 2 / n_0 + std_1 ** 2 / n_1)

    alternative = kwargs.get('alternative')
    if alternative in ["two-sided", "2-sided", "2s"]:
        pvalue = norm.sf(np.abs(statistic)) * 2
    elif alternative in ["larger", "l"]:
        pvalue = norm.sf(statistic)
    elif alternative in ["smaller", "s"]:
        pvalue = norm.cdf(statistic)
    else:
        raise ValueError("invalid alternative")


    return result(statistic, pvalue)


def cuped(x: np.array, y: np.array, pre_x: np.array, pre_y: np.array):
    theta_x = np.cov(x, pre_x, ddof=0)[0][1]
    theta_x /= np.var(pre_x)

    theta_y = np.cov(y, pre_y, ddof=0)[0][1]
    theta_y /= np.var(pre_y)

    cuped_x = x - (pre_x - np.mean(pre_x)) * theta_x
    cuped_y = y - (pre_y - np.mean(pre_y)) * theta_y

    return cuped_x, cuped_y


def calc_cuped_t_test(x: np.array, y: np.array, pre_x: np.array, pre_y: np.array, **kwargs):
    cuped_x, cuped_y = cuped(x, y, pre_x, pre_y)
    return ttest_ind(cuped_x, cuped_y, **kwargs)


def get_metric_stats(x_0: np.array, y_0: Optional[np.array] = None):

    if y_0 is None:
        y_0 = np.ones(len(x_0))

    mean_nom, var_nom = np.mean(x_0), np.var(x_0)
    mean_den, var_den = np.mean(y_0), np.var(y_0)

    cov = np.mean((x_0 - mean_nom) * (y_0 - mean_den))

    std = np.sqrt(
        var_nom / mean_den ** 2 + var_den * mean_nom ** 2 / mean_den ** 4 - 2 * mean_nom / mean_den ** 3 * cov
    )
    mean = np.sum(x_0) / np.sum(y_0)

    return mean, std


