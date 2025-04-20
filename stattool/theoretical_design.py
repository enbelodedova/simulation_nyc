from typing import Iterable, List, Optional

import numpy as np
from statsmodels.stats.power import tt_ind_solve_power

from stattool.stat_test import get_metric_stats

FIRST_TYPE_ERROR = 0.05
SECOND_TYPE_ERROR = 0.2
ddof = 1


def get_parameter_size(
    nominator: str,
    parameter: str,
    split_count: int = 2,
    effect: float = 0.01,
    denominator: Optional[str] = None,
    ratio: float = 1.0,
    alpha: float = FIRST_TYPE_ERROR,
    beta: float = SECOND_TYPE_ERROR,
    alternative: str = "two-sided",
    cnt_metrics: Optional[int] = 3,
    default_cnt_metrics: Optional[int] = 3,
):
    pairing_cnt = split_count * (split_count - 1) / 2

    if cnt_metrics > default_cnt_metrics:
        pairing_cnt *= cnt_metrics

    alpha = alpha / pairing_cnt

    mean, std = get_metric_stats(nominator, denominator)

    nobs1 = len(nominator) / (1 + ratio)

    if parameter == "effect":
        return get_effect_size(mean, std, nobs1, ratio=ratio, alpha=alpha, beta=beta, alternative=alternative)
    elif parameter == "size":
        return get_sample_size(mean, std, effect, ratio=ratio, alpha=alpha, beta=beta, alternative=alternative)
    elif parameter == "power":
        return get_power_size(mean, std, nobs1, effect, ratio=ratio, alpha=alpha, alternative=alternative)
    elif parameter == "correctness":
        return get_correctness_size(mean, std, nobs1, effect, ratio=ratio, beta=beta, alternative=alternative)
    else:
        raise Exception('Uknown parameter. Use from available "effect", "size", "power", "correctness"')


def get_effect_size(
    mean, std, nobs1, alpha=FIRST_TYPE_ERROR, beta=SECOND_TYPE_ERROR, ratio=1.0, alternative="two-sided"
):
    std_effect = tt_ind_solve_power(
        effect_size=None, nobs1=nobs1, alpha=alpha, power=1 - beta, ratio=ratio, alternative=alternative
    )
    mde = std_effect * std / mean

    return np.round(100.0 * mde, 2)


def get_sample_size(
    mean, std, effect, alpha=FIRST_TYPE_ERROR, beta=SECOND_TYPE_ERROR, ratio=1.0, alternative="two-sided"
):
    std_effect = mean * effect / std

    sample_size = tt_ind_solve_power(
        effect_size=std_effect, nobs1=None, alpha=alpha, power=1 - beta, ratio=ratio, alternative=alternative
    )

    return int(sample_size * (1 + ratio))


def get_power_size(mean, std, nobs1, effect, alpha=FIRST_TYPE_ERROR, ratio=1.0, alternative="two-sided"):
    std_effect = mean * effect / std

    power_size = tt_ind_solve_power(
        effect_size=std_effect, nobs1=nobs1, alpha=alpha, power=None, ratio=ratio, alternative=alternative
    )

    return np.round(100.0 * power_size, 2)


def get_correctness_size(mean, std, nobs1, effect, beta=SECOND_TYPE_ERROR, ratio=1.0, alternative="two-sided"):
    std_effect = mean * effect / std

    correctness_size = tt_ind_solve_power(
        effect_size=std_effect, nobs1=nobs1, alpha=None, power=1 - beta, ratio=ratio, alternative=alternative
    )

    return np.round(100.0 * correctness_size, 2)
