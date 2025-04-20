from itertools import combinations
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from scipy.stats import kstest, uniform

from stattool.config import DEFAULT_ARGS, DEFAULT_METHOD
from stattool.splitter import split_data

FIRST_TYPE_ERROR = 0.05
SECOND_TYPE_ERROR = 0.2
ddof = 1


def empiric_mde(
    data: pd.DataFrame,
    nominator: str,
    method: callable = DEFAULT_METHOD,
    beta: float = SECOND_TYPE_ERROR,
    alpha: float = FIRST_TYPE_ERROR,
    N: int = 100,
    denominator: str = None,
    eps: float = 0.001,
    delta: int = 3,
    right_bound: int = 1,
):
    left = 0
    right = right_bound
    power_level = (1 - beta) * 100

    while right - left > eps:
        middle = (left + right) / 2
        power_pvals = check_aa(
            data, nominator, split_count=2, method=method, effects=[0, middle], N=N, denominator=denominator
        )

        power_estimation = (
            100
            * sum(
                np.array(
                    power_pvals.reshape(
                        N,
                    )
                )
                < alpha
            )
            / len(power_pvals)
        )

        if power_estimation >= power_level:
            right = middle
        else:
            left = middle

        if abs(power_estimation - power_level) <= delta:
            return middle

    return right


def check_uniform_distribution(p_values: list):
    _, p_value = kstest(p_values, uniform.cdf)
    return p_value



def empiric_sample_size(
    data: pd.DataFrame,
    nominator: str,
    expected_effect: float,
    method: callable = DEFAULT_METHOD,
    beta: float = SECOND_TYPE_ERROR,
    alpha: float = FIRST_TYPE_ERROR,
    delta: int = 3,
    eps: int = 500,
    denominator: str = None,
    N: int = 100,
):
    left = 10
    right = len(data)
    power_level = (1 - beta) * 100

    while right - left > eps:
        middle = int(np.ceil((left + right) / 2))

        df = data.sample(middle)
        power_pvals = check_aa(
            data, nominator, split_count=2, method=method, effects=[0, middle], N=N, denominator=denominator
        )

        power_estimation = (
            100
            * sum(
                np.array(
                    power_pvals.reshape(
                        N,
                    )
                )
                < alpha
            )
            / len(power_pvals)
        )

        if power_estimation >= power_level:
            right = middle
        else:
            left = middle

        if abs(power_estimation - power_level) <= delta:
            return middle

    return right


def check_aa(
    data: pd.DataFrame,
    nominator: str,
    denominator: Optional[str] = None,
    percentage_groups: Optional[Iterable[float]] = None,
    method: Optional[callable] = DEFAULT_METHOD,
    split_type: Optional[str] = "random",
    split_count: Optional[int] = 2,
    effects: Optional[list] = None,
    N: int = 1000,
    **kwargs,
) -> np.array:
    p_values = []

    df = data.copy()

    if effects is None:
        effects = [0 for _ in range(split_count)]

    combination = list(combinations([i for i in range(split_count)], 2))

    for _ in tqdm(range(N)):
        split_data(df, split_count, split_type, percentage_groups=percentage_groups, **kwargs)

        p_values_per_combo = []

        for pair in combination:
            values_a_nom, values_b_nom = (
                df.loc[df["split"] == pair[0], nominator],
                df.loc[df["split"] == pair[1], nominator],
            )

            if denominator:
                values_a_den, values_b_den = (
                    df.loc[df["split"] == pair[0], denominator],
                    df.loc[df["split"] == pair[1], denominator],
                )

                # подумать над структурой как здесь расчитывать эффекты лучше
                p_values_per_combo.append(
                    method(
                        apply_effect(values_a_nom, effects[pair[0]], denominator=values_a_den),
                        values_a_den,
                        apply_effect(values_b_nom, effects[pair[1]], denominator=values_b_den),
                        values_b_den,
                        **kwargs.get("criteria_args", DEFAULT_ARGS),
                    ).pvalue
                )

            else:
                p_values_per_combo.append(
                    method(
                        apply_effect(values_a_nom, effects[pair[0]]),
                        apply_effect(values_b_nom, effects[pair[1]]),
                        **kwargs.get("criteria_args", DEFAULT_ARGS),
                    ).pvalue
                )

        p_values.append(p_values_per_combo)

    return np.array(p_values)


def apply_pvalue_correction(p_values, alpha=0.05, method="holm-sidak"):
    p_values_corrected = []

    for row in p_values:
        p_values_corrected.append(multipletests(row, alpha=alpha, method=method)[1])

    return np.array(p_values_corrected)


def calculate_fpr_FWER(p_values, alpha=0.05):
    fpr = 100 * sum(np.any(np.array(p_values) < alpha, axis=1)) / len(p_values)
    return fpr


def apply_effect(nominator, relative_shift, denominator=None):
    if relative_shift <= 0:
        return nominator

    if ((nominator == 0) | (nominator == 1)).all():

        if denominator is not None:
            p_est = sum(nominator) / sum(denominator) * (1 + relative_shift)
        else:
            p_est = sum(nominator) / len(nominator) * (1 + relative_shift)

        if p_est > 1:
            p_est = 1

        return np.random.binomial(1, p_est, len(nominator))

    elif (denominator != 1).any():
        p_est = sum(nominator) / sum(denominator) * (1 + relative_shift)

        if p_est > 1:
            p_est = 1

        return np.random.binomial(denominator, p_est)

    mu, sigma = (relative_shift + 1), 0.2
    s = np.random.normal(mu, sigma, size=len(nominator))

    return s * np.where(nominator == 0, 1e-5, nominator)
