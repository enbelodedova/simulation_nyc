from collections import namedtuple
from itertools import combinations
from typing import Iterable, List, Optional
from copy import deepcopy

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from stattool.config import DEFAULT_ARGS, DEFAULT_CORRECTION, DEFAULT_METHOD


def conduct_test(
        data: pd.DataFrame,
        nominator: str,
        metric_name: str,
        tag_names: List[str],
        combination: Optional[list] = None,
        denominator: Optional[str] = None,
        split_by: Optional[str] = "group",
        alpha: Optional[float] = 0.05,
        aa: bool = False,
        method: Optional[callable] = DEFAULT_METHOD,
        criteria_args: Optional[dict] = DEFAULT_ARGS,
        correction_method: Optional[str] = DEFAULT_CORRECTION,
        cnt_metrics: Optional[int] = 3,
        default_cnt_metrics: Optional[int] = 3,
) -> pd.DataFrame:
    results = pd.DataFrame(columns=["metric_name", "group", "metric_value_left", "metric_value_right", "lift", "pval"])

    df = data.copy()

    if combination is None:
        combination = list(combinations([i for i in tag_names], 2))

    if aa == True:
        combination = list(
            filter(
                lambda x: (x[0] in ("test_aa", "control_aa") and x[1] in ("test_aa", "control_aa"))
                          or (x[0] not in ("test_aa", "control_aa") and x[1] not in ("test_aa", "control_aa")),
                combination,
            )
        )

    if cnt_metrics > default_cnt_metrics:
        alpha /= cnt_metrics

    for pair in combination:
        row = {}
        row["metric_name"] = metric_name
        row["group"] = f"{pair[0]} vs {pair[1]}"
        values_a_nom, values_b_nom = (
            df.loc[df[split_by] == pair[0], nominator],
            df.loc[df[split_by] == pair[1], nominator],
        )

        if denominator:
            values_a_den, values_b_den = (
                df.loc[df[split_by] == pair[0], denominator],
                df.loc[df[split_by] == pair[1], denominator],
            )

            row["metric_value_left"] = values_a_nom.sum() / values_a_den.sum()
            row["metric_value_right"] = values_b_nom.sum() / values_b_den.sum()

            row["pval"] = method(values_a_nom, values_a_den, values_b_nom, values_b_den, **criteria_args).pvalue
        else:
            row["metric_value_left"] = values_a_nom.mean()
            row["metric_value_right"] = values_b_nom.mean()
            row["pval"] = method(values_a_nom, values_b_nom, **criteria_args).pvalue

        row["lift"] = 100 * (row["metric_value_right"] - row["metric_value_left"]) / row["metric_value_left"]

        results = pd.concat([pd.DataFrame(row, index=[0]), results], axis=0)

    if (len(tag_names) == 4 and aa == True) or (len(tag_names) == 2 and aa == False):
        results["is_reject"] = results["pval"] < alpha
    else:
        corrected = multipletests(
            np.array(results["pval"].astype(float)),
            alpha=alpha,
            method=correction_method,
        )[1]
        results["is_reject"] = corrected < alpha

    return results


def format_results(results: pd.DataFrame):
    df = results.copy()

    index = ["group", "metric_name"]
    values = ["metric_value_left", "metric_value_right", "lift", "is_reject", "pval"]
    aggfunc = {i: lambda x: x for i in values}
    df["is_reject"] = df["is_reject"].apply(lambda x: "yes" if x is True else "no")
    df = df.sort_values(by=["is_reject", "lift"])

    pivot = pd.pivot_table(df, values=values, index=index, aggfunc=aggfunc)[
        ["metric_value_left", "metric_value_right", "lift", "pval", "is_reject"]
    ]

    pivot.index = pivot.index.rename({"group": "Group pair", "metric_name": "Metric"})
    pivot = pivot.rename(
        columns={
            "metric_value_left": "Mean left group",
            "metric_value_right": "Mean right group",
            "lift": "Lift, %",
            "is_reject": "Rejection H0",
            "pval": "p-value",
        }
    )

    table = deepcopy(pivot)

    table.loc[(table["Rejection H0"] == "yes") & (table["Lift, %"] > 0), "is_green"] = True
    table.loc[(table["Rejection H0"] == "yes") & (table["Lift, %"] < 0), "is_green"] = False

    table.loc[table["is_green"] == True, "p-value"] = "True"
    table.loc[table["is_green"] == False, "p-value"] = "False"

    table.loc[table["is_green"] == True, "Lift, %"] = "True"
    table.loc[table["is_green"] == False, "Lift, %"] = "False"

    table.loc[table["is_green"] == True, "Rejection H0"] = "True"
    table.loc[table["is_green"] == False, "Rejection H0"] = "False"

    pivot = (
        pivot.style.set_table_styles(
            [
                {"selector": "th", "props": "font-size: 11pt; text-align: center;"},
                {"selector": "td", "props": "font-size: 11pt; text-align: center;"},
                {
                    "selector": "table",
                    "props": " border: 1px solid black; border-collapse: collapse; color: black; font-size: 12px; table-layout: fixed;",
                },
                {"selector": "thead", "props": "border-bottom: 1px solid black; vertical-align: bottom;"},
                {
                    "selector": "tr, th, td",
                    "props": "text-align: right; vertical-align: middle; padding: 0.5em 0.5em; line-height: normal; white-space: normal; max-width: none; border: none;",
                },
                {"selector": "th", "props": "font-weight: bold"},
                {"selector": "tbody tr:hover", "props": "background: rgba(66, 165, 245, 0.2);"},
            ],
            overwrite=False,
        )
        .set_td_classes(table)
        .format(precision=4)
    )
    return pivot


def copy_to_html(table):
    return table.to_html()
