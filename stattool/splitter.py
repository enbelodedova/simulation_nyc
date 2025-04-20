import os
from hashlib import sha256
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def split_data(
    df: pd.DataFrame,
    split_count: Optional[int] = 2,
    split_type: Optional[str] = "random",
    percentage_groups: Optional[Iterable[float]] = None,
    **split_kwargs,
):
    split_types = ["random", "hash"]

    if split_type == "random":
        random_split(df, split_count, percentage_groups)

    elif split_type == "hash":
        hash_split(df, split_count, percentage_groups, **split_kwargs)

    else:
        raise Exception(f"{split_type} not implemented. Use the following {split_types}")


def hash_split(df, split_count, percentage_groups: Optional[Iterable[float]] = None, **kwargs):
    len_hash = 6
    hash_base = 16
    k_variants = hash_base**len_hash

    salt = os.urandom(16).hex()

    try:
        df["hashed_int"] = df[kwargs.get("split_by", "user_id")].apply(
            lambda x: int(sha256((str(x) + str(salt)).encode()).hexdigest()[:len_hash], hash_base)
        )
    except:
        raise Exception('No "userid" in DataFrame. Point value to hash in kwargs via split_by')

    if percentage_groups:
        percentage_groups.insert(0, 0)
        percentage_groups = np.cumsum([x * k_variants for x in percentage_groups])

        for i in range(len(percentage_groups) - 1):
            df.loc[
                (df["hashed_int"] > percentage_groups[i]) & (df["hashed_int"] <= percentage_groups[i + 1]), "split"
            ] = i
    else:
        df["split"] = df["hashed_int"] % split_count

    df.drop("hashed_int", axis=1, inplace=True)


def random_split(df, split_count, percentage_groups: Optional[Iterable[float]] = None):
    df["split"] = np.random.choice(split_count, len(df), p=percentage_groups)
