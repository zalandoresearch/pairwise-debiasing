from collections import Counter
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def get_global_np_random_number_generator():
    return np.random.random.__self__


def get_random_number_generator(
    rng: Optional[np.random.Generator],
) -> np.random.Generator:
    return rng if rng is not None else get_global_np_random_number_generator()


def pad_array(arr: Sequence, length: int) -> np.ndarray:
    return np.concatenate([arr, list(arr[-1:]) * (length - len(arr))])


def get_positions(qids: Sequence) -> np.ndarray:
    return (
        pd.DataFrame(qids, columns=["qid"])
        .assign(dummy=1)
        .groupby("qid")
        .dummy.rank(method="first")
        .astype(int)
        .to_numpy()
    )


def get_group_lengths(qids: Sequence) -> Sequence[int]:
    return list(pd.Series(pd.unique(qids)).map(Counter(qids)))
