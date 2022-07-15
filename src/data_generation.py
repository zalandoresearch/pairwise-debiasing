import fsspec
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.datasets import load_svmlight_file

from src.click_model import ClickModel
from src.common import GeneratedData, LoadedData
from src.preprocessing import DataPart, PROCESSED_DATA_SUFFIX
from src.utils import get_group_lengths, get_positions


OUTPUT_SUFFIX = "_out"


def load_data(
    dataset_location: str,
    data_part: DataPart,
    list_truncation_length: int,
) -> LoadedData:
    with fsspec.open(
        os.path.join(
            dataset_location,
            f"{data_part.value}{PROCESSED_DATA_SUFFIX}/{data_part.value}.id",
        )
    ) as f:
        qids = (
            pd.read_table(f, header=None)[0]
            .str.split("_")
            .str[:2]
            .str.join("_")
            .to_numpy()
        )

    positions = get_positions(qids)
    truncation_mask = positions <= list_truncation_length

    qids = qids[truncation_mask]

    with fsspec.open(
        os.path.join(
            dataset_location,
            f"{data_part.value}{PROCESSED_DATA_SUFFIX}/{data_part.value}.label",
        )
    ) as f:  # weights are actually the relevance values (levels) set by assessors
        true_rel = pd.read_table(f, header=None)[0].to_numpy()
    relevance_values = true_rel[truncation_mask]

    with fsspec.open(
        os.path.join(
            dataset_location,
            f"{data_part.value}{PROCESSED_DATA_SUFFIX}/{data_part.value}.feature",
        )
    ) as f:
        features, _ = load_svmlight_file(f, query_id=False)
    features = features[truncation_mask, :]

    assert features.shape[0] == len(qids)
    assert len(qids) == len(relevance_values)

    group_lengths = np.array(get_group_lengths(qids))
    assert group_lengths.sum() == len(qids)

    return LoadedData(
        features=features,
        relevance_values=relevance_values,
        qids=qids,
        group_lengths=group_lengths,
    )


def repeat_data(data: LoadedData, num_repetitions: int = 16) -> LoadedData:
    return LoadedData(
        features=scipy.sparse.vstack([data.features] * num_repetitions),
        relevance_values=np.concatenate([data.relevance_values] * num_repetitions),
        qids=np.array(
            [qid + f"_{rep:02d}" for rep in range(num_repetitions) for qid in data.qids]
        ),
        group_lengths=np.concatenate([data.group_lengths] * num_repetitions),
    )


def remove_lists_with_no_relevant_items(loaded_data: LoadedData) -> LoadedData:
    qids_with_relevant_items = set(loaded_data.qids[loaded_data.relevance_values > 0])
    mask = pd.Series(loaded_data.qids).isin(qids_with_relevant_items)
    return LoadedData(
        features=loaded_data.features[mask, :],
        relevance_values=loaded_data.relevance_values[mask],
        qids=loaded_data.qids[mask],
        group_lengths=np.array(get_group_lengths(loaded_data.qids[mask])),
    )


def select_lists_with_clicks(
    query_ids: Sequence[str], click_indicators: np.ndarray
) -> np.ndarray:
    """Returns the mask of rows corresponding to lists with at least one click."""
    qids = pd.Series(query_ids)
    return qids.isin(set(qids.loc[click_indicators == 1])).to_numpy()


def generate_data(
    query_ids: np.ndarray,
    relevance_values: np.ndarray,
    click_model: ClickModel,
    num_additional_generations: int = 50,
) -> GeneratedData:
    generated_data = click_model.gen_click_indicators(query_ids, relevance_values, 100)
    for i in range(num_additional_generations):
        no_click_mask = ~select_lists_with_clicks(
            query_ids, generated_data.click_indicators
        )
        if sum(no_click_mask) > 0:
            additional_generated_train_data = click_model.gen_click_indicators(
                query_ids[no_click_mask], relevance_values[no_click_mask], 1000
            )
            generated_data.set_with_mask(no_click_mask, additional_generated_train_data)
    return generated_data


def get_share_lists_with_interactions(
    loaded_data: LoadedData, generated_data: GeneratedData
):
    return len(
        set(
            loaded_data.qids[
                select_lists_with_clicks(
                    loaded_data.qids, generated_data.click_indicators
                )
            ]
        )
    ) / len(set(loaded_data.qids))


def save_loaded_data(loaded_data: LoadedData, data_part: DataPart, output_loc: str):
    full_output_loc = os.path.join(output_loc, f"{data_part.value}{OUTPUT_SUFFIX}")
    with fsspec.open(os.path.join(full_output_loc, "features.npz"), "wb") as f:
        scipy.sparse.save_npz(f, loaded_data.features)
    with fsspec.open(os.path.join(full_output_loc, "relevance_values.npy"), "wb") as f:
        np.save(f, loaded_data.relevance_values)
    with fsspec.open(os.path.join(full_output_loc, "qids.npy"), "wb") as f:
        np.save(f, loaded_data.qids)


def save_generated_data(
    generated_data: GeneratedData,
    data_part: DataPart,
    click_model_name: str,
    output_loc: str,
):
    full_output_loc = os.path.join(output_loc, f"{data_part.value}{OUTPUT_SUFFIX}")
    with fsspec.open(
        os.path.join(full_output_loc, f"{click_model_name}_data.npz"), "wb"
    ) as f:
        np.savez(
            f,
            relevance_indicators=generated_data.relevance_indicators,
            examination_indicators=generated_data.examination_indicators,
            click_indicators=generated_data.click_indicators,
        )


def restore_loaded_data(data_part: DataPart, input_loc: str) -> LoadedData:
    full_input_loc = os.path.join(input_loc, f"{data_part.value}{OUTPUT_SUFFIX}")
    with fsspec.open(os.path.join(full_input_loc, "features.npz"), "rb") as f:
        features = scipy.sparse.load_npz(f)
    with fsspec.open(os.path.join(full_input_loc, "relevance_values.npy"), "rb") as f:
        relevance_values = np.load(f)
    with fsspec.open(os.path.join(full_input_loc, "qids.npy"), "rb") as f:
        qids = np.load(f)
    return LoadedData(
        features=features,
        relevance_values=relevance_values,
        qids=qids,
        group_lengths=np.array(get_group_lengths(qids)),
    )


def restore_generated_data(
    data_part: DataPart, click_model_name: str, input_loc: str
) -> GeneratedData:
    with fsspec.open(
        os.path.join(
            os.path.join(input_loc, f"{data_part.value}{OUTPUT_SUFFIX}"),
            f"{click_model_name}_data.npz",
        ),
        "rb",
    ) as f:
        with np.load(f) as npz:
            return GeneratedData(
                relevance_indicators=npz["relevance_indicators"],
                examination_indicators=npz["examination_indicators"],
                click_indicators=npz["click_indicators"],
            )
