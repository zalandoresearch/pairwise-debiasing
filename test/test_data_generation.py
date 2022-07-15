import numpy as np
import pandas as pd
import scipy.sparse

from src.click_model import ClickModel, PositionBasedModel
from src.common import LoadedData
from src.data_generation import (
    remove_lists_with_no_relevant_items,
    select_lists_with_clicks,
)
from src.data_generation import generate_data


def test_remove_lists_with_no_relevant_items():
    input_loaded_data = LoadedData(
        features=scipy.sparse.csr_matrix(np.identity(10)),
        relevance_values=np.array([0, 0, 0, 0, 3, 0, 1, 0, 0, 2]),
        qids=np.array(["a", "a", "a", "b", "c", "c", "d", "e", "e", "f"]),
        group_lengths=np.array([3, 1, 2, 1, 2, 1]),
    )
    filtered_loaded_data = remove_lists_with_no_relevant_items(input_loaded_data)
    assert filtered_loaded_data.features.shape[0] == 4
    assert np.all(filtered_loaded_data.relevance_values == np.array([3, 0, 1, 2]))
    assert np.all(filtered_loaded_data.qids == np.array(["c", "c", "d", "f"]))
    assert np.all(filtered_loaded_data.group_lengths == np.array([2, 1, 1]))


def test_select_lists_with_clicks():
    assert np.all(
        select_lists_with_clicks(
            ["a", "a", "b", "c", "c", "c", "d", "d", "e"],
            np.array([0, 0, 1, 0, 1, 0, 1, 0, 0]),
        )
        == np.array([False, False, True, True, True, True, True, True, False])
    )


def test_generate_data():  # similar to PositionBasedModel.test_gen_click_indicators above
    rng = np.random.default_rng(2021)

    n = 100000
    num_tries = 5
    position_biases = 1 / np.arange(1, 11)
    qids = np.repeat(np.arange(n), len(position_biases))
    positions = np.tile(np.arange(len(position_biases)), n)
    relevances = rng.choice([0, 1, 2, 3, 4], size=len(position_biases))
    click_model = PositionBasedModel(position_biases, rng)
    generated_data = generate_data(qids, np.tile(relevances, n), click_model, num_tries)

    freq_df = (
        pd.DataFrame(
            {
                "position": positions,
                "click_indicator": generated_data.click_indicators,
                "examination_indicator": generated_data.examination_indicators,
                "relevance_indicators": generated_data.relevance_indicators,
            }
        )
        .groupby("position")
        .mean()
    )
    bcp = np.array(position_biases) * ClickModel._compute_relevance_probs(
        relevances
    )  # bcp means base click probs.
    # These are the click probabilities
    # that would be true if we didn't filter
    # for lists with at least one interaction
    click_probs = []
    for i in range(len(bcp)):
        q = (1 - bcp).prod()
        click_probs.append(
            bcp[i] * sum([q**j for j in range(num_tries)]) / (1 - q**num_tries)
        )
    # These are the "corrected" true probabilities (where the correction is needed because of the filtering).
    # The logic behind the correction is as follows: in the resulting sample, all cases where a given click
    # indicator is one are retained. At the same time, the cases where it is 0 are discarded iff all of the
    # other indicators are 0 too. The probability of that can be easily computed because the click indicators
    # are independent in this click model.
    assert np.allclose(freq_df.click_indicator, click_probs, rtol=1e-1)
