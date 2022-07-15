import numpy as np
import pandas as pd

from src import click_model


def test_compute_relevance_probs():
    assert np.allclose(
        click_model.ClickModel._compute_relevance_probs([0.0, 1.0, 4.0, 3.0, 3.0, 2.0]),
        [0.0, 1.0 / 15.0, 1.0, 7.0 / 15.0, 7.0 / 15.0, 1.0 / 5.0],
    )


def test_get_first_successful_try_index():
    assert np.all(
        click_model.ClickModel._get_first_successful_try_index(
            ["a", "a", "a", "b", "b", "c", "c", "d", "e", "f"],
            np.array(
                [
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0],
                    [0, 1, 1, 0],
                ]
            ),
        )
        == np.array([0, 0, 0, 2, 2, 3, 3, 3, 3, 1])
    )


def test_position_based_model_gen_click_indicators():
    rng = np.random.default_rng(2021)

    n = 100000
    num_tries = 10
    position_biases = 1 / np.arange(1, 11)
    qids = np.repeat(np.arange(n), len(position_biases))
    positions = np.tile(np.arange(len(position_biases)), n)
    relevances = rng.choice([0, 1, 2, 3, 4], size=len(position_biases))
    generated_data = click_model.PositionBasedModel(
        position_biases, rng
    ).gen_click_indicators(qids, np.tile(relevances, n), num_tries)
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
    # bcp below means base click probs.
    bcp = np.array(position_biases) * click_model.ClickModel._compute_relevance_probs(
        relevances
    )
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


def test_no_skipping_model_gen_click_indicators():
    rng = np.random.default_rng(2021)

    n = 1000
    position_biases = 1 / np.arange(1, 11)
    qids = np.repeat(np.arange(n), len(position_biases))
    relevances = rng.choice([0, 1, 2, 3, 4], size=len(position_biases))
    generated_data = click_model.NoSkippingModel(
        position_biases, len(position_biases), rng
    ).gen_click_indicators(qids, np.tile(relevances, n), 10)
    assert (
        pd.DataFrame({"qid": qids, "e": generated_data.examination_indicators})
        .groupby("qid")
        .e.diff()
        .dropna()
        <= 0
    ).all()
