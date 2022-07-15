from abc import ABCMeta, abstractmethod

import enum
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.common import GeneratedData
from src.utils import get_random_number_generator, pad_array, get_positions


class ClickModelType(enum.Enum):
    POSITION_BASED = "pbm"
    NO_SKIPPING = "nsm"


class ClickModel(metaclass=ABCMeta):
    @abstractmethod
    def gen_click_indicators(
        self, query_ids: Sequence[str], relevance_values: Sequence[int], num_tries: int
    ) -> GeneratedData:
        pass

    @staticmethod
    def _compute_relevance_probs(relevance_values: Sequence[float]):
        assert np.min(relevance_values) >= 0.0
        return (np.power(2, relevance_values) - 1) / (
            np.power(2, np.max(relevance_values)) - 1
        )

    @staticmethod
    def _get_first_successful_try_index(
        query_ids: Sequence, indicators: np.ndarray
    ) -> np.ndarray:
        """Returns a series of column indices for each row in the indicator matrix.

        Learning-to-rank methods don't make use of input lists containing no interactions (i.e. where
        all of the items have the same label of 0). In the original (Unbiased LambdaMART) experiments,
        they generate clicks for a given list until there's at least one item with an interaction. In
        our numpy-based implementation, we take an equivalent but a slightly different approach: for
        each input list, we generate multiple lists of click indicators and then pick the first list
        of click indicators with at least one click. This is a helper function that implements this
        selection.
        """
        assert len(indicators.shape) == 2
        assert len(query_ids) == indicators.shape[0]
        _, nonzero_col_indices = np.nonzero(
            np.diff(
                indicators.cumsum(axis=1).astype(bool),
                axis=1,
                prepend=False,
                append=True,
            )
        )
        return np.minimum(
            pd.Series(query_ids).map(
                pd.DataFrame({"qid": query_ids, "nonzero_col_idx": nonzero_col_indices})
                .groupby("qid")
                .nonzero_col_idx.min()
                .to_dict()
            ),
            indicators.shape[1] - 1,
        )


class PositionBasedModel(ClickModel):
    def __init__(
        self,
        position_biases: Sequence[float],
        rng: Optional[np.random.Generator] = None,
    ):
        self.type = ClickModelType.POSITION_BASED
        assert np.min(position_biases) >= 0.0 and np.max(position_biases) <= 1.0
        self.position_biases = np.array(position_biases)
        self.rng = get_random_number_generator(rng)

    def gen_click_indicators(
        self,
        query_ids: Sequence[str],
        relevance_values: Sequence[int],
        num_tries: int  # num_tries is the number of times we generate the clicks for a single list
        # (we then pick only one collection of clicks per an input list - the first
        # one with at least one click)
    ) -> GeneratedData:
        total_impressions = len(relevance_values)
        relevance_indicators = self.rng.binomial(
            1,
            p=ClickModel._compute_relevance_probs(relevance_values)[:, None],
            size=(total_impressions, num_tries),
        )  # shape: total_impressions x num_tries
        positions = (
            pd.DataFrame({"qid": query_ids})
            .assign(dummy=1)
            .groupby("qid")
            .dummy.rank(method="first")
            .astype(int)
        )
        examination_indicators = self.rng.binomial(
            1,
            p=self.position_biases[
                np.minimum((positions - 1), len(self.position_biases) - 1)
            ][:, None],
            size=(total_impressions, num_tries),
        )  # shape: total_impressions x num_tries
        click_indicators = (
            examination_indicators * relevance_indicators
        )  # shape: total_impressions x num_tries

        # To generate the result, we pick the first collection of click indicators with at least one interaction.
        # The index of this first collection is generally different for different input lists (queries). Those
        # indices are obtained using the ClickModel._get_first_successful_try_index helper method.
        successful_try_col_indices = ClickModel._get_first_successful_try_index(
            query_ids, click_indicators
        )
        all_rows = np.arange(
            click_indicators.shape[0]
        )  # we are going to select all the rows of the generated
        # matrices and pick only one item for each row given
        # by successful_try_col_indices

        return GeneratedData(
            examination_indicators=examination_indicators[
                all_rows, successful_try_col_indices
            ],
            relevance_indicators=relevance_indicators[
                all_rows, successful_try_col_indices
            ],
            click_indicators=click_indicators[all_rows, successful_try_col_indices],
        )


class NoSkippingModel(ClickModel):
    def __init__(
        self,
        position_biases: Sequence[float],
        max_position: int,
        rng: Optional[np.random.Generator] = None,
    ):
        self.type = ClickModelType.NO_SKIPPING
        assert np.min(position_biases) >= 0.0 and np.max(position_biases) <= 1.0
        self.position_biases = pad_array(position_biases, max_position)
        self.rng = get_random_number_generator(rng)

    def gen_click_indicators(
        self,
        query_ids: Sequence[str],
        relevance_values: Sequence[int],
        num_tries: int  # num_tries is the number of times we generate the clicks for a single list
        # (we then pick only one collection of clicks per an input list - the first
        # one with at least one click)
    ) -> GeneratedData:
        unique_queries = pd.unique(query_ids)
        last_viewed_pos_probs = (  # these are the stopping (or "exit") probabilities
            self.position_biases - np.concatenate([self.position_biases[1:], [0]])
        )
        last_viewed_positions = np.random.choice(
            np.arange(1, len(self.position_biases) + 1),
            (len(unique_queries), num_tries),
            p=last_viewed_pos_probs,
        )[
            pd.Series(query_ids)
            .map({q: i for i, q in enumerate(unique_queries)})
            .to_numpy(),
            :,
        ]  # shape: total_impressions x num_tries
        positions = get_positions(query_ids)
        examination_indicators = (positions[:, None] <= last_viewed_positions).astype(
            int
        )  # shape: total_impressions x num_tries

        total_impressions = len(relevance_values)
        relevance_indicators = self.rng.binomial(
            1,
            p=ClickModel._compute_relevance_probs(relevance_values)[:, None],
            size=(total_impressions, num_tries),
        )

        click_indicators = (
            examination_indicators * relevance_indicators
        )  # shape: total_impressions x num_tries

        # To generate the result, we pick the first collection of click indicators with at least one interaction.
        # The index of this first collection is generally different for different input lists (queries). Those
        # indices are obtained using the ClickModel._get_first_successful_try_index helper method.
        successful_try_col_indices = ClickModel._get_first_successful_try_index(
            query_ids, click_indicators
        )
        all_rows = np.arange(
            click_indicators.shape[0]
        )  # we are going to select all the rows of the generated
        # matrices and pick only one item for each row given
        # by successful_try_col_indices

        return GeneratedData(
            examination_indicators=examination_indicators[
                all_rows, successful_try_col_indices
            ],
            relevance_indicators=relevance_indicators[
                all_rows, successful_try_col_indices
            ],
            click_indicators=click_indicators[all_rows, successful_try_col_indices],
        )
