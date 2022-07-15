from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse


@dataclass
class GeneratedData:
    click_indicators: np.ndarray
    examination_indicators: np.ndarray
    relevance_indicators: np.ndarray

    def set_with_mask(
        self, mask: Sequence[bool], other_generated_data: "GeneratedData"
    ):
        assert len(other_generated_data.relevance_indicators) == np.sum(mask)
        self.click_indicators[mask] = other_generated_data.click_indicators
        self.examination_indicators[mask] = other_generated_data.examination_indicators
        self.relevance_indicators[mask] = other_generated_data.relevance_indicators


@dataclass
class LoadedData:
    features: scipy.sparse.csr_matrix
    relevance_values: np.ndarray  # those are the values set by assessors on the scale from 0 to 4
    qids: np.ndarray
    group_lengths: np.ndarray
