"""Base classes and helper functions"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Set

import pandas as pd
import pyspark
import pyspark.sql.functions as F


@dataclass
class MetricData:
    """A simple class for storing metric values

    Used to store metric values computed over a set of samples. In the
    case of ranking metrics, each sample is a collection of items, e.g.
    as a list of articles or brands.
    """

    name: str
    values: pd.Series


class MetricCalculator(ABC):  # pylint: disable=too-few-public-methods
    """A base class for metric calculators

    The difference between a metric and a metric calculator is that the
    latter computes the metric for different values of its parameters,
    e.g. an NDCG metric calculator would be able to compute NDCG@k for
    multiple values of k. The idea is that by defining a calculator, we
    can optimise the computation of the same metric for diferent values
    of its parameters.
    """

    COLLECTION_ID_COLUMN = "collection_id"
    LABEL_COLUMN = "label"
    TIE_BREAKER_COLUMN = "item_id"

    required_columns: Set[str] = {COLLECTION_ID_COLUMN, TIE_BREAKER_COLUMN}

    @abstractmethod
    def compute(  # pylint: disable=invalid-name
        self, df: pyspark.sql.DataFrame, score_col: str
    ) -> List[MetricData]:
        """Compute metric values for a given ranking score."""


def check_columns(  # pylint: disable=invalid-name
    df: pyspark.sql.DataFrame, required_columns: Set[str]
):
    """Checks that the input data has all of the needed columns."""
    assert required_columns <= set(df.columns)


def get_metric_data(
    grouped_by_collection: pyspark.sql.DataFrame, metric_cols: List[str]
) -> List[MetricData]:
    """Collects values of metrics into MetricData objects."""
    metric_df = (
        grouped_by_collection.select(
            [MetricCalculator.COLLECTION_ID_COLUMN] + metric_cols
        )
        .toPandas()
        .set_index(MetricCalculator.COLLECTION_ID_COLUMN)
    )
    return [MetricData(name=col, values=metric_df[col]) for col in metric_cols]


def add_rank(  # pylint: disable=invalid-name
    df: pyspark.sql.DataFrame, score_col: str, new_rank_column_name: str
) -> pyspark.sql.DataFrame:
    """Computes ranks from ranking score values.

    A higher value of the score corresponds to a higher position (lower
    rank).
    """
    window = pyspark.sql.Window.partitionBy(
        MetricCalculator.COLLECTION_ID_COLUMN
    ).orderBy(
        F.col(score_col).desc(), F.xxhash64(F.col(MetricCalculator.TIE_BREAKER_COLUMN))
    )
    return df.withColumn(new_rank_column_name, F.row_number().over(window))


def apply_cutoff(
    rank_column: str, value_column: str, cutoff: int
) -> pyspark.sql.Column:
    """
    Constructs o copy of value_column with values below the cutoff
    point set to zero.
    """
    return F.when(F.col(rank_column) <= cutoff, F.col(value_column)).otherwise(0.0)
