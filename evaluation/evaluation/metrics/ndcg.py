"""Normalised Discounted Cumulative Gain"""

from typing import Callable, List
from typing import Dict

import numpy as np
import pyspark
import pyspark.sql.functions as F

from evaluation.evaluation.metrics import base


def log_discount(rank_column: str) -> pyspark.sql.Column:
    """Computes the discounts."""
    return F.log2(1 + F.col(rank_column))


def calculate_ndcg_for_dataframe_rows_list(
    pyspark_rows_list: List[pyspark.sql.Row],
    rel_name: str,
    score_name: str,
    k: int = 10,
) -> List[Dict]:

    """Computes NDCG for a list of pyspark.sql.Row objects from a dataframe."""

    metric = NDCGMetric(topn=k)

    ret = []

    for row in pyspark_rows_list:
        # update the state with the relevances and scores (weights)
        metric.update_state(
            np.array(row[rel_name], ndmin=2), np.array(row[score_name], ndmin=2)
        )

        # convert row to a dict
        row_dict = row.asDict()
        # add the result to the row
        row_dict["ndcg@10"] = float(metric.result().numpy())

        ret.append(row_dict)

        # reset the metric state
        metric.reset_states()

    return ret


class NDCGMetricCalculator(
    base.MetricCalculator
):  # pylint: disable=too-few-public-methods,missing-class-docstring
    def __init__(self, cutoffs: List[int], discount_func: Callable = log_discount):
        self.cutoffs = cutoffs
        self.discount_func = discount_func
        self.required_columns.add(base.MetricCalculator.LABEL_COLUMN)

    def compute(
        self, df: pyspark.sql.DataFrame, score_col: str
    ) -> List[base.MetricData]:
        base.check_columns(df, self.required_columns | {score_col})

        exponential_gain_column = "exponential_gain"
        df = df.withColumn(
            exponential_gain_column,
            F.pow(2, F.col(base.MetricCalculator.LABEL_COLUMN)) - 1,
        )

        score_rank_column = "score_rank"
        df = base.add_rank(df, score_col, score_rank_column)
        discounted_gain_column = "discounted_gain"
        df = df.withColumn(
            discounted_gain_column,
            F.col(exponential_gain_column) / self.discount_func(score_rank_column),
        )

        ideal_rank_column = "ideal_rank"
        df = base.add_rank(df, base.MetricCalculator.LABEL_COLUMN, ideal_rank_column)
        ideal_discounted_gain_column = "ideal_discounted_gain"
        df = df.withColumn(
            ideal_discounted_gain_column,
            F.col(exponential_gain_column) / self.discount_func(ideal_rank_column),
        )

        ndcg_column_prefix = "ndcg_at_"
        grouped_by_instance = df.groupby(
            base.MetricCalculator.COLLECTION_ID_COLUMN
        ).agg(
            *[
                (
                    F.sum(
                        base.apply_cutoff(
                            score_rank_column, discounted_gain_column, cutoff
                        )
                    )
                    / F.sum(
                        base.apply_cutoff(
                            ideal_rank_column, ideal_discounted_gain_column, cutoff
                        )
                    )
                ).alias(ndcg_column_prefix + str(cutoff))
                for cutoff in self.cutoffs
            ]
        )

        metric_cols = [ndcg_column_prefix + str(cutoff) for cutoff in self.cutoffs]
        return base.get_metric_data(grouped_by_instance, metric_cols)
