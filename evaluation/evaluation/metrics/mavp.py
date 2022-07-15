"""Mean Average Precision"""

from typing import List

import pyspark
import pyspark.sql.functions as F

import evaluation.evaluation.metrics.base as base


class MAPMetricCalculator(
    base.MetricCalculator
):  # pylint: disable=too-few-public-methods,missing-class-docstring
    def __init__(self, cutoffs: List[int]):
        self.cutoffs = cutoffs
        self.required_columns.add(base.MetricCalculator.LABEL_COLUMN)

    def compute(
        self, df: pyspark.sql.DataFrame, score_col: str
    ) -> List[base.MetricData]:
        base.check_columns(df, self.required_columns | {score_col})

        score_rank_column = "score_rank"
        df = base.add_rank(df, score_col, score_rank_column)

        relevance_indicator_column = "is_relevant"
        df = df.withColumn(
            relevance_indicator_column,
            F.when(F.col(base.MetricCalculator.LABEL_COLUMN) > 0, 1).otherwise(0),
        )

        window = pyspark.sql.Window.partitionBy(
            base.MetricCalculator.COLLECTION_ID_COLUMN
        ).orderBy(score_rank_column)
        precision_times_relevance_indicator_column = (
            "precision_times_relevance_indicator"
        )
        df = df.withColumn(
            precision_times_relevance_indicator_column,
            F.sum(F.col(relevance_indicator_column)).over(window)
            / F.col(score_rank_column)
            * F.col(relevance_indicator_column),
        )

        map_column_prefix = "map_at_"
        grouped_by_instance = df.groupby(
            base.MetricCalculator.COLLECTION_ID_COLUMN
        ).agg(
            *[
                (
                    F.sum(
                        base.apply_cutoff(
                            score_rank_column,
                            precision_times_relevance_indicator_column,
                            cutoff,
                        )
                    )
                    / F.least(F.sum(relevance_indicator_column), F.lit(cutoff))
                ).alias(map_column_prefix + str(cutoff))
                for cutoff in self.cutoffs
            ]
        )

        metric_cols = [map_column_prefix + str(cutoff) for cutoff in self.cutoffs]
        return base.get_metric_data(grouped_by_instance, metric_cols)
