"""Mean Reciprocal Rank"""

from typing import List

import pyspark
import pyspark.sql.functions as F

from evaluation.evaluation.metrics import base


class MRRMetricCalculator(
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

        reciprocal_rank_column = "reciprocal_rank"
        df = df.withColumn(
            reciprocal_rank_column,
            F.when(
                F.col(base.MetricCalculator.LABEL_COLUMN) > 0,
                1 / F.col(score_rank_column),
            ).otherwise(0.0),
        )

        mrr_column_prefix = "mrr_at_"
        max_label_column = "max_label"
        grouped_by_instance = df.groupby(
            base.MetricCalculator.COLLECTION_ID_COLUMN
        ).agg(
            *(
                [
                    F.max(
                        base.apply_cutoff(
                            score_rank_column, reciprocal_rank_column, cutoff
                        )
                    ).alias(mrr_column_prefix + str(cutoff))
                    for cutoff in self.cutoffs
                ]
                + [F.max(base.MetricCalculator.LABEL_COLUMN).alias(max_label_column)]
            )
        )
        metric_cols = [mrr_column_prefix + str(cutoff) for cutoff in self.cutoffs]
        for col in metric_cols:
            grouped_by_instance = grouped_by_instance.withColumn(
                col,
                F.when(F.col(max_label_column) > 0, F.col(col)).otherwise(F.lit(None)),
            )

        return base.get_metric_data(grouped_by_instance, metric_cols)
