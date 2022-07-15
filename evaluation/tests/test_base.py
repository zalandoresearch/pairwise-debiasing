# pylint: disable=missing-module-docstring,missing-function-docstring

import numpy as np
import pandas as pd
import pyspark

from evaluation.metrics import base


def test_add_rank():
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(  # pylint: disable=invalid-name
        pd.DataFrame(
            [
                ("response_0", "sku-a", 1),
                ("response_0", "sku-b", 2),
                ("response_0", "sku-c", 0),
                ("response_0", "sku-d", 1),
                ("response_1", "sku-e", 0),
            ],
            columns=[
                base.MetricCalculator.COLLECTION_ID_COLUMN,
                base.MetricCalculator.TIE_BREAKER_COLUMN,
                base.MetricCalculator.LABEL_COLUMN,
            ],
        )
    )
    with_ideal_rank = base.add_rank(
        df, base.MetricCalculator.LABEL_COLUMN, "ideal_rank"
    )
    ideal_ranks = (
        with_ideal_rank.toPandas()
        .sort_values(by=base.MetricCalculator.TIE_BREAKER_COLUMN)["ideal_rank"]
        .to_numpy()
    )
    print(ideal_ranks)
    assert np.all(ideal_ranks == np.array([3, 1, 4, 2, 1]))
