# pylint: disable=missing-module-docstring,missing-function-docstring

import numpy as np
import pandas as pd
import pyspark

from evaluation.metrics import base
from evaluation.metrics import mrr


def test_compute():
    score_column = "score"

    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(  # pylint: disable=invalid-name
        pd.DataFrame(
            [
                ("response_0", 0.5, "sku-a", 1),
                ("response_0", 0.3, "sku-c", 2),
                ("response_0", 1.0, "sku-b", 0),
                ("response_1", 0.0, "sku-e", 1),
                ("response_2", 0.0, "sku-z", 0),
            ],
            columns=[
                base.MetricCalculator.COLLECTION_ID_COLUMN,
                score_column,
                base.MetricCalculator.TIE_BREAKER_COLUMN,
                base.MetricCalculator.LABEL_COLUMN,
            ],
        )
    )
    calc = mrr.MRRMetricCalculator([1, 2])
    mrr_at_1_data, mrr_at_2_data = calc.compute(df, score_column)

    assert np.allclose(mrr_at_1_data.values["response_0"], 0.0)
    assert np.allclose(mrr_at_2_data.values["response_0"], 0.5)
    assert np.allclose(mrr_at_1_data.values["response_1"], 1.0)
    assert np.allclose(mrr_at_2_data.values["response_1"], 1.0)
    assert pd.isnull(mrr_at_1_data.values["response_2"])
    assert pd.isnull(mrr_at_2_data.values["response_2"])
