# pylint: disable=missing-module-docstring,missing-function-docstring

import numpy as np
import pandas as pd
import pyspark

from evaluation.metrics import base
from evaluation.metrics import mavp


def test_compute():
    score_column = "score"

    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(  # pylint: disable=invalid-name
        pd.DataFrame(
            [
                ("response_0", 0.5, "sku-a", 1),
                ("response_0", 0.3, "sku-b", 2),
                ("response_0", 1.0, "sku-c", 0),
                ("response_1", 0.0, "sku-a", 1),
                ("response_2", 0.0, "sku-d", 0),
                ("response_3", 0.5, "sku-e", 1),
                ("response_3", 0.3, "sku-f", 2),
            ],
            columns=[
                base.MetricCalculator.COLLECTION_ID_COLUMN,
                score_column,
                base.MetricCalculator.TIE_BREAKER_COLUMN,
                base.MetricCalculator.LABEL_COLUMN,
            ],
        )
    )
    calc = mavp.MAPMetricCalculator([1, 2, 3])
    map_at_1_data, map_at_2_data, map_at_3_data = calc.compute(df, score_column)

    assert np.allclose(map_at_1_data.values["response_0"], 0.0)
    assert np.allclose(map_at_2_data.values["response_0"], 0.25)
    assert np.allclose(map_at_3_data.values["response_0"], (1 / 2 + 2 / 3) / 2)
    assert np.allclose(map_at_1_data.values["response_1"], 1.0)
    assert np.allclose(map_at_2_data.values["response_1"], 1.0)
    assert np.allclose(map_at_3_data.values["response_1"], 1.0)
    assert pd.isnull(map_at_1_data.values["response_2"])
    assert pd.isnull(map_at_2_data.values["response_2"])
    assert pd.isnull(map_at_3_data.values["response_2"])
    assert np.allclose(map_at_1_data.values["response_3"], 1.0)
    assert np.allclose(map_at_2_data.values["response_3"], 1.0)
    assert np.allclose(map_at_3_data.values["response_3"], 1.0)
