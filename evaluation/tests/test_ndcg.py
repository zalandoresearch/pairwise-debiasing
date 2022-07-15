# pylint: disable=missing-module-docstring,missing-function-docstring

import math
import numpy as np
import pandas as pd

import pyspark
from pyspark.sql import Row

from evaluation.metrics import base
from evaluation.metrics.ndcg import NDCGMetricCalculator
from evaluation.metrics.ndcg import calculate_ndcg_for_dataframe_rows_list


def _dcg(ordered_rel):

    ret = 0
    for i, rel in enumerate(ordered_rel):
        ret += (math.pow(2.0, rel) - 1) / math.log2(i + 2)
    return ret


def test_calculate_ndcg_for_dataframe_rows_list():

    pyspark_rows_list = [
        Row(rel=[1, 0, 1], scores=[3, 1, 2]),
        Row(rel=[1, 0, 1], scores=[10, 4, 1]),
        Row(rel=[0.7, 0, 0.5], scores=[3, 1, 2]),
        Row(rel=[0.7, 0, 0.5], scores=[6, 4, 2]),
    ]

    # calculate the ndcg
    pyspark_rows_list = calculate_ndcg_for_dataframe_rows_list(
        pyspark_rows_list, "rel", "scores"
    )

    # iterate over the list to check the ndcg scores
    assert pyspark_rows_list[0]["ndcg@10"] == np.round(
        _dcg([1, 1, 0]) / _dcg([1, 1, 0]), 7
    ).astype(np.float32)
    assert pyspark_rows_list[1]["ndcg@10"] == np.round(
        _dcg([1, 0, 1]) / _dcg([1, 1, 0]), 7
    ).astype(np.float32)
    assert pyspark_rows_list[2]["ndcg@10"] == np.round(
        _dcg([0.7, 0.5, 0]) / _dcg([0.7, 0.5, 0]), 7
    ).astype(np.float32)
    assert pyspark_rows_list[3]["ndcg@10"] == np.round(
        _dcg([0.7, 0, 0.5]) / _dcg([0.7, 0.5, 0]), 7
    ).astype(np.float32)


def test_compute():
    score_column = "score"

    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(  # pylint: disable=invalid-name
        pd.DataFrame(
            [
                ("response_0", 0.5, "sku-y", 1),
                ("response_0", 0.3, "sku-a", 2),
                ("response_0", 1.0, "sku-z", 0),
                ("response_0", 0.0, "sku-x", 1),
                ("response_1", 0.0, "sku-a", 1),
                ("response_2", 0.0, "sku-b", 0),
            ],
            columns=[
                base.MetricCalculator.COLLECTION_ID_COLUMN,
                score_column,
                base.MetricCalculator.TIE_BREAKER_COLUMN,
                base.MetricCalculator.LABEL_COLUMN,
            ],
        )
    )
    calc = NDCGMetricCalculator([2, 3])
    ndcg_at_2_data, ndcg_at_3_data = calc.compute(df, score_column)

    assert np.allclose(
        ndcg_at_2_data.values["response_0"], (1 / np.log2(3)) / (3 + 1 / np.log2(3))
    )
    assert np.allclose(
        ndcg_at_3_data.values["response_0"],
        (1 / np.log2(3) + 3 / np.log2(4)) / (3 + 1 / np.log2(3) + 1 / np.log2(4)),
    )
    assert np.allclose(ndcg_at_2_data.values["response_1"], 1.0)
    assert np.allclose(ndcg_at_3_data.values["response_1"], 1.0)
    assert pd.isnull(ndcg_at_2_data.values["response_2"])
    assert pd.isnull(ndcg_at_3_data.values["response_2"])
