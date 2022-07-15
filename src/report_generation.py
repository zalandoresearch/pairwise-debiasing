import fsspec
from dataclasses import dataclass
import os
from tempfile import mkdtemp
from typing import List

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyspark.sql

from src.data_generation import (
    DataPart,
    LoadedData,
    load_data,
    remove_lists_with_no_relevant_items,
)
from evaluation.evaluation import summary
from evaluation.evaluation.metrics import mavp, ndcg
from evaluation.evaluation.metrics.base import MetricCalculator


@dataclass
class Model:
    name: str
    booster: lgb.Booster


MODEL_NAMES = [
    "c_ulambdamart_reg0.0",
    "c_lambdamart",
    "c_ulambdamart_reg1.0",
    "c_ulambdamart_reg2.0",
    "c_d1lambdamart",
    "r_lambdamart",
]


def model_name_to_col(model_name: str) -> str:
    return model_name.rsplit(".0", 1)[0]


def load_model(location: str, file_name: str) -> lgb.Booster:
    temp_dir = mkdtemp()
    local_path = os.path.join(temp_dir, file_name)
    with fsspec.open(os.path.join(location, file_name), "rb") as fin:
        with open(local_path, "wb") as fout:
            fout.write(fin.read())
    return lgb.Booster(model_file=local_path)


def load_models(model_location: str) -> List[Model]:
    return [
        Model(model_name, load_model(model_location, model_name))
        for model_name in MODEL_NAMES
    ]


def prep_test_df(
    spark: pyspark.sql.SparkSession, filtered_test_data: LoadedData, models: List[Model]
) -> pd.DataFrame:
    col_dict = {
        "collection_id": filtered_test_data.qids,
        "item_id": np.arange(filtered_test_data.features.shape[0]),
        "label": filtered_test_data.relevance_values.astype(int),
    }
    for model in models:
        col_dict[model_name_to_col(model.name)] = model.booster.predict(
            filtered_test_data.features
        )
    return spark.createDataFrame(pd.DataFrame(col_dict))


def get_summary(
    filtered_test_data: LoadedData,
    model_location: str,
    model_names: List[str],
    metric_calcs: List[MetricCalculator],
    spark: pyspark.sql.SparkSession,
) -> pd.DataFrame:
    return summary.generate_summary(
        prep_test_df(spark, filtered_test_data, load_models(model_location)),
        model_name_to_col(model_names[0]),  # the baseline score to evaluate against
        [
            model_name_to_col(model_name) for model_name in model_names[1:]
        ],  # the scores to be evaluated
        metric_calcs,
    )


def save_summary(experiment_output_location: str, summary_df: pd.DataFrame):
    summary_df.to_csv(os.path.join(experiment_output_location, "summary.csv"))
    with open(os.path.join(experiment_output_location, "short_summary.txt"), "wt") as f:
        f.write(summary.format_summary(summary_df))


def load_summary(experiment_output_location: str):
    return pd.read_csv(
        os.path.join(experiment_output_location, "summary.csv"), index_col=0
    )


def generate_report(
    dataset_location: str,
    experiment_output_location: str,
    model_location: str,
    spark: pyspark.sql.SparkSession,
) -> None:
    metric_calcs = [
        mavp.MAPMetricCalculator([150]),
        ndcg.NDCGMetricCalculator([1, 3, 5, 10]),
    ]
    loaded_test_data = load_data(dataset_location, DataPart.TEST, 150)
    filtered_test_data = remove_lists_with_no_relevant_items(loaded_test_data)
    summary_df = get_summary(
        filtered_test_data, model_location, MODEL_NAMES, metric_calcs, spark
    )
    save_summary(experiment_output_location, summary_df)
