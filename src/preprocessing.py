import argparse
import enum
import os
import shutil
import tempfile
from typing import *

import pyspark
from pyspark.sql import functions as F
from pyspark.sql import DataFrame


FEATURE_FILE_EXT = "feature"
ID_FILE_EXT = "id"
LABEL_FILE_EXT = "label"
PROCESSED_DATA_SUFFIX = "_processed"


class DataPart(enum.Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


def get_text_file_paths(dir: str) -> List[str]:
    text_file_paths = []
    for file_name in os.listdir(dir):
        if file_name.endswith(".txt"):
            text_file_paths.append(os.path.join(dir, file_name))
    return text_file_paths


def save_as_a_text_file(
    df: DataFrame, temp_location: str, output_location: str, output_file_name: str
):
    temp_output_directory = os.path.join(temp_location, output_file_name)
    df.repartition(1).write.mode("overwrite").format("text").save(temp_output_directory)
    output_text_file_paths = get_text_file_paths(temp_output_directory)
    assert len(output_text_file_paths) == 1

    shutil.move(
        output_text_file_paths[0], os.path.join(output_location, output_file_name)
    )


def split_id_and_features(df: DataFrame) -> DataFrame:
    return (
        df.withColumn(
            "id", F.element_at(F.split(F.col("value"), pattern="\s+", limit=2), 1)
        )
        .withColumn(
            "features",
            F.concat(
                F.lit("0 "),
                F.element_at(F.split(F.col("value"), pattern=r"\s+", limit=2), -1),
            ),
        )
        .drop("value")
    )


def fix_features(
    source: str,
    destination: str,
    part: DataPart,
    spark: pyspark.sql.SparkSession,
    hex_prefix: str,
):
    libsvm_as_text_df = split_id_and_features(
        spark.read.format("text").load(
            os.path.join(source, f"{part.value}/{part.value}.feature")
        )
    )

    labels_df = (
        spark.read.format("text")
        .load(os.path.join(source, f"{part.value}/{part.value}.weights"))
        .withColumn("value", F.split("value", " "))
        .select(
            F.element_at("value", 1).alias("qid"),
            F.slice("value", F.lit(2), F.size("value")).alias("labels"),
        )
        .select("qid", F.explode("labels").alias("label"))
    )

    assert libsvm_as_text_df.count() == labels_df.count()
    df = libsvm_as_text_df.withColumn(
        "row_number", F.monotonically_increasing_id()
    ).join(
        labels_df.withColumn("row_number", F.monotonically_increasing_id()),
        on="row_number",
        how="inner",
    )
    assert (
        not df.select(
            F.when(F.element_at(F.split(F.col("id"), "_"), 2) == F.col("qid"), 0)
            .otherwise(1)
            .alias("mismatch_indicator")
        )
        .agg(F.max("mismatch_indicator").alias("has_mismatches"))
        .head()
        .has_mismatches
    )

    if hex_prefix != "":
        print(
            f"{part.value}: subsampling by using only queries which id hash starts with {hex_prefix}."
        )
        df = df.where(F.md5("id").startswith(hex_prefix.lower()))

    temp_location = os.path.join(tempfile.mkdtemp(), f"{part.value}")
    output_location = os.path.join(destination, f"{part.value}{PROCESSED_DATA_SUFFIX}")
    os.makedirs(output_location, exist_ok=True)

    save_as_a_text_file(
        df.select("id"),
        temp_location,
        output_location,
        f"{part.value}.{ID_FILE_EXT}",
    )
    save_as_a_text_file(
        df.select("features"),
        temp_location,
        output_location,
        f"{part.value}.{FEATURE_FILE_EXT}",
    )
    save_as_a_text_file(
        df.select("label"),
        temp_location,
        output_location,
        f"{part.value}.{LABEL_FILE_EXT}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess data to make it loadable with svmlight utils."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="ranked_yahoo_c14_dataset",
        help="Input dataset path.",
    )
    parser.add_argument(
        "--destination",
        type=str,
        default="ranked_yahoo_c14_dataset",
        help="Input dataset path.",
    )
    parser.add_argument(
        "--hex-prefix",
        type=str,
        default="",
        help="Only use queries with this prefix of the id hash. Pass an empty prefix to use all queries.",
    )
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    fix_features(args.source, args.destination, DataPart.TRAIN, spark, args.hex_prefix)
    fix_features(args.source, args.destination, DataPart.VALID, spark, args.hex_prefix)
    fix_features(args.source, args.destination, DataPart.TEST, spark, args.hex_prefix)
