import numpy as np
import pandas as pd
import pyspark

from src.preprocessing import split_id_and_features


def test_split_id_and_features():
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        pd.DataFrame({"value": ["id1 0.1 0.3 0.005 10", "id2 0.8 0.0 0.01 0"]})
    )
    output_df = split_id_and_features(df).toPandas()
    assert (output_df.id.to_numpy() == np.array(["id1", "id2"])).all()
    assert (
        output_df.features.to_numpy()
        == np.array(["0 0.1 0.3 0.005 10", "0 0.8 0.0 0.01 0"])
    ).all()
