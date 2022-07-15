import argparse
import os
import time
import yaml

import pyspark

from src.click_model import ClickModelType
from src.experiment import run
from src.report_generation import generate_report


def get_click_model_type_by_name(click_model_name: str) -> ClickModelType:
    if click_model_name == "position-based":
        return ClickModelType.POSITION_BASED
    elif click_model_name == "no-skipping":
        return ClickModelType.NO_SKIPPING
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a pairwise debiasing experiment.")
    parser.add_argument(
        "--conf",
        type=str,
        default="conf.yaml",
        help="Path to a yaml-format configuration file.",
    )

    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder.getOrCreate()

    with open(args.conf, "rt") as f:
        experiment_config = yaml.load(f, yaml.SafeLoader)

    print(f"Loaded the following experiment configuration from {args.conf}:")
    print(
        f"  {'max positions':19} = [{', '.join([str(position) for position in experiment_config['max-positions']])}]"
    )
    print(f"  {'click models':19} = [{', '.join(experiment_config['click-models'])}]")
    print(f"  {'position bias power'} = {experiment_config['position-bias-power']:.1f}")
    print(f"  {'experiment prefix':19} = {experiment_config['experiment-prefix']}")
    print(f"----------")

    print()
    print(f"Running {experiment_config['experiment-prefix']}")
    print()
    for max_position in experiment_config["max-positions"]:
        for click_model_name in experiment_config["click-models"]:
            print(
                f"Parameters: max_position = {max_position:3d}, click_model = {click_model_name}"
            )

            click_model_type = get_click_model_type_by_name(click_model_name)

            experiment_id = f"trunc{max_position}_{click_model_type.value}"
            experiment_output_location = os.path.join(
                experiment_config["output-location"],
                experiment_config["experiment-prefix"],
                experiment_id,
            )
            os.makedirs(experiment_output_location, exist_ok=True)
            output_model_location = os.path.join(experiment_output_location, "models")
            os.makedirs(output_model_location, exist_ok=True)

            st = time.time()
            run(
                experiment_config,
                experiment_output_location,
                output_model_location,
                max_position,
                click_model_type,
            )
            generate_report(
                experiment_config["dataset-location"],
                experiment_output_location,
                output_model_location,
                spark,
            )
            print(f"The run took {int((time.time() - st) / 60):3d} minutes.")
