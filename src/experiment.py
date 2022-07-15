import os
import tempfile
from typing import Any, Mapping, Optional, Tuple

import fsspec
import lightgbm as lgb
import numpy as np


from dlambdamart.dlambdamart.objective import DatasetWithCalculatorRanks0TPlusAndP
from dlambdamart.dlambdamart.objective import debiased_lambdarank_objective_fixed_tplus
from dlambdamart.dlambdamart.objective import unbiased_lambdarank_objective_fixed_tplus
from src.click_model import ClickModelType, NoSkippingModel, PositionBasedModel
from src.data_generation import LoadedData, GeneratedData
from src.data_generation import (
    load_data,
    repeat_data,
    generate_data,
    save_loaded_data,
    save_generated_data,
)
from src.data_generation import (
    get_share_lists_with_interactions,
    remove_lists_with_no_relevant_items,
)
from src.data_generation import select_lists_with_clicks
from src.preprocessing import DataPart
from src.utils import get_group_lengths, get_positions, pad_array

REG_P_VALUES = [0.0, 1.0, 2.0]  # regularisation parameter for the baseline method

LGB_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": 1,
    "train_metric": True,
    "max_bin": 255,
    "num_trees": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "tree_learner": "serial",
    "feature_fraction": 0.9,
    "bagging_freq": 1,
    "bagging_fraction": 0.9,
    "min_data_in_leaf": 50,
    "min_sum_hessian_in_leaf": 5.0,
    "is_enable_sparse": True,
    "use_two_round_loading": False,
    "lambdarank_truncation_level": 30,
}


def save_model(model: lgb.Booster, name: str, output_loc: str):
    tmp_dir = tempfile.mkdtemp()
    local_path = os.path.join(tmp_dir, name)
    model.save_model(local_path)

    output_url = os.path.join(output_loc, name)
    with fsspec.open(output_url, "wb") as f_out:
        with open(local_path, "rb") as f_in:
            f_out.write(f_in.read())


def create_data(
    dataset_location: str,
    position_biases: np.ndarray,
    max_position: int,
    click_model_type: ClickModelType,
) -> Tuple[LoadedData, GeneratedData]:
    loaded_train_data = load_data(dataset_location, DataPart.TRAIN, max_position)

    # For simplicity, we assume that users don't click on irrelevant items.
    # We, therefore, exclude lists in which the relevance of all items is zero
    # because no clicks can be generated for those lists and hence they can't
    # be used for training.
    filtered_train_data = remove_lists_with_no_relevant_items(loaded_train_data)
    print(
        "The percentage of removed lists is roughly "
        + f"{(1 - len(filtered_train_data.group_lengths) / len(loaded_train_data.group_lengths)) * 100:.0f}%."
    )

    # We follow the experiment from the Unbiased LambdaMART paper and repeat each list
    # of the data set 16 times (before generating clicks).
    repeated_train_data = repeat_data(filtered_train_data, 16)

    rng = np.random.default_rng(2022)

    if click_model_type == ClickModelType.POSITION_BASED:
        click_model = PositionBasedModel(position_biases, rng)
    else:
        click_model = NoSkippingModel(
            position_biases, repeated_train_data.group_lengths.max(), rng
        )

    generated_train_data = generate_data(
        repeated_train_data.qids, repeated_train_data.relevance_values, click_model, 10
    )
    print(
        "The percentage of lists with at least one click is "
        + f"{get_share_lists_with_interactions(repeated_train_data, generated_train_data) * 100:.2f}%."
    )

    return repeated_train_data, generated_train_data


def make_lgb_dataset(
    loaded_data: LoadedData,
    generated_data: GeneratedData,
    position_biases: np.ndarray,
    pad_t_plus: bool,
    use_true_relevance_indicators: bool = False,
    reg_p: float = 0.0,
) -> lgb.Dataset:
    lists_with_clicks_indices = select_lists_with_clicks(
        loaded_data.qids, generated_data.click_indicators
    )
    max_pos = np.max(loaded_data.group_lengths)
    return DatasetWithCalculatorRanks0TPlusAndP(
        max_ndcg_pos=max_pos,
        ranks=get_positions(loaded_data.qids),
        p=reg_p,
        t_plus=pad_array(position_biases, max_pos)
        if pad_t_plus
        else np.array(position_biases),
        label=(
            generated_data.click_indicators[lists_with_clicks_indices]
            if not use_true_relevance_indicators
            else loaded_data.relevance_values[lists_with_clicks_indices].astype(int)
        ),
        data=loaded_data.features[lists_with_clicks_indices, :],
        group=np.array(get_group_lengths(loaded_data.qids[lists_with_clicks_indices])),
    )


def train_and_save_lambdamart_models(
    repeated_train_data: LoadedData,
    generated_train_data: GeneratedData,
    position_biases: np.ndarray,
    output_model_location: str,
) -> None:
    # -- a LambdaMART model trained on (biased) click data
    c_train_set = make_lgb_dataset(
        repeated_train_data, generated_train_data, position_biases, pad_t_plus=False
    )
    c_lambdamart = lgb.train(params=LGB_PARAMS, train_set=c_train_set)
    save_model(c_lambdamart, f"c_lambdamart", output_model_location)

    # -- a LambdaMART model trained on the (unobserved) relevance data
    r_train_set = make_lgb_dataset(
        repeated_train_data,
        generated_train_data,
        position_biases,
        pad_t_plus=False,
        use_true_relevance_indicators=True,
    )
    r_lambdamart = lgb.train(params=LGB_PARAMS, train_set=r_train_set)
    save_model(r_lambdamart, "r_lambdamart", output_model_location)


def train_and_save_unbiased_lambdamart_models(
    repeated_train_data: LoadedData,
    generated_train_data: GeneratedData,
    position_biases: np.ndarray,
    output_model_location: str,
) -> None:
    """Trains Unbiased LambdaMART models on click data with different levels of regularisation"""
    for reg_p in REG_P_VALUES:
        c_train_set = make_lgb_dataset(
            repeated_train_data,
            generated_train_data,
            position_biases,
            pad_t_plus=False,
            reg_p=reg_p,
        )

        t_minus = np.ones(len(c_train_set.t_plus))

        def u_custom_objective(preds, dataset):
            return unbiased_lambdarank_objective_fixed_tplus(preds, dataset, t_minus)

        c_ulambdamart = lgb.train(
            params=LGB_PARAMS, train_set=c_train_set, fobj=u_custom_objective
        )
        save_model(
            c_ulambdamart, f"c_ulambdamart_reg{reg_p:.1f}", output_model_location
        )


def train_and_save_robust_unbiased_lambdamart_model(
    repeated_train_data: LoadedData,
    generated_train_data: GeneratedData,
    position_biases: np.ndarray,
    output_model_location: str,
) -> None:
    """Trains a "Robust" LambdaMART model trained on click data"""

    def d1_custom_objective(preds, dataset):
        return debiased_lambdarank_objective_fixed_tplus(preds, dataset, 0)

    c_train_set = make_lgb_dataset(
        repeated_train_data, generated_train_data, position_biases, pad_t_plus=True
    )
    c_d1lambdamart = lgb.train(
        params=LGB_PARAMS, train_set=c_train_set, fobj=d1_custom_objective
    )
    save_model(c_d1lambdamart, "c_d1lambdamart", output_model_location)


def train_and_save_models(
    repeated_train_data: LoadedData,
    generated_train_data: GeneratedData,
    position_biases: np.ndarray,
    output_model_location: str,
) -> None:
    train_and_save_lambdamart_models(
        repeated_train_data,
        generated_train_data,
        position_biases,
        output_model_location,
    )

    train_and_save_unbiased_lambdamart_models(
        repeated_train_data,
        generated_train_data,
        position_biases,
        output_model_location,
    )

    train_and_save_robust_unbiased_lambdamart_model(
        repeated_train_data,
        generated_train_data,
        position_biases,
        output_model_location,
    )


def run(
    experiment_config: Mapping[str, Any],
    experiment_output_location: str,
    output_model_location: str,
    max_position: int,
    click_model_type: ClickModelType,
) -> None:
    position_biases = np.power(
        1 / np.arange(1, max_position + 1), experiment_config["position-bias-power"]
    )

    print("Generating data..")
    repeated_train_data, generated_train_data = create_data(
        experiment_config["dataset-location"],
        position_biases,
        max_position,
        click_model_type,
    )
    save_loaded_data(repeated_train_data, DataPart.TRAIN, experiment_output_location)
    save_generated_data(
        generated_train_data,
        DataPart.TRAIN,
        click_model_type.value,
        experiment_output_location,
    )

    print("Training models..")
    train_and_save_models(
        repeated_train_data,
        generated_train_data,
        position_biases,
        output_model_location,
    )
