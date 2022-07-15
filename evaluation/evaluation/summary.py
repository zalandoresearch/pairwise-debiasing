# pylint: disable=missing-module-docstring,missing-function-docstring

from dataclasses import asdict, dataclass
from typing import List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyspark
from scipy.stats import norm

from evaluation.evaluation.metrics import base


@dataclass
class EvaluationResult:  # pylint: disable=too-many-instance-attributes,missing-class-docstring
    metric_name: str
    avg: float
    num_obs: int
    num_null_obs: int
    ci_low: float
    ci_high: float
    rel_change_pct: Optional[float]
    pv: Optional[float]  # pylint: disable=invalid-name
    num_nonnull_diff_obs: Optional[int]

    def to_dict(self):
        er_dict = asdict(self)
        metric_name = er_dict.pop("metric_name")
        return {f"{metric_name}_{key}": value for key, value in er_dict.items()}


def compute_metrics_for_one_score(
    calcs: List[base.MetricCalculator],
    df: pyspark.sql.DataFrame,  # pylint: disable=invalid-name
    score_col: str,
) -> List[base.MetricData]:
    return [
        metric_data for calc in calcs for metric_data in calc.compute(df, score_col)
    ]


def compute_evaluation_result(  # pylint: disable=too-many-locals
    metric_data: base.MetricData,
    baseline_metric_data: Optional[base.MetricData],
    sig_level: float,
) -> EvaluationResult:
    assert baseline_metric_data is None or metric_data.name == baseline_metric_data.name

    num_obs = len(metric_data.values)
    num_null_obs = metric_data.values.isnull().sum()
    avg = metric_data.values.mean()
    std = metric_data.values.std()
    ci_half_len = norm.ppf(1 - sig_level / 2) * std / np.sqrt(num_obs - num_null_obs)
    ci_low = avg - ci_half_len
    ci_high = avg + ci_half_len

    if baseline_metric_data is not None:
        diff = metric_data.values - baseline_metric_data.values
        diff_mean = diff.mean()
        diff_std = diff.std()
        num_nonnull_diff_obs = (~diff.isnull()).sum()
        rel_change_pct = diff_mean / (metric_data.values - diff).mean() * 100
        t = (  # pylint: disable=invalid-name
            np.sqrt(num_nonnull_diff_obs) * diff_mean / diff_std
        )  # pylint: disable=invalid-name
        pv = 2 * (1 - norm.cdf(abs(t)))  # pylint: disable=invalid-name
    else:
        rel_change_pct = None
        pv = None  # pylint: disable=invalid-name
        num_nonnull_diff_obs = None

    return EvaluationResult(
        metric_name=metric_data.name,
        avg=avg,
        num_obs=num_obs,
        num_null_obs=num_null_obs,
        ci_low=ci_low,
        ci_high=ci_high,
        rel_change_pct=rel_change_pct,
        pv=pv,
        num_nonnull_diff_obs=num_nonnull_diff_obs,
    )


BASELINE_SUFFIX = " (baseline)"


def compute_metrics_for_multiple_scores(
    df: pd.DataFrame,  # pylint: disable=invalid-name
    calcs: List[base.MetricCalculator],
    score_cols: List[str],
) -> Mapping[str, List[base.MetricData]]:
    return {
        score_col: compute_metrics_for_one_score(calcs, df, score_col)
        for score_col in score_cols
    }


def create_metric_df(
    score_to_metrics_map: Mapping[str, List[base.MetricData]]
) -> pd.DataFrame:
    return pd.concat(
        [
            metric_data.values.rename(f"{score_name}__{metric_data.name}")
            for score_name, metrics in score_to_metrics_map.items()
            for metric_data in metrics
        ],
        axis=1,
    )


def generate_summary(  # pylint: disable=too-many-arguments,too-many-locals
    df: pyspark.sql.DataFrame,  # pylint: disable=invalid-name
    baseline_score: str,
    scores: List[str],
    calcs: List[base.MetricCalculator],
    sig_level: float = 0.05,
    return_per_list_metrics: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Produce a summary table and (optionally) a dataframe with per-list metric values."""
    score_to_metrics_map = compute_metrics_for_multiple_scores(
        df, calcs, [baseline_score] + scores
    )

    baselines = score_to_metrics_map[baseline_score]
    num_intervals = (len(scores) + 1) * len(baselines)
    adj_sig_level = sig_level / num_intervals
    results = [
        [
            compute_evaluation_result(baseline_metric_data, None, adj_sig_level)
            for baseline_metric_data in baselines
        ]
    ]

    for score in scores:
        metrics = score_to_metrics_map[score]
        results.append(
            [
                compute_evaluation_result(
                    metric_data, baseline_metric_data, adj_sig_level
                )
                for (metric_data, baseline_metric_data) in zip(metrics, baselines)
            ]
        )

    rows = []
    for result_list in results:
        row = {}
        for er in result_list:  # pylint: disable=invalid-name
            row.update(er.to_dict())
        rows.append(row)

    summary_df = pd.DataFrame(
        rows, index=[f"{baseline_score}{BASELINE_SUFFIX}"] + scores
    )
    if return_per_list_metrics:  # pylint: disable=no-else-return
        return summary_df, create_metric_df(score_to_metrics_map)
    else:
        return summary_df


def format_rel_change_pct(
    rel_change_pct: float, pv: float  # pylint: disable=invalid-name
) -> str:
    if pv < 0.01:
        sig_str = "***"
    elif pv < 0.05:
        sig_str = "**"
    elif pv < 0.1:
        sig_str = "*"
    else:
        sig_str = ""
    sign_str = "+" if rel_change_pct > 0 else ""
    return f"{sign_str}{rel_change_pct:.2f}%{sig_str}"


def get_short_summary(summary_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    baseline_idx = [idx for idx in summary_df.index if BASELINE_SUFFIX in idx]

    metric_names = [
        c.split("_avg")[0] for c in summary_df.columns if c.endswith("_avg")
    ]
    num_comparisons = (len(summary_df) - 1) * len(metric_names)
    short_summary_df = pd.concat(
        [
            summary_df.drop(baseline_idx).apply(
                lambda row, mn=metric_name: format_rel_change_pct(
                    row[f"{mn}_rel_change_pct"], row[f"{mn}_pv"] * num_comparisons
                ),
                axis=1,
            )
            for metric_name in metric_names
        ],
        axis=1,
    )
    short_summary_df.columns = metric_names

    assert len(baseline_idx) == 1
    baseline_name = baseline_idx[0].split(BASELINE_SUFFIX)[0]

    return short_summary_df, baseline_name


def format_summary(summary_df: pd.DataFrame) -> str:
    short_summary_df, baseline_name = get_short_summary(summary_df)
    return f"Evaluation Results (relative to the `{baseline_name}` baseline)\n" + str(
        short_summary_df
    )


def print_short_summary(summary_df: pd.DataFrame):
    print(format_summary(summary_df))
