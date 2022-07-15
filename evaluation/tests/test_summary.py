# pylint: disable=missing-module-docstring,missing-function-docstring

import numpy as np
import pandas as pd
from scipy.stats import norm

from evaluation.metrics import base
from evaluation import summary


def test_compute_evaluation_result():
    md_1 = base.MetricData(
        name="metric",
        values=pd.Series([0.3, 0.2, 0.5, np.nan], index=["a", "b", "c", "d"]),
    )
    md_0 = base.MetricData(
        name="metric",
        values=pd.Series([1.0, 0.1, 0.4, np.nan], index=["b", "a", "x", "y"]),
    )
    er = summary.compute_evaluation_result(  # pylint: disable=invalid-name
        md_1, md_0, 0.05
    )

    assert er.metric_name == "metric"
    avg = 1.0 / 3
    assert np.allclose(er.avg, avg)
    assert er.num_obs == 4
    assert er.num_null_obs == 1
    std = np.std([0.3, 0.2, 0.5], ddof=1)
    assert np.allclose(er.ci_low, avg - norm.ppf(1 - 0.05 / 2) * std / np.sqrt(3))
    assert np.allclose(er.ci_high, avg + 1.96 * std / np.sqrt(3))
    assert er.num_nonnull_diff_obs == 2
    diff_mean = np.mean([0.2, -0.8])
    diff_std = np.std([0.2, -0.8], ddof=1)
    assert np.allclose(er.rel_change_pct, diff_mean / np.mean([1.0, 0.1]) * 100)
    t = np.sqrt(2) * diff_mean / diff_std  # pylint: disable=invalid-name
    assert np.allclose(er.pv, 2 * (1 - norm.cdf(abs(t))))


def test_create_metric_df():
    scores_to_metrics_map = {
        "score_1": [
            base.MetricData(
                name="metric_a",
                values=pd.Series([0.3, 0.2, 0.5, np.nan], index=["a", "b", "c", "d"]),
            ),
            base.MetricData(
                name="metric_b",
                values=pd.Series([0.0, 0.1, np.nan, 0.2], index=["a", "d", "c", "b"]),
            ),
        ],
        "score_2": [
            base.MetricData(
                name="metric_a",
                values=pd.Series([1.0, 0.1, 0.4, np.nan], index=["b", "a", "y", "x"]),
            ),
            base.MetricData(
                name="metric_b",
                values=pd.Series([1.0, 0.1, 0.0, np.nan], index=["b", "a", "x", "y"]),
            ),
        ],
    }
    metric_df = summary.create_metric_df(scores_to_metrics_map)
    index = ["a", "b", "c", "d", "x", "y"]
    assert np.allclose(
        metric_df.loc[index, "score_1__metric_a"].to_numpy(),
        np.array([0.3, 0.2, 0.5, np.nan, np.nan, np.nan]),
        equal_nan=True,
    )
    assert np.allclose(
        metric_df.loc[index, "score_1__metric_b"].to_numpy(),
        np.array([0.0, 0.2, np.nan, 0.1, np.nan, np.nan]),
        equal_nan=True,
    )
    assert np.allclose(
        metric_df.loc[index, "score_2__metric_a"].to_numpy(),
        np.array([0.1, 1.0, np.nan, np.nan, np.nan, 0.4]),
        equal_nan=True,
    )
    assert np.allclose(
        metric_df.loc[index, "score_2__metric_b"].to_numpy(),
        np.array([0.1, 1.0, np.nan, np.nan, 0.0, np.nan]),
        equal_nan=True,
    )
