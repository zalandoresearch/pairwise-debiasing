import numpy as np

from dlambdamart.dlambdamart.calculator import get_query_boundaries, Calculator, MIN_ARG, MAX_ARG


def test_get_query_boundaries():
    assert(get_query_boundaries([2, 3]) == [0, 2, 5])
    assert(get_query_boundaries([1]) == [0, 1])
    assert(get_query_boundaries([1, 1]) == [0, 1, 2])


def test__calculator():
    gains = [0, 2, 1, 1, 0]
    groups = [3, 2]

    calculator = Calculator(gains, groups, 3)

    assert (np.allclose(calculator.discounts[1], 1.0))
    assert (np.allclose(calculator.discounts[2], 1.0 / np.log2(3)))
    assert (np.allclose(calculator.discounts[3], 1.0 / np.log2(4)))

    assert (np.allclose(calculator.inverse_max_dcgs[1], 1.0))
    assert (np.allclose(calculator.inverse_max_dcgs[0], 1 / (2 + 1 / np.log2(3))))
    assert (
        np.allclose(
            calculator.compute_ndcg([-0.5, 1.0, 0.5, 0.5, 1.0]),
            0.5 * (1 + 1 / np.log2(3))
        )
    )
    assert (
        np.allclose(
            calculator.compute_ndcg([-0.5, 1.0, 0.5, 1.0, 0.5]),
            1.0
        )
    )

    assert (
        np.allclose(
            calculator.get_sigmoid(MIN_ARG - 1),
            1.0 / (1 + np.exp(1.0 * MIN_ARG)),
            atol=1e-6
        )
    )
    assert (
        np.allclose(
            calculator.get_sigmoid(MIN_ARG),
            1.0 / (1 + np.exp(1.0 * MIN_ARG)),
            atol=1e-6
        )
    )
    for arg in MIN_ARG \
               + np.random.random(3) * (MAX_ARG - MIN_ARG):
        assert (
            np.allclose(
                calculator.get_sigmoid(arg),
                1.0 / (1 + np.exp(1.0 * arg)),
                atol=1e-6
            )
        )
    assert (
        np.allclose(
            calculator.get_sigmoid(MAX_ARG),
            1.0 / (1 + np.exp(1.0 * MAX_ARG)),
            atol=1e-6
        )
    )
    assert (
        np.allclose(
            calculator.get_sigmoid(MAX_ARG + 1),
            1.0 / (1 + np.exp(1.0 * MAX_ARG)),
            atol=1e-6
        )
    )

    assert (
        np.allclose(
            calculator.get_log(MIN_ARG - 1),
            np.log(1 + np.exp(-1.0 * MIN_ARG)),
            atol=1e-6
        )
    )
    assert (
        np.allclose(
            calculator.get_log(MIN_ARG),
            np.log(1 + np.exp(-1.0 * MIN_ARG)),
            atol=1e-6
        )
    )
    for arg in MIN_ARG \
               + np.random.random(3) * (MAX_ARG - MIN_ARG):
        assert (
            np.allclose(
                calculator.get_log(arg),
                np.log(1 + np.exp(-1.0 * arg)),
                atol=1e-6
            )
        )
    assert (
        np.allclose(
            calculator.get_log(MAX_ARG),
            np.log(1 + np.exp(-1.0 * MAX_ARG)),
            atol=1e-6
        )
    )
    assert (
        np.allclose(
            calculator.get_log(MAX_ARG + 1),
            np.log(1 + np.exp(-1.0 * MAX_ARG)),
            atol=1e-6
        )
    )

    calculator = Calculator(gains, groups, 1)

    assert (np.allclose(calculator.discounts[1], 1.0))
    assert (np.allclose(calculator.discounts[2], 1.0 / np.log2(3)))
    assert (np.allclose(calculator.discounts[3], 1.0 / np.log2(4)))

    assert (np.allclose(calculator.inverse_max_dcgs[0], 0.5))
    assert (
        np.allclose(
            calculator.compute_ndcg([-0.5, 1.0, 0.5, 0.5, 1.0]),
            0.5
        )
    )