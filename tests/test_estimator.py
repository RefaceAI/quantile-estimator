import math
import random

import pytest

from quantile_estimator import Estimator


@pytest.mark.parametrize("num_observations", [1, 10, 100, 1000, 10000, 100000])
def test_random_observations(num_observations):
    invariants = (0.5, 0.01), (0.9, 0.01), (0.99, 0.01)
    estimator = Estimator(*invariants)

    values = [random.uniform(0, 100) for _ in range(num_observations)]
    for value in values:
        estimator.observe(value)

    values.sort()
    for quantile, inaccuracy in invariants:
        min_rank = math.floor(quantile * num_observations - inaccuracy * num_observations)
        max_rank = min(math.ceil(quantile * num_observations + inaccuracy * num_observations), num_observations - 1)
        assert 0 <= values[min_rank] <= estimator.query(quantile) <= values[max_rank] <= 100


def test_border_invariants():
    estimator = Estimator((0.0, 0.0), (1.0, 0.0))

    values = [random.uniform(0, 100) for _ in range(500)]
    for value in values:
        estimator.observe(value)

    assert estimator.query(0.0) == min(values)
    assert estimator.query(1.0) == max(values)
