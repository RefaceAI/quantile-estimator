import random

import pytest

from quantile_estimator import Estimator


@pytest.mark.parametrize("num_observations", [1, 10, 100, 1000, 10000, 100000])
def test_random_observations(num_observations):
    estimator = Estimator()
    sum_observations = 0
    for _ in range(num_observations):
        value = random.randint(1, 1000)
        estimator.observe(value)
        sum_observations += value

    assert 1 <= estimator.query(0.5) <= estimator.query(0.9) <= estimator.query(0.99) <= 1000
