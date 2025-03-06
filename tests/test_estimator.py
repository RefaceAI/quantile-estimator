import random

import pytest

from quantile_estimator import Estimator


@pytest.mark.parametrize("num_observations", [1, 10, 100, 1000, 10000, 100000])
def test_random_observations(num_observations):
    estimator = Estimator()
    for _ in range(num_observations):
        estimator.observe(random.randint(1, 1000) / 100)

    assert 0 <= estimator.query(0.5) <= estimator.query(0.9) <= estimator.query(0.99) <= 10


def test_border_invariants():
    estimator = Estimator((0.0, 0.0), (1.0, 0.0))

    values = [random.randint(1, 1000) for _ in range(1000)]
    for x in values:
        estimator.observe(x)

    assert estimator.query(0) == min(values)
    assert estimator.query(1) == max(values)
