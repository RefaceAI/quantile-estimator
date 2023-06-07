import math
from time import sleep
from unittest.mock import MagicMock

import pytest

from quantile_estimator import TimeWindowEstimator

INTERVAL_BETWEEN_OBSERVATIONS_SECONDS = 0.1


def get_estimator(max_age_seconds):
    estimator = TimeWindowEstimator(max_age_seconds=max_age_seconds)
    estimator.rotate_buckets = MagicMock(side_effect=estimator.rotate_buckets)
    estimator.get_new_bucket = MagicMock(side_effect=estimator.get_new_bucket)
    return estimator


@pytest.mark.parametrize(
    "max_age_seconds,num_observations,expected_rotations",
    [
        (200, 30, 0),
        (10, 30, 1),
        (5, 35, 3),
        (5, 75, 7),
    ],
)
def test_estimator_buckets_rotation(max_age_seconds, num_observations, expected_rotations):
    estimator = get_estimator(max_age_seconds=max_age_seconds)
    sum_observations = 0
    for i in range(num_observations):
        sleep(INTERVAL_BETWEEN_OBSERVATIONS_SECONDS)
        sum_observations += i
        estimator.observe(i)

    assert estimator._sum == sum_observations
    assert estimator._observations == num_observations
    assert estimator.rotate_buckets.call_count == num_observations

    assert estimator.current_bucket == expected_rotations % estimator.age_buckets
    assert estimator.get_new_bucket.call_count == expected_rotations


def test_query_non_empty_estimator():
    estimator = get_estimator(max_age_seconds=5)
    estimator.observe(2)
    assert not math.isnan(estimator.query(0.5))


def test_query_empty_estimator():
    estimator = get_estimator(max_age_seconds=5)
    assert math.isnan(estimator.query(0.5))
