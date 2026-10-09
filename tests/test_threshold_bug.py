"""The 99th-percentile threshold bug, on scores with a known distribution.

Normal scores are drawn from N(0, 1), so the true false-alarm rate of any
threshold t is exactly P(Z > t). The sizes are carpet's: the old rule had 28
held-out images, the 5-fold rule scores all 280.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "edd"))

from threshold_bug import synthetic  # noqa: E402

OLD = "old: 10% holdout, 99th percentile"
Q99 = "5-fold, 99th percentile"
FIX = "5-fold, tolerance bound (shipped)"


@pytest.fixture(scope="module")
def res():
    return synthetic(trials=1500)


def test_old_rule_misses_the_target_most_of_the_time(res):
    # 20000 trials in the script give a mean of 4.0%, four times the target
    assert res[OLD]["mean_far"] > 0.03
    assert res[OLD]["share_of_runs_above_1pct"] > 0.75


def test_more_calibration_data_alone_is_not_enough(res):
    # a point estimate lands on the wrong side of 1% about half the time or more
    assert 0.01 < res[Q99]["mean_far"] < 0.02
    assert res[Q99]["share_of_runs_above_1pct"] > 0.5


def test_tolerance_bound_holds_the_target(res):
    # n = 280 is below the 299 the 95% guarantee needs, so the rule falls back
    # to the sample maximum, which exceeds 1% FAR with probability 0.99^280 = 6.0%
    assert res[FIX]["mean_far"] < 0.01
    assert res[FIX]["share_of_runs_above_1pct"] == pytest.approx(0.99**280, abs=0.02)
