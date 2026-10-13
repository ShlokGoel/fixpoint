# Held-out test for the toy task (the "FAIL_TO_PASS" test). The agent never sees it;
# scripts/demo_toy.py copies it in only after the episode ends, to grade the patch.
from stats_utils import median


def test_median_even():
    assert median([1, 2, 3, 4]) == 2.5
    assert median([4, 1]) == 2.5
