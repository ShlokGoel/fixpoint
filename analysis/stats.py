"""Small-sample statistics for resolve rates and ablation deltas.

At n=30 one task is 3.3 percentage points, so every rate is reported with an
interval and every ablation delta with a paired test. Paired, because both arms
run on the same instances: the comparison should use per-task outcomes, not two
independent proportions.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def paired_bootstrap_delta(
    a: Sequence[int], b: Sequence[int], n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean(b - a) with a 95% percentile-bootstrap CI over tasks."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    n = len(a)
    if n == 0:
        return (0.0, 0.0, 0.0)
    diffs = [bi - ai for ai, bi in zip(a, b, strict=True)]
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return (sum(diffs) / n, percentile(means, 0.025), percentile(means, 0.975))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)
