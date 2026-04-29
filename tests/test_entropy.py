"""Tests for entropy scoring."""

import math
from muninn.scoring.entropy import calculate_shannon_entropy

def test_calculate_shannon_entropy_edge_cases():
    # Empty and single items
    assert calculate_shannon_entropy([]) == 0.0
    assert calculate_shannon_entropy([0.8]) == 0.0

    # Negative scores total
    assert calculate_shannon_entropy([-0.5, -0.1]) == 0.0
    assert calculate_shannon_entropy([0.0, 0.0]) == 0.0

    # Uniform distribution max entropy
    n = 4
    assert math.isclose(calculate_shannon_entropy([1.0] * n), math.log2(n))
    assert math.isclose(calculate_shannon_entropy([0.5] * n), math.log2(n))
