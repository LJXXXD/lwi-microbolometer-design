"""Tests for CLODD (Clustering in Ordered Dissimilarity Data)."""

import numpy as np

from lwi_microbolometer_design.analysis.clodd import (
    clodd_objective,
    clodd_partition,
    eedge,
    esquare,
    labels_from_cuts,
    map_labels_to_original_order,
    normalize_ordered_dissimilarity,
)


def test_havens_example1_partition_prefers_two_two_split() -> None:
    """Figure 3 / Example 1: U = {2:3} should score higher than V = {3:2}."""
    d_star = np.array(
        [
            [0.0, 0.12, 0.59, 0.73, 0.78],
            [0.12, 0.0, 0.55, 0.71, 0.74],
            [0.59, 0.55, 0.0, 0.19, 0.19],
            [0.73, 0.71, 0.19, 0.0, 0.16],
            [0.78, 0.74, 0.19, 0.16, 0.0],
        ],
        dtype=np.float64,
    )
    d_n = normalize_ordered_dissimilarity(d_star)
    u_cuts = (2,)
    v_cuts = (3,)
    assert esquare(d_n, u_cuts) > esquare(d_n, v_cuts)
    assert eedge(d_n, u_cuts) > eedge(d_n, v_cuts)
    gamma = 0.5  # a = 2.5 > min cluster size 2 → spline = 1 for both partitions
    assert clodd_objective(d_n, u_cuts, alpha=0.5, gamma=gamma) > clodd_objective(
        d_n, v_cuts, alpha=0.5, gamma=gamma
    )


def test_clodd_exhaustive_finds_expected_cuts() -> None:
    d_star = np.array(
        [
            [0.0, 0.12, 0.59, 0.73, 0.78],
            [0.12, 0.0, 0.55, 0.71, 0.74],
            [0.59, 0.55, 0.0, 0.19, 0.19],
            [0.73, 0.71, 0.19, 0.0, 0.16],
            [0.78, 0.74, 0.19, 0.16, 0.0],
        ],
        dtype=np.float64,
    )
    d_n = normalize_ordered_dissimilarity(d_star)
    res = clodd_partition(d_n, c_max=4, mode="exhaustive", gamma=0.5)
    assert res.n_clusters == 2
    assert res.cuts == (2,)


def test_labels_and_inverse_mapping() -> None:
    labels = labels_from_cuts(5, (2, 4))
    assert list(labels) == [0, 0, 1, 1, 2]
    reorder = [2, 0, 4, 1, 3]
    orig = map_labels_to_original_order(labels, reorder)
    assert orig[2] == 0 and orig[0] == 0
    assert orig[4] == 1 and orig[1] == 1
    assert orig[3] == 2
