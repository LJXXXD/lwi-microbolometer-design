"""Independent expectations for distance, clustering and plotting contracts."""

from itertools import permutations
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from lwi_microbolometer_design.analysis import (
    calculate_optimal_pairing_distance,
    compute_distance_matrix,
    spectral_angle_mapper,
)
from lwi_microbolometer_design.analysis.clodd import (
    clodd_partition,
    normalize_ordered_dissimilarity,
)
from lwi_microbolometer_design.analysis.elite_clustering import (
    graph_single_linkage_from_distance_matrix,
)
from lwi_microbolometer_design.ga import population_analysis
from lwi_microbolometer_design.map_elites.visualization import plot_polished_elites
from lwi_microbolometer_design.simulation import gaussian_parameters_to_unit_amplitude_curves
from lwi_microbolometer_design.visualization import (
    visualize_distance_matrix,
    visualize_sensor_output,
)


def test_batched_sam_matches_independent_scalar_pairs_and_negative_axis():
    vectors = np.array(
        [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [1, 2, 3], [0, 0, 0], [1e-16, 0, 0]],
        dtype=float,
    )
    expected = np.zeros((len(vectors), len(vectors)))
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            expected[i, j] = expected[j, i] = spectral_angle_mapper(vectors[i], vectors[j])
    actual = compute_distance_matrix(vectors.T, spectral_angle_mapper, axis=-1)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert actual[0, 1] == 90 and actual[0, 2] == 180
    np.testing.assert_array_equal(actual[4:], np.zeros((2, 6)))
    with pytest.raises(ValueError, match="out of bounds"):
        compute_distance_matrix(vectors, spectral_angle_mapper, axis=-3)


def test_optimal_pairing_matrix_matches_brute_force_assignments():
    chromosomes = np.array([[0, 1, 4, 2, 9, 3], [9, 3, 0, 1, 4, 2], [1, 2, 5, 4, 8, 1]])
    actual = compute_distance_matrix(
        chromosomes, use_optimal_pairing=True, params_per_group=2, metric="euclidean"
    )
    expected = np.zeros((3, 3))
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = chromosomes[i].reshape(3, 2), chromosomes[j].reshape(3, 2)
            expected[i, j] = expected[j, i] = min(
                sum(np.linalg.norm(a[k] - b[p[k]]) for k in range(3))
                for p in permutations(range(3))
            )
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert actual[0, 1] == 0


def test_mahalanobis_pairing_supports_single_component_groups():
    distance = calculate_optimal_pairing_distance([0.0, 2.0], [1.0, 3.0], metric="mahalanobis")
    assert distance == pytest.approx(2 * np.sqrt(3 / 5))


def test_graph_single_linkage_handles_a_chain_longer_than_recursion_limit():
    n = 1100
    d = np.full((n, n), 2.0)
    np.fill_diagonal(d, 0.0)
    d[np.arange(n - 1), np.arange(1, n)] = 1.0
    d[np.arange(1, n), np.arange(n - 1)] = 1.0
    labels = graph_single_linkage_from_distance_matrix(d, 1.0)
    np.testing.assert_array_equal(labels, np.zeros(n, dtype=int))


def test_clodd_rejects_unknown_search_and_normalizes_singleton():
    with pytest.raises(ValueError, match="mode"):
        clodd_partition(np.ones((4, 4)), mode="unknown")
    np.testing.assert_array_equal(normalize_ordered_dissimilarity(np.array([[7.0]])), [[0.0]])
    with pytest.raises(ValueError, match="square"):
        normalize_ordered_dissimilarity(np.ones((3, 2)))


def test_kmeans_is_available_when_density_clustering_finds_no_clusters(monkeypatch):
    genes = np.array([[-4, -4], [-4, -3], [-3, -4], [3, 4], [4, 3], [4, 4]], dtype=float)
    solutions = [SimpleNamespace(genes=g, fitness=100 - i) for i, g in enumerate(genes)]
    monkeypatch.setattr(population_analysis, "_adaptive_dbscan", lambda _: {"n_clusters": 0})
    monkeypatch.setattr(population_analysis, "_find_optimal_k", lambda *_: 2)
    result = population_analysis._perform_clustering_analysis(
        solutions,
        genes,
        {"top_n": 6},
        lambda a, b: np.linalg.norm(a - b),
        population_analysis.AnalysisConfig(),
    )
    assert result["method"] == "kmeans"
    assert result["cluster_count"] == 2
    assert result["silhouette_score"] > 0.8


def test_default_plots_accept_missing_labels_and_use_radiance_units(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        visualize_distance_matrix(np.array([[0.0, 90.0], [90.0, 0.0]]))
        assert len(plt.gca().get_xticklabels()) == 2
        visualize_sensor_output(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert "W/(m²·sr)" in plt.gca().get_ylabel()
    finally:
        plt.close("all")


def test_polish_labels_follow_the_result_record_after_completion_order_changes(
    monkeypatch, tmp_path
):
    close = plt.close
    monkeypatch.setattr(plt, "close", lambda *args: None)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda *args, **kwargs: None)
    chromosome = np.array([5.0, 1.0, 12.0, 2.0])
    records = [
        dict(
            elite_id=1,
            initial_fitness=2.0,
            polished_fitness=20.0,
            fitness_gain=18.0,
            polished_chromosome=chromosome,
        ),
        dict(
            elite_id=0,
            initial_fitness=1.0,
            polished_fitness=10.0,
            fitness_gain=9.0,
            polished_chromosome=chromosome,
        ),
    ]
    try:
        plot_polished_elites(
            [{"fitness": 1.0}, {"fitness": 2.0}],
            records,
            np.array([4.0, 8.0, 12.0, 16.0]),
            gaussian_parameters_to_unit_amplitude_curves,
            tmp_path / "unused.png",
        )
        text = [t.get_text() for t in plt.gca().texts]
        assert "2.00 → 20.00" in text[0]
        assert "1.00 → 10.00" in text[1]
    finally:
        close("all")
