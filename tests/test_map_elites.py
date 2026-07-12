"""Tests for MAP-Elites archive geometry and CMA normalization."""

from __future__ import annotations

import numpy as np

from lwi_microbolometer_design.map_elites import (
    archive_coverage_pct,
    bin_coordinates,
    extract_features,
    reachable_cell_count,
    run_cma_me,
)
from lwi_microbolometer_design.map_elites.emitters import OptimizingEmitter
from lwi_microbolometer_design.map_elites.normalization import UnitCubeScaler
from lwi_microbolometer_design.map_elites.visualization import (
    _chromosome_family_labels,
    _graph_component_labels_from_distance_matrix,
    family_labels_for_top_elites,
    family_labels_for_top_elites_graph_threshold,
)


def make_gene_space() -> list[dict[str, float]]:
    """Eight genes: four (mu, sigma) pairs with mixed parameter scales."""
    gene_space: list[dict[str, float]] = []
    for _ in range(4):
        gene_space.append({"low": 4.0, "high": 20.0})
        gene_space.append({"low": 0.1, "high": 4.0})
    return gene_space


def toy_fitness(_ga_instance: object, chromosome: np.ndarray, _solution_idx: int) -> float:
    """Smooth mixed-scale objective with an interior optimum."""
    target = np.array([8.0, 0.7, 11.0, 1.2, 14.0, 0.9, 17.0, 1.5])
    distance = float(np.linalg.norm(chromosome - target))
    return 100.0 - distance


def test_reachable_cell_count_matches_sorted_descriptor_geometry() -> None:
    """Sorted descriptors only fill the upper-triangular half of the grid."""
    chromosome = np.array([15.0, 0.4, 6.0, 1.1, 12.0, 0.8, 4.5, 1.7])

    mu_1, mu_2 = extract_features(chromosome)
    x_bin, y_bin = bin_coordinates(mu_1, mu_2, grid_resolution=20, mu_range=(4.0, 20.0))

    assert reachable_cell_count(20) == 210
    assert x_bin <= y_bin
    assert np.isclose(archive_coverage_pct(210, 20), 100.0)


def test_unit_cube_scaler_round_trip_preserves_mixed_scale_values() -> None:
    """Normalization should not distort wavelengths and widths."""
    scaler = UnitCubeScaler.from_bounds(
        [4.0, 0.1, 4.0, 0.1],
        [20.0, 4.0, 20.0, 4.0],
    )
    chromosome = np.array([15.2, 0.35, 6.4, 3.2])

    normalized = scaler.normalize(chromosome)
    restored = scaler.denormalize(normalized)

    assert np.all(normalized >= 0.0)
    assert np.all(normalized <= 1.0)
    assert np.allclose(restored, chromosome)


def test_optimizing_emitter_returns_bounded_physical_candidates() -> None:
    """Emitter `ask()` should return denormalized candidates inside bounds."""
    bounds_low = [4.0, 0.1, 4.0, 0.1]
    bounds_high = [20.0, 4.0, 20.0, 4.0]
    emitter = OptimizingEmitter(
        x0=np.array([12.0, 1.0, 13.0, 1.5]),
        sigma0=0.2,
        bounds_low=bounds_low,
        bounds_high=bounds_high,
        batch_size=6,
        restart_patience=5,
        seed=123,
    )

    solutions = emitter.ask()

    assert len(solutions) == 6
    for solution in solutions:
        assert solution.shape == (4,)
        assert np.all(solution >= np.array(bounds_low))
        assert np.all(solution <= np.array(bounds_high))

    fitnesses = [float(np.sum(solution)) for solution in solutions]
    improvements = [0.0] * len(solutions)
    improvements[int(np.argmax(fitnesses))] = 1.0
    emitter.tell(solutions, improvements, fitnesses)

    assert not emitter.converged


def test_run_cma_me_tracks_reachable_coverage() -> None:
    """CMA-ME metadata should use the reachable triangular archive size."""
    archive, metadata = run_cma_me(
        fitness_func=toy_fitness,
        gene_space=make_gene_space(),
        grid_resolution=4,
        mu_range=(4.0, 20.0),
        num_initial=32,
        total_evals=128,
        num_emitters=2,
        initial_sigma=0.2,
        restart_patience=5,
        log_interval=32,
        random_seed=7,
        show_progress=False,
    )

    assert metadata["reachable_cells"] == reachable_cell_count(4)
    assert len(archive) <= metadata["reachable_cells"]
    assert 0.0 <= metadata["coverage_pct"] <= 100.0
    assert np.isclose(metadata["coverage_pct"], archive_coverage_pct(len(archive), 4))
    assert all(x_bin <= y_bin for x_bin, y_bin in archive)


def test_chromosome_family_labels_invariant_to_basis_permutation() -> None:
    """Reordering (μ,σ) groups should not create extra families (optimal pairing)."""
    base = np.array([6.0, 0.3, 13.0, 0.3, 16.0, 0.3, 19.0, 0.3], dtype=np.float64)
    permuted = np.array([13.0, 0.3, 6.0, 0.3, 16.0, 0.3, 19.0, 0.3], dtype=np.float64)
    chrom = np.vstack((base, permuted, base, base, base, base, base, base, base, base))
    labels, n_fam = _chromosome_family_labels(chrom)
    assert n_fam == 1
    assert np.all(labels == 0)


def test_family_labels_for_top_elites_shape() -> None:
    """Labels length matches elite list (suite sweep slices this vector for 10/20/50)."""
    base = np.array([6.0, 0.3, 13.0, 0.3, 16.0, 0.3, 19.0, 0.3], dtype=np.float64)
    ranked: list[dict] = [
        {"chromosome": base * (1.0 + 0.002 * i), "fitness": float(100.0 - i)} for i in range(5)
    ]
    lab = family_labels_for_top_elites(ranked)
    assert lab.shape == (5,)


def test_graph_component_labels_single_linkage() -> None:
    """At threshold 0.2, points 0-1 form one component, point 2 is separate."""
    d = np.array(
        [[0.0, 0.1, 1.0], [0.1, 0.0, 1.0], [1.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    lab = _graph_component_labels_from_distance_matrix(d, 0.2)
    assert int(lab[0]) == int(lab[1])
    assert int(lab[2]) != int(lab[0])


def test_family_labels_graph_threshold_uses_tau() -> None:
    """At τ=0, only identical genotypes link; a huge τ gives one component before reordering."""
    c0 = np.array([6.0, 0.3, 13.0, 0.3, 16.0, 0.3, 19.0, 0.3], dtype=np.float64)
    c1 = c0 * 1.0
    c2 = np.array([6.0, 0.3, 13.0, 0.3, 20.0, 2.0, 19.0, 0.3], dtype=np.float64)
    inds: list[dict] = [
        {"chromosome": c0, "fitness": 100.0},
        {"chromosome": c1, "fitness": 99.0},
        {"chromosome": c2, "fitness": 95.0},
    ]
    lab, tau = family_labels_for_top_elites_graph_threshold(inds, distance_threshold=0.0)
    assert tau == 0.0
    assert int(lab[0]) == int(lab[1])
    assert int(lab[2]) != int(lab[0])
    assert len(np.unique(lab)) == 2

    lab_all, t_big = family_labels_for_top_elites_graph_threshold(inds, distance_threshold=1e9)
    assert t_big == 1e9
    assert len(np.unique(lab_all)) == 1
