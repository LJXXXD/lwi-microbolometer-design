"""Minimal AdvancedGA smoke tests on a toy bimodal landscape."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pygad

from lwi_microbolometer_design.ga import (
    AdvancedGA,
    NichingConfig,
    calculate_population_diversity,
)


def toy_bimodal_fitness(ga_instance: pygad.GA, chromosome: np.ndarray, solution_idx: int) -> float:
    """Two-peak fitness landscape to exercise diversity / niching.

    Peaks near centers c1 and c2 in gene space; fitness is the max of two
    Gaussian bumps so multiple niches can co-exist if niching works.
    """
    _ = ga_instance, solution_idx
    c1 = np.array([5.0, 1.0, 5.0, 1.0, 5.0, 1.0, 5.0, 1.0])
    c2 = np.array([15.0, 1.0, 15.0, 1.0, 15.0, 1.0, 15.0, 1.0])

    d1 = float(np.linalg.norm(chromosome - c1))
    d2 = float(np.linalg.norm(chromosome - c2))

    f1 = float(np.exp(-0.5 * (d1 / 2.0) ** 2))
    f2 = float(np.exp(-0.5 * (d2 / 2.0) ** 2))
    return float(max(f1, f2)) * 100.0


def make_gene_space() -> list[dict[str, float]]:
    """Eight genes: four (mu, sigma) pairs."""
    mu_min, mu_max = 4.0, 20.0
    sigma_min, sigma_max = 0.1, 4.0
    gene_space: list[dict[str, float]] = []
    for _ in range(4):
        gene_space.append({"low": mu_min, "high": mu_max})
        gene_space.append({"low": sigma_min, "high": sigma_max})
    return gene_space


def _diversity_callback(
    history: list[float], niching: NichingConfig | None
) -> Callable[[pygad.GA], None]:
    """Build an on_generation hook that records mean pairwise spread each generation."""

    def on_generation(ga_instance: pygad.GA) -> None:
        if ga_instance.population is None:
            history.append(0.0)
            return
        history.append(float(calculate_population_diversity(ga_instance.population, niching)))

    return on_generation


def _run_ga(niching: NichingConfig | None) -> tuple[float, float]:
    """Return (best fitness, final population diversity) for one run."""
    gene_space = make_gene_space()
    diversity_history: list[float] = []

    ga = AdvancedGA(
        num_generations=80,
        num_parents_mating=20,
        fitness_func=toy_bimodal_fitness,
        sol_per_pop=60,
        num_genes=8,
        gene_space=gene_space,
        mutation_probability=0.15,
        crossover_probability=0.8,
        keep_elitism=4,
        random_seed=123,
        niching_config=niching,
        on_generation=_diversity_callback(diversity_history, niching),
    )
    ga.run()
    _solution, best_fitness, _solution_idx = ga.best_solution()
    final_div = diversity_history[-1] if diversity_history else 0.0
    return float(best_fitness), final_div


def test_reproducibility_and_diversity() -> None:
    """Deterministic reruns and non-regression on final diversity with niching."""
    cfg_no = NichingConfig(enabled=False, use_optimal_pairing=False)

    cfg_yes = NichingConfig(
        enabled=True,
        use_optimal_pairing=True,
        params_per_group=2,
        sigma_share=2.0,
        alpha=1.0,
        distance_metric="euclidean",
    )

    best_no, diversity_no = _run_ga(cfg_no)
    best_yes, diversity_yes = _run_ga(cfg_yes)
    best_yes_2, _ = _run_ga(cfg_yes)

    assert np.isclose(best_yes, best_yes_2)
    assert best_no > 0.0 and best_yes > 0.0
    # Niching should not collapse diversity relative to the non-niched baseline.
    assert diversity_yes >= diversity_no * 0.9
