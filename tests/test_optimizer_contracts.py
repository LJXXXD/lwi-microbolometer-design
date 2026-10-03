"""Optimizer budgets, reproducibility, and shared fitness regressions."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from lwi_microbolometer_design.analysis import spectral_angle_mapper, min_based_dissimilarity_score
from lwi_microbolometer_design.analysis.robustness import (
    evaluate_elite_fitness,
    evaluate_solutions_robustness,
    summarise_robustness,
)
from lwi_microbolometer_design.data import SceneConfig
from lwi_microbolometer_design.ga import AdvancedGA, NichingConfig, MutationConfig
from lwi_microbolometer_design.ga.fitness import MinDissimilarityFitnessEvaluator
from lwi_microbolometer_design.ga.mutations import diversity_preserving_mutation
from lwi_microbolometer_design.ga.result_extraction import extract_basic_results
from lwi_microbolometer_design.ga.ga_configuration import load_ga_configuration_from_csv
from lwi_microbolometer_design.map_elites import cma_me, run_cma_me
from lwi_microbolometer_design.map_elites.polish import (
    polish_single_elite_cma,
    polish_single_elite_hc,
)
from lwi_microbolometer_design.simulation import gaussian_parameters_to_unit_amplitude_curves


def small_scene():
    return SceneConfig(
        np.array([4.0, 6.0, 10.0, 16.0]),
        np.array([[0.8, 0.1, 0.3], [0.4, 0.7, 0.2], [0.2, 0.4, 0.9], [0.7, 0.2, 0.3]]),
        np.array([0.1, 0.3, 0.8, 1.0]),
        300.0,
        0.1,
        1.0,
        np.array(["a", "b", "c"]),
    )


def evaluator(
    scene=None, aggregation="single", callback=gaussian_parameters_to_unit_amplitude_curves
):
    return MinDissimilarityFitnessEvaluator(
        scene or small_scene(), callback, 2, aggregation=aggregation
    )


@pytest.mark.parametrize("budget", [1, 2, 7, 8, 31])
def test_cma_me_honors_total_budget_and_inserts_zero_fitness_new_cells(monkeypatch, budget):
    calls = []
    tells = []

    class Emitter:
        batch_size = 6
        converged = False
        total_restarts = 0

        def __init__(self, **kwargs):
            pass

        def ask(self):
            return [np.array([4.0, 0.1, 4.0, 0.1, 4.0, 0.1, 4.0, 0.1]) for _ in range(6)]

        def tell(self, solutions, improvements, fitnesses):
            tells.append(len(solutions))
            assert len(solutions) == 6

    monkeypatch.setattr(cma_me, "OptimizingEmitter", Emitter)

    def fitness(_, chromosome, index):
        calls.append(chromosome.copy())
        return 0.0

    gene_space = [{"low": 4.0, "high": 20.0}, {"low": 0.1, "high": 4.0}] * 4
    archive, metadata = run_cma_me(
        fitness,
        gene_space,
        num_initial=1,
        total_evals=budget,
        num_emitters=2,
        random_seed=0,
        show_progress=False,
    )
    assert len(calls) == budget == metadata["total_evals"]
    assert metadata["history"]["evals"][-1] == budget
    assert len(tells) == (budget - 1) // 6
    if budget > 1:
        assert (0, 0) in archive
        assert archive[(0, 0)]["fitness"] == 0.0


@pytest.mark.parametrize("budget", [0, 1, 7, 13])
def test_cma_polish_counts_all_calls_and_never_exceeds_budget(budget):
    calls = []

    def objective(_, chromosome, index):
        calls.append(chromosome.copy())
        return 100.0 - float(np.linalg.norm(chromosome - 1.0))

    result = polish_single_elite_cma(
        0,
        np.full(4, 0.5),
        99.0,
        objective,
        [{"low": 0.0, "high": 2.0}] * 4,
        max_fevals=budget,
        population_size=6,
        random_seed=123,
    )
    assert len(calls) == result["fevals_used"] <= budget
    assert result["polished_fitness"] >= 99.0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_optimizer_rejects_nonfinite_callback_scores(bad):
    with pytest.raises(ValueError, match="finite scalar"):
        run_cma_me(
            lambda *_: bad,
            [{"low": 4.0, "high": 20.0}, {"low": 0.1, "high": 4.0}] * 2,
            num_initial=1,
            total_evals=1,
            num_emitters=1,
            show_progress=False,
        )


def test_seeded_hill_climb_reproduces_worker_result():
    kwargs = dict(
        elite_id=0,
        chromosome=np.array([0.3, 0.4, 0.5, 0.6]),
        initial_fitness=0.0,
        fitness_func=lambda _, x, i: float(x.sum()),
        gene_space=[{"low": 0.0, "high": 1.0}] * 4,
        num_iterations=40,
        adaptive_iterations=False,
        random_seed=123,
    )
    a, b = polish_single_elite_hc(**kwargs), polish_single_elite_hc(**kwargs)
    np.testing.assert_array_equal(a["polished_chromosome"], b["polished_chromosome"])


@pytest.mark.parametrize("polisher", [polish_single_elite_hc, polish_single_elite_cma])
@pytest.mark.parametrize(
    "chromosome,fitness", [(np.ones(4), np.nan), (np.full(4, np.nan), 1.0), (np.full(4, 2.0), 1.0)]
)
def test_polish_rejects_invalid_incumbents_before_evaluating(polisher, chromosome, fitness):
    calls = []
    with pytest.raises(ValueError):
        polisher(
            0, chromosome, fitness, lambda *args: calls.append(args), [{"low": 0, "high": 1}] * 4
        )
    assert calls == []


def mutation_instance(space, nested, offspring):
    return SimpleNamespace(
        gene_space=space,
        gene_space_nested=nested,
        mutation_probability=0.5,
        population=offspring,
        num_generations=20,
        generations_completed=2,
        best_solutions_fitness=[],
        mutation_config=MutationConfig.aggressive(),
    )


def test_custom_mutation_uses_seed_for_every_draw():
    offspring = np.full((20, 4), 0.5)
    ga = mutation_instance([{"low": 0.0, "high": 1.0}] * 4, True, offspring)
    np.random.seed(123)
    a = diversity_preserving_mutation(offspring, ga)
    np.random.seed(123)
    b = diversity_preserving_mutation(offspring, ga)
    np.testing.assert_array_equal(a, b)
    np.random.seed(124)
    c = diversity_preserving_mutation(offspring, ga)
    assert not np.array_equal(a, c)
    np.testing.assert_array_equal(offspring, np.full((20, 4), 0.5))


@pytest.mark.parametrize("space,nested", [([[3.0, 7.0]] * 4, True), ([3.0, 7.0, 9.0, 11.0], False)])
def test_discrete_gene_spaces_do_not_become_ranges_or_fixed_per_gene(space, nested):
    offspring = np.full((80, 4), 3.0)
    ga = mutation_instance(space, nested, offspring)
    np.random.seed(1)
    out = diversity_preserving_mutation(offspring, ga)
    allowed = [3.0, 7.0] if nested else space
    assert np.all(np.isin(out, allowed))
    assert np.any(out != 3.0)


def test_step_space_stays_on_high_exclusive_pygad_grid():
    offspring = np.full((100, 4), 0.9)
    ga = mutation_instance([{"low": 0.0, "high": 1.0, "step": 0.3}] * 4, True, offspring)
    np.random.seed(1)
    out = diversity_preserving_mutation(offspring, ga)
    assert np.all(out < 1.0)
    np.testing.assert_allclose(out / 0.3, np.rint(out / 0.3), atol=1e-14)


def test_robustness_uses_identical_fitness_contract_and_aggregation():
    scene = small_scene()
    scenes = [scene, replace(scene, temperature_k=500.0, atmospheric_distance_ratio=0.8)]
    chrom = np.array([6.0, 1.0, 14.0, 2.0])
    scores = [evaluator(s).fitness_func(None, chrom, 0) for s in scenes]
    assert scores[0] != pytest.approx(scores[1])
    for aggregation, expected in [("min", min(scores)), ("mean", np.mean(scores))]:
        assert evaluator(scenes, aggregation).fitness_func(None, chrom, 0) == pytest.approx(
            expected
        )
    results = evaluate_solutions_robustness(
        [{"chromosome": chrom, "fitness": 1.0}],
        scenes,
        gaussian_parameters_to_unit_amplitude_curves,
        2,
        show_progress=False,
    )
    np.testing.assert_allclose(results[0].fitness_per_condition, scores, rtol=1e-14)
    assert (
        evaluate_elite_fitness(chrom, scene, gaussian_parameters_to_unit_amplitude_curves, 2)
        == scores[0]
    )
    for call in [
        lambda: evaluator(scene).fitness_func(None, chrom[:-1], 0),
        lambda: evaluate_elite_fitness(
            chrom[:-1], scene, gaussian_parameters_to_unit_amplitude_curves, 2
        ),
    ]:
        with pytest.raises(ValueError, match="divisible"):
            call()


def test_nonfinite_callback_is_not_a_degenerate_score():
    e = evaluator(callback=lambda params, wl: np.full((len(wl), len(params)), np.nan))
    with pytest.raises(ValueError, match="finite"):
        e.fitness_func(None, np.array([6.0, 1.0, 14.0, 2.0]), 0)
    with pytest.raises(ValueError, match="finite"):
        spectral_angle_mapper([0.0, 0.0], [np.nan, 1.0])


def test_sam_large_vectors_and_threshold_boundary():
    assert spectral_angle_mapper([1e300, 0.0], [0.0, 1e300]) == 90.0
    assert spectral_angle_mapper([1e-15, 0.0], [0.0, 1e-15]) == 90.0
    assert spectral_angle_mapper([np.nextafter(1e-15, 0.0), 0.0], [0.0, 1e-15]) == 0.0
    assert (
        min_based_dissimilarity_score(
            distance_matrix=np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 2.0], [3.0, 2.0, 0.0]])
        )
        == 0.0
    )
    with pytest.raises(ValueError, match="two items"):
        min_based_dissimilarity_score(distance_matrix=np.zeros((1, 1)))
    with pytest.raises(ValueError, match="empty"):
        summarise_robustness([])


def test_ga_config_csv_defaults_missing_cells_and_honors_mutation(tmp_path):
    csv = tmp_path / "config.csv"
    csv.write_text(
        "num_generations,sol_per_pop,num_parents_mating,mutation_type,k_tournament,niching_enabled\n2,8,4,random,2,false\n"
    )
    config = load_ga_configuration_from_csv(csv)
    assert config["mutation_type"] == "random"
    assert config["K_tournament"] == 2
    assert not config["niching_config"].enabled
    csv.write_text("niching_enabled,num_generations\nfalse,\n")
    assert load_ga_configuration_from_csv(csv)["num_generations"] == 2000
    csv.write_text("niching_enabled\nunknown\n")
    with pytest.raises(ValueError, match="Invalid boolean"):
        load_ga_configuration_from_csv(csv)


def test_fitness_sharing_matches_scalar_formula_and_restores_on_callback_error():
    config = NichingConfig(enabled=True, use_optimal_pairing=False, sigma_share=1.5, alpha=0.5)
    ga = AdvancedGA(
        1,
        2,
        lambda _, x, i: float(x.sum()),
        4,
        2,
        niching_config=config,
        mutation_config=MutationConfig.conservative(),
        keep_elitism=1,
        random_seed=42,
        mutation_probability=0.1,
    )
    ga.population = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    fitness = np.array([8.0, 6.0, 4.0, 2.0])
    ga.last_generation_fitness = fitness.copy()
    expected = []
    for i in range(4):
        count = 1.0 + sum(
            1.0 - (abs(ga.population[i, 0] - ga.population[j, 0]) / 1.5) ** 0.5
            for j in range(4)
            if j != i and abs(ga.population[i, 0] - ga.population[j, 0]) < 1.5
        )
        expected.append(fitness[i] / count)
    np.testing.assert_allclose(ga._calculate_shared_fitness(), expected, rtol=1e-14)

    def fail(*args):
        raise RuntimeError("callback failure")

    ga.on_parents = fail
    with pytest.raises(RuntimeError, match="callback failure"):
        ga.run_select_parents()
    np.testing.assert_array_equal(ga.last_generation_fitness, fitness)


def test_result_extraction_does_not_invent_history_or_hide_failures():
    ga = AdvancedGA(
        2,
        2,
        lambda _, x, i: float(x.sum()),
        4,
        2,
        keep_elitism=1,
        random_seed=42,
        mutation_probability=0.1,
    )
    ga.run()
    result = extract_basic_results(ga)
    assert result["mean_fitness_history"] == []
    assert result["diversity_history"] == []
    assert result["final_diversity"] >= 0.0
    ga.mean_fitness_history = [1.0, 2.0]
    ga.diversity_history = [3.0, 4.0]
    assert extract_basic_results(ga)["mean_fitness_history"] == [1.0, 2.0]
    ga.best_solution = lambda: (_ for _ in ()).throw(RuntimeError("result failure"))
    with pytest.raises(RuntimeError, match="result failure"):
        extract_basic_results(ga)
