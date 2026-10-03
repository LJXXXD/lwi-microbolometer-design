"""Experiment overrides and worker metadata without writing production outputs."""

from concurrent.futures import Future
import importlib
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import pytest

from lwi_microbolometer_design.ga import tuning
from lwi_microbolometer_design.map_elites import run_map_elites
from lwi_microbolometer_design.map_elites.normalization import UnitCubeScaler


class StopBeforeWorker(RuntimeError):
    """Stop an entrypoint after capturing the real worker configuration."""


@pytest.fixture
def suite(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts" / "v2_paper_2026"))
    monkeypatch.setattr(mp, "set_start_method", lambda *a, **k: None)
    return importlib.import_module("_budgets")


@pytest.mark.parametrize("quick,expected", [(False, (17, 13)), (True, (12, 20))])
def test_multistart_forwards_per_run_overrides_and_explicit_workers(
    monkeypatch, tmp_path, suite, quick, expected
):
    module = importlib.import_module("run_03_multi_start_ga")
    monkeypatch.setattr(suite, "MULTI_START_GENERATIONS", 17)
    monkeypatch.setattr(suite, "MULTI_START_POP", 13)
    monkeypatch.setattr(suite, "MULTI_START_NUM_RUNS", 3)
    monkeypatch.setattr(module._paths, "step_output_dir", lambda _: tmp_path)
    captured = {}

    class Executor:
        def __init__(self, *, max_workers):
            captured["workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, worker, payload):
            captured.update(payload)
            raise StopBeforeWorker

    monkeypatch.setattr(module, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(
        sys, "argv", [module.__file__, "--max-workers", "3"] + (["--quick"] if quick else [])
    )
    with pytest.raises(StopBeforeWorker):
        module.main()
    assert captured["workers"] == 3
    assert (captured["num_generations"], captured["sol_per_pop"]) == expected
    assert captured["random_seed"] == 42 and captured["mutation_type"] == "random"


def test_map_budget_rejects_seed_count_exceeding_total(monkeypatch, suite):
    monkeypatch.setattr(suite, "MAP_ELITES_TOTAL_FITNESS_EVALS", 3)
    monkeypatch.setattr(suite, "MAP_ELITES_NUM_INITIAL", 4)
    with pytest.raises(ValueError, match="PRES_MAP_ELITES_INITIAL"):
        suite.map_elites_budget()


def test_advanced_ga_uses_its_own_environment_budget(monkeypatch, tmp_path, suite):
    module = importlib.import_module("run_02_advanced_ga_niching")
    monkeypatch.setattr(suite, "GA_NUM_GENERATIONS_ADVANCED", 7)
    monkeypatch.setattr(suite, "GA_SOL_PER_POP_ADVANCED", 13)
    monkeypatch.setattr(module._paths, "step_output_dir", lambda _: tmp_path)
    captured = {}

    def create_ga(**kwargs):
        captured.update(kwargs)
        raise StopBeforeWorker

    monkeypatch.setattr(module, "AdvancedGA", create_ga)
    monkeypatch.setattr(sys, "argv", [module.__file__])
    with pytest.raises(StopBeforeWorker):
        module.main()
    assert captured["num_generations"] == 7 and captured["sol_per_pop"] == 13


def test_quick_hill_climb_keeps_explicit_worker_override(monkeypatch, tmp_path, suite):
    module = importlib.import_module("run_05_map_elites_hc")
    monkeypatch.setattr(module._paths, "step_output_dir", lambda _: tmp_path)
    archive = {
        (i, i): {
            "chromosome": np.array([5.0, 1.0, 9.0, 1.0, 13.0, 1.0, 18.0, 1.0]),
            "fitness": 46.0 + i,
        }
        for i in range(3)
    }
    monkeypatch.setattr(module, "run_map_elites", lambda **kwargs: archive)
    monkeypatch.setattr(module._plotting, "plot_descriptor_scatter", lambda *a, **k: None)
    monkeypatch.setattr(
        module._plotting, "export_presentation_top_elite_spectra_sweep", lambda *a, **k: None
    )
    captured = {}

    def executor(*, max_workers):
        captured["workers"] = max_workers
        raise StopBeforeWorker

    monkeypatch.setattr(module, "ProcessPoolExecutor", executor)
    monkeypatch.setattr(sys, "argv", [module.__file__, "--quick", "--hc-workers", "3"])
    with pytest.raises(StopBeforeWorker):
        module.main()
    assert captured["workers"] == 3


@pytest.mark.parametrize("iterations,probability", [(-1, 0.1), (1, np.nan), (1, 1.1)])
def test_invalid_map_settings_fail_before_any_fitness_call(iterations, probability):
    calls = []
    with pytest.raises(ValueError):
        run_map_elites(
            lambda *args: calls.append(args),
            [{"low": 0, "high": 1}] * 4,
            num_iterations=iterations,
            mutation_probability=probability,
        )
    assert calls == []


@pytest.mark.parametrize("low,high", [([0], [np.nan]), ([0], [np.inf]), ([[0]], [[1]]), ([], [])])
def test_cma_scaler_rejects_invalid_bounds(low, high):
    with pytest.raises(ValueError):
        UnitCubeScaler.from_bounds(low, high)


def test_tuner_forwards_seed_and_records_success_counts(monkeypatch, tmp_path):
    captured = []

    class Executor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, worker, config, *args):
            captured.append(args[-1])
            future = Future()
            future.set_result(tuning.TuningResult(config, 45.0, 20.0, 2.0, 1, 1))
            return future

    monkeypatch.setattr(tuning, "ProcessPoolExecutor", Executor)
    tuner = tuning.HyperparameterTuner(
        lambda *_: 1.0,
        [{"low": 0, "high": 1}] * 4,
        tuning.HyperparameterSearchSpace(),
        2,
        num_runs=1,
        max_workers=1,
        random_seed_base=321,
    )
    monkeypatch.setattr(tuner, "generate_configurations", lambda: [{"sol_per_pop": 8}])
    tuner.tune(tmp_path / "nested")
    assert captured == [321]
    summary_path = next((tmp_path / "nested").glob("tuning_summary_*.json"))
    summary = json.loads(summary_path.read_text())
    assert summary["random_seed_base"] == 321
    assert summary["successful_configurations"] == 1
    assert summary["failed_configurations"] == 0
