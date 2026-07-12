# System Audit & Architectural Planning Report

**Project:** `lwi-microbolometer-design`
**Date:** 2026-03-21
**Scope:** Full codebase audit with hyper-focus on `simulation/sensor_simulation.py`, `ga/fitness.py`, and the `analysis/` directory.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Data & Physical Reality](#2-the-data--physical-reality)
3. [GA-Simulation Interface Diagnosis](#3-ga-simulation-interface-diagnosis)
4. [SENS-D Legacy Assessment: `analysis/` Module](#4-sens-d-legacy-assessment-analysis-module)
5. [Structural Anti-Pattern Inventory](#5-structural-anti-pattern-inventory)
6. [Strategic Pythonic Refactoring Plan](#6-strategic-pythonic-refactoring-plan)
7. [Dependency Graph & Module Coupling Map](#7-dependency-graph--module-coupling-map)
8. [Prioritized Action Items](#8-prioritized-action-items)

---

## 1. Executive Summary

This codebase implements a sensor design optimization pipeline for infrared microbolometer arrays. The physical problem: given a set of target substances with known IR emissivity spectra and fixed atmospheric conditions, find the optimal set of Gaussian basis functions (representing sensor filter response curves) that maximizes the spectral distinguishability between all substance pairs.

**The pipeline in one sentence:** A GA chromosome encodes Gaussian parameters `[(mu, sigma), ...]`, which get converted to sensor response curves, convolved with physics (blackbody + atmosphere + emissivity), producing a `(m, n)` signal matrix, from which a pairwise SAM distance matrix yields a scalar fitness score.

**Current state:** The physics simulation layer (`simulation/`) is clean and well-isolated. The fitness evaluator (`ga/fitness.py`) is well-designed with proper dependency injection via `parameters_to_curves`. The `analysis/` module is solid but has an identity problem — it serves two masters (post-hoc analysis and in-loop fitness computation) without a clear boundary. The `ga/` package has grown organically and contains significant complexity in `mutations.py`, `tuning.py`, and `analysis.py` that could benefit from decomposition.

**Primary risks:**
- The `dict[str, np.ndarray | float]` return type from `load_substance_atmosphere_data` is a Data Clump that propagates untyped, unvalidated data through the entire system.
- The `ga/tuning.py` module (~1000 lines) is a God Module that conflates experiment configuration, fitness evaluator construction, data loading, GA execution, and visualization.
- DEBUG `print()` statements are live in production code (`advanced_ga.py:221`, `mutations.py:607`).

---

## 2. The Data & Physical Reality

### 2.1 ELI5: What Flows Through This System

Imagine you have 4 white powders on a table. Each powder glows differently in infrared light — that "glow pattern" across wavelengths is its **emissivity spectrum**. You're building a special camera with 4 filters (each shaped like a bell curve). Each filter lets through a different slice of the infrared spectrum. When you point this camera at a powder, each filter produces a single number (voltage). So for one powder, you get 4 numbers. For 4 powders, you get a 4×4 table of numbers.

The question is: **can you tell the powders apart just by looking at those numbers?** The GA's job is to find the filter positions and widths that make those numbers as different as possible across powders.

### 2.2 Data Shape Inventory

The following is the authoritative reference for every array shape in the system.

#### 2.2.1 Input Data (from Excel files via `load_substance_atmosphere_data`)

| Variable | Shape | Dtype | Physical Meaning |
|---|---|---|---|
| `wavelengths` | `(d, 1)` | `float64` | Discrete wavelength sampling points in µm. Typically d≈100-200 points spanning 4-20 µm. |
| `emissivity_curves` | `(d, n)` | `float64` | Emissivity spectrum for each of `n` substances at each wavelength point. Values in [0, 1]. |
| `air_transmittance` | `(d, 1)` | `float64` | Atmospheric transmission coefficient at each wavelength. Values in [0, 1]. |
| `substance_names` | `(n,)` | `object` (str) | Human-readable names of the n substances. |
| `temperature_K` | scalar | `float64` | Scene temperature in Kelvin (e.g., 293.15 K = 20°C). |
| `atmospheric_distance_ratio` | scalar | `float64` | Exponent for atmospheric path modeling (e.g., 0.11). |
| `air_refractive_index` | scalar | `float64` | Refractive index of air (≈1.0). |

**Key dimension variables:**
- `d` = number of discrete wavelength points (spectral resolution)
- `n` = number of target substances (typically 4-21 in current datasets)
- `m` = number of basis functions / sensor channels (typically 4, optimized by GA)

#### 2.2.2 GA Chromosome → Basis Functions

| Stage | Shape | Description |
|---|---|---|
| Chromosome (raw) | `(m * p,)` | Flat 1D gene array. For Gaussians: `p=2`, so 4 basis functions = 8 genes: `[mu1, sigma1, mu2, sigma2, mu3, sigma3, mu4, sigma4]`. |
| Parameter tuples | `List[Tuple[float, float]]` of length `m` | Parsed: `[(mu1, sigma1), (mu2, sigma2), ...]`. The `parameters_to_curves` callback consumes this. |
| Basis functions | `(d, m)` | Each column is a unit-amplitude Gaussian curve evaluated at wavelength grid. Output of `gaussian_parameters_to_unit_amplitude_curves`. |

#### 2.2.3 Physics Simulation Pipeline (`simulate_sensor_output`)

The simulation computes `sensor_outputs = ∫ τ_air^r · B(λ,T) · ε(λ) · φ(λ) dλ` for each substance and basis function.

| Step | Intermediate | Shape | Formula |
|---|---|---|---|
| 1. Blackbody emission | `bb_emit` | `(d, 1)` | Planck's law: `B(λ, T, n)` |
| 2. Atmospheric transmission | `tau_air` | `(d, 1)` | `air_transmittance^atmospheric_distance_ratio` |
| 3. Per-substance radiation | `abso_spec` | `(d, m)` | `tau_air * bb_emit * emissivity[:, i:i+1] * basis_functions` |
| 4. Integration | `sensor_outputs[:, i]` | `(m,)` | `np.trapezoid(abso_spec, wavelengths, axis=0)` |
| **Final output** | `sensor_outputs` | **(m, n)** | **Each column is one substance's "fingerprint" as seen through all m sensor channels.** |

#### 2.2.4 Scoring Pipeline

| Step | Input Shape | Output Shape | Function |
|---|---|---|---|
| Distance matrix | `(m, n)` sensor_outputs | `(n, n)` symmetric | `compute_distance_matrix(..., axis=1)` with SAM |
| Dissimilarity score | `(n, n)` distance matrix | scalar `float` | `min_based_dissimilarity_score` → min off-diagonal value |

**The fitness value is a single float: the minimum SAM angle (in degrees) between any pair of substance fingerprints.** This is a maximin objective — the GA tries to maximize the worst-case separation.

### 2.3 Throughput & Scale Analysis

For a typical run with the "4 White Powders" dataset:
- `d = ~200`, `n = 4`, `m = 4`, `p = 2` → chromosome length = 8
- One fitness evaluation: `O(d*m*n)` for simulation + `O(n²*m)` for SAM distance matrix = very fast (sub-millisecond)
- GA population: 200 chromosomes × 2000 generations = 400,000 evaluations
- MAP-Elites: 200,000 evaluations

**Conclusion:** The per-evaluation cost is trivially small. The computational bottleneck is not the physics — it's the O(pop²) niching distance matrix computed every generation in `_calculate_shared_fitness`, which uses the Hungarian algorithm (`O(m³)`) per pair when `use_optimal_pairing=True`. For `pop=200`, that's 19,900 pairs × Hungarian = ~20K optimizations per generation.

---

## 3. GA-Simulation Interface Diagnosis

### 3.1 Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  MinDissimilarityFitnessEvaluator (ga/fitness.py)           │
│                                                             │
│  Owns: wavelengths, emissivity_curves, temperature_k,       │
│        atmospheric_distance_ratio, air_refractive_index,     │
│        air_transmittance, distance_metric,                   │
│        parameters_to_curves, params_per_basis_function       │
│                                                             │
│  .fitness_func(ga_instance, chromosome, idx) → float        │
│      │                                                      │
│      ├─ 1. Parse chromosome → List[Tuple]                   │
│      ├─ 2. self.parameters_to_curves(params, wavelengths)   │
│      │      → basis_functions (d, m)                        │
│      ├─ 3. simulate_sensor_output(...)  → (m, n)            │
│      ├─ 4. compute_distance_matrix(...)  → (n, n)           │
│      └─ 5. min_based_dissimilarity_score(...)  → float      │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 What's Good

1. **`parameters_to_curves` as a callback (Strategy Pattern).** This is the single best design decision in the codebase. The fitness evaluator doesn't know it's dealing with Gaussians — it receives a `Callable[[List[Tuple], ndarray], ndarray]`. This means you can swap in Lorentzian, top-hat, or measured filter responses without touching the evaluator.

2. **Class-based evaluator for picklability.** The docstring explicitly justifies why this is a class and not a closure. This is correct and important for `multiprocessing`.

3. **`simulate_sensor_output` is a pure function.** It takes arrays in, returns arrays out, has no state, no side effects. This is the ideal simulation function.

4. **`distance_metric` as an injectable dependency.** The SAM metric is a default, not a hardcoded choice.

### 3.3 Where Abstractions Leak

#### Issue 1: The Untyped Data Dictionary

The data returned by `load_substance_atmosphere_data` is a `dict[str, np.ndarray | float]`. This dictionary is then destructured by every consumer — `fitness.py`, `tuning.py`, `run_map_elites.py`, `run_ensemble_demo.py` — each with its own boilerplate:

```python
wavelengths_val = data["wavelengths"]
emissivity_val = data["emissivity_curves"]
temp_k_val = data["temperature_K"]
# ... 4 more lines of extraction ...
# ... followed by isinstance checks and float() casts ...
```

This pattern appears at least **4 times** across the codebase. The dictionary keys are stringly-typed — a typo like `"temperature_k"` vs `"temperature_K"` produces a silent `KeyError` at runtime, not a type error at development time. The `dict` also mixes arrays and scalars in the same value type, forcing every consumer to do `isinstance` checks.

**Verdict:** This is a textbook **Data Clump** anti-pattern. The data has a fixed, known structure that should be a typed container.

#### Issue 2: The GA Knows About Physics Parameters

`MinDissimilarityFitnessEvaluator.__init__` accepts `temperature_k`, `atmospheric_distance_ratio`, `air_refractive_index`, and `air_transmittance` as individual parameters. These are physics domain parameters that the GA module has no business understanding — it just forwards them to `simulate_sensor_output`.

The evaluator is essentially doing manual dependency plumbing: it stores 7 physics parameters as attributes just to pass them through to one function call. This couples the GA to the simulation's parameter signature.

**Verdict:** The evaluator's constructor mirrors the simulation function's signature. If the simulation adds a parameter (e.g., detector noise model), the evaluator's constructor, all call sites in `tuning.py`, `run_map_elites.py`, etc. all need to change. This is **Shotgun Surgery**.

#### Issue 3: `ga/tuning.py` — The 1000-Line God Module

`tuning.py` contains:
- `GenerationTracker` (callback tracker)
- `HyperparameterSearchSpace` (dataclass)
- `TuningResult` (dataclass)
- `run_single_configuration()` (GA runner)
- `HyperparameterTuner` (grid search orchestrator)
- `create_default_search_space()`, `create_focused_search_space()` (presets)
- `ExperimentConfig` (YAML experiment loading)
- `load_experiment_config()` (YAML parser)
- `create_fitness_evaluator_from_experiment()` (wires data loading → fitness evaluator)
- `create_search_space_from_experiment()`, `create_gene_space_from_experiment()` (experiment helpers)
- `visualize_top_configurations()` (visualization runner)

This module imports from `analysis`, `data`, `simulation`, `ga.advanced_ga`, `ga.diversity`, `ga.fitness`, `ga.ga_configuration`, `ga.mutations`, and `ga.visualization`. It is coupled to every module in the package.

**Verdict:** SRP violation. This module has at least 4 distinct responsibilities: experiment configuration, GA execution, result aggregation, and visualization. It should be decomposed.

#### Issue 4: Script-Level Data Wiring Boilerplate

Every script (`run_map_elites.py`, `run_ensemble_demo.py`, `tune_ga.py`) duplicates the same 30-line block:

1. Load data via `load_substance_atmosphere_data`
2. Extract and cast each field from the dict
3. Construct `MinDissimilarityFitnessEvaluator` with all extracted fields
4. Call `.fitness_func`

The `create_fitness_evaluator_from_experiment` in `tuning.py` attempts to solve this, but it's buried inside a module that's tightly coupled to the YAML experiment system, so scripts that don't use YAML can't reuse it.

### 3.4 What the GA Shouldn't Know

Currently, the GA layer (`ga/`) directly imports and uses:
- `simulate_sensor_output` — through `MinDissimilarityFitnessEvaluator`
- `gaussian_parameters_to_unit_amplitude_curves` — through `tuning.py` and `visualization.py`
- `load_substance_atmosphere_data` — through `tuning.py`
- `spectral_angle_mapper` — through `fitness.py` (as default)

The GA should only know:
- It has a `fitness_func: Callable[[GA, ndarray, int], float]`
- It has a `gene_space: List[Dict[str, float]]`
- It has a `niching_config: NichingConfig`

Everything else is physics domain knowledge that should be composed at the script/application layer.

---

## 4. SENS-D Legacy Assessment: `analysis/` Module

### 4.1 Module Inventory

| File | LOC | Role in Fitness Loop? | Role in Post-Hoc Analysis? |
|---|---|---|---|
| `distance_metrics.py` | 63 | **YES** — SAM is the default distance metric in the fitness function | YES — used for any spectral comparison |
| `distance_matrix.py` | 273 | **YES** — `compute_distance_matrix` called inside `fitness_func` every evaluation | YES — used for population diversity (IVAT, clustering) |
| `dissimilarity_scoring.py` | 266 | **YES** — `min_based_dissimilarity_score` is the fitness function's output | Partially — `group_based_dissimilarity_score` is post-hoc only |
| `optimal_pairing_distance.py` | 342 | NO — not used in fitness loop | YES — used for niching distance and IVAT in `ga/diversity.py` |
| `vat.py` | 104 | NO — not used in fitness loop | YES — used for IVAT visualization |

### 4.2 `optimal_pairing_distance.py` — Relevance Assessment

**What it does:** Implements the Hungarian algorithm to find the minimum-cost matching between two sets of parameter tuples (e.g., two chromosomes' Gaussian `(mu, sigma)` pairs), then sums the matched distances. This makes chromosome comparison invariant to the ordering of basis functions.

**Where it's used:**
1. `ga/diversity.py` → `compute_population_distance_matrix()` when `niching_config.use_optimal_pairing=True`
2. `ga/analysis.py` → `_determine_clustering_distance_func()` for population clustering
3. `ga/visualization.py` → `plot_ivat_analysis()` for IVAT distance matrix (hardcoded `params_per_group=2`)

**Is it relevant to fitness?** No. It's purely a chromosome-space distance metric used for diversity management. The fitness function operates in signal-space (sensor outputs) using SAM.

**Assessment:** This module is well-designed and solves a real problem (permutation invariance of grouped parameters). However, it contains 11 vectorized distance functions (`_compute_euclidean_distance`, `_compute_manhattan_distance`, etc.) that duplicate functionality available in `scipy.spatial.distance`. The custom implementations add maintenance burden for marginal benefit. Only Euclidean is actually used via `optimal_pairing_metric='euclidean'`.

**Recommendation:** Keep the Hungarian matching logic. Replace the 11 custom distance functions with `scipy.spatial.distance.cdist` calls. This shrinks the module from 342 to ~80 lines.

### 4.3 `dissimilarity_scoring.py` — Relevance Assessment

**What it does:** Provides 4 scoring functions that reduce an `(n, n)` distance matrix to a single scalar:
1. `min_based_dissimilarity_score` — min off-diagonal (maximin)
2. `mean_min_based_dissimilarity_score` — `mean * min^alpha`
3. `group_based_dissimilarity_score` — mean inter-group distance
4. `weighted_mean_min_dissimilarity_score` — `beta*mean + (1-beta)*min`

**Which are used in production?**
- `min_based_dissimilarity_score` — **critical**, it's the fitness function
- The other 3 — **never imported outside this module or tests**

**Assessment:** The 3 unused scoring functions are not dead code in the negative sense — they're reasonable alternative fitness objectives. But they represent speculative generality. They should be kept available but clearly marked as alternatives rather than sitting alongside the production scorer undifferentiated.

**The `_ensure_distance_matrix` helper** adds a layer of indirection that lets scoring functions accept either raw items or precomputed distance matrices. This is fine for a convenience API but adds cognitive overhead. In the hot fitness loop, only the `distance_matrix=` path is used.

### 4.4 `distance_matrix.py` — Dual-Use Diagnosis

This module serves two distinct purposes:
1. **In the fitness loop:** `compute_distance_matrix(sensor_outputs, distance_func=SAM, axis=1)` — compares substance signal vectors.
2. **In GA diversity analysis:** `compute_distance_matrix(population, metric='euclidean', use_optimal_pairing=True, params_per_group=2)` — compares chromosomes.

These are fundamentally different operations applied to different data in different coordinate spaces. They're unified under one API because they share the mathematical concept of "pairwise distance matrix," but the code paths diverge immediately after the entry point (`_compute_standard_distances` vs `_compute_optimal_pairing_distances`).

**Assessment:** The unification is defensible but creates a complex parameter surface (`distance_func` vs `metric`, `axis`, `use_optimal_pairing`, `params_per_group`, `metric_params`, `attribute`). New users must understand which parameter combinations are valid for which use case. This is a **Leaky Abstraction** — the internal routing logic leaks into the public API.

---

## 5. Structural Anti-Pattern Inventory

### 5.1 Data Clump: The Scene Configuration Dictionary

**Location:** `load_substance_atmosphere_data` return value, consumed in `fitness.py`, `tuning.py`, `run_map_elites.py`, `run_ensemble_demo.py`.

**Evidence:** The exact same 7 keys (`wavelengths`, `emissivity_curves`, `air_transmittance`, `temperature_K`, `atmospheric_distance_ratio`, `air_refractive_index`, `substance_names`) are destructured in 4+ locations with identical boilerplate.

**Severity:** High. Any new parameter (e.g., `detector_noise_std`) requires changes in every consumer.

### 5.2 Shotgun Surgery: Adding a Simulation Parameter

**Scenario:** Add detector noise to the simulation.

**Files that must change:**
1. `sensor_simulation.py` — add parameter to `simulate_sensor_output`
2. `fitness.py` — add parameter to `MinDissimilarityFitnessEvaluator.__init__` and forward it
3. `substance_atmosphere_data.py` — add field to returned dict (or load from elsewhere)
4. `tuning.py` — update `create_fitness_evaluator_from_experiment`
5. Every script that constructs the evaluator manually

**Severity:** High. The coupling path is `data_loader → dict → evaluator.__init__ → simulation`.

### 5.3 SRP Violation: `ga/tuning.py`

**Evidence:** 1021 lines, 12 top-level definitions, imports from 9 internal modules. Contains experiment I/O, GA execution, statistical aggregation, and visualization code.

**Severity:** Medium. Works but is unmaintainable. The `visualize_top_configurations` function alone is 170 lines and re-implements the entire GA setup pipeline.

### 5.4 Dead DEBUG Statements

| File | Line | Statement |
|---|---|---|
| `advanced_ga.py` | 221 | `print(f"DEBUG: Niching penalty applied: ...")` |
| `mutations.py` | 607 | `print(f"DEBUG: Mutation function CALLED ...")` |

**Severity:** Low but unprofessional. These will spam stdout during any GA run (200 solutions × 2000 generations = 400K print calls for mutations alone).

### 5.5 Hardcoded Gaussian Assumption in Visualization

`ga/visualization.py:242` and `ga/visualization.py:304` hardcode the chromosome-to-curve conversion:
```python
gaussian_params = [(chromosome[j], chromosome[j + 1]) for j in range(0, len(chromosome), 2)]
basis_functions = gaussian_parameters_to_unit_amplitude_curves(gaussian_params, wavelengths)
```

Despite the fitness evaluator being agnostic to basis function type (via `parameters_to_curves` callback), the visualization layer re-hardcodes Gaussians. The IVAT analysis also hardcodes `params_per_group=2`.

**Severity:** Medium. Breaks the abstraction that `fitness.py` correctly implements.

### 5.6 Inconsistent Wavelength Shape Convention

`wavelengths` enters as `(d, 1)` from Excel loading, gets `squeeze`d to `(d,)` in the fitness evaluator, then gets `reshape(-1, 1)` back in `simulate_sensor_output`. The shape bounces between 1D and column-vector across the pipeline:

| Location | Shape |
|---|---|
| `load_substance_atmosphere_data` return | `(d, 1)` |
| `MinDissimilarityFitnessEvaluator.wavelengths_1d` | `(d,)` |
| `simulate_sensor_output` internal (after reshape) | `(d, 1)` |
| `gaussian_parameters_to_unit_amplitude_curves` input | `(d,)` (flattened if needed) |

**Severity:** Low. The defensive reshaping handles it, but the inconsistency creates confusion and unnecessary code.

---

## 6. Strategic Pythonic Refactoring Plan

### 6.1 Guiding Principles

1. **Solve real problems, not hypothetical ones.** Every proposed change maps to an anti-pattern identified above.
2. **Preserve what works.** The `simulate_sensor_output` pure function and the `parameters_to_curves` callback pattern are correct — don't touch them.
3. **Prioritize the data path.** The highest-ROI change is typing the data that flows through the system.
4. **Use `dataclass` as DTO, not as framework.** No metaclass magic, no descriptor protocols, no runtime validation beyond `__post_init__`.

### 6.2 Proposal 1: `SceneConfig` Dataclass (replaces the data dictionary)

**Problem solved:** Data Clump (#5.1), Shotgun Surgery (#5.2), Script Boilerplate (#3.4).

```python
@dataclass(frozen=True, slots=True)
class SceneConfig:
    """Immutable snapshot of the physical scene for sensor simulation.

    Contains all environment/substance parameters needed by
    simulate_sensor_output. Sensor basis functions are NOT included
    because they are the optimization variable.
    """
    wavelengths: np.ndarray        # (d,) — 1D, canonical shape
    emissivity_curves: np.ndarray  # (d, n)
    air_transmittance: np.ndarray  # (d,) — 1D, canonical shape
    temperature_k: float
    atmospheric_distance_ratio: float
    air_refractive_index: float
    substance_names: np.ndarray    # (n,) str array, informational only
```

**Key design decisions:**
- `frozen=True` — immutable. Physics parameters don't change during optimization. This makes the object safe to share across processes without copies.
- `slots=True` — memory-efficient, prevents accidental attribute assignment.
- **Wavelengths are 1D.** The `(d, 1)` column-vector convention is an artifact of the original Excel loading. Force 1D at the boundary and let consumers reshape if needed.
- **No validation in `__post_init__`.** Shape validation is the responsibility of a factory function or the data loader. The dataclass is a typed container, not a validation framework.

**Impact:**
- `MinDissimilarityFitnessEvaluator.__init__` drops from 9 parameters to 4: `scene: SceneConfig`, `parameters_to_curves`, `params_per_basis_function`, `distance_metric`.
- `load_substance_atmosphere_data` returns `SceneConfig` instead of `dict`.
- All scripts replace 15-line destructuring blocks with `scene = load_substance_atmosphere_data(...)`.

### 6.3 Proposal 2: Simplify `MinDissimilarityFitnessEvaluator`

**Problem solved:** GA knowing about physics parameters (#3.3, Issue 2).

The evaluator should accept the scene configuration as a single object:

```python
class MinDissimilarityFitnessEvaluator:
    def __init__(
        self,
        scene: SceneConfig,
        parameters_to_curves: Callable[[list[tuple], np.ndarray], np.ndarray],
        params_per_basis_function: int,
        distance_metric: Callable = spectral_angle_mapper,
    ): ...
```

The evaluator no longer stores individual physics parameters. It holds a `SceneConfig` and delegates to `simulate_sensor_output(scene, basis_functions)` — or continues to unpack the scene internally if the simulation function signature stays the same.

**Note:** This does NOT require changing `simulate_sensor_output`'s signature. The evaluator can unpack `scene` when calling the simulation. The benefit is that the evaluator's public API is decoupled from the simulation's parameter list.

### 6.4 Proposal 3: Decompose `ga/tuning.py`

**Problem solved:** SRP violation (#5.3).

Split into:

| New Module | Contents | LOC (est.) |
|---|---|---|
| `ga/tuning.py` | `HyperparameterTuner`, `run_single_configuration`, search space presets | ~300 |
| `ga/experiment.py` | `ExperimentConfig`, `load_experiment_config`, `create_*_from_experiment` | ~200 |
| Move `visualize_top_configurations` into `ga/visualization.py` | Already the right home | ~170 |

`GenerationTracker` can stay in `tuning.py` since it's only used there.

### 6.5 Proposal 4: Standardize Wavelength as 1D

**Problem solved:** Inconsistent shape convention (#5.6).

**Rule:** `wavelengths` is always `np.ndarray` with `ndim == 1` and shape `(d,)`. Period.

**Changes:**
- `load_substance_atmosphere_data`: squeeze wavelengths to 1D before returning (or in `SceneConfig` factory)
- `simulate_sensor_output`: remove the `if wavelengths.ndim == 1: wavelengths = wavelengths.reshape(-1, 1)` defensive code; instead, the function internally reshapes if needed for broadcasting
- `MinDissimilarityFitnessEvaluator`: drop the `wavelengths_1d` attribute; `self.scene.wavelengths` is already 1D

### 6.6 Proposal 5: Trim `optimal_pairing_distance.py`

**Problem solved:** Unnecessary code surface area (#4.2).

Replace the 11 hand-rolled distance functions with:

```python
from scipy.spatial.distance import cdist

def _pairwise_distances_vectorized(a, b, metric, metric_params=None):
    return cdist(a, b, metric=metric, **(metric_params or {}))
```

This eliminates ~200 lines of code that duplicates scipy functionality and is only tested indirectly.

### 6.7 Proposal 6: Clean GA Visualization of Gaussian Assumption

**Problem solved:** Hardcoded Gaussian assumption in visualization (#5.5).

Two options:

**Option A (Pragmatic):** Accept that visualization is always Gaussian-specific for now. Document this explicitly and add a `parameters_to_curves` parameter to `plot_top_sensor_designs` for future extensibility. This is the recommended option since the entire system currently only uses Gaussians.

**Option B (Principled):** Store the `parameters_to_curves` callable in the GA result dictionary so visualization can use it. This adds complexity without current benefit.

### 6.8 What NOT to Change

1. **`simulate_sensor_output` function signature.** It's a clean, well-documented pure function. Wrapping it in a class or changing its interface would add indirection without benefit.
2. **`AdvancedGA` inheriting from `pygad.GA`.** The inheritance is justified — it's extending a framework class with niching behavior.
3. **The `analysis/` module structure.** The 5-file decomposition (metrics, matrix, scoring, optimal pairing, VAT) follows logical boundaries. Don't collapse them.
4. **`MutationConfig` and `NichingConfig` dataclasses.** These are well-designed configuration objects with sensible defaults and presets.
5. **The `parameters_to_curves` callback pattern.** This is the correct abstraction. Protect it.

---

## 7. Dependency Graph & Module Coupling Map

### 7.1 Current Import Graph (simplified)

```
scripts/*
  └─ data.load_substance_atmosphere_data
  └─ ga.MinDissimilarityFitnessEvaluator
  └─ ga.AdvancedGA
  └─ simulation.gaussian_parameters_to_unit_amplitude_curves
  └─ analysis.spectral_angle_mapper

ga/fitness.py
  └─ simulation.simulate_sensor_output
  └─ analysis.compute_distance_matrix
  └─ analysis.min_based_dissimilarity_score
  └─ analysis.spectral_angle_mapper

ga/advanced_ga.py
  └─ ga.diversity.compute_population_distance_matrix

ga/diversity.py
  └─ analysis.compute_distance_matrix   (for optimal pairing path)
  └─ scipy.spatial.distance.cdist       (for standard path)

ga/tuning.py
  └─ analysis.spectral_angle_mapper
  └─ data.load_substance_atmosphere_data
  └─ simulation.gaussian_parameters_to_unit_amplitude_curves
  └─ ga.advanced_ga, ga.diversity, ga.fitness, ga.ga_configuration
  └─ ga.mutations, ga.visualization

ga/visualization.py
  └─ analysis.compute_distance_matrix, vat_reorder, ivat_transform
  └─ simulation.gaussian_parameters_to_unit_amplitude_curves

ga/analysis.py
  └─ analysis.calculate_optimal_pairing_distance
  └─ ga.diversity.compute_population_distance_matrix
```

### 7.2 Ideal Layering

```
Layer 0 (Pure Math):     analysis/distance_metrics.py
                          analysis/vat.py

Layer 1 (Math + Scipy):  analysis/distance_matrix.py
                          analysis/optimal_pairing_distance.py
                          analysis/dissimilarity_scoring.py

Layer 2 (Physics):       simulation/blackbody.py
                          simulation/gaussian_parameter_to_curves.py
                          simulation/sensor_simulation.py

Layer 3 (Data I/O):      data/substance_atmosphere_data.py

Layer 4 (Optimization):  ga/advanced_ga.py
                          ga/fitness.py
                          ga/diversity.py
                          ga/mutations.py

Layer 5 (Orchestration): ga/tuning.py, ga/experiment.py
                          ga/ga_configuration.py
                          ga/result_extraction.py

Layer 6 (Presentation):  ga/visualization.py
                          visualization/*
                          ga/analysis.py

Layer 7 (Application):   scripts/*
```

**Violation:** `ga/tuning.py` (Layer 5) imports from Layer 2 and Layer 3 directly, bypassing Layer 4. The `create_fitness_evaluator_from_experiment` function in `tuning.py` is actually a Layer 7 concern (application wiring).

---

## 8. Prioritized Action Items

### Tier 1: Immediate (Low Risk, High Value)

| # | Action | Files | Effort |
|---|---|---|---|
| 1.1 | Remove DEBUG print statements | `advanced_ga.py`, `mutations.py` | 5 min |
| 1.2 | Introduce `SceneConfig` dataclass in `data/` | `data/scene_config.py` (new), `data/__init__.py` | 1 hr |
| 1.3 | Update `load_substance_atmosphere_data` to return `SceneConfig` | `data/substance_atmosphere_data.py` | 30 min |
| 1.4 | Standardize wavelengths as 1D `(d,)` everywhere | `data/`, `simulation/`, `ga/fitness.py` | 1 hr |

### Tier 2: Moderate (Medium Risk, High Value)

| # | Action | Files | Effort |
|---|---|---|---|
| 2.1 | Refactor `MinDissimilarityFitnessEvaluator` to accept `SceneConfig` | `ga/fitness.py`, all scripts | 2 hr |
| 2.2 | Extract `ga/experiment.py` from `ga/tuning.py` | `ga/tuning.py` → `ga/experiment.py` | 2 hr |
| 2.3 | Replace custom distance functions in `optimal_pairing_distance.py` with `scipy.cdist` | `analysis/optimal_pairing_distance.py` | 1 hr |
| 2.4 | Create a convenience factory: `SceneConfig.from_excel(spectral_path, transmittance_path, ...)` | `data/scene_config.py` | 30 min |

### Tier 3: Refinement (Low Risk, Medium Value)

| # | Action | Files | Effort |
|---|---|---|---|
| 3.1 | Add `parameters_to_curves` parameter to `plot_top_sensor_designs` | `ga/visualization.py` | 30 min |
| 3.2 | Mark unused scoring functions with "Alternative Fitness Objectives" docstring section | `analysis/dissimilarity_scoring.py` | 15 min |
| 3.3 | Add type narrowing to `load_substance_atmosphere_data` return (overloads for single vs multi-condition) | `data/substance_atmosphere_data.py` | 1 hr |
| 3.4 | Remove or deprecate `generated_basis_functions_deprecated.py` | `simulation/` | 15 min |

### Not Recommended

| Action | Why Not |
|---|---|
| Abstract `simulate_sensor_output` behind a Protocol/ABC | Only one implementation exists. YAGNI. |
| Replace `dict` returns in `ga/analysis.py` with dataclasses | The analysis dicts are consumed by scripts for ad-hoc inspection, not by typed code paths. Dictionaries are appropriate here. |
| Introduce a Registry/Plugin pattern for distance metrics | The system uses exactly one metric (SAM) in production. A registry adds complexity for a swap that has never been needed. |
| Move `analysis/` into `ga/` to reduce cross-package imports | The analysis functions are domain-independent math. They should stay general. |

---

*End of audit.*
