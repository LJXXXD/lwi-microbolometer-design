# Project Architecture & Data Flow

> **LWI Microbolometer Sensor Design and Optimization**
>
> The ultimate onboarding reference for understanding the physics, mathematics,
> software architecture, and end-to-end data flow of this project.

---

## Table of Contents

1. [The Core Problem & Mathematical Intuition](#1-the-core-problem--mathematical-intuition)
2. [The Refactoring Journey (The "Why")](#2-the-refactoring-journey-the-why)
3. [The Data Reality (Strict Shapes & Types)](#3-the-data-reality-strict-shapes--types)
4. [Core Module Breakdown (The Current Architecture)](#4-core-module-breakdown-the-current-architecture)
5. [End-to-End Data Flow (Step-by-Step)](#5-end-to-end-data-flow-step-by-step)
6. [Next Steps & Strategic Targets](#6-next-steps--strategic-targets)

---

## 1. The Core Problem & Mathematical Intuition

### 1.1 What Are We Actually Building?

We are designing **infrared microbolometer sensors** that can identify unknown
substances by measuring their thermal infrared emissions. Think of it like this:

> **ELI5:** Every material "glows" differently in the infrared. A hot road and a
> hot chemical spill both radiate heat, but they radiate at *different
> wavelengths*. Our sensor needs tunable "color filters" (basis functions) that
> look at specific wavelength bands so the readings from each substance produce a
> unique fingerprint. The optimization problem is: *which set of filters makes
> every substance look as different as possible?*

In more rigorous terms, we have `n` target substances, each with a known
spectral emissivity curve ε(λ). We design `m` sensor basis functions B(λ)
(parameterized Gaussians) such that the integrated sensor responses form an
`m`-dimensional fingerprint vector per substance. The optimization objective is
to **maximize the minimum pairwise angular separation** between all fingerprint
vectors.

### 1.2 The Physics Pipeline (From Photons to Numbers)

The sensor response for substance `j` through basis function `i` is:

```
S(i,j) = ∫ τ(λ)^r · B(λ,T,n) · ε_j(λ) · Φ_i(λ) dλ
```

Where:

| Symbol | Meaning | Units |
|--------|---------|-------|
| `λ` | Wavelength | µm |
| `τ(λ)` | Atmospheric transmittance | dimensionless [0, 1] |
| `r` | Atmospheric distance ratio | dimensionless |
| `B(λ,T,n)` | Planck blackbody spectral radiance at temperature `T` and refractive index `n` | W/(m²·µm·sr) |
| `ε_j(λ)` | Emissivity spectrum of substance `j` | dimensionless [0, 1] |
| `Φ_i(λ)` | Sensor basis function `i` (unit-amplitude Gaussian) | dimensionless |

Planck's law itself:

```
B(λ, T, n) = n² · (2πhc² / λ⁵) · 1 / (exp(hc / λkT) − 1)
```

The `n²` factor accounts for the increased photon density of states in a medium
(for air, n ≈ 1.0003 — a small but physically correct correction). The
integration is performed numerically via the trapezoidal rule (`np.trapezoid`)
over the discrete wavelength grid.

### 1.3 Why Shapes, Not Absolute Energy?

A critical insight: we care about the **angular direction** of each substance's
fingerprint vector in `m`-space, not its magnitude. Two substances that
produce the exact same ratio of sensor readings across all `m` channels —
even if one produces 10× more total signal — would be indistinguishable.

This is precisely why we use the **Spectral Angle Mapper (SAM)** as our
distance metric:

```
θ(u, v) = arccos( (u · v) / (‖u‖ · ‖v‖) )
```

SAM measures the angle (in degrees) between two vectors. It is
**magnitude-invariant**: only the direction matters. This directly maps to the
physical reality that a sensor must distinguish substances by *spectral shape*,
not by *how much total energy* they emit.

### 1.4 What Makes a "Good" Sensor? The Maximin Strategy

Given `n` substances, the pairwise SAM distance matrix is an `n × n` symmetric
matrix with zero diagonal. Our fitness function extracts the **minimum
off-diagonal value** — the worst-case pairwise separation:

```
fitness = min{ θ(u_i, u_j) : i ≠ j }
```

This is a **maximin** (maximize-the-minimum) strategy. We are designing for the
hardest case: if the two most confusable substances are still well-separated,
then every pair is. It is the conservative, robust choice — especially important
for hazardous material detection where misclassification of *any* pair is
equally catastrophic.

> **Analogy:** Imagine placing `n` points on a sphere and trying to spread them
> apart. SAM measures angular separation. Our fitness is the distance between the
> *closest* pair. We want to push that closest pair as far apart as possible —
> the classic "packing problem" on a hypersphere.

---

## 2. The Refactoring Journey (The "Why")

### 2.1 Previous Bottlenecks

The original codebase suffered from several architectural anti-patterns that
compounded into real engineering pain:

| Anti-Pattern | Where It Lived | Concrete Symptoms |
|---|---|---|
| **Data Clumps** | Every function that needed scene data accepted 6-7 loose parameters: `wavelengths`, `emissivity_curves`, `air_transmittance`, `temperature_k`, `atmospheric_distance_ratio`, `air_refractive_index` | Functions had 10+ arguments. Passing data through call chains meant threading the same six arrays everywhere. Forgetting one parameter caused silent shape mismatches. |
| **God Module** | `tuning.py` (~800+ lines) | Contained experiment loading, fitness wiring, search space definition, tuner orchestration, and result analysis — all in one file. Circular imports were inevitable when other modules needed its types. |
| **Circular Imports** | `tuning.py` ↔ `ga_configuration.py` ↔ `advanced_ga.py` | `HyperparameterSearchSpace` lived in `tuning.py` but was needed by `experiment.py` (which also needed fitness wiring from tuning). Import cycles caused `ImportError` at startup. |
| **Zero-Norm SAM Crashes** | `distance_metrics.py` | During GA optimization, degenerate chromosomes could produce zero-energy sensor outputs. Dividing by `‖v‖ = 0` in SAM caused `NaN` propagation, crashing the entire GA run. |
| **Visualization in the Hot Path** | `tuning.py`, scattered scripts | Plotting functions were interleaved with fitness evaluation and GA setup, making the core loop harder to test and profile. |
| **Inconsistent Array Shapes** | Throughout | Wavelengths sometimes arrived as `(d,)`, sometimes `(d, 1)`. Air transmittance could be `(d, k)` when only the first column was needed. No single point canonicalized shapes, leading to defensive reshaping scattered across every consumer. |

### 2.2 The Conceptual Solution

The refactoring followed three core principles:

1. **Introduce a Data Transfer Object (DTO)** — `SceneConfig` — to bundle all
   scene-invariant physics data into a single frozen, validated container.
2. **Decompose the God Module** — extract `experiment.py` for YAML-driven
   experiment wiring, keeping `tuning.py` focused on hyperparameter search.
3. **Decouple visualization from computation** — move all plotting into
   `visualization/` and `ga/visualization.py`, completely out of the simulation
   and fitness evaluation path.

### 2.3 How We Implemented It

#### Step 1: `SceneConfig` DTO (commit `2fae1a8`)

A frozen, slotted dataclass that canonicalizes all array shapes on construction:

```python
@dataclass(frozen=True, slots=True)
class SceneConfig:
    wavelengths: np.ndarray           # (d,)
    emissivity_curves: np.ndarray     # (d, n)
    air_transmittance: np.ndarray     # (d,)
    temperature_k: float
    atmospheric_distance_ratio: float
    air_refractive_index: float
    substance_names: np.ndarray       # (n,)
```

- **Immutable** (`frozen=True`): cannot be accidentally mutated mid-optimization.
- **Self-validating** (`__post_init__`): squeezes `(d, 1)` → `(d,)`, extracts
  first column from multi-column transmittance, and raises `ValueError` if
  dimensions are inconsistent.
- **Single source of truth**: every function downstream receives one object
  instead of six loose arrays.

#### Step 2: Decoupled `MinDissimilarityFitnessEvaluator` (commit `c6af287`)

Refactored from a factory-function closure to a **picklable class** that stores
`SceneConfig` + a `parameters_to_curves` callback:

```python
class MinDissimilarityFitnessEvaluator:
    def __init__(self, scene, parameters_to_curves, params_per_basis_function, distance_metric):
        ...
    def fitness_func(self, _ga_instance, chromosome, _chromosome_idx) -> float:
        ...
```

Why a class? Factory closures capture free variables and are **not picklable**.
Multiprocessing with `concurrent.futures.ProcessPoolExecutor` requires
serializable fitness functions. This class stores everything as attributes.

#### Step 3: God Module Decomposition (commit `07777c8`)

`tuning.py` was split:

| Before (in `tuning.py`) | After |
|---|---|
| YAML experiment loading | → `experiment.py :: load_experiment_config()` |
| Fitness evaluator wiring | → `experiment.py :: create_fitness_evaluator_from_experiment()` |
| Search space / gene space creation | → `experiment.py :: create_search_space_from_experiment()` |
| `HyperparameterSearchSpace`, `TuningResult`, `HyperparameterTuner` | Remain in `tuning.py` |

Circular imports resolved by using `TYPE_CHECKING`-guarded imports where
`experiment.py` needs `HyperparameterSearchSpace` only for type annotations.

#### Step 4: Zero-Norm SAM Guard (commit `ed01ad7`)

```python
_SAM_NORM_EPS = 1e-15

def spectral_angle_mapper(vector1, vector2) -> float:
    ...
    if norm1 < _SAM_NORM_EPS or norm2 < _SAM_NORM_EPS:
        return 0.0  # degenerate fingerprint → maximally similar
    ...
```

Returning `0.0` (zero degrees = identical) is the correct physical
interpretation: a sensor design that produces no output for a substance provides
zero discriminating information. This makes fitness `0.0` for degenerate
chromosomes, giving the GA a smooth gradient to evolve away from them.

#### Step 5: Visualization Decoupling (commit `2ed5d4b`)

All plotting moved to:

- `visualization/distance_matrix_visualization.py` — heatmaps
- `visualization/sensor_output_visualization.py` — sensor response curves
- `ga/visualization.py` — GA-specific plots (fitness evolution, top designs, iVAT)

The simulation and fitness modules now have **zero matplotlib imports**.

---

## 3. The Data Reality (Strict Shapes & Types)

### 3.1 Dimension Glossary

| Symbol | Meaning | Typical Range |
|--------|---------|---------------|
| **`d`** | Number of discrete wavelength sampling points | ~100–500 |
| **`n`** | Number of target substances | ~5–20 |
| **`m`** | Number of sensor basis functions (Gaussian filters) | ~2–8 |
| **`p`** | Parameters per basis function (e.g., 2 for Gaussian: µ, σ) | 2 |

### 3.2 Tensor Shape Table

This table traces the exact array shapes through the entire evaluation pipeline:

| Stage | Variable | Shape | dtype | Source |
|-------|----------|-------|-------|--------|
| **Data Loading** | `wavelengths` | `(d,)` | `float64` | Excel col 0, canonicalized by `SceneConfig` |
| | `emissivity_curves` | `(d, n)` | `float64` | Excel cols 1..n |
| | `air_transmittance` | `(d,)` | `float64` | Excel, squeezed from `(d, 1)` or `(d, k)[:, 0]` |
| | `substance_names` | `(n,)` | `object` | Excel column headers |
| **SceneConfig** | All of the above bundled | — | — | `SceneConfig.__post_init__` validates consistency |
| **GA Chromosome** | `chromosome` | `(m * p,)` | `float64` | e.g., `[µ₁, σ₁, µ₂, σ₂, …, µₘ, σₘ]` |
| **Parameter Extraction** | `basis_params` | `list[tuple]` of length `m` | — | `chromosome` chunked by `p` |
| **Basis Functions** | `basis_functions` | `(d, m)` | `float64` | `gaussian_parameters_to_unit_amplitude_curves` |
| **Physics Simulation** | `bb_emit` (blackbody) | `(d, 1)` | `float64` | `blackbody_emit` |
| | `tau_air` (atmospheric) | `(d, 1)` | `float64` | `air_transmittance ** r`, reshaped |
| | `abso_spec` (per-substance) | `(d, m)` | `float64` | `tau_air * bb_emit * ε_j * Φ` (broadcast) |
| **Sensor Outputs** | `sensor_outputs` | `(m, n)` | `float64` | `np.trapezoid(abso_spec, λ)` per substance |
| **Distance Matrix** | `distance_matrix` | `(n, n)` | `float64` | Pairwise SAM over columns of `sensor_outputs` |
| **Fitness Score** | scalar | `float` | — | `min(off-diagonal values of distance_matrix)` |

### 3.3 Shape Canonicalization Rules

`SceneConfig.__post_init__` enforces these invariants at construction time:

1. `wavelengths`: any 2-D input is squeezed; must be exactly 1-D of length `d`.
2. `air_transmittance`: `(d, 1)` → squeeze to `(d,)`; `(d, k)` → take column 0.
   Must reduce to 1-D of length `d`.
3. `emissivity_curves`: must be exactly 2-D with first axis `d`.
4. `substance_names`: flattened to 1-D; length must equal `n` (second axis of
   emissivity).

If any shape is inconsistent, a descriptive `ValueError` is raised **before**
any computation begins.

---

## 4. Core Module Breakdown (The Current Architecture)

### 4.1 Package Layout

```
src/lwi_microbolometer_design/
├── __init__.py                          # Public API re-exports
├── data/
│   ├── scene_config.py                  # SceneConfig DTO
│   └── substance_atmosphere_data.py     # Excel → SceneConfig loader
├── simulation/
│   ├── blackbody.py                     # Planck's law (B(λ, T, n))
│   ├── gaussian_parameter_to_curves.py  # (µ, σ) → Gaussian basis curves
│   └── sensor_simulation.py             # Full physics pipeline → (m, n)
├── analysis/
│   ├── distance_metrics.py              # SAM (and extensible for others)
│   ├── distance_matrix.py               # Pairwise distance computation
│   ├── dissimilarity_scoring.py         # min, mean-min, group, weighted scores
│   ├── optimal_pairing_distance.py      # Hungarian-algorithm grouped distance
│   └── vat.py                           # VAT/iVAT reordering for visualization
├── ga/
│   ├── advanced_ga.py                   # AdvancedGA (PyGAD + niching)
│   ├── fitness.py                       # MinDissimilarityFitnessEvaluator
│   ├── diversity.py                     # Population distance matrices
│   ├── mutations.py                     # Adaptive diversity-preserving mutation
│   ├── ga_configuration.py              # GA config builder
│   ├── experiment.py                    # YAML experiment loading & wiring
│   ├── tuning.py                        # Hyperparameter search (Tuner)
│   ├── result_extraction.py             # Post-run result extraction
│   ├── population_analysis.py           # Clustering / diversity analytics
│   └── visualization.py                 # GA-specific plots
└── visualization/
    ├── distance_matrix_visualization.py # Heatmaps (seaborn)
    └── sensor_output_visualization.py   # Sensor response curve plots
```

### 4.2 Module Responsibilities & Boundaries

#### `data/` — Scene Data Layer

| File | Responsibility |
|------|---------------|
| `scene_config.py` | Defines the `SceneConfig` frozen dataclass. Pure data container with shape validation. Zero external dependencies beyond NumPy. |
| `substance_atmosphere_data.py` | Reads Excel files (spectral emissivity + air transmittance) and constructs `SceneConfig` instance(s). Supports multi-condition sweeps via parameter meshgrid. |

**Boundary:** This module knows about file I/O and data formats. It knows
nothing about optimization, fitness, or sensors.

#### `simulation/` — Physics Engine

| File | Responsibility |
|------|---------------|
| `blackbody.py` | Implements Planck's law: `blackbody_emit(λ, T, n) → B(λ)`. Pure physics, no awareness of substances or sensors. |
| `gaussian_parameter_to_curves.py` | Converts `(µ, σ)` parameter tuples → unit-amplitude Gaussian curves on a wavelength grid. Aligns means to nearest discrete wavelength for exact peak = 1.0. Fully vectorized. |
| `sensor_simulation.py` | The core physics pipeline: computes blackbody emission, applies atmospheric attenuation, multiplies by emissivity, convolves with basis functions, integrates. Returns `(m, n)` sensor output matrix. |

**Boundary:** Pure numerical computation. No GA awareness, no fitness concepts,
no file I/O. Accepts raw arrays (not `SceneConfig` directly) — the fitness
evaluator unpacks the DTO.

#### `analysis/` — Spectral Analysis & Scoring

| File | Responsibility |
|------|---------------|
| `distance_metrics.py` | `spectral_angle_mapper(u, v) → float` in degrees. Guarded against zero-norm vectors. |
| `distance_matrix.py` | `compute_distance_matrix(items, distance_func, axis) → (n, n)`. Unified API for lists, arrays, and optimal-pairing mode. |
| `dissimilarity_scoring.py` | Scoring functions: `min_based_dissimilarity_score` (primary fitness), plus `mean_min`, `group_based`, and `weighted_mean_min` variants for experimental objectives. |
| `optimal_pairing_distance.py` | Hungarian-algorithm based distance for grouped parameters (order-invariant comparison of `[(µ₁,σ₁), (µ₂,σ₂)]` vs `[(µ₂,σ₂), (µ₁,σ₁)]`). |
| `vat.py` | Visual Assessment of Tendency: reorders distance matrices for cluster visualization. |

**Boundary:** Operates on abstract vectors and distance matrices. Does not
import anything from `simulation/` or `ga/`. Could be used for any
distance-based analysis task.

#### `ga/` — Genetic Algorithm Engine

| File | Responsibility |
|------|---------------|
| `fitness.py` | **`MinDissimilarityFitnessEvaluator`** — the central wiring class. Holds `SceneConfig` + `parameters_to_curves` callback. Its `fitness_func` method is the PyGAD-compatible callable. |
| `advanced_ga.py` | **`AdvancedGA`** — extends `pygad.GA` with optional fitness-sharing niching. Overrides `run_select_parents` to inject shared fitness for parent selection while preserving original fitness for elitism. Also defines `NichingConfig`. |
| `diversity.py` | Computes population-level distance matrices (Euclidean or optimal-pairing) for niching. |
| `mutations.py` | `diversity_preserving_mutation` — adaptive mutation that detects stagnation and adjusts step sizes. `MutationConfig` provides presets (`conservative`, `aggressive`, `balanced`). |
| `ga_configuration.py` | `create_ga_config()` builds the full kwargs dict for `AdvancedGA`. `load_ga_configuration_from_csv()` loads tuned configs from previous tuning runs. |
| `experiment.py` | YAML experiment loading (`ExperimentConfig`), fitness evaluator construction, search space and gene space generation. Decoupled from `tuning.py` to break circular imports. |
| `tuning.py` | `HyperparameterTuner` orchestrates grid/random search over GA hyperparameters. `HyperparameterSearchSpace` defines the parameter grid. `run_single_configuration()` executes one GA trial. |
| `result_extraction.py` | `extract_basic_results(ga) → dict` for post-run analytics. |
| `population_analysis.py` | DBSCAN/K-Means clustering of final populations. Silhouette scoring, adaptive cluster count selection. |
| `visualization.py` | All GA-specific plots: fitness evolution, diversity curves, top sensor designs, iVAT heatmaps, tuning result comparisons. |

**Boundary (critical design rule):** The GA module **knows nothing about
physics**. `MinDissimilarityFitnessEvaluator` receives a `SceneConfig`
and a generic `parameters_to_curves` callback. The GA optimizes chromosomes;
it has no idea they represent Gaussian filter parameters. This decoupling means
we can swap in Lorentzian curves, Fabry-Pérot responses, or any other
parameterized basis function by providing a different callback — zero GA code
changes.

#### `visualization/` — Standalone Plotting

| File | Responsibility |
|------|---------------|
| `distance_matrix_visualization.py` | Seaborn heatmaps for `(n, n)` distance matrices. |
| `sensor_output_visualization.py` | Matplotlib bar/line plots for `(m, n)` sensor output matrices. |

**Boundary:** Pure presentation. Imports only matplotlib/seaborn + numpy.
Never imported by computation modules.

### 4.3 Dependency Graph (Simplified)

```
data/
  └─ scene_config.py                   ← numpy only
  └─ substance_atmosphere_data.py      ← pandas, scene_config

simulation/
  └─ blackbody.py                      ← numpy only
  └─ gaussian_parameter_to_curves.py   ← numpy only
  └─ sensor_simulation.py              ← blackbody

analysis/
  └─ distance_metrics.py               ← numpy only
  └─ distance_matrix.py                ← optimal_pairing_distance
  └─ dissimilarity_scoring.py          ← distance_matrix, distance_metrics
  └─ optimal_pairing_distance.py       ← scipy
  └─ vat.py                            ← numpy only

ga/
  └─ fitness.py                        ← analysis, data, simulation
  └─ advanced_ga.py                    ← pygad, diversity
  └─ diversity.py                      ← analysis (distance_matrix only)
  └─ mutations.py                      ← pygad, numpy
  └─ experiment.py                     ← data, simulation, analysis, fitness
  └─ tuning.py                         ← advanced_ga, diversity, ga_configuration, mutations
  └─ ga_configuration.py               ← advanced_ga (NichingConfig), mutations

visualization/
  └─ (matplotlib, seaborn, numpy only — never imported by computation code)
```

Notice: **no arrows point from `simulation/` or `analysis/` back into `ga/`**.
The dependency flow is strictly layered.

---

## 5. End-to-End Data Flow (Step-by-Step)

This traces the complete lifecycle of **one fitness evaluation** — the function
called thousands of times per GA run.

### Step 0: Data Loading (Once, Before GA Starts)

```
Excel files on disk
    │
    ├─ spectral_data.xlsx  →  wavelengths (d,), emissivity (d, n), names (n,)
    └─ air_transmittance.xlsx  →  air_transmittance (d,)
    │
    ▼
load_substance_atmosphere_data()
    │
    ▼
SceneConfig (frozen, validated)
    ├── wavelengths         (d,)    float64
    ├── emissivity_curves   (d, n)  float64
    ├── air_transmittance   (d,)    float64
    ├── temperature_k       float
    ├── atmospheric_distance_ratio  float
    ├── air_refractive_index        float
    └── substance_names     (n,)    object
```

### Step 1: Fitness Evaluator Construction (Once)

```python
evaluator = MinDissimilarityFitnessEvaluator(
    scene=scene_config,                                     # SceneConfig
    parameters_to_curves=gaussian_parameters_to_unit_amplitude_curves,
    params_per_basis_function=2,                             # p = 2 (µ, σ)
    distance_metric=spectral_angle_mapper,
)
```

This object is passed to `AdvancedGA(fitness_func=evaluator.fitness_func, ...)`.

### Step 2: Chromosome → Parameter Tuples

For each individual in the population, PyGAD calls `evaluator.fitness_func`:

```
chromosome: (m*p,) = [µ₁, σ₁, µ₂, σ₂, ..., µₘ, σₘ]
                              │
                              ▼  chunk by p=2
basis_params: [(µ₁, σ₁), (µ₂, σ₂), ..., (µₘ, σₘ)]
```

### Step 3: Parameter Tuples → Basis Function Curves

```
basis_params + scene.wavelengths
         │
         ▼  gaussian_parameters_to_unit_amplitude_curves()
basis_functions: (d, m)
```

Each column is a unit-amplitude Gaussian. Means are snapped to the nearest
discrete wavelength so the peak is exactly 1.0.

### Step 4: Physics Simulation

Inside `simulate_sensor_output()`:

```
1.  wavelengths reshaped:           (d,) → (d, 1)
2.  air_transmittance reshaped:     (d,) → (d, 1)
3.  bb_emit = blackbody_emit(λ, T, n):  (d, 1)
4.  tau_air = air_transmittance ^ r:     (d, 1)

For each substance j = 0..n-1:
    5.  emissivity_curve = ε[:, j:j+1]:  (d, 1)
    6.  abso_spec = tau_air * bb_emit * emissivity_curve * basis_functions:  (d, m)
              ↑ element-wise broadcast: (d,1)·(d,1)·(d,1)·(d,m) → (d, m)
    7.  sensor_outputs[:, j] = trapezoid(abso_spec, λ, axis=0):  (m,)

Result: sensor_outputs (m, n)
```

Each entry `sensor_outputs[i, j]` is the integrated energy that substance `j`
deposits through basis function `i`.

### Step 5: Distance Matrix

```
sensor_outputs: (m, n)
       │
       ▼  compute_distance_matrix(axis=1)  ← compare columns
       │   For each pair (j₁, j₂):
       │     extract column j₁ → (m,) vector
       │     extract column j₂ → (m,) vector
       │     SAM(v_j₁, v_j₂) → angle in degrees
       │
       ▼
distance_matrix: (n, n)   symmetric, zero diagonal
```

### Step 6: Fitness Score

```
distance_matrix: (n, n)
       │
       ▼  min_based_dissimilarity_score()
       │   mask diagonal with np.eye
       │   extract off-diagonal values → (n²−n,) vector
       │   return min(off_diagonal)
       │
       ▼
fitness: float (degrees)   ← higher is better
```

### Visual Summary

```
┌───────────────────────────────────────────────────────────────┐
│                    ONE FITNESS EVALUATION                      │
│                                                               │
│  Chromosome (m*p,)                                            │
│       │                                                       │
│       ▼  chunk by p                                           │
│  [(µ₁,σ₁), ..., (µₘ,σₘ)]                                    │
│       │                                                       │
│       ▼  gaussian_parameters_to_unit_amplitude_curves         │
│  Basis Functions (d, m)                                       │
│       │                                                       │
│       ├── SceneConfig ──┐                                     │
│       │                 ▼                                     │
│       ▼  simulate_sensor_output                               │
│  Sensor Outputs (m, n)                                        │
│       │                                                       │
│       ▼  compute_distance_matrix (SAM, axis=1)                │
│  Distance Matrix (n, n)                                       │
│       │                                                       │
│       ▼  min_based_dissimilarity_score                        │
│  Fitness: float (min SAM angle in degrees)                    │
│       │                                                       │
│       ▼  returned to PyGAD                                    │
└───────────────────────────────────────────────────────────────┘
```

### Step 7: GA Evolution (Outer Loop)

```
For each generation:
    1. PyGAD evaluates fitness_func for all sol_per_pop chromosomes
    2. If niching enabled:
         a. Compute population distance matrix (Euclidean or optimal-pairing)
         b. Calculate niche counts via sharing coefficient:
              sh(d) = 1 - (d/σ_share)^α  if d < σ_share, else 0
         c. shared_fitness[i] = original_fitness[i] / niche_count[i]
         d. Use shared_fitness for parent selection
         e. Restore original_fitness for elitism
    3. Select parents → crossover → mutation (diversity_preserving_mutation)
    4. Repeat
```

---

## 6. Next Steps & Strategic Targets

With the architectural plumbing clean, the focus shifts to **algorithmic
improvements** — the places where better science yields better sensors.

### 6.1 MAP-Elites Exploration

A MAP-Elites grid indexed by basis function center wavelengths (e.g., µ₁ × µ₂)
is already prototyped in `scripts/run_map_elites.py`. The strategic value:

- **Illumination over optimization:** MAP-Elites explores the full phenotype
  space, revealing *qualitatively different* high-fitness designs that a
  single-objective GA would collapse past.
- **Next step:** Extend to higher-dimensional behavior descriptors (e.g.,
  µ₁ × σ₁ × µ₂) and integrate with multi-condition `SceneConfig` sweeps to
  find designs robust across temperature ranges.

### 6.2 Precomputing Fixed Scene Physics

In the current pipeline, `blackbody_emit(λ, T, n)` and `tau_air^r` are
recomputed on every fitness call despite being constant for a given
`SceneConfig`. The optimization:

```python
# Computed ONCE per SceneConfig:
fixed_spectrum = (air_transmittance ** r) * blackbody_emit(λ, T, n)  # (d, 1)
weighted_emissivity = fixed_spectrum * emissivity_curves              # (d, n)

# Per fitness call (only the basis-dependent part):
sensor_outputs[:, j] = trapezoid(weighted_emissivity[:, j:j+1] * basis_functions, λ, axis=0)
```

This eliminates redundant blackbody + atmospheric computation from the hot loop.
With `d ≈ 300` and populations of 200+, this is a meaningful constant-factor
speedup.

### 6.3 Euclidean vs. SAM Distance Mismatch in Niching

A subtle but important inconsistency: the **fitness metric** uses SAM (angular
distance in sensor-output space), but the **niching distance** between
chromosomes uses Euclidean distance in gene space (µ, σ parameter space).

This means two chromosomes can be far apart in gene space but produce nearly
identical sensor outputs (e.g., two Gaussians with different σ but similar
integrated response). Conversely, small gene-space perturbations near spectral
features can cause large fitness jumps.

**Potential fix:** Compute niching distance in *phenotype space* (SAM distance
between the sensor output vectors) rather than *genotype space*. This requires
caching `sensor_outputs` per chromosome per generation, trading memory for
alignment between what we optimize and how we maintain diversity.

### 6.4 Vectorized Fitness Evaluation

The current `simulate_sensor_output` loops over substances (`for i in
range(n)`). A fully vectorized version using 3-D broadcasting:

```python
# (d, 1, n) * (d, 1, 1) * (d, 1, 1) * (d, m, 1) → (d, m, n)
abso_spec_3d = tau_air[:, :, None] * bb_emit[:, :, None] * emissivity[:, None, :] * basis[:, :, None]
sensor_outputs = np.trapezoid(abso_spec_3d, wavelengths, axis=0)  # (m, n)
```

This replaces the Python loop with a single NumPy broadcast, which is especially
impactful when `n` is large.

### 6.5 Alternative Fitness Objectives

The `dissimilarity_scoring` module already contains experimental alternatives:

| Function | Formula | Use Case |
|----------|---------|----------|
| `min_based_dissimilarity_score` | `min(off_diag)` | **Current default.** Conservative maximin. |
| `mean_min_based_dissimilarity_score` | `mean · min^α` | Balances average separation with worst case. |
| `weighted_mean_min_dissimilarity_score` | `β·mean + (1−β)·min` | Linear blend, tunable conservatism. |
| `group_based_dissimilarity_score` | `mean(inter-group distances)` | When substances belong to known categories (e.g., hazardous vs. benign). |

These are ready to plug in as alternative fitness objectives for
multi-objective or experimental runs.

### 6.6 Multi-Condition Robustness

`load_substance_atmosphere_data` already supports parameter meshgrids (multiple
temperatures, distance ratios, refractive indices), returning a
`list[SceneConfig]`. The fitness evaluator currently handles single conditions.

**Next step:** A meta-fitness that evaluates across multiple `SceneConfig`
conditions and returns the *worst-case* fitness:

```
meta_fitness(chromosome) = min( fitness(chromosome, scene) for scene in scenes )
```

This would optimize for designs robust to environmental variation — critical for
field-deployed sensors.

---

*Document generated: March 2026. Reflects the codebase state at commit `ed01ad7`
(finalize architecture sweep).*
