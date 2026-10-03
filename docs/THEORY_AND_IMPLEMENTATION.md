# LWI Microbolometer Design — Unified Onboarding & Codebase Reference

Single reference for **what this repository implements**: research motivation, problem intuition, data and PoC scope, how the search strategy evolved, what the key plots mean, then the full technical stack—physics and fitness, tensor conventions, package layout, scripts, and tests. Dependencies and tool versions live in **`pyproject.toml`** (Python ≥ 3.12; NumPy, pandas, SciPy, scikit-learn, matplotlib, seaborn, openpyxl, PyGAD, tqdm, numba, PyYAML, **`cma`** for CMA-ES / CMA-ME).

**Keeping this file current:** When a change affects public behavior, module boundaries, or how someone runs experiments, update this document in the same change as the code.

**Documentation map**

- **`THEORY_AND_IMPLEMENTATION.md` (this file):** Strategy, physics, tensor contracts, package layout, and compact “how it computes” notes for key algorithms.
- **`docs/01_EXPERIMENT_PIPELINE.md`:** v2_paper_2026 suite reproduction: step dataflow, budgets, environment variables, pickle artifacts, and detailed algorithm walkthroughs (optimal pairing, MAP-Elites, CMA-ME).
- **`scripts/v2_paper_2026/README.md`:** Runnable `uv run` commands, CPU scheduling waves, and a concise environment-variable table.

---

## Table of contents

1. [Project philosophy & problem intuition](#1-project-philosophy--problem-intuition)
2. [Data origin & specifics (proof of concept)](#2-data-origin--specifics-proof-of-concept)
3. [Evolution of the search strategy](#3-evolution-of-the-search-strategy)
4. [Key experiments, visualizations & insights](#4-key-experiments-visualizations--insights)
5. [Problem, goals, and technical summary](#5-problem-goals-and-technical-summary)
6. [Physics and fitness](#6-physics-and-fitness)
7. [Tensor shapes and conventions](#7-tensor-shapes-and-conventions)
8. [Architecture layers and dependency direction](#8-architecture-layers-and-dependency-direction)
9. [Package map: `lwi_microbolometer_design`](#9-package-map-lwi_microbolometer_design)
10. [Module reference: `data/`](#10-module-reference-data)
11. [Module reference: `simulation/`](#11-module-reference-simulation)
12. [Module reference: `analysis/`](#12-module-reference-analysis)
13. [Module reference: `ga/`](#13-module-reference-ga)
14. [Module reference: `map_elites/` (QD algorithms)](#14-module-reference-map_elites-qd-algorithms)
15. [Module reference: `visualization/`](#15-module-reference-visualization)
16. [Top-level public API](#16-top-level-public-api)
17. [Scripts layout](#17-scripts-layout)
18. [Tests](#18-tests)
19. [Design choices, limitations, and writing-up](#19-design-choices-limitations-and-writing-up)
20. [Optional extensions and known gaps](#20-optional-extensions-and-known-gaps)

---

## 1. Project philosophy & problem intuition

### 1.1 The problem in one sentence

We are **optimizing the spectral response functions (channels)** of a **single-pixel multispectral** sensor so that **specific target substances** produce fingerprints that are **as separable as possible**—not by picking a classifier after the fact, but by **jointly shaping the physical channel responses** under a forward radiance model.

### 1.2 Physical intuition: the “white powder” story

To human vision in the **RGB** band, many powdered materials look interchangeable: fine white grains can correspond to very different chemistries. The running PoC uses substances that illustrate this idea dramatically—**Cocaine, Heroin, Sodium Bicarbonate (baking soda), and Sucralose (sugar substitute)** can all present as similar-looking white powders under visible light.

In the **thermal infrared**, the story changes. Each material has a characteristic **spectral emissivity** ε(λ): the way it “leaks” thermal radiation by wavelength differs even when bulk appearance does not. If the sensor could tune **narrowband-ish spectral cones**—here modeled as **Gaussian basis responsivities** with tunable center **μ** and width **σ** per channel—it can place those cones where emissivity curves **diverge**, turning subtle spectral differences into **large angular separation** between short fingerprint vectors (one scalar per channel per substance).

That mental model is the bridge from “microbolometer application” to “optimize Φᵢ(λ) so integrated outputs separate substances.”

### 1.3 How this differs from a typical industry pattern

Many workflows **separate** design and decision: choose a few fixed bands or filters, collect data, then train a **downstream classifier**. This codebase instead treats **channel shapes as design variables** and evaluates discrimination inside the **same forward physics loop** that maps (scene + substance spectra + channels) → fingerprints.

The objective is deliberately conservative: we maximize **worst-case pairwise separation** in **sensor-output space** using the **Spectral Angle Mapper (SAM)** (see §6). That is **end-to-end parametric optimization of physical response curves** for **maximin** spectral geometry—not a substitute for a full noise and calibration model unless those layers are added explicitly.

---

## 2. Data origin & specifics (proof of concept)

### 2.1 The four-substance PoC is intentional

Abstract symbols **`n`** (substances) and **`d`** (wavelength samples) in tensor tables (§7) generalize the code. The **current proof-of-concept** driving much of the presentation and development narrative uses **exactly four substances**: **Cocaine, Heroin, Sodium Bicarbonate, Sucralose**.

That choice is **not** a claim that production deployment stops at four targets. It **controls problem scale** so we can:

- stress-test optimizers and archives on a loop that is cheap enough to iterate;
- visualize fingerprints, distance structure, and QD maps without drowning in combinatorial clutter;
- establish that **parametric channel optimization + maximin SAM** behaves sensibly before scaling **`n`**, richer spectra, or fabrication constraints.

Excel-driven loading and `SceneConfig` do **not** hard-code those four names in the core DTO; the **dataset** (column headers and emissivity sheets) defines **`n`** and `substance_names`.

### 2.2 Scene configuration and multi-environment readiness

**`SceneConfig`** bundles everything about the **scene** except the sensor being optimized: wavelengths, emissivity matrix **`(d, n)`**, atmospheric transmittance, temperature, path-length handle (`atmospheric_distance_ratio`), refractive index, and substance names (see §10).

**`load_substance_atmosphere_data`** can emit **one** `SceneConfig` or a **list** of them when temperature, distance ratio, or refractive index are given as **lists**—internally a **meshgrid** over combinations. That is the hook for:

- **robust training** (e.g. minimax over scenes inside **`MinDissimilarityFitnessEvaluator`**);
- **robustness evaluation** sweeps (§12 `analysis/robustness.py`, §15 plots).

So the PoC is small in **`n`**, but the **pipeline** is already shaped for **multi-environment** studies.

---

## 3. Evolution of the search strategy

This section is the **compass** for why the repository contains **both** classic GAs and **quality–diversity (QD)** methods—not an accident of feature creep.

### 3.1 V1: grid search

Early exploration used **exhaustive or dense grid search** over discrete parameterizations. That gave interpretable baselines but **hit a computational ceiling** as the number of channels and continuous parameters grew. The cost of covering the space uniformly scales poorly; we needed **search** that concentrates evaluations in promising regions.

### 3.2 Genetic algorithms: higher scores, but a structural bias

**Genetic algorithms (GA)** scaled better and **raised** achievable fitness. PyGAD-based **`AdvancedGA`** adds **niching**, adaptive mutation, and optional **optimal pairing** so permutations of channel order do not artifactually inflate genotype distance (§13).

Still, a standard GA is **“survival of the fittest”** in a single soup: selection pressure plus elitism tends to **collapse the population toward one high-performing mode**. **Niching** and **custom mutation** mitigate overlap in **parameter space**, but they do not change the core outcome that **one run** often returns **one dominant family** of solutions. That is **mode collapse** in practice: excellent for a single peak, misleading if the design landscape is **multimodal**.

### 3.3 Evidence of multimodality: ensemble GAs and the QD pivot

To test whether “one best chromosome” was an artifact of the optimizer, we ran **ensembles of many independent GA runs** (e.g. on the order of **50** in the presentation pipeline). Independently seeded runs **repeatedly** landed on **several distinct high-scoring “families”** of channel layouts—**not** minor perturbations of the same design, but **qualitatively different** allocations of spectral mass.

That empirical picture motivated **quality–diversity**:

- **MAP-Elites** maintains an **archive** of elites indexed by **behavior descriptors** (here, low-dimensional features of the μ layout), so **different regions of design space** can each keep a champion.
- **CMA-ME** couples **CMA-ES**-style adaptation to **archive improvement**, illuminating behavior space faster than random local mutation alone (§14).

### 3.4 Goal shift: from “the best sensor” to a **sensor library**

For **fabrication and integration**, a single argmax design is brittle. If the top design is **sensitive to process variation**, **mask alignment**, or **forbidden μ/σ combinations**, you want **many** nearly equivalent options.

The QD track therefore targets a **library of high-performance sensor architectures**: tens of **diverse** elites with **similar top-tier fitness**, not one lonely optimum. Downstream steps (cost, manufacturability, robustness under environmental grids) can **prune** that library without having to **re-run** a global search from scratch.

---

## 4. Key experiments, visualizations & insights

Plots are not endpoints; they answer specific **research questions**. Below ties **figures** to **claims** and **modules**.

### 4.1 iVAT and KMeans: multimodality in **solution** space

**Question:** Are high-fitness solutions **one cloud** or **several separated families**?

- **VAT / iVAT** (`analysis/vat.py`: **`vat_reorder`**, **`ivat_transform`**) reorder a **distance matrix** so block structure (clusters) becomes visible along the diagonal. Applied to **distances between solutions** (e.g. chromosomes or their sensor outputs), sharp blocks indicate **disjoint families**.
- **iVAT** uses a path-based (bottleneck) transform so cluster boundaries are often **cleaner** than raw distance heatmaps.
- **KMeans / population analysis** (`ga/population_analysis.py` with **`AnalysisConfig`**) groups the **population or elite set** and reports cluster-quality heuristics and textual summaries.

**Insight:** When ensemble runs and iVAT/KMeans **agree** on **multiple** well-separated high-fitness groups, that is strong evidence the **design landscape is multimodal**—supporting the QD narrative in §3.3.

**Where to look in code:** **`ga/visualization.py`** (`plot_ivat_analysis`, orchestrated by **`visualize_ga_results`**), MAP-Elites **`map_elites/visualization.py`** (permutation / labeling helpers for comparable curves).

### 4.2 Robustness degradation heatmaps: the simulation–reality **reality check**

**Question:** If we **freeze** a design that looked brilliant under **nominal** temperature and path settings, how much does fitness **collapse** under **other** plausible environments?

The main scientific skepticism toward simulation-only optimization is the **simulation–reality gap**. We address a **slice** of that honestly: **same forward model**, but **perturbed environmental parameters** (temperature, atmospheric distance ratio, refractive index grids via lists in data loading → multiple **`SceneConfig`**).

- **`analysis/robustness.py`** re-evaluates stored chromosomes across a **sequence of scenes**, tracking per-condition fitness, **retention** vs. nominal, worst case, etc.
- **`visualization/robustness_visualization.py`** turns those vectors into **heatmaps**, **distributions by condition**, **worst vs. nominal**, and **sensitivity** views.

**Insight:** Heatmaps that show **graceful** degradation support **transfer** claims; **cliffs** flag designs that **overfit** nominal physics. When training uses **`aggregation="min"`** across scenes (**minimax** robustness, §6.5), the optimizer is **explicitly** aligned with that critique rather than hoping average-case luck generalizes.

**Paper pipeline:** `scripts/v2_paper_2026/` includes robustness and minimax-oriented steps; YAML such as **`experiments/v2_paper_2026_minimax.yaml`** documents multi-scene / minimax-style runs.

---

## 5. Problem, goals, and technical summary

### 5.1 What the code is trying to do

The LWI microbolometer work package treats **spectral channel design** as an optimization problem. **Target substances** are given with **infrared emissivity vs. wavelength** on a fixed grid (typically from measurements or a library). The sensor is modeled as **several independent spectral channels**—here, **Gaussian-shaped spectral responsivities** with tunable center **μ** and width **σ** per channel. Passing scene radiance through those channels and integrating over wavelength yields a **short fingerprint vector** per substance (one scalar per channel). The search adjusts channel parameters so those fingerprints are **easy to tell apart** under a small forward-physics chain (blackbody, simple atmosphere, emissivity, integration).

This setup is **single-pixel / non-imaging multispectral** (or narrowband multichannel) **response synthesis**: spatial imaging, optics layout, and readout electronics are out of scope unless added elsewhere. **Microbolometer** names the application; mathematically the code optimizes **spectral responsivity** shapes.

The discrimination objective is deliberately conservative: maximize the **worst pairwise separation** among substances in a **spectral-angle** metric (SAM), not a full classifier trained on noise—see §6.

### 5.2 Short technical summary (for abstracts or methods blurbs)

- **Task:** Computational **channel design** / **filter optimization** for **material discrimination** from thermal-IR-style measurements, using a discrete-wavelength forward model.
- **Forward model:** At each wavelength, radiance scales as **τ(λ)^r · B(λ,T,n) · ε(λ)**; multiply by channel responsivity **Φ_i(λ)**, then **trapezoidal integration** over λ → one scalar per channel per substance.
- **Objective:** **Maximin** over substance pairs: maximize the **minimum** pairwise **Spectral Angle Mapper (SAM)** angle between fingerprint vectors (degrees). Optimizers **maximize** this scalar.

---

## 6. Physics and fitness

### 6.0 How formulas are written here

Many Markdown viewers **do not render LaTeX**. **§6.1** uses **plain text and Unicode** so the integral and symbols stay readable everywhere. **§6.1a** repeats the same content as **LaTeX** for copy-paste into papers (`$…$` / `\[…\]`); treat that block as **optional** if the preview looks like raw code.

### 6.1 Forward model (per substance, per channel)

Integrated sensor response for channel **i** and substance **j** (discrete grid, trapezoid rule):

```text
S[i,j] ≈ ∫ τ(λ)^r · B(λ, T, n) · ε[j](λ) · Φ[i](λ) dλ
```

**λ** is wavelength in **µm**. Implemented in `simulate_sensor_output` using nonuniform trapezoidal weights and matrix multiplication across all channels and substances. With dimensionless emissivity, transmission and basis functions, the output is integrated radiance in **W/(m²·sr)**; a voltage output requires a separate calibrated transfer model.

| Symbol | Meaning |
|--------|---------|
| λ | Wavelength (µm) |
| τ(λ) | Atmospheric transmittance (loaded curve) |
| r | `atmospheric_distance_ratio` — exponent on τ (path / visibility handle) |
| B(λ, T, n) | Spectral radiance per µm of vacuum wavelength; **n²** applies to a homogeneous, isotropic, nonattenuating medium with a fixed index, not arbitrary interfaces or dispersive optical stacks |
| ε[j] | Emissivity of substance j |
| Φ[i] | Channel i responsivity on the grid (default: unit-amplitude Gaussian from genes) |

#### 6.1a LaTeX copy (for manuscripts; may not render in Markdown)

```latex
S_{i,j} \approx \int \tau(\lambda)^{r} \, B(\lambda, T, n) \, \varepsilon_j(\lambda) \, \Phi_i(\lambda) \, d\lambda
```

### 6.2 SAM and why magnitude is de-emphasized

The fingerprint for substance **j** is column **j** of **`sensor_outputs`**, shape `(m,)`. **SAM** is the angle between two such vectors, in **degrees**. It is **insensitive to overall scale**, so a common gain across channels does not change the angle—useful when total radiance varies but **relative channel pattern** carries discrimination.

### 6.3 Scalar fitness: maximin

Pairwise SAM values form an `(n, n)` matrix. **`min_based_dissimilarity_score`** returns the **minimum off-diagonal** entry; **larger is better**. GA, MAP-Elites, and CMA-ME **maximize** this scalar.

### 6.4 Degenerate fingerprints

If a fingerprint vector has (near) **zero norm**, **`spectral_angle_mapper`** returns **`0.0`** degrees (treated as maximally similar) so **arccos** never receives invalid input. The guard applies only to finite vectors with norm below **`1e-15`**. Nonfinite inputs raise errors; execution failures are not optimization scores. Genuine zero off-diagonal angles remain part of maximin scoring.

### 6.5 Multi-condition (robust) evaluation

**`MinDissimilarityFitnessEvaluator`** can hold **one** or **many** **`SceneConfig`** instances. With multiple scenes, per-scene scores combine as:

- **`"min"`** — worst case across conditions (**minimax**),
- **`"mean"`** — average across conditions.

Single-scene mode uses **`aggregation="single"`**. YAML wiring: `experiment.data["robustness_aggregation"]` with list-valued temperature, distance ratio, or refractive index in data (see **`ga/experiment.py`**). Passing **multiple** scenes with **`aggregation="single"`** logs a **warning** and coerces to **`"min"`**.

---

## 7. Tensor shapes and conventions

| Symbol | Meaning | Typical size |
|--------|---------|--------------|
| `d` | Wavelength samples | ~100–500 |
| `n` | Substances | PoC **4**; code supports larger **`n`** (e.g. ~4–21 in experiments) |
| `m` | Basis functions / channels | Often 4 |
| `p` | Genes per basis (Gaussian) | 2 (μ, σ) |

| Object | Shape | Notes |
|--------|-------|-------|
| `SceneConfig.wavelengths` | `(d,)` | Canonical **1D** after DTO validation |
| `SceneConfig.emissivity_curves` | `(d, n)` | |
| `SceneConfig.air_transmittance` | `(d,)` | First column used if Excel gives `(d, k)` |
| Chromosome | `(m * p,)` | Flattened parameters |
| `basis_functions` | `(d, m)` | From `parameters_to_curves` |
| `sensor_outputs` | `(m, n)` | Column index j = substance j |
| Distance matrix | `(n, n)` | SAM between columns when `axis=1` |

**Input domains:** The integration grid is finite, positive, strictly increasing and has at least two samples. Transmission is finite in [0, 1]; temperature and index are positive; distance ratio is nonnegative. Basis functions and emissivity are finite. Emissivity retains signed and out-of-range measurements without clipping; these values cannot automatically be interpreted as physical emissivity. The nominal workbook includes a minimum of **-0.013948**. The spectral and atmosphere wavelength columns must match exactly; the loader performs no interpolation.

**Integration:** For trapezoidal weights **`w`**, the matrix operation is **`basis.T @ ((w * tau**r * B)[:, None] * emissivity)`**. At **`r=0`**, attenuation is one, including where tau is zero. Floating-point summation order can differ from per-substance integration.

---

## 8. Architecture layers and dependency direction

```text
data  →  simulation  →  analysis  →  ga / map_elites  →  visualization
(load)    (physics)      (metrics)     (search)            (plots only)
```

Core simulation and distance/scoring modules are independent of optimizers. Environmental robustness is an evaluation orchestration layer: it imports **`MinDissimilarityFitnessEvaluator`** locally so training and evaluation share one fitness contract without an import-time cycle.

**Useful patterns:** **`parameters_to_curves`** is a **Strategy**-style hook: swap Gaussian curves for another parameterization without changing the GA. **`MinDissimilarityFitnessEvaluator`** is a **class** (not a closure) so **`fitness_func`** can be **pickled** for multiprocessing.

---

## 9. Package map: `lwi_microbolometer_design`

```text
src/lwi_microbolometer_design/
├── __init__.py                 # Curated public re-exports
├── data/
│   ├── scene_config.py
│   └── substance_atmosphere_data.py
├── simulation/
│   ├── _validation.py
│   ├── blackbody.py
│   ├── gaussian_parameter_to_curves.py
│   └── sensor_simulation.py
├── analysis/
│   ├── distance_metrics.py
│   ├── distance_matrix.py
│   ├── dissimilarity_scoring.py
│   ├── optimal_pairing_distance.py
│   ├── robustness.py
│   ├── elite_clustering.py
│   ├── clodd.py
│   ├── vat.py
│   └── __init__.py
├── ga/
│   ├── advanced_ga.py
│   ├── diversity.py
│   ├── experiment.py
│   ├── fitness.py
│   ├── ga_configuration.py
│   ├── mutations.py
│   ├── population_analysis.py
│   ├── result_extraction.py
│   ├── tuning.py
│   ├── visualization.py
│   └── __init__.py
├── map_elites/
│   ├── algorithm.py
│   ├── archive.py
│   ├── cma_me.py
│   ├── emitters.py
│   ├── normalization.py
│   ├── polish.py
│   ├── visualization.py
│   └── __init__.py
└── visualization/
    ├── distance_matrix_visualization.py
    ├── sensor_output_visualization.py
    ├── robustness_visualization.py
    └── __init__.py
```

---

## 10. Module reference: `data/`

### `scene_config.py`

- **`SceneConfig`** — Frozen, slots dataclass; field bindings are frozen but NumPy array contents remain mutable. It **canonicalizes** wavelengths and transmittance to 1D and validates domains, shapes and names. No cached forward-model quantities depend on array immutability. Single **DTO** for “everything about the scene except the sensor being optimized.” See §2 for PoC substance count and multi-environment use.

### `substance_atmosphere_data.py`

- **`load_substance_atmosphere_data(...)`** — Reads the spectral workbook with headers (λ + emissivity columns), and the atmosphere workbook without headers (λ + transmission). Only the first transmission column is used; extra columns are not additional environments. Scalars and length-one sequences return one **`SceneConfig`**. Nonempty 1D sequences form a Cartesian product in distance, temperature, index order and return a list when there is more than one combination. Empty and multidimensional sequences are rejected. Typed with **`@overload`** for static checkers.

**Implicit behavior:** Multi-condition lists support **robust training** (min/mean aggregation) and **robustness evaluation** (grids of scenes).

---

## 11. Module reference: `simulation/`

### `blackbody.py`

- **`blackbody_emit(spectra, temperature_k, refractive_index)`** — Planck spectral radiance **`2 h c² n² × 10²⁴ / (λ⁵ (exp(hc×10⁶/(λkT))−1))`**, in **W/(m²·sr·µm)**. Historical CODATA 2010 constants are retained. Log-domain evaluation avoids intermediate overflow in the Wien tail; only the final radiance may underflow. The integral over vacuum wavelength is **`n² σ T⁴ / π`**; no extra π belongs in the radiance formula. See [NIST TN 910-8, Eq. 12.3a](https://nvlpubs.nist.gov/nistpubs/Legacy/TN/nbstechnicalnote910-8.pdf) for the wavelength and medium convention.

### `gaussian_parameter_to_curves.py`

- **`gaussian_parameters_to_unit_amplitude_curves(gaussian_parameters, wavelengths)`** — List of `(μ, σ)` → matrix `(d, m)`. **Snaps μ to nearest discrete wavelength** so peak amplitude is **exactly 1** on the grid. Default **`parameters_to_curves`** callback for GA/MAP-Elites.

### `sensor_simulation.py`

- **`simulate_sensor_output(...)`** — Full chain: validated inputs, blackbody, τ^r, nonuniform trapezoidal weights and matrix integration. Output shape is **`(m, n)`**. It preserves the trapezoidal integral, not bitwise equality of different summation orders.

---

## 12. Module reference: `analysis/`

### `distance_metrics.py`

- **`spectral_angle_mapper`** — SAM in degrees; **norm guard** at `1e-15`; clamps dot product for `arccos` stability.

### `distance_matrix.py`

- **`compute_distance_matrix`** — Unified API for:
  - **Array mode:** e.g. `sensor_outputs` with **`axis=1`** → compare substance columns with **`distance_func`** (SAM).
  - **List / object mode:** arbitrary items.
  - **`use_optimal_pairing=True`:** treats chromosomes as grouped parameters; uses Hungarian matching so **permutation of groups** does not inflate distance (see below).

### `optimal_pairing_distance.py`

- **`calculate_optimal_pairing_distance`** — Builds cost matrix between **groups** of parameters (e.g. `(μ, σ)` pairs), solves **linear sum assignment** (SciPy), returns minimum-cost pairing distance. Vectorized group distances use **`scipy.spatial.distance.cdist`**.
- **Computation (concise):** Each chromosome is split into **equal-length** groups of **`params_per_group`** adjacent genes. Build an **n×n** matrix of pairwise group distances (`cdist` for built-in metrics, or a nested loop for a custom **`distance_func`**). **`scipy.optimize.linear_sum_assignment`** (Hungarian algorithm) finds a **minimum-cost perfect matching**; the returned distance is the **sum** of costs on matched pairs (**O(n³)** in the number of groups). This affects **niching / genotype distance** only; **fitness** remains **SAM** on sensor outputs (§6). Step-by-step narrative: **`docs/01_EXPERIMENT_PIPELINE.md`** §8.1.

### `dissimilarity_scoring.py`

- **`min_based_dissimilarity_score`** — **Production fitness** reducer (min off-diagonal).
- **`mean_min_based_dissimilarity_score`**, **`group_based_dissimilarity_score`**, **`weighted_mean_min_dissimilarity_score`** — Alternative objectives; available for experiments; not the default hot path.
- **`_ensure_distance_matrix`** — Internal helper to accept raw items or precomputed matrix.

### `vat.py`

- **`vat_reorder`** — Reorders distance matrix for **VAT** visualization.
- **`ivat_transform`** — **iVAT** (path-based / bottleneck) transform for sharper cluster visuals. See §4.1 for interpretation.

### `robustness.py` (environmental stress testing of fixed designs)

**Phase-1-style** analysis: given elites or any list of solutions with **`chromosome`** + **`fitness`**, re-evaluate across a **sequence of `SceneConfig`** (e.g. temperature / distance / index grid).

- **`ConditionLabel`** / **`RobustnessResult`** — Per-solution storage: fitness vector across conditions, **retention ratio** (`min_fitness / nominal_fitness`), CV, worst-condition index, etc.
- **`evaluate_elite_fitness`** — One chromosome, one scene → scalar (duplicates evaluator logic for standalone use).
- **`evaluate_elite_robustness`** — One elite × many scenes.
- **`evaluate_solutions_robustness`** — Generic batch (sorts by fitness, optional **`top_n`**, tqdm).
- **`evaluate_archive_robustness`** — Wrapper: MAP-Elites **`archive.values()`** dicts.
- **`find_nominal_scene_index`** / **`align_nominal_fitness_to_training_scene`** — Align “nominal” to a grid cell so retention metrics match **re-evaluated** baselines (archive fitness can differ slightly from a fresh forward pass).
- **`summarise_robustness`** — Aggregate stats over a list of `RobustnessResult`.

**Note:** Robustness here is **environmental parameter sweeps** on the **same forward model**, not detector noise injection (unless added in simulation or post-process `sensor_outputs` elsewhere). See §4.2 for how plots support claims.

### `analysis/__init__.py`

Re-exports scoring, `compute_distance_matrix`, `spectral_angle_mapper`, `calculate_optimal_pairing_distance`, robustness symbols, VAT helpers.

---

## 13. Module reference: `ga/`

### `fitness.py`

- **`MinDissimilarityFitnessEvaluator`** — Pickle-friendly class; **`fitness_func(ga, chromosome, idx)`**:
  1. Decode chromosome into tuples of length `params_per_basis_function`.
  2. `parameters_to_curves(basis_params, scene.wavelengths)`.
  3. `simulate_sensor_output` with that scene’s arrays.
  4. `compute_distance_matrix(..., axis=1)`.
  5. `min_based_dissimilarity_score`.
  6. If multiple scenes: **`min`** or **`mean`** of per-scene scores.

**`SUPPORTED_AGGREGATIONS`** — `("single", "min", "mean")`.

### `advanced_ga.py`

- **`AdvancedGA`** — Subclass of **`pygad.GA`** with optional **fitness sharing / niching** in parent selection (`run_select_parents` override). Elitism uses **raw** fitness; selection can use **shared** fitness.
- **`NichingConfig`** — `enabled`, `sigma_share`, `alpha`, `use_optimal_pairing`, `params_per_group`, `optimal_pairing_metric`, etc. (see module docstring examples).
- **`niche_sharing_coefficient`**, **`compute_chromosome_distance`** — Helpers for sharing and distance reporting.

**Nuance:** Niching distances live in **genotype / parameter space** (Euclidean or optimal-pairing), while fitness uses **phenotype SAM** in sensor-output space. The two need not agree.

### `diversity.py`

- **`compute_population_distance_matrix`** — Routes to standard or optimal-pairing distances for **population** vectors.
- **`calculate_population_diversity`** — Scalar summary (e.g. mean pairwise distance).

### `mutations.py`

- **`MutationConfig`** — Presets (`balanced`, `conservative`, `aggressive`) and thresholds for stagnation / diversity.
- **`diversity_preserving_mutation`** — PyGAD mutation callback: progress-aware step size, stagnation detection, Cauchy heavy tails, optional push away from mean, gene restarts, discrete vs. continuous gene handling. All draws use the NumPy stream seeded by PyGAD. Two-value lists remain discrete; stepped spaces use PyGAD’s high-exclusive grid. Large module; treat as **one adaptive operator** unless debugging.

### `ga_configuration.py`

- **`create_ga_config`** — Builds kwargs dict for **`AdvancedGA`** / PyGAD (mutation, crossover, elitism, callbacks, niching wiring).
- **`load_ga_configuration_from_csv`** — Hydrate GA params from tuning CSV output (+ internal helpers).

### `experiment.py`

- **`ExperimentConfig`** — Dataclass mirroring YAML `experiment` block.
- **`load_experiment_config`** — Parses YAML, resolves paths (project root detection; special case if YAML lives under `experiments/`).
- **`create_fitness_evaluator_from_experiment`** — Loads data → `SceneConfig` or list → builds **`MinDissimilarityFitnessEvaluator`** with Gaussian curves and SAM; reads **`robustness_aggregation`** for multi-scene.
- **`create_search_space_from_experiment`** — Builds **`HyperparameterSearchSpace`** for **`tuning`** (local import to avoid cycles).
- **`create_gene_space_from_experiment`** — Repeats `sensor.param_bounds` for `num_basis_functions` channels.

### `tuning.py`

- **`GenerationTracker`** — Records generation-wise stats (mean/best fitness, diversity).
- **`HyperparameterSearchSpace`**, **`TuningResult`**, **`HyperparameterTuner`** — Grid / random search over GA hyperparameters, parallel execution, CSV/JSON output.
- **`run_single_configuration`** — One GA trial worker.
- **`create_default_search_space`**, **`create_focused_search_space`** — Presets.

### `result_extraction.py`

- **`extract_basic_results`** — Standard dict from finished **`AdvancedGA`**. Per-generation mean fitness and diversity are returned only when actually tracked; missing histories are empty. Final diversity is a separate value. A failed `best_solution()` propagates its error rather than fabricating a zero-fitness candidate.

### `population_analysis.py`

- **`AnalysisConfig`**, **`analyze_population_diversity`** — Post-run clustering (KMeans / DBSCAN / adaptive), silhouette-based choices, fitness segmentation, textual **recommendations**. Many **`_`** helpers for internal pipeline. See §4.1.

### `visualization.py`

- **`ParametersToCurves`** type alias — Callable for plotting with non-Gaussian bases.
- **`visualize_ga_results`** — Orchestrates multiple figures (fitness, diversity, distributions, top designs, iVAT, …). Accepts **`parameters_to_curves`** and **`params_per_basis_function`** (defaults to Gaussian).
- **`plot_top_sensor_designs`**, **`plot_best_design`**, **`plot_ivat_analysis`**, **`plot_high_fitness_evolution`**, **`plot_fitness_spread_evolution`**, **`visualize_top_configurations`** — Focused plots / tuning visualization.

Helper **`_chromosome_to_basis_param_tuples`** centralizes chromosome decoding for plots.

---

## 14. Module reference: `map_elites/` (QD algorithms)

First-class package (not only scripts). **Feature extraction** still assumes **Gaussian-style** chromosomes (**even indices = μ**); changing basis parameterization requires revisiting **`extract_features`** / **`bin_coordinates`**.

### `archive.py`

- **`extract_features(chromosome)`** — **`(min(μ), second_min(μ))`** across all channels — **order-invariant** descriptor in μ space.
- **`bin_coordinates`** — Maps two μs to **`(x_bin, y_bin)`** in a `grid_resolution`² grid; clamps to `mu_range`.
- **`reachable_cell_count`** — Features are sorted (μ₁ ≤ μ₂), so only **upper-triangle** cells are reachable: **`R = G(G+1)/2`** for grid size **G**.
- **`archive_coverage_pct`** — Filled / reachable.
- **`initialize_archive`** — Random uniform seeds; keeps best per cell; optional tqdm.

### `algorithm.py`

- **`mutate_chromosome`** — Gaussian perturbation; **larger sigma for μ (even indices)**, smaller for σ (odd indices) — encourages **cross-bin** movement along wavelength.
- **`run_map_elites`** — Main loop: draw a **parent** uniformly from the archive (or uniform random genes if the archive is empty); **`mutate_chromosome`**; evaluate fitness; map to **`(x_bin, y_bin)`** via **`extract_features`** / **`bin_coordinates`**; **replace** the cell if vacant or if the child has **strictly higher** fitness. Returns **`dict[(i,j), elite_dict]`** with `chromosome`, `fitness`, `mu_1`, `mu_2`. Pseudocode: **`docs/01_EXPERIMENT_PIPELINE.md`** §8.2.

### `normalization.py`

- **`UnitCubeScaler`** — Affine map gene bounds → **[0,1]ⁿ** for CMA-ES stability; **`normalize` / `denormalize`** with clip.

### `emitters.py`

- **`EmitterBase`** — Abstract emitter: **`ask`**, **`tell(...)`**, **`restart`**, **`batch_size`**, **`converged`**.
- **`OptimizingEmitter`** — Wraps **`cma.CMAEvolutionStrategy`** on the unit cube **[0,1]ⁿ** (`UnitCubeScaler`). **`ask`:** CMA proposes candidates in **normalized** coordinates; values are **denormalized** to **`gene_space`** bounds for fitness evaluation. **`tell`:** builds a list of scalar **objectives** for CMA’s ranking—if **any** candidate had **archive improvement** `> 0`, **improvers** are ranked by **higher improvement first** (non-improvers get a worse objective than the best improver); if **no** improver in the batch, objectives are **negative raw fitness** so CMA still moves toward better nominal scores. **`cma.tell`** receives **normalized** points. Exact transform: **`docs/01_EXPERIMENT_PIPELINE.md`** §8.3.

### `cma_me.py`

- **`run_cma_me`** — (**1**) **Seed** the MAP-Elites archive with **`num_initial`** random evaluations (same geometry as vanilla MAP-Elites). (**2**) Create **`num_emitters`** **`OptimizingEmitter`** instances, each initialized at a **random archive elite’s** chromosome. (**3**) Until the **evaluation budget** is exhausted: for each emitter, **`ask`** a batch → evaluate **`fitness_func(None, chromosome, 0)`** for each → compute per-candidate **archive improvement** (new bin: `imp = fitness`; occupied bin: `imp = max(0, fitness − incumbent)`; else `0`) → update the shared archive → **`tell`** for a full batch. The last batch is limited to remaining evaluations; its archive updates count even when there are too few points for `tell`. Vacant cells accept every finite score, including zero and negative scores. (**4**) When an emitter **converges** (CMA internal stop or **`restart_patience`** generations without any archive improvement in its batches), **`restart`** its mean at a **new random elite**. Returns **`(archive, metadata)`** including **`history`** (evals, archive size, best fitness, coverage). Reference: Fontaine et al., *Covariance Matrix Adaptation for the Rapid Illumination of Behavior Space* (GECCO 2020). Loop detail: **`docs/01_EXPERIMENT_PIPELINE.md`** §8.3–8.4.

### `polish.py`

- **`polish_single_elite_hc`** — Random perturbation hill-climbing; **`adaptive_iterations`** scales effort vs. gap to a hardcoded target fitness (`59.0` in code — **presentation-era calibration**; change if the fitness scale shifts).
- **`polish_single_elite_cma`** — Local **ask/evaluate/tell** CMA-ES from an elite in normalized coordinates. The known initial score is reused, all candidate evaluations count against **`max_fevals`**, and a final partial batch is evaluated without calling `tell`. Both polish APIs accept a seed; HC workers in the presentation suite use **`42 + elite index`**.

### `map_elites/visualization.py`

- **`plot_map_elites_heatmap`**, **`plot_top_elites`**, **`plot_polished_elites`**, **`plot_cma_me_progress`**, **`plot_top_from_population`**, **`plot_top_individual_curves`**, **`set_tight_ylim_stacked_spectra`** — QD-specific figures; includes helpers to cluster / label **permutation families** of chromosomes for legibility.

---

## 15. Module reference: `visualization/`

Standalone plotting (matplotlib/seaborn); safe to import from notebooks/scripts — **not** from hot physics paths.

- **`distance_matrix_visualization.py`** — `visualize_distance_matrix`, `visualize_distance_matrix_simple`; **`visualize_distance_matrix_large`** alias for compatibility.
- **`sensor_output_visualization.py`** — `visualize_sensor_output` for `(m, n)` matrices.
- **`robustness_visualization.py`** — `plot_fitness_degradation_heatmap`, `plot_fitness_distribution_by_condition`, `plot_worst_case_vs_nominal`, `plot_condition_sensitivity`, `plot_retention_histogram`. See §4.2 for interpretation.

---

## 16. Top-level public API

`lwi_microbolometer_design/__init__.py` re-exports a **curated** set of symbols (analysis + robustness + GA + MAP-Elites + simulation + visualization). It does **not** export everything (e.g. `SceneConfig`, `load_substance_atmosphere_data`, `ExperimentConfig` come from subpackages).

**Typical imports:** `from lwi_microbolometer_design.data import SceneConfig`, etc.

---

## 17. Scripts layout

Scripts are **applications**; they may duplicate small wiring patterns. Prefer adding **reusable logic under `src/`** (as with `map_elites` and `robustness`).

| Area | Path | Role |
|------|------|------|
| GA demos / tuning | `scripts/ga/` | `tune_ga.py`, `run_ensemble_demo.py`, `run_diversity_search.py`, `tune_niching_strategy.py` |
| MAP-Elites drivers | `scripts/map_elites/` | `run_map_elites_raw.py`, `run_map_elites_hc_polish.py`, `run_map_elites_cma_polish.py`, `run_cma_me.py` |
| Robustness driver | `scripts/robustness/` | `run_robustness_test.py` |
| v2_paper_2026 suite | `scripts/v2_paper_2026/` | **Numbered pipeline** `run_01_…`–`run_08_…` with shared helpers (`_budgets.py`, `_paths.py`, `_experiment_body.py`, `_plotting*.py`, `_ga_multistart.py`, `_artifacts.py`, `_progress.py`). README documents **CPU waves**, **env overrides**, and **`uv run python`** usage. Outputs under `outputs/v2_paper_2026/`. **Dataflow, artifacts, and full env catalog:** **`docs/01_EXPERIMENT_PIPELINE.md`**. |
| Prototypes | `scripts/prototypes/` | Ad-hoc analysis / verification |

**YAML:** Example **`experiments/v2_paper_2026_minimax.yaml`** for minimax / multi-scene runs (e.g. multi-cell temperature–distance grids).

---

## 18. Tests

Under `tests/`:

- **`test_spectral_contracts.py`** — Planck reference, Wien/Rayleigh–Jeans limits, radiance integral, nonuniform signed integration, real Excel alignment and domains.
- **`test_optimizer_contracts.py`** — Exact budgets, seeds, shared fitness, degeneracy, CSV overrides and honest histories.
- **`test_analysis_contracts.py`**, **`test_experiment_entrypoints.py`** — Pairwise expectations, clustering, plot defaults, worker forwarding and invalid configuration boundaries.
- **`test_scene_config.py`** — DTO validation / shapes.
- **`test_simulation.py`** — Forward model sanity.
- **`test_analysis.py`** — Metrics / matrices / scoring.
- **`test_advanced_ga_minimal.py`** — GA smoke / integration.
- **`test_map_elites.py`** — MAP-Elites archive / binning / loop behaviors.
- **`test_robustness_nominal_scene.py`** — Robustness alignment and evaluation against nominal scene.

Use **`pytest`**; floating-point checks should use **`math.isclose` / `np.allclose`** per project rules.

---

## 19. Design choices, limitations, and writing-up

1. **Gaussian channels** are a **modeling ansatz**, not a fabrication guarantee — describe as **parameterized spectral responsivity** unless tied to a specific stack model.
2. **SAM + maximin** is a **conservative discrimination surrogate** — not the same as calibrated misclassification rate under a noise model unless noise and a decision rule are added.
3. **Multi-condition `min`** is **explicit robust optimization**; **`mean`** is a softer aggregate.
4. **MAP-Elites descriptor** (sorted μs) is **low-dimensional** and **permutation-invariant in μ only** — σ does not index the archive; same μs with different σs **share a cell**.
5. **CMA-ME** ties learning to **archive improvement**; hyperparameters (`num_emitters`, `batch_size`, `initial_sigma`, `restart_patience`) trade coverage vs. peak fitness.
6. **Niching vs. SAM:** genotype distances need not match phenotype angles; output-space niching would be heavier.
7. **Performance:** matrix quadrature and batched SAM avoid repeated substance/pair loops. Blackbody and τ^r are recomputed per call because SceneConfig arrays are mutable. Any future cache needs explicit ownership or invalidation.
8. **Robustness scope:** common positive gains cancel in SAM above the norm threshold. With the nominal transmission identically one, distance-ratio sweeps do not probe wavelength-dependent atmospheric attenuation. The n² factor alone also cancels. These sweeps evaluate the configured surrogate, not measured identification performance.

**Reasonable paper angles:** simulation-informed channel design; QD archives for diverse high-fitness configs; minimax / multi-environment training vs. nominal-only; clear limits (single-pixel forward model, optional noise, no diffraction model unless added).

---

## 20. Optional extensions and known gaps

The active stack includes: **`SceneConfig`**, split **`ga/experiment.py`**, multi-scene **`MinDissimilarityFitnessEvaluator`**, SAM zero-norm guard, **`map_elites/`** with CMA-ME and polish, **`analysis/robustness`** and plots, GA visualization **`parameters_to_curves`** hook.

Still optional or open, depending on project goals:

- **Detector noise / calibration layers** — not first-class in core fitness unless added.
- **RLC or Fabry–Pérot responsivity** — an eventual alternative **`parameters_to_curves`** and gene layout. The supplied **`RLC Model/`** package is separate from the active LWI forward model and requires calibration/validation first; see **`docs/RLC_MODEL_TECHNICAL_ASSESSMENT.md`**.
- **`ga/tuning.py` boundaries** — experiment wiring lives in `experiment.py`; further splitting should address a concrete maintenance need.

Other Markdown files under **`docs/`** may include older narratives or superseded plans; **this file** is the implementation-aligned overview.
