# LWI Microbolometer Design — Project Master Reference

**Status:** Canonical onboarding, implementation, and paper-drafting reference.
**Scope:** Reflects the **current** `src/lwi_microbolometer_design/` and `scripts/` tree as implemented (not legacy audit TODOs).

This document is the single place to answer: *what problem we solve, what tensors mean, where to plug in a new basis model, and how GA / MAP-Elites / CMA-ME relate.*

---

## Table of contents

1. [Problem and physics foundation](#i-problem-and-physics-foundation)
2. [Data contract and shapes](#ii-data-contract-and-shapes)
3. [Modular deep dive](#iii-modular-deep-dive)
4. [Execution flow](#iv-execution-flow-one-chromosome-end-to-end)
5. [Research and strategic context](#v-research-and-strategic-context)

---

## I. Problem and physics foundation

### I.1 Multispectral material discrimination (what we optimize)

We treat the sensor as a **single-pixel multispectral** device: `m` tunable spectral channels (basis functions \(\Phi_i(\lambda)\)) integrate radiance over wavelength into an **`m`-dimensional fingerprint** per target substance. Given `n` substances with known emissivity \(\varepsilon_j(\lambda)\) on a fixed grid, the design problem is to choose channel shapes/positions so that **fingerprints are as separable as possible** in a chosen metric.

The codebase does **not** optimize spatial imaging, readout noise budgets, or classifier architectures in the core loop; it optimizes **spectral responsivity synthesis** under a forward radiance model.

### I.2 Forward model (implemented)

For substance index \(j\) and channel \(i\), the simulation implements:

\[
S_{i,j} = \int \tau_{\mathrm{air}}(\lambda)^{r}\, B(\lambda, T, n)\, \varepsilon_j(\lambda)\, \Phi_i(\lambda)\, d\lambda
\]

| Symbol | Code / meaning |
|--------|----------------|
| \(\lambda\) | Wavelength (µm), discrete grid length `d` |
| \(B(\lambda, T, n)\) | Planck spectral radiance via `blackbody_emit` |
| \(\tau_{\mathrm{air}}(\lambda)\) | Loaded atmospheric transmittance; raised to power `atmospheric_distance_ratio` (`r`) |
| \(\varepsilon_j(\lambda)\) | Column `j` of `emissivity_curves` |
| \(\Phi_i(\lambda)\) | Column `i` of `basis_functions` (design variables) |

**Implementation:** `simulate_sensor_output` in `simulation/sensor_simulation.py` uses `np.trapezoid` along wavelength. Internally it reshapes 1D wavelength and transmittance to `(d, 1)` for broadcasting with `(d, m)` basis functions and `(d, 1)` per-substance emissivity.

### I.3 Planck’s law (as coded)

`blackbody.blackbody_emit` evaluates (in a medium with refractive index \(n\)):

\[
B(\lambda, T) = n^2 \cdot \frac{2\pi h c^2}{\pi \lambda^5} \cdot \frac{1}{\exp(hc/(\lambda k T)) - 1}
\]

with \(\lambda\) in µm and constants as in the module docstring. The **\(n^2\)** factor is the medium density-of-states scaling (small for air, but kept for physical consistency).

### I.4 Atmospheric attenuation (as coded)

Effective transmission is **not** a separate Beer's-law path integral in code; it is the loaded curve raised to a scalar exponent:

\[
\tau_{\mathrm{eff}}(\lambda) = \tau_{\mathrm{loaded}}(\lambda)^{\,r}
\]

where `r` is `atmospheric_distance_ratio`. This is a **phenomenological** handle on path length / visibility; it is not a full MODTRAN line-by-line model.

### I.5 SAM and why “angle,” not “absolute energy”

**Spectral Angle Mapper (SAM):** for two fingerprint vectors \(\mathbf{u}, \mathbf{v} \in \mathbb{R}^m\),

\[
\theta(\mathbf{u}, \mathbf{v}) = \arccos\left( \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \, \|\mathbf{v}\|} \right)
\]

returned in **degrees** by `analysis.distance_metrics.spectral_angle_mapper`.

**Why angles:** The objective cares about **relative response pattern across channels** (spectral shape in channel space), not total integrated power. Scaling all channels by a common positive factor does not change SAM. That matches a common remote-sensing interpretation: discriminability from **direction** in feature space.

**Degeneracy guard:** If either norm is below `1e-15`, SAM returns `0.0` (treat as maximally similar). This prevents `NaN` fitness when a design yields a null fingerprint.

### I.6 Scalar fitness: maximin over pairs

Given pairwise SAM matrix of shape `(n, n)`, production fitness uses `min_based_dissimilarity_score`: the **minimum off-diagonal** entry. The GA **maximizes** this value — a **maximin** (worst-case pair) objective: push apart the hardest-to-separate substance pair in **sensor-output space**.

---

## II. Data contract and shapes

Treat this section as the **nervous system** of the project: any new feature must preserve these conventions or explicitly extend them.

### II.1 Dimension glossary

| Symbol | Meaning |
|--------|---------|
| `d` | Spectral grid length (# wavelength samples) |
| `n` | Number of substances |
| `m` | Number of basis functions / channels |
| `p` | Genes per basis function (e.g. 2 for Gaussian \((\mu, \sigma)\)) |

Typical demo setups: `n = 4`, `m = 4`, `p = 2` → chromosome length `m·p = 8`.

### II.2 `SceneConfig` (immutable DTO)

**Module:** `data.scene_config.SceneConfig`
**Pattern:** Frozen, slotted dataclass — **the canonical carrier** for everything the forward model needs except the sensor basis.

| Field | Canonical shape / type | Role |
|-------|-------------------------|------|
| `wavelengths` | `(d,)`, `float64` | µm; column vectors squeezed to 1D in `__post_init__` |
| `emissivity_curves` | `(d, n)` | \(\varepsilon\) per substance |
| `air_transmittance` | `(d,)` | Base \(\tau(\lambda)\); `(d,1)` squeezed; `(d,k)` uses **first column** |
| `temperature_k` | `float` | Scene temperature |
| `atmospheric_distance_ratio` | `float` | Exponent `r` |
| `air_refractive_index` | `float` | Planck medium index |
| `substance_names` | `(n,)`, `object` | Labels; must match `n` columns of emissivity |

**Hard constraints (fail fast):** `__post_init__` raises `ValueError` if `d` mismatches emissivity/transmittance or if `len(substance_names) != n`. This is the **central gate** that prevents silent `(d, n)` vs `(d,)` bugs.

**Loader:** `data.substance_atmosphere_data.load_substance_atmosphere_data` returns **`SceneConfig`** for scalar env args, or **`list[SceneConfig]`** when any of `temperature_kelvin`, `atmospheric_distance_ratio`, or `air_refractive_index` is a multi-value sequence — **Cartesian product** of conditions.

### II.3 Chromosome and basis tensors

| Object | Shape | Notes |
|--------|-------|------|
| Chromosome | `(m·p,)` | Flat gene vector |
| `basis_params` | `m` tuples of length `p` | Parsed in `MinDissimilarityFitnessEvaluator.fitness_func` |
| `basis_functions` | `(d, m)` | Columns are \(\Phi_i\) on the grid; from `parameters_to_curves` |
| `sensor_outputs` | `(m, n)` | Column `j` = substance `j` fingerprint |
| SAM distance matrix | `(n, n)` | Symmetric; diagonal ignored for min score |
| Fitness | `float` | Degrees (SAM) after `min_based_dissimilarity_score` |

**Axis convention for SAM:** `compute_distance_matrix(..., axis=1)` treats **columns** of `(m, n)` as items — consistent with “column = substance.”

### II.4 Multi-condition fitness (robust evaluation)

**Class:** `ga.fitness.MinDissimilarityFitnessEvaluator`

- **Constructor:** `scene` may be one `SceneConfig` or a **sequence** of scenes.
- **`aggregation`:** `"single"` | `"min"` | `"mean"` (see `SUPPORTED_AGGREGATIONS`).
- **Implicit behavior:** If you pass **multiple** scenes with `aggregation="single"`, the code **logs a warning** and **defaults to `"min"`** (worst-case / minimax across conditions).

This is the implemented bridge from “single nominal scene” to **explicit multi-scene** objectives without ad-hoc dict plumbing.

### II.5 Hidden assumptions worth calling out

1. **Gaussian gene layout in MAP-Elites helpers:** `map_elites.archive.extract_features` assumes **even indices** of the chromosome are \(\mu\) (wavelength centers). Odd indices are not used for behavior descriptors. If you change `p` or encoding, **replace feature extraction** accordingly.
2. **`simulate_sensor_output` docstring** still mentions `(d, 1)` for wavelengths; **runtime accepts 1D** and reshapes internally. Authoritative 1D contract is **`SceneConfig`**.
3. **Niching distance vs fitness metric:** Niching uses **genotype-space** distances (Euclidean or optimal-pairing on genes), while fitness uses **SAM in output space**. They are deliberately different objects (see Section V).

---

## III. Modular deep dive

### III.1 Standard modules (black-box API)

| Module | Responsibility | Inputs → outputs |
|--------|----------------|------------------|
| `data.scene_config` | Typed scene DTO | Fields as in §II.2 |
| `data.substance_atmosphere_data` | Excel → `SceneConfig` / list | Paths + scalars or sequences → `SceneConfig \| list[SceneConfig]` |
| `simulation.blackbody` | Planck \(B(\lambda,T,n)\) | `spectra` (µm), `temperature_k`, `refractive_index` → radiance array |
| `simulation.gaussian_parameter_to_curves` | Gaussian basis factory | `list[tuple[float,float]]`, `wavelengths (d,)` → `(d, m)` unit peaks |
| `simulation.sensor_simulation` | Forward integration | wavelengths, `(d,n)` emissivity, `(d,m)` bases, scalars, `(d,)` τ → `(m,n)` |
| `analysis.distance_metrics` | SAM | two length-`m` vectors → degrees |
| `analysis.distance_matrix` | Pairwise distances | arrays or lists; optional **optimal pairing** mode |
| `analysis.dissimilarity_scoring` | Matrix → scalar | `min_based_dissimilarity_score` (production); alternates for experiments |
| `analysis.optimal_pairing_distance` | Hungarian pairing cost | two equal-length group lists → scalar sum of matched distances |
| `analysis.robustness` | Post-hoc multi-condition testing | elites + scenes + optional noise → `RobustnessResult` / archives |
| `ga.fitness` | **Strategy-assembled** evaluator | `SceneConfig`(s), `parameters_to_curves`, `p`, optional `distance_metric`, `aggregation` |
| `ga.advanced_ga` | PyGAD + fitness sharing | `AdvancedGA`, `NichingConfig` |
| `ga.diversity` | Population distances for niching / reporting | population `(N, genes)` + `NichingConfig` → `(N,N)` |
| `ga.experiment` | YAML wiring | experiment files → evaluator, gene space, search space |
| `ga.tuning` | Hyperparameter sweeps | `HyperparameterTuner`, `run_single_configuration`, etc. |
| `map_elites.*` | QD archive + algorithms | See §III.3 |
| `visualization/*` | Plots | Distance matrices, sensor outputs, robustness figures |

### III.2 Design patterns (explicit)

- **Strategy (basis functions):** `parameters_to_curves: Callable[[list[tuple], np.ndarray], np.ndarray]` injects how genes become `(d, m)`. **This is the correct extension point** for RLC / Fabry–Pérot / measured filters: implement a function with that signature and pass it to `MinDissimilarityFitnessEvaluator`.
- **Strategy (metric):** `distance_metric` defaults to SAM but remains injectable.
- **Serializable evaluator:** `MinDissimilarityFitnessEvaluator` is a **class** (not a closure) so `fitness_func` can be pickled for `multiprocessing`.
- **DTO boundary:** `SceneConfig` replaces the former untyped dict clump at the loader boundary.

### III.3 AdvancedGA niching (implicit logic)

**Class:** `ga.advanced_ga.AdvancedGA` (extends `pygad.GA`).
**Config:** `NichingConfig` — note **`enabled` and `use_optimal_pairing` are mandatory** (no silent default pairing mode).

**Per generation when niching is on:**

1. Cache **raw** fitness in `original_fitness_scores`.
2. Build pairwise population distance matrix via `ga.diversity.compute_population_distance_matrix` (respects `NichingConfig`).
3. For each individual \(i\), niche count
   \(\mathrm{nc}_i = 1 + \sum_{j \neq i} \mathrm{sh}(d_{ij})\)
   with \(\mathrm{sh}(d) = 1 - (d/\sigma_{\mathrm{share}})^\alpha\) if \(d < \sigma_{\mathrm{share}}\), else `0`.
4. **Shared fitness** \(f'_i = f_i / \mathrm{nc}_i\).
5. Temporarily replace `last_generation_fitness` with shared values, call parent selection, then **restore raw fitness** so elitism uses true objective values.

**Why optimal pairing:** Chromosomes encode unordered channels as \([(\mu_1,\sigma_1), \ldots, (\mu_m,\sigma_m)]\) but the flat gene order is arbitrary. **Permuting whole groups** should not make individuals “far apart” in niching distance. With `use_optimal_pairing=True`, `diversity` routes to `compute_distance_matrix(..., use_optimal_pairing=True, params_per_group=p)`.

**Optimal pairing mechanics:** For each chromosome pair, genes are reshaped to `(m, p)` row-vectors. A **cost matrix** of pairwise distances between groups from individual A and B is built (`scipy.spatial.distance.cdist` in the vectorized path). **`scipy.optimize.linear_sum_assignment`** (Hungarian algorithm) finds the minimum-sum matching; the distance is the **sum** of matched edge costs.

**Complexity note:** Niching builds an `N×N` distance matrix each generation; optimal pairing uses \(O(N^2 \cdot m^3)\)-scale work from repeated Hungarian solves — acceptable at modest population sizes, painful at very large `sol_per_pop`.

### III.4 MAP-Elites: archive, emitter, CMA-ME

**Package:** `map_elites/` (first-class module, not only scripts).

**Behavior descriptor (current default):** `extract_features` sorts all \(\mu\) from **even gene indices** and takes the **smallest and second-smallest** → 2D binning \((\mu_{(1)}, \mu_{(2)})\). **`reachable_cell_count`** enforces that only **`x_bin ≤ y_bin`** cells exist (upper triangle of an `L×L` grid) — **do not assume a full `L²` rectangular archive is usable**.

**MAP-Elites loop (`run_map_elites`):** Emitter = **parent uniform from current archive** + `mutate_chromosome` (Gaussian perturbation; **even indices** use larger relative step than odd). Child is evaluated; if its bin is empty or fitness improves, replace occupant. This is **quality–diversity** competition **within bins**, not global single-objective collapse.

**CMA-ME (`run_cma_me`):** Multiple `OptimizingEmitter` instances (CMA-ES in **unit-cube** normalized gene space via `map_elites.normalization.UnitCubeScaler`). Each batch:

1. **Ask** candidates from CMA-ES.
2. Evaluate fitness and **archive improvement** per candidate: new cell = improvement `fitness`; better in existing cell = marginal gain; else `0`.
3. **Tell:** CMA-ES ranking uses **improvement-first objectives** (negated for minimization API); if no improvement in the batch, falls back to raw fitness ranking.
4. On convergence / stall, **restart** emitter from a random archive elite.

**Implicit lesson:** CMA-ME is not “CMA-ES with a fitness wrapper”; the **emitter’s learning signal is tied to archive progress**, aligning search with **both** quality and coverage in descriptor space.

**Polish:** `map_elites.polish` provides **hill-climbing** and **CMA** local polish utilities for elites after illumination.

---

## IV. Execution flow: one chromosome end-to-end

**Goal:** Trace **genotype → phenotype → fitness** for the default Gaussian sensor.

1. **Genotype:** `chromosome` of length `m·p` (e.g. 8 floats).
2. **Parse:** Chunk into `m` tuples of length `p` (`MinDissimilarityFitnessEvaluator.fitness_func`).
3. **Phenotype (spectral):** `parameters_to_curves(basis_params, scene.wavelengths)` → `(d, m)`; default `gaussian_parameters_to_unit_amplitude_curves` snaps each \(\mu\) to nearest grid point so peak = 1.
4. **Physics:** `simulate_sensor_output` with fields from `SceneConfig` → `(m, n)`.
5. **Dissimilarity:** `compute_distance_matrix(sensor_outputs, distance_func=SAM, axis=1)` → `(n, n)`.
6. **Scalarize:** `min_based_dissimilarity_score(distance_matrix=...)` → **degrees**, higher is better.
7. **Multi-scene:** repeat step 4–6 per scene; aggregate with `min` or `mean` per constructor.

**Where to add a new `parameters_to_curves`:**

1. Implement `fn(basis_params: list[tuple], wavelengths: np.ndarray) -> np.ndarray` with output `(d, m)`.
2. Pass `fn` and correct `params_per_basis_function` into `MinDissimilarityFitnessEvaluator`.
3. Update **gene_space** / chromosome length in scripts (`m * p`).
4. If using MAP-Elites / CMA-ME, update **`extract_features`** and bin ranges if the behavior descriptor should no longer be “two smallest \(\mu\)”.

---

## V. Research and strategic context

### V.1 Current modeling limitations (honest scope)

- **Basis family:** Production path assumes **parametric Gaussian channels** unless you inject another `parameters_to_curves`. Real microbolometer stacks (RLC, resonant cavities, coupling) are **not** in the forward model until you replace that callback.
- **Atmosphere:** Single transmittance curve + scalar exponent — not full radiative transfer or aerosol models.
- **Detector physics:** No NETD, 1/f noise, drift, nonlinearity, or saturation in the core `simulate_sensor_output` → SAM pipeline.
- **Noise in optimization:** Core fitness is **deterministic** given `SceneConfig`. The **`analysis.robustness`** path supports evaluating elites under grids of conditions and optional noise — suitable for **Phase 1** degradation studies; it is not automatically the objective unless you wire a stochastic evaluator.
- **Objective semantics:** SAM + maximin encodes **geometric separability of noiseless fingerprints**, not Bayes-optimal classification under priors or cost asymmetry.
- **Descriptor / fitness mismatch in QD:** MAP-Elites bins use **sorted \(\mu\)** — permutation-invariant in centers only — while raw chromosomes can still permute \(\sigma\) with centers; the descriptor is a **deliberate choice**, not a theorem of optimality.

### V.2 Implemented directions (useful for “related work / methods”)

- **Quality–diversity:** `map_elites.run_map_elites`, `run_cma_me`, polish utilities, visualization.
- **Robust analysis:** `evaluate_*_robustness`, `summarise_robustness`, `ConditionLabel`, and plotting helpers exported from the package root.
- **Multi-scene fitness:** `MinDissimilarityFitnessEvaluator` with `aggregation="min"` or `"mean"`.
- **Presentation / experiment scripts:** `scripts/presentation_2026/` (standard GA, niching, multi-start, MAP-Elites, HC polish, CMA-ME, robustness, minimax); `scripts/map_elites/`, `scripts/ga/`, `scripts/robustness/`.

### V.3 Roadmap themes (future work for papers)

1. **Environmental robustness:** Treat `list[SceneConfig]` as first-class training distribution; compare nominal vs minimax (`aggregation="min"`) vs noisy Monte Carlo objectives; report retention curves (infrastructure exists in `robustness` + visualizations).
2. **Noise-aware objectives:** Inject noise **after** `(m, n)` (or inside a wrapper) and optimize expectation / CVaR of min-SAM.
3. **Physics-rich responsivity:** Replace Gaussian `parameters_to_curves` with **RLC / coupled-mode** models when available; revisit MAP-Elites descriptors (they should track **physically meaningful** behavior coordinates).
4. **Phenotype-aligned diversity:** Optional niching distance in **output space** (e.g. SAM between full `(m, n)` stacks flattened per individual) to align diversity with the SAM objective — trades memory for consistency.
5. **Surrogate-assisted search:** Cheap regressors on `(chromosome → fitness)` or `(chromosome → sensor_outputs)` for MAP-Elites / CMA-ME budget extension — not required at current `d, m, n` scales but relevant if costs grow.
6. **Experimental closure:** Domain randomization + calibration layers + measured filter / stack spectra to reduce simulation–reality gap; position as **simulation-informed design** rather than certified field performance.

---

## Quick reference: file map

| Concern | Primary location |
|---------|------------------|
| Scene typing | `data/scene_config.py`, `data/substance_atmosphere_data.py` |
| Forward model | `simulation/sensor_simulation.py`, `simulation/blackbody.py` |
| Fitness wiring | `ga/fitness.py` |
| GA + niching | `ga/advanced_ga.py`, `ga/diversity.py`, `ga/mutations.py`, `ga/ga_configuration.py` |
| MAP-Elites / CMA-ME | `map_elites/archive.py`, `algorithm.py`, `cma_me.py`, `emitters.py`, `polish.py` |
| SAM / distances / scoring | `analysis/distance_metrics.py`, `distance_matrix.py`, `dissimilarity_scoring.py` |
| Robustness testing | `analysis/robustness.py`, `visualization/robustness_visualization.py` |
| Public exports | `lwi_microbolometer_design/__init__.py` |

---

*End of master reference.*
