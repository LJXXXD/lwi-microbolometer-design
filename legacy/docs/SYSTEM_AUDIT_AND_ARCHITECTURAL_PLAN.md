# LWI Microbolometer Design Optimization
## System Audit and Architectural Plan

### Scope
This audit reviewed the current production path and the nearby research tooling with emphasis on:

- `src/lwi_microbolometer_design/simulation/sensor_simulation.py`
- `src/lwi_microbolometer_design/ga/fitness.py`
- `src/lwi_microbolometer_design/analysis/*`
- `src/lwi_microbolometer_design/data/substance_atmosphere_data.py`
- `src/lwi_microbolometer_design/ga/*`
- Core entrypoint scripts such as `scripts/run_map_elites.py`, `scripts/run_ensemble_demo.py`, and `scripts/run_diversity_search.py`
- Selected `legacy/tools/scoring/*` modules to understand lineage

Notes:

- The Excel data files are not present in the repository, so exact wavelength count `d` and substance count `n` are inferred symbolically from the APIs and scripts.
- The canonical scripts target the `4 White Powders` dataset, so `n` is often 4 in practice.
- Local test execution was not possible from the current shell because `pytest` is not installed; confidence is based on static code inspection plus the existing test suite contents.

## Executive Summary
The real system is smaller and cleaner than it first appears:

1. A chromosome parameterizes a small set of spectral basis functions.
2. The physics simulation compresses each substance spectrum into a short sensor-response vector.
3. Fitness is the minimum pairwise angular separation between those response vectors.

The core physical idea is coherent. The main architectural debt is not the math itself. The main debt is plumbing:

- scene data is passed around as untyped dicts and sometimes as `dict | list[dict]`
- scripts repeatedly unpack and normalize the same scene fields by hand
- the GA fitness adapter owns too many responsibilities at once
- response-space analysis and genotype-space analysis are mixed together under `analysis/`
- several research/prototype paths have already drifted from the current runtime API

Bottom line:

- `dissimilarity_scoring.py` is directly relevant to the optimization goal
- `optimal_pairing_distance.py` is not part of the physics objective; it is a GA diversity utility
- the highest-value refactor is small and pragmatic: normalize scene data once, precompute fixed scene terms once, and make the GA fitness object a thin adapter instead of the owner of the entire physics/data schema

## 1. The Data and Physical Reality

### Explain Like I'm 5
The optimizer is trying to design a sensor with a few wavelength-sensitive "bumps."

- Each material has a true emissivity curve over wavelength.
- The scene also has temperature and air transmission.
- The sensor design says where its bumps are and how wide they are.
- For each material, the simulator asks: "How much total energy does each bump collect?"
- That produces a short fingerprint for each material.
- The GA rewards designs where even the two most similar fingerprints are still clearly separated.

So the GA is not comparing raw spectra directly. It is comparing compressed sensor fingerprints.

### What actually flows through the system

#### Main symbols

- `d`: number of wavelength samples
- `n`: number of substances
- `m`: number of basis functions / subpixels
- `p`: parameters per basis function

In the current scripts:

- `m` is usually 4
- `p` is usually 2 for Gaussian `(mu, sigma)` pairs
- chromosome length is therefore usually `m * p = 8`

#### Fixed scene data versus variable design data

| Layer | Owner today | Type | Shape | Fixed per run? | Meaning |
| --- | --- | --- | --- | --- | --- |
| `wavelengths` | loader | `np.ndarray` | `(d, 1)` from loader, often squeezed to `(d,)` later | Yes | spectral grid in micrometers |
| `substance_names` | loader | `np.ndarray[str]` | `(n,)` | Yes | human labels only |
| `emissivity_curves` | loader | `np.ndarray` | `(d, n)` | Yes | emissivity spectrum for each substance |
| `air_transmittance` | loader | `np.ndarray` | expected `(d, 1)`, not rigorously validated | Yes | atmospheric transmission over wavelength |
| `temperature_K` | loader | `float` | scalar | Yes | scene temperature |
| `atmospheric_distance_ratio` | loader | `float` | scalar | Yes | exponent applied to transmission |
| `air_refractive_index` | loader | `float` | scalar | Yes | medium refractive index |
| chromosome | GA | `np.ndarray` | `(m * p,)` | No | design variables for one candidate sensor |
| basis parameter list | fitness adapter | `list[tuple]` | length `m` | No | decoded chromosome, usually Gaussian params |
| `basis_functions` | curve factory | `np.ndarray` | `(d, m)` | No | one spectral response curve per basis function |

#### Intermediate and final signals

| Stage | Type | Shape | Physical meaning |
| --- | --- | --- | --- |
| `bb_emit` | `np.ndarray` | `(d, 1)` or `(d,)` | blackbody emission over wavelength |
| `tau_air` | `np.ndarray` | `(d, 1)` or `(d,)` | air transmission after path-length exponent |
| per-substance spectral product | `np.ndarray` | `(d, m)` | scene radiance for one substance multiplied by all basis functions |
| `sensor_outputs` | `np.ndarray` | `(m, n)` | final compressed sensor fingerprint for each substance |
| one substance fingerprint | `np.ndarray` | `(m,)` | one column of `sensor_outputs` |
| `distance_matrix` | `np.ndarray` | `(n, n)` | pairwise separation between substance fingerprints |
| fitness | `float` | scalar | worst-case pairwise separation score |

### The actual forward model
The current simulator implements this logic:

1. Generate basis functions from chromosome parameters.
2. Compute blackbody emission from wavelength, temperature, and refractive index.
3. Compute atmospheric attenuation as `air_transmittance ** atmospheric_distance_ratio`.
4. For each substance, multiply:
   - blackbody emission
   - atmospheric attenuation
   - that substance's emissivity curve
   - every basis function
5. Integrate each resulting spectrum over wavelength.

That produces `sensor_outputs` with shape `(m, n)`.

The fitness path then does this:

1. Treat each column of `sensor_outputs` as one substance fingerprint in an `m`-dimensional response space.
2. Compute pairwise distances between those columns using spectral angle mapper (SAM).
3. Return the smallest off-diagonal distance.

So the current objective is:

- maximize the worst-separated pair of substances
- in the low-dimensional sensor-response space
- using an angle-based metric that ignores overall magnitude scale

### Important physical interpretation
This is the most important top-down insight in the current codebase:

- the optimizer is not maximizing raw absorbed energy
- it is not maximizing SNR
- it is not modeling noise, detector saturation, or class priors
- it is maximizing angular separation between response fingerprints

Because SAM normalizes vectors before comparing them, a design that scales all substances up or down together will not improve the current objective. The current fitness values directional separability, not absolute response strength.

That is not inherently wrong. It is simply the actual contract of the current system, and all architectural choices should reflect that reality.

### 1D versus 2D signals in this repository

#### 1D signals

- wavelength grid
- one substance emissivity curve
- air transmittance curve
- blackbody emission curve
- one basis function curve
- one substance response fingerprint after integration

#### 2D signals

- emissivity matrix `(d, n)`
- basis-function matrix `(d, m)`
- sensor-output matrix `(m, n)`
- substance distance matrix `(n, n)`

The final signal used for fitness is not a spectrum anymore. It is the short `m`-dimensional response vector for each substance.

### Throughput and where the time really goes
Per fitness evaluation, the main work is:

1. Decode chromosome into `m` parameter groups.
2. Build basis functions over `d` wavelengths.
3. Recompute fixed scene terms:
   - blackbody emission
   - atmospheric attenuation
4. Integrate across `d` for each of `n` substances and `m` basis functions.
5. Compute an `n x n` distance matrix over `m`-length vectors.

For the current problem scale, this means:

- response-space scoring is cheap because `m` and `n` are small
- the spectral-side work over `d` is the hot path
- fixed scene terms are currently recomputed for every chromosome even though they do not change within a run

Approximate workload from current scripts:

- `scripts/run_map_elites.py`: about 201,000 fitness calls
- `scripts/run_diversity_search.py`: about 60,000 fitness calls
- `scripts/run_ensemble_demo.py`: about 5,000,000 candidate evaluations at current settings
- `src/lwi_microbolometer_design/ga/tuning.py` default search space expands to 2,916 configurations; at 5 runs each and average population/generation settings, a literal full sweep implies on the order of billions of candidate evaluations

That last point is why the correct performance refactor is not "add a fancy architecture." The correct performance refactor is "stop recomputing fixed scene physics on every chromosome."

## 2. GA-Simulation Interface Diagnosis

### Current boundary
Nominally, the optimizer boundary is clean:

- PyGAD only needs `fitness_func(ga_instance, chromosome, chromosome_idx) -> float`

In practice, the boundary leaks badly because every script that builds that callable has to know:

- the loader's dict keys
- the loader's scalar-versus-array conventions
- the naming mismatch between `temperature_K`, `temperature_k`, and `temperature_kelvin`
- the wavelength shape expectations
- which basis generator is in use
- which distance metric and score reducer define the objective

The GA runtime does not see the physics directly, but the script layer absolutely does.

### Where coupling is too tight

| Problem | Evidence | Why it matters |
| --- | --- | --- |
| Scene data clump | `load_substance_atmosphere_data()` returns a mixed `dict[str, np.ndarray | float]` or `list[dict[...]]` | every consumer must know the schema and normalize it by hand |
| Script-level evaluator reconstruction | repeated unpacking in `ga/tuning.py`, `scripts/run_map_elites.py`, `scripts/run_ensemble_demo.py`, `scripts/run_diversity_search.py`, and multiple prototypes | changes in scene schema or fitness API require edits in many places |
| Fitness object does too much | `ga/fitness.py` decodes chromosome, builds curves, runs simulation, computes distance matrix, and applies objective reduction | it mixes optimizer adaptation with domain evaluation |
| Physics invariants exposed to GA-adjacent code | scripts manually pass `wavelengths`, `emissivity_curves`, `temperature_k`, `air_transmittance`, etc. into evaluator construction | the optimizer layer knows more about scene physics than it should |
| Single-scene and multi-scene are unresolved | loader supports multiple conditions, but most consumers immediately select the first element if a list is returned | the public API suggests a capability the optimization path does not truly support |

### Where abstractions are leaking

#### Naming drift

- loader returns `temperature_K`
- fitness evaluator expects `temperature_k`
- experiment configs use `temperature_kelvin`

This is small but important. It proves there is no single internal runtime schema.

#### Shape drift

- `simulate_sensor_output()` documents wavelength input as `(d, 1)`
- the fitness evaluator squeezes it to `(d,)`
- curve factories operate on 1D arrays

The current code handles this informally by reshaping and squeezing in multiple places. It works, but the shape contract is not centralized.

#### Prototype drift
`scripts/prototypes/demo_ga_comprehensive.py` already shows API drift from the supported path by using `temperature_K=` and a mismatched data key. That is strong evidence that the boundary is not stable enough to survive normal research iteration.

#### Analysis distance mismatch
The GA's niching logic can use optimal-pairing distance in gene space, but several production callbacks and tuning trackers compute population diversity with plain Euclidean defaults instead of the active `niching_config`. The result is that reporting and selection are not always talking about the same notion of distance.

### What the GA should know
The GA should know:

- chromosome length and gene space
- how to mutate/crossover/select
- optionally, how to measure chromosome distance for niching

### What the GA should not know
The GA should not know:

- wavelength grids
- atmospheric transmittance arrays
- emissivity matrices
- scene temperature and refractive index
- spectral integration details

Those belong to a scene/objective layer, not to scripts that happen to launch a GA.

### The main hot-path design issue
`simulate_sensor_output()` is mathematically the physics kernel, but its current signature requires all fixed scene terms every time:

- `wavelengths`
- `temperature_k`
- `atmospheric_distance_ratio`
- `air_refractive_index`
- `air_transmittance`
- `substances_emissivity`

Only `basis_functions` actually changes per chromosome. This is the clearest sign that the current interface is shaped around convenience, not around the real runtime boundary.

## 3. Analysis Package and SENS-D Legacy Assessment

### What the current `analysis/` directory really contains

| File | Role today | Hot path? | Strategic assessment |
| --- | --- | --- | --- |
| `analysis/distance_metrics.py` | SAM response-space distance metric | Yes | core objective support |
| `analysis/distance_matrix.py` | generic pairwise matrix builder | Yes | useful core utility, but more generic than current hot path needs |
| `analysis/dissimilarity_scoring.py` | scalar reducers over distance matrices | Yes, partially | `min_based` is core; other scorers are exploratory |
| `analysis/optimal_pairing_distance.py` | order-insensitive grouped-parameter distance using Hungarian matching | No for fitness | useful, but belongs conceptually to GA diversity/niching |
| `analysis/vat.py` | VAT/iVAT visualization transforms | No | diagnostic and reporting only |

### Lineage from legacy code
The modern `analysis` package is clearly descended from:

- `legacy/tools/scoring/spectral_angle_mapper.py`
- `legacy/tools/scoring/distance_matrix.py`
- `legacy/tools/scoring/distance_matrix_evaluation.py`

This is not a bad thing. In fact, it explains why the core scoring logic is compact and understandable.

What matters is that the current package still carries the shape of a generic scoring toolbox, while the production optimization path only depends on a narrow subset of it.

### `dissimilarity_scoring.py`: relevance to the fitness function
This module is highly relevant, but only one part of it is currently essential.

#### Directly relevant

- `min_based_dissimilarity_score()` is the current objective reducer used by `ga/fitness.py`

This is the actual fitness scalar. It answers:

"What is the closest pair of substances in the final sensor-response space, and how large is that separation?"

That makes it central to the optimization goal.

#### Potentially relevant, but not active today

- `group_based_dissimilarity_score()` would matter if the product requirement were group-level discrimination rather than all-pairs discrimination
- `mean_min_based_dissimilarity_score()` and `weighted_mean_min_dissimilarity_score()` are reasonable alternative objectives, but they are not wired into the current runtime path and are not exported from the package root

Conclusion:

- keep `dissimilarity_scoring.py`
- treat `min_based_dissimilarity_score()` as part of the production objective path
- treat the other scorers as experimental until there is a real product need for them

### `optimal_pairing_distance.py`: relevance to the fitness function
This module is useful, but it does not serve the physics objective directly.

What it actually does:

- compares grouped chromosome parameters such as `(mu, sigma)` pairs
- ignores ordering of those groups by using Hungarian matching
- supports GA niching, diversity analysis, and clustering of candidate designs

What it does not do:

- it does not compare simulated substance fingerprints
- it does not appear in `ga/fitness.py`
- it does not influence the current physics-based score returned to the optimizer

So in the context of the fitness function, `optimal_pairing_distance.py` is orthogonal.

This is a strong architectural clue: it should live next to GA diversity logic, not next to response-space scoring logic.

One practical note: it returns the sum of matched distances, so its scale grows with the number of basis-function groups. That is fine within one run where group count is fixed, but it makes `sigma_share` less portable across experiments with different sensor sizes.

### VAT and other analysis utilities
`analysis/vat.py` and the IVAT plot path in `ga/visualization.py` are useful for research diagnostics and paper-quality explanation. They are not part of the optimization contract.

For a production-ready architecture, these tools should be considered reporting utilities, not core pipeline dependencies.

### Net assessment of the `analysis/` directory
The directory currently contains three conceptually different things:

1. response-space objective support
2. genotype-space diversity support
3. visualization-only diagnostics

That is the real reason it feels like "legacy SENS-D" tooling: it is not one cohesive responsibility.

## 4. Strategic Pythonic Refactoring Plan

### Design goals
The refactor should optimize for:

- rigor: one internal runtime schema for scene data
- explainability: the data flow should be teachable in one whiteboard session
- cohesion: physics, objective, and optimizer adapters should each have one job
- practicality: optimize the hot path that actually dominates runtime
- restraint: do not build a framework for imagined future basis families or optimizers

### Recommended minimal structure
Do not replace the whole package layout. The existing top-level split is workable:

- `data/`
- `simulation/`
- `analysis/`
- `ga/`
- `visualization/`

The recommended change is to add one small typed runtime layer and tighten the boundaries around it.

### Recommended DTOs and dataclasses

| Object | Purpose | Suggested fields | Suggested config |
| --- | --- | --- | --- |
| `SceneData` | normalized runtime scene object returned by the loader | `wavelengths`, `substance_names`, `emissivity_curves`, `air_transmittance`, `temperature_k`, `atmospheric_distance_ratio`, `air_refractive_index` | `@dataclass(frozen=True, slots=True, kw_only=True)` |
| `PrecomputedScene` | hot-path cached physics invariants | `wavelengths`, `substance_names`, `scene_radiance` or at least `bb_emit` and `tau_air` | `@dataclass(frozen=True, slots=True, kw_only=True)` |
| `ChromosomeEncoding` | the design-side schema | `params_per_basis_function`, `parameters_to_curves`, optionally `gene_space` | small dataclass or plain callable bundle |
| `GARunSummary` | honest output DTO for reporting and plots | `best_chromosome`, `best_fitness`, `final_population`, `final_fitness_scores`, optional tracked histories | dataclass preferred over ad-hoc dicts if the reporting surface is meant to be stable |

Important detail:

- large arrays should be normalized and validated in `__post_init__`
- `wavelengths` and `air_transmittance` should become one canonical 1D representation internally
- `repr` for large arrays should be kept concise

### Recommended runtime layers

#### 1. Data normalization layer
Responsibility:

- load Excel
- normalize names and shapes
- return `SceneData`

This should be the only place that knows about:

- `temperature_kelvin`
- `temperature_K`
- pandas Excel conventions
- whether transmittance came in as `(d, 1)` or `(d,)`

#### 2. Physics layer
Responsibility:

- convert `SceneData` into `PrecomputedScene`
- accept `basis_functions`
- return `sensor_outputs`

The key refactor here is to compute fixed scene terms once:

- blackbody emission
- atmospheric attenuation
- optionally the full per-substance scene radiance before basis-function multiplication

#### 3. Objective layer
Responsibility:

- compare columns of `sensor_outputs`
- compute pairwise distances
- reduce them to one score

This layer should own:

- `spectral_angle_mapper`
- `compute_distance_matrix` for response vectors
- `min_based_dissimilarity_score`

#### 4. Optimizer adapter layer
Responsibility:

- decode chromosome
- build basis functions
- call the objective
- expose a pickleable GA-compatible `fitness_func`

This is where `ga/fitness.py` should live conceptually.

### Dependency injection, but only where it earns its keep
The codebase already hints at the right level of dependency injection:

- `parameters_to_curves` is injected
- `distance_metric` is injected

That is good and should continue.

Recommended next step:

- inject the score reducer as well
- optionally inject a preconfigured simulator object or callable

What not to do:

- do not add abstract base classes for every layer
- do not add a service container
- do not add a plugin registry

Plain callables and a few dataclasses are enough for this codebase.

### Concrete package-boundary changes

#### Keep

- `simulation/blackbody.py`
- `simulation/sensor_simulation.py` as the physics kernel
- `analysis/dissimilarity_scoring.py` as response-space objective support
- `analysis/distance_metrics.py`

#### Move or rename

- move `analysis/optimal_pairing_distance.py` next to GA diversity code, or re-export it there and treat the `analysis` location as compatibility-only
- rename `ga/analysis.py` to something like `ga/population_analysis.py` to stop colliding conceptually with `analysis/`

#### Centralize

- one supported builder that turns `SceneData` plus encoding choices into a GA evaluator

Today that logic is duplicated across scripts. It should exist once.

### The multi-condition decision that should be made explicitly
Right now the loader supports multiple scene conditions, but most optimization paths immediately do:

- "if this is a list, use the first item"

That is the worst of both worlds. The system advertises multi-condition support without architecting around it.

Choose one of these two models explicitly:

1. Core optimization is single-scene only.
   - loader returns exactly one `SceneData`
   - multi-condition sweeps happen outside the optimizer

2. Core optimization is aggregated across scenes.
   - evaluator accepts `list[SceneData]`
   - score aggregation across scenes is explicit, such as `min`, `mean`, or weighted mean

For this codebase, the pragmatic recommendation is model 1 unless there is an immediate product requirement for multi-condition optimization.

### Ordered migration plan

| Priority | Change | Why it is worth doing now |
| --- | --- | --- |
| P0 | Introduce `SceneData` and central shape validation | removes data clumps and schema drift immediately |
| P0 | Centralize evaluator creation into one supported builder | stops script-level unpacking from spreading further |
| P0 | Precompute fixed scene terms once per run | biggest real performance win for GA, ensemble, MAP-Elites, and tuning |
| P0 | Decide single-scene versus multi-scene behavior explicitly | removes the current `dict | list[dict]` ambiguity |
| P1 | Move `optimal_pairing_distance` next to GA diversity code | fixes conceptual package boundaries |
| P1 | Rename `ga/analysis.py` to a population-specific name | reduces confusion between spectral analysis and GA analytics |
| P1 | Make reporting histories honest and consistent | avoids misleading plots and metrics |
| P1 | Pass active niching distance config consistently into diversity analytics | aligns reporting with runtime behavior |
| P2 | Prune or clearly label experimental scorers and prototype-only paths | reduces cognitive load for future maintainers |

## 5. What Not To Over-Engineer
This repository does not need:

- a full domain-driven design layer
- Pydantic models for internal hot-path objects
- xarray for current data volume
- a generic plugin architecture for arbitrary basis-function families
- a rewrite away from PyGAD
- distributed compute infrastructure before the per-evaluation hot path is cleaned up

The actual data volume is small in memory. The bottleneck is repeated CPU work across many evaluations, not large persistent datasets or complex orchestration.

The right amount of architecture here is:

- one stable scene DTO
- one cached scene representation for the hot path
- one thin GA adapter
- a clear separation between response-space objective logic and genotype-space diversity logic

## 6. Production-Readiness Observations Outside the Core Refactor
These are not the main architectural story, but they matter:

- There are debug `print()` statements still in hot-path GA code such as `ga/advanced_ga.py` and `ga/mutations.py`.
- The current tests cover isolated simulation math, basic analysis utilities, and a toy GA scenario, but not a full loader-to-fitness integration path with real scene data.
- Some reporting tools assume histories that may only be approximations of final-generation values.
- Several prototype scripts appear to be research artifacts rather than stable entrypoints; they should not define the public runtime contract.

## Final Recommendation
The production architecture should be centered on one sentence:

The optimizer should only hand a chromosome to a small adapter; everything after that should operate on typed scene and response objects, not on ad-hoc dict schemas.

If you do only three things, do these:

1. Introduce a canonical `SceneData` object and stop passing raw scene dicts around.
2. Precompute fixed scene physics once per run and keep only design-dependent work in the chromosome loop.
3. Separate response-space objective utilities from genotype-space diversity utilities so `analysis/` stops carrying unrelated concerns.

That plan is rigorous, easy to explain, tightly grounded in the actual physics/data flow, and avoids speculative over-engineering.
