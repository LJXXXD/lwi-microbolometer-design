# Experiment pipeline and reproducibility — v2_paper_2026 suite

This document is the **companion** to **`THEORY_AND_IMPLEMENTATION.md`**. That file explains **what** the codebase optimizes and **why** the stack is shaped as it is (physics, tensors, module boundaries). **Here** we document **how to run and reproduce** the numbered **`scripts/v2_paper_2026/`** pipeline: shared configuration, evaluation budgets, **dataflow between steps**, pickle artifacts, environment overrides, and **algorithm-level mechanics** that are too operational for the blueprint.

**Quick commands and CPU waves** (copy-paste `uv run` lines, minimal env table) remain in **`scripts/v2_paper_2026/README.md`**. This file **extends** that README with narrative, a **complete** environment-variable catalog, and deep enough algorithm detail to re-derive behavior months later.

---

## Table of contents

1. [Purpose and reading order](#1-purpose-and-reading-order)
2. [Reproducibility baseline](#2-reproducibility-baseline)
3. [Shared experiment SSOT](#3-shared-experiment-ssot)
4. [Budget and override SSOT](#4-budget-and-override-ssot)
5. [Output layout and artifact contract](#5-output-layout-and-artifact-contract)
6. [Pipeline graph (dataflow)](#6-pipeline-graph-dataflow)
7. [Per-step narrative: `run_01` … `run_08`](#7-per-step-narrative-run_01--run_08)
8. [Algorithm annex](#8-algorithm-annex)
   - [8.1 Optimal pairing (Hungarian / linear sum assignment)](#81-optimal-pairing-hungarian--linear-sum-assignment)
   - [8.2 Vanilla MAP-Elites](#82-vanilla-map-elites)
   - [8.3 CMA-ME: archive improvement and `OptimizingEmitter.tell`](#83-cma-me-archive-improvement-and-optimizingemittertell)
   - [8.4 CMA-ME: `run_cma_me` orchestration](#84-cma-me-run_cma_me-orchestration)
   - [8.5 Hill-climbing polish](#85-hill-climbing-polish)
9. [Alternate entry points (`scripts/map_elites`, `scripts/ga`, `scripts/robustness`)](#9-alternate-entry-points-scriptsmap_elites-scriptsga-scriptsrobustness)
10. [Repro checklist](#10-repro-checklist)

---

## 1. Purpose and reading order

| Document | Use it when you need… |
|----------|------------------------|
| **`docs/THEORY_AND_IMPLEMENTATION.md`** | Physical model, SAM/maximin fitness, tensor shapes, package map, **high-level** algorithm behavior. |
| **`docs/01_EXPERIMENT_PIPELINE.md` (this file)** | **Which script produces which file**, env vars, seeds, evaluation counts, and **step-by-step** logic for pairing / MAP-Elites / CMA-ME. |
| **`scripts/v2_paper_2026/README.md`** | **Immediate execution**: `uv run python …`, CPU scheduling waves, short env table. |
| **`scripts/v2_paper_2026/_budgets.py`** | **Authoritative numeric defaults**; every production default is defined here (often overridable by environment variables). |

---

## 2. Reproducibility baseline

- **Working directory:** Repository **root** (parent of `scripts/`). Helpers resolve paths from `Path(__file__)`; do not rely on changing cwd into `scripts/`.
- **Tooling:** Prefer **`uv run python <script>`** so the environment matches **`pyproject.toml`** (Python ≥ 3.12, editable install of `lwi_microbolometer_design`).
- **Multiprocessing:** Presentation GA and pool steps call **`multiprocessing.set_start_method("spawn", force=True)`** for macOS/Windows-safe workers.
- **Seeds (nominal presentation wiring):**
  - Steps **01**, **02**, **04**, **05**, **06**, **08**: **`random_seed=42`** where the driver passes it; NumPy is seeded inside library code where applicable.
  - Step **03** multi-start: base seed **`42`**, per-run **`random_seed = 42 + run_id`** inside each worker payload.
  - **CMA-ME emitters:** emitter **`i`** uses **`seed=random_seed + i + 1`** in **`OptimizingEmitter`** (see **`map_elites/cma_me.py`**).

GA **generations × population** is a budget proxy, not an exact callback count: PyGAD can reuse fitness for retained solutions. MAP-Elites and CMA-ME counts refer to actual candidate evaluations; HC adds a separate polish budget.

Record **`git` commit hash**, **`uv`/`python` versions**, and any **`PRES_*`** overrides in lab notes when publishing numbers.

---

## 3. Shared experiment SSOT

**Module:** `scripts/v2_paper_2026/_experiment_body.py`

| Symbol | Role |
|--------|------|
| **`spectral_data_path()`** | Default Excel: `data/Test 3 - 4 White Powers/white_powders_with_labels.xlsx` |
| **`air_transmittance_path()`** | `data/Test 3 - 4 White Powers/Air transmittance.xlsx` |
| **`load_nominal_scene()`** | Single **`SceneConfig`** matching legacy drivers: `atmospheric_distance_ratio=0.11`, `temperature_kelvin=293.15`, `air_refractive_index=1.0` |
| **`default_gene_space()`** | **4** Gaussian channels × **`(μ, σ)`** bounds: `{"low": 4.0, "high": 20.0}` and `{"low": 0.1, "high": 4.0}` repeated |
| **`default_map_elites_geometry()`** | **`(grid_resolution=20, mu_range=(4.0, 20.0))`** — must stay consistent with **`archive.bin_coordinates`** / QD plots |
| **`num_params_per_basis_function()`** | **`2`** (μ, σ per channel) |
| **`wavelengths_array(scene)`** | Wavelength grid for plotting exports |

**Implication:** Changing paths or nominal scene parameters here changes **every** presentation step that imports this module unless a step explicitly overrides (only **08** uses YAML-driven scenes for fitness).

---

## 4. Budget and override SSOT

**Module:** `scripts/v2_paper_2026/_budgets.py`

**CLI:** Every `run_0X_*.py` accepts **`--quick`**, which swaps in **smoke-test** budgets via functions like **`quickrun_ga_baseline()`**, **`quickrun_map_elites()`**, **`quickrun_cma_me()`**, etc.

**Important:** Step **01** (standard GA) and step **02** (advanced / niching GA) use **different** constants on purpose (`PRES_GA_GENERATIONS` vs `PRES_GA_GENERATIONS_ADVANCED`, etc.).

### Environment variables (full catalog)

Variables are read with **`os.environ.get(..., default)`** in **`_budgets.py`**. Defaults below are the **code defaults** when the variable is unset.

| Variable | Default | Step(s) / meaning |
|----------|---------|-------------------|
| `PRES_GA_GENERATIONS` | `5000` | **01** standard GA generations |
| `PRES_GA_POP` | `240` | **01** population size |
| `PRES_GA_PARENTS_FRACTION_STANDARD` | `0.52` | **01** mating pool size as fraction of population |
| `PRES_GA_GENERATIONS_ADVANCED` | `4000` | **02** advanced GA generations |
| `PRES_GA_POP_ADVANCED` | `200` | **02** population size |
| `PRES_MULTI_START_RUNS` | `200` | **03** independent GA runs |
| `PRES_MULTI_START_GENS` | step 01 generations (`5000` by default) | **03** generations per run |
| `PRES_MULTI_START_POP` | step 01 population (`240` by default) | **03** population per run |
| `PRES_MULTI_START_WORKERS_CAP` | `200` | Cap when defaulting worker count: `min(CPU, cap)` |
| `PRES_MULTI_START_WORKERS` | *(derived)* | **03** explicit process pool size (overrides default) |
| `PRES_MAP_ELITES_TOTAL_EVALS` | `PRES_GA_GENERATIONS × PRES_GA_POP` (`1200000`) | **04**, **05** total calls: initial seeds + mutations |
| `PRES_MAP_ELITES_INITIAL` | step 01 population (`240` by default) | **04**, **05** random seeds for initial archive |
| `PRES_CMA_ME_EVALS` | `PRES_GA_GENERATIONS × PRES_GA_POP` (`1200000`) | **06** total fitness evaluations (includes initial seeding) |
| `PRES_CMA_ME_EMITTERS` | `6` | **06** number of CMA-ME emitters |
| `PRES_CMA_ME_INITIAL` | step 01 population (`240` by default) | **06** initial random archive size |
| `PRES_MINIMAX_EVALS` | `200000` | **08** total MAP-Elites calls (each eval runs **all** YAML conditions) |
| `PRES_MINIMAX_MULT` | `1.0` | **08** multiplier on **`PRES_MINIMAX_EVALS`** (CLI `--budget-multiplier` wins if set) |
| `PRES_ROBUSTNESS_TOP_N` | `64` | **07** cap on elites (`0` in env means evaluate **entire** archive — see script) |
| `PRES_HC_MAX_ELITES` | `160` | **05** max elites selected for hill climbing |
| `PRES_HC_THRESHOLD` | `45.0` | **05** minimum archive fitness to enter the “promising” polish pool |
| `PRES_HC_WORKERS_CAP` | `49` | Cap for default HC worker pool |
| `PRES_HC_WORKERS` | *(derived)* | **05** explicit HC pool size |

**Shared GA stop rule:** **`GA_STOP_CRITERIA = "saturate_10000"`** is passed into **`create_ga_config`** for steps **01** and **02**; **`_ga_multistart.run_single_ga_worker`** receives the same via payload from **03**.

---

## 5. Output layout and artifact contract

**Root:** `outputs/v2_paper_2026/<step_key>/`

Step keys are defined in **`scripts/v2_paper_2026/_paths.py`**:

| Key | Directory |
|-----|-----------|
| `STEP_STANDARD_GA` | `01_standard_ga` |
| `STEP_ADVANCED_GA` | `02_advanced_ga_niching` |
| `STEP_MULTI_START_GA` | `03_multi_start_ga` |
| `STEP_MAP_ELITES` | `04_map_elites` |
| `STEP_MAP_ELITES_HC` | `05_map_elites_hc` |
| `STEP_CMA_ME` | `06_cma_me` |
| `STEP_ROBUSTNESS` | `07_robustness_test` |
| `STEP_MINIMAX` | `08_minimax_optimization` |

### `run_summary.json`

Written by **`_artifacts.save_run_summary`**: always includes **`suite`: `"v2_paper_2026"`**, **`generated_utc`**, and step-specific **`step`**, **`method`**, **`budgets`**, **`wall_time_seconds`** (or split times for **05**), metrics, and **`output_files`**. Treat it as the **manifest** for what a completed run produced.

### Archive pickle schema (MAP-Elites / CMA-ME)

Both **`run_map_elites`** and **`run_cma_me`** use the same elite record shape:

- **Key:** **`(x_bin, y_bin)`** — integers from **`bin_coordinates`**
- **Value:** **`dict`** with at least:
  - **`chromosome`**: `np.ndarray` (length **8** for four `(μ, σ)` pairs)
  - **`fitness`**: `float` (maximin SAM objective)
  - **`mu_1`**, **`mu_2`**: descriptor features (sorted μ pair; see **`THEORY_AND_IMPLEMENTATION.md`** §14 `archive.py`)

**Robustness and minimax** consume this structure via **`evaluate_archive_robustness`**, which only requires **`chromosome`** (+ optional **`fitness`** for sorting).

---

## 6. Pipeline graph (dataflow)

```text
01_standard_ga ──────────────┐
02_advanced_ga_niching ──────┤  independent GA baselines (no downstream consumer in-suite)
03_multi_start_ga ───────────┘

04_map_elites ──► (no later step reads 04 outputs by default)

05_map_elites_hc ──► runs its own full MAP-Elites pass, then HC; does not ingest 04’s pickle

05_map_elites_hc/map_elites_archive.pkl ──► 07_robustness_test (default; pre-HC archive)

06_cma_me ──► cma_me_archive.pkl (optional input to 07 via --archive)

08_minimax_optimization ──► uses experiments/v2_paper_2026_minimax.yaml (not 06’s archive)
```

**Critical dependency:** **07** expects **`05_map_elites_hc/map_elites_archive.pkl`** unless you pass **`--archive`**. This is the archive before HC, not the polished result list.

**Independence:** **01–06** and **08** are independent training runs. **07** evaluates the archive from **05** by default. **04**, **05**, and **08** are separate MAP-Elites tracks; **06** uses CMA-ME.

---

## 7. Per-step narrative: `run_01` … `run_08`

### Step 01 — `run_01_standard_ga.py`

- **Intent:** Single-population GA **without** niching; **`mutation_type="random"`**.
- **Library:** **`AdvancedGA`**, **`create_ga_config`**, **`MinDissimilarityFitnessEvaluator`**, **`spectral_angle_mapper`**.
- **Budgets:** **`PRES_GA_GENERATIONS`**, **`PRES_GA_POP`**, parents from **`PRES_GA_PARENTS_FRACTION_STANDARD`**.
- **Outputs:** `final_population.pkl`, `best_chromosome.pkl`, descriptor scatter, top-elite spectrum PNGs, **`run_summary.json`**.
- **Evaluations (estimate):** `num_generations × sol_per_pop` (plus PyGAD internal semantics; same order of magnitude).

### Step 02 — `run_02_advanced_ga_niching.py`

- **Intent:** **`niching_enabled=True`**, **`diversity_preserving_mutation`**, default **`NichingConfig`** from **`create_ga_config`** (includes optimal pairing for grouped distances when configured in GA config — see **`ga/ga_configuration.py`**).
- **Budgets:** **`PRES_GA_GENERATIONS_ADVANCED`**, **`PRES_GA_POP_ADVANCED`**.
- **Outputs:** Same pattern as **01** (`final_population.pkl`, `best_chromosome.pkl`, plots, **`run_summary.json`**). Diversity metrics use **`calculate_population_diversity(..., ga.niching_config)`**.

### Step 03 — `run_03_multi_start_ga.py`

- **Intent:** **`PRES_MULTI_START_RUNS`** independent GAs in a **`ProcessPoolExecutor`**; each worker runs **`_ga_multistart.run_single_ga_worker`** (**`niching_enabled=False`**, **`mutation_type="random"`**, **`keep_elitism=5`**). Per-run budgets use **`PRES_MULTI_START_GENS`** and **`PRES_MULTI_START_POP`**, defaulting to step 01.
- **CLI:** **`--max-workers`** overrides default `min(num_runs, PRES_MULTI_START_WORKERS)`.
- **Outputs:** `champions.pkl` (`champions`, `champion_fitness`, `per_run`), `per_run_fitness.json`, overlay plot, top-elite PNGs, **`run_summary.json`**. Optional **mean pairwise distance** among champions via **`compute_population_distance_matrix`**.
- **Evaluations (estimate):** `num_runs × num_generations × sol_per_pop`.

### Step 04 — `run_04_map_elites.py`

- **Intent:** Vanilla **`run_map_elites`** (mutation-only QD).
- **Budgets:** **`PRES_MAP_ELITES_TOTAL_EVALS`**, **`PRES_MAP_ELITES_INITIAL`**; mutation iterations equal total minus initial seeds, and an initial count exceeding the total is rejected.
- **Outputs:** `map_elites_archive.pkl`, heatmap, top-elite PNGs, **`run_summary.json`**.

### Step 05 — `run_05_map_elites_hc.py`

- **Intent:** Full **`run_map_elites`** (same budgets as **04**), then **parallel hill climbing** on up to **`PRES_HC_MAX_ELITES`** elites with fitness **`> PRES_HC_THRESHOLD`**, capped by pool size and **`--hc-workers`** / **`PRES_HC_WORKERS`**.
- **Polish:** **`polish_single_elite_hc`** with **`num_iterations=5000`**, **`mutation_sigma=0.05`**, **`mutation_probability=0.2`**, **`adaptive_iterations=True`**, with **`random_seed=42 + elite index`** (see §8.5).
- **Outputs:** `map_elites_archive.pkl` (post-MAP state), `polished_results.pkl` (polished chromosomes + gains), heatmap, comparison plots, **`run_summary.json`**.

### Step 06 — `run_06_cma_me.py`

- **Intent:** **`run_cma_me`** on the nominal scene with presentation geometry.
- **Budgets:** **`PRES_CMA_ME_EVALS`**, **`PRES_CMA_ME_INITIAL`**, **`PRES_CMA_ME_EMITTERS`**; **`initial_sigma=0.2`**, **`restart_patience=30`** in script.
- **Outputs:** `cma_me_archive.pkl`, `cma_me_metadata.pkl`, heatmap, progress plot, top-elite PNGs, **`run_summary.json`**.
- **Metadata:** Includes **`history`** with keys **`evals`**, **`archive_size`**, **`best_fitness`**, **`coverage_pct`** for **`plot_cma_me_progress`**.

### Step 07 — `run_07_robustness_test.py`

- **Intent:** Phase-1 environmental stress: re-evaluate elites from a QD archive across a **temperature × distance × refractive index** grid loaded via **`load_substance_atmosphere_data`** (lists → Cartesian product of **`SceneConfig`**).
- **Input:** **`--archive`** or default **`outputs/v2_paper_2026/05_map_elites_hc/map_elites_archive.pkl`** (**`default_robustness_archive_path()`**), the pre-HC archive.
- **Grid:** **`ROBUSTNESS_TEMPERATURES_K`**, **`ROBUSTNESS_DISTANCE_RATIOS`**, **`ROBUSTNESS_REFRACTIVE_INDICES`** in **`_budgets.py`** (must include the **nominal** training triple for alignment).
- **Processing:** **`evaluate_archive_robustness`** → **`align_nominal_fitness_to_training_scene`** → **`summarise_robustness`**.
- **Outputs:** `robustness_summary.json`, `robustness_results.pkl`, `fitness_degradation_heatmap.png`, `fitness_distribution_by_condition.png`, `worst_case_vs_nominal.png`, `condition_sensitivity.png`, `retention_histogram.png`, **`run_summary.json`**.

### Step 08 — `run_08_minimax_optimization.py`

- **Intent:** Vanilla MAP-Elites where **each** fitness call uses **`MinDissimilarityFitnessEvaluator`** built from **`experiments/v2_paper_2026_minimax.yaml`**: **25 scenes** (5×5 temperature × distance), **`robustness_aggregation: min`** (worst-case across conditions).
- **CLI:** **`--experiment-yaml`**, **`--budget-multiplier`**, **`--quick`**.
- **Budgets:** **`PRES_MINIMAX_EVALS × PRES_MINIMAX_MULT`** total **`run_map_elites`** calls (initial seeds + mutations); **physics cost** ≈ **`total_evals × num_conditions`** forward passes.
- **Outputs:** `map_elites_archive.pkl`, `final_population.pkl`, `best_chromosome.pkl`, descriptor scatter, minimax heatmap/top-elite PNGs, and **`run_summary.json`** with the YAML path, aggregation, and estimated number of physics calls. Full runs use **`PRES_MAP_ELITES_INITIAL`** seeds, capped at the total budget; quick runs use the quick seed count. Budget multiplier and resulting total must be positive.

---

## 8. Algorithm annex

### 8.1 Optimal pairing (Hungarian / linear sum assignment)

**Code:** `src/lwi_microbolometer_design/analysis/optimal_pairing_distance.py`, integrated into population distances via `analysis/distance_matrix.py` and `ga/diversity.py`.

**Procedure**

1. Partition each chromosome vector into **`n`** groups of **`params_per_group`** consecutive parameters (for Gaussians, **`params_per_group=2`** → `(μ, σ)` pairs).
2. Require **`len(items_a) == len(items_b)`** (same number of groups).
3. Compute **`C ∈ ℝ^{n×n}`** where **`C[i,j]`** is the distance between group **`i`** of **A** and group **`j`** of **B**, using **`scipy.spatial.distance.cdist`** when a string **`metric`** is supplied, or a double loop over **`distance_func`**.
4. Solve **`linear_sum_assignment(C)`** (Jonker–Volgenant / Hungarian implementation in SciPy). Obtain a permutation **`π`** minimizing **`Σ_i C[i, π(i)]`**.
5. Return that **minimum total cost** as the distance between **A** and **B**.

**Complexity:** **O(n³)** in the number of groups (dominated by assignment; building **`C`** is **O(n²)** times cost of one group distance).

**Role in the project:** **Niching / diversity only.** It does **not** change the fitness landscape; **SAM** on **`sensor_outputs`** still defines fitness.

### 8.2 Vanilla MAP-Elites

**Code:** `src/lwi_microbolometer_design/map_elites/algorithm.py` (`run_map_elites`, `mutate_chromosome`), `archive.py` (initialization and binning).

**Pseudocode (aligned with implementation)**

```text
seed archive with num_initial random chromosomes (keep best per cell)
archive_list ← list of archive values

repeat num_iterations times:
    if archive_list empty:
        parent ← uniform random within gene_space
    else:
        parent ← uniform random elite from archive_list
    child ← mutate_chromosome(parent, gene_space, mutation_probability)
    f ← fitness_func(None, child, 0)
    (μ₁, μ₂) ← extract_features(child)
    (bx, by) ← bin_coordinates(μ₁, μ₂, grid_resolution, mu_range)
    if cell (bx, by) empty OR f > archive[(bx, by)].fitness:
        archive[(bx, by)] ← {chromosome: child, fitness: f, mu_1: μ₁, mu_2: μ₂}
        archive_list ← list(archive.values())
```

**Mutation detail:** For each gene, with probability **`mutation_probability`**, add Gaussian noise with **`mutation_sigma = 0.75 × gene_range`** for **even** indices (μ) and **`0.2 × gene_range`** for **odd** indices (σ), then clip to bounds.

### 8.3 CMA-ME: archive improvement and `OptimizingEmitter.tell`

**Improvement scalar** (for one evaluated candidate **`x`**), from **`run_cma_me`**:

- Let **`key = bin_coordinates(extract_features(x))`**.
- If **`key`** not in archive: **`imp = fitness(x)`** (new cell).
- Else if **`fitness(x) > archive[key].fitness`**: **`imp = fitness(x) - archive[key].fitness`**.
- Else: **`imp = 0`**.

**`OptimizingEmitter.tell`** (`map_elites/emitters.py`) converts **`improvements`** and **`fitnesses`** into **`objectives`** for **`cma.tell`** (CMA **minimizes** objectives):

- If **any** **`imp > 0`**: let **`M = max(improvements)`**. For each candidate **`k`**,
  **`objective_k = -imp_k`** if **`imp_k > 0`**, else **`objective_k = M + 1.0`**.
  So archive-improvers are strictly better (lower objective) than non-improvers; among improvers, **larger improvement** ⇒ **lower objective** ⇒ better rank.
- If **no** improver: **`objective_k = -fitness_k`** (standard fitness ascent in minimization form).

**Coordinates:** **`ask`** stores **normalized** CMA samples and returns **denormalized** chromosomes. **`tell`** prefers passing **normalized** vectors back to **`cma`** when they match the last **`ask`**.

### 8.4 CMA-ME: `run_cma_me` orchestration

**Code:** `src/lwi_microbolometer_design/map_elites/cma_me.py`.

1. **`initialize_archive`** — same random seeding contract as MAP-Elites; consumes **`num_initial`** evaluations.
2. Build **`UnitCubeScaler`** bounds from **`gene_space`**; each **`OptimizingEmitter`** wraps **`cma.CMAEvolutionStrategy`** on **[0,1]ⁿ** with **`initial_sigma`** in normalized space.
3. **`num_emitters`** emitters; emitter **`i`** starts at **`elite.chromosome`** for a **uniformly random** elite from the seeded archive.
4. Outer loop rounds until evaluation budget: for each emitter (until budget):
   - **`solutions = emitter.ask()`**
   - Keep at most the remaining budget’s number of solutions. For each **`solution`**: evaluate **`fitness_func`**, compute **`imp`**, maybe update **shared** `archive` dict
   - **`emitter.tell(solutions, improvements, fitnesses)`** only for a full batch
   - Increment the global eval counter by the number actually evaluated
   - If **`emitter.converged`**: pick a **new random archive elite** and **`emitter.restart(elite_chromosome)`**
5. Append snapshots to **`history`** on **`log_interval`** evals.
6. Return **`(archive, metadata)`**.

**Budget note:** **`total_evals`** includes initial seeding and is exact. The final partial batch updates archive cells without a CMA distribution update. Require **`1 <= num_initial <= total_evals`**. Empty cells accept any finite score, including zero or negative values; positive improvement alone governs emitter ranking. Nonfinite callback scores raise errors.

### 8.5 Hill-climbing polish

**Code:** `src/lwi_microbolometer_design/map_elites/polish.py` — **`polish_single_elite_hc`**.

Greedy hill climb: for **`actual_iterations`** steps (scaled from **`num_iterations`** when **`adaptive_iterations`** is true using a **hardcoded target fitness `59.0`**), propose **`candidate`** by Gaussian perturbation per gene (**`mutation_probability`**, **`mutation_sigma × range`**), clip to bounds, accept if fitness improves; track **best** seen.

**Caveat:** The **`59.0`** target is **presentation-era calibration**; if the global fitness scale shifts, adjust **`polish.py`** or interpret adaptive budgets accordingly.

---

## 9. Alternate entry points (`scripts/map_elites`, `scripts/ga`, `scripts/robustness`)

| Location | Default outputs / behavior |
|----------|----------------------------|
| **`scripts/map_elites/run_cma_me.py`** | Writes under **`outputs/map_elites/cma_me/`** (self-contained loader; not the presentation step layout). |
| **`scripts/map_elites/run_map_elites_raw.py`**, **`run_map_elites_*_polish.py`** | Legacy MAP-Elites drivers under **`outputs/map_elites/...`**. |
| **`scripts/ga/*.py`** | Tuning, demos, diversity studies — unrelated directory layout to **`outputs/v2_paper_2026/`**. |
| **`scripts/robustness/run_robustness_test.py`** | Scans **`outputs/map_elites/cma_me/cma_me_archive.pkl`** or **`raw/map_elites_raw_archive.pkl`**; writes **`outputs/robustness/`**. **Not** the same default archive as presentation **07** (which points at **`v2_paper_2026/05_map_elites_hc/`**). |

When comparing results across time, always note **which driver** produced an archive and **which path** robustness or plotting scripts load.

---

## 10. Repro checklist

1. **Repository root**; **`uv sync`** (or equivalent) so **`lwi_microbolometer_design`** imports cleanly.
2. **Data files** present at paths in **`_experiment_body`** (and YAML for **08**).
3. Record **`PRES_*`** overrides and **`--quick`** vs production flags.
4. **Order:** run **05** before **07** (or pass **`--archive`** explicitly).
5. **Robustness grid** includes the **nominal** training **`(T, distance_ratio, n)`** so **`align_nominal_fitness_to_training_scene`** is meaningful.
6. **Minimax:** confirm **`experiments/v2_paper_2026_minimax.yaml`** `robustness_aggregation` matches intent (**`min`** for worst-case training).
7. **Artifacts:** archive **`pickle`** files match the schema in §5; keep **`run_summary.json`** next to them for manifest recovery.

---

*End of experiment pipeline reference.*
