# QD Algorithms & Architecture Discussion Summary

This document summarizes a Q&A session about Quality-Diversity algorithms, the relationship between GA and MAP-Elites, and the architecture of the optimization codebase.

---

## 1. MAP-Elites Module Architecture

### Question: Should MAP-Elites be a separate module like GA, or does it share too much with GA?

**What MAP-Elites reuses from GA code:**

| Component | File | Reuse Level |
|---|---|---|
| Fitness evaluation | `ga/fitness.py` (`MinDissimilarityFitnessEvaluator`) | 100% identical |
| Adaptive mutation | `ga/mutations.py` (`diversity_preserving_mutation`) | Can use directly; MAP-Elites script currently has its own simpler version |
| Experiment config loading | `ga/experiment.py` (`ExperimentConfig`, YAML loading) | 90%+ reusable |
| Population diversity analysis | `ga/population_analysis.py` | Mostly reusable |
| Some visualization | `ga/visualization.py` (top designs, IVAT) | Same patterns, different details |

**What MAP-Elites does NOT need from GA:**

| Component | File | Reason |
|---|---|---|
| PyGAD wrapper | `ga/advanced_ga.py` (`AdvancedGA`) | MAP-Elites has no generation/population concept |
| Niching | `ga/advanced_ga.py` (`NichingConfig`) | Archive itself IS the diversity mechanism |
| GA hyperparameter tuning | `ga/tuning.py` | MAP-Elites has its own hyperparameters |
| GA config factory | `ga/ga_configuration.py` | GA-specific parameter structure |
| GA result extraction | `ga/result_extraction.py` | GA's population/generation concept doesn't apply |

**What MAP-Elites has that GA doesn't:**
- Archive (elite library indexed by feature descriptors)
- Feature extraction (chromosome → bin coordinates)
- Binning / Grid management
- Archive-specific visualization (fitness heatmap landscape)

### Recommended Architecture (clean version)

```
src/lwi_microbolometer_design/
├── data/              (unchanged)
├── simulation/        (unchanged)
├── analysis/          (unchanged)
├── optimization/      (NEW - shared optimization infrastructure)
│   ├── fitness.py              ← moved from ga/fitness.py
│   ├── mutations.py            ← moved from ga/mutations.py
│   └── experiment.py           ← moved from ga/experiment.py
├── ga/                (GA-specific code only)
│   ├── advanced_ga.py, ga_configuration.py, tuning.py, diversity.py,
│   │   population_analysis.py, result_extraction.py, visualization.py
├── map_elites/        (NEW - MAP-Elites specific)
│   ├── archive.py, algorithm.py, visualization.py
└── visualization/     (unchanged - general visualization)
```

### Pragmatic approach (recommended for now)

1. Create `map_elites/` module
2. Have it import directly from `ga/` (e.g., `from lwi_microbolometer_design.ga.fitness import ...`)
3. Extract shared `optimization/` layer later when adding more algorithms (CMA-ES, surrogate, etc.)

---

## 2. GA vs MAP-Elites: Unified Mental Model

### They are both instances of the same pattern

Both GA and MAP-Elites are iterative optimization algorithms that:
1. Generate new candidate solutions
2. Evaluate each candidate's fitness
3. Use old solutions to generate new ones
4. Iterate to find the best

### The key difference: competition scope

| | GA | MAP-Elites |
|---|---|---|
| Who competes with whom | **All solutions compete against each other.** Worse ones get eliminated. | **Only solutions in the same bin compete.** Different bins don't interfere. |
| Result | Entire population tends to converge to one peak (mode collapse) | Each bin independently retains its region's best (diversity guaranteed) |
| Capacity per "cell" | No cell concept; all mixed together; capacity = population_size | Each cell capacity = 1 |
| New solution generation | Crossover (combining parents' strengths) + mutation (perturbation based on existing solutions) | Currently: random parent from archive + mutation only |

### Analogy

- **GA** = One big classroom, all students take the same exam, ranked by total score. Survivors are all the same type of "top student."
- **MAP-Elites** = Many small classrooms by major, each keeps only its #1 student. Math champion and Literature champion don't compete with each other.
- **MAP-Elites + GA emitter** = Classrooms by major, but new students are created by "crossing" champions from two different classrooms, then placed in whichever classroom matches their major.

### MAP-Elites mutation is NOT purely random

It picks an existing good solution from the archive as parent, then perturbs it:

```python
parent = archive_list[np.random.randint(len(archive_list))]
parent_chromosome = parent["chromosome"]
child_chromosome = mutate_chromosome(parent_chromosome, gene_space, mutation_probability)
```

The parent provides the starting point; mutation explores around it.

### MAP-Elites exploitation is weaker than GA

Because each bin only has 1 solution, there's no population to do crossover and selection within a bin. Improvements:
1. Add crossover (select 2 parents from different bins)
2. Add local polish (CMA-ES or hill climbing on high-fitness bin occupants)
3. Increase grid resolution (more bins = finer exploration)

---

## 3. Quality-Diversity (QD) Algorithm Framework

### Using GA to generate solutions for MAP-Elites is NOT a chimera

This is a well-studied family of algorithms called **Quality-Diversity (QD) algorithms**, based on the **Emitter + Archive** architecture:

- **Archive**: MAP-Elites grid — stores elites by feature descriptor, ensures diversity
- **Emitter**: Any algorithm that generates new candidate solutions — can be mutation, crossover, CMA-ES, random, etc.

```
┌─────────────────────────────────────────────┐
│                   Archive                    │
│  ┌───┬───┬───┬───┐                          │
│  │ . │ . │56 │ . │  ← binned by (μ₁, μ₂)   │
│  ├───┼───┼───┼───┤     each cell keeps best  │
│  │52 │ . │59 │ . │                          │
│  ├───┼───┼───┼───┤                          │
│  │ . │55 │ . │48 │                          │
│  └───┴───┴───┴───┘                          │
│         ↑ place (if better)    ↓ sample      │
│         │                     │             │
│    ┌────┴─────────────────────┴────┐        │
│    │         Emitter(s)            │        │
│    │  mutation / crossover / CMA   │        │
│    └───────────────────────────────┘        │
└─────────────────────────────────────────────┘
```

### Different QD algorithm variants

| Algorithm | Archive | Emitter |
|---|---|---|
| Current MAP-Elites | MAP-Elites grid | Random parent from archive → Gaussian mutation |
| MAP-Elites + GA | MAP-Elites grid | 2 parents from archive → crossover + mutation |
| MAP-Elites + Hill Climbing polish | MAP-Elites grid | Mutation (exploration) + greedy local search (exploitation, post-hoc) |
| **CMA-ME** | MAP-Elites grid | **CMA-ES** (learns search direction, runs as emitter) |
| Multi-Emitter MAP-Elites | MAP-Elites grid | Multiple emitters simultaneously (mutation + crossover + CMA-ES + random) |

---

## 4. Hill Climbing Polish Explained

The current `polish_map_elites.py` does:

1. Take an existing good solution (an elite from the MAP-Elites archive)
2. Add a **very small** random perturbation to each parameter (`mutation_sigma = 0.05`, only 5% of range)
3. Evaluate the new solution's fitness
4. **If better → accept. If worse → discard.** (This is the "greedy" / "hill climbing" acceptance criterion)
5. Repeat thousands to tens of thousands of times

"Hill climbing" is named for its **behavior pattern** — it only moves uphill (to higher fitness), never downhill. The name describes the **acceptance criterion** (only accept improvements), not the sampling method.

### Limitations
- Can only reach **local** optima, cannot escape current "mountain"
- Inefficient — purely random probing, doesn't learn search direction
- Each parameter is perturbed independently — cannot discover correlations between parameters

---

## 5. CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

### Core idea: Learn which direction to search, instead of searching blindly

**Hill climbing** perturbs each parameter independently with equal variance — the search region is a hypersphere (all directions equally likely).

**CMA-ES** adapts the search distribution based on which directions were successful — the search region becomes an oriented ellipsoid aligned with the fitness landscape.

### How CMA-ES learns (without knowing the landscape)

**Generation 1**: Sample from isotropic Gaussian (circle). Evaluate all candidates. Observe that successful candidates cluster in a particular direction.

**Generation 2**: Update the covariance matrix to stretch the distribution toward the successful direction (circle → ellipse). Move the center toward the successful candidates.

**Generation 3+**: Continue refining. After a few generations, the ellipse naturally aligns with the direction of steepest fitness improvement.

**Key insight**: CMA-ES never "computes" the fitness landscape. It implicitly learns the local shape by observing which samples succeed and which fail. The covariance matrix C encodes "which directions are more likely to produce improvements."

### Hill Climbing vs CMA-ES comparison

| | Hill Climbing (current polish) | CMA-ES |
|---|---|---|
| How candidates are generated | Each parameter independently gets random noise | Sampled from multivariate Gaussian with learned covariance (parameters are correlated) |
| How many candidates per step | 1 (generate 1, keep if better, discard if worse) | Generate λ candidates, select best μ, use them to update distribution |
| Does it learn? | No. Step 1000 uses same strategy as step 1. | Yes. Each generation adjusts search direction and step size based on success/failure. |
| Parameter coupling | Invisible. mu and sigma perturbations are fully independent. | Learnable. "increasing mu while decreasing sigma is good" gets encoded in covariance matrix. |

### Drop-in replacement for hill climbing

```python
import cma  # pip install cma

def polish_with_cma(chromosome, fitness_func, gene_space):
    bounds_low = [g['low'] for g in gene_space]
    bounds_high = [g['high'] for g in gene_space]

    result = cma.fmin(
        lambda x: -fitness_func(None, x, 0),  # negate because cma minimizes
        chromosome,                             # starting point = MAP-Elites elite
        0.5,                                    # initial step size
        options={
            'bounds': [bounds_low, bounds_high],
            'maxfevals': 5000,
        }
    )
    return result[0]  # optimized solution
```

---

## 6. CMA-ME (CMA-MAP-Elites) Detailed

### CMA-ME = CMA-ES used as emitter inside MAP-Elites

- **CMA-ES** = A local optimizer ("find the peak of one mountain")
- **CMA-ME** = Put CMA-ES inside MAP-Elites as an emitter ("use CMA-ES to find peaks on every mountain")

### How CMA-ME works step by step

**Initialization:**
1. Create empty 20×20 archive
2. Create K CMA-ES emitters (e.g., K=5), each with a random starting point
3. Each emitter has its own mean (center) and C (covariance matrix, initially identity = isotropic)

**Main loop (for each emitter, e.g., emitter_1):**

**Step 1: Sample.** emitter_1 samples λ=20 candidates from its distribution N(mean₁, C₁).

**Step 2: Evaluate.** For each candidate, compute:
- fitness (via simulate_sensor_output → SAM → min_dissimilarity)
- feature descriptor (smallest two mu values → bin coordinates)

**Step 3: Try to place in archive.** Each candidate goes to its corresponding bin. If the bin is empty or the candidate is better than the current occupant → place it (success). Otherwise → reject (failure).

**Step 4: Update CMA-ES distribution.** This is the key innovation. CMA-ME has two emitter strategies:

- **Improvement Emitter**: "Success" = candidate that improved the archive (either filled an empty bin or replaced a worse occupant). CMA-ES updates toward these candidates. Effect: learns to search in directions that improve the archive — could be higher fitness OR exploring new empty bins.

- **Optimizing Emitter**: Standard CMA-ES — ranks candidates purely by fitness. Effect: pure fitness optimization, diversity comes from candidates naturally landing in different bins.

**In practice, run multiple emitters simultaneously:**
```
emitter_1: Improvement strategy (encourages exploring new bins)
emitter_2: Improvement strategy
emitter_3: Optimizing strategy (pure fitness pursuit)
emitter_4: Optimizing strategy
emitter_5: Random emitter (fills blind spots)
```

**Step 5: Check for restart.** If an emitter's step size shrinks below threshold (converged, nothing left to search):
- Pick a random elite from the archive as new starting point
- Reset covariance matrix to identity
- Restart search from there

### Comparison with current two-phase approach

```
Current approach (two stages, serial):

  Stage 1: MAP-Elites (random mutation emitter, 200,000 iterations)
     → Gets an archive with many filled bins, but solutions aren't polished

  Stage 2: Hill Climbing polish (top 120 elites, 5,000-20,000 iterations each)
     → Polishes good regions to near-optimal

CMA-ME (integrated, simultaneous):

  Each generation does both:
     - Improvement emitters explore new bins (≈ Stage 1)
     - Optimizing emitters polish existing bins (≈ Stage 2)
  → Exploration and exploitation interleave every generation
```

Core advantage: The current approach is coarse-then-fine in two steps. CMA-ME is coarse-and-fine simultaneously every generation. Plus, CMA-ES emitters are much smarter than random mutation (they learn direction), so the same compute budget produces better archive quality and coverage.

### Upgrade paths

| Path | Change Required | Effect |
|---|---|---|
| Replace hill climbing polish with CMA-ES | Change 1 function (`polish_single_elite`) | Polish stage efficiency improves, exploration stage unchanged |
| Replace entire MAP-Elites + polish with CMA-ME | Rewrite main loop using pyribs library | Exploration and polish happen simultaneously; theoretically optimal |

**Pragmatic recommendation**: Start with path 1 (swap hill climbing for CMA-ES) — small change, immediate benefit. CMA-ME can come later, or use the [pyribs](https://pyribs.org/) library which already implements CMA-ME and multiple emitter types — you just provide a fitness function and feature extraction function.
