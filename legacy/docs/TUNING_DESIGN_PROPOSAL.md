# GA Hyperparameter Tuning Design Proposal

## Executive Summary

This document proposes a complete redesign of the hyperparameter tuning system for the GA-based microbolometer sensor optimization. The new system will provide robust, scalable grid search capabilities with comprehensive logging, visualization, and parallel execution support.

## Architecture Decision: Module vs Script

**Decision: Hybrid Approach**

- **Core tuning logic**: Module under `src/lwi_microbolometer_design/ga/tuning.py`
  - Reusable classes and functions
  - Can be imported and used programmatically
  - Maintains consistency with existing codebase structure

- **Experiment runner**: Standalone script under `scripts/tune_ga.py`
  - CLI interface for launching experiments
  - Handles experiment definition loading
  - Orchestrates the tuning workflow
  - Easy to run without Python imports

**Rationale**: This separation follows the pattern already established in the codebase (e.g., `scripts/prototypes/demo_ga.py` uses GA modules from `src/`). The module provides flexibility for programmatic use, while the script provides convenience for command-line experimentation.

## Experiment Definition Format

**Format: YAML** (human-readable, supports comments, easy to edit)

**Structure**:
```yaml
experiment:
  name: "initial_grid_search"
  description: "Wide grid search over key hyperparameters"

  data:
    spectral_data_file: "data/Test 3 - 4 White Powers/white_powders_with_labels.xlsx"
    air_transmittance_file: "data/Test 3 - 4 White Powers/Air transmittance.xlsx"
    atmospheric_distance_ratio: 0.11
    temperature_kelvin: 293.15
    air_refractive_index: 1.0

  sensor:
    num_basis_functions: 4
    params_per_basis_function: 2
    param_bounds:
      - {low: 4.0, high: 20.0}  # mu
      - {low: 0.1, high: 4.0}   # sigma

  search_space:
    sol_per_pop: [100, 200, 300]
    num_generations: [1000, 2000]
    num_parents_mating: [40, 50, 60]
    parent_selection_type: ["tournament", "SSS", "SUS"]
    K_tournament: [3, 5]  # Only used if parent_selection_type is "tournament"
    crossover_type: ["uniform", "single_point", "two_points"]
    crossover_probability: [0.7, 0.8, 0.9]
    mutation_probability: [0.05, 0.1, 0.15, 0.2]
    keep_elitism: [3, 5, 10]
    niching_enabled: [true, false]
    niching_sigma_share: [0.5, 1.0, 2.0]
    niching_alpha: [0.5, 1.0]

  execution:
    num_runs_per_config: 3  # Number of independent runs per configuration
    random_seed_base: 42
    max_workers: 12  # Parallel workers (defaults to CPU count)
    fitness_threshold: 45.0  # For counting high-quality solutions

  validation:
    enabled: false  # Set to true for quick validation runs
    num_generations_override: 100  # Override generations for validation
    max_configs: 10  # Limit number of configs for validation
```

**Rationale**: YAML is human-readable, supports comments for documentation, and is easy to edit. It provides a clear structure that separates data loading, search space definition, and execution parameters.

## Parallel Execution Model

**Model: ProcessPoolExecutor with proper serialization**

- Use `multiprocessing.ProcessPoolExecutor` (already used in `scripts/prototypes/demo_ga.py`)
- Each worker runs a complete GA experiment (single configuration)
- Workers are stateless - all data passed as function arguments
- Use `mp.set_start_method('spawn')` for cross-platform compatibility

**Worker Function Design**:
```python
def run_single_experiment(
    experiment_id: str,
    config: dict,
    data: dict,
    gene_space: list,
    params_per_basis_function: int,
    num_generations: int,
    random_seed: int,
    fitness_threshold: float,
) -> dict:
    """Run single GA experiment and return results."""
    # Create fitness evaluator
    # Create GA config
    # Run GA
    # Extract metrics
    # Return structured result dict
```

**Rationale**:
- ProcessPoolExecutor handles worker lifecycle automatically
- Spawn method ensures clean state (important for reproducibility)
- Stateless workers avoid shared state issues
- Each experiment is independent and can fail without affecting others

## Orchestration Strategy

**Strategy: Staged Grid Search with Smart Filtering**

1. **Phase 1: Coarse Grid Search**
   - Wide parameter ranges, 2-3 values per parameter
   - Quick validation runs (100 generations) to eliminate poor configurations
   - Identify promising regions

2. **Phase 2: Refined Search**
   - Focus on top-performing regions from Phase 1
   - Narrower ranges around best values
   - Full runs (2000 generations)
   - More seeds per configuration (5-10 runs)

3. **Phase 3: Deep Dive (Optional)**
   - Very narrow search around best configurations
   - Fine-tune niching parameters
   - Custom mutation parameter tuning (if needed)

**Implementation Approach**:
- Use `itertools.product` for full factorial grid search
- Support filtering/invalid configurations (e.g., `num_parents_mating >= sol_per_pop`)
- Support early stopping based on intermediate results
- Cache results to avoid re-running identical configurations

**Rationale**:
- Grid search is simple, interpretable, and parallelizable
- Staged approach balances exploration vs exploitation
- Validation runs reduce wasted compute on poor configurations
- Full factorial ensures comprehensive coverage

## Result Logging and Artifacts

**CSV Results** (`tuning_results_TIMESTAMP.csv`):
- All hyperparameters
- Aggregated metrics (mean best_fitness, mean diversity, etc.)
- Statistical metrics (std, min, max across runs)
- Convergence statistics
- High-quality solution counts

**JSON Summary** (`tuning_summary_TIMESTAMP.json`):
- Experiment metadata
- Top N configurations
- Parameter importance analysis
- Niching effectiveness comparison
- Best configuration ready to load

**Visualization Artifacts**:
- For top K configurations (default K=5):
  - `plot_top_sensor_designs` - **CRITICAL** - shows diversity of top solutions
  - Fitness evolution plots
  - Diversity evolution plots
  - IVAT analysis (if applicable)

**Reproducibility**:
- Store random seeds for each run
- Save full experiment YAML config
- Store data loading parameters
- Log exact GA configuration dictionaries

**Rationale**: CSV for analysis, JSON for programmatic access, plots for visual assessment. The `plot_top_sensor_designs` visualization is critical for validating multimodal optimization success.

## API Changes Required

**Minimal changes needed** - existing API is good!

1. **No changes to `create_ga_config()`** - already supports all needed parameters
2. **No changes to `AdvancedGA`** - already compatible
3. **No changes to visualization** - `plot_top_sensor_designs` exists and works

**Potential enhancement** (optional):
- Add helper function to load experiment YAML and create fitness evaluator
- This could go in a new `ga.experiments` module or stay in `tuning.py`

## Implementation Plan

### Phase 1: Core Tuning Module
1. Refactor `tuning.py`:
   - Remove outdated `AdvancedGAConfig` references
   - Update to use `create_ga_config()` API
   - Implement new `HyperparameterTuner` class
   - Add experiment result aggregation and statistics

### Phase 2: Experiment Runner Script
2. Create `scripts/tune_ga.py`:
   - YAML experiment loader
   - CLI argument parsing
   - Data loading orchestration
   - Tuning execution
   - Result saving and visualization

### Phase 3: Integration and Testing
3. Test with validation runs:
   - Small search space, few generations
   - Verify parallel execution
   - Verify result logging
   - Verify visualization generation

4. Test with full runs:
   - Medium search space
   - Full generations
   - Verify performance and correctness

### Phase 4: Documentation
5. Create example experiment YAML files
6. Update README with tuning instructions
7. Document hyperparameter recommendations

## Risk Mitigation

**Risk: Too many configurations (combinatorial explosion)**
- **Mitigation**: Support validation mode, staged search, config filtering

**Risk: Long runtime**
- **Mitigation**: Parallel execution, early stopping, validation runs

**Risk: Reproducibility issues**
- **Mitigation**: Store seeds, save configs, use spawn method

**Risk: Memory issues with large result sets**
- **Mitigation**: Stream results to disk, don't keep all in memory

## Success Criteria

1. ✅ Can run grid search over 10+ hyperparameters
2. ✅ Parallel execution scales to 12 cores
3. ✅ Results logged in CSV/JSON format
4. ✅ Top configurations visualized with `plot_top_sensor_designs`
5. ✅ Experiment definitions are easy to create and modify
6. ✅ Validation runs work correctly
7. ✅ Full runs complete in reasonable time
8. ✅ Results are reproducible

## Next Steps

1. Review and approve this design
2. Implement Phase 1 (core module)
3. Implement Phase 2 (script)
4. Test with validation runs
5. Run full experiments
6. Iterate based on results
