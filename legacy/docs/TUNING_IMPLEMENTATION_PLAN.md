# GA Hyperparameter Tuning Implementation Plan

## Overview

This document provides a step-by-step implementation plan for building the robust, scalable tuning pipeline for GA hyperparameters.

## Architecture Summary

- **Module**: `src/lwi_microbolometer_design/ga/tuning.py` - Core tuning logic
- **Script**: `scripts/tune_ga.py` - CLI experiment runner
- **Format**: YAML experiment definitions
- **Strategy**: Grid search with staged refinement
- **Execution**: Multiprocessing with ProcessPoolExecutor

## Implementation Steps

### Step 1: Refactor Core Tuning Module (`tuning.py`)

**File**: `src/lwi_microbolometer_design/ga/tuning.py`

**Changes**:
1. Remove outdated `AdvancedGAConfig` references (lines 153-167)
2. Update `run_single_configuration()` to use `create_ga_config()` API
3. Refactor `HyperparameterTuner` class:
   - Update `generate_configurations()` to use new parameter names
   - Update `tune()` to use modern GA API
   - Add support for YAML experiment definitions
   - Add result aggregation and statistics
4. Add new result dataclass that matches modern GA result structure
5. Add experiment loading functions

**Key Functions to Update**:
- `run_single_configuration()` → Use `create_ga_config()` and modern `AdvancedGA` API
- `HyperparameterTuner.generate_configurations()` → Support new parameter names
- `HyperparameterTuner.tune()` → Use modern result extraction

**New Functions to Add**:
- `load_experiment_config(yaml_path: Path) -> dict`
- `create_fitness_evaluator_from_experiment(experiment_config: dict) -> Callable`
- `aggregate_experiment_results(results: list[dict]) -> dict`

**Dependencies**:
- Keep existing imports
- Add `yaml` import for experiment loading
- Use `create_ga_config` from `ga_configuration.py`
- Use `MinDissimilarityFitnessEvaluator` from `fitness.py`

### Step 2: Create Experiment Runner Script

**File**: `scripts/tune_ga.py`

**Structure**:
```python
#!/usr/bin/env python3
"""GA Hyperparameter Tuning Script.

Launches grid search experiments over GA hyperparameters.
"""

import argparse
import logging
from pathlib import Path

from lwi_microbolometer_design.ga.tuning import (
    HyperparameterTuner,
    load_experiment_config,
    create_fitness_evaluator_from_experiment,
)

def main():
    parser = argparse.ArgumentParser(...)
    parser.add_argument('experiment', type=Path, help='Path to experiment YAML')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--validation', action='store_true')
    # ... more args

    args = parser.parse_args()

    # Load experiment config
    experiment = load_experiment_config(args.experiment)

    # Load data
    data = load_data_from_experiment(experiment)

    # Create fitness evaluator
    fitness_func = create_fitness_evaluator_from_experiment(experiment)

    # Create search space from experiment config
    search_space = create_search_space_from_experiment(experiment)

    # Create tuner
    tuner = HyperparameterTuner(...)

    # Run tuning
    results = tuner.tune(...)

    # Generate visualizations for top configs
    visualize_top_configurations(results, experiment, args.output_dir)
```

**Key Features**:
- Load YAML experiment definition
- Handle validation mode (fewer generations, fewer configs)
- Orchestrate data loading, fitness creation, tuning execution
- Generate visualizations for top configurations
- Save results and summaries

### Step 3: Create Experiment Definition Template

**File**: `experiments/tuning_experiment_template.yaml`

**Purpose**: Template showing all available options with comments

**Structure**: See design proposal for example structure

### Step 4: Update Tuning Module API

**File**: `src/lwi_microbolometer_design/ga/tuning.py`

**New Classes/Functions**:

1. **`ExperimentConfig` dataclass**:
   ```python
   @dataclass
   class ExperimentConfig:
       name: str
       data: dict
       sensor: dict
       search_space: dict
       execution: dict
       validation: dict | None
   ```

2. **`TuningResult` (updated)**:
   ```python
   @dataclass
   class TuningResult:
       experiment_id: str
       config: dict
       runs: list[dict]  # Individual run results
       aggregated: dict  # Mean, std, etc.
       random_seeds: list[int]
   ```

3. **`load_experiment_config(yaml_path: Path) -> ExperimentConfig`**:
   - Load and parse YAML
   - Validate structure
   - Set defaults
   - Return ExperimentConfig

4. **`create_fitness_evaluator_from_experiment(experiment: ExperimentConfig) -> Callable`**:
   - Load data files
   - Create MinDissimilarityFitnessEvaluator
   - Return fitness function

5. **`run_single_experiment(...) -> dict`**:
   - Updated to use modern API
   - Accept experiment_id, config dict, data, gene_space, etc.
   - Return structured result dict

6. **`HyperparameterTuner` (refactored)**:
   - Update `__init__()` to accept ExperimentConfig
   - Update `generate_configurations()` to handle new parameter names
   - Update `tune()` to:
     - Generate configurations from search space
     - Run experiments in parallel
     - Aggregate results
     - Save CSV/JSON
     - Return DataFrame

7. **`visualize_top_configurations(results_df: pd.DataFrame, experiment: ExperimentConfig, output_dir: Path, top_k: int = 5)`**:
   - Select top K configurations
   - For each, run quick GA with that config
   - Generate `plot_top_sensor_designs` visualizations
   - Save plots

### Step 5: Handle Parameter Dependencies

**Issue**: Some parameters are conditional (e.g., `K_tournament` only matters if `parent_selection_type == 'tournament'`)

**Solution**: Filter configurations in `generate_configurations()`:
```python
def generate_configurations(self) -> list[dict]:
    configs = []
    for combination in itertools.product(...):
        config = dict(zip(param_names, combination))

        # Skip invalid configurations
        if config['num_parents_mating'] >= config['sol_per_pop']:
            continue
        if config['keep_elitism'] >= config['sol_per_pop']:
            continue

        # Handle conditional parameters
        if config['parent_selection_type'] != 'tournament':
            # K_tournament doesn't matter, but we'll still include it
            pass

        configs.append(config)
    return configs
```

### Step 6: Implement Result Aggregation

**Function**: `aggregate_experiment_results(runs: list[dict]) -> dict`

**Metrics to Aggregate**:
- `best_fitness`: mean, std, min, max
- `mean_fitness`: mean, std
- `diversity_score`: mean, std
- `convergence_generation`: mean, std
- `high_quality_solutions`: mean, std, sum
- `final_population`: not aggregated (too large)
- `final_fitness_scores`: not aggregated (too large)

**Result Structure**:
```python
{
    'best_fitness_mean': float,
    'best_fitness_std': float,
    'best_fitness_min': float,
    'best_fitness_max': float,
    'mean_fitness_mean': float,
    'diversity_score_mean': float,
    'convergence_generation_mean': float,
    'high_quality_solutions_mean': float,
    'num_runs': int,
    'random_seeds': list[int],
}
```

### Step 7: Implement Visualization for Top Configurations

**Function**: `visualize_top_configurations(...)`

**Process**:
1. Sort results by `best_fitness_mean` (descending)
2. Select top K configurations
3. For each top config:
   - Load experiment data
   - Create fitness evaluator
   - Run GA with that configuration (full run or quick run)
   - Extract high-quality solutions
   - Call `plot_top_sensor_designs()` from visualization module
   - Save plot to output directory

**Output**:
- `top_1_sensor_designs.png`
- `top_2_sensor_designs.png`
- etc.

### Step 8: Add Validation Mode Support

**Implementation**:
- Check `experiment.validation.enabled`
- If enabled:
  - Override `num_generations` with `validation.num_generations_override`
  - Limit configurations to `validation.max_configs` (random sample or first N)
  - Reduce `num_runs_per_config` if specified

**Example**:
```python
if experiment.validation and experiment.validation.enabled:
    num_generations = experiment.validation.num_generations_override
    max_configs = experiment.validation.max_configs
    # Limit search space
else:
    num_generations = experiment.search_space.num_generations
    max_configs = None
```

### Step 9: Testing Strategy

**Unit Tests** (`tests/test_tuning.py`):
1. Test `load_experiment_config()` with valid/invalid YAML
2. Test `generate_configurations()` with various search spaces
3. Test `run_single_experiment()` with mock data
4. Test result aggregation functions
5. Test parameter filtering logic

**Integration Tests**:
1. Run validation mode with small experiment
2. Verify CSV/JSON output format
3. Verify visualization generation
4. Verify parallel execution

**Manual Testing**:
1. Create example experiment YAML
2. Run validation mode (should complete in < 5 minutes)
3. Run small full experiment (should complete in < 30 minutes)
4. Verify results are correct and reproducible

### Step 10: Documentation

**Files to Create/Update**:
1. `docs/TUNING_DESIGN_PROPOSAL.md` ✅ (created)
2. `docs/TUNING_IMPLEMENTATION_PLAN.md` ✅ (this file)
3. `experiments/README.md` - How to create experiment definitions
4. `experiments/example_validation.yaml` - Example validation experiment
5. `experiments/example_full_search.yaml` - Example full search experiment
6. Update main `README.md` with tuning instructions

## Code Organization

### Module Structure (`tuning.py`)

```python
# Imports
# Constants
# Dataclasses (ExperimentConfig, TuningResult, HyperparameterSearchSpace)
# Helper Functions (load_experiment_config, create_fitness_evaluator, etc.)
# Core Functions (run_single_experiment, aggregate_results)
# Classes (HyperparameterTuner)
# Factory Functions (create_default_search_space, etc.)
```

### Script Structure (`tune_ga.py`)

```python
# Imports
# Constants
# Helper Functions (load_data, create_gene_space, etc.)
# Main Function (main())
# CLI Parsing (parse_arguments())
# Entry Point (if __name__ == '__main__')
```

## Parameter Priority (What to Tune First)

**High Priority** (tune first):
1. `niching_enabled` - Critical for multimodal optimization
2. `niching_sigma_share` - Key niching parameter
3. `niching_alpha` - Secondary niching parameter
4. `mutation_probability` - Important for exploration
5. `sol_per_pop` - Population size affects diversity
6. `num_generations` - Convergence time

**Medium Priority** (tune second):
7. `num_parents_mating` - Affects selection pressure
8. `crossover_type` - Crossover strategy
9. `crossover_probability` - Crossover rate
10. `parent_selection_type` - Selection method
11. `keep_elitism` - Elitism size

**Low Priority** (fix defaults, tune later):
12. `K_tournament` - Only if using tournament selection
13. Custom mutation internals - Use defaults initially

## Implementation Order

1. ✅ Design proposal (done)
2. ✅ Implementation plan (this document)
3. **Next**: Refactor `tuning.py` core module
4. Create `tune_ga.py` script
5. Create example experiment YAML files
6. Test validation mode
7. Test full runs
8. Generate documentation
9. Iterate based on results

## Success Metrics

- [ ] Can load experiment from YAML
- [ ] Can run grid search over 10+ parameters
- [ ] Parallel execution uses all 12 cores efficiently
- [ ] Results saved in CSV format
- [ ] Results saved in JSON format
- [ ] Top configurations visualized with `plot_top_sensor_designs`
- [ ] Validation mode works (quick runs)
- [ ] Full runs complete successfully
- [ ] Results are reproducible (seeds stored)
- [ ] Code passes linting and type checking

## Estimated Effort

- **Step 1** (Refactor module): 4-6 hours
- **Step 2** (Create script): 2-3 hours
- **Step 3** (Templates): 1 hour
- **Step 4-7** (Features): 4-6 hours
- **Step 8** (Validation): 1-2 hours
- **Step 9** (Testing): 2-3 hours
- **Step 10** (Documentation): 2-3 hours

**Total**: ~16-24 hours

## Notes

- Start with validation mode to test quickly
- Use staged search to manage compute cost
- Focus on niching parameters first (most important)
- Keep custom mutation internals as defaults initially
- Iterate based on initial results
