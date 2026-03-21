# GA Hyperparameter Tuning - Prioritized Task List

## Task Summary

Build a robust, scalable tuning pipeline to find optimal GA hyperparameters using grid search with multiprocessing support.

## Prioritized Tasks

### Phase 1: Core Infrastructure (Critical Path)

#### Task 1.1: Refactor `tuning.py` - Remove Outdated Code
**Priority**: 🔴 Critical
**File**: `src/lwi_microbolometer_design/ga/tuning.py`
**Estimated Time**: 2 hours

**Subtasks**:
- [ ] Remove `AdvancedGAConfig` references (lines 153-167)
- [ ] Update imports to use modern API (`create_ga_config`, `AdvancedGA`)
- [ ] Remove TODO comments about outdated API
- [ ] Clean up unused code

**Acceptance Criteria**:
- No references to `AdvancedGAConfig`
- All imports are correct
- Code compiles without errors

---

#### Task 1.2: Update `run_single_configuration()` Function
**Priority**: 🔴 Critical
**File**: `src/lwi_microbolometer_design/ga/tuning.py`
**Estimated Time**: 3 hours

**Subtasks**:
- [ ] Update function signature to match modern API
- [ ] Use `create_ga_config()` to build configuration
- [ ] Use `AdvancedGA` with dict-based config
- [ ] Extract results using modern result structure
- [ ] Handle `MinDissimilarityFitnessEvaluator` properly
- [ ] Return structured result dict matching demo_ga.py pattern

**Acceptance Criteria**:
- Function works with modern GA API
- Returns result dict compatible with demo_ga.py
- Handles fitness evaluator correctly
- Tested with single configuration

**Dependencies**: Task 1.1

---

#### Task 1.3: Refactor `HyperparameterTuner` Class
**Priority**: 🔴 Critical
**File**: `src/lwi_microbolometer_design/ga/tuning.py`
**Estimated Time**: 4 hours

**Subtasks**:
- [ ] Update `__init__()` to accept experiment config or search space
- [ ] Update `generate_configurations()` to use correct parameter names:
  - `sol_per_pop` (not `population_size`)
  - `mutation_probability` (not `mutation_rate_base`)
  - `crossover_probability` (not `crossover_rate`)
  - `keep_elitism` (not `elitism_size`)
  - Add support for `parent_selection_type`, `crossover_type`, etc.
- [ ] Update `tune()` to use modern API
- [ ] Add support for YAML experiment loading
- [ ] Update result aggregation to match new result structure

**Acceptance Criteria**:
- Class works with modern GA API
- Generates valid configurations
- Parameter names match `create_ga_config()` API
- Tested with small search space

**Dependencies**: Task 1.2

---

### Phase 2: Experiment Definition System

#### Task 2.1: Create Experiment Config Dataclass
**Priority**: 🟡 High
**File**: `src/lwi_microbolometer_design/ga/tuning.py`
**Estimated Time**: 1 hour

**Subtasks**:
- [ ] Create `ExperimentConfig` dataclass
- [ ] Add fields: name, data, sensor, search_space, execution, validation
- [ ] Add validation methods
- [ ] Add helper methods for data extraction

**Acceptance Criteria**:
- Dataclass defined with all required fields
- Type hints correct
- Can instantiate from dict

---

#### Task 2.2: Implement YAML Loading Function
**Priority**: 🟡 High
**File**: `src/lwi_microbolometer_design/ga/tuning.py`
**Estimated Time**: 2 hours

**Subtasks**:
- [ ] Add `yaml` to dependencies (check pyproject.toml)
- [ ] Create `load_experiment_config(yaml_path: Path) -> ExperimentConfig`
- [ ] Parse YAML structure
- [ ] Validate required fields
- [ ] Set defaults for optional fields
- [ ] Handle path resolution (relative to YAML file)

**Acceptance Criteria**:
- Can load example YAML file
- Validates structure correctly
- Sets defaults appropriately
- Handles missing fields gracefully

**Dependencies**: Task 2.1

---

#### Task 2.3: Create Fitness Evaluator Factory
**Priority**: 🟡 High
**File**: `src/lwi_microbolometer_design/ga/tuning.py`
**Estimated Time**: 2 hours

**Subtasks**:
- [ ] Create `create_fitness_evaluator_from_experiment(experiment: ExperimentConfig) -> Callable`
- [ ] Load data files using `load_substance_atmosphere_data()`
- [ ] Create `MinDissimilarityFitnessEvaluator`
- [ ] Return fitness function
- [ ] Handle errors gracefully

**Acceptance Criteria**:
- Can create fitness function from experiment config
- Uses correct data loading functions
- Returns callable compatible with GA

**Dependencies**: Task 2.2

---

### Phase 3: Result Handling and Aggregation

#### Task 3.1: Create Result Aggregation Function
**Priority**: 🟡 High
**File**: `src/lwi_microbolometer_design/ga/tuning.py`
**Estimated Time**: 2 hours

**Subtasks**:
- [ ] Create `aggregate_experiment_results(runs: list[dict]) -> dict`
- [ ] Aggregate: best_fitness (mean, std, min, max)
- [ ] Aggregate: mean_fitness, diversity_score, convergence_generation
- [ ] Aggregate: high_quality_solutions
- [ ] Store random seeds list
- [ ] Handle empty runs gracefully

**Acceptance Criteria**:
- Aggregates all required metrics
- Computes statistics correctly
- Handles edge cases (empty, single run)

**Dependencies**: Task 1.2

---

#### Task 3.2: Update Result Saving
**Priority**: 🟡 High
**File**: `src/lwi_microbolometer_design/ga/tuning.py`
**Estimated Time**: 2 hours

**Subtasks**:
- [ ] Update CSV saving to include all parameters
- [ ] Update CSV to include aggregated metrics
- [ ] Create JSON summary with metadata
- [ ] Store experiment config in summary
- [ ] Store random seeds for reproducibility
- [ ] Sort results by best_fitness (descending)

**Acceptance Criteria**:
- CSV contains all hyperparameters and metrics
- JSON summary is complete and valid
- Results are sorted correctly
- Seeds are stored for reproducibility

**Dependencies**: Task 3.1

---

### Phase 4: Visualization Integration

#### Task 4.1: Implement Top Configuration Visualization
**Priority**: 🟠 Medium (but CRITICAL for user requirement)
**File**: `src/lwi_microbolometer_design/ga/tuning.py`
**Estimated Time**: 3 hours

**Subtasks**:
- [ ] Create `visualize_top_configurations()` function
- [ ] Select top K configurations from results
- [ ] For each top config:
  - Load experiment data
  - Create fitness evaluator
  - Run GA with that configuration
  - Extract high-quality solutions
  - Call `plot_top_sensor_designs()` from visualization module
  - Save plot
- [ ] Handle errors gracefully (if GA run fails)

**Acceptance Criteria**:
- Generates `plot_top_sensor_designs` for top configurations
- Plots saved to output directory
- Handles failures gracefully
- Works with top 5 configurations (default)

**Dependencies**: Task 1.3, Task 2.3

---

### Phase 5: CLI Script

#### Task 5.1: Create `tune_ga.py` Script
**Priority**: 🟡 High
**File**: `scripts/tune_ga.py`
**Estimated Time**: 3 hours

**Subtasks**:
- [ ] Create script structure
- [ ] Add CLI argument parsing:
  - `experiment` (required): Path to YAML file
  - `--output-dir`: Output directory (default: `outputs/tuning/`)
  - `--validation`: Enable validation mode
  - `--top-k`: Number of top configs to visualize (default: 5)
- [ ] Load experiment config
- [ ] Create tuner instance
- [ ] Run tuning
- [ ] Generate visualizations
- [ ] Print summary

**Acceptance Criteria**:
- Script runs without errors
- CLI arguments work correctly
- Can run validation mode
- Can run full mode
- Outputs results correctly

**Dependencies**: Task 1.3, Task 2.2, Task 4.1

---

### Phase 6: Validation Mode Support

#### Task 6.1: Implement Validation Mode
**Priority**: 🟠 Medium
**File**: `src/lwi_microbolometer_design/ga/tuning.py`
**Estimated Time**: 2 hours

**Subtasks**:
- [ ] Check `experiment.validation.enabled` flag
- [ ] Override `num_generations` if validation enabled
- [ ] Limit number of configurations if `max_configs` specified
- [ ] Optionally reduce `num_runs_per_config` in validation mode
- [ ] Add validation mode indicators to output

**Acceptance Criteria**:
- Validation mode uses fewer generations
- Validation mode limits configurations
- Results marked as validation runs
- Quick to run (< 5 minutes for small search space)

**Dependencies**: Task 2.2, Task 1.3

---

### Phase 7: Testing and Examples

#### Task 7.1: Create Example Experiment YAML Files
**Priority**: 🟠 Medium
**File**: `experiments/` directory
**Estimated Time**: 1 hour

**Subtasks**:
- [ ] Create `experiments/example_validation.yaml`
  - Small search space (2-3 values per parameter)
  - Validation mode enabled
  - 100 generations
- [ ] Create `experiments/example_full_search.yaml`
  - Medium search space
  - Full generations (2000)
  - All key parameters
- [ ] Add comments explaining each section

**Acceptance Criteria**:
- YAML files are valid
- Can be loaded by script
- Comments are helpful
- Examples are realistic

**Dependencies**: Task 2.2

---

#### Task 7.2: Write Unit Tests
**Priority**: 🟠 Medium
**File**: `tests/test_tuning.py`
**Estimated Time**: 3 hours

**Subtasks**:
- [ ] Test `load_experiment_config()` with valid/invalid YAML
- [ ] Test `generate_configurations()` with various search spaces
- [ ] Test `run_single_experiment()` with mock data
- [ ] Test result aggregation functions
- [ ] Test parameter filtering logic
- [ ] Test validation mode logic

**Acceptance Criteria**:
- All tests pass
- Good coverage (>80%)
- Tests are fast (< 1 minute total)

**Dependencies**: Task 1.3, Task 2.2, Task 3.1, Task 6.1

---

#### Task 7.3: Manual Integration Testing
**Priority**: 🟠 Medium
**Estimated Time**: 2 hours

**Subtasks**:
- [ ] Run validation mode with example experiment
- [ ] Verify CSV output format
- [ ] Verify JSON summary format
- [ ] Verify visualization generation
- [ ] Verify parallel execution (check CPU usage)
- [ ] Verify reproducibility (run twice with same seed)

**Acceptance Criteria**:
- Validation run completes successfully
- All outputs are correct
- Visualizations generated
- Parallel execution works
- Results are reproducible

**Dependencies**: Task 5.1, Task 7.1

---

### Phase 8: Documentation

#### Task 8.1: Create Experiment Guide
**Priority**: 🟢 Low
**File**: `experiments/README.md`
**Estimated Time**: 1 hour

**Subtasks**:
- [ ] Document YAML structure
- [ ] Explain each section
- [ ] Provide examples
- [ ] Document parameter options
- [ ] Document validation mode

**Acceptance Criteria**:
- Documentation is clear
- Examples work
- All parameters documented

**Dependencies**: Task 7.1

---

#### Task 8.2: Update Main README
**Priority**: 🟢 Low
**File**: `README.md`
**Estimated Time**: 1 hour

**Subtasks**:
- [ ] Add tuning section
- [ ] Link to experiment guide
- [ ] Provide quick start example
- [ ] Document CLI usage

**Acceptance Criteria**:
- README updated
- Links work
- Examples are clear

**Dependencies**: Task 5.1, Task 8.1

---

## Task Dependencies Graph

```
Task 1.1 (Remove outdated code)
  └─> Task 1.2 (Update run_single_configuration)
      └─> Task 1.3 (Refactor HyperparameterTuner)
          └─> Task 4.1 (Visualization)
          └─> Task 5.1 (CLI Script)

Task 2.1 (ExperimentConfig dataclass)
  └─> Task 2.2 (YAML loading)
      └─> Task 2.3 (Fitness evaluator factory)
          └─> Task 4.1 (Visualization)
          └─> Task 5.1 (CLI Script)
      └─> Task 6.1 (Validation mode)
          └─> Task 5.1 (CLI Script)

Task 1.2 (Update run_single_configuration)
  └─> Task 3.1 (Result aggregation)
      └─> Task 3.2 (Result saving)
          └─> Task 5.1 (CLI Script)

Task 5.1 (CLI Script)
  └─> Task 7.3 (Integration testing)

Task 2.2 (YAML loading)
  └─> Task 7.1 (Example YAML files)
      └─> Task 7.3 (Integration testing)
      └─> Task 8.1 (Experiment guide)

Task 1.3, 2.2, 3.1, 6.1
  └─> Task 7.2 (Unit tests)
```

## Recommended Implementation Order

1. **Phase 1** (Core Infrastructure): Tasks 1.1 → 1.2 → 1.3
2. **Phase 2** (Experiment System): Tasks 2.1 → 2.2 → 2.3
3. **Phase 3** (Results): Tasks 3.1 → 3.2
4. **Phase 4** (Visualization): Task 4.1
5. **Phase 5** (CLI): Task 5.1
6. **Phase 6** (Validation): Task 6.1
7. **Phase 7** (Testing): Tasks 7.1 → 7.2 → 7.3
8. **Phase 8** (Documentation): Tasks 8.1 → 8.2

## Critical Path

The fastest path to a working system:
1. Task 1.1 (Remove outdated code) - 2h
2. Task 1.2 (Update run_single_configuration) - 3h
3. Task 1.3 (Refactor HyperparameterTuner) - 4h
4. Task 2.1 (ExperimentConfig) - 1h
5. Task 2.2 (YAML loading) - 2h
6. Task 2.3 (Fitness evaluator) - 2h
7. Task 3.1 (Result aggregation) - 2h
8. Task 3.2 (Result saving) - 2h
9. Task 5.1 (CLI Script) - 3h
10. Task 7.1 (Example YAML) - 1h
11. Task 7.3 (Integration test) - 2h

**Total Critical Path**: ~24 hours

## Notes

- Start with Phase 1 to get core functionality working
- Can test incrementally after each phase
- Validation mode (Task 6.1) can be added later if needed for initial testing
- Visualization (Task 4.1) is critical for user requirement but can be added after core works
- Focus on getting basic grid search working first, then add features
