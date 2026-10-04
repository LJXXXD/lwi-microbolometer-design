# LWI Project Instructions

## 1. Working Approach

Use current source and task-relevant documentation to understand the work. Preserve intended goals and justified constraints; choose methods, tools, and structure according to current needs and evidence. Existing layouts and implementations are not permanent requirements.

Prefer clear, cohesive code. Reuse existing functionality where suitable; introduce abstractions, dependencies, or compatibility handling for concrete needs. Use validation where external inputs or independently callable APIs require it; internal helpers may rely on valid upstream guarantees. Avoid speculative fallbacks and repeated checks without a distinct purpose.

Give adjustable settings clear ownership, defaults, and override behavior. Choose function parameters, local constants, or shared configuration according to actual use.

## 2. Scientific Correctness

For physical models, quantities and units, spectral/tensor contracts, or objective interpretation, consult the relevant sections of `docs/THEORY_AND_IMPLEMENTATION.md` and affected API contracts. For reproducing current experiments, consult `docs/01_EXPERIMENT_PIPELINE.md`. Resolve disagreements between intended requirements and current behavior explicitly; neither code nor documentation alone establishes scientific correctness.

Preserve raw spectra, material/reference data, and experimental metadata. Make transformations, exclusions, units, missing-data handling, and assumptions explicit. Keep spectral grids aligned and respect the applicable physical domains when combining inputs. Keep blackbody, atmospheric transmission, emissivity, and spectral integration assumptions explicit, with consistent spectral densities and dimensional factors. Do not disguise invalid input or substitute a different mathematical quantity merely to complete execution.

Keep optimization objectives and evaluation metrics distinct from broader scientific claims. An improved SAM or optimization score alone does not establish better substance identification or physical sensor performance. Preserve explicitly defined handling of degenerate or infeasible candidates; do not turn unexpected execution failures into valid scores.

Preserve intended behavior during refactoring, including supported entry points, configuration semantics, and consumed result formats. Explain and validate intentional changes, especially those affecting scientific meaning. Match evaluation to the intended independent unit and claim; prevent leakage where held-out evaluation is required.

Keep consequential results reproducible through relevant input provenance, effective parameters, seeds, scene grids, and evaluation budgets. Choose precision and optimization from numerical requirements and demonstrated needs. Support non-obvious scientific choices with appropriate sources or validation.

## 3. Code and Documentation

Use `pyproject.toml` for dependencies and formatter/linter settings. Follow consistent naming, absolute package imports, and practical type hints.

Use NumPy conventions for structured docstrings. Simple helpers may use one-line docstrings. Explain non-obvious contracts, units, shapes, assumptions, and side effects; avoid repeating obvious code or rewriting unrelated documentation solely for style.

Keep affected documentation and comments accurate. Scientific figures should clearly identify quantities, units, and series.

## 4. Verification and Changes

Retain reviewed tests for consequential behavior, mathematical expectations, and regressions. Prefer independent expected results and justified numerical tolerances; revise expectations explicitly for intentional behavior changes. Use small real files or mocks according to the contract being tested.

Run relevant configured checks, expanding validation when risks or failures justify it. Resolve underlying issues rather than weakening checks; necessary exceptions should be narrowly scoped and justified.

Update affected callers, tests, and documentation together. Use `git mv`/`git rm` for tracked moves/deletions. When committing, capitalize the type prefix and first word, for example `FIX: Preserve integral semantics`.
