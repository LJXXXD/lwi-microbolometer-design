# Phase-3 LWI analysis provenance

The analysis harness in `experiments_v3_ai_driven/` reads the stored archives
in `outputs/v2_paper_2026/` that back the Chapter-3 prose (see the Phase-2 note
`00_Phase2_README.md` R1). `analyze_archives.py` reports archive quality and
structural diversity. These are summaries of existing runs, not new
optimization runs or evidence of measured identification performance.

## Archive metrics (`results/archive_metrics.json`)

| Archive | peak | coverage | QD-score | elite-20 | #≥55 |
|---|---|---|---|---|---|
| MAP-Elites (nominal) | **58.794** | 100% (210/210) | 7612 | 49.95 | **7** |
| CMA-ME | **58.776** | 100% | 6613 | **48.28** | 4 |
| Minimax MAP-Elites | **56.687** | 100% | 6718 | 46.13 | 1 |
| Multi-start GA champions | **59.540** | — | — | — | — |

Cross-check against the chapter text:
- MAP-Elites peak 58.79 ✓; "7 distinct families scoring above 55.0" ✓ (#≥55 = 7).
- CMA-ME peak 58.78 ✓; "dropping to 48.28 by the 20th elite" ✓ (elite-20 = 48.28, exact).
- Minimax peak/floor 56.69 ✓.
- Multi-start peak 59.54 ✓.

**QD-score** is the sum of positive elite fitnesses over filled cells:
MAP-Elites (7612) > Minimax (6718) > CMA-ME (6613) in these stored archives.
This ranking describes this run set; it does not establish general algorithm
superiority. Historical budget labels are not a measurement of actual
fitness-evaluation counts.

## Family-count sweep (permutation-invariant optimal-pairing single-linkage)

Distinct structural families vs threshold tau:

| Set | tau=0.5 | 1.0 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|
| Multi-start champions | **6** | 5 | 5 | 4 | 4 |
| MAP-Elites top-20 | 17 | 15 | 14 | 8 | 6 |
| CMA-ME top-20 | 14 | 10 | 10 | 8 | 6 |
| Minimax top-20 | 18 | 16 | 13 | 10 | 6 |

- The chapter's **"6 distinct design families"** for the Multi-start GA is
  reproduced at **tau = 0.5** — i.e. the chapter's implicit clustering threshold is
  ~0.5 in the 8-D optimal-pairing metric. Report tau explicitly in the methods.
- At the same tau, MAP-Elites top-20 yields **17** families vs CMA-ME's 14,
  quantifying MAP-Elites' superior structural diversity retention (the chapter's
  "over 20 distinct design families" refers to the full 210-cell archive, of which
  the top-20 alone already span 17 families).

## Reproduce
```
cd lwi-microbolometer-design
.venv.nosync/bin/python experiments_v3_ai_driven/analyze_archives.py
```

## Pending off-grid validation

Grid-density robustness validation evaluates the Minimax elites on a denser
or off-grid scene set using `MinDissimilarityFitnessEvaluator`. It is specified
in `Phase2_Architecture_Design/01_Ch3_LWI_Experiment_and_Code_Architecture.md`
(A5). No off-grid result is recorded here.

## Stored 2026-07-13 in-grid robustness (`analyze_minimax_robustness.py`)

`analyze_minimax_robustness.py` evaluates the step-08 Minimax MAP-Elites archive
with the same evaluator and default 25-scene grid as `run_07`, retaining the
top 64 archive elites and aligning the nominal baseline to the training scene.
The stored JSON is `results/minimax_robustness.json`; its nominal comparison
comes from `outputs/v2_paper_2026/07_robustness_test/robustness_summary.json`.
The grid uses temperatures 273.15–313.15 K, distance ratios
0.05/0.08/0.11/0.15/0.20 and refractive index 1.0. This is in-grid evaluation.

| Archive (top-64, 25-scene grid) | mean worst-case retention | worst-elite retention | mean CV |
|---|---|---|---|
| Nominal MAP-Elites+HC (step 05) | 91.70% | **71.6%** (bad tail) | 0.035 |
| **Minimax MAP-Elites (step 08)** | **97.51%** | **92.45%** | **0.013** |

The Minimax archive has higher relative retention, especially in its least
retained elite, and lower mean CV. Relative retention is distinct from absolute
SAM: mean worst-case SAM is 44.24 degrees for Minimax and 44.59 degrees for the
nominal archive in the stored summaries. These numbers do not show an increase
in mean worst-case SAM.

The supplied atmospheric transmission is identically one, so the distance
sweep does not test wavelength-dependent attenuation. The stored results
therefore characterize the configured spectral surrogate and temperature
grid, not general atmospheric or detector robustness (see
`docs/THEORY_AND_IMPLEMENTATION.md`, Section 19).

The stored JSON and optimizer archives retain their historical values. A
current-code recheck should use a separate output path; future runs must record
their code revision, input provenance, effective grid and top-N selection.

Reproduce:
```
cd lwi-microbolometer-design
.venv.nosync/bin/python experiments_v3_ai_driven/analyze_minimax_robustness.py --output .ai-tmp/minimax-recheck/summary.json
```
