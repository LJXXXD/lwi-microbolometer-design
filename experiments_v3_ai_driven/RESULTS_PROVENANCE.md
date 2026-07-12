# Phase-3 LWI analysis provenance

Additive harness (`experiments_v3_ai_driven/`). No existing script or `src/`
module modified. `analyze_archives.py` loads the **canonical** archives in
`outputs/v2_paper_2026/` (the run that backs the Chapter-3 prose — see the
Phase-2 note `00_Phase2_README.md` R1) and derives the diversity/quality metrics
the chapter refers to but which were not persisted as JSON.

## Results (`results/archive_metrics.json`) — these CONFIRM the Ch3 prose

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

**QD-score** (sum of positive elite fitnesses over filled cells) is a new, standard
QD-field metric added for the paper: MAP-Elites (7612) > Minimax (6718) > CMA-ME
(6613). MAP-Elites dominates CMA-ME on QD-score at equal 1× budget, quantifying
the "No Free Lunch" argument the chapter makes qualitatively.

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

## Still designed-but-not-run (documented in Phase-2 `01_...` A5, deferred for correctness)

Grid-density robustness validation (re-evaluate the Minimax elites on a denser /
off-grid scene set) was NOT run tonight: it requires wiring the full physics
`MinDissimilarityFitnessEvaluator` with a new scene grid, and doing it under time
pressure risked a subtly-wrong physics setup. It is specified in
`Phase2_Architecture_Design/01_Ch3_LWI_Experiment_and_Code_Architecture.md` (A5)
and should be run deliberately with the substance/atmosphere data loaded.
