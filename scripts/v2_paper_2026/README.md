# v2_paper_2026 — “Let It Rip” execution suite

Run all commands from the **repository root** with dependencies available. Recommended: **`uv run python`** (uses the project virtualenv and editable install).

Outputs go to `outputs/v2_paper_2026/<step_dir>/` (pickles, PNGs, `run_summary.json`).

## Progress bars (ETA and rate)

Each script prints a visible **step banner**, then **tqdm** progress with **ETA** and throughput (**`gen/s`**, **`it/s`**, **`eval/s`**, **`elite/s`**, or **`run/s`** as appropriate). Steps 01–02 hook PyGAD `on_generation`; 04–06 and 08 use MAP-Elites / CMA-ME loops in `src`; 03 and 05 wrap process-pool completion; 07 uses the robustness elite loop.

## CPU and scheduling

Running **all eight jobs at once** will oversubscribe the CPU (especially step 03 multi-start and step 05 HC polish pools). Prefer **waves**:

1. **Wave A (parallel):** 01, 02, 03 (default pool = `min(runs, min(CPU, cap))`; override with `--max-workers`), 06, 08
2. **Wave B:** 04 (heavy MAP-Elites)
3. **Wave C:** 05 (includes another full MAP-Elites + HC pool)
4. **Wave D:** 07 **after** 06 finishes (needs `06_cma_me/cma_me_archive.pkl`)

## Environment overrides

| Variable | Meaning |
|----------|---------|
| `PRES_GA_GENERATIONS` | Step 01 (standard GA) generations |
| `PRES_GA_POP` | Step 01 population |
| `PRES_GA_PARENTS_FRACTION_STANDARD` | Step 01 mating pool as fraction of population |
| `PRES_GA_GENERATIONS_ADVANCED` | Step 02 (niching GA) generations |
| `PRES_GA_POP_ADVANCED` | Step 02 population |
| `PRES_MULTI_START_RUNS` | Step 03 number of independent GAs |
| `PRES_MULTI_START_WORKERS_CAP` | Upper bound on step 03 default pool (`min(CPU, cap)`); default `200` |
| `PRES_MULTI_START_WORKERS` | Step 03 explicit process pool size (overrides default) |
| `PRES_HC_WORKERS_CAP` | Upper bound on step 05 HC default pool; default `49` (under 50 cores) |
| `PRES_HC_WORKERS` | Step 05 explicit HC pool size |
| `PRES_MAP_ELITES_ITERS` | Steps 04–05 MAP-Elites iterations |
| `PRES_CMA_ME_EVALS` | Step 06 total fitness evaluations |
| `PRES_MINIMAX_EVALS` | Step 08 total evaluations (each ×25 scenes) |

## Launch commands

```bash
# Smoke test (tiny budgets)
uv run python scripts/v2_paper_2026/run_01_standard_ga.py --quick
uv run python scripts/v2_paper_2026/run_02_advanced_ga_niching.py --quick
uv run python scripts/v2_paper_2026/run_03_multi_start_ga.py --quick --max-workers 2
uv run python scripts/v2_paper_2026/run_04_map_elites.py --quick
uv run python scripts/v2_paper_2026/run_05_map_elites_hc.py --quick --hc-workers 2
uv run python scripts/v2_paper_2026/run_06_cma_me.py --quick
uv run python scripts/v2_paper_2026/run_07_robustness_test.py --quick
uv run python scripts/v2_paper_2026/run_08_minimax_optimization.py --quick

# Production-style (default budgets in _budgets.py / env)
uv run python scripts/v2_paper_2026/run_01_standard_ga.py
uv run python scripts/v2_paper_2026/run_02_advanced_ga_niching.py
uv run python scripts/v2_paper_2026/run_03_multi_start_ga.py
uv run python scripts/v2_paper_2026/run_04_map_elites.py
uv run python scripts/v2_paper_2026/run_05_map_elites_hc.py
uv run python scripts/v2_paper_2026/run_06_cma_me.py
uv run python scripts/v2_paper_2026/run_07_robustness_test.py
uv run python scripts/v2_paper_2026/run_08_minimax_optimization.py
```

Step 07 accepts `--archive PATH` to override the default archive from step 06.

Step 08 uses `experiments/v2_paper_2026_minimax.yaml` (25-scene grid, `robustness_aggregation: min`).
