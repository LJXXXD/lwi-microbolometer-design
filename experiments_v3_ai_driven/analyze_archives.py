#!/usr/bin/env python3
"""Phase-3 additive analysis of the canonical v2_paper_2026 archives.

Does NOT modify any existing script or module. It loads the already-computed
canonical archives (outputs/v2_paper_2026/) and derives the diversity/quality
metrics the Chapter-3 prose refers to but which were not persisted as JSON:
per-archive QD-score, coverage, top-20 fitness ladder, count of elites above
55, and the number of distinct structural families (single-linkage clustering on
the permutation-invariant optimal-pairing distance, swept over the threshold tau).

Writes results/archive_metrics.json.

Run:
    cd lwi-microbolometer-design
    .venv.nosync/bin/python experiments_v3_ai_driven/analyze_archives.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from lwi_microbolometer_design.analysis.elite_clustering import (
    family_labels_from_optimal_pairing_d,
    optimal_pairing_distance_matrix,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "v2_paper_2026"


def archive_chromo_fit(path: Path):
    """Return (chromosomes [n,8], fitnesses [n]) from a MAP-Elites/CMA-ME archive dict."""
    d = pickle.load(open(path, "rb"))
    chromo, fit = [], []
    for cell in d.values():
        if isinstance(cell, dict) and "chromosome" in cell:
            chromo.append(np.asarray(cell["chromosome"], float))
            fit.append(float(cell["fitness"]))
    return np.array(chromo), np.array(fit)


def qd_metrics(fit: np.ndarray, n_cells_reachable: int = 210) -> dict:
    fit_sorted = np.sort(fit)[::-1]
    return {
        "n_filled": int(fit.size),
        "coverage_pct": round(100.0 * fit.size / n_cells_reachable, 2),
        "peak": float(fit_sorted[0]) if fit.size else None,
        "mean_archive_fitness": float(fit.mean()) if fit.size else None,
        "qd_score": float(fit[fit > 0].sum()),  # sum of positive elite fitnesses
        "top20": [round(float(x), 3) for x in fit_sorted[:20]],
        "elite20_value": float(fit_sorted[19]) if fit.size >= 20 else None,
        "n_above_55": int((fit >= 55.0).sum()),
        "n_above_58": int((fit >= 58.0).sum()),
    }


def family_sweep(chromo: np.ndarray, fit: np.ndarray, taus) -> dict:
    """Family count vs tau (permutation-invariant optimal-pairing single-linkage)."""
    if len(chromo) < 2:
        return {}
    dist = optimal_pairing_distance_matrix([c for c in chromo], params_per_group=2)
    order = np.argsort(fit)[::-1]
    dist_o = dist[np.ix_(order, order)]
    fit_o = fit[order]
    out = {}
    for tau in taus:
        labels, _ = family_labels_from_optimal_pairing_d(dist_o, fit_o, tau=float(tau))
        out[f"{tau:.2f}"] = int(len(set(labels.tolist())))
    return out


def main() -> None:
    taus = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    res: dict = {"archives": {}, "family_sweeps": {}}

    archives = {
        "map_elites": OUT / "04_map_elites" / "map_elites_archive.pkl",
        "cma_me": OUT / "06_cma_me" / "cma_me_archive.pkl",
        "minimax_map_elites": OUT / "08_minimax_optimization" / "map_elites_archive.pkl",
    }
    for name, path in archives.items():
        if not path.exists():
            continue
        chromo, fit = archive_chromo_fit(path)
        res["archives"][name] = qd_metrics(fit)
        # family structure of the top-20 elites (the chapter's diversity claim)
        order = np.argsort(fit)[::-1][:20]
        res["family_sweeps"][name + "_top20"] = family_sweep(chromo[order], fit[order], taus)
        print(
            f"[{name}] peak={res['archives'][name]['peak']:.3f} "
            f"cov={res['archives'][name]['coverage_pct']}% "
            f"QD={res['archives'][name]['qd_score']:.0f} "
            f"elite20={res['archives'][name]['elite20_value']} "
            f">55={res['archives'][name]['n_above_55']}"
        )

    # Multi-start GA champions: expected ~6 distinct families in the chapter.
    champ_path = OUT / "03_multi_start_ga" / "champions.pkl"
    if champ_path.exists():
        cd = pickle.load(open(champ_path, "rb"))
        chromo = np.asarray(cd["champions"], float)
        fit = np.asarray(cd["champion_fitness"], float)
        res["multistart_champions"] = {
            "n": int(len(fit)),
            "peak": float(fit.max()),
            "min": float(fit.min()),
            "mean": float(fit.mean()),
        }
        res["family_sweeps"]["multistart_champions"] = family_sweep(chromo, fit, taus)
        print(
            f"[multistart] n={len(fit)} peak={fit.max():.3f} "
            f"family_sweep={res['family_sweeps']['multistart_champions']}"
        )

    outp = REPO / "experiments_v3_ai_driven" / "results" / "archive_metrics.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=2))
    print(f"[done] {outp}")


if __name__ == "__main__":
    main()
