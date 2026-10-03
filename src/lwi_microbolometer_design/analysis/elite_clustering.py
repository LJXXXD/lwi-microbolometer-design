"""
Family labels for top elites from optimal-pairing distance **D** (single-linkage at τ).

:func:`family_labels_from_optimal_pairing_d` uses **only** graph connectivity: link *i*–*j*
if ``D[i,j] <= τ``; families are **connected components**. This is stable for nearly identical
high-fitness genotypes, unlike *k*-means / spectral on small *n* which can spuriously split them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lwi_microbolometer_design.analysis.distance_matrix import compute_distance_matrix


def _reorder_families_by_fitness(labels: np.ndarray, fitnesses: Sequence[float]) -> np.ndarray:
    """Permute family ids so family ``0`` has the highest mean fitness."""
    unique = np.unique(labels)
    means: dict[int, float] = {}
    for c in unique:
        idx = np.nonzero(labels == c)[0]
        means[int(c)] = float(np.mean([fitnesses[i] for i in idx]))
    order = sorted(unique, key=lambda c: -means[int(c)])
    remap = {int(old): new for new, old in enumerate(order)}
    return np.array([remap[int(c)] for c in labels], dtype=int)


def graph_single_linkage_from_distance_matrix(dist: np.ndarray, edge_max: float) -> np.ndarray:
    """Connect *i*–*j* if ``D[i,j] <= edge_max``; return component ids 0..k-1 (unreordered)."""
    d = np.asarray(dist, dtype=np.float64)
    n = int(d.shape[0])
    if n <= 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    thr = float(edge_max)
    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if float(d[i, j]) <= thr:
                union(i, j)

    roots = np.empty(n, dtype=np.int64)
    for k in range(n):
        roots[k] = find(k)
    order = {int(r): i for i, r in enumerate(sorted(np.unique(roots).tolist()))}
    return np.array([order[int(roots[i])] for i in range(n)], dtype=int)


def optimal_pairing_distance_matrix(
    items: list[np.ndarray],
    *,
    params_per_group: int = 2,
) -> np.ndarray:
    return compute_distance_matrix(
        items,
        metric="euclidean",
        use_optimal_pairing=True,
        params_per_group=int(params_per_group),
    )


def family_labels_from_optimal_pairing_d(
    dist: np.ndarray,
    fitness: Sequence[float],
    *,
    tau: float,
) -> tuple[np.ndarray, str]:
    """Single-linkage components on *D* at threshold ``τ``; then reorder families by mean fitness.

    Parameters
    ----------
    dist
        Symmetric distance matrix, shape ``(n, n)`` (Euclidean, optimal-pairing between chromosomes).
    fitness
        Length ``n``, same order as rows of ``dist`` (e.g. fitness descending in rank order).
    tau
        Edge threshold: link pairs with :math:`D_{ij} \\le \\tau` (same units as *D*).
    """
    d = np.asarray(dist, dtype=np.float64)
    n = d.shape[0]
    n_f = len(fitness)
    if n == 0 or n_f == 0:
        return np.array([], dtype=int), f"graph single-linkage, τ={tau:.4g}"
    if n != n_f:
        raise ValueError(f"dist and fitness size mismatch: {n} vs {n_f}")
    if n == 1:
        return np.zeros(1, dtype=int), f"graph single-linkage, τ={float(tau):.4g}"

    raw = graph_single_linkage_from_distance_matrix(d, float(tau))
    re = _reorder_families_by_fitness(np.asarray(raw, dtype=int), list(fitness))
    note = f"graph single-linkage, τ={float(tau):.4g}"
    return re, note


def graph_component_labels_from_distance_matrix(
    dist: np.ndarray, edge_distance_max: float
) -> np.ndarray:
    return graph_single_linkage_from_distance_matrix(dist, edge_distance_max)
