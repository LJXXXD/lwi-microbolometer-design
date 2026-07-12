"""
CLODD — Clustering in Ordered Dissimilarity Data (Havens et al., 2009).

Finds an aligned crisp partition (vertical/horizontal cuts in VAT order) by
maximizing contrast and boundary sharpness on a normalized ordered matrix D*.
Typically D* comes from VAT or iVAT; iVAT often gives cleaner blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from itertools import combinations
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CLODDResult:
    """Best partition found over c = 2 .. c_max on the ordered index axis."""

    n_clusters: int
    """Number of clusters (crisp c)."""
    cuts: tuple[int, ...]
    """Length c-1: first row/column index of clusters 1 .. c-1 (0-based VAT order)."""
    objective: float
    """Maximized CLODD score E(U; D*) in [0, 1] under the paper's construction."""
    per_c_best: dict[int, float]
    """Best objective found for each c tried."""
    labels_ordered: np.ndarray
    """Shape (n,), cluster ids 0..c-1 along the VAT/iVAT ordering axis."""


# ---------------------------------------------------------------------------
# Normalization & labels
# ---------------------------------------------------------------------------


def normalize_ordered_dissimilarity(d: np.ndarray) -> np.ndarray:
    """
    Min–max scale off-diagonal entries to [0, 1]; diagonal set to 0.

    Matches the usual VAT image scaling assumption in the CLODD paper.
    """
    d = np.asarray(d, dtype=np.float64)
    n = d.shape[0]
    mask = ~np.eye(n, dtype=bool)
    vals = d[mask]
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi <= lo:
        out = np.zeros_like(d, dtype=np.float64)
        np.fill_diagonal(out, 0.0)
        return out
    out = (d - lo) / (hi - lo)
    np.fill_diagonal(out, 0.0)
    return out


def labels_from_cuts(n: int, cuts: tuple[int, ...]) -> np.ndarray:
    """Build cluster labels 0..c-1 from strictly increasing cut positions."""
    labels = np.zeros(n, dtype=np.int64)
    cid = 0
    start = 0
    for cut in cuts:
        labels[start:cut] = cid
        start = cut
        cid += 1
    labels[start:] = cid
    return labels


def cluster_sizes_from_cuts(n: int, cuts: tuple[int, ...]) -> list[int]:
    """Cardinalities n1..nc for an aligned partition."""
    boundaries = (0,) + tuple(cuts) + (n,)
    return [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]


# ---------------------------------------------------------------------------
# Objective (Havens et al., Eqs. 6–11)
# ---------------------------------------------------------------------------


def _spline_s(min_cluster: int, a: float) -> float:
    """Equation (9): s-curve on minimum cluster size, inflection tied to a = γ n."""
    x = float(min_cluster)
    if x <= 1.0:
        return 0.0
    if x >= a:
        return 1.0
    if x <= a / 2.0:
        return 2.0 * (x / a) ** 2
    return 1.0 - 2.0 * ((a - x) / a) ** 2


def esquare(d_star: np.ndarray, cuts: tuple[int, ...]) -> float:
    """Contrast: mean between-cluster dissimilarity minus mean within-cluster."""
    n = d_star.shape[0]
    labels = labels_from_cuts(n, cuts)
    between_sum = 0.0
    between_cnt = 0
    within_sum = 0.0
    within_cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            dij = float(d_star[i, j])
            if labels[i] == labels[j]:
                within_sum += dij
                within_cnt += 1
            else:
                between_sum += dij
                between_cnt += 1
    if between_cnt == 0 or within_cnt == 0:
        return 0.0
    return between_sum / between_cnt - within_sum / within_cnt


def eedge(d_star: np.ndarray, cuts: tuple[int, ...]) -> float:
    """Average horizontal gradient magnitude across interior block boundaries."""
    n = d_star.shape[0]
    c = len(cuts) + 1
    if c < 2:
        return 0.0
    sizes = cluster_sizes_from_cuts(n, cuts)
    totals: list[float] = []
    for j in range(c - 1):
        left_lo = 0 if j == 0 else sum(sizes[:j])
        left_hi = sum(sizes[: j + 1])
        right_lo = left_hi
        right_hi = sum(sizes[: j + 2])
        col_l = left_hi - 1
        col_r = right_lo
        acc = 0.0
        for i in range(left_lo, right_hi):
            acc += abs(float(d_star[i, col_l]) - float(d_star[i, col_r]))
        totals.append(acc / float(sizes[j] + sizes[j + 1]))
    return float(np.mean(totals))


def clodd_objective(
    d_star: np.ndarray,
    cuts: tuple[int, ...],
    *,
    alpha: float = 0.5,
    gamma: float,
) -> float:
    """
    Full CLODD objective E(U; D*) = S_γ(U) · (α E_sq + (1-α) E_edge).

    Parameters
    ----------
    d_star :
        Ordered dissimilarity, assumed roughly in [0, 1] (use normalize_ordered_dissimilarity).
    gamma :
        Spline set-point parameter; internally uses a = γ · n with γ ∈ (2/n, 1].
    """
    n = d_star.shape[0]
    sizes = cluster_sizes_from_cuts(n, cuts)
    min_sz = min(sizes)
    a = gamma * n
    s = _spline_s(min_sz, a)
    e_a = alpha * esquare(d_star, cuts) + (1.0 - alpha) * eedge(d_star, cuts)
    return s * e_a


def _gamma_bounds(n: int) -> tuple[float, float]:
    return (2.0 / n + 1e-9, 1.0)


def resolve_gamma(n: int, gamma: float | None) -> float:
    """Default γ = 0.05 per paper examples, clipped to (2/n, 1]."""
    lo, hi = _gamma_bounds(n)
    g = 0.05 if gamma is None else float(gamma)
    return float(np.clip(g, lo, hi))


# ---------------------------------------------------------------------------
# Search: exhaustive (small n) or PSO
# ---------------------------------------------------------------------------


def _best_for_c_exhaustive(
    d_star: np.ndarray,
    c: int,
    *,
    alpha: float,
    gamma: float,
) -> tuple[tuple[int, ...], float]:
    n = d_star.shape[0]
    best_cuts: tuple[int, ...] = tuple()
    best_e = -1.0
    for cuts in combinations(range(1, n), c - 1):
        ct = tuple(cuts)
        e = clodd_objective(d_star, ct, alpha=alpha, gamma=gamma)
        if e > best_e:
            best_e = e
            best_cuts = ct
    return best_cuts, best_e


def _pso_best_for_c(
    d_star: np.ndarray,
    c: int,
    *,
    alpha: float,
    gamma: float,
    n_particles: int,
    max_iter: int,
    seed: int | None,
    k_inertia: float,
    a_local: float,
    a_global: float,
) -> tuple[tuple[int, ...], float]:
    n = d_star.shape[0]
    rng = np.random.default_rng(seed)
    dim = c - 1

    def random_valid() -> np.ndarray:
        pts = rng.choice(range(1, n), size=dim, replace=False)
        return np.sort(pts)

    particles = np.array([random_valid() for _ in range(n_particles)], dtype=np.float64)
    velocities = rng.uniform(-1.0, 1.0, size=(n_particles, dim))
    p_best = particles.copy()
    p_best_e = np.array(
        [clodd_objective(d_star, tuple(map(int, p)), alpha=alpha, gamma=gamma) for p in p_best]
    )
    g_idx = int(np.argmax(p_best_e))
    g_best = p_best[g_idx].copy()
    g_best_e = float(p_best_e[g_idx])

    for _ in range(max_iter):
        for i in range(n_particles):
            r1, r2 = rng.random(dim), rng.random(dim)
            velocities[i] = (
                k_inertia * velocities[i]
                + a_local * r1 * (p_best[i] - particles[i])
                + a_global * r2 * (g_best - particles[i])
            )
            cand = np.round(particles[i] + velocities[i]).astype(np.int64)
            cand = np.clip(cand, 1, n - 1)
            cand = np.sort(cand)
            if len(np.unique(cand)) < dim:
                continue
            e = clodd_objective(d_star, tuple(map(int, cand)), alpha=alpha, gamma=gamma)
            particles[i] = cand.astype(np.float64)
            if e > p_best_e[i]:
                p_best_e[i] = e
                p_best[i] = cand.astype(np.float64)
                if e > g_best_e:
                    g_best_e = e
                    g_best = cand.astype(np.float64)

    return tuple(map(int, g_best)), g_best_e


def _exhaustive_work_estimate(n: int, c_max: int) -> int:
    return sum(math.comb(n - 1, c - 1) for c in range(2, min(c_max, n) + 1))


def clodd_partition(
    d_ordered: np.ndarray,
    *,
    c_max: int | None = None,
    alpha: float = 0.5,
    gamma: float | None = None,
    mode: Literal["auto", "exhaustive", "pso"] = "auto",
    exhaustive_max_work: int = 200_000,
    n_particles: int = 20,
    max_iter: int = 400,
    seed: int | None = None,
    k_inertia: float = 0.75,
    a_local: float = 2.0,
    a_global: float = 2.0,
) -> CLODDResult:
    """
    Partition objects in VAT/iVAT order by maximizing CLODD's E.

    Parameters
    ----------
    d_ordered :
        Square ordered dissimilarity (e.g. VAT or iVAT matrix). Should be finite,
        symmetric, zero diagonal. It is **not** normalized inside this function;
        pass the same matrix you will visualize, after ``normalize_ordered_dissimilarity``
        if you want scores comparable to the paper.
    c_max :
        Largest c to try (default ``n - 1``).
    mode :
        ``exhaustive`` for guaranteed optimum per c (small n), ``pso`` for heuristic,
        ``auto`` picks exhaustive when the work estimate is below ``exhaustive_max_work``.
    """
    d_star = np.asarray(d_ordered, dtype=np.float64)
    n = d_star.shape[0]
    if n < 3:
        raise ValueError("CLODD requires at least 3 objects (n >= 3).")
    if c_max is None:
        c_max = n - 1
    c_max = int(min(max(2, c_max), n - 1))
    gamma_r = resolve_gamma(n, gamma)

    per_c: dict[int, float] = {}
    best_global_cuts: tuple[int, ...] = ()
    best_global_e = -1.0
    best_c = 2

    use_exhaustive = mode == "exhaustive" or (
        mode == "auto" and _exhaustive_work_estimate(n, c_max) <= exhaustive_max_work
    )

    for c in range(2, c_max + 1):
        if use_exhaustive:
            cuts, e = _best_for_c_exhaustive(d_star, c, alpha=alpha, gamma=gamma_r)
        else:
            cuts, e = _pso_best_for_c(
                d_star,
                c,
                alpha=alpha,
                gamma=gamma_r,
                n_particles=n_particles,
                max_iter=max_iter,
                seed=seed if seed is None else seed + c,
                k_inertia=k_inertia,
                a_local=a_local,
                a_global=a_global,
            )
        per_c[c] = e
        if e > best_global_e:
            best_global_e = e
            best_global_cuts = cuts
            best_c = c

    labels = labels_from_cuts(n, best_global_cuts)
    return CLODDResult(
        n_clusters=best_c,
        cuts=best_global_cuts,
        objective=float(best_global_e),
        per_c_best=per_c,
        labels_ordered=labels,
    )


def map_labels_to_original_order(labels_ordered: np.ndarray, reorder: list[int]) -> np.ndarray:
    """Map cluster labels from VAT order back to original row/column indices."""
    n = len(reorder)
    r = np.array(reorder, dtype=np.int64)
    inv = np.empty(n, dtype=np.int64)
    inv[r] = np.arange(n)
    return labels_ordered[inv].copy()


def add_clodd_partition_lines(
    ax,
    cuts: tuple[int, ...],
    *,
    n: int,
    color: str = "red",
    linewidth: float = 1.8,
    alpha: float = 0.95,
) -> None:
    """Draw CLODD boundaries on an ``imshow`` of an n×n ordered matrix (index origin top-left)."""
    for cut in cuts:
        ax.axvline(cut - 0.5, color=color, linewidth=linewidth, alpha=alpha)
        ax.axhline(cut - 0.5, color=color, linewidth=linewidth, alpha=alpha)
