"""Visualisation utilities for MAP-Elites archives and polished results."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .archive import extract_features, reachable_cell_count


def plot_map_elites_heatmap(
    archive: dict[tuple[int, int], dict[str, Any]],
    grid_resolution: int,
    mu_range: tuple[float, float],
    output_path: Path,
) -> None:
    """Render the MAP-Elites archive as a fitness heatmap.

    Parameters
    ----------
    archive : dict
        ``(x_bin, y_bin) -> individual dict``.
    grid_resolution : int
        Bins per dimension.
    mu_range : tuple[float, float]
        (min, max) for both axes.
    output_path : Path
        Where to save the figure.
    """
    mu_min, mu_max = mu_range
    fitness_grid = np.full((grid_resolution, grid_resolution), np.nan)
    for (x_bin, y_bin), individual in archive.items():
        fitness_grid[x_bin, y_bin] = individual["fitness"]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(
        fitness_grid.T,
        origin="lower",
        extent=[mu_min, mu_max, mu_min, mu_max],
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_xlabel("Smallest \u03bc (\u00b5m)", fontsize=14)
    ax.set_ylabel("Second Smallest \u03bc (\u00b5m)", fontsize=14)
    ax.set_title(
        f"MAP-Elites Fitness Landscape\n"
        f"Reachable Coverage: {len(archive)}/{reachable_cell_count(grid_resolution)} cells",
        fontsize=16,
        fontweight="bold",
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Fitness", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"      Saved heatmap to: {output_path}")
    plt.close()


def plot_top_elites(
    archive: dict[tuple[int, int], dict[str, Any]],
    wavelengths: np.ndarray,
    parameters_to_curves: Callable[..., np.ndarray],
    output_path: Path,
    top_n: int = 10,
    title_prefix: str = "MAP-Elites Raw",
) -> None:
    """Plot the top-*N* elite response curves from the archive.

    Parameters
    ----------
    archive : dict
        ``(x_bin, y_bin) -> individual dict``.
    wavelengths : np.ndarray
        Wavelength axis.
    parameters_to_curves : callable
        ``f(params, wavelengths) -> curves``.
    output_path : Path
        Where to save the figure.
    top_n : int
        How many elites to show.
    title_prefix : str
        Title label.
    """
    all_individuals = sorted(archive.values(), key=lambda x: x["fitness"], reverse=True)
    top_elites = all_individuals[:top_n]

    plt.figure(figsize=(14, 10))
    num_individuals = len(top_elites)
    offset_step = 0.2
    all_curve_maxes: list[float] = []

    for i, individual in enumerate(top_elites):
        chromosome = individual["chromosome"]
        gaussian_params = [(chromosome[j], chromosome[j + 1]) for j in range(0, len(chromosome), 2)]
        basis_functions = parameters_to_curves(gaussian_params, wavelengths)
        vertical_offset = (num_individuals - 1 - i) * offset_step

        for basis_func in basis_functions.T:
            scaled_basis = basis_func * 0.3
            all_curve_maxes.append(float(np.max(scaled_basis + vertical_offset)))
            plt.plot(
                wavelengths,
                scaled_basis + vertical_offset,
                color="red",
                alpha=0.7,
                linewidth=1.5,
            )

    y_max = max(all_curve_maxes) + 0.5 if all_curve_maxes else 2.0
    plt.ylim(-0.1, y_max)
    plt.xlabel("Wavelength (\u00b5m)", fontsize=14)
    plt.ylabel("Absorptivity (Offset Applied)", fontsize=14)
    plt.title(
        f"{title_prefix}: Top {num_individuals} Elite Solutions",
        fontsize=16,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)

    for i, individual in enumerate(top_elites):
        mu_1, mu_2 = extract_features(individual["chromosome"])
        text_y_pos = 0.98 - (i * (0.95 / num_individuals))
        plt.text(
            0.02,
            text_y_pos,
            f"Rank {i + 1}: F={individual['fitness']:.2f} "
            f"(\u03bc\u2081={mu_1:.1f}, \u03bc\u2082={mu_2:.1f})",
            transform=plt.gca().transAxes,
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
            verticalalignment="top",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"      Saved top elites plot to: {output_path}")
    plt.close()


def plot_polished_elites(
    initial_elites: list[dict[str, Any]],
    polished_elites: list[dict[str, Any]],
    wavelengths: np.ndarray,
    parameters_to_curves: Callable[..., np.ndarray],
    output_path: Path,
    top_n: int = 20,
    title_prefix: str = "MAP-Elites Polished",
) -> None:
    """Plot polished top-*N* elites with before/after fitness labels.

    Parameters
    ----------
    initial_elites : list[dict]
        Pre-polish elite dicts (parallel to *polished_elites*).
    polished_elites : list[dict]
        Post-polish result dicts.
    wavelengths : np.ndarray
        Wavelength axis.
    parameters_to_curves : callable
        ``f(params, wavelengths) -> curves``.
    output_path : Path
        Where to save the figure.
    top_n : int
        How many to show.
    title_prefix : str
        Title label.
    """
    sorted_indices = np.argsort([e["polished_fitness"] for e in polished_elites])[::-1]
    sorted_polished = [polished_elites[i] for i in sorted_indices]
    sorted_initial = [initial_elites[i] for i in sorted_indices]

    top_n_actual = min(top_n, len(sorted_polished))
    sorted_polished = sorted_polished[:top_n_actual]
    sorted_initial = sorted_initial[:top_n_actual]

    plt.figure(figsize=(14, 10))
    num_individuals = len(sorted_polished)
    offset_step = 0.2
    all_curve_maxes: list[float] = []

    for i, (polished, _initial) in enumerate(zip(sorted_polished, sorted_initial, strict=False)):
        chromosome = polished["polished_chromosome"]
        gaussian_params = [(chromosome[j], chromosome[j + 1]) for j in range(0, len(chromosome), 2)]
        basis_functions = parameters_to_curves(gaussian_params, wavelengths)
        vertical_offset = (num_individuals - 1 - i) * offset_step

        for basis_func in basis_functions.T:
            scaled_basis = basis_func * 0.3
            all_curve_maxes.append(float(np.max(scaled_basis + vertical_offset)))
            plt.plot(
                wavelengths,
                scaled_basis + vertical_offset,
                color="red",
                alpha=0.7,
                linewidth=1.5,
            )

    y_max = max(all_curve_maxes) + 0.5 if all_curve_maxes else 2.0
    plt.ylim(-0.1, y_max)
    plt.xlabel("Wavelength (\u00b5m)", fontsize=14)
    plt.ylabel("Absorptivity (Offset Applied)", fontsize=14)
    plt.title(
        f"{title_prefix}: Top {top_n_actual} Refined Solutions",
        fontsize=16,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)

    for i, (polished, initial) in enumerate(zip(sorted_polished, sorted_initial, strict=False)):
        text_y_pos = 0.98 - (i * (0.95 / num_individuals))
        plt.text(
            0.02,
            text_y_pos,
            f"Rank {i + 1}: {initial['fitness']:.2f} \u2192 {polished['polished_fitness']:.2f} "
            f"(+{polished['fitness_gain']:.2f})",
            transform=plt.gca().transAxes,
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
            verticalalignment="top",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"      Saved polished elites plot to: {output_path}")
    plt.close()


def plot_cma_me_progress(
    history: dict[str, list[float]],
    output_path: Path,
) -> None:
    """Plot CMA-ME convergence curves (coverage and best fitness).

    Parameters
    ----------
    history : dict[str, list[float]]
        Must contain keys ``"evals"``, ``"archive_size"``,
        ``"best_fitness"``, and ``"coverage_pct"`` as returned by
        :func:`~.cma_me.run_cma_me` metadata.
    output_path : Path
        Where to save the figure.
    """
    evals = history["evals"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(evals, history["coverage_pct"], color="steelblue", linewidth=2)
    ax1.set_ylabel("Reachable Coverage (%)", fontsize=12)
    ax1.set_title("CMA-ME Convergence", fontsize=16, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.plot(evals, history["best_fitness"], color="firebrick", linewidth=2)
    ax2.set_xlabel("Fitness Evaluations", fontsize=12)
    ax2.set_ylabel("Best Fitness", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"      Saved CMA-ME progress plot to: {output_path}")
    plt.close()
