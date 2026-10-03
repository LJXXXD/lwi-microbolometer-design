"""Multi-start champion overlay (presentation styling)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

import _plotting
from lwi_microbolometer_design import gaussian_parameters_to_unit_amplitude_curves
from lwi_microbolometer_design.map_elites import set_tight_ylim_stacked_spectra


def plot_multi_start_champions(
    champions: np.ndarray,
    champion_fitness: np.ndarray,
    wavelengths: np.ndarray,
    mean_pairwise_distance: float,
    output_path: Path,
    num_runs: int,
) -> None:
    """Overlay basis curves for all champions (red, offset), sorted by fitness."""
    _plotting.apply_presentation_style()
    sorted_indices = np.argsort(champion_fitness)[::-1]
    sorted_champions = champions[sorted_indices]
    sorted_fitness = champion_fitness[sorted_indices]

    fig, ax = plt.subplots(figsize=(14, 8))
    num_individuals = len(sorted_champions)
    y_mins: list[float] = []
    y_maxs: list[float] = []

    for i, (chromosome, fitness) in enumerate(zip(sorted_champions, sorted_fitness, strict=False)):
        gaussian_params = [(chromosome[j], chromosome[j + 1]) for j in range(0, len(chromosome), 2)]
        basis_functions = gaussian_parameters_to_unit_amplitude_curves(gaussian_params, wavelengths)
        vertical_offset = (num_individuals - 1 - i) * 0.1
        for _j, basis_func in enumerate(basis_functions.T):
            scaled_basis = basis_func * 0.3
            stacked = scaled_basis + vertical_offset
            y_mins.append(float(np.min(stacked)))
            y_maxs.append(float(np.max(stacked)))
            ax.plot(
                wavelengths,
                stacked,
                color="red",
                alpha=0.3,
                linewidth=1.5,
            )

    if y_mins and y_maxs:
        set_tight_ylim_stacked_spectra(ax, min(y_mins), max(y_maxs))

    ax.set_xlabel("Wavelength (µm)", fontsize=12)
    ax.set_ylabel("Spectral Responsivity (scaled, offset applied)", fontsize=12)
    ax.set_title(
        f"Multi-start GA: {num_runs} independent runs — champions overlay\n"
        f"Mean pairwise chrom. distance: {mean_pairwise_distance:.4f} | "
        f"Fitness max {np.max(champion_fitness):.2f} | mean {np.mean(champion_fitness):.2f}",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    trans_axes_x_data_y = blended_transform_factory(ax.transAxes, ax.transData)
    num_annotations = min(20, len(sorted_fitness))
    for i in range(num_annotations):
        fit = sorted_fitness[i]
        y_base = (num_individuals - 1 - i) * 0.1
        ax.text(
            0.01,
            y_base,
            f"Rank {i + 1}: {fit:.2f}",
            transform=trans_axes_x_data_y,
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
            verticalalignment="baseline",
            horizontalalignment="left",
            clip_on=False,
        )

    stats = (
        f"Runs: {num_runs}\n"
        f"Best: {np.max(champion_fitness):.2f}\n"
        f"Mean: {np.mean(champion_fitness):.2f}\n"
        f"Diversity: {mean_pairwise_distance:.4f}"
    )
    ax.text(
        0.98,
        0.02,
        stats,
        transform=ax.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.9},
        verticalalignment="bottom",
        horizontalalignment="right",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
