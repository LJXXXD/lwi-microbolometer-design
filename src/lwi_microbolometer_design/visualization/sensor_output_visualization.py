"""
Sensor output visualization functions.

This module contains functions for visualizing simulated sensor outputs
across substances and basis functions.
"""

import matplotlib.pyplot as plt
import numpy as np


def visualize_sensor_output(
    sensor_outputs: np.ndarray,
    substances_names: list[str] | None = None,
    basis_funcs_labels: list[str] | None = None,
    fontsize: int = 10,
    figure_size: tuple[int, int] = (8, 6),
) -> None:
    """
    Visualize sensor outputs as curves for different substances.

    Parameters
    ----------
    sensor_outputs : np.ndarray
        Sensor output values, shape (m, n) where m is number of basis functions
        and n is number of substances
    substances_names : List[str], optional
        Names of substances (columns of sensor_outputs)
    basis_funcs_labels : List[str], optional
        Labels for basis functions (rows of sensor_outputs)
    fontsize : int, optional
        Font size for text in the plot (default: 10)
    figure_size : Tuple[int, int], optional
        Figure size (width, height) in inches (default: (8, 6))
    """
    m, n = sensor_outputs.shape  # m = number of basis functions, n = number of substances

    # X-axis values for the basis functions
    x = np.arange(1, m + 1)

    # Create the plot
    plt.figure(figsize=figure_size)

    # Plot each substance's sensor output as a curve
    for i in range(n):
        label = substances_names[i] if substances_names is not None else f'Substance {i + 1}'
        plt.plot(x, sensor_outputs[:, i], marker='o', label=label, linewidth=2)

    # Add labels, title, and legend
    plt.xlabel('Basis Function Index', fontsize=fontsize)
    plt.ylabel('Sensor Output Values (Volt)', fontsize=fontsize)
    plt.title('Sensor Output Comparison', fontsize=fontsize + 2, fontweight='bold')
    plt.legend(loc='best', fontsize=fontsize)

    # Optionally label x-ticks with basis function labels
    if basis_funcs_labels is not None:
        plt.xticks(ticks=x, labels=basis_funcs_labels, fontsize=fontsize)
    else:
        plt.xticks(ticks=x, fontsize=fontsize)

    # Show grid for better readability
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
