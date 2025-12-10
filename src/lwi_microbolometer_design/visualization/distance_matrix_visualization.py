"""
Distance matrix visualization functions for microbolometer sensor analysis.

This module provides specialized visualization functions for distance matrices,
which are commonly used in spectral analysis and sensor optimization. The functions
are optimized for symmetric distance matrices with identical x/y labels.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def visualize_distance_matrix(
    distance_matrix: np.ndarray,
    labels: list[str] | None = None,
    title: str = 'Distance Matrix',
    cmap: str = 'viridis',
    fontsize: int = 10,
    colorbar_min: float | None = None,
    colorbar_max: float | None = None,
    figure_size: tuple[int, int] = (8, 6),
    show_values: bool = True,
    symmetric: bool = True,
) -> None:
    """
    Visualize a distance matrix as a heatmap with customizable options.

    This is the main function for visualizing distance matrices with comprehensive
    customization options for labels, colors, and display properties.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Distance matrix, shape (n, n) where n is number of items
    labels : List[str], optional
        Labels for the items (default: None, uses generic labels)
    title : str, optional
        Title of the plot (default: "Distance Matrix")
    cmap : str, optional
        Colormap for the heatmap (default: "viridis")
    fontsize : int, optional
        Font size for text in the plot (default: 10)
    colorbar_min : float, optional
        Minimum value for the color scale (default: None for auto)
    colorbar_max : float, optional
        Maximum value for the color scale (default: None for auto)
    figure_size : Tuple[int, int], optional
        Figure size (width, height) in inches (default: (8, 6))
    show_values : bool, optional
        Whether to display numerical values in cells (default: True)
    symmetric : bool, optional
        Whether the matrix is symmetric (affects display optimization) (default: True)

    Returns
    -------
    None
        Displays the plot
    """
    plt.figure(figsize=figure_size)
    ax = plt.gca()

    # Define color range limits
    vmin = colorbar_min if colorbar_min is not None else np.min(distance_matrix)
    vmax = colorbar_max if colorbar_max is not None else np.max(distance_matrix)

    # Create the heatmap
    if symmetric and show_values:
        # Use seaborn for symmetric matrices with annotations
        sns.heatmap(
            distance_matrix,
            annot=True,
            fmt='.2f',
            cmap=cmap,
            square=True,
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Distance'},
            vmin=vmin,
            vmax=vmax,
        )
    else:
        # Use matplotlib for more control
        img = plt.imshow(distance_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(img)
        cbar.set_label('Distance', fontsize=fontsize)

        # Add labels if provided
        if labels is not None and len(labels) > 0:
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, fontsize=fontsize)
            ax.set_yticks(np.arange(len(labels)))
            ax.set_yticklabels(labels, fontsize=fontsize)
        else:
            n = distance_matrix.shape[0]
            ax.set_xticks(np.arange(n))
            ax.set_xticklabels([f'Item {i + 1}' for i in range(n)], fontsize=fontsize)
            ax.set_yticks(np.arange(n))
            ax.set_yticklabels([f'Item {i + 1}' for i in range(n)], fontsize=fontsize)

        # Move x-axis labels to the top for better readability
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()

        # Add cell values if requested
        if show_values:
            for i in range(distance_matrix.shape[0]):
                for j in range(distance_matrix.shape[1]):
                    plt.text(
                        j,
                        i,
                        f'{distance_matrix[i, j]:.2f}',
                        ha='center',
                        va='center',
                        fontsize=fontsize - 2,
                        color='black' if distance_matrix[i, j] > (vmax / 2) else 'white',
                    )

    plt.title(title, fontsize=fontsize + 2, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualize_distance_matrix_simple(
    distance_matrix: np.ndarray,
    title: str = 'Distance Matrix',
    figure_size: tuple[int, int] = (8, 6),
) -> None:
    """
    Visualize a distance matrix as a heatmap.

    This is a simplified version for quick visualization without customization.
    Best for large matrices or when you just need a quick view.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Distance matrix to visualize
    title : str, optional
        Title for the plot (default: "Distance Matrix")
    figure_size : Tuple[int, int], optional
        Figure size (width, height) in inches (default: (8, 6))
    """
    plt.figure(figsize=figure_size)
    plt.imshow(distance_matrix, cmap='viridis', origin='upper')
    plt.colorbar(label='Distance')
    plt.title(title)
    plt.xlabel('Indices')
    plt.ylabel('Indices')
    plt.tight_layout()
    plt.show()


# Backward compatibility aliases
visualize_distance_matrix_large = visualize_distance_matrix_simple
