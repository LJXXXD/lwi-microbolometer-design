
import numpy as np
import matplotlib.pyplot as plt


def compute_distance_matrix(matrix, distance_func):
    """
    Computes a distance matrix for the given matrix using a specified distance function.

    Parameters:
    - matrix (2D array): Input matrix (shape = (m, n)), where each column represents a vector.
    - distance_func (function): A function that computes the distance between two vectors.

    Returns:
    - distance_matrix (2D array): Distance matrix (shape = (n, n)).
    """
    n = matrix.shape[1]  # Number of columns (vectors)
    distance_matrix = np.zeros((n, n))  # Initialize distance matrix

    for i in range(n):
        for j in range(i + 1, n):
            distance = distance_func(matrix[:, i], matrix[:, j])  # Apply distance function
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Symmetric matrix

    return distance_matrix




def visualize_distance_matrix(distance_matrix, labels=None, title="Distance Matrix", cmap="viridis", fontsize=10):
    """
    Plots the distance matrix as a heatmap with cell values and x-axis labels on top.

    Parameters:
    - distance_matrix (2D array): SAM distance matrix (shape = (n, n)).
    - labels (array-like of str, optional): Labels for the substances. Defaults to None.
    - title (str, optional): Title of the plot. Defaults to "Distance Matrix".
    - cmap (str, optional): Colormap for the heatmap. Defaults to "viridis".
    - fontsize (int, optional): Font size for text in the plot. Defaults to 10.

    Returns:
    - None
    """
    # Format labels to break into multiple lines if they have more than one word
    if labels is not None:
        labels = ["\n".join(label.split()) for label in labels]  # Split by spaces and rejoin with newlines

    plt.figure(figsize=(8, 6))
    ax = plt.gca()  # Get current axis
    img = plt.imshow(distance_matrix, cmap=cmap, aspect='auto')
    cbar = plt.colorbar(img)
    cbar.set_label("SAM Distance (degrees)", fontsize=fontsize)

    # Add labels if provided
    if labels is not None and len(labels) > 0:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=fontsize)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=fontsize)
    else:
        n = distance_matrix.shape[0]
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels([f"Substance {i+1}" for i in range(n)], fontsize=fontsize)
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels([f"Substance {i+1}" for i in range(n)], fontsize=fontsize)

    # Move x-axis labels to the top
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    # Add cell values (distances)
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            plt.text(
                j, i, f"{distance_matrix[i, j]:.2f}",
                ha='center', va='center',
                fontsize=fontsize - 2, color="black" if distance_matrix[i, j] > (distance_matrix.max() / 2) else "white"
            )

    # Set title (below the heatmap now)
    plt.title(title, fontsize=fontsize + 2, pad=20)
    plt.tight_layout()
    plt.show()