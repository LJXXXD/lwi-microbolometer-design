
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




import matplotlib.pyplot as plt
import numpy as np

def visualize_distance_matrix(distance_matrix, labels=None, title="Distance Matrix", cmap="viridis", 
                              fontsize=10, colorbar_min=None, colorbar_max=None, figure_size=(8, 6)):
    """
    Plots the distance matrix as a heatmap with cell values and x-axis labels on top.
    
    Parameters:
    - distance_matrix (2D array): SAM distance matrix (shape = (n, n)).
    - labels (array-like of str, optional): Labels for the substances. Defaults to None.
    - title (str, optional): Title of the plot. Defaults to "Distance Matrix".
    - cmap (str, optional): Colormap for the heatmap. Defaults to "viridis".
    - fontsize (int, optional): Font size for text in the plot. Defaults to 10.
    - colorbar_min (float, optional): Minimum value for the color scale. Defaults to None (auto).
    - colorbar_max (float, optional): Maximum value for the color scale. Defaults to None (auto).

    Returns:
    - None
    """

    # Format labels to break into multiple lines if they have more than one word
    if labels is not None:
        labels = ["\n".join(label.split()) for label in labels]  # Split by spaces and rejoin with newlines

    plt.figure(figsize=figure_size)
    ax = plt.gca()  # Get current axis

    # Define color range limits
    vmin = colorbar_min if colorbar_min is not None else np.min(distance_matrix)
    vmax = colorbar_max if colorbar_max is not None else np.max(distance_matrix)

    # Plot heatmap with custom color range
    img = plt.imshow(distance_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
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
                fontsize=fontsize - 2, 
                color="black" if distance_matrix[i, j] > (vmax / 2) else "white"
            )

    # Set title (below the heatmap now)
    # plt.title(title, fontsize=fontsize + 2, pad=20)
    plt.tight_layout()
    plt.show()




def visualize_distance_matrix_large(distance_matrix, title="Distance Matrix Visualization"):
    """
    Visualizes a distance matrix as a heatmap.

    Parameters:
    - distance_matrix (2D ndarray): Pairwise distance matrix.
    - title (str): Title for the plot.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(distance_matrix, cmap='viridis', origin='upper')
    plt.colorbar(label="Distance")
    plt.title(title, fontsize=16)
    plt.xlabel("Reordered Indices", fontsize=12)
    plt.ylabel("Reordered Indices", fontsize=12)
    plt.tight_layout()
    plt.show()