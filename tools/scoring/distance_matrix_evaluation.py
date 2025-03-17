
import numpy as np


def min_based_dissimilarity_score(distance_matrix):
    """
    Computes a dissimilarity score based purely on the minimum value of the SAM distance matrix.
    The larger the minimum value, the better the separation between the closest pair of substances.

    Parameters:
    - distance_matrix (2D array): Pairwise SAM distance matrix (shape = (n, n)).

    Returns:
    - score (float): The minimum off-diagonal distance (largest among worst-case pairs).
    """
    n = distance_matrix.shape[0]
    # Extract off-diagonal values (exclude diagonal elements)
    off_diag_values = distance_matrix[~np.eye(n, dtype=bool)]
    
    # Find the minimum distance among all pairs
    min_distance = np.min(off_diag_values)
    
    return min_distance


def mean_min_based_dissimilarity_score(distance_matrix, alpha=1.0):
    """
    Computes a dissimilarity score based on the mean and minimum values
    of a SAM distance matrix, with an adjustable penalty for minimum separability.

    Parameters:
    - distance_matrix (2D array): SAM distance matrix (shape = (n, n)).
    - alpha (float): Exponent for penalizing the minimum distance. Defaults to 1.0.

    Returns:
    - score (float): Mean-min based dissimilarity score.
    """
    n = distance_matrix.shape[0]
    # Extract off-diagonal values
    off_diag_values = distance_matrix[~np.eye(n, dtype=bool)]
    
    # Calculate mean of off-diagonal values
    mean_distance = np.mean(off_diag_values)
    
    # Calculate minimum off-diagonal value
    min_distance = np.min(off_diag_values)
    
    # Weighted combined metric: mean * (min ^ alpha)
    score = mean_distance * (min_distance ** alpha)
    return score



def group_based_dissimilarity_score(distance_matrix, groups):
    """
    Computes a dissimilarity score based on the mean inter-group distance.

    Parameters:
    - distance_matrix (2D array): SAM distance matrix (shape = (n, n)).
    - groups (list of lists): List of groups, where each group is a list of indices.

    Returns:
    - score (float): Group-based dissimilarity score (mean inter-group distance).
    """
    num_groups = len(groups)
    if num_groups < 2:
        raise ValueError("At least two groups are required to compute dissimilarity.")

    # Calculate inter-group distances
    inter_group_distances = []
    for g1 in range(num_groups):
        for g2 in range(g1 + 1, num_groups):  # Only consider distinct group pairs
            for i in groups[g1]:
                for j in groups[g2]:
                    inter_group_distances.append(distance_matrix[i, j])

    # Calculate mean inter-group distance
    mean_inter_group_distance = np.mean(inter_group_distances)

    return mean_inter_group_distance



def weighted_mean_min_dissimilarity_score(distance_matrix, beta=0.5):
    """
    Combines mean and minimum distances with a weighting factor.
    
    Parameters:
    - distance_matrix (2D array): Pairwise distance matrix.
    - beta (float): Weight for the mean distance (0 <= beta <= 1).
    
    Returns:
    - score (float): Weighted score.
    """
    n = distance_matrix.shape[0]
    off_diag_values = distance_matrix[~np.eye(n, dtype=bool)]
    mean_distance = np.mean(off_diag_values)
    min_distance = np.min(off_diag_values)
    score = beta * mean_distance + (1 - beta) * min_distance
    return score