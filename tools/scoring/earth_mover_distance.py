import numpy as np
from pyemd import emd

def earth_mover_distance(vector1, vector2):
    """
    Computes the Earth Mover's Distance (EMD) between two vectors.

    Parameters:
    - vector1 (1D array): The first vector representing a distribution.
    - vector2 (1D array): The second vector representing a distribution.

    Returns:
    - distance (float): The Earth Mover's Distance between the two vectors.
    """
    # Ensure input vectors are NumPy arrays and normalized to represent distributions
    vector1 = np.array(vector1, dtype=np.float64)
    vector2 = np.array(vector2, dtype=np.float64)

    # Normalize the vectors to ensure they sum to 1
    if np.sum(vector1) == 0 or np.sum(vector2) == 0:
        raise ValueError("Input vectors must represent valid distributions and sum to a positive value.")

    vector1 /= np.sum(vector1)
    vector2 /= np.sum(vector2)

    # Create the distance matrix (1D indices are assumed as the ground distance metric)
    bins = len(vector1)
    distance_matrix = np.abs(np.subtract.outer(np.arange(bins), np.arange(bins)))

    # Compute the EMD
    distance = emd(vector1, vector2, distance_matrix)
    
    return distance