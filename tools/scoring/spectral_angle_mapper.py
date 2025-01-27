
import numpy as np


def spectral_angle_mapper(vector1, vector2):
    """
    Computes the spectral angle (in degrees) between two vectors.

    Parameters:
    - vector1 (1D array): The first vector.
    - vector2 (1D array): The second vector.

    Returns:
    - angle (float): The spectral angle in degrees between the two vectors.
    """
    # Ensure input vectors are NumPy arrays
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    # Normalize the vectors
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input vectors must not have zero magnitude.")

    vector1 = vector1 / norm1
    vector2 = vector2 / norm2

    # Compute dot product and clamp to [-1, 1] to avoid numerical issues
    dot_product = np.clip(np.dot(vector1, vector2), -1.0, 1.0)

    # Compute the spectral angle
    angle = np.arccos(dot_product)

    return np.degrees(angle)