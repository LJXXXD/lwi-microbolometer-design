"""Distance metrics for spectral analysis and substance discrimination."""

import numpy as np

# Below this norm, treat a vector as zero-magnitude so SAM stays finite during optimization.
_SAM_NORM_EPS = 1e-15


def spectral_angle_mapper(vector1: np.ndarray | list, vector2: np.ndarray | list) -> float:
    """
    Compute the spectral angle (in degrees) between two vectors.

    The Spectral Angle Mapper (SAM) is a widely used metric in remote sensing
    and spectral analysis for measuring the similarity between two spectral signatures.
    It measures the angle between two vectors in n-dimensional space, where
    smaller angles indicate greater similarity.

    Parameters
    ----------
    vector1 : np.ndarray or list
        The first vector (spectral signature)
    vector2 : np.ndarray or list
        The second vector (spectral signature)

    Returns
    -------
    float
        The spectral angle in degrees between the two vectors (typically 0–90° for
        nonzero norms). If either vector has norm below ``1e-15``, returns ``0.0``
        so fitness evaluation stays finite (degenerate fingerprints read as
        maximally similar).

    Notes
    -----
    The spectral angle is calculated as:
    θ = arccos((v1 · v2) / (||v1|| ||v2||))

    Where:
    - v1, v2 are the normalized vectors
    - θ is the angle in radians, converted to degrees
    - Smaller angles indicate greater similarity
    """
    # Ensure input vectors are NumPy arrays
    vec1_array = np.array(vector1) if not isinstance(vector1, np.ndarray) else vector1
    vec2_array = np.array(vector2) if not isinstance(vector2, np.ndarray) else vector2

    # Normalize the vectors
    norm1 = float(np.linalg.norm(vec1_array))
    norm2 = float(np.linalg.norm(vec2_array))

    if norm1 < _SAM_NORM_EPS or norm2 < _SAM_NORM_EPS:
        return 0.0

    vec1_normalized = vec1_array / norm1
    vec2_normalized = vec2_array / norm2

    # Compute dot product and clamp to [-1, 1] to avoid numerical issues
    dot_product = float(np.clip(np.dot(vec1_normalized, vec2_normalized), -1.0, 1.0))

    # Compute the spectral angle
    angle = float(np.arccos(dot_product))

    return float(np.degrees(angle))
