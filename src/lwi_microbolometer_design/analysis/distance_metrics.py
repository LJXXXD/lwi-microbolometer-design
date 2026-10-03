"""Distance metrics for spectral analysis and substance discrimination."""

import numpy as np

# Below this norm, treat a vector as zero-magnitude so SAM stays finite during optimization.
_SAM_NORM_EPS = 1e-15


def _sam_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute SAM between finite equal-length rows, preserving zero-norm handling."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] == 0 or not np.all(np.isfinite(vectors)):
        raise ValueError("SAM vectors must be nonempty finite 1D arrays of equal length.")
    scales = np.max(np.abs(vectors), axis=1)
    scaled = np.divide(
        vectors, scales[:, None], out=np.zeros_like(vectors), where=scales[:, None] != 0
    )
    norms = np.linalg.norm(scaled, axis=1)
    degenerate = scales < _SAM_NORM_EPS / np.maximum(norms, 1.0)
    normalized = np.divide(
        scaled, norms[:, None], out=np.zeros_like(scaled), where=norms[:, None] != 0
    )
    angles = np.degrees(np.arccos(np.clip(normalized @ normalized.T, -1.0, 1.0)))
    angles[degenerate, :] = 0.0
    angles[:, degenerate] = 0.0
    np.fill_diagonal(angles, 0.0)
    return angles


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
        The spectral angle in degrees between the two vectors, in [0, 180].
        Nonnegative spectra lie in [0, 90]. If either vector has norm below
        ``1e-15``, returns ``0.0``
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
    vec1 = np.asarray(vector1, dtype=np.float64)
    vec2 = np.asarray(vector2, dtype=np.float64)
    if vec1.ndim != 1 or vec2.shape != vec1.shape or vec1.size == 0:
        raise ValueError("SAM vectors must be nonempty 1D arrays with identical shape.")
    if not np.all(np.isfinite(vec1)) or not np.all(np.isfinite(vec2)):
        raise ValueError("SAM vectors must be finite, including degenerate comparisons.")

    # Scaling keeps norms representable for very small or very large vectors.
    scale1, scale2 = float(np.max(np.abs(vec1))), float(np.max(np.abs(vec2)))
    if scale1 == 0 or scale2 == 0:
        return 0.0
    scaled1, scaled2 = vec1 / scale1, vec2 / scale2
    norm1, norm2 = float(np.linalg.norm(scaled1)), float(np.linalg.norm(scaled2))
    if scale1 < _SAM_NORM_EPS / norm1 or scale2 < _SAM_NORM_EPS / norm2:
        return 0.0
    vec1_normalized, vec2_normalized = scaled1 / norm1, scaled2 / norm2

    # Compute dot product and clamp to [-1, 1] to avoid numerical issues
    dot_product = float(np.clip(np.dot(vec1_normalized, vec2_normalized), -1.0, 1.0))

    # Compute the spectral angle
    angle = float(np.arccos(dot_product))

    return float(np.degrees(angle))
