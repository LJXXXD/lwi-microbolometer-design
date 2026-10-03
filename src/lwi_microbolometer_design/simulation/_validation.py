"""Shared input contracts for scene construction and public simulation APIs."""

import numpy as np


def spectral_vector(values: np.ndarray, name: str) -> np.ndarray:
    """Canonicalize a 1D spectral vector or a row/column vector."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a spectral vector; got shape {array.shape}")
    return array


def validate_wavelengths(wavelengths: np.ndarray) -> None:
    """Require at least two finite, positive, strictly increasing samples in µm."""
    if wavelengths.size < 2:
        raise ValueError("wavelengths must contain at least two samples for integration.")
    if not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0):
        raise ValueError("wavelengths must be finite and positive.")
    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError("wavelengths must be strictly increasing.")


def positive_scalar(value: float, name: str, *, allow_zero: bool = False) -> float:
    """Require a finite positive scalar, optionally allowing zero."""
    if np.ndim(value) != 0:
        raise ValueError(f"{name} must be a scalar.")
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0 or (scalar == 0 and not allow_zero):
        domain = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {domain}.")
    return scalar


def validate_transmittance(transmittance: np.ndarray) -> None:
    """Require finite fractional transmission without clipping measurements."""
    if (
        not np.all(np.isfinite(transmittance))
        or np.any(transmittance < 0)
        or np.any(transmittance > 1)
    ):
        raise ValueError("air_transmittance must be finite and within [0, 1].")


def emissivity_matrix(values: np.ndarray, num_wavelengths: int) -> np.ndarray:
    """Require aligned finite spectra; preserve signed measured values verbatim."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"emissivity_curves must be 2D with shape (d, n); got shape {array.shape}")
    if array.shape[0] != num_wavelengths:
        raise ValueError(
            f"emissivity_curves first axis ({array.shape[0]}) must match "
            f"wavelengths length ({num_wavelengths})"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("emissivity_curves must be finite.")
    return array
