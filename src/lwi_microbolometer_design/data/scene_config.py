"""Frozen scene configuration for sensor simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lwi_microbolometer_design.simulation._validation import (
    emissivity_matrix,
    positive_scalar,
    spectral_vector,
    validate_transmittance,
    validate_wavelengths,
)


@dataclass(frozen=True, slots=True)
class SceneConfig:
    """Validated configuration of the physical scene for sensor simulation.

    Contains all environment and substance parameters needed by
    ``simulate_sensor_output``. Sensor basis functions are not included
    because they are the optimization variable.

    Parameters
    ----------
    wavelengths : np.ndarray
        Discrete wavelength sampling points in µm. Canonical shape is ``(d,)``;
        column vectors ``(d, 1)`` are squeezed on construction.
    emissivity_curves : np.ndarray
        Finite measured emissivity spectrum for each substance, shape ``(d, n)``.
        Raw values outside [0, 1] are preserved; they require a separate physical
        interpretation or preprocessing decision before claims about emissivity.
    air_transmittance : np.ndarray
        Atmospheric transmission per wavelength. Canonical shape ``(d,)``;
        ``(d, 1)`` or the first column of ``(d, k)`` is used if needed.
    temperature_k : float
        Scene temperature in Kelvin.
    atmospheric_distance_ratio : float
        Exponent for atmospheric path modeling (e.g., 0.11).
    air_refractive_index : float
        Refractive index of air (≈1.0).
    substance_names : np.ndarray
        Human-readable names of the ``n`` substances, shape ``(n,)``, dtype object/str.

    Raises
    ------
    ValueError
        If shapes, spectral ordering, finiteness, or scene scalar domains are invalid.

    Notes
    -----
    Fields are frozen, but NumPy array contents remain mutable. Do not cache
    derived quantities across edits to those arrays. Wavelengths are vacuum
    wavelengths in µm. Temperature and refractive index must be positive;
    distance ratio may be zero (no path attenuation).
    """

    wavelengths: np.ndarray
    emissivity_curves: np.ndarray
    air_transmittance: np.ndarray
    temperature_k: float
    atmospheric_distance_ratio: float
    air_refractive_index: float
    substance_names: np.ndarray

    def __post_init__(self) -> None:
        """Canonicalize array shapes and enforce spectral grid consistency."""
        wl = spectral_vector(self.wavelengths, "wavelengths")
        validate_wavelengths(wl)
        em = emissivity_matrix(self.emissivity_curves, wl.size)
        at = np.asarray(self.air_transmittance, dtype=np.float64)
        if at.ndim == 2 and at.shape[1] > 0:
            at = at[:, 0]
        at = spectral_vector(at, "air_transmittance")
        if at.size != wl.size:
            raise ValueError(
                f"air_transmittance length ({at.size}) must match wavelengths length ({wl.size})"
            )
        validate_transmittance(at)
        names = np.asarray(self.substance_names, dtype=object).reshape(-1)
        if names.size != em.shape[1]:
            raise ValueError(
                f"substance_names length ({names.size}) must match "
                f"number of emissivity columns ({em.shape[1]})"
            )

        object.__setattr__(self, "wavelengths", wl)
        object.__setattr__(self, "emissivity_curves", em)
        object.__setattr__(self, "air_transmittance", at)
        object.__setattr__(self, "substance_names", names)
        for name in ("temperature_k", "air_refractive_index", "atmospheric_distance_ratio"):
            object.__setattr__(
                self,
                name,
                positive_scalar(
                    getattr(self, name), name, allow_zero=name == "atmospheric_distance_ratio"
                ),
            )
