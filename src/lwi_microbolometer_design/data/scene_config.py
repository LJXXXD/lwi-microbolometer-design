"""Immutable scene configuration for sensor simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SceneConfig:
    """Immutable snapshot of the physical scene for sensor simulation.

    Contains all environment and substance parameters needed by
    ``simulate_sensor_output``. Sensor basis functions are not included
    because they are the optimization variable.

    Parameters
    ----------
    wavelengths : np.ndarray
        Discrete wavelength sampling points in µm. Canonical shape is ``(d,)``;
        column vectors ``(d, 1)`` are squeezed on construction.
    emissivity_curves : np.ndarray
        Emissivity spectrum for each substance, shape ``(d, n)``, values in [0, 1].
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
        If array shapes are inconsistent after canonicalization.
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
        wl = np.asarray(self.wavelengths, dtype=np.float64)
        wl = np.squeeze(wl)
        if wl.ndim != 1:
            msg = (
                "wavelengths must reduce to a 1D array of length d; "
                f"got shape {np.asarray(self.wavelengths).shape}"
            )
            raise ValueError(msg)
        wl = wl.reshape(-1)
        object.__setattr__(self, "wavelengths", wl)

        em = np.asarray(self.emissivity_curves, dtype=np.float64)
        object.__setattr__(self, "emissivity_curves", em)

        at = np.asarray(self.air_transmittance, dtype=np.float64)
        if at.ndim == 2:
            if at.shape[1] == 1:
                at = np.squeeze(at, axis=1)
            else:
                at = at[:, 0]
        at = np.squeeze(at)
        if at.ndim != 1:
            msg = (
                "air_transmittance must reduce to a 1D array of length d; "
                f"got shape {np.asarray(self.air_transmittance).shape}"
            )
            raise ValueError(msg)
        at = at.reshape(-1)
        object.__setattr__(self, "air_transmittance", at)

        names = np.asarray(self.substance_names, dtype=object)
        object.__setattr__(self, "substance_names", names.reshape(-1))

        d = int(wl.shape[0])
        if em.ndim != 2:
            msg = f"emissivity_curves must be 2D with shape (d, n); got shape {em.shape}"
            raise ValueError(msg)
        if em.shape[0] != d:
            msg = (
                f"emissivity_curves first axis ({em.shape[0]}) must match wavelengths length ({d})"
            )
            raise ValueError(msg)

        if at.shape[0] != d:
            msg = f"air_transmittance length ({at.shape[0]}) must match wavelengths length ({d})"
            raise ValueError(msg)

        n_sub = int(em.shape[1])
        if names.shape[0] != n_sub:
            msg = (
                f"substance_names length ({names.shape[0]}) must match "
                f"number of emissivity columns ({n_sub})"
            )
            raise ValueError(msg)
