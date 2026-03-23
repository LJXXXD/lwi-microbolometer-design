"""Normalisation helpers for mixed-scale MAP-Elites parameter spaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UnitCubeScaler:
    """Affine map between physical gene bounds and the unit hypercube.

    CMA-based optimizers behave much better on this project when wavelength
    and width parameters are scaled onto comparable ranges before adaptation.
    """

    lower: np.ndarray
    upper: np.ndarray
    span: np.ndarray

    @classmethod
    def from_bounds(
        cls,
        bounds_low: list[float] | np.ndarray,
        bounds_high: list[float] | np.ndarray,
    ) -> "UnitCubeScaler":
        """Build a scaler from lower/upper bound vectors."""
        lower = np.asarray(bounds_low, dtype=float)
        upper = np.asarray(bounds_high, dtype=float)
        span = upper - lower
        if lower.shape != upper.shape:
            raise ValueError("Lower and upper bounds must have identical shape.")
        if np.any(span <= 0.0):
            raise ValueError("Every gene bound must satisfy high > low.")
        return cls(lower=lower, upper=upper, span=span)

    @property
    def dimension(self) -> int:
        """Number of parameters represented by the scaler."""
        return int(self.lower.size)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Map physical parameters onto the unit hypercube."""
        array = np.asarray(values, dtype=float)
        normalized = (array - self.lower) / self.span
        return np.clip(normalized, 0.0, 1.0)

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        """Map unit-hypercube parameters back to physical bounds."""
        array = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
        physical = self.lower + array * self.span
        return np.clip(physical, self.lower, self.upper)
