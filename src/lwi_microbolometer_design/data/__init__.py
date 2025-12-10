"""Data loading utilities for substance and atmosphere data.

This package provides functions to load experimental data files, including
substance spectral emissivity data and atmospheric transmittance data, along
with simulation parameters, for use in sensor simulation and optimization.
"""

from .substance_atmosphere_data import load_substance_atmosphere_data

__all__ = ['load_substance_atmosphere_data']
