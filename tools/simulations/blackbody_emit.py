
from math import pi

import numpy as np

def blackbody_emit(spectra, temp_K, refractive_index=1):
    H = 6.626_069_57e-34 # Planck constant
    C = 299_792_458 # Speed of light
    K = 1.380_648_8e-23 # Boltzmann constant

    C1 = 2 * pi * H * C**2
    C2 = H * C / K

    # return C1*1e24 / ((specta**5) * (np.exp(C2*1e6 / (temp_K * specta)) - 1))
    return (C1*1e24 / (pi * spectra**5)) * (1 / (np.exp(C2*1e6 / (temp_K * spectra) ) - 1))

    