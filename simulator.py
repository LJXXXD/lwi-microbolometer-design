
from math import pi

import numpy as np
import pandas as pd

from tools.blackbody_emit import blackbody_emit

def simulator(spectra, mat_em, temp_K, air_trans, atm_dist_ratio, air_RI, basis_funcs):
    bb_emit = blackbody_emit(spectra, temp_K, air_RI)
    tau_air = air_trans ** atm_dist_ratio
    abso_spec = tau_air * bb_emit * mat_em * basis_funcs
    out = np.trapz(abso_spec, spectra, axis=0)

    return out


if __name__ == '__main__':
    basis_funcs = np.array(pd.read_excel('./data/Test 2 - 21 Substances/Basis functions.xlsx', header=None))

    # Substance related parameters
    spectra = basis_funcs[:, 0:1]
    substances_emit = np.array(pd.read_excel('./data/Test 2 - 21 Substances/substances.xlsx', header=None))
    mat_proportion = np.array(pd.read_excel('./data/Test 2 - 21 Substances/proportion.xlsx', header=None)) # Material mixture proportion
    weights = mat_proportion[:, 0] / np.sum(mat_proportion[:, 0])
    mat_em = np.average(substances_emit, weights=weights, axis=1)
    mat_em = np.expand_dims(mat_em, 1)
    
    # Enviroment related parameters
    temp_K = 293.15 # Environmental temperature in K
    air_trans = np.array(pd.read_excel('./data/Test 2 - 21 Substances/Air transmittance.xlsx', header=None))
    air_trans = air_trans[:, 1:]
    atm_dist_ratio = 0.11 # Atomsphere distance ratio
    air_RI = 1 # Refractive index of air

    # Sensor related parameters
    basis_funcs = basis_funcs[:, 1:]


    out = simulator(spectra, mat_em, temp_K, air_trans, atm_dist_ratio, air_RI, basis_funcs)

    print(out)