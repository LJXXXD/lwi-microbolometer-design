
from math import pi

import numpy as np
import pandas as pd

from tools.blackbody_emit import blackbody_emit

def simulator(sim_params, mat_em):
    bb_emit = blackbody_emit(sim_params.spectra, sim_params.temp_K, sim_params.air_RI)
    tau_air = sim_params.air_trans ** sim_params.atm_dist_ratio
    abso_spec = tau_air * bb_emit * mat_em * sim_params.basis_funcs
    out = np.trapz(abso_spec, sim_params.spectra, axis=0)

    return out


if __name__ == '__main__':
    # Enviroment related parameters
    temp_K = 293.15 # Environmental temperature in K
    air_trans = np.array(pd.read_excel('./data/Test 2 - 21 Substances/Air transmittance.xlsx', header=None))
    air_trans = air_trans[:, 1:]
    atm_dist_ratio = 0.11 # Atomsphere distance ratio
    air_RI = 1 # Refractive index of air

    # Sensor related parame ters
    basis_funcs = np.array(pd.read_excel('./data/Test 2 - 21 Substances/Basis functions.xlsx', header=None))
    basis_funcs = basis_funcs[:, 1:]
    
    # Substance related parameters
    spectra = np.array(pd.read_excel('./data/Test 2 - 21 Substances/spectra.xlsx', header=None))
    substances_emit = np.array(pd.read_excel('./data/Test 2 - 21 Substances/substances.xlsx', header=None))
    mat_proportion = np.array(pd.read_excel('./data/Test 2 - 21 Substances/proportion.xlsx', header=None)) # Material mixture proportion
    weights = mat_proportion[:, 0] / np.sum(mat_proportion[:, 0])
    # mat_em = np.average(substances_emit, weights=weights, axis=1)
    mat_em = substances_emit[:, 0]
    mat_em = np.expand_dims(mat_em, 1)
    
    

    out = simulator(spectra, mat_em, temp_K, air_trans, atm_dist_ratio, air_RI, basis_funcs)

    np.savetxt('temp.csv', out.transpose(), delimiter = ",")

    print(out)