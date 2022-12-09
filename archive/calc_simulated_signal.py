

from operator import delitem
from tokenize import Double
import numpy as np
import csv

from func import blackbody_emit, blackbody_emit_old
from simulator import simulator


def calc_simulated_signal():
    ################################################################################
    # Read args <START>

    spec = np.genfromtxt('./data/spectra.csv')
    spec = np.expand_dims(spec, 1)
    abosorptivity = np.genfromtxt('./data/absorptivity.csv', delimiter=",")
    emissivity = np.genfromtxt('./data/emissivity.csv', delimiter=",")

    # emissivity_CO = np.genfromtxt('./output/CO-IR-adjusted.csv', delimiter=",")[:, 1]
    # emissivity_CO2 = np.genfromtxt('./output/CO2-IR-adjusted.csv', delimiter=",")[:, 1]
    # emissivity_Methane = np.genfromtxt('./output/Methane-IR-adjusted.csv', delimiter=",")[:, 1]
    # emissivity_Propane = np.genfromtxt('./output/Propane-IR-adjusted.csv', delimiter=",")[:, 1]
    # emissivity = np.column_stack([emissivity_CO, emissivity_CO2, emissivity_Methane, emissivity_Propane])
    # print(emissivity.shape)


    # Read args <END>
    ################################################################################
    
    h = 6.62606957e-34
    c = 299792458
    kb = 1.3806488e-23
    temp_C = 10
    # spec = 4

    c1 = 2 * h * c**2
    c2 = h * c / kb
    c1_prime = c1 * 1e24
    c2_prime = c2 * 1e6
    temp_K = temp_C + 273.15

    # bb_emit = blackbody_emit_old(c1_prime, c2_prime, spec, temp_K)
    bb_emit = blackbody_emit(spec, temp_K)
    
    # mat_emit = np.multiply(bb_emit, emissivity)
    mat_emit = bb_emit * emissivity

    # signature_matrix = np.zeros((emissivity.shape[1], abosorptivity.shape[1]))

    # for i in range(abosorptivity.shape[1]):
    #     pix_absorb = np.multiply(mat_emit, abosorptivity[:, i:i+1])
    #     signature_matrix[:, i] = np.trapz(pix_absorb, spec[:, 0], axis=0)
    
    # print(signature_matrix.transpose())

    result = simulator(spec, emissivity[:, 0:1], 1, bb_emit, abosorptivity)
    print(result)

if __name__ == '__main__':
    calc_simulated_signal()