

from operator import delitem
from tokenize import Double
import numpy as np
import csv

from func import blackbody_emit


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
    c1_prime = c1 * 1e+24
    c2_prime = c2 * 1000000
    temp_K = temp_C + 273.15

    bb_emit = blackbody_emit(c1_prime, c2_prime, spec, temp_K)
    
    t1 = np.multiply(bb_emit, abosorptivity)

    signature_matrix = np.zeros((emissivity.shape[1], abosorptivity.shape[1]))

    for i in range(emissivity.shape[1]):
        t2 = np.multiply(t1, emissivity[:, i:i+1])
        for j in range (abosorptivity.shape[1]):
            signature_matrix[i, j] = np.trapz(t2[:, j], spec[:, 0])
    
    print(signature_matrix.transpose())


if __name__ == '__main__':
    calc_simulated_signal()