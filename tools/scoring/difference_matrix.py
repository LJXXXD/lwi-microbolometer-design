
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist,squareform
from scipy.stats import wasserstein_distance

from tools.simulations import simulator


def difference_matrix(items):
    diff_matrix = squareform(pdist(items, metric='euclidean'))
    return diff_matrix


def emd_matrix(items):
    emd_matrix = np.zeros((len(items), len(items)))
    for i, sub_a in enumerate(items):
        for j, sub_b in enumerate(items):
            sub_a /= np.sum(sub_a)
            sub_b /= np.sum(sub_b)
            emd_matrix[i, j] = wasserstein_distance(u_values=np.arange(len(sub_a)),
                                                    v_values=np.arange(len(sub_b)),
                                                    u_weights=sub_a,
                                                    v_weights=sub_b)
    return emd_matrix



if __name__ == '__main__':
    # Enviroment related parameters
    temp_K = 293.15 # Environmental temperature in K
    air_trans = np.array(pd.read_excel('./data/Test 2 - 21 Substances/Air transmittance.xlsx', header=None))
    air_trans = air_trans[:, 1:]
    atm_dist_ratio = 0.11 # Atomsphere distance ratio
    air_RI = 1 # Refractive index of air

    # Sensor related parameters
    basis_funcs = np.array(pd.read_excel('./data/Test 2 - 21 Substances/Basis functions.xlsx', header=None))
    basis_funcs = basis_funcs[:, 1:]
    
    # Substance related parameters
    spectra = np.array(pd.read_excel('./data/Test 2 - 21 Substances/spectra.xlsx', header=None))
    substances_emit = np.array(pd.read_excel('./data/Test 2 - 21 Substances/substances.xlsx', header=None))
    
    
    out = []
    for i in range(substances_emit.shape[1]):
        mat_em = substances_emit[:, i:i+1]
        out.append(simulator(spectra, mat_em, temp_K, air_trans, atm_dist_ratio, air_RI, basis_funcs))
        # print(out[i])

    # print(len(out))


    diff_matrix = difference_matrix(out)
    print(diff_matrix)
    np.savetxt("diff_matrix.csv", diff_matrix, delimiter=",")