
import numpy as np

from tools import Dataset, simulator


def create_dataset(sim_params):
    data = []
    labels = []
    for i in range(sim_params.mat_proportion.shape[1]):
        weights = sim_params.mat_proportion[:, i]
        mat_em = np.average(sim_params.substances_emit, weights=weights, axis=1)
        mat_em = np.expand_dims(mat_em, 1)
        out = simulator(sim_params, mat_em)
        data.append(out)
        labels.append(weights)
        
    dataset = Dataset(data, labels)
    return dataset