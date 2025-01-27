
import numpy as np
from sklearn.decomposition import PCA

from tools import Dataset
from tools.simulations import simulator


def create_dataset(sim_params):
    data = []
    labels = []
    sim_params.cal_mix_prop()
    for i in range(sim_params.mat_proportion.shape[1]):
        weights = sim_params.mat_proportion[:, i]
        mat_em = np.average(sim_params.substances_emit, weights=weights, axis=1)
        mat_em = np.expand_dims(mat_em, 1)
        out = simulator(sim_params, mat_em)
        # print("out shape", out.shape)
        data.append(out)
        labels.append(weights)

    # Convert the list to a NumPy array
    # data_array = np.array(data)

    # Calculate mean and standard deviation along axis 0 (across the 4 values)
    # mean_values = np.mean(data_array, axis=0)
    # std_values = np.std(data_array, axis=0)

    # # Normalize each item in the list
    # normalized_data = [(item - mean_values) / std_values for item in data]

    # # If you want to convert the normalized data back to a list
    # normalized_data_list = [item.tolist() for item in normalized_data]

    # print(mean_values, std_values)
        
    # dataset = Dataset(normalized_data_list, labels)
    dataset = Dataset(data, labels)
    return dataset


def create_dataset_PCA(sim_params):
    data = []
    labels = []
    sim_params.cal_mix_prop()

    num_components = sim_params.basis_funcs.shape[1]
    # print(num_components)
    pca = PCA(n_components=num_components)
    # print(sim_params.substances_emit.shape)
    data_pca = pca.fit_transform(np.transpose(sim_params.substances_emit))
    # print(data_pca.shape)

    # sim_params.substances_emit
    for i in range(sim_params.mat_proportion.shape[1]):
        weights = sim_params.mat_proportion[:, i]
        mat_em = np.average(sim_params.substances_emit, weights=weights, axis=1)
        mat_em = np.expand_dims(mat_em, 0)
        # print(mat_em.shape)
        out = pca.transform(mat_em)
        # print(out.shape)
        out = np.squeeze(out, 0)
        # print(out.shape)
        # print(out)
        data.append(out)
        labels.append(weights)
        
    dataset = Dataset(data, labels)
    return dataset