
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from tools import Sim_Parameters, Train_Parameters, create_dataset, RMSELoss, train_val_test, test_epoch




test_loss_temp_list = []

for num_substances in range(2, 8):
    print('\nnum substances', num_substances)

    sim_params = Sim_Parameters(air_trans_file='./data/Test 2 - 21 Substances/Air transmittance.xlsx',
                                air_RI=1,
                                atm_dist_ratio=0.11,
                                basis_func_file='./data/Test 2 - 21 Substances/Basis functions.xlsx',
                                num_substances=num_substances,
                                spectra_file='./data/Test 2 - 21 Substances/spectra.xlsx',
                                substances_emit_file='./data/Test 2 - 21 Substances/substances.xlsx',
                                temp_K=293.15)


    dataset = create_dataset(sim_params)
    len(dataset)


    train_params = Train_Parameters(train_percentage=0.8,
                                    batch_size=len(dataset) // 10,
                                    criterions=[nn.L1Loss(), RMSELoss, nn.MSELoss()],
                                    learning_rate=1e-3,
                                    num_epochs=100,
                                    device=torch.device("cpu"),
                                    k_fold_flag=True,
                                    k=5,
                                    random_flag=True,
                                    random_seed=28)


    history, models, test_loss, pred_list, targ_list = train_val_test(dataset, train_params, sim_params)


    test_loss_temp_list.append(test_loss)

    print(test_loss)

test_loss_temp_list = np.array(test_loss_temp_list)
plt.plot(test_loss_temp_list)

plt.show()