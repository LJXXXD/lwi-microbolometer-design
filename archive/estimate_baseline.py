
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools import RMSELoss, Sim_Parameters, create_dataset_PCA, load_data



matplotlib.use("Qt5Agg") # Changing the backend to QT solves ctrl-C not quiting issue in terminal

# Base line

epoch = 10000



mseloss = nn.MSELoss()
l1loss = nn.L1Loss()

baseline = {}


for num_substances in range(2, 3):

    print('\nnum substances', num_substances)

    L1Loss = []
    avg_L1Loss = []
    RMSE = []
    avg_RMSE = []
    MSE = []
    avg_MSE = []

    air_trans, basis_funcs, spectra, substances_emit = load_data(air_trans_file='./data/Test 2 - 21 Substances/Air transmittance.xlsx',
                                                                 basis_func_file='./data/Test 2 - 21 Substances/Basis functions.xlsx',
                                                                 spectra_file='./data/Test 2 - 21 Substances/spectra.xlsx', 
                                                                 substances_emit_file='./data/Test 2 - 21 Substances/substances.xlsx')

    sim_params = Sim_Parameters(air_trans=air_trans,
                                air_RI=1,
                                atm_dist_ratio=0.11,
                                basis_funcs=basis_funcs,
                                basis_func_comb=np.array(range(7)),
                                substance_ind_list=np.array(range(num_substances)),
                                spectra=spectra,
                                substances_emit=substances_emit,
                                temp_K=293.15)

    dataset = create_dataset_PCA(sim_params)
    print('Dataset Length', len(dataset))
    # print(dataset)

    for ind in tqdm(range(epoch)):
        
        target_loader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=True)
        pred_loader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=True)
        target_loader_iter = iter(target_loader)
        pred_loader_iter = iter(pred_loader)

        _, target = next(target_loader_iter)
        _, pred = next(pred_loader_iter)

        L1Loss.append(l1loss(pred, target).item())
        RMSE.append(RMSELoss(pred, target).item())
        MSE.append(mseloss(pred, target).item())
            
        avg_L1Loss.append(np.mean(L1Loss))
        avg_RMSE.append(np.mean(RMSE))
        avg_MSE.append(np.mean(MSE))
        
    print(avg_MSE[-1])
    print(avg_RMSE[-1])
    print(avg_L1Loss[-1])


    fig, ax = plt.subplots()
    ax.plot(avg_L1Loss)
    ax.plot(avg_RMSE)
    ax.plot(avg_MSE)

    loss_func_names = ['L1Loss', 'RMSELoss', 'MSELoss']

    ax.legend(loss_func_names)
    ax.set_title(str(num_substances) + '-substance random guesser error')
    ax.set_xlabel('Number of guesses')
    ax.set_ylabel('Error')

    baseline[num_substances] = dict(zip(loss_func_names, [avg_L1Loss[-1], avg_RMSE[-1], avg_MSE[-1]]))


print(baseline)


baseline_file_name = 'baseline_test.pkl'

try:
    with open(baseline_file_name, 'wb') as f:
        pickle.dump(baseline, f)
except Exception as e:
    showerror(type(e).__name__, str(e))
plt.show()
