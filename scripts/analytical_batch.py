
import fcntl
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import scipy.stats as stats

import sys
sys.path.insert(0, os.path.abspath('.'))
from tools import Sim_Parameters, Train_Parameters, create_dataset, RMSELoss, train_val_test, test_epoch, simulator, ballsINTObins, load_data, NoisyDataset, df_to_excel





def solve_for_B(A, C):
    # Calculate the Moore-Penrose pseudoinverse of matrix A
    A_pseudo_inv = np.linalg.pinv(A)
    
    # Calculate matrix B
    B = np.dot(A_pseudo_inv, C)
    
    return B


substance_ind_list = [0, 1, 2, 3]
substance_ind_list.sort()

air_trans, basis_funcs, spectra, substances_emit = load_data(air_trans_file='./data/Test 3 - 4 White Powers/Air transmittance.xlsx',
                                                             basis_func_file='./data/Test 3 - 4 White Powers/Basis functions_4-20um.xlsx',
                                                             spectra_file='./data/Test 3 - 4 White Powers/white_powders_spectra.xlsx', 
                                                             substances_emit_file='./data/Test 3 - 4 White Powers/white_powders.xlsx',)


substance_names = np.array(pd.read_excel('./data/Test 3 - 4 White Powers/white_powders_names.xlsx', header=None))

comb = [0, 1, 4, 6]
temp_K = 293.15

sim_params = Sim_Parameters(air_trans=air_trans,
                            air_RI=1,
                            atm_dist_ratio=0.11,
                            basis_funcs=basis_funcs,
                            basis_func_comb=comb,
                            substance_ind_list=substance_ind_list,
                            spectra=spectra,
                            substances_emit=substances_emit,
                            temp_K=temp_K,
                            percentage_step=0.05)

dataset = create_dataset(sim_params)




sensor_output_list = []

for idx in substance_ind_list:
    sub_signal = substances_emit[:, idx]
    sub_signal = np.expand_dims(sub_signal, 1)
    
    out = simulator(sim_params, sub_signal)  # Apply simulator() function to the column
    sensor_output_list.append(out)

sensor_output_list = np.asarray(sensor_output_list).transpose()
print(sensor_output_list.shape)
A = np.asarray(sensor_output_list)
print('A', A)


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
print('Num of test samples', test_size)

test_noise_perc_list = [i * 0.05 for i in range(41)]
print('Test_Noise', test_noise_perc_list)

results = []

for test_noise_perc in test_noise_perc_list:
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    if test_noise_perc != 0:
        test_dataset = NoisyDataset(test_dataset, test_noise_perc)


    dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    loss_list = []
    targ_list = []
    pred_list = []

    yi_pred_exclude = []


    for batch_idx, (x, y) in enumerate(dataloader):
        print('Data Shape', x.shape)
        print('Label shape', y.shape)
        
        sum_x = sum(x).numpy()
        print('sum x', sum_x)
        sum_y_pred, _, _, _ = np.linalg.lstsq(np.asarray(sensor_output_list), sum_x, rcond=None)
        if test_noise_perc != 0:
            print('test noise', test_noise_perc)
            print('sum y pred', sum_y_pred)
            print('sum y targ', sum(y))

        for i in range(len(y)):
            sum_x_exclude = sum_x - x[i].numpy()
            sum_y_pred_exclude, _, _, _ = np.linalg.lstsq(np.asarray(sensor_output_list), sum_x_exclude, rcond=-1)
            yi_pred_exclude.append(sum_y_pred_exclude)
            yi = sum_y_pred - sum_y_pred_exclude
            # if test_noise_perc != 0:
                # print('test noise', test_noise_perc)
                # print('pred_y', yi)
                # print('targ_y', y[i])
        # print('A - shape', sensor_output_list.shape)
        # print('B - shape', y.numpy().transpose().shape)
        # print('C - shape', x.numpy().transpose().shape)

        # X = np.matmul(sensor_output_list, y.numpy().transpose())
        # print('Cal C\n', X.transpose())
        # print('Cal C - shape', X.shape)
        # print('C original', x.numpy())

        # Cal_B_by_Cal_C = solve_for_B(sensor_output_list, X)
        # print('Cal B by Cal C\n', Cal_B_by_Cal_C)

        # prediction_y = solve_for_B(sensor_output_list, x.numpy().transpose())
        # if test_noise_perc == 0:
        #     print('pred', prediction_y.transpose())
        #     print('targ', y.numpy())
        # print('sum of pred', np.sum(prediction_y))
        # normalized_prediction = prediction_y / np.sum(prediction_y)
        # prediction_y = normalized_prediction
        # print('pred', prediction_y.transpose())
        # print('sum of pred', np.sum(prediction_y))

            criterion = nn.L1Loss()

            loss = criterion(torch.from_numpy(yi.transpose()), y[i])
            # print('loss', loss)
            loss_list.append(loss.numpy())
            targ_list.append(np.squeeze(y[i].numpy()))
            pred_list.append(np.squeeze(yi))
    
    sum_yi_pred_exclude = sum(yi_pred_exclude)
    sum_yi = sum_yi_pred_exclude / (len(test_dataset) - 1)
    print('sum_yi', sum_yi)

    avg_loss = np.mean(loss_list)
    targ_list = np.asarray(targ_list)
    pred_list = np.asarray(pred_list)
    # print('avg loss', avg_loss)
    # print(pred_list.shape)
    # print(targ_list.shape)
    # diff_list = pred_list - targ_list

    # print('diff_list', diff_list)


    row = {
        'Temperature_K': temp_K,
        'Substance Number': len(substance_ind_list),
        'Substance Comb': tuple(substance_ind_list),
        'Basis Function Number': len(comb),
        'Comb of Basis Functions': tuple(comb),
        'Train Noise Max Percentage': 0,
        'Test Noise Max Percentage': test_noise_perc,
        'L1Loss': avg_loss
    }


    results.append(row)

df = pd.DataFrame(results)
print('Final Print')
print(df)
df_to_excel(df, './output', 'analytical_batch', df.columns.to_list())