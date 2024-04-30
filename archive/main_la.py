import fcntl
import itertools
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special
import scipy.stats as stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.abspath('.'))
from tools import Sim_Parameters, Train_Parameters, create_dataset, RMSELoss, train_val_test, load_data, simulator, NoisyDataset



matplotlib.use("TkAgg") # Changing the backend to QT solves ctrl-C not quiting issue in terminal




def solve_for_B(A, C):
    # Calculate the Moore-Penrose pseudoinverse of matrix A
    A_pseudo_inv = np.linalg.pinv(A)
    
    # Calculate matrix B
    B = np.dot(A_pseudo_inv, C)
    
    return B




# User Input
substance_ind_list =  [0, 1, 2, 3]
substance_ind_list.sort()

basis_func_ind_list = [0, 1, 2, 3, 4, 5, 6]
basis_func_ind_list.sort()

temp_K_list = np.asarray([293.15])
temp_K_list.sort()

train_noise_list = [0]
test_noise_list = [0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 1, 2]


train_noise_list = [0]
# test_noise_list = [0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 1, 2]
test_noise_list = [i * 0.05 for i in range(21)]


air_trans, basis_funcs, spectra, substances_emit = load_data(air_trans_file='./data/Test 3 - 4 White Powers/Air transmittance.xlsx',
                                                             basis_func_file='./data/Test 3 - 4 White Powers/Basis functions_4-20um.xlsx',
                                                             spectra_file='./data/Test 3 - 4 White Powers/white_powders_spectra.xlsx', 
                                                             substances_emit_file='./data/Test 3 - 4 White Powers/white_powders.xlsx',)


substance_names = np.array(pd.read_excel('./data/Test 3 - 4 White Powers/white_powders_names.xlsx', header=None))

# Start
print('\nsubstances', str(substance_ind_list))

results = []


for temp_K_ind, temp_K in enumerate(temp_K_list):
    sub_combs = list(itertools.combinations(range(4), 4))

    for substance_ind_list in sub_combs:
        for num_basis_func_ind, num_basis_func in enumerate(tqdm(range(1, len(basis_func_ind_list)+1))):
            if num_basis_func != len(substance_ind_list):
                continue
            basis_func_combs = list(itertools.combinations(range(len(basis_func_ind_list)), num_basis_func))

            for comb_ind, comb in enumerate(basis_func_combs):
                sim_params = Sim_Parameters(air_trans=air_trans,
                                            air_RI=1,
                                            atm_dist_ratio=0.11,
                                            basis_funcs=basis_funcs,
                                            basis_func_comb=comb,
                                            substance_ind_list=substance_ind_list,
                                            spectra=spectra,
                                            substances_emit=substances_emit,
                                            temp_K=temp_K)
                
                dataset = create_dataset(sim_params)

                sensor_output_list = []
                for idx in substance_ind_list:
                    sub_signal = substances_emit[:, idx]
                    sub_signal = np.expand_dims(sub_signal, 1)
                    out = simulator(sim_params, sub_signal)  # Apply simulator() function to the column
                    sensor_output_list.append(out)
                sensor_output_list = np.asarray(sensor_output_list).transpose()
                # print(sensor_output_list.shape)

                for train_noise in train_noise_list:
                    for test_noise in test_noise_list:
                        # train_params = Train_Parameters(train_percentage=0.8,
                        #                                 # batch_size=len(dataset) // 10,
                        #                                 batch_size=100,
                        #                                 criterions=[nn.L1Loss(), RMSELoss, nn.MSELoss()],
                        #                                 # criterions=[nn.MSELoss()],
                        #                                 loss_func_names = ['L1Loss', 'RMSELoss', 'MSELoss'],
                        #                                 learning_rate=1e-3,
                        #                                 num_epochs=100,
                        #                                 device=torch.device("cpu"),
                        #                                 k_fold_flag=True,
                        #                                 k=5,
                        #                                 random_flag=True,
                        #                                 random_seed=28,
                        #                                 train_noise_perc=train_noise,
                        #                                 test_noise_perc=test_noise)
                        train_size = int(0.8 * len(dataset))
                        test_size = len(dataset) - train_size

                        torch.manual_seed(28)
                        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
                        if test_noise != 0:
                            test_dataset = NoisyDataset(test_dataset, test_noise)


                        dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

                        loss_list = []
                        targ_list = []
                        pred_list = []

                        for batch_idx, (x, y) in enumerate(dataloader):
                            # print(batch_idx)
                            # print('x - C - mixture - Norm', x)
                            # print('y - B - ratio', y)

                            # print('A - shape', sensor_output_list.shape)
                            # print('B - shape', y.numpy().transpose().shape)
                            # print('C - shape', x.numpy().transpose().shape)

                            X = np.matmul(sensor_output_list, y.numpy().transpose())
                            # print('Cal C\n', X.transpose())
                            # print('Cal C - shape', X.shape)

                            Cal_B_by_Cal_C = solve_for_B(sensor_output_list, X)
                            # print('Cal B by Cal C\n', Cal_B_by_Cal_C)

                            prediction = solve_for_B(sensor_output_list, x.numpy().transpose())
                            # print('pred', prediction.transpose())
                            # print('sum of pred', np.sum(prediction))
                            normalized_prediction = prediction / np.sum(prediction)
                            prediction = normalized_prediction
                            # print('pred', prediction.transpose())
                            # print('sum of pred', np.sum(prediction))

                            criterion = nn.L1Loss()

                            loss = criterion(torch.from_numpy(prediction.transpose()), y)
                            # print('loss', loss)
                            loss_list.append(loss.numpy())
                            targ_list.append(np.squeeze(y.numpy()))
                            pred_list.append(np.squeeze(prediction))

                        avg_loss = np.mean(loss_list)
                        targ_list = np.asarray(targ_list)
                        pred_list = np.asarray(pred_list)
                        # print('avg loss', avg_loss)
                        # print(pred_list.shape)
                        # print(targ_list.shape)
                        diff_list = pred_list - targ_list

                        # print('diff_list', diff_list)


                        row = {
                            'Temperature_K': temp_K,
                            'Substance Number': len(substance_ind_list),
                            'Substance Comb': tuple(substance_ind_list),
                            'Basis Function Number': len(comb),
                            'Comb of Basis Functions': comb,
                            'Train Noise': 0,
                            'Test Noise': test_noise,
                            'L1Loss': avg_loss
                        }



                        perc95_interval = []
                        for i, sub_ind in enumerate(substance_ind_list):

                            data = diff_list[:, i]
                            # params = stats.norm.fit(data)
                            # data_mean, data_std = params
                            ae, loce, scalee = stats.skewnorm.fit(data)
                            
                            # Add mean and std to row
                            row[f'Substance {sub_ind} skewnorm a'] = ae
                            row[f'Substance {sub_ind} skewnorm loc'] = loce
                            row[f'Substance {sub_ind} skewnorm scale'] = scalee


                            # Create a skewed normal distribution object
                            dist = stats.skewnorm(ae, loce, scalee)


                            for confidence_perc in (0.68, 0.95, 0.997):
                                # Calculate the quantiles for a specific percentage
                                lower_quantile = dist.ppf(0.5-confidence_perc/2)  # Lower quantile covering % of the data
                                upper_quantile = dist.ppf(0.5+confidence_perc/2)  # Upper quantile covering % of the data

                                # row[f'Substance {sub_ind} skewnorm {confidence_perc*100}% low'] = lower_quantile
                                # row[f'Substance {sub_ind} skewnorm {confidence_perc*100}% high'] = upper_quantile
                                # row[f'Substance {sub_ind} skewnorm {confidence_perc*100}% interval range'] = upper_quantile - lower_quantile

                                if confidence_perc == 0.95:
                                    perc95_interval.append(upper_quantile - lower_quantile)


                        perc95_min = np.min(perc95_interval)
                        row[f'MIN skewnorm 95% interval range'] = perc95_min
                        perc95_max = np.max(perc95_interval)
                        row[f'MAX skewnorm 95% interval range'] = perc95_max
                        perc95_avg = np.mean(perc95_interval)
                        row[f'AVG skewnorm 95% interval range'] = perc95_avg


                        # Append the row to the results list
                        results.append(row)

# Create a pandas DataFrame from the results list
df = pd.DataFrame(results)



# df.to_csv('.csv', index=False)

output_file_name = 'loss_la.csv'
# Check if the file "loss.pkl" already exists
if os.path.isfile(output_file_name):

    # If the file exists, load the existing DataFrame from the file
    df_existing = pd.read_csv(output_file_name)

    # Open the file in binary mode
    with open(output_file_name, 'w') as f:
        while True:
            try:
                # Get a lock on the file
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except IOError:
                # Lock is held by another process, wait and try again
                time.sleep(1)

        # Merge df to df_existing, and if there's duplicates, keep the new values from df
        df = pd.concat([df, df_existing]).drop_duplicates(
            subset=['Temperature_K', 'Substance Number', 'Substance Comb', 'Basis Function Number', 
                    'Comb of Basis Functions', 'Train Noise', 'Test Noise'], keep='first')
        df = df.sort_values(by=list(df.columns))
        print(df)

        # Save the DataFrame to a CSV file "loss.csv"
        df.to_csv(output_file_name, index=False)

        # Release the lock on the file
        fcntl.flock(f, fcntl.LOCK_UN)
else:
    # Save the DataFrame to a CSV file "loss.csv"
    df.to_csv(output_file_name, index=False) 

