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
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.abspath('.'))
from tools import Sim_Parameters, Train_Parameters, create_dataset_PCA, RMSELoss, train_val_test, load_data, train_val_test_pretrained



matplotlib.use("TkAgg") # Changing the backend to QT solves ctrl-C not quiting issue in terminal




# User Input
# substance_ind_list =  [1, 3, 5, 18]
# substance_ind_list =  [6, 7, 8, 12]
# substance_ind_list =  [0, 6, 11, 13]
# substance_ind_list =  [3, 4, 5, 12]
# substance_ind_list = [1, 3, 5, 7, 9, 10, 14, 18]
substance_ind_list =  [0, 1, 2, 3]
substance_ind_list.sort()

basis_func_ind_list = [0, 1, 2, 3]
basis_func_ind_list.sort()

# temp_K_list = np.asarray(list(range(-20, 51, 10))) + 273.15
temp_K_list = np.asarray([293.15])
temp_K_list.sort()

test_aggregation = np.asarray([1])

train_noise_perc_list = [0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 1]
test_noise_perc_list = [0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 1]


air_trans, basis_funcs, spectra, substances_emit = load_data(air_trans_file='./data/Test 3 - 4 White Powers/Air transmittance.xlsx',
                                                             basis_func_file='./data/Test 3 - 4 White Powers/Basis functions_4-20um.xlsx',
                                                             spectra_file='./data/Test 3 - 4 White Powers/white_powders_spectra.xlsx', 
                                                             substances_emit_file='./data/Test 3 - 4 White Powers/white_powders.xlsx',)


substance_names = np.array(pd.read_excel('./data/Test 3 - 4 White Powers/white_powders_names.xlsx', header=None))


# Start
print('\nsubstances', str(substance_ind_list))

results = []

for temp_K_ind, temp_K in enumerate(temp_K_list):

    for num_basis_func_ind, num_basis_func in enumerate(range(1, len(basis_func_ind_list)+1)):
        if num_basis_func != 4:
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

            dataset = create_dataset_PCA(sim_params)

            for agg in test_aggregation:
                for train_noise_perc in tqdm(train_noise_perc_list):
                    for test_noise_perc in test_noise_perc_list:
                        train_params = Train_Parameters(train_percentage=0.8,
                                                        # batch_size=len(dataset) // 10,
                                                        batch_size=100,
                                                        criterions=[nn.L1Loss(), RMSELoss, nn.MSELoss()],
                                                        # criterions=[nn.MSELoss()],
                                                        loss_func_names = ['L1Loss', 'RMSELoss', 'MSELoss'],
                                                        learning_rate=1e-3,
                                                        num_epochs=100,
                                                        device=torch.device("cpu"),
                                                        k_fold_flag=True,
                                                        k=5,
                                                        random_flag=True,
                                                        random_seed=28,
                                                        train_noise_perc=train_noise_perc,
                                                        test_noise_perc=test_noise_perc,
                                                        test_aggregation=agg)
                        
                        history, models, best_model_index, avg_test_loss, pred_list, targ_list = train_val_test(dataset, train_params, sim_params)

                        loss_values = dict(zip(['L1Loss', 'RMSELoss', 'MSELoss'], avg_test_loss))

                        row = {
                            'Temperature_K': temp_K,
                            'Substance Number': len(substance_ind_list),
                            'Substance Comb': tuple(substance_ind_list),
                            'Basis Function Number': num_basis_func,
                            'Comb of Basis Functions': comb,
                            'Train Noise Max Percentage': train_params.train_noise_perc,
                            'Test Noise Max Percentage': train_params.test_noise_perc,
                            'Test Aggregation': train_params.test_aggregation,
                            **loss_values,
                        }

                        diff_list = pred_list - targ_list

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

                                row[f'Substance {sub_ind} skewnorm {confidence_perc*100}% low'] = lower_quantile
                                row[f'Substance {sub_ind} skewnorm {confidence_perc*100}% high'] = upper_quantile
                                row[f'Substance {sub_ind} skewnorm {confidence_perc*100}% interval range'] = upper_quantile - lower_quantile


                            
                        # Append the row to the results list
                        results.append(row)

# Create a pandas DataFrame from the results list
df = pd.DataFrame(results)
print(df)

# Check if the file "loss.pkl" already exists
if os.path.isfile('loss_PCA.pkl'):

    # If the file exists, load the existing DataFrame from the file
    df_existing = pd.read_pickle('loss_PCA.pkl')

    # Open the file in binary mode
    with open('loss_PCA.pkl', 'wb') as f:
        while True:
            try:
                # Get a lock on the file
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except IOError:
                # Lock is held by another process, wait and try again
                time.sleep(1)

        # Merge df to df_existing, and if there's duplicates, keep the new values from df
        df = pd.concat([df, df_existing]).drop_duplicates(subset=['Temperature_K', 'Substance Number', 'Substance Comb', 'Basis Function Number', 'Comb of Basis Functions', 'Train Noise Max Percentage', 'Test Noise Max Percentage', 'Test Aggregation'], keep='first')
        df = df.sort_values(by=list(df.columns))
        print(df)

        # Save the DataFrame to the file "loss.pkl"
        df.to_pickle('loss_PCA.pkl')

        # Save the DataFrame to a CSV file "loss.csv"
        df.to_csv('loss_PCA.csv', index=False)

        # Release the lock on the file
        fcntl.flock(f, fcntl.LOCK_UN)
else:
    # Save the DataFrame to the file "loss.pkl"
    df.to_pickle('loss_PCA.pkl')

    # Save the DataFrame to a CSV file "loss.csv"
    df.to_csv('loss_PCA.csv', index=False) 