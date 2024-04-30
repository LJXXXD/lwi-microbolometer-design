import ast
import fcntl
from io import BytesIO
import itertools
import os
import time


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from openpyxl.drawing.image import Image
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.utils import get_column_letter
import pandas as pd
from scipy import special
import scipy.stats as stats
import torch
import torch.nn as nn
from tqdm import tqdm


import sys
sys.path.insert(0, os.path.abspath('.'))
from tools import Sim_Parameters, Train_Parameters, create_dataset, RMSELoss, train_val_test, load_data, simulator



matplotlib.use("TkAgg") # Changing the backend to QT solves ctrl-C not quiting issue in terminal




# User Input
# substance_ind_list =  [1, 3, 5, 18]
# substance_ind_list =  [6, 7, 8, 12]
# substance_ind_list =  [0, 6, 11, 13]
# substance_ind_list =  [3, 4, 5, 12]
# substance_ind_list = [1, 3, 5, 7, 9, 10, 14, 18]
substance_ind_list =  [0, 1, 2, 3]
substance_ind_list.sort()

basis_func_ind_list = [0, 1, 2, 3, 4, 5, 6]
# basis_func_ind_list = [0, 1,  2, 5]
basis_func_ind_list.sort()

# temp_K_list = np.asarray(list(range(-30, 51, 10))) + 273.15
temp_K_list = np.asarray([293.15])
temp_K_list.sort()

# test_aggregation = np.asarray([1])

train_noise_list = [0]
train_noise_list = [i * 0.2 for i in range(11)]
# test_noise_list = [0]
test_noise_list = [i * 0.2 for i in range(11)]


air_trans, basis_funcs, spectra, substances_emit = load_data(air_trans_file='./data/Test 3 - 4 White Powers/Air transmittance.xlsx',
                                                             basis_func_file='./data/Test 3 - 4 White Powers/Basis functions_4-20um.xlsx',
                                                             spectra_file='./data/Test 3 - 4 White Powers/white_powders_spectra.xlsx', 
                                                             substances_emit_file='./data/Test 3 - 4 White Powers/white_powders.xlsx',)


substance_names = np.array(pd.read_excel('./data/Test 3 - 4 White Powers/white_powders_names.xlsx', header=None))

# air_trans, basis_funcs, spectra, substances_emit = load_data(air_trans_file='./data/Test 2 - 21 Substances/Air transmittance.xlsx',
#                                                              basis_func_file='./data/Test 2 - 21 Substances/Basis functions.xlsx',
#                                                              spectra_file='./data/Test 2 - 21 Substances/spectra.xlsx', 
#                                                              substances_emit_file='./data/Test 2 - 21 Substances/substances.xlsx')

# substance_names = np.array(pd.read_excel('./data/Test 2 - 21 Substances/sequence.xlsx', header=None))

# Start
print('\nsubstances', str(substance_ind_list))

results = []

for temp_K_ind, temp_K in enumerate(temp_K_list):
    sub_combs = list(itertools.combinations(range(4), 4))

    for substance_ind_list in sub_combs:
        for num_basis_func_ind, num_basis_func in enumerate(range(1, len(basis_func_ind_list)+1)):
            if num_basis_func != len(substance_ind_list):
                continue
            basis_func_combs = list(itertools.combinations(range(len(basis_func_ind_list)), num_basis_func))
            basis_func_combs = [tuple([basis_func_ind_list[i] for i in ind]) for ind in basis_func_combs]
            # print(basis_func_combs)

            for comb_ind, comb in enumerate(tqdm(basis_func_combs)):
                sim_params = Sim_Parameters(air_trans=air_trans,
                                            air_RI=1,
                                            atm_dist_ratio=0.11,
                                            basis_funcs=basis_funcs,
                                            basis_func_comb=comb,
                                            substance_ind_list=substance_ind_list,
                                            spectra=spectra,
                                            substances_emit=substances_emit,
                                            temp_K=temp_K,
                                            percentage_step=0.1)

                dataset = create_dataset(sim_params)

                # A_matrix = []
                # # calc A matrix ####################################################
                # for idx in substance_ind_list:
                #     sub_signal = substances_emit[:, idx]
                #     sub_signal = np.expand_dims(sub_signal, 1)
                    
                #     out = simulator(sim_params, sub_signal)  # Apply simulator() function to the column
                #     A_matrix.append(out)
                # # calc A matrix ####################################################
                # A_matrix = np.asarray(A_matrix).transpose()
                # A_det = np.linalg.det(A_matrix)
                # A_con = np.linalg.cond(A_matrix, p=2)
                # correlation_matrix = np.corrcoef(A_matrix)
                # A_avg_cor = np.mean(correlation_matrix[np.triu_indices(correlation_matrix.shape[0], k=1)])
                # cov_matrix = np.cov(A_matrix)
                # A_cov_trace = np.trace(cov_matrix)
                # A_sum_covariances = np.sum(cov_matrix) - np.trace(cov_matrix)
                # from scipy.spatial import distance
                # distances = distance.cdist(A_matrix.T, A_matrix.T, 'euclidean')
                # A_mean_disimilarity = np.mean(distances)

    

                for train_noise in train_noise_list:
                    for test_noise in test_noise_list:
                        train_params = Train_Parameters(train_percentage=0.8,
                                                        # batch_size=len(dataset) // 10,
                                                        batch_size=100,
                                                        criterions=[nn.L1Loss(), RMSELoss, nn.MSELoss()],
                                                        # criterions=[nn.MSELoss()],
                                                        loss_func_names = ['L1Loss', 'RMSELoss', 'MSELoss'],
                                                        learning_rate=2e-3,
                                                        num_epochs=200,
                                                        device=torch.device("cpu"),
                                                        k_fold_flag=True,
                                                        k=5,
                                                        random_flag=True,
                                                        random_seed=28,
                                                        train_noise=train_noise,
                                                        test_noise=test_noise)
                        
                        history, models, best_model_index, avg_test_loss, pred_list, targ_list = train_val_test(dataset, train_params, sim_params)

                        loss_values = dict(zip(['L1Loss', 'RMSELoss', 'MSELoss'], avg_test_loss))

                        row = {
                            'Temperature_K': temp_K,
                            'Substance Number': len(substance_ind_list),
                            'Substance Comb': tuple(substance_ind_list),
                            'Basis Function Number': num_basis_func,
                            'Basis Function Comb': comb,
                            'Train Noise': train_params.train_noise,
                            'Test Noise': train_params.test_noise,
                            # 'A determinant': A_det,
                            # 'A conditional number': A_con,
                            # 'A avg cor': A_avg_cor,
                            # 'A cov trace': A_cov_trace,
                            # 'A sum cov exc trace': A_sum_covariances,
                            # 'A mean disimilarity': A_mean_disimilarity,
                            **loss_values,
                            'Best Model': models[best_model_index],
                        }

                        diff_list = pred_list - targ_list

                        perc95_interval = []


                        # Function to create a sparkline for grouped data
                        # def create_grouped_sparkline(grouped_diffs):
                        #     plt.figure(figsize=(1.5, 0.6))  # Adjust figure size for sparkline
                        #     plt.bar(range(len(grouped_diffs)), grouped_diffs, width=0.5)
                        #     plt.axis('off')
                        #     buf = BytesIO()
                        #     plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=300)  # Increase DPI for better resolution
                        #     plt.close()
                        #     buf.seek(0)
                        #     return Image(buf)


                        # Group data and calculate mean absolute differences
                        def group_data_and_calculate(targ_list, diff_list):
                            grouped_diffs = []
                            for i in range(10):
                                lower_bound = i * 0.1
                                upper_bound = lower_bound + 0.1

                                if i == 9:  # Special case for the last group to include 1.0
                                    mask = (targ_list >= lower_bound) & (targ_list <= upper_bound)
                                else:
                                    mask = (targ_list >= lower_bound) & (targ_list < upper_bound)

                                grouped_diff = np.mean(np.abs(diff_list[mask])) if np.any(mask) else 0
                                grouped_diffs.append(grouped_diff)
                            return grouped_diffs

                        

                        for i, sub_ind in enumerate(substance_ind_list):

                            diff_per_sub = diff_list[:, i]
                            targ_per_sub = targ_list[:, i]
                            # print(len(diff_per_sub))
                            row[f'Substance {sub_ind} avg diff'] = np.mean(np.abs(diff_per_sub))
                            grouped_diffs = group_data_and_calculate(targ_per_sub, diff_per_sub)

                            # Store each value of grouped_diffs in a separate column
                            for group_index, diff_value in enumerate(grouped_diffs):
                                row[f'Substance {sub_ind} Group {group_index}'] = diff_value


                            # Leave an empty column for manually adding sparklines in Excel
                            row[f'Substance {sub_ind} Sparkline'] = ''

                            # params = stats.norm.fit(data)
                            # data_mean, data_std = params
                            ae, loce, scalee = stats.skewnorm.fit(diff_per_sub)
                            
                            # Add mean and std to row
                            # row[f'Substance {sub_ind} skewnorm a'] = ae
                            # row[f'Substance {sub_ind} skewnorm loc'] = loce
                            # row[f'Substance {sub_ind} skewnorm scale'] = scalee


                            # Create a skewed normal distribution object
                            dist = stats.skewnorm(ae, loce, scalee)


                            
                            # for confidence_perc in (0.68, 0.95, 0.997):
                            for confidence_perc in [0.95]:
                                # Calculate the quantiles for a specific percentage
                                lower_quantile = dist.ppf(0.5-confidence_perc/2)  # Lower quantile covering % of the data
                                upper_quantile = dist.ppf(0.5+confidence_perc/2)  # Upper quantile covering % of the data

                                # row[f'Substance {sub_ind} skewnorm {confidence_perc*100}% low'] = lower_quantile
                                # row[f'Substance {sub_ind} skewnorm {confidence_perc*100}% high'] = upper_quantile
                                # row[f'Substance {sub_ind} skewnorm {confidence_perc*100}% interval range'] = upper_quantile - lower_quantile

                                
                                if confidence_perc == 0.95:
                                    perc95_interval.append(upper_quantile - lower_quantile)
                            
                        # print(perc95_interval)
                        # perc95_min = np.min(perc95_interval)
                        # row[f'MIN skewnorm 95% interval range'] = perc95_min
                        # perc95_max = np.max(perc95_interval)
                        # row[f'MAX skewnorm 95% interval range'] = perc95_max
                        perc95_avg = np.mean(perc95_interval)
                        row[f'AVG skewnorm 95% interval range'] = perc95_avg


                            
                        # Append the row to the results list
                        results.append(row)

# Create a pandas DataFrame from the results list
df = pd.DataFrame(results)
df[['Train Noise', 'Test Noise']] = df[['Train Noise', 'Test Noise']].astype(float)

data_types = df.dtypes
# print(data_types)


output_pickle = 'loss.pkl'
# output_csv = output_pickle.split('.')[0] + '.csv'
output_excel = output_pickle.split('.')[0] + '.xlsx'

# Check if the file "loss.pkl" already exists
if os.path.isfile(output_pickle):

    # If the file exists, load the existing DataFrame from the file
    df_existing = pd.read_pickle(output_pickle)

    # Open the file in binary mode
    with open(output_pickle, 'w') as f:
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
                    'Basis Function Comb', 'Train Noise', 'Test Noise'], keep='first')
        df = df.sort_values(by=df.columns[:7].tolist())

        # Save the DataFrame to a CSV file "loss.csv"
        df.to_pickle(output_pickle)

        # Drop 'Best Model' column and save to csv
        df = df.drop(['Best Model'], axis=1)
        # df.to_csv(output_csv, index=False)

        df.to_excel(output_excel, index=False)  # Save to Excel file

        # Release the lock on the file
        fcntl.flock(f, fcntl.LOCK_UN)
else:
    # Save the DataFrame to a CSV file "loss.csv"
    df.to_pickle(output_pickle)

    # Drop 'Best Model' column and save to csv
    df = df.drop(['Best Model'], axis=1)
    # df.to_csv(output_csv, index=False)

    df.to_excel(output_excel, index=False)  # Save to Excel file