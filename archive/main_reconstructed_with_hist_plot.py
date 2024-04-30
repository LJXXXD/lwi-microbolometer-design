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
substance_ind_list = [0, 1, 2, 3, 4, 5]
# substance_ind_list = [1, 7, 9, 10, 14]
substance_ind_list.sort()

basis_func_ind_list = [0, 1, 2, 3, 4, 5, 6]
basis_func_ind_list.sort()

# temp_K_list = np.asarray(list(range(-20, 51, 10))) + 273.15
temp_K_list = np.asarray([293.15])
temp_K_list.sort()


air_trans, basis_funcs, spectra, substances_emit = load_data(air_trans_file='./data/Test 2 - 21 Substances/Air transmittance.xlsx',
                                                             basis_func_file='./data/Test 2 - 21 Substances/Basis functions.xlsx',
                                                             spectra_file='./data/Test 2 - 21 Substances/spectra.xlsx', 
                                                             substances_emit_file='./data/Test 2 - 21 Substances/substances.xlsx')


# Start
print('\nsubstances', str(substance_ind_list))

results = []

for temp_K_ind, temp_K in enumerate(temp_K_list):

    for num_basis_func_ind, num_basis_func in enumerate(range(1, len(basis_func_ind_list)+1)):
        basis_func_combs = list(itertools.combinations(range(len(basis_func_ind_list)), num_basis_func))

        for comb_ind, comb in enumerate(tqdm(basis_func_combs)):
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
                                            random_seed=28)
            
            history, models, best_model_index, avg_test_loss, pred_list, targ_list = train_val_test(dataset, train_params, sim_params)

            loss_values = dict(zip(['L1Loss', 'RMSELoss', 'MSELoss'], avg_test_loss))

            row = {
                'Temperature_K': temp_K,
                'Substance Number': len(substance_ind_list),
                'Substance Comb': tuple(substance_ind_list),
                'Basis Function Number': num_basis_func,
                'Comb of Basis Functions': comb,
                **loss_values,
            }

            diff_list = pred_list - targ_list

            num_subplots = len(substance_ind_list)

            # Determine the number of rows and columns for the subplots
            num_rows = int(np.sqrt(num_subplots))
            num_cols = int(np.ceil(num_subplots / num_rows))

            # Create a new figure and subplots
            fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 8))

            # Flatten the axs array if necessary
            if num_subplots == 1:
                axs = np.array([axs])

            # Loop over the data and subplots
            for i, ax in enumerate(axs.flat):
                # Plot histogram for the current data
                if i == num_subplots:
                    break
                n, bins, _ = ax.hist(diff_list[:, i], bins=15, alpha=0.7)
                ax.margins(0)

                # Fit the best-fitting normal distribution to the current data
                data = diff_list[:, i]
                params = stats.norm.fit(data)
                data_mean, data_std = params
                ae, loce, scalee = stats.skewnorm.fit(data)
                
                # Add mean and std to row
                row[f'Substance {substance_ind_list[i]} pred mean'] = data_mean
                row[f'Substance {substance_ind_list[i]} pred std'] = data_std

                # Create an array of x-values for the assumed normal distribution
                x = np.linspace(diff_list[:, i].min(), diff_list[:, i].max(), 1000)

                # Create a skewnorm distribution object
                dist = stats.skewnorm(ae, loce, scalee)

                # Calculate the y-values of the assumed normal distribution
                y =  dist.pdf(x)

                # Scale the distribution curve based on the maximum frequency in the histogram
                max_freq = np.max(n)
                y_scaled = y * max_freq / np.max(y)

                # Plot the scaled assumed normal distribution
                ax.plot(x, y_scaled, color='red', linewidth=2)

                # Set the title for the current subplot
                ax.set_title(f'Histogram of Substance pred error {i}')

                # Create a new y-axis on the right side for the distribution
                ax_new = ax.twinx()
                ax_new.set_ylabel('Distribution')

                # Hide the ticks and labels on the new y-axis
                ax_new.yaxis.set_tick_params(left=False, labelleft=False)
                ax_new.yaxis.set_tick_params(right=True, labelright=True)

                # Adjust the color of the new y-axis labels
                ax_new.yaxis.label.set_color('red')
                ax_new.tick_params(axis='y', colors='red')

                # Calculate the maximum value of the scaled distribution
                max_scaled_value = np.max(y)

                # Set the limits of the new y-axis to match the scaled distribution
                ax_new.set_ylim(0, max_scaled_value)  # Adjust the scale factor as needed

            # Set common y-axis label
            fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical')

            # Adjust spacing between subplots
            fig.tight_layout()

            # Show the figure
            plt.show()

            
            # Append the row to the results list
            results.append(row)

# Create a pandas DataFrame from the results list
df = pd.DataFrame(results)
print(df)

# Check if the file "loss.pkl" already exists
if os.path.isfile('loss.pkl'):

    # If the file exists, load the existing DataFrame from the file
    df_existing = pd.read_pickle('loss.pkl')

    # Open the file in binary mode
    with open('loss.pkl', 'wb') as f:
        while True:
            try:
                # Get a lock on the file
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except IOError:
                # Lock is held by another process, wait and try again
                time.sleep(1)

        # Merge df to df_existing, and if there's duplicates, keep the new values from df
        df = pd.concat([df, df_existing]).drop_duplicates(subset=['Temperature_K', 'Substance Number', 'Substance Comb', 'Basis Function Number', 'Comb of Basis Functions'], keep='first')
        df = df.sort_values(by=list(df.columns))
        print(df)

        # Save the DataFrame to the file "loss.pkl"
        df.to_pickle('loss.pkl')

        # Save the DataFrame to a CSV file "loss.csv"
        df.to_csv('loss.csv', index=False)

        # Release the lock on the file
        fcntl.flock(f, fcntl.LOCK_UN)
else:
    # Save the DataFrame to the file "loss.pkl"
    df.to_pickle('loss.pkl')

    # Save the DataFrame to a CSV file "loss.csv"
    df.to_csv('loss.csv', index=False) 