import fcntl
import itertools
import multiprocessing
from multiprocessing import Manager
import os
import sys
import time
import yaml

import matplotlib
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools import Sim_Parameters, Train_Parameters, create_dataset, train_val_test, load_config, load_excel_data, get_list_values, calculate_runtime

# Changing the backend to QT solves ctrl-C not quiting issue in terminal
matplotlib.use("TkAgg")


def main():

    # Load variables from config.yaml

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        config = load_config(config_file)
        print(f"Loaded configuration from {config_file}")
    else:
        print("No configuration file provided. Exiting")
        sys.exit(1)


    basis_func_combs = list(itertools.combinations(config['basis_func_ind_list'], config['num_bf']))

    print('cpu count:', multiprocessing.cpu_count())

    with Manager() as manager:
        shared_config = manager.dict(config)
        args = [(comb, shared_config) for comb in basis_func_combs]
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            pool.starmap(worker_function, args)
            pool.close()
            pool.join()


    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #     pool.map(worker_function, basis_func_combs)
    #     pool.close()
    #     pool.join()

    # # Create a multiprocessing pool
    # pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # # Distribute the work among the worker processes
    # pool.map(worker_function, basis_func_combs)

    # # Close the pool and wait for the work to finish
    # pool.close()
    # pool.join()


def worker_helper(args):
    return worker_function(*args)


def worker_function(comb, config):
    print(f"Worker process {os.getpid()} started for combination: {comb}")
    # for comb in tqdm(basis_func_combs):
    # config_file = sys.argv[1]
    # config = load_config(config_file)
    results = []
    for temp_K in get_list_values(config['temp_K_list']):    
        file_paths = config['file_paths']
        sim_params = Sim_Parameters(air_trans=load_excel_data(file_paths['air_trans_file']),
                                    air_RI=config['air_refractive_index'],
                                    atm_dist_ratio=config['atm_dist_ratio'],
                                    basis_funcs=load_excel_data(file_paths['basis_func_file']),
                                    basis_func_comb=comb,
                                    substance_ind_list=config['substance_ind_list'],
                                    spectra=load_excel_data(file_paths['spectra_file']),
                                    substances_emit=load_excel_data(file_paths['substances_emit_file']),
                                    temp_K=temp_K,
                                    percentage_step=config['percentage_step'])
        dataset = create_dataset(sim_params)
        for train_noise in tqdm(get_list_values(config['train_noise'])):
            for test_noise in get_list_values(config['test_noise']):

                

                # Get criteria from file
                criteria = []
                for name in config['loss_func_names']:
                    # Check if the loss function is a custom one or from nn module
                    loss_func_class = getattr(nn, name, None) or globals().get(name)
                    if loss_func_class:
                        criteria.append(loss_func_class())
                    else:
                        raise ValueError(f"Loss function {name} not found.")

                train_params = Train_Parameters(train_percentage=config['train_percentage'],
                                                batch_size=config['batch_size'],
                                                criteria=criteria,
                                                loss_func_names = config['loss_func_names'],
                                                learning_rate=config['learning_rate'],
                                                num_epochs=config['num_epochs'],
                                                device=torch.device(config['device']),
                                                k_fold_flag=config['k_fold_flag'],
                                                k=config['k'],
                                                random_flag=config['random_flag'],
                                                random_seed=config['random_seed'],
                                                train_noise=train_noise,
                                                test_noise=test_noise)

                history, models, best_model_index, avg_test_loss, pred_list, targ_list = train_val_test(dataset, train_params, sim_params)

                loss_values = dict(zip(config['loss_func_names'], avg_test_loss))
                row = {
                    'Temperature_K': temp_K,
                    'Substance Number': len(sim_params.substance_ind_list),
                    'Substance Comb': tuple(sim_params.substance_ind_list),
                    'Basis Function Number': config['num_bf'],
                    'Basis Function Comb': comb,
                    'Train Noise': train_params.train_noise,
                    'Test Noise': train_params.test_noise,
                    **loss_values,
                    # 'Best Model': models[best_model_index],
                }


                # Calculate 95% confident interval for each substance then take average
                diff_list = pred_list - targ_list
                perc95_interval = []
                for i, sub_ind in enumerate(sim_params.substance_ind_list):
                    data = diff_list[:, i]
                    ae, loce, scalee = stats.skewnorm.fit(data)
                    dist = stats.skewnorm(ae, loce, scalee)
                    lower_quantile = dist.ppf(0.5-config['confidence']/2)  # Lower quantile covering % of the data
                    upper_quantile = dist.ppf(0.5+config['confidence']/2)  # Upper quantile covering % of the data
                    interval = upper_quantile - lower_quantile
                    perc95_interval.append(interval)
                    row[f'Substance {sub_ind} skewnorm 95% interval'] = interval

                perc95_avg = np.mean(perc95_interval)
                row[f'AVG skewnorm 95% interval range'] = perc95_avg


                
                def group_data(array, matrix):
                    bins = np.arange(0.1, 1, 0.1)
                    bin_indices = np.digitize(array, bins, right=False)
                    bin_matrices = []
                    for i in range(0, len(bins)+1):
                        mask = bin_indices == i
                        bin_m = matrix[mask]
                        if len(bin_m) > 0:  # Check if there are elements in this bin
                            bin_matrices.append(bin_m)
                        else:
                            bin_matrices.append(np.asarray([0]))
                    return bin_matrices
                
                # for i, sub_ind in enumerate(substance_ind_list):
                #     targ_per_sub = targ_list[:, i]
                #     grouped_diffs = group_data(targ_per_sub, diff_list)
                #     bined_L1 = []
                #     for g_d in grouped_diffs:
                #         if g_d != np.asarry([]):
                #             loss = torch.nn.L1Loss(g_d)
                #             bined_L1.append(loss)
                #         else:
                #             bined_L1.append(-0.1)




                # Group data and calculate mean absolute differences
                # def group_data_and_calculate(targ_list, diff_list):
                #     grouped_diffs = []
                #     for i in range(10):
                #         lower_bound = i * 0.1
                #         upper_bound = lower_bound + 0.1

                #         if i == 9:  # Special case for the last group to include 1.0
                #             mask = (targ_list >= lower_bound) & (targ_list <= upper_bound)
                #         else:
                #             mask = (targ_list >= lower_bound) & (targ_list < upper_bound)

                #         grouped_diff = np.mean(np.abs(diff_list[mask])) if np.any(mask) else 0
                #         grouped_diffs.append(grouped_diff)
                #     return grouped_diffs
                
                for i, sub_ind in enumerate(sim_params.substance_ind_list):

                    diff_per_sub = diff_list[:, i]
                    targ_per_sub = targ_list[:, i]
                    # print(len(diff_per_sub))
                    row[f'Substance {sub_ind} avg diff'] = np.mean(np.abs(diff_per_sub))
                    grouped_diffs = group_data(targ_per_sub, diff_per_sub)

                    # Store each value of grouped_diffs in a separate column
                    for group_index, diff_value in enumerate(grouped_diffs):
                        row[f'Substance {sub_ind} Group {group_index}'] = np.mean(np.abs(diff_value))
                    
                # Append the row to the results list
                results.append(row)

                

    # Create a pandas DataFrame from the results list
    df = pd.DataFrame(results)
    df[['Train Noise', 'Test Noise']] = df[['Train Noise', 'Test Noise']].astype(float)

    # output_pickle = config['output_folder'] + config['output_file_name'] + '.pkl'
    output_excel = config['output_folder'] + config['output_file_name'] + '.xlsx'

    # Check if the file "loss.pkl" already exists
    if os.path.isfile(output_excel):

        # If the file exists, load the existing DataFrame from the file
        df_existing = pd.read_excel(output_excel)

        # Open the file in binary mode
        with open(output_excel, 'w') as f:
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
            # df.to_pickle(output_pickle)

            # Drop 'Best Model' column and save to csv
            # df = df.drop(['Best Model'], axis=1)
            # df.to_csv(output_csv, index=False)

            df.to_excel(output_excel, index=False)  # Save to Excel file

            # Release the lock on the file
            fcntl.flock(f, fcntl.LOCK_UN)
    else:
        # Save the DataFrame to a CSV file "loss.csv"
        # df.to_pickle(output_pickle)

        # Drop 'Best Model' column and save to csv
        # df = df.drop(['Best Model'], axis=1)
        # df.to_csv(output_csv, index=False)

        df.to_excel(output_excel, index=False)  # Save to Excel file


if __name__ == "__main__":
    main()