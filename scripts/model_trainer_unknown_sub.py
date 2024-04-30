import itertools
import sys

import matplotlib
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Subset
from tqdm import tqdm

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools import Sim_Parameters, Train_Parameters, create_dataset, train_val, load_config, load_excel_data, get_list_values, df_to_pickle, data_split, NoisyDataset

# Changing the backend to QT solves ctrl-C not quiting issue in terminal
matplotlib.use("TkAgg")



# Load variables from config.yaml

if len(sys.argv) > 1:
    config_file = sys.argv[1]
    config = load_config(config_file)
    print(f"Loaded configuration from {config_file}")
else:
    print("No configuration file provided. Exiting")
    sys.exit(1)


# Start
basis_func_combs = list(itertools.combinations(config['basis_func_ind_list'], config['num_bf']))

results = []

for comb in tqdm(basis_func_combs, desc="Combs", leave=True):
    for temp_K in tqdm(get_list_values(config['temp_K_list']), desc="Temps", leave=False):
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

        for unknown_idx in config['substance_ind_list']:


            # Indices for splitting
            train_indices = [i for i, (_, label) in enumerate(dataset) if label[unknown_idx] == 0]
            test_indices = [i for i, (_, label) in enumerate(dataset) if label[unknown_idx] != 0]

            # Creating subsets for training and testing
            train_dataset = Subset(dataset, train_indices)
            test_dataset = Subset(dataset, test_indices)
            print('unknown index', unknown_idx)
            for _, label in test_dataset:
                print(label)

            for train_noise in tqdm(get_list_values(config['train_noise']), desc="Train Noise", leave=False):
                if train_noise != 0:
                    train_dataset = NoisyDataset(train_dataset, train_params.train_noise)
                # Get criteria from file
                criteria = []
                for name in config['loss_func_names']:
                    # Check if the loss function is a custom one or from nn module
                    loss_func_class = getattr(nn, name, None) or globals().get(name)
                    if loss_func_class:
                        criteria.append(loss_func_class())
                    else:
                        raise ValueError(f"Loss function {name} not found.")

                train_params = Train_Parameters(num_in=sim_params.basis_funcs.shape[1],
                                                num_out=sim_params.num_substances,
                                                train_percentage=config['train_percentage'],
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
                                                test_noise_list=get_list_values(config['test_noise']))

                history, models, best_model_index = train_val(train_dataset, train_params)

                row = {
                    'Substance Number': len(config['substance_ind_list']),
                    'Substance Comb': tuple(config['substance_ind_list']),
                    'Unknown Sub idx': unknown_idx,
                    'Basis Function Number': config['num_bf'],
                    'Basis Function Comb': comb,
                    'Temperature_K': temp_K,
                    'Train Noise': train_noise,
                    'air_refractive_index': config['air_refractive_index'],
                    'atm_dist_ratio': config['atm_dist_ratio'],
                    'percentage_step': config['percentage_step'],
                    'train_percentage': config['train_percentage'],
                    'batch_size': config['batch_size'],
                    'learning_rate': config['learning_rate'],
                    'num_epochs': config['num_epochs'],
                    'k_fold_flag': config['k_fold_flag'],
                    'k': config['k'],
                    'random_flag': config['random_flag'],
                    'random_seed': config['random_seed'],
                    'Best Model': models[best_model_index],
                    'Test Dataset': test_dataset
                }
                
                results.append(row)

df = pd.DataFrame(results)
df[['Train Noise']] = df[['Train Noise']].astype(float)
df[['Substance Comb', 'Basis Function Comb']] = df[['Substance Comb', 'Basis Function Comb']].astype(str)


drop_duplicates_columns = df.columns[:config['drop_duplicates_columns_num']].to_list()
df_to_pickle(df, config['output_folder'], config['output_file_name'], drop_duplicates_columns)
