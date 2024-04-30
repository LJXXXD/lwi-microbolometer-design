import sys

import matplotlib
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools import load_config, test_epoch, df_to_excel

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

# Load the DataFrame from the pickle file
df = pd.read_pickle(config['input_file_path'])

group_criteria = config['group_criteria']
target_group = tuple(config['target_group'])
filtered_df = df.groupby(group_criteria).get_group(target_group)



# Get criteria from file
criteria = []
for name in config['loss_func_names']:
    # Check if the loss function is a custom one or from nn module
    loss_func_class = getattr(nn, name, None) or globals().get(name)
    if loss_func_class:
        criteria.append(loss_func_class())
    else:
        raise ValueError(f"Loss function {name} not found.")
    



test_df = []
# Iterate over the rows in the filtered DataFrame
for index, row in filtered_df.iterrows():
    print(index)
    model = row['Best Model']
    for i_test, r_test in filtered_df.iterrows():
        test_temp_K = r_test['Temperature_K']
        test_dataset = r_test['Test Dataset']
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        test_loss, pred_list, targ_list = test_epoch(model, torch.device(config['device']), test_loader, criteria)
        avg_test_loss = test_loss / len(test_dataset)
        loss_values = dict(zip(config['loss_func_names'], avg_test_loss))

        new_row = row[:-2].copy()
        new_row['Test_Temperature_K'] = test_temp_K
        
        for k, v in loss_values.items():
            new_row[k] = v
        test_df.append(new_row)

test_df = pd.DataFrame(test_df)
df_to_excel(test_df, config['output_folder'], config['output_file_name'], test_df.columns[:-2].to_list())