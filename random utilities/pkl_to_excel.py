import itertools
import sys

import matplotlib
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools import Sim_Parameters, Train_Parameters, create_dataset, train_val, load_config, load_excel_data, get_list_values, df_to_pickle, df_to_excel

# Changing the backend to QT solves ctrl-C not quiting issue in terminal
matplotlib.use("TkAgg")


if len(sys.argv) > 1:
    config_file = sys.argv[1]
    config = load_config(config_file)
    print(f"Loaded configuration from {config_file}")
else:
    print("No configuration file provided. Exiting")
    sys.exit(1)

input_folder = os.path.join(config['output_folder'], config['output_file_name'] + '.' + 'pkl')

df = pd.read_pickle(input_folder)


df = df.drop(['Best Model', 'Test Dataset'], axis=1)
print(df.columns)
df_to_excel(df, config['output_folder'], config['output_file_name'], df.columns.to_list())

