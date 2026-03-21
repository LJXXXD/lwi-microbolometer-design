import sys

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools import load_config, test_epoch, df_to_excel, calc_conf_interval

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


df_model = pd.read_pickle(config["model_file_path"])

df_testset = pd.read_pickle(config["testset_file_path"])
group_criteria = config["group_criteria"]
target_group = tuple(config["target_group"])
filtered_df_testset = df_testset.groupby(group_criteria).get_group(target_group)


# Get criteria from file
criteria = []
for name in config["loss_func_names"]:
    # Check if the loss function is a custom one or from nn module
    loss_func_class = getattr(nn, name, None) or globals().get(name)
    if loss_func_class:
        criteria.append(loss_func_class())
    else:
        raise ValueError(f"Loss function {name} not found.")


test_df = []
# Iterate over the rows in the filtered DataFrame
for index, row in df_model.iterrows():
    print(index)
    model = row["Best Model"]
    for i_test, r_test in filtered_df_testset.iterrows():
        test_atmdr = r_test["atm_dist_ratio"]
        test_dataset = r_test["Test Dataset"]
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        test_loss, pred_list, targ_list = test_epoch(
            model, torch.device(config["device"]), test_loader, criteria
        )
        avg_test_loss = test_loss / len(test_dataset)
        loss_values = dict(zip(config["loss_func_names"], avg_test_loss))

        new_row = row[:-2].copy()
        new_row["Test_atmdr"] = test_atmdr

        for k, v in loss_values.items():
            new_row[k] = v

        conf_interval = calc_conf_interval(pred_list, targ_list, config["confidence"])
        for k, v in zip(config["substance_ind_list"], conf_interval):
            new_row[f"Substance {k} 95% confidence interval"] = v
        new_row["AVG 95% confidence interval"] = np.mean(conf_interval)

        test_df.append(new_row)

test_df = pd.DataFrame(test_df)
df_to_excel(
    test_df, config["output_folder"], config["output_file_name"], test_df.columns[:-2].to_list()
)
