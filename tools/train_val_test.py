

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tools import RMSELoss, k_fold_train_val, test_epoch, NoisyDataset



def train_val(train_dataset, train_params):
    # train_size = int(train_params.train_percentage * len(train_dataset))
    # test_size = len(train_dataset) - train_size
    # if train_params.random_flag:
    #     torch.manual_seed(train_params.random_seed)
    # train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

    # if train_params.train_noise != 0:
    #     train_dataset = NoisyDataset(train_dataset, train_params.train_noise)

    # if train_params.test_noise != 0:
    #     test_dataset = NoisyDataset(test_dataset, train_params.test_noise)

    if train_params.k_fold_flag == True:
        history, models = k_fold_train_val(train_dataset, train_params)

    final_loss_list = [history[i]["valid_loss"][:, -1][-1] for i in range(train_params.k)]
    best_model_index = np.argmin(final_loss_list)

    return history, models, best_model_index



def train_val_test(dataset, train_params):
    train_size = int(train_params.train_percentage * len(dataset))
    test_size = len(dataset) - train_size
    if train_params.random_flag:
        torch.manual_seed(train_params.random_seed)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    if train_params.train_noise != 0:
        train_dataset = NoisyDataset(train_dataset, train_params.train_noise)

    if train_params.k_fold_flag == True:
        history, models = k_fold_train_val(train_dataset, train_params)
        
    final_loss_list = [history[i]["valid_loss"][:, -1][-1] for i in range(train_params.k)]
    best_model_index = np.argmin(final_loss_list)

    avg_test_loss_dict = {}
    pred_list_dict = {}
    targ_list_dict = {}
    for test_noise in train_params.test_noise_list:
        if test_noise != 0:
            test_dataset_noise = NoisyDataset(test_dataset, test_noise)
        else:
            test_dataset_noise = test_dataset

        test_loader = DataLoader(test_dataset_noise, batch_size=1, shuffle=False)
        test_loss, pred_list, targ_list = test_epoch(models[best_model_index], train_params.device, test_loader, train_params.criteria)

        # test_loss[0] = test_loss[0] * sim_params.num_substances
        avg_test_loss = test_loss / len(test_dataset_noise)

        avg_test_loss_dict[test_noise] = avg_test_loss
        pred_list_dict[test_noise] = pred_list
        targ_list_dict[test_noise] = targ_list


    return history, models, best_model_index, avg_test_loss_dict, pred_list_dict, targ_list_dict



def train_val_test_pretrained(dataset, train_params, model):
    

    train_size = int(train_params.train_percentage * len(dataset))
    test_size = len(dataset) - train_size
    if train_params.random_flag:
        torch.manual_seed(train_params.random_seed)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    
    # if train_params.k_fold_flag == True:
    #     history, models = k_fold_train_val(train_dataset, train_params, sim_params)

    # final_loss_list = [history[i]["valid_loss"][:, -1][-1] for i in range(train_params.k)]
    # best_model_index = np.argmin(final_loss_list)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_loss, pred_list, targ_list = test_epoch(model, train_params.device, test_loader, train_params.criteria)
    

    test_loss = test_loss / len(test_dataset)

    return model, model, model, test_loss, pred_list, targ_list