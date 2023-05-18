
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from tools import RMSELoss, k_fold_train_val, test_epoch

def train_val_test(dataset, train_params, sim_params):
    train_size = int(train_params.train_percentage * len(dataset))
    test_size = len(dataset) - train_size
    if train_params.random_flag:
        torch.manual_seed(train_params.random_seed)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    
    if train_params.k_fold_flag == True:
        history, models = k_fold_train_val(train_dataset, train_params, sim_params)

    final_loss_list = [history[i]["valid_loss"][:, -1][-1] for i in range(train_params.k)]
    best_model_index = np.argmin(final_loss_list)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_loss, pred_list, targ_list = test_epoch(models[best_model_index], train_params.device, test_loader, train_params.criterions)

    test_loss[0] = test_loss[0] * sim_params.num_substances
    avg_test_loss = test_loss / len(test_dataset)

    return history, models, best_model_index, avg_test_loss, pred_list, targ_list



def train_val_test_pretrained(dataset, train_params, sim_params, model):
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
    test_loss, pred_list, targ_list = test_epoch(model, train_params.device, test_loader, train_params.criterions)
    

    test_loss = test_loss / len(test_dataset)

    return model, model, model, test_loss, pred_list, targ_list