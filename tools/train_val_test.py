
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

    loss_func_index = 2 # Needs to be fixed!
    final_loss_list = []
    for i in range(train_params.k):
        final_loss_list.append(history[i]["valid_loss"][:, loss_func_index][-1])

    avg_loss = np.mean(final_loss_list)
    best_loss = np.min(final_loss_list)
    best_model_index = np.argmin(final_loss_list)

    print("avg loss =", avg_loss)
    print("best loss =", best_loss)
    print("best model index =", best_model_index)

    test_loader = DataLoader(train_dataset, batch_size=train_params.batch_size, shuffle=True)
    best_model = models[best_model_index]
    test_loss, pred_list, targ_list = test_epoch(best_model, train_params.device, test_loader, train_params.criterions)

    print(test_loss)

    test_loss = test_loss / len(test_dataset)

    # fig, ax = plt.subplots(figsize=(6, 4))
    # ax.plot(history[best_model_index]['train_loss'][:, loss_func_index])
    # ax.plot(history[best_model_index]['valid_loss'][:, loss_func_index])

    # ax.axhline(y=test_loss[loss_func_index], color='g')

    # ax.set_ylim(bottom=0)
    # ax.set_xlabel('Epoch')

    # ax.legend(['train loss', 'valid loss', 'test loss'], 
    #       title=f'Last Loss: {"{:0.3e}".format(history[best_model_index]["valid_loss"][:, loss_func_index][-1])}')
    
    # plt.show()

    return history, models, test_loss, pred_list, targ_list