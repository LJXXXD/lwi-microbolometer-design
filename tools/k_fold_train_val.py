
import numpy as np
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from tools import MLP, train_epoch, valid_epoch

def k_fold_train_val(train_dataset, train_params, sim_params):
    
    kfold = KFold(n_splits=train_params.k, shuffle=True, random_state=train_params.random_seed)
    
    history = []
    models = []

    for fold, (train_ids, valid_ids) in enumerate(kfold.split(train_dataset)):
        # print(valid_ids)
        # print(f'\nFOLD {fold + 1}')
    #     print('--------------------------------')
        train_subsampler = SubsetRandomSampler(train_ids)
        valid_subsampler = SubsetRandomSampler(valid_ids)
        train_loader = DataLoader(train_dataset, batch_size=train_params.batch_size, sampler=train_subsampler, num_workers=0)
        valid_loader = DataLoader(train_dataset, batch_size=train_params.batch_size, sampler=valid_subsampler, num_workers=0)

        if train_params.random_flag:
            torch.manual_seed(train_params.random_seed)
        model = MLP(num_in=sim_params.basis_funcs.shape[1], num_out=sim_params.num_substances)
        model.to(train_params.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=train_params.learning_rate)

        model_loss_records = {'train_loss': [], 'valid_loss': []}


        

        for epoch in range(train_params.num_epochs):
            train_loss = train_epoch(model, train_params.device, train_loader, train_params.criteria, optimizer)
            valid_loss = valid_epoch(model, train_params.device, valid_loader, train_params.criteria)

            train_loss = train_loss / len(train_loader.sampler)
            valid_loss = valid_loss / len(valid_loader.sampler)

            model_loss_records['train_loss'].append(train_loss)
            model_loss_records['valid_loss'].append(valid_loss)


        

        model_loss_records['train_loss'] = np.asarray(model_loss_records['train_loss'])
        model_loss_records['valid_loss'] = np.asarray(model_loss_records['valid_loss'])
        history.append(model_loss_records)
        models.append(model)
    
    return history, models


