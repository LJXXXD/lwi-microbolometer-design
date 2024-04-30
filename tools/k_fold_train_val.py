
import numpy as np
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from tools import MLP, train_epoch, valid_epoch


class CustomSubsetRandomSampler(SubsetRandomSampler):
    """SubsetRandomSampler that drops the last batch only if it has exactly one sample."""
    def __init__(self, indices, batch_size):
        super().__init__(indices)
        self.indices = indices
        self.batch_size = batch_size

    def __iter__(self):
        # Shuffle indices
        indices = torch.randperm(len(self.indices)).tolist()
        indices = [self.indices[i] for i in indices]

        # Check the size of the last batch
        total_size = len(indices)
        last_batch_size = total_size % self.batch_size
        if last_batch_size == 1:
            indices = indices[:-1]  # Drop the last single index

        return iter(indices)

    def __len__(self):
        total_len = len(self.indices)
        if total_len % self.batch_size == 1:
            return total_len - 1
        return total_len



def k_fold_train_val(train_dataset, train_params):
    
    kfold = KFold(n_splits=train_params.k, shuffle=True, random_state=train_params.random_seed)
    
    history = []
    models = []

    for fold, (train_ids, valid_ids) in enumerate(kfold.split(train_dataset)):
        # print(valid_ids)
        # print(f'\nFOLD {fold + 1}')
    #     print('--------------------------------')
        train_subsampler = CustomSubsetRandomSampler(train_ids, train_params.batch_size)
        valid_subsampler = CustomSubsetRandomSampler(valid_ids, train_params.batch_size)
        train_loader = DataLoader(train_dataset, batch_size=train_params.batch_size, sampler=train_subsampler, num_workers=0)
        valid_loader = DataLoader(train_dataset, batch_size=train_params.batch_size, sampler=valid_subsampler, num_workers=0)

        if train_params.random_flag:
            torch.manual_seed(train_params.random_seed)
        model = MLP(num_in=train_params.num_in, num_out=train_params.num_out)
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


