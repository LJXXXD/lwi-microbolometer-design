
import math
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm.notebook import tqdm

from simulator import simulator
from tools import ballsINTObins, RMSELoss, train_epoch, valid_epoch, test_epoch


# Define - Dataset, Model
    
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, samples, labels):
        
        super(Dataset, self).__init__()

        self.dataset = []
        
        for sample, label in zip(samples, labels):
            self.dataset.append((torch.tensor(sample).float(), torch.tensor(label).float()))

    def __getitem__(self, index):
                                        
        return self.dataset[index]
        
    def __len__(self):
        
        return len(self.dataset)


    
class MLP(nn.Module):

    def __init__(self, num_in, num_out):
        super().__init__()
        
        self.network = nn.Sequential(nn.Linear(num_in, 32),
                                     nn.ReLU(),
                                     nn.Linear(32, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, num_out),
                                     nn.Sigmoid())

    def forward(self, x):
        return self.network(x)
    

class Parameters:
    def __init__(self, air_trans_file, air_RI, atm_dist_ratio, basis_func_file,
                 num_substances, spectra_file, substances_emit_file):
        self.air_trans_file = air_trans_file
        self.air_RI = air_RI
        self.atm_dist_ratio = atm_dist_ratio
        self.basis_func_file = basis_func_file
        self.num_substances = num_substances
        self.spectra_file = spectra_file
        self.substances_emit_file = substances_emit_file


def select_basis_funcs(num_bf):

    # Simulation parameters
   
    sim_params = Parameters(air_trans_file='./data/Test 2 - 21 Substances/Air transmittance.xlsx',
                            air_RI=1,
                            atm_dist_ratio=0.11,
                            basis_func_file='./data/Test 2 - 21 Substances/Basis functions.xlsx',
                            num_substances=3,
                            spectra_file='./data/Test 2 - 21 Substances/spectra.xlsx',
                            substances_emit_file='./data/Test 2 - 21 Substances/substances.xlsx')

    # Enviroment related parameters
    temp_K = 293.15 # Environmental temperature in K
    air_trans = np.array(pd.read_excel(sim_params.air_trans_file, header=None))
    air_trans = air_trans[:, 1:]
    atm_dist_ratio = sim_params.atm_dist_ratio # Atomsphere distance ratio
    air_RI = sim_params.air_RI # Refractive index of air

    # Sensor related parameters
    basis_funcs = np.array(pd.read_excel(sim_params.basis_func_file, header=None))
    basis_funcs = basis_funcs[:, 1:]
    combs = itertools.combinations(range(basis_funcs.shape[1]), num_bf)

    # Substance related parameters
    num_substances = sim_params.num_substances
    spectra = np.array(pd.read_excel(sim_params.spectra_file, header=None))
    substances_emit = np.array(pd.read_excel(sim_params.substances_emit_file, header=None))
    substances_emit = substances_emit[:, 0:sim_params.num_substances]
    # Material mixture proportion
    mat_proportion = ballsINTObins(10, sim_params.num_substances).transpose() / 10


    for comb in combs:
        data = []
        labels = []
        b_funcs = basis_funcs[:, comb]
        for i in range(mat_proportion.shape[1]):
            weights = mat_proportion[:, i]
            mat_em = np.average(substances_emit, weights=weights, axis=1)
            mat_em = np.expand_dims(mat_em, 1)
            out = simulator(spectra, mat_em, temp_K, air_trans, atm_dist_ratio, air_RI, b_funcs)
            data.append(out)
            labels.append(weights)


        # Training parameters
        dataset = Dataset(data, labels)

        batch_size = len(data) // 10

        train_percentage = 0.8
        train_size = int(train_percentage * len(data))
        test_size = len(dataset) - train_size
        torch.manual_seed(28)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        criterions = [nn.L1Loss(), RMSELoss, nn.MSELoss()]

        learning_rate = 1e-3

        num_epochs = 1000

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("mps")
        device = torch.device("cpu")

        kfold = KFold(n_splits=5, shuffle=True)

        # Train - k fold
        history = []
        models = []

        for fold, (train_ids, valid_ids) in enumerate(kfold.split(train_dataset)):
            print(f'\nFOLD {fold + 1}')
        #     print('--------------------------------')
            train_subsampler = SubsetRandomSampler(train_ids)
            valid_subsampler = SubsetRandomSampler(valid_ids)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler)
            valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_subsampler)
            
            model = MLP(num_in=b_funcs.shape[1], num_out=sim_params.num_substances)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            model_loss_records = {'train_loss': [], 'valid_loss': []}
            for epoch in tqdm(range(num_epochs)):
                train_loss = train_epoch(model, device, train_loader, criterions, optimizer)
                valid_loss = valid_epoch(model, device, valid_loader, criterions)
                
                train_loss = train_loss / len(train_loader.sampler)
                valid_loss = valid_loss / len(valid_loader.sampler)
                
                model_loss_records['train_loss'].append(train_loss)
                model_loss_records['valid_loss'].append(valid_loss)
                
            model_loss_records['train_loss'] = np.asarray(model_loss_records['train_loss'])
            model_loss_records['valid_loss'] = np.asarray(model_loss_records['valid_loss'])
            history.append(model_loss_records)
            models.append(model)

        loss = []
        loss_func_index = 0
        for i in range (5):
            loss.append(history[i]["valid_loss"][:, loss_func_index][-1])
        best_model_index = np.argmin(loss)
        model = models[best_model_index]
        test_loss, pred_list, targ_list = test_epoch(model, device, test_loader, criterions)
        
        print(comb, 'test loss:', test_loss)

    return








if __name__ == '__main__':




    select_basis_funcs(6)






