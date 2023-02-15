
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from simulator import simulator


def percentage_test(x, y):
    
    x_train = torch.FloatTensor(x)
    y_train = torch.FloatTensor(y)

    model = MLP()
    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.MSELoss()
    lr = 1e-4
    # optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    model.train()
    epoch = 10000
    loss_all = []
    for ep in range(epoch):
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(x_train)
        # Compute Loss
        loss = criterion(y_pred.squeeze(), y_train)
        loss_all.append(loss.item())
    #     if epoch % 1000 == 0:
    #         print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        # Backward pass
        loss.backward()
        optimizer.step()

    plt.figure()
    plt.plot(loss_all)
    plt.show()


    model.eval()
    y_pred = model(x_train)
    y_pred = y_pred.detach().numpy()
    print(y_pred)
    print(y_pred.shape)
    return


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(7, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


if __name__ == '__main__':
    # Enviroment related parameters
    temp_K = 293.15 # Environmental temperature in K
    air_trans = np.array(pd.read_excel('./data/Test 2 - 21 Substances/Air transmittance.xlsx', header=None))
    air_trans = air_trans[:, 1:]
    atm_dist_ratio = 0.11 # Atomsphere distance ratio
    air_RI = 1 # Refractive index of air

    # Sensor related parameters
    basis_funcs = np.array(pd.read_excel('./data/Test 2 - 21 Substances/Basis functions.xlsx', header=None))
    basis_funcs = basis_funcs[:, 1:]
    
    # Substance related parameters
    spectra = np.array(pd.read_excel('./data/Test 2 - 21 Substances/spectra.xlsx', header=None))
    substances_emit = np.array(pd.read_excel('./data/Test 2 - 21 Substances/substances.xlsx', header=None))
    substances_emit = substances_emit[:, [4, 16]]

    mat_proportion = np.array(pd.read_excel('./data/Test 2 - 21 Substances/proportion_NN_test.xlsx', header=None)) # Material mixture proportion

    data = []
    labels = []
    for i in range(mat_proportion.shape[1]):
        weights = mat_proportion[:, i]
        mat_em = np.average(substances_emit, weights=weights, axis=1)
        mat_em = np.expand_dims(mat_em, 1)
        out = simulator(spectra, mat_em, temp_K, air_trans, atm_dist_ratio, air_RI, basis_funcs)
        data.append(out)
        labels.append(weights)


    percentage_test(data, labels)