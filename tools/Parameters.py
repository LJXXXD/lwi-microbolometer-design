
import numpy as np
import pandas as pd

from tools import ballsINTObins



class Sim_Parameters:
    def __init__(self, air_trans, atm_dist_ratio, air_RI, basis_funcs, basis_func_comb,
                 substance_ind_list, spectra, substances_emit, temp_K, percentage_step=0.1):

        # Enviroment related parameters
        self.temp_K = temp_K # Environmental temperature in K
        self.air_trans = air_trans[:, 1:]
        self.atm_dist_ratio = atm_dist_ratio # Atomsphere distance ratio
        self.air_RI = air_RI # Refractive index of air

        # Sensor related parameters
        self.basis_funcs = basis_funcs[:, 1:][:, basis_func_comb]

        # Substance related parameters
        self.num_substances = len(substance_ind_list)
        self.substance_ind_list = substance_ind_list
        self.spectra = spectra
        self.substances_emit = substances_emit[:, substance_ind_list]
        self.percentage_step = percentage_step

    def cal_mix_prop(self):
        # Material mixture proportion
        num_balls = int(1/self.percentage_step)
        self.mat_proportion = ballsINTObins(num_balls, self.num_substances).transpose() / num_balls




class Train_Parameters:
    def __init__(self, train_percentage, batch_size, criteria, loss_func_names, learning_rate, num_epochs, device, k_fold_flag, k, random_flag, random_seed, train_noise, test_noise):
        self.train_percentage = train_percentage
        self.batch_size = batch_size
        self.criteria = criteria
        self.loss_func_names = loss_func_names
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device
        self.k_fold_flag = k_fold_flag
        self.k = k
        self.random_flag = random_flag
        self.random_seed = random_seed
        self.train_noise = train_noise
        self.test_noise = test_noise
        # self.test_aggregation = test_aggregation