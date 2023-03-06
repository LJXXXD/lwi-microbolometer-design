
import numpy as np
import pandas as pd

from tools import ballsINTObins

class Sim_Parameters:
    def __init__(self, air_trans_file, atm_dist_ratio, air_RI, basis_func_file,
                 num_substances, spectra_file, substances_emit_file, temp_K):
        # self.air_trans_file = air_trans_file
        # self.air_RI = air_RI
        # self.atm_dist_ratio = atm_dist_ratio
        self.basis_func_file = basis_func_file
        self.num_substances = num_substances
        self.spectra_file = spectra_file
        self.substances_emit_file = substances_emit_file
        self.temp_K = temp_K

        # Enviroment related parameters
        self.temp_K = temp_K # Environmental temperature in K
        self.air_trans = np.array(pd.read_excel(air_trans_file, header=None))
        self.air_trans = self.air_trans[:, 1:]
        self.atm_dist_ratio = atm_dist_ratio # Atomsphere distance ratio
        self.air_RI = air_RI # Refractive index of air

        # Sensor related parameters
        self.basis_funcs = np.array(pd.read_excel(basis_func_file, header=None))
        self.basis_funcs = self.basis_funcs[:, 1:]

        # Substance related parameters
        self.num_substances = num_substances
        self.spectra = np.array(pd.read_excel(spectra_file, header=None))
        self.substances_emit = np.array(pd.read_excel(substances_emit_file, header=None))
        self.substances_emit = self.substances_emit[:, 0:self.num_substances]
        # Material mixture proportion
        self.mat_proportion = ballsINTObins(10, self.num_substances).transpose() / 10




class Train_Parameters:
    def __init__(self, train_percentage, batch_size, criterions, learning_rate, num_epochs, device, k_fold_flag, k, random_flag, random_seed):
        self.train_percentage=train_percentage
        self.batch_size=batch_size
        self.criterions=criterions
        self.learning_rate=learning_rate
        self.num_epochs=num_epochs
        self.device=device
        self.k_fold_flag=k_fold_flag
        self.k=k
        self.random_flag=random_flag
        self.random_seed=random_seed
