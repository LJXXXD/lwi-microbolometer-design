
import numpy as np
import pandas as pd


def load_data(air_trans_file, basis_func_file, spectra_file, substances_emit_file):
    air_trans = np.array(pd.read_excel(air_trans_file, header=None))
    basis_funcs = np.array(pd.read_excel(basis_func_file, header=None))
    spectra = np.array(pd.read_excel(spectra_file, header=None))
    substances_emit = np.array(pd.read_excel(substances_emit_file, header=None))

    return air_trans, basis_funcs, spectra, substances_emit