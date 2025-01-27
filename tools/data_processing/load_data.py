import yaml

import numpy as np
import pandas as pd


def get_list_values(list_config):

    def generate_range_list(start, stop, step):
        tolerance = 1e-10 # Tolerance to account for floating-point imprecision
        return [start + step * i for i in range(int((stop - start + tolerance) / step) + 1)]

    if 'values' in list_config:
        return list_config['values']
    else:
        return generate_range_list(list_config['start'], list_config['stop'], list_config['step'])



def load_excel_data(*file_paths):
    loaded_data = []
    for path in file_paths:
        try:
            data = np.array(pd.read_excel(path, header=None))
            loaded_data.append(data)
        except FileNotFoundError:
            print(f"Error: Excel file '{path}' not found.")
            continue  # Optionally, you could also choose to exit the program here
        except ValueError as e:
            print(f"Error loading Excel file '{path}': {e}")
            continue
    if len(loaded_data) == 1:
        return loaded_data[0]
    else:
        return loaded_data


def load_config(file_path):
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{file_path}' not found.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        exit(1)


# Obsolete
def load_data(air_trans_file, basis_func_file, spectra_file, substances_emit_file):
    air_trans = np.array(pd.read_excel(air_trans_file, header=None))
    basis_funcs = np.array(pd.read_excel(basis_func_file, header=None))
    spectra = np.array(pd.read_excel(spectra_file, header=None))
    substances_emit = np.array(pd.read_excel(substances_emit_file, header=None))
    # substance_names = np.array(pd.read_excel(substance_names_file, header=None))

    return air_trans, basis_funcs, spectra, substances_emit


