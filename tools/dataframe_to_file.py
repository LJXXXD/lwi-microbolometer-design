import fcntl
import os
import time

import pandas as pd

def df_to_file(df, output_folder, file_name, file_extension, drop_duplicates_columns, write_function, read_function):

    output_file = os.path.join(output_folder, file_name + '.' + file_extension)

    # Check if the file already exists
    if os.path.isfile(output_file):

        # If the file exists, load the existing DataFrame from the file
        df_existing = read_function(output_file)

        # Open the file in binary mode
        with open(output_file, 'w') as f:
            while True:
                try:
                    # Get a lock on the file
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except IOError:
                    # Lock is held by another process, wait and try again
                    time.sleep(1)

            # Merge df to df_existing, and if there's duplicates, keep the new values from df
            df = pd.concat([df, df_existing], ignore_index=True).drop_duplicates(subset=drop_duplicates_columns, keep='first')
            df = df.sort_values(by=drop_duplicates_columns)

            write_function(df, output_file)
            
            fcntl.flock(f, fcntl.LOCK_UN)
    
    else:
        df = df.sort_values(by=drop_duplicates_columns)
        write_function(df, output_file)


def df_to_csv(df, output_folder, file_name, drop_duplicates_columns):
    def write_csv(dataframe, file_path):
        dataframe.to_csv(file_path, index=False)

    file_extension = 'csv'

    df_to_file(df, output_folder, file_name, file_extension, drop_duplicates_columns, write_csv, pd.read_csv)


def df_to_excel(df, output_folder, file_name, drop_duplicates_columns):
    def write_excel(dataframe, file_path):
        dataframe.to_excel(file_path, index=False)

    file_extension = 'xlsx'

    df_to_file(df, output_folder, file_name, file_extension, drop_duplicates_columns, write_excel, pd.read_excel)


def df_to_pickle(df, output_folder, file_name, drop_duplicates_columns):
    def write_pickle(dataframe, file_path):
        dataframe.to_pickle(file_path)

    file_extension = 'pkl'

    df_to_file(df, output_folder, file_name, file_extension, drop_duplicates_columns, write_pickle, pd.read_pickle)
