import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')



# Read the data from the CSV file
file_name = 'loss.csv'
df = pd.read_csv(file_name, converters={'Substance Comb': ast.literal_eval, 'Basis Function Comb': ast.literal_eval})

# file_name = 'loss_PCA.csv'
# df_PCA = pd.read_csv(file_name, converters={'Substance Comb': ast.literal_eval, 'Comb of Basis Functions': ast.literal_eval})

# file_name_la = 'loss_la.csv'
# df_la = pd.read_csv(file_name_la, converters={'Substance Comb': ast.literal_eval, 'Basis Function Comb': ast.literal_eval})


temp_K_list = np.round(np.arange(-30, 51, 10, dtype=int) + 273.15, 2)
# temp_K_list = (293.15,)
substance_ind_list =  [0, 1, 2, 3]
substance_ind_list.sort()

basis_func_ind_list = [0, 3, 4, 6]
# basis_func_ind_list = [0, 1, 2, 5]
basis_func_ind_list.sort()

confidence_perc_list = [0.68, 0.95, 0.997]
confidence_perc = confidence_perc_list[1]

# Define the group criteria
group_criteria = ['Temperature_K', 'Substance Number', 'Substance Comb', 'Basis Function Number', 'Basis Function Comb',
                  'Train Noise']


train_noise_list = [0]
# train_noise_list = [0, 0.5, 1]
train_noise = 0
# test_noise_list = [0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 1, 2]
test_noise_list = [i * 0.1 for i in range(11)]



plt.figure()
for temp_K in temp_K_list:
# for train_noise in train_noise_list:
    # Specify the target group
    target_group = (temp_K, len(substance_ind_list), tuple(substance_ind_list), len(basis_func_ind_list), tuple(basis_func_ind_list), train_noise)

    # Filter the DataFrame based on the target group
    filtered_df = df.groupby(group_criteria).get_group(target_group)
    
    # print(filtered_df.iloc[:, 0:8])

    # index = range(len(filtered_train_df['Test Noise Max Percentage']))
    index = test_noise_list
    print(index)
    plt.plot(index, filtered_df[f'L1Loss'], label=f'NN with Temp={temp_K}')

    

    # if train_noise == 0:
    #     filtered_df_la = df_la.groupby(group_criteria).get_group(target_group)
    #     plt.plot(index, filtered_df_la[f'AVG skewnorm 95% interval range'], label=f'LA')

    plt.xlabel('Noise (Volt)')
    plt.ylabel(f'Metric')
    plt.title(f'NN V.S. Noise')

    # plt.ylim(0, 0.8)

    # Set logarithmic scale on x-axis with base 5
    # plt.xticks(index, filtered_train_df['Test Noise Max Percentage'])

# Display a legend
plt.legend()
plt.show()

    # Save the plot to a file
# plt.savefig('noise output/' + 'ML.png')

