import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')



# Read the data from the CSV file
file_name = 'model_perform_diff_temp_loss.pkl'
df = pd.read_pickle(file_name)

# file_name = 'loss_PCA.csv'
# df_PCA = pd.read_csv(file_name, converters={'Substance Comb': ast.literal_eval, 'Comb of Basis Functions': ast.literal_eval})

# file_name_la = 'loss_la.csv'
# df_la = pd.read_csv(file_name_la, converters={'Substance Comb': ast.literal_eval, 'Basis Function Comb': ast.literal_eval})

model_temp_K_list = [243.15, 283.15, 323.15, 'All Temp']
temp_K_list = np.round(np.arange(-30, 51, 10, dtype=int) + 273.15, 2)

substance_ind_list =  [0, 1, 2, 3]
substance_ind_list.sort()

basis_func_ind_list = [0, 3, 4, 6]
# basis_func_ind_list = [0, 1, 2, 5]
basis_func_ind_list.sort()

confidence_perc_list = [0.68, 0.95, 0.997]
confidence_perc = confidence_perc_list[1]

# Define the group criteria
group_criteria = ['Model Temp K', 'Substance Comb', 'Basis Function Comb', 'Train Noise', 'Test Noise']


train_noise_list = [0]
# train_noise_list = [0, 0.5, 1]
train_noise = 0

test_noise_list = [0]
# test_noise_list = [i * 0.05 for i in range(21)]
test_noise = 0



plt.figure()
for model_temp_K in model_temp_K_list:
# for train_noise in train_noise_list:
    # Specify the target group
    target_group = (model_temp_K, tuple(substance_ind_list), tuple(basis_func_ind_list), train_noise, test_noise)

    # Filter the DataFrame based on the target group
    filtered_df = df.groupby(group_criteria).get_group(target_group)
    filtered_df = filtered_df.sort_values(by='Temperature_K')
    
    print(filtered_df.iloc[:, 0:8])

    plt.plot(temp_K_list, filtered_df[f'AVG skewnorm 95% interval range'], label=f'Model Trained Temp (F)={model_temp_K}')

    

    # if train_noise == 0:
    #     filtered_df_la = df_la.groupby(group_criteria).get_group(target_group)
    #     plt.plot(index, filtered_df_la[f'AVG skewnorm 95% interval range'], label=f'LA')

    plt.xlabel('Temperature')
    plt.ylabel(f'Metric (Lower is better)')
    plt.title(f'Model Performance under different Temperature')

    # plt.ylim(0, 0.8)

    # Set logarithmic scale on x-axis with base 5
    # plt.xticks(index, filtered_train_df['Test Noise Max Percentage'])

# Display a legend
plt.legend()
plt.show()

    # Save the plot to a file
# plt.savefig('noise output/' + 'ML.png')

