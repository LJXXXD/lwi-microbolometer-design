import ast

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')



# Read the data from the CSV file
file_name = 'loss.csv'
df = pd.read_csv(file_name, converters={'Substance Comb': ast.literal_eval, 'Comb of Basis Functions': ast.literal_eval})

file_name = 'loss_PCA.csv'
df_PCA = pd.read_csv(file_name, converters={'Substance Comb': ast.literal_eval, 'Comb of Basis Functions': ast.literal_eval})

file_name = 'loss_test_mtx_m.csv'
df_mtx_mul = pd.read_csv(file_name, converters={'Substance Comb': ast.literal_eval, 'Comb of Basis Functions': ast.literal_eval})

substance_list = [0, 1, 2, 3]
# substance_list = [1, 3, 5, 7, 9, 10, 14, 18]

confidence_perc_list = [0.68, 0.95, 0.997]
confidence_perc = confidence_perc_list[1]

# Define the group criteria
group_criteria_4train = ['Temperature_K', 'Substance Number', 'Substance Comb', 'Basis Function Number',
                       'Comb of Basis Functions', 'Train Noise Max Percentage', 'Test Aggregation']

# group_criteria_4test = ['Temperature_K', 'Substance Number', 'Substance Comb', 'Basis Function Number',
#                        'Comb of Basis Functions', 'Test Noise Max Percentage']

train_noise_list = [0]
test_noise_list = [0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 1]


# df_mtx_mul = pd.read_csv('loss_test_mtx_m.csv', converters={'Substance Comb': ast.literal_eval, 'Comb of Basis Functions': ast.literal_eval})
group_criteria_mtx_mul = ['Temperature_K', 'Substance Number', 'Substance Comb', 'Basis Function Number',
                        'Comb of Basis Functions']

substance_names = np.array(pd.read_excel('./data/Test 3 - 4 White Powers/white_powders_names.xlsx', header=None))


# Create a new plot


for substance in substance_list:
    plt.figure()
    # plt.subplots_adjust(top=0.85, bottom=0.15)

    for train_noise in train_noise_list:
        # Specify the target group
        target_train_group = (293.15, len(substance_list), tuple(substance_list), 4, (0, 1, 2, 3), train_noise, 1)
        # Filter the DataFrame based on the target group
        filtered_train_df = df.groupby(group_criteria_4train).get_group(target_train_group)
        filtered_train_df_pca = df_PCA.groupby(group_criteria_4train).get_group(target_train_group)

        index = range(len(filtered_train_df['Test Noise Max Percentage']))

        plt.plot(index, filtered_train_df[f'Substance {substance} skewnorm {confidence_perc*100}% interval range'], label=f'non PCA Train Noise={train_noise}')
        plt.plot(index, filtered_train_df_pca[f'Substance {substance} skewnorm {confidence_perc*100}% interval range'], label=f'PCA Train Noise={train_noise}')

        # Set plot labels and title
        plt.xlabel('Noise')
        plt.ylabel(f'skewnorm {confidence_perc*100}% interval range')
        plt.title(f'{substance_names[substance, 1]} {confidence_perc*100}% confidence interval range')

        # Set logarithmic scale on x-axis with base 5
        plt.xticks(index, filtered_train_df['Test Noise Max Percentage'])

        # Set the maximum scale of the y-axis
        lower_limit = 0
        upper_limit = 2
        plt.ylim(lower_limit, upper_limit)

        
        ######################################
        # TEMP ADD ON! PLZ FIX AND CLEAN UP!!!

        target_group_mtx_mul = (293.15, len(substance_list), tuple(substance_list), 4, (0, 1, 2, 3))

        filtered_df_mtx_mul = df.groupby(group_criteria_mtx_mul).get_group(target_group_mtx_mul)

        print(filtered_df_mtx_mul[f'Substance {substance} skewnorm {confidence_perc*100}% interval range'])

        plt.plot(index, df_mtx_mul[f'Substance {substance} skewnorm {confidence_perc*100}% interval range'], label=f'Linear Algebra (No normalization)')


        # TEMP ADD ON! PLZ FIX AND CLEAN UP!!!
        ######################################


        # Display a legend
        # plt.legend()
    # print(df_mtx_mul[f'Substance {substance} skewnorm {confidence_perc*100}% interval range'])
    # plt.plot(index, df_mtx_mul[f'Substance {substance} skewnorm {confidence_perc*100}% interval range'], label=f'Linear Algebra')
    plt.legend()
    # plt.show()
    # Save the plot to a file
    plt.savefig('noise output/' + file_name.split('.')[0] + f' Substance {substance} skewnorm\n{confidence_perc*100}% interval range (Performance) vs Noise Level.png', dpi=300)
    plt.close()


plt.figure()
for train_noise in train_noise_list:
    # Specify the target group
    target_train_group = (293.15, len(substance_list), tuple(substance_list), 4, (0, 1, 2, 3), train_noise, 1)

    # Filter the DataFrame based on the target group
    filtered_train_df = df.groupby(group_criteria_4train).get_group(target_train_group)

    

    index = range(len(filtered_train_df['Test Noise Max Percentage']))
    plt.plot(index, filtered_train_df[f'MSELoss'], label=f'ML Model')

    plt.xlabel('Noise')
    plt.ylabel(f'MSE Loss')
    plt.title(f'MSE Loss Performance Comparison')

    # Set logarithmic scale on x-axis with base 5
    plt.xticks(index, filtered_train_df['Test Noise Max Percentage'])

    # Display a legend
    plt.legend()

######################################
# TEMP ADD ON! PLZ FIX AND CLEAN UP!!!

target_group_mtx_mul = (293.15, len(substance_list), tuple(substance_list), 4, (0, 1, 2, 3))

filtered_df_mtx_mul = df.groupby(group_criteria_mtx_mul).get_group(target_group_mtx_mul)

print(filtered_df_mtx_mul[f'Substance {substance} skewnorm {confidence_perc*100}% interval range'])

plt.plot(index, df_mtx_mul[f'Substance {substance} skewnorm {confidence_perc*100}% interval range'], label=f'Linear Algebra (No normalization)')


# TEMP ADD ON! PLZ FIX AND CLEAN UP!!!
######################################

plt.legend()
    # Save the plot to a file
plt.savefig('noise output/' + 'ML.png')

# for train_noise in train_noise_list:
#     # Specify the target group
#     target_train_group = (293.15, len(substance_list), tuple(substance_list), 7, (0, 1, 2, 3, 4, 5, 6), train_noise)

#     # Filter the DataFrame based on the target group
#     filtered_train_df = df.groupby(group_criteria_4train).get_group(target_train_group)

#     # Create a separate plot for each substance
#     for substance in substance_list:

#         for confidence_perc in confidence_perc_list:

#             # Create a new plot
#             plt.figure()

#             # Create a plot for each 'Test Noise Max Percentage'
#             for test_noise_max_percentage in filtered_train_df['Test Noise Max Percentage'].unique():
#                 # Filter the data for the current 'Test Noise Max Percentage'
#                 data = filtered_train_df[filtered_train_df['Test Noise Max Percentage'] == test_noise_max_percentage]

#                 # Create the plot
#                 plt.plot(data['Test Aggregation'], data[f'Substance {substance} skewnorm {confidence_perc*100}% interval range'], label=f'Test Noise Max Percentage: {test_noise_max_percentage}')
                
#                 # Set plot labels and title
#                 plt.xlabel('Time/Test Aggregation')
#                 plt.ylabel(f'skewnorm {confidence_perc*100}% interval range/Performance')
#                 plt.title(f'Train Noise = {train_noise}\nSubstance {substance} skewnorm\n{confidence_perc*100}% interval range (Performance) vs Time(Test Aggregation)')

#                 # Set logarithmic scale on x-axis with base 5
#                 plt.xscale('log', base=5)

#                 # Display a legend
#                 plt.legend()

#             # Save the plot to a file
#             plt.savefig('noise output/' + f'Train Noise = {train_noise}-Substance {substance} skewnorm-{confidence_perc*100}% interval range (Performance) vs Time(Test Aggregation).png')
#             plt.close()


#     group_criteria = ['Temperature_K', 'Substance Number', 'Substance Comb', 'Basis Function Number',
#                        'Comb of Basis Functions', 'Test Aggregation']
#     target_group = (293.15, len(substance_list), tuple(substance_list), 7, (0, 1, 2, 3, 4, 5, 6), 1)
#     filtered_df = df.groupby(group_criteria).get_group(target_group)
#     data = filtered_df[filtered_df['Test Aggregation'] == 1]
#     plt.figure()

    # for substance in substance_list:

    #     plt.plot(data['Test Noise Max Percentage'], data[f'Substance {substance} skewnorm {confidence_perc*100}% interval range'], label=f'Substance{substance}')
    # plt.show()

# for test_noise in test_noise_list:
#     # Specify the target group
#     target_test_group = (293.15, len(substance_list), tuple(substance_list), 7, (0, 1, 2, 3, 4, 5, 6), test_noise)

#     # Filter the DataFrame based on the target group
#     filtered_test_df = df.groupby(group_criteria_4test).get_group(target_test_group)

#     # Create a separate plot for each substance
#     for substance in substance_list:

#         for confidence_perc in confidence_perc_list:

#             # Create a new plot
#             plt.figure()

#             # Create a plot for each 'Test Noise Max Percentage'
#             for train_noise_max_percentage in filtered_test_df['Train Noise Max Percentage'].unique():
#                 # Filter the data for the current 'Test Noise Max Percentage'
#                 data = filtered_test_df[filtered_test_df['Train Noise Max Percentage'] == train_noise_max_percentage]

#                 # Create the plot
#                 plt.plot(data['Test Aggregation'], data[f'Substance {substance} skewnorm {confidence_perc*100}% interval range'], label=f'Train Noise Max Percentage: {train_noise_max_percentage}')
                
#                 # Set plot labels and title
#                 plt.xlabel('Time/Test Aggregation')
#                 plt.ylabel(f'skewnorm {confidence_perc*100}% interval range/Performance')
#                 plt.title(f'Test Noise = {test_noise}\nSubstance {substance} skewnorm\n{confidence_perc*100}% interval range (Performance) vs Time(Test Aggregation)')

#                 # Set logarithmic scale on x-axis with base 5
#                 plt.xscale('log', base=5)

#                 # Display a legend
#                 plt.legend()

#             # Save the plot to a file
#             plt.savefig('noise output/' + f'Test Noise = {test_noise}-Substance {substance} skewnorm- {confidence_perc*100}% interval range (Performance) vs Time(Test Aggregation).png')
#             plt.close()




#     # # Create a plot for l1 loss
#     # plt.figure()

#     # # Create a plot for each 'Test Noise Max Percentage'
#     # for test_noise_max_percentage in filtered_df['Test Noise Max Percentage'].unique():
#     #     # Filter the data for the current 'Test Noise Max Percentage'
#     #     data = filtered_df[filtered_df['Test Noise Max Percentage'] == test_noise_max_percentage]

#     #     # Create the plot
#     #     plt.plot(data['Test Aggregation'], data[f'L1Loss'], label=f'Test Noise Max Percentage: {test_noise_max_percentage}')
        
#     #     # Set plot labels and title
#     #     plt.xlabel('Test Aggregation')
#     #     plt.ylabel(f'skewnorm {confidence_perc*100}% interval range')
#     #     plt.title(f'L1Loss')

#     #     # Set logarithmic scale on x-axis with base 5
#     #     plt.xscale('log', base=5)

#     #     # Display a legend
#     #     plt.legend()

#     # # Save the plot to a file
#     # plt.savefig(f'{confidence_perc*100}%_L1Loss_plot.png')

#     # # Show the plot
#     # plt.show()
