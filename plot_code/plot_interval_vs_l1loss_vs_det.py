import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("TkAgg") # Changing the backend to QT solves ctrl-C not quiting issue in terminal

# Read the data from the CSV file
file_name = 'loss.csv'
df = pd.read_csv(file_name, converters={'Substance Comb': ast.literal_eval, 'Basis Function Comb': ast.literal_eval})

file_name = 'loss_la.csv'
df_la = pd.read_csv(file_name, converters={'Substance Comb': ast.literal_eval, 'Basis Function Comb': ast.literal_eval})

substance_list = [0, 1, 2, 3]

confidence_perc_list = [0.68, 0.95, 0.997]
confidence_perc = confidence_perc_list[1]

# Define the group criteria
group_criteria = ['Temperature_K', 'Substance Number', 'Substance Comb', 'Basis Function Number',
                  'Train Noise', 'Test Noise']

# Specify the target group
target_group = (293.15, len(substance_list), tuple(substance_list), 4, 0, 0)

filtered_df = df.groupby(group_criteria).get_group(target_group)
filtered_df_la = df_la.groupby(group_criteria).get_group(target_group)

# Sort the DataFrame by the "AVG skewnorm 95% interval range" column
filtered_df = filtered_df.sort_values(by='AVG skewnorm 95% interval range')
filtered_df_la = filtered_df_la.sort_values(by='AVG skewnorm 95% interval range')

# Create a plot
# plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

# Plot the data, where x is the row number and y is "AVG skewnorm 95% interval range"

x_values = range(len(filtered_df))

avg_interval = filtered_df['AVG skewnorm 95% interval range']
max_interval = filtered_df['MAX skewnorm 95% interval range']
min_interval = filtered_df['MIN skewnorm 95% interval range']

plt.plot(x_values, avg_interval, color='orange', label='AVG Interval')
plt.plot(x_values, max_interval, color='blue', alpha=0.3, label='MAX Interval')
plt.plot(x_values, min_interval, color='green', alpha=0.3, label='MIN Interval')
plt.fill_between(x_values, max_interval, avg_interval, color='blue', alpha=0.1)
plt.fill_between(x_values, min_interval, avg_interval, color='green', alpha=0.1)

# plt.plot(x_values, filtered_df['L1Loss'], label='L1')

plt.plot(x_values, filtered_df_la['AVG skewnorm 95% interval range'], color='red', label='AVG Interval(Linear Algebra)')

# plt.plot(range(1, len(df) + 1), df['A determinant'].abs()/10, label='A matrix det')
# plt.plot(range(1, len(df) + 1), df['A conditional number'].abs()/12000, label='A matrix Conditional #')
# plt.plot(range(1, len(df) + 1), df['A avg cor'], label='A matrix avg correlation')
# plt.plot(range(1, len(df) + 1), df['A cov trace']/700, label='A matrix cov trace')
# plt.plot(range(1, len(df) + 1), df['A sum cov exc trace']/2000, label='sum cov exc trace')
# plt.plot(range(1, len(df) + 1), df['A mean disimilarity']/20, label='A matrix mean disimilarity')

# plt.ylim(0, 1)

# Label the axes and title
plt.xlabel('Sensor Design No. (Ranked by AVG interval range as metric)')
plt.ylabel('Metric Score')
plt.title('Comparison of Performance (NN V.S. LA)')
plt.legend()

# Show the plot
plt.show()
