import itertools
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

# Load the baseline values from the pickle file
baseline_file_name = 'output/estimated_baseline.pkl'

try:
    with open(baseline_file_name, 'rb') as f:
        baseline = pickle.load(f)
except Exception as e:
    showerror(type(e).__name__, str(e))

baseline_loss = []
for i in range(3, 11):
    base_line_loss = list(baseline[i].values())
    baseline_loss.append(base_line_loss[0])

# Read the data from the pickle file
df = pd.read_pickle('loss.pkl')

# Get the rows with the minimum L1 loss for each group
min_l1loss_rows = df.groupby(['Temperature_K', 'Substance Number', 'Substance Comb', 'Basis Function Number', 'Comb of Basis Functions', 'Train Noise Max Percentage', 'Test Noise Max Percentage']).apply(lambda x: x[x.L1Loss == x.L1Loss.min()]).reset_index(drop=True)

# Pivot the table to have the L1 loss values as columns
table = pd.pivot_table(min_l1loss_rows, values='L1Loss', index='Basis Function Number', columns='Substance Number')

# Calculate the metric score
# for i, col in enumerate(table):
    # table[col] = (1 - table[col] / baseline_loss[i]) * 100

# Rename the table
table.columns.name = 'Number of Substances'
table.index.name = 'Basis Functions'
metric_table = table.rename(columns={'L1Loss': 'Metric Score (1-L1Loss/BaselineLoss)'})

# Set seaborn style
# sns.set(style='white')

# Create the figure and axes for the first plot
fig1, ax1 = plt.subplots()

# Plot the metric table as a line plot without markers (dots)
metric_table.plot.line(marker='', ax=ax1)

# Set the x-label and y-label
ax1.set_xlabel('Number of Basis Functions')
ax1.set_ylabel('L1 Loss (Not AVG)')

# Set the title
ax1.set_title('Metric Score vs Number of Basis Functions for Different Number of Substances')

# Add a legend with the title "Number of Substances"
ax1.legend(title='Number of Substances', loc='upper right')

# Set the tick labels to be integers only
ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Create the figure and axes for the second plot
fig2, ax2 = plt.subplots()

# Plot the transposed metric table as a line plot without markers (dots)
transposed_table = metric_table.transpose()
transposed_table.plot.line(marker='', ax=ax2)

# Set the x-label and y-label
ax2.set_xlabel('Number of Substances')
ax2.set_ylabel('L1 Loss (Not AVG)')

# Set the title
ax2.set_title('Metric Score vs Number of Substances for Different Number of Basis Functions')

# Add a legend with the title "Number of Basis Functions"
ax2.legend(title='Number of Basis Functions', loc='lower right')

# Set the tick labels to be integers only
ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

ax1.grid(axis='y')
ax2.grid(axis='y')

# Show the plots
plt.show()
