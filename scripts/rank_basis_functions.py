import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.use('TkAgg')

# Read the data from the pickle file
df = pd.read_pickle('loss.pkl')

# Group by the first 4 columns and assign rank within each group based on L1Loss in reverse order
df['Value in group'] = df.groupby(['Temperature_K', 'Substance Number', 'Substance Comb', 'Basis Function Number'])['L1Loss'].rank(method='min', ascending=False) - 1
print(df[:20])

# df.to_csv('1 Value in group.csv', index=False)

# Add new columns with default values of 0
for i in range(7):
    col_name = f'b_func {i} value'
    df[col_name] = 0

# Update the values in the new columns based on the 'Comb of Basis Functions' column
def update_b_func_value(row):
    comb = row['Comb of Basis Functions']
    value_in_group = row['Value in group']
    for i in range(7):
        if i in comb:
            col_name = f'b_func {i} value'
            row[col_name] = value_in_group
    return row

df = df.apply(update_b_func_value, axis=1)

print(df[:20])

# df.to_csv('2 Value in group to b func.csv', index=False)

df = df[df['Basis Function Number'] != 7]

# Group by the first 4 columns and sum the 'b_func' columns within each group
b_func_value_sums = df.groupby(['Temperature_K', 'Substance Number', 'Substance Comb', 'Basis Function Number']).agg({
    'b_func 0 value': 'sum',
    'b_func 1 value': 'sum',
    'b_func 2 value': 'sum',
    'b_func 3 value': 'sum',
    'b_func 4 value': 'sum',
    'b_func 5 value': 'sum',
    'b_func 6 value': 'sum'
})

# Print the sums
print(b_func_value_sums)

# b_func_value_sums.to_csv('3 b_func value sums.csv', index=True)

# Create a copy of b_func_sums
rank_df = b_func_value_sums.copy()

# Rename the columns with the suffix "Rank"
rank_df = rank_df.rename(columns=lambda x: x.replace('value', 'Rank'))

# Convert the values to ranks within each row
rank_df = rank_df.rank(axis=1, method='min', ascending=False).astype(int)

# Print the resulting DataFrame
print(rank_df)
# rank_df.to_csv('4 Value to rank.csv', index=True)


# Count the occurrences of each rank (1, 2, 3, etc.) for each column
counts = rank_df.apply(pd.Series.value_counts).fillna(0).astype(int)

# Print the resulting counts
print(counts)
# counts.to_csv('5 Count of rank.csv', index=True)

# Set column names
column_names = [f'Basis F {i}' for i in range(len(counts.columns))]
counts.columns = column_names

# Plot the heatmap with red-to-blue colormap
plt.figure(figsize=(10, 6))
sns.heatmap(counts, annot=True, cmap='RdBu_r', fmt='d')
plt.title('Counts of Ranks for Each Basis Function')
plt.xlabel('Basis Functions')
plt.ylabel('Ranks')
plt.xticks(rotation=0)  # Rotate x-axis labels to horizontal
plt.tight_layout()  # Adjust layout for better visibility
plt.show()


# Calculate the weighted mean (Average Rank) for each column
average_rank = counts.apply(lambda col: np.average(col.index, weights=col.values))

# average_rank.to_csv('6 Average rank.csv', index=False)

# Print the Average Rank for each column
print("Average Rank:")
for i, col in enumerate(counts.columns):
    print(f"{col}: {average_rank[i]}")
