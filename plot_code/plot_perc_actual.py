import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data_path = './loss.xlsx'
data = pd.read_excel(data_path)

# Filter the data for the specified 'Basis Function Comb' and 'Train Noise'
filtered_data = data[(data['Basis Function Comb'] == "(0, 1, 2, 3)") & (data['Train Noise'] == 0)]

# Extract the test noise values and the substance columns for plotting
substance_columns = [col for col in filtered_data if col.startswith('Substance 3 Group')]

# Define the positions for the bars to sit between the intervals
intervals = list(range(0, 110, 10))  # From 0 to 100 with a step of 10
bar_positions = [(intervals[i] + intervals[i+1]) / 2 for i in range(len(intervals)-1)]

# Extract the RGBA values of the bar color from the provided plot
bar_color_rgba = (0.5294117647058824, 0.807843137254902, 0.9215686274509803, 1.0)
bar_color = bar_color_rgba[:-1]  # Exclude the alpha channel

# Find the maximum value across all subsets to set as the y-axis limit
max_value = filtered_data[substance_columns].max().max()
max_value = 0.6

# Generate and save the plots
plot_file_paths_intervals = []
for noise in [0, 0.2, 0.4, 0.8]:
    subset = filtered_data[filtered_data['Test Noise'] == noise]
    if len(subset) == 1:
        # Create a new figure for each plot
        plt.figure(figsize=(6, 4))
        plt.bar(bar_positions, subset.iloc[0][substance_columns], color=bar_color, width=5)  # width chosen for visual appeal
        plt.title(f'Actual Performance (Test Noise = {noise})')
        plt.xlabel('Heroin Percentage')
        plt.xticks(intervals)  # Set x-ticks to be at the intervals
        plt.ylabel('L1 Loss')
        plt.ylim(0, max_value * 1.1)  # Set y-axis limit to be 10% above the max value for a bit of headroom
        plt.grid(True, which='both', linestyle='--')  # Add grid to the plot

        
        # Save the figure
        plot_path_intervals = f'./substance_group_intervals_{noise}_plot.png'
        plt.savefig(plot_path_intervals)
        plot_file_paths_intervals.append(plot_path_intervals)
        
        # Close the figure to avoid display
        plt.close()

# Print the paths to the saved plots
print(plot_file_paths_intervals)
