import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = './loss (original).xlsx'
data = pd.read_excel(file_path)

# Filter the data for the specific Basis Function Combination
basis_function_comb = (0, 1, 3, 6)
filtered_data = data[data['Basis Function Comb'].astype(str) == str(basis_function_comb)]

# Updated 'x' and 'y' values for the "additional data"
updated_x_values = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2]
updated_y_values = [2.91E-07, 0.035962564, 0.052052954, 0.121607232, 0.12801152, 0.173648211, 0.230552487, 0.213644756, 0.278273178, 0.31083042, 0.410427253, 0.512123464, 0.608413169, 0.878882877, 0.530554157, 3.103836454, 2.932986338, 2.683899411, 2.243380164, 1.623565335, 1.015923187, 2.130395856, 6.164501156, 1.372412073, 1.768877988, 8.339773358, 1.880572804, 7.221781029, 15.28240662, 3.405864579, 4.85466456, 6.55216298, 2.257344483, 3.032720565, 16.55340372, 3.318065381, 2.284189385, 1.749626793, 4.572803599, 5.733950281, 11.31816987]

# Filtering the additional data to only include x-values up to 0.8
filtered_additional_data = {'x': [], 'y': []}
for x, y in zip(updated_x_values, updated_y_values):
    # if x <= :
    filtered_additional_data['x'].append(x)
    filtered_additional_data['y'].append(y)







# Updated 'x' and 'y' values for the "additional data"
updated_x_values2 = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2]
updated_y_values2 = [0.088691144, 0.105641957, 0.164653718, 0.199157117, 0.229322152, 0.262199698, 0.284840293, 0.380257524, 0.379928152, 0.416239978, 0.495952731, 0.438970835, 0.48842891, 0.646635919, 0.625056428, 0.728760298, 0.694273336, 0.778641971, 0.806355692, 0.932803464, 1.046286393, 1.222710326, 1.080329662, 1.24754951, 1.343230295, 1.408438464, 1.603516894, 1.291380688, 1.254246037, 1.708402439, 1.444735155, 1.669353697, 1.531317854, 2.560897096, 2.168019821, 2.518866929, 8.03980356, 2.047206821, 2.880972839, 2.396591002, 2.532696809]

# Filtering the additional data to only include x-values up to 0.8
filtered_additional_data2 = {'x': [], 'y': []}
for x, y in zip(updated_x_values2, updated_y_values2):
    # if x <= :
    filtered_additional_data2['x'].append(x)
    filtered_additional_data2['y'].append(y)









# Plotting
plt.figure(figsize=(8, 6))
plt.title("Updated NN V.S. Analytical")
plt.xlabel("Test Noise")
plt.ylabel("L1Loss")
plt.grid(which='both', linestyle='--', linewidth=0.5)

# Color scheme for better distinction
colors = ['g', 'r', 'b']  # Blue, Green, Red

# Plotting for each train noise value
for i, train_noise in enumerate([0]):
    subset = filtered_data[filtered_data['Train Noise'] == train_noise]
    plt.plot(subset['Test Noise'], subset['L1Loss'], label=f'Train Noise = {train_noise}', color=colors[i])

# Plot the filtered additional data with updated legend name
plt.plot(filtered_additional_data["x"], filtered_additional_data["y"], label="Analytical", color='r')  # Magenta for analytical data




# Plot the filtered additional data with updated legend name
plt.plot(filtered_additional_data2["x"], filtered_additional_data2["y"], label="Updated Analytical (100 sampless)", color='b')  # Magenta for analytical data




# Setting the y-axis limit
# plt.xlim(0, 2)
plt.ylim(0, 2)

# Adding legend
plt.legend()

# Show the plot
plt.show()
