import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load the DataFrame from the Excel file
df = pd.read_excel("./output/test_with_trained_models.xlsx")

# Group the DataFrame by "Model Temp K"
grouped = df.groupby("Temperature_K")

# Iterate over each group and plot "Test Temp K" vs "L1Loss"
for model_temp, group_df in grouped:
    # plt.plot(group_df["Test_Temperature_K"], group_df["L1Loss"], label=f"Model Train Temperature (K) = {model_temp}")
    plt.plot(group_df["Test_Temperature_K"], group_df["L1Loss"], alpha=0.3)

# Create a list to store unique Test Temp K values
test_temp_values = df["Test_Temperature_K"].unique()

# Set x-axis ticks to match the Test Temp K values from the DataFrame
plt.xticks(test_temp_values)




df_multi_temp = pd.read_excel("./output/test_with_trained_models_multi_temp.xlsx")
df_multi_temp_5 = df_multi_temp[df_multi_temp["Temperature_K_step"]==5]
df_multi_temp_10 = df_multi_temp[df_multi_temp["Temperature_K_step"]==10]
df_multi_temp_20 = df_multi_temp[df_multi_temp["Temperature_K_step"]==20]
df_multi_temp_50 = df_multi_temp[df_multi_temp["Temperature_K_step"]==50]
plt.plot(df_multi_temp_5["Test_Temperature_K"], df_multi_temp_5["L1Loss"], label=f"Model Train Temp [253.15, 258.15, ..., 303.15] (Step of 5K)", linestyle='--', color='red')
plt.plot(df_multi_temp_10["Test_Temperature_K"], df_multi_temp_10["L1Loss"], label=f"Model Train Temp [253.15, 263.15, ..., 303.15] (Step of 10K)", linestyle='--', color='blue')
plt.plot(df_multi_temp_20["Test_Temperature_K"], df_multi_temp_20["L1Loss"], label=f"Model Train Temp [253.15, 273.15, 293.15] (Step of 20K)", linestyle='--', color='green')
plt.plot(df_multi_temp_50["Test_Temperature_K"], df_multi_temp_50["L1Loss"], label=f"Model Train Temp [253.15, 303.15] (Step of 50K)", linestyle='--', color='purple')



# Add labels and legend
plt.xlabel("Test Temperature (K)")
plt.ylabel("L1Loss (Lower means better)")
plt.legend()
plt.title("Temperature-Dependent Model Evaluation Across Variable Testing Temperatures")
plt.show()
