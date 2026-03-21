import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load the DataFrame from the Excel file
df = pd.read_excel("./outputs/legacy_output/test_with_trained_models_test_noise.xlsx")

plt.figure(figsize=(6, 5))
# Group the DataFrame by "Model Temp K"
grouped = df.groupby("Train Noise")

# Iterate over each group and plot "Test Temp K" vs "L1Loss"
for index, (train_noise, group_df) in enumerate(grouped):
    if index % 2 != 0:
        continue
    plt.plot(
        group_df["Test_Noise"],
        group_df["AVG 95% confidence interval"],
        label=f"Model Train Noise = {round(train_noise, 2)}",
    )
    # plt.plot(group_df["Test_Noise"], group_df["L1Loss"], alpha=0.3)

# Create a list to store unique Test Temp K values
test_noise_values = df["Test_Noise"].unique()
# Set x-axis ticks to match the Test Temp K values from the DataFrame
plt.xticks(test_noise_values)

# Set the start of the y-axis to 0
plt.ylim(bottom=0)  # Automatically adjusts the upper limit

# df_analytical = pd.read_excel("./output/analytical.xlsx")
# plt.plot(df_analytical["Test Noise Max Percentage"], df_analytical["L1Loss"], label=f"analytical", linestyle='--', color='red')

# df_analytical_batch = pd.read_excel("./output/analytical_batch.xlsx")
# plt.plot(df_analytical_batch["Test Noise Max Percentage"], df_analytical_batch["L1Loss"], label=f"analytical_batch", linestyle='--', color='blue')

# df_multi_temp = pd.read_excel("./output/test_with_trained_models_multi_temp.xlsx")
# df_multi_temp_5 = df_multi_temp[df_multi_temp["Temperature_K_step"]==5]
# df_multi_temp_10 = df_multi_temp[df_multi_temp["Temperature_K_step"]==10]
# df_multi_temp_20 = df_multi_temp[df_multi_temp["Temperature_K_step"]==20]
# df_multi_temp_50 = df_multi_temp[df_multi_temp["Temperature_K_step"]==50]
# plt.plot(df_multi_temp_5["Test_Temperature_K"], df_multi_temp_5["L1Loss"], label=f"Model Train Temp [253.15, 258.15, ..., 303.15] (Step of 5K)", linestyle='--', color='red')
# plt.plot(df_multi_temp_10["Test_Temperature_K"], df_multi_temp_10["L1Loss"], label=f"Model Train Temp [253.15, 263.15, ..., 303.15] (Step of 10K)", linestyle='--', color='blue')
# plt.plot(df_multi_temp_20["Test_Temperature_K"], df_multi_temp_20["L1Loss"], label=f"Model Train Temp [253.15, 273.15, 293.15] (Step of 20K)", linestyle='--', color='green')
# plt.plot(df_multi_temp_50["Test_Temperature_K"], df_multi_temp_50["L1Loss"], label=f"Model Train Temp [253.15, 303.15] (Step of 50K)", linestyle='--', color='purple')


# Add labels and legend
plt.xlabel("Test Noise")
plt.ylabel("95% confidence interval (Lower means better)")
plt.legend(loc="upper left")
plt.title("Noise-Dependent Model Evaluation")
plt.show()
