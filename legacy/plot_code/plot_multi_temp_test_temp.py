import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load the DataFrame from the Excel file
df = pd.read_excel("./outputs/legacy_output/test_with_trained_models_temp.xlsx")


plt.figure(figsize=(8, 6))

# Group the DataFrame by "Model Temp K"
grouped = df.groupby("Temperature_K")

keys = list(grouped.groups.keys())[::2]

# Iterate over each group and plot "Test Temp K" vs "L1Loss"
for index, (model_temp, group_df) in enumerate(grouped):
    if index % 2 != 0:
        continue
    # plt.plot(group_df["Test_Temperature_K"], group_df["AVG 95% confidence interval"], label=f"Model Train Temperature (K) = {model_temp}")
    plt.plot(group_df["Test_Temperature_K"], group_df["AVG 95% confidence interval"], alpha=0.2)

# Create a list to store unique Test Temp K values
test_temp_values = df["Test_Temperature_K"].unique()

# Set x-axis ticks to match the Test Temp K values from the DataFrame
plt.xticks(test_temp_values)

# Set the start of the y-axis to 0
plt.ylim(bottom=0)  # Automatically adjusts the upper limit


df_multi_temp = pd.read_excel("./outputs/legacy_output/test_with_trained_models_multi_temp.xlsx")
df_multi_temp_5 = df_multi_temp[df_multi_temp["Temperature_K_step"] == 5]
df_multi_temp_10 = df_multi_temp[df_multi_temp["Temperature_K_step"] == 10]
df_multi_temp_20 = df_multi_temp[df_multi_temp["Temperature_K_step"] == 20]
df_multi_temp_50 = df_multi_temp[df_multi_temp["Temperature_K_step"] == 50]
plt.plot(
    df_multi_temp_5["Test_Temperature_K"],
    df_multi_temp_5["AVG 95% confidence interval"],
    label="Model Train Temp [253.15, 258.15, ..., 303.15]",
    linestyle="--",
    color="red",
)
plt.plot(
    df_multi_temp_10["Test_Temperature_K"],
    df_multi_temp_10["AVG 95% confidence interval"],
    label="Model Train Temp [253.15, 263.15, ..., 303.15]",
    linestyle="--",
    color="blue",
)
plt.plot(
    df_multi_temp_20["Test_Temperature_K"],
    df_multi_temp_20["AVG 95% confidence interval"],
    label="Model Train Temp [253.15, 273.15, 293.15]",
    linestyle="--",
    color="green",
)
plt.plot(
    df_multi_temp_50["Test_Temperature_K"],
    df_multi_temp_50["AVG 95% confidence interval"],
    label="Model Train Temp [253.15, 303.15]",
    linestyle="--",
    color="purple",
)


# Add labels and legend
plt.xlabel("Test Temperature (K)")
plt.ylabel("95% confidence interval (Lower means better)")
plt.legend(loc="upper left")
plt.title("Temperature-Dependent Model Evaluation")
plt.show()
