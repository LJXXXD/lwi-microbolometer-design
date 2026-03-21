import pandas as pd
import matplotlib.pyplot as plt

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load the DataFrame from the Excel file
df = pd.read_excel("./outputs/legacy_output/test_with_trained_models_dist.xlsx")

plt.figure(figsize=(6, 5))

# Group the DataFrame by "Model Temp K"
grouped = df.groupby("atm_dist_ratio")

# Iterate over each group and plot "Test Temp K" vs "L1Loss"
for index, (atmdr, group_df) in enumerate(grouped):
    if index % 2 != 0:
        continue
    plt.plot(
        group_df["Test_atm_dist_ratio"],
        group_df["AVG 95% confidence interval"],
        label=f"Model Train Dist Ratio = {round(atmdr, 2)}",
    )
    # plt.plot(group_df["Test_atm_dist_ratio"], group_df["AVG 95% confidence interval"], alpha=0.2)

# Create a list to store unique Test Temp K values
test_atmdr = df["Test_atm_dist_ratio"].unique()
# Set x-axis ticks to match the Test Temp K values from the DataFrame
plt.xticks(test_atmdr)

# Set the start of the y-axis to 0
plt.ylim(bottom=0)  # Automatically adjusts the upper limit

# df_analytical = pd.read_excel("./output/analytical.xlsx")
# plt.plot(df_analytical["Test Noise Max Percentage"], df_analytical["L1Loss"], label=f"analytical", linestyle='--', color='red')

# df_analytical_batch = pd.read_excel("./output/analytical_batch.xlsx")
# plt.plot(df_analytical_batch["Test Noise Max Percentage"], df_analytical_batch["L1Loss"], label=f"analytical_batch", linestyle='--', color='blue')

# df_multi_dist = pd.read_excel("./output/test_with_trained_models_multi_dist.xlsx")
# df_multi_dist_01 = df_multi_dist[df_multi_dist["atm_dist_ratio_step"]==0.1]
# df_multi_temp_02 = df_multi_dist[df_multi_dist["atm_dist_ratio_step"]==0.2]
# df_multi_temp_04 = df_multi_dist[df_multi_dist["atm_dist_ratio_step"]==0.4]
# df_multi_temp_09 = df_multi_dist[df_multi_dist["atm_dist_ratio_step"]==0.9]
# plt.plot(df_multi_dist_01["Test_atmdr"], df_multi_dist_01[f"AVG 95% confidence interval"], label=f"Model Train Dist Ratio [0.1, 0.2, ..., 1]", linestyle='--', color='red')
# plt.plot(df_multi_temp_02["Test_atmdr"], df_multi_temp_02[f"AVG 95% confidence interval"], label=f"Model Train Dist Ratio [0.1, 0.3, ..., 0.9]", linestyle='--', color='blue')
# plt.plot(df_multi_temp_04["Test_atmdr"], df_multi_temp_04[f"AVG 95% confidence interval"], label=f"Model Train Dist Ratio [0.1, 0.5, ..., 0.9]", linestyle='--', color='green')
# plt.plot(df_multi_temp_09["Test_atmdr"], df_multi_temp_09[f"AVG 95% confidence interval"], label=f"Model Train Dist Ratio [0.1, 1]", linestyle='--', color='purple')


# Add labels and legend
plt.xlabel("Test atm distance ratio")
plt.ylabel("95% confidence interval (Lower means better)")
plt.legend(loc="upper left")
plt.title("Distance-Dependent Model Evaluation")
plt.show()
