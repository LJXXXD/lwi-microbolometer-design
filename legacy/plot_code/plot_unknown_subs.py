import pandas as pd
import matplotlib.pyplot as plt

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load the DataFrame from the Excel file
df1 = pd.read_excel("./outputs/legacy_output/test_with_trained_models_temp.xlsx")
df2 = pd.read_excel("./outputs/legacy_output/test_with_trained_models_unknown_sub.xlsx")

plt.figure(figsize=(6, 5))

# Group the DataFrame by "Model Temp K"
grouped1 = df1.groupby(["Temperature_K", "Test_Temperature_K"]).get_group((293.15, 293.15))
# print(grouped1[f'AVG 95% confidence interval'])
# print(df2[f'AVG 95% confidence interval'])


# Sample data
categories = ["None", "Sub 0", "Sub 1", "Sub 2", "Sub 3"]
values = pd.concat([grouped1["AVG 95% confidence interval"], df2["AVG 95% confidence interval"]])
# # Define a list of colors for the bars
colors = ["red", "green", "blue", "yellow", "purple"]
# Creating the bar plot
plt.bar(categories, values, color=colors)  # You can customize the color


# Add labels and legend
plt.xlabel("Unknown substance")
plt.ylabel("95% confidence interval (Lower means better)")
# plt.legend(labels=['None', 'sub1 missing', 'sub2 missing', 'sub3 missing', 'sub4 missing'], loc="upper left")
# Creating custom legend
from matplotlib.patches import Patch

legend_labels = [
    "Model trained with 4 substances",
    "Model trained with sub0 missing",
    "Model trained with sub1 missing",
    "Model trained with sub2 missing",
    "Model trained with sub3 missing",
]
legend_handles = [Patch(color=colors[i], label=legend_labels[i]) for i in range(len(legend_labels))]
plt.legend(handles=legend_handles, loc="upper left")


plt.title("Model Evaluation Across Variable Missing Substance")
plt.show()

# # Iterate over each group and plot "Test Temp K" vs "L1Loss"
# for index, (atmdr, group_df) in enumerate(grouped):
#     if index % 2 != 0:
#         continue
#     plt.plot(group_df["Test_atm_dist_ratio"], group_df[f"AVG 95% confidence interval"], label=f"Model Train Dist Ratio = {round(atmdr, 2)}")
#     # plt.plot(group_df["Test_atm_dist_ratio"], group_df["AVG 95% confidence interval"], alpha=0.2)

# # Create a list to store unique Test Temp K values
# test_atmdr = df["Test_atm_dist_ratio"].unique()
# # Set x-axis ticks to match the Test Temp K values from the DataFrame
# plt.xticks(test_atmdr)

# # Set the start of the y-axis to 0
# plt.ylim(bottom=0)  # Automatically adjusts the upper limit

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
# plt.xlabel("Test atmdr")
# plt.ylabel(f"95% confidence interval (Lower means better)")
# plt.legend(loc="upper left")
# plt.title("Distance-Dependent Model Evaluation Across Variable Testing atmdr")
# plt.show()
