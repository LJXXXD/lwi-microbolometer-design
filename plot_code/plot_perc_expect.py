import matplotlib.pyplot as plt

# Adjusting the plot with correct x-axis range and bar positions
plt.figure(figsize=(4, 4))  # Reducing the plot width to 1/4

# Adjusted sample data: Decreasing from 0.2 to 0.05
adjusted_sample_values = [0.2 - (0.2 - 0.05) / 9 * i for i in range(10)]

# Creating the adjusted bar plot
plt.bar([x + 0.5 for x in range(10)], adjusted_sample_values, color='skyblue')
plt.title('Expected Performance')
plt.xlabel('Heroin Percentage')
plt.ylabel('L1 Loss')
plt.xticks(range(11), ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])
plt.grid(True, linestyle='dotted')  # Add dotted gridlines


plt.tight_layout()
plt.show()
